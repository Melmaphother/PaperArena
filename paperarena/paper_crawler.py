import requests
import os
from pathlib import Path
from urllib.parse import urlparse, parse_qs
import time
from typing import Optional, List, Dict
import openreview
import getpass
import jsonlines
import re


def sanitize_filename(filename: str, max_length: int = 200) -> str:
    """
    Clean filename by removing/replacing illegal characters

    Args:
        filename (str): Original filename
        max_length (int): Maximum filename length

    Returns:
        str: Sanitized filename
    """
    # Replace illegal characters
    illegal_chars = {
        ":": "-",
        "/": "-",
        "\\": "-",
        "?": "",
        "*": "",
        '"': "",
        "<": "",
        ">": "",
        "|": "",
        "\n": " ",
        "\r": " ",
        "\t": " ",
    }

    # Apply replacements
    for illegal, replacement in illegal_chars.items():
        filename = filename.replace(illegal, replacement)

    # Remove multiple spaces and trim
    filename = re.sub(r"\s+", " ", filename).strip()

    # Truncate if too long (leave space for .pdf extension)
    if len(filename) > max_length - 4:
        filename = filename[: max_length - 4]

    return filename


def get_openreview_client(username: str = None, password: str = None):
    """
    Create and return OpenReview client with authentication

    Args:
        username (str): OpenReview username (optional, will prompt if not provided)
        password (str): OpenReview password (optional, will prompt if not provided)

    Returns:
        openreview.api.OpenReviewClient: Authenticated client
    """
    try:
        if not username:
            username = input(
                "🔐 Enter OpenReview username (or press Enter to use anonymous access): "
            ).strip()

        if not password and username:
            password = getpass.getpass("🔐 Enter OpenReview password: ")

        if username and password:
            print(f"🔑 Authenticating as: {username}")
            client = openreview.api.OpenReviewClient(
                baseurl="https://api2.openreview.net",
                username=username,
                password=password,
            )
        else:
            print("🔓 Using anonymous access (some PDFs may not be available)")
            client = openreview.api.OpenReviewClient(
                baseurl="https://api2.openreview.net"
            )

        return client
    except Exception as e:
        print(f"❌ Failed to create OpenReview client: {str(e)}")
        return None


def download_openreview_pdf(
    paper_id: str, output_dir: str, filename: str = None, client=None
) -> bool:
    """
    Download PDF from OpenReview using official client

    Args:
        paper_id (str): OpenReview paper ID (e.g., "038rEwbChh")
        output_dir (str): Directory to save the PDF
        filename (str): Custom filename (optional, will use paper_id if not provided)
        client: OpenReview client (optional, will create if not provided)

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Create client if not provided
        if client is None:
            client = get_openreview_client()
            if client is None:
                return False

        print(f"📄 Downloading OpenReview paper: {paper_id}")

        # Get PDF content using official client
        pdf_content = client.get_pdf(id=paper_id)

        if not pdf_content:
            print(f"❌ Failed to get PDF content for {paper_id}")
            return False

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Determine filename
        if not filename:
            filename = f"{paper_id}.pdf"
        if not filename.endswith(".pdf"):
            filename += ".pdf"

        output_path = Path(output_dir) / filename

        # Save PDF
        with open(output_path, "wb") as f:
            f.write(pdf_content)

        print(f"✅ Successfully downloaded: {output_path}")
        print(f"📊 File size: {len(pdf_content) / 1024 / 1024:.2f} MB")
        return True

    except Exception as e:
        print(f"❌ Error downloading {paper_id}: {str(e)}")
        return False


def download_cvpr_pdf(pdf_url: str, output_dir: str, filename: str = None) -> bool:
    """
    Download PDF directly from CVPR website - simple HTTP download

    Args:
        pdf_url (str): Direct URL to CVPR PDF
        output_dir (str): Directory to save the PDF
        filename (str): Custom filename (optional, will extract from URL if not provided)

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        print(f"📄 Downloading CVPR paper from: {pdf_url}")

        # Simple request with basic headers
        headers = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"}

        # Download with stream=True for large files
        response = requests.get(pdf_url, headers=headers, timeout=30, stream=True)
        response.raise_for_status()

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Determine filename
        if not filename:
            # Extract filename from URL
            filename = Path(urlparse(pdf_url).path).name
            if not filename.endswith(".pdf"):
                filename = "cvpr_paper.pdf"

        output_path = Path(output_dir) / filename

        # Download and save PDF in chunks
        total_size = 0
        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    total_size += len(chunk)

        print(f"✅ Successfully downloaded: {output_path}")
        print(f"📊 File size: {total_size / 1024 / 1024:.2f} MB")
        return True

    except Exception as e:
        print(f"❌ Error downloading CVPR PDF: {str(e)}")
        return False


def extract_openreview_id(pdf_url: str) -> Optional[str]:
    """
    Extract paper ID from OpenReview PDF URL

    Args:
        pdf_url (str): OpenReview PDF URL

    Returns:
        str: Paper ID if found, None otherwise
    """
    try:
        parsed_url = urlparse(pdf_url)
        if "openreview.net" in parsed_url.netloc:
            query_params = parse_qs(parsed_url.query)
            paper_id = query_params.get("id", [None])[0]
            return paper_id
    except Exception:
        pass
    return None


def download_sampled_papers(
    sampled_papers_file: str, output_dir: str = "sampled_papers", client=None
) -> Dict[str, int]:
    """
    Download all papers from sampled papers JSONL file

    Args:
        sampled_papers_file (str): Path to sampled papers JSONL file
        output_dir (str): Directory to save PDFs
        client: OpenReview client (optional, will create if needed)

    Returns:
        Dict[str, int]: Statistics of download results
    """
    print("🚀 Starting batch download of sampled papers...")
    print(f"📁 Input file: {sampled_papers_file}")
    print(f"📁 Output directory: {output_dir}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load sampled papers
    papers = []
    try:
        with jsonlines.open(sampled_papers_file, "r") as reader:
            papers = list(reader)
        print(f"📊 Found {len(papers)} papers to download")
    except Exception as e:
        print(f"❌ Error loading sampled papers: {str(e)}")
        return {"error": 1}

    # Initialize statistics
    stats = {
        "total": len(papers),
        "success": 0,
        "failed": 0,
        "skipped": 0,
        "openreview": 0,
        "cvpr": 0,
    }

    # Create OpenReview client if needed
    if client is None:
        print("🔑 Setting up OpenReview client...")
        client = get_openreview_client()

    # Process each paper
    for i, paper in enumerate(papers, 1):
        try:
            paper_id = paper.get("id", "unknown")
            title = paper.get("title", "untitled")
            pdf_url = paper.get("pdf", "")

            if not pdf_url:
                print(f"⚠️  [{i}/{len(papers)}] No PDF URL for paper {paper_id}")
                stats["skipped"] += 1
                continue

            # Create filename: id_title.pdf
            sanitized_title = sanitize_filename(title)
            filename = f"{paper_id}_{sanitized_title}.pdf"

            print(f"📄 [{i}/{len(papers)}] Downloading: {title[:60]}...")

            # Check if file already exists
            output_path = Path(output_dir) / filename
            if output_path.exists():
                print(
                    f"✅ [{i}/{len(papers)}] File already exists, skipping: {filename}"
                )
                stats["skipped"] += 1
                continue

            # Determine download method based on URL
            success = False
            if "openreview.net" in pdf_url:
                # OpenReview download
                openreview_id = extract_openreview_id(pdf_url)
                if openreview_id and client:
                    success = download_openreview_pdf(
                        openreview_id, output_dir, filename, client=client
                    )
                    if success:
                        stats["openreview"] += 1
                else:
                    print(
                        f"❌ [{i}/{len(papers)}] Failed to extract OpenReview ID or no client"
                    )

            elif "thecvf.com" in pdf_url:
                # CVPR download
                success = download_cvpr_pdf(pdf_url, output_dir, filename)
                if success:
                    stats["cvpr"] += 1

            else:
                print(f"❌ [{i}/{len(papers)}] Unsupported PDF URL: {pdf_url}")

            # Update statistics
            if success:
                stats["success"] += 1
                print(f"✅ [{i}/{len(papers)}] Successfully downloaded: {filename}")
            else:
                stats["failed"] += 1
                print(f"❌ [{i}/{len(papers)}] Failed to download: {filename}")

            # Add delay between downloads to be respectful
            if i < len(papers):  # Don't sleep after the last download
                time.sleep(1)

        except Exception as e:
            print(f"❌ [{i}/{len(papers)}] Error processing paper {paper_id}: {str(e)}")
            stats["failed"] += 1
            continue

    # Print final statistics
    print("\n" + "=" * 60)
    print("📊 Download Summary")
    print("=" * 60)
    print(f"📈 Total papers: {stats['total']}")
    print(f"✅ Successfully downloaded: {stats['success']}")
    print(f"❌ Failed downloads: {stats['failed']}")
    print(f"⏭️  Skipped (already exists): {stats['skipped']}")
    print(f"🔬 OpenReview papers: {stats['openreview']}")
    print(f"🏛️  CVPR papers: {stats['cvpr']}")

    success_rate = (
        (stats["success"] / stats["total"] * 100) if stats["total"] > 0 else 0
    )
    print(f"📊 Success rate: {success_rate:.1f}%")

    return stats


def test_sample_downloads():
    """Test the download functions with sample URLs"""
    print("🚀 Testing PDF download functionality...")

    # Create test output directory
    output_dir = "test_pdfs"

    # Test OpenReview download
    print("\n" + "=" * 60)
    print("🔬 Testing OpenReview Download")
    print("=" * 60)

    # Create OpenReview client once for all downloads
    print("🔑 Setting up OpenReview client...")
    client = get_openreview_client()

    if client:
        icml_url = "https://openreview.net/pdf?id=038rEwbChh"
        paper_id = extract_openreview_id(icml_url)

        if paper_id:
            print(f"📋 Extracted paper ID: {paper_id}")
            success = download_openreview_pdf(
                paper_id, output_dir, "icml_sample.pdf", client=client
            )
            if success:
                print("✅ OpenReview download test: PASSED")
            else:
                print("❌ OpenReview download test: FAILED")
        else:
            print("❌ Failed to extract paper ID from URL")
    else:
        print("❌ Failed to create OpenReview client")

    # Wait a bit between requests
    time.sleep(2)

    # Test CVPR download
    print("\n" + "=" * 60)
    print("🔬 Testing CVPR Download")
    print("=" * 60)

    cvpr_url = "https://openaccess.thecvf.com/content/CVPR2025/papers/Zhang_2DMamba_Efficient_State_Space_Model_for_Image_Representation_with_Applications_CVPR_2025_paper.pdf"

    success = download_cvpr_pdf(cvpr_url, output_dir, "cvpr_sample.pdf")
    if success:
        print("✅ CVPR download test: PASSED")
    else:
        print("❌ CVPR download test: FAILED")

    print("\n" + "=" * 60)
    print("📊 Test Summary")
    print("=" * 60)

    # Check downloaded files
    test_dir = Path(output_dir)
    if test_dir.exists():
        pdf_files = list(test_dir.glob("*.pdf"))
        print(f"📁 Downloaded files in {output_dir}:")
        for pdf_file in pdf_files:
            size_mb = pdf_file.stat().st_size / 1024 / 1024
            print(f"  📄 {pdf_file.name}: {size_mb:.2f} MB")
    else:
        print("❌ No files downloaded")


def main():
    """Main function to download sampled papers"""
    print("📚 Sampled Papers Downloader")
    print("=" * 50)

    # Default paths
    base_dir = Path(__file__).parent
    sampled_papers_file = base_dir / "sampling_outputs" / "sampled_papers.jsonl"
    output_dir = base_dir / "sampled_papers"

    # Check if sampled papers file exists
    if not sampled_papers_file.exists():
        print(f"❌ Sampled papers file not found: {sampled_papers_file}")
        print("Please run paper_sampling.py first to generate sampled papers.")
        return

    print(f"📁 Input file: {sampled_papers_file}")
    print(f"📁 Output directory: {output_dir}")

    # Ask user for confirmation
    response = (
        input("\n🤔 Do you want to proceed with downloading? (y/N): ").strip().lower()
    )
    if response not in ["y", "yes"]:
        print("❌ Download cancelled by user.")
        return

    # Start download
    stats = download_sampled_papers(str(sampled_papers_file), str(output_dir))

    if "error" not in stats:
        print(f"\n🎉 Download process completed!")
        print(f"📁 Check the downloaded PDFs in: {output_dir}")
    else:
        print(f"\n❌ Download process failed!")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_sample_downloads()
    else:
        main()
