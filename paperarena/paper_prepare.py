import json
import os
from pathlib import Path


def extract_paper_fields(
    input_file, output_file, required_fields=None, filter_func=None
):
    """
    Extract specific fields from paper data and save to a new file.

    Args:
        input_file (str): Path to input JSON file
        output_file (str): Path to output JSON file
        required_fields (list): List of fields to extract
        filter_func (callable): Optional function to filter papers
    """
    if required_fields is None:
        required_fields = [
            "title",
            "status",
            "track",
            "id",
            "abstract",
            "primary_area",
            "pdf",
        ]

    # Read input data
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Extract required fields
    extracted_data = []
    for paper in data:
        # Apply filter if provided
        if filter_func and not filter_func(paper):
            continue

        extracted_paper = {}
        for field in required_fields:
            if field == "site":
                extracted_paper["pdf"] = paper.get("site", "").replace("forum", "pdf")
            else:
                extracted_paper[field] = paper.get(
                    field, ""
                )  # Use empty string if field doesn't exist
        extracted_data.append(extracted_paper)

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Write extracted data
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(extracted_data, f, indent=2, ensure_ascii=False)

    print(
        f"✅ Processed {len(extracted_data)} papers from {input_file} -> {output_file}"
    )


def main():
    """Process all conference paper data"""
    base_path = Path(__file__).parent / "paperlists"
    output_base = Path(__file__).parent / "selected_papers"

    # Define filter functions
    def iclr_filter(paper):
        """Filter out rejected papers for ICLR"""
        return (
            paper.get("status", "").lower() != "reject"
            and paper.get("status", "").lower() != "withdraw"
            and paper.get("status", "").lower() != "desk reject"
            and paper.get("status", "").lower() != ""
        )

    def icml_filter(paper):
        """Filter out rejected papers for ICML"""
        return (
            paper.get("status", "").lower() != "reject"
            and paper.get("status", "").lower() != ""
        )

    def nips_filter(paper):
        """Filter out rejected papers for NIPS"""
        return (
            paper.get("status", "").lower() != "reject"
            and paper.get("status", "").lower() != "journal"
            and paper.get("status", "").lower() != ""
        )

    def cvpr_filter(paper):
        """Filter out rejected papers for CVPR"""
        return paper.get("status", "").lower() != ""

    # Define conference configurations
    conferences = [
        {
            "name": "icml2025",
            "input": base_path / "icml" / "icml2025.json",
            "output": output_base / "icml2025.json",
            "fields": [
                "title",
                "status",
                "track",
                "id",
                "abstract",
                "primary_area",
                "pdf",
            ],
            "filter": icml_filter,
        },
        {
            "name": "nips2024",
            "input": base_path / "nips" / "nips2024.json",
            "output": output_base / "nips2024.json",
            "fields": [
                "title",
                "status",
                "track",
                "id",
                "abstract",
                "primary_area",
                "pdf",
            ],
            "filter": nips_filter,
        },
        {
            "name": "iclr2025",
            "input": base_path / "iclr" / "iclr2025.json",
            "output": output_base / "iclr2025.json",
            "fields": [
                "title",
                "status",
                "track",
                "id",
                "abstract",
                "primary_area",
                "pdf",
            ],
            "filter": iclr_filter,
        },
        {
            "name": "cvpr2025",
            "input": base_path / "cvpr" / "cvpr2025.json",
            "output": output_base / "cvpr2025.json",
            "fields": [
                "title",
                "status",
                "track",
                "id",
                "abstract",
                "pdf",
            ],  # CVPR doesn't have primary_area
            "filter": cvpr_filter,
        },
        {
            "name": "www2025",
            "input": base_path / "www" / "www2025.json",
            "output": output_base / "www2025.json",
            "fields": [
                "title",
                "status",
                "track",
                "id",
                "abstract",
                "site",
            ],  # WWW doesn't have primary_area
        },
    ]

    print("🚀 Starting paper data extraction...")

    # Process each conference
    for conf in conferences:
        try:
            extract_paper_fields(
                input_file=str(conf["input"]),
                output_file=str(conf["output"]),
                required_fields=conf["fields"],
                filter_func=conf.get("filter"),
            )
        except Exception as e:
            print(f"❌ Error processing {conf['name']}: {str(e)}")

    print("🎉 All conferences processed successfully!")


if __name__ == "__main__":
    main()
