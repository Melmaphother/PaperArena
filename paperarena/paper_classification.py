import json
import jsonlines
import os
from pathlib import Path
from openai import OpenAI
from template import PAPER_CLASSIFICATION_PROMPT
import time
import re
from typing import Dict, List, Optional
from tqdm import tqdm
from dotenv import load_dotenv


class PaperClassifier:
    def __init__(
        self, api_key: str = None, model: str = "gpt-4o", base_url: str = None
    ):
        """Initialize the paper classifier"""
        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url

        self.client = OpenAI(**client_kwargs)
        self.model = model
        self.classification_cache = {}

    def extract_json_from_response(self, response_text: str) -> Optional[Dict]:
        """Extract JSON from LLM response using direct brace matching"""
        try:
            # First try direct parsing
            return json.loads(response_text.strip())
        except json.JSONDecodeError:
            pass

        # Extract JSON using brace matching - more reliable than code blocks
        brace_count = 0
        start_idx = -1

        for i, char in enumerate(response_text):
            if char == "{":
                if brace_count == 0:
                    start_idx = i
                brace_count += 1
            elif char == "}":
                brace_count -= 1
                if brace_count == 0 and start_idx != -1:
                    # Found complete JSON object
                    json_str = response_text[start_idx : i + 1]
                    try:
                        return json.loads(json_str)
                    except json.JSONDecodeError:
                        # Continue searching for next JSON object
                        start_idx = -1
                        continue

        print(f"❌ Failed to extract JSON from response: {response_text[:200]}...")
        return None

    def classify_paper(
        self, title: str, abstract: str, max_retries: int = 3
    ) -> Optional[Dict]:
        """Classify a single paper using GPT-4o"""
        cache_key = f"{title}|{abstract}"
        if cache_key in self.classification_cache:
            return self.classification_cache[cache_key]

        prompt = PAPER_CLASSIFICATION_PROMPT.substitute(title=title, abstract=abstract)

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert AI research taxonomist. Provide classification results in valid JSON format only.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.1,
                    max_tokens=500,
                )

                result = self.extract_json_from_response(
                    response.choices[0].message.content
                )

                if result and all(
                    field in result
                    for field in [
                        "primary_category",
                        "research_type",
                        "eval_method",
                        "confidence",
                    ]
                ):
                    self.classification_cache[cache_key] = result
                    return result

                if attempt < max_retries - 1:
                    time.sleep(2**attempt)

            except Exception as e:
                print(f"❌ Error classifying paper (attempt {attempt + 1}): {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(2**attempt)

        return None

    def load_existing_results(self, output_file: Path) -> Dict[str, Dict]:
        """Load existing classification results for resume functionality"""
        existing_results = {}

        if output_file.exists():
            try:
                with jsonlines.open(output_file, "r") as reader:
                    for paper in reader:
                        paper_id = paper.get("id", "")
                        if paper_id:
                            existing_results[paper_id] = paper
                print(
                    f"📂 Loaded {len(existing_results)} existing results from {output_file}"
                )
            except Exception as e:
                print(f"⚠️  Error loading existing results: {str(e)}")

        return existing_results

    def process_conference_papers(
        self,
        json_file: Path,
        conference_name: str,
        output_file: Path = None,
        resume: bool = True,
    ) -> List[Dict]:
        """Process all papers from a conference JSON file with resume capability"""
        print(f"📚 Processing {conference_name.upper()} papers from {json_file}")

        try:
            with open(json_file, "r", encoding="utf-8") as f:
                papers = json.load(f)
        except Exception as e:
            print(f"❌ Error reading {json_file}: {str(e)}")
            return []

        # Load existing results for resume
        existing_results = {}
        if resume and output_file:
            existing_results = self.load_existing_results(output_file)

        processed_papers = []
        skipped_count = 0

        for paper in tqdm(
            papers, desc=f"Classifying {conference_name.upper()}", unit="paper"
        ):
            paper_id = paper.get("id", "")
            title = paper.get("title", "").strip()
            abstract = paper.get("abstract", "").strip()

            if not title or not abstract:
                continue

            # Check if already processed (resume functionality)
            if resume and paper_id in existing_results:
                processed_papers.append(existing_results[paper_id])
                skipped_count += 1
                continue

            classification = self.classify_paper(title, abstract)

            result = {
                "title": paper.get("title", ""),
                "status": paper.get("status", ""),
                "track": paper.get("track", ""),
                "primary_area": paper.get("primary_area", ""),
                "id": paper.get("id", ""),
                "abstract": paper.get("abstract", ""),
                "pdf": paper.get("pdf", ""),
                "conference": conference_name,
            }

            if classification:
                result.update(
                    {
                        "ai_primary_category": classification.get(
                            "primary_category", ""
                        ),
                        "ai_secondary_category": classification.get(
                            "secondary_category", ""
                        ),
                        "ai_research_type": classification.get("research_type", ""),
                        "ai_eval_method": classification.get("eval_method", ""),
                        "ai_confidence": classification.get("confidence", 0.0),
                        "ai_reasoning": classification.get("reasoning", ""),
                        "classification_status": "success",
                    }
                )
            else:
                result.update(
                    {
                        "ai_primary_category": "",
                        "ai_secondary_category": "",
                        "ai_research_type": "",
                        "ai_eval_method": "",
                        "ai_confidence": 0.0,
                        "ai_reasoning": "",
                        "classification_status": "failed",
                    }
                )

            processed_papers.append(result)

            # Save incrementally for resume capability
            if output_file:
                self.save_single_result(output_file, result)

            time.sleep(0.1)  # Rate limiting

        if skipped_count > 0:
            print(f"⏭️  Skipped {skipped_count} already processed papers")

        return processed_papers

    def save_single_result(self, output_file: Path, result: Dict):
        """Save a single result to JSONL file (append mode)"""
        try:
            os.makedirs(output_file.parent, exist_ok=True)
            with jsonlines.open(output_file, "a") as writer:
                writer.write(result)
        except Exception as e:
            print(f"⚠️  Error saving single result: {str(e)}")

    def process_all_conferences(
        self, input_dir: Path, output_dir: Path, resume: bool = True
    ):
        """Process all conference files and output to separate JSONL files"""
        conferences = {
            # "icml2025": input_dir / "icml2025.json",
            # "nips2024": input_dir / "nips2024.json",
            # "iclr2025": input_dir / "iclr2025.json",
            # "cvpr2025": input_dir / "cvpr2025.json",
            "www2025": input_dir / "www2025.json",
        }

        os.makedirs(output_dir, exist_ok=True)
        total_papers = 0
        conference_stats = {}

        print("🚀 Starting paper classification for all conferences...")
        print(f"📁 Output directory: {output_dir}")
        print(f"🔄 Resume mode: {'Enabled' if resume else 'Disabled'}")

        for conf_name, conf_file in conferences.items():
            if not conf_file.exists():
                print(f"⚠️  File not found: {conf_file}")
                continue

            # Define output file for this conference
            output_file = output_dir / f"{conf_name}_classified.jsonl"

            # Check if conference is already completed
            if resume and output_file.exists():
                existing_count = self.count_existing_results(output_file)
                with open(conf_file, "r", encoding="utf-8") as f:
                    total_expected = len(json.load(f))

                if existing_count >= total_expected:
                    print(
                        f"✅ {conf_name.upper()} already completed ({existing_count} papers)"
                    )
                    conference_stats[conf_name] = existing_count
                    total_papers += existing_count
                    continue

            print(f"\n{'='*60}")
            print(f"🎯 Processing {conf_name.upper()}")
            print(f"{'='*60}")

            papers = self.process_conference_papers(
                conf_file, conf_name, output_file, resume
            )

            conference_stats[conf_name] = len(papers)
            total_papers += len(papers)

            print(
                f"✅ {conf_name.upper()} completed: {len(papers)} papers -> {output_file}"
            )

        # Generate summary
        self.print_final_summary(conference_stats, total_papers, output_dir)

    def count_existing_results(self, output_file: Path) -> int:
        """Count existing results in JSONL file"""
        if not output_file.exists():
            return 0

        try:
            with jsonlines.open(output_file, "r") as reader:
                return sum(1 for _ in reader)
        except Exception:
            return 0

    def print_final_summary(
        self, conference_stats: Dict[str, int], total_papers: int, output_dir: Path
    ):
        """Print final classification summary"""
        print(f"\n{'='*70}")
        print("🎉 CLASSIFICATION COMPLETED!")
        print(f"{'='*70}")

        print(f"📊 Total Papers Processed: {total_papers:,}")
        print(f"📁 Output Directory: {output_dir}")

        print(f"\n📋 Conference Breakdown:")
        for conf_name, count in conference_stats.items():
            output_file = output_dir / f"{conf_name}_classified.jsonl"
            print(
                f"  {conf_name.upper():<12}: {count:>6,} papers -> {output_file.name}"
            )

        print(f"\n🔗 Individual Files:")
        for conf_name in conference_stats.keys():
            output_file = output_dir / f"{conf_name}_classified.jsonl"
            if output_file.exists():
                size_mb = output_file.stat().st_size / 1024 / 1024
                print(f"  📄 {output_file}: {size_mb:.2f} MB")

        print(f"\n{'='*70}")
        print("✨ All classifications saved as separate JSONL files!")
        print("🔄 Use resume=True to continue interrupted classifications")
        print(f"{'='*70}")


def main():
    """Main function to run paper classification"""
    print("🤖 AI Paper Classification System")
    print("=" * 50)

    # Check for OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY environment variable not found!")
        print("Please set your API key: export OPENAI_API_KEY='your-key-here'")
        return

    # Check for optional base_url
    base_url = os.getenv("OPENAI_BASE_URL")
    if base_url:
        print(f"🔗 Using custom base URL: {base_url}")

    # Initialize classifier
    classifier = PaperClassifier(
        api_key=api_key, base_url=base_url, model="gpt-4o-mini-2024-07-18"
    )

    # Define paths
    base_dir = Path(__file__).parent
    input_dir = base_dir / "selected_papers"
    output_dir = base_dir / "classified_results"

    # Process all conferences with resume capability
    classifier.process_all_conferences(input_dir, output_dir, resume=True)


if __name__ == "__main__":
    load_dotenv(".env")
    main()
