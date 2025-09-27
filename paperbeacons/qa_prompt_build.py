import os
import jsonlines
import re
from pathlib import Path
from typing import Dict, Optional
from template import QA_GENERATION_PROMPT


def sanitize_filename(filename: str, max_length: int = 200) -> str:
    """
    Clean filename by removing/replacing illegal characters
    Same as in paper_crawler.py to maintain consistency

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

    # Truncate if too long (leave space for .txt extension)
    if len(filename) > max_length - 4:
        filename = filename[: max_length - 4]

    return filename


def get_category_full_name(category_code: str) -> str:
    """
    Convert category code to full name

    Args:
        category_code (str): Category abbreviation

    Returns:
        str: Full category name
    """
    category_mapping = {
        "ML": "Machine Learning",
        "NLP": "Natural Language Processing",
        "CV": "Computer Vision",
        "RL": "Reinforcement Learning",
        "Gen AI": "Generative AI",
        "IR": "Information Retrieval",
        "AI4Science": "AI for Science",
        "MLSys": "Machine Learning Systems",
    }
    if category_code not in category_mapping:
        print(f"❌ Unknown category: {category_code}")
        return category_code
    return category_mapping.get(category_code)


def get_research_type_description(research_type: str) -> str:
    """
    Get full description for research type

    Args:
        research_type (str): Research type

    Returns:
        str: Full description
    """
    type_mapping = {
        "Theoretical": "Theoretical: Focuses on mathematical proofs and formal analysis",
        "Empirical": "Empirical: Focuses on experimental validation and comparisons",
        "Architectural": "Architectural: Proposes a new model or system architecture",
        "Resource": "Resource: Creates a new dataset, benchmark, or framework",
    }
    if research_type not in type_mapping:
        print(f"❌ Unknown research type: {research_type}")
        return research_type
    return type_mapping.get(research_type)


def get_eval_method_description(eval_method: str) -> str:
    """
    Get full description for evaluation method

    Args:
        eval_method (str): Evaluation method

    Returns:
        str: Full description
    """
    eval_mapping = {
        "Benchmark": "Evaluates on **established, public benchmarks** (e.g., ImageNet, SQuAD, GLUE) to compare performance against other existing work.",
        "Empirical": "Empirical: Validates claims through controlled experiments on **non-standard, private, or custom-designed datasets/environments**. The focus is on proving a specific hypothesis, where a public benchmark is unavailable or unsuitable.",
        "Ablation": "Ablation: Uses ablation studies (systematically removing components) to justify internal design choices.",
        "Qualitative": "Relies on non-numerical evidence, such as case studies, visualizations, or human evaluations.",
        "System": "System: Measures system-level metrics like latency, throughput, or memory usage.",
    }
    if eval_method not in eval_mapping:
        print(f"❌ Unknown evaluation method: {eval_method}")
        return eval_method
    return eval_mapping.get(eval_method)


def convert_status_to_tier(status: str) -> str:
    """
    Convert paper status to tier level

    Args:
        status (str): Paper status

    Returns:
        str: Tier level
    """
    status_mapping = {
        "Poster": "Standard-Tier",
        "Spotlight": "Mid-Tier",
        "Highlight": "Mid-Tier",
        "Oral": "Top-Tier",
        "Award Candidate": "Top-Tier",
    }
    if status not in status_mapping:
        print(f"❌ Unknown status: {status}")
        return status
    return status_mapping.get(status)


def build_qa_prompt(paper: Dict) -> str:
    """
    Build QA generation prompt for a single paper

    Args:
        paper (Dict): Paper information from JSONL

    Returns:
        str: Generated prompt
    """
    # Extract paper information
    title = paper.get("title", "")
    abstract = paper.get("abstract", "")
    conference = paper.get("conference", "")
    status = convert_status_to_tier(paper.get("status", ""))

    # Get AI classification results
    primary_category = paper.get("ai_primary_category", "")
    secondary_category = paper.get("ai_secondary_category", "")
    research_type = paper.get("ai_research_type", "")
    eval_method = paper.get("ai_eval_method", "")

    # Convert to full names/descriptions
    primary_category_full = get_category_full_name(primary_category)
    secondary_category_full = (
        get_category_full_name(secondary_category) if secondary_category else ""
    )
    research_type_desc = get_research_type_description(research_type)
    eval_method_desc = get_eval_method_description(eval_method)

    # Build category string
    if secondary_category_full:
        category = f"{primary_category_full}, {secondary_category_full}"
    else:
        category = primary_category_full

    # Generate prompt using template
    prompt = QA_GENERATION_PROMPT.substitute(
        title=title,
        abstract=abstract,
        conference=conference,
        status=status,
        category=category,
        research_type=research_type_desc,
        eval_method=eval_method_desc,
    )

    return prompt


def build_all_qa_prompts(
    sampled_papers_file: str, output_dir: str = "qa_prompts"
) -> Dict[str, int]:
    """
    Build QA generation prompts for all papers in the sampled papers file

    Args:
        sampled_papers_file (str): Path to sampled papers JSONL file
        output_dir (str): Directory to save prompt files

    Returns:
        Dict[str, int]: Statistics of the process
    """
    print("🚀 Starting QA prompt generation for sampled papers...")
    print(f"📁 Input file: {sampled_papers_file}")
    print(f"📁 Output directory: {output_dir}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load sampled papers
    papers = []
    try:
        with jsonlines.open(sampled_papers_file, "r") as reader:
            papers = list(reader)
        print(f"📊 Found {len(papers)} papers to process")
    except Exception as e:
        print(f"❌ Error loading sampled papers: {str(e)}")
        return {"error": 1}

    # Initialize statistics
    stats = {"total": len(papers), "success": 0, "failed": 0}

    # Process each paper
    for i, paper in enumerate(papers, 1):
        try:
            paper_id = paper.get("id", "unknown")
            title = paper.get("title", "untitled")

            # Create filename: id_title.txt
            sanitized_title = sanitize_filename(title)
            filename = f"{paper_id}_{sanitized_title}.txt"

            print(f"📝 [{i}/{len(papers)}] Generating prompt for: {title[:60]}...")

            # Build prompt
            prompt = build_qa_prompt(paper)

            # Save prompt to file (overwrite if exists)
            output_path = Path(output_dir) / filename
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(prompt)

            stats["success"] += 1
            print(f"✅ [{i}/{len(papers)}] Successfully generated: {filename}")

        except Exception as e:
            print(f"❌ [{i}/{len(papers)}] Error processing paper {paper_id}: {str(e)}")
            stats["failed"] += 1
            continue

    # Print final statistics
    print("\n" + "=" * 60)
    print("📊 QA Prompt Generation Summary")
    print("=" * 60)
    print(f"📈 Total papers: {stats['total']}")
    print(f"✅ Successfully generated: {stats['success']}")
    print(f"❌ Failed generations: {stats['failed']}")

    success_rate = (
        (stats["success"] / stats["total"] * 100) if stats["total"] > 0 else 0
    )
    print(f"📊 Success rate: {success_rate:.1f}%")

    return stats


def main():
    """Main function to build QA prompts for sampled papers"""
    print("📝 QA Prompt Builder")
    print("=" * 50)

    # Default paths
    base_dir = Path(__file__).parent
    sampled_papers_file = base_dir / "sampling_outputs" / "sampled_papers.jsonl"
    output_dir = base_dir / "qa_prompts"

    # Check if sampled papers file exists
    if not sampled_papers_file.exists():
        print(f"❌ Sampled papers file not found: {sampled_papers_file}")
        print("Please run paper_sampling.py first to generate sampled papers.")
        return

    print(f"📁 Input file: {sampled_papers_file}")
    print(f"📁 Output directory: {output_dir}")

    # Ask user for confirmation
    response = (
        input("\n🤔 Do you want to proceed with QA prompt generation? (y/N): ")
        .strip()
        .lower()
    )
    if response not in ["y", "yes"]:
        print("❌ QA prompt generation cancelled by user.")
        return

    # Start generation
    stats = build_all_qa_prompts(str(sampled_papers_file), str(output_dir))

    if "error" not in stats:
        print(f"\n🎉 QA prompt generation completed!")
        print(f"📁 Check the generated prompts in: {output_dir}")
    else:
        print(f"\n❌ QA prompt generation failed!")


if __name__ == "__main__":
    main()
