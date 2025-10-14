#!/usr/bin/env python3
"""
QA File Builder Script

Extract ID prefixes from all files in qa_prompts folder, then create corresponding JSON files
in qa folder containing paper information from sampled_papers.jsonl and empty qa field.
"""

import os
import json
import glob
from typing import Dict, Any


def extract_filenames_from_qa_prompts(qa_prompts_dir: str) -> list[tuple[str, str]]:
    """
    Extract filenames from qa_prompts folder and return (id, full_filename_without_ext) pairs.

    Args:
        qa_prompts_dir: Path to qa_prompts folder

    Returns:
        List of (paper_id, filename_without_ext) tuples
    """
    print("🔍 Extracting filenames from qa_prompts folder...")

    txt_files = glob.glob(os.path.join(qa_prompts_dir, "*.txt"))
    file_info = []

    for file_path in txt_files:
        filename = os.path.basename(file_path)
        # Remove .txt extension
        filename_without_ext = filename[:-4]  # Remove .txt
        # Extract ID part before underscore
        if "_" in filename_without_ext:
            paper_id = filename_without_ext.split("_")[0]
            file_info.append((paper_id, filename_without_ext))

    print(f"✅ Successfully extracted {len(file_info)} files")
    return file_info


def load_sampled_papers(sampled_papers_file: str) -> Dict[str, Any]:
    """
    Read sampled_papers.jsonl file and create mapping from ID to paper information.

    Args:
        sampled_papers_file: Path to sampled_papers.jsonl file

    Returns:
        Dictionary mapping from ID to paper information
    """
    print("📖 Reading sampled_papers.jsonl file...")

    id_to_paper = {}

    with open(sampled_papers_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():  # Skip empty lines
                paper_data = json.loads(line.strip())
                paper_id = paper_data.get("id")
                if paper_id:
                    id_to_paper[paper_id] = paper_data

    print(f"✅ Successfully loaded {len(id_to_paper)} paper records")
    return id_to_paper


def create_qa_json_files(
    file_info: list[tuple[str, str]], id_to_paper: Dict[str, Any], qa_dir: str
) -> None:
    """
    Create corresponding JSON files for each file, containing paper info and empty qa field.

    Args:
        file_info: List of (paper_id, filename_without_ext) tuples
        id_to_paper: Mapping from ID to paper information
        qa_dir: Path to qa folder
    """
    print("📝 Creating QA JSON files...")

    # Ensure qa directory exists
    os.makedirs(qa_dir, exist_ok=True)

    created_count = 0
    missing_count = 0

    for paper_id, filename_without_ext in file_info:
        if paper_id in id_to_paper:
            # Get paper information
            paper_info = id_to_paper[paper_id].copy()

            # Add empty qa field
            paper_info["qa"] = []

            # Create JSON filename using the same name as qa_prompts but with .json extension
            json_filename = f"{filename_without_ext}.json"
            json_filepath = os.path.join(qa_dir, json_filename)

            # Write JSON file with standard format (4 spaces indentation)
            with open(json_filepath, "w", encoding="utf-8") as f:
                json.dump(paper_info, f, ensure_ascii=False, indent=4)

            created_count += 1
            print(f"    ✅ Created: {json_filename}")

        else:
            missing_count += 1
            print(f"    ⚠️  ID {paper_id} not found in paper data")

    print(f"\n📊 Statistics:")
    print(f"    Successfully created: {created_count} files")
    print(f"    Missing data: {missing_count} IDs")


def main():
    """Main function"""
    print("🚀 Starting QA file building...")

    # Set up paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    qa_prompts_dir = os.path.join(base_dir, "qa_prompts")
    sampled_papers_file = os.path.join(
        base_dir, "sampling_outputs", "sampled_papers.jsonl"
    )
    qa_dir = os.path.join(base_dir, "qa")

    # Check if necessary files and directories exist
    if not os.path.exists(qa_prompts_dir):
        print(f"❌ Error: qa_prompts directory does not exist: {qa_prompts_dir}")
        return

    if not os.path.exists(sampled_papers_file):
        print(
            f"❌ Error: sampled_papers.jsonl file does not exist: {sampled_papers_file}"
        )
        return

    try:
        # Step 1: Extract file information from qa_prompts
        file_info = extract_filenames_from_qa_prompts(qa_prompts_dir)

        if not file_info:
            print("❌ Error: No valid files found")
            return

        # Step 2: Load paper information
        id_to_paper = load_sampled_papers(sampled_papers_file)

        if not id_to_paper:
            print("❌ Error: No paper information loaded")
            return

        # Step 3: Create JSON files
        create_qa_json_files(file_info, id_to_paper, qa_dir)

        print("\n🎉 QA file building completed!")

    except Exception as e:
        print(f"❌ Error occurred: {str(e)}")
        raise


if __name__ == "__main__":
    main()
