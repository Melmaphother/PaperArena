#!/usr/bin/env python3
"""
QA Statistics Analysis Tool

This script analyzes the quality assurance (QA) data from paper JSON files,
providing comprehensive statistics about question types, difficulty levels,
tool chains, and other QA metrics.
"""

import json
import os
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Any, Tuple
import statistics


class QAStatistics:
    """
    A class to analyze QA statistics from paper JSON files.

    This class provides methods to load QA data from JSON files and compute
    various statistics including question counts, type distributions, difficulty
    levels, and tool chain usage patterns.
    """

    def __init__(self, qa_folder_path: str):
        """
        Initialize the QA statistics analyzer.

        Args:
            qa_folder_path (str): Path to the folder containing QA JSON files
        """
        self.qa_folder_path = Path(qa_folder_path)
        self.papers_data: List[Dict[str, Any]] = []
        self.total_papers = 0
        self.total_qas = 0

    def load_qa_data(self) -> None:
        """
        Load QA data from all JSON files in the specified folder.

        Raises:
            FileNotFoundError: If the QA folder doesn't exist
            json.JSONDecodeError: If a JSON file is malformed
        """
        if not self.qa_folder_path.exists():
            raise FileNotFoundError(f"QA folder not found: {self.qa_folder_path}")

        print("🚀 Loading QA data from JSON files...")

        json_files = list(self.qa_folder_path.glob("*.json"))
        self.total_papers = len(json_files)

        for json_file in json_files:
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if "qa" in data and isinstance(data["qa"], list):
                        self.papers_data.append(data)
                        self.total_qas += len(data["qa"])
            except json.JSONDecodeError as e:
                print(f"⚠️ Warning: Failed to parse {json_file}: {e}")
            except Exception as e:
                print(f"⚠️ Warning: Error processing {json_file}: {e}")

        print(f"✅ Successfully loaded {len(self.papers_data)} papers with QA data")
        print(f"📊 Total QAs found: {self.total_qas}")

    def analyze_qa_counts_per_paper(self) -> Dict[str, Any]:
        """
        Analyze QA counts per paper and compute related statistics.

        Returns:
            Dict[str, Any]: Statistics about QA counts per paper
        """
        print("\n📈 Analyzing QA counts per paper...")

        qa_counts = []
        paper_qa_details = []

        for paper in self.papers_data:
            qa_count = len(paper.get("qa", []))
            qa_counts.append(qa_count)

            paper_info = {
                "title": paper.get("title", "Unknown"),
                "id": paper.get("id", "Unknown"),
                "qa_count": qa_count,
            }
            paper_qa_details.append(paper_info)

        # Sort papers by QA count for better analysis
        paper_qa_details.sort(key=lambda x: x["qa_count"], reverse=True)

        stats = {
            "total_papers": len(qa_counts),
            "total_qas": sum(qa_counts),
            "mean_qas_per_paper": statistics.mean(qa_counts) if qa_counts else 0,
            "median_qas_per_paper": statistics.median(qa_counts) if qa_counts else 0,
            "min_qas_per_paper": min(qa_counts) if qa_counts else 0,
            "max_qas_per_paper": max(qa_counts) if qa_counts else 0,
            "std_dev_qas_per_paper": (
                statistics.stdev(qa_counts) if len(qa_counts) > 1 else 0
            ),
            "papers_with_most_qas": paper_qa_details[:5],  # Top 5
            "papers_with_least_qas": paper_qa_details[-5:],  # Bottom 5
            "qa_count_distribution": Counter(qa_counts),
        }

        return stats

    def analyze_qa_types_and_difficulty(self) -> Dict[str, Any]:
        """
        Analyze QA types and difficulty distribution.

        Returns:
            Dict[str, Any]: Statistics about QA types and difficulty levels
        """
        print("\n🎯 Analyzing QA types and difficulty levels...")

        qa_type_counts = Counter()
        difficulty_counts = Counter()
        type_difficulty_matrix = defaultdict(lambda: defaultdict(int))

        for paper in self.papers_data:
            for qa in paper.get("qa", []):
                qa_type = qa.get("qa_type", "Unknown")
                difficulty = qa.get("difficulty", "Unknown")

                qa_type_counts[qa_type] += 1
                difficulty_counts[difficulty] += 1
                type_difficulty_matrix[qa_type][difficulty] += 1

        # Calculate specific statistics requested
        multi_choice_total = qa_type_counts["Multi-Choice Answer"]
        concise_answer_total = qa_type_counts["Concise Answer"]
        open_answer_total = qa_type_counts["Open Answer"]

        stats = {
            "qa_type_distribution": dict(qa_type_counts),
            "difficulty_distribution": dict(difficulty_counts),
            "type_difficulty_matrix": {
                k: dict(v) for k, v in type_difficulty_matrix.items()
            },
            "multi_choice_total": multi_choice_total,
            "concise_answer_total": concise_answer_total,
            "open_answer_total": open_answer_total,
        }

        return stats

    def analyze_tool_chain_statistics(self) -> Dict[str, Any]:
        """
        Analyze tool chain usage patterns and statistics.

        Returns:
            Dict[str, Any]: Statistics about tool chain usage
        """
        print("\n🔧 Analyzing tool chain statistics...")

        tool_chain_lengths = []
        tool_usage_counts = Counter()
        tool_chain_combinations = Counter()
        unique_tools = set()

        for paper in self.papers_data:
            for qa in paper.get("qa", []):
                tool_chain = qa.get("tool_chain", [])

                # Tool chain length statistics
                chain_length = len(tool_chain)
                tool_chain_lengths.append(chain_length)

                # Individual tool usage
                for tool in tool_chain:
                    tool_usage_counts[tool] += 1
                    unique_tools.add(tool)

                # Tool chain combinations
                chain_signature = tuple(sorted(tool_chain))
                tool_chain_combinations[chain_signature] += 1

        # Calculate statistics
        stats = {
            "total_tool_chains": len(tool_chain_lengths),
            "unique_tools_count": len(unique_tools),
            "unique_tools_list": sorted(list(unique_tools)),
            "tool_usage_frequency": dict(tool_usage_counts.most_common()),
            "most_common_tool_combinations": dict(
                tool_chain_combinations.most_common(10)
            ),
            "tool_chain_length_stats": {
                "mean_length": (
                    statistics.mean(tool_chain_lengths) if tool_chain_lengths else 0
                ),
                "median_length": (
                    statistics.median(tool_chain_lengths) if tool_chain_lengths else 0
                ),
                "min_length": min(tool_chain_lengths) if tool_chain_lengths else 0,
                "max_length": max(tool_chain_lengths) if tool_chain_lengths else 0,
                "std_dev_length": (
                    statistics.stdev(tool_chain_lengths)
                    if len(tool_chain_lengths) > 1
                    else 0
                ),
                "length_distribution": dict(Counter(tool_chain_lengths)),
            },
        }

        return stats

    def analyze_additional_qa_metrics(self) -> Dict[str, Any]:
        """
        Analyze additional QA-related metrics.

        Returns:
            Dict[str, Any]: Additional QA statistics
        """
        print("\n📋 Analyzing additional QA metrics...")

        question_lengths = []
        answer_lengths = []
        reasoning_lengths = []
        qa_ids = []

        for paper in self.papers_data:
            for qa in paper.get("qa", []):
                # Question and answer length analysis
                question = qa.get("question", "")
                answer = qa.get("answer", "")
                reasoning = qa.get("reasoning", "")
                qa_id = qa.get("qa_id", 0)

                question_lengths.append(len(question))
                answer_lengths.append(len(answer))
                reasoning_lengths.append(len(reasoning))
                qa_ids.append(qa_id)

        # Calculate text length statistics
        def calculate_text_stats(lengths: List[int], name: str) -> Dict[str, float]:
            if not lengths:
                return {
                    f"{name}_mean": 0,
                    f"{name}_median": 0,
                    f"{name}_min": 0,
                    f"{name}_max": 0,
                    f"{name}_std_dev": 0,
                }

            return {
                f"{name}_mean": statistics.mean(lengths),
                f"{name}_median": statistics.median(lengths),
                f"{name}_min": min(lengths),
                f"{name}_max": max(lengths),
                f"{name}_std_dev": statistics.stdev(lengths) if len(lengths) > 1 else 0,
            }

        stats = {
            "question_length_stats": calculate_text_stats(question_lengths, "question"),
            "answer_length_stats": calculate_text_stats(answer_lengths, "answer"),
            "reasoning_length_stats": calculate_text_stats(
                reasoning_lengths, "reasoning"
            ),
            "qa_id_range": {
                "min_qa_id": min(qa_ids) if qa_ids else 0,
                "max_qa_id": max(qa_ids) if qa_ids else 0,
                "unique_qa_ids": len(set(qa_ids)),
            },
        }

        return stats

    def generate_comprehensive_report(self) -> None:
        """
        Generate and display a comprehensive statistics report.
        """
        print("\n" + "=" * 80)
        print("📊 QA STATISTICS COMPREHENSIVE REPORT")
        print("=" * 80)

        # Load data
        self.load_qa_data()

        # Analyze different aspects
        qa_count_stats = self.analyze_qa_counts_per_paper()
        type_difficulty_stats = self.analyze_qa_types_and_difficulty()
        tool_chain_stats = self.analyze_tool_chain_statistics()
        additional_stats = self.analyze_additional_qa_metrics()

        # Display results
        self._display_qa_count_report(qa_count_stats)
        self._display_type_difficulty_report(type_difficulty_stats)
        self._display_tool_chain_report(tool_chain_stats)
        self._display_additional_metrics_report(additional_stats)

        print("\n✅ Report generation completed!")

    def _display_qa_count_report(self, stats: Dict[str, Any]) -> None:
        """Display QA count statistics."""
        print(f"\n📈 1. QA COUNT STATISTICS PER PAPER")
        print(f"   Total papers analyzed: {stats['total_papers']}")
        print(f"   Total QAs: {stats['total_qas']}")
        print(f"   Average QAs per paper: {stats['mean_qas_per_paper']:.2f}")
        print(f"   Median QAs per paper: {stats['median_qas_per_paper']}")
        print(f"   Min QAs per paper: {stats['min_qas_per_paper']}")
        print(f"   Max QAs per paper: {stats['max_qas_per_paper']}")
        print(f"   Standard deviation: {stats['std_dev_qas_per_paper']:.2f}")

        print(f"\n   📊 QA Count Distribution:")
        for count, frequency in sorted(stats["qa_count_distribution"].items()):
            print(f"      {count} QAs: {frequency} papers")

        print(f"\n   🏆 Top 5 papers with most QAs:")
        for i, paper in enumerate(stats["papers_with_most_qas"], 1):
            print(f"      {i}. {paper['title'][:50]}... ({paper['qa_count']} QAs)")

    def _display_type_difficulty_report(self, stats: Dict[str, Any]) -> None:
        """Display QA type and difficulty statistics."""
        print(f"\n🎯 2. QA TYPE AND DIFFICULTY DISTRIBUTION")

        print(f"   📋 QA Type Distribution:")
        for qa_type, count in stats["qa_type_distribution"].items():
            percentage = (count / sum(stats["qa_type_distribution"].values())) * 100
            print(f"      {qa_type}: {count} ({percentage:.1f}%)")

        print(f"\n   🎚️ Difficulty Level Distribution:")
        for difficulty, count in stats["difficulty_distribution"].items():
            percentage = (count / sum(stats["difficulty_distribution"].values())) * 100
            print(f"      {difficulty}: {count} ({percentage:.1f}%)")

        print(f"\n   📊 Requested Specific Statistics:")
        print(f"      Multi-Choice Answer (total): {stats['multi_choice_total']}")
        print(f"      Concise Answer (total): {stats['concise_answer_total']}")
        print(f"      Open Answer (total): {stats['open_answer_total']}")

        print(f"\n   🔄 Type-Difficulty Cross-tabulation:")
        for qa_type, difficulties in stats["type_difficulty_matrix"].items():
            print(f"      {qa_type}:")
            for difficulty, count in difficulties.items():
                print(f"         {difficulty}: {count}")

    def _display_tool_chain_report(self, stats: Dict[str, Any]) -> None:
        """Display tool chain statistics."""
        print(f"\n🔧 3. TOOL CHAIN STATISTICS")

        length_stats = stats["tool_chain_length_stats"]
        print(f"   📏 Tool Chain Length Statistics:")
        print(f"      Total tool chains: {stats['total_tool_chains']}")
        print(f"      Average length: {length_stats['mean_length']:.2f}")
        print(f"      Median length: {length_stats['median_length']}")
        print(f"      Min length: {length_stats['min_length']}")
        print(f"      Max length: {length_stats['max_length']}")
        print(f"      Standard deviation: {length_stats['std_dev_length']:.2f}")

        print(f"\n   📊 Length Distribution:")
        for length, count in sorted(length_stats["length_distribution"].items()):
            percentage = (count / stats["total_tool_chains"]) * 100
            print(f"      {length} tools: {count} chains ({percentage:.1f}%)")

        print(f"\n   🛠️ Tool Usage Frequency (Top 10):")
        for i, (tool, count) in enumerate(
            list(stats["tool_usage_frequency"].items())[:10], 1
        ):
            percentage = (count / sum(stats["tool_usage_frequency"].values())) * 100
            print(f"      {i}. {tool}: {count} ({percentage:.1f}%)")

        print(f"\n   🔗 Most Common Tool Combinations (Top 5):")
        for i, (combination, count) in enumerate(
            list(stats["most_common_tool_combinations"].items())[:5], 1
        ):
            tools_str = " + ".join(combination)
            print(f"      {i}. [{tools_str}]: {count} times")

    def _display_additional_metrics_report(self, stats: Dict[str, Any]) -> None:
        """Display additional QA metrics."""
        print(f"\n📋 4. ADDITIONAL QA METRICS")

        print(f"   📝 Text Length Statistics:")
        q_stats = stats["question_length_stats"]
        a_stats = stats["answer_length_stats"]
        r_stats = stats["reasoning_length_stats"]

        print(f"      Question lengths (characters):")
        print(f"         Mean: {q_stats['question_mean']:.1f}")
        print(f"         Median: {q_stats['question_median']:.1f}")
        print(f"         Range: {q_stats['question_min']}-{q_stats['question_max']}")

        print(f"      Answer lengths (characters):")
        print(f"         Mean: {a_stats['answer_mean']:.1f}")
        print(f"         Median: {a_stats['answer_median']:.1f}")
        print(f"         Range: {a_stats['answer_min']}-{a_stats['answer_max']}")

        print(f"      Reasoning lengths (characters):")
        print(f"         Mean: {r_stats['reasoning_mean']:.1f}")
        print(f"         Median: {r_stats['reasoning_median']:.1f}")
        print(f"         Range: {r_stats['reasoning_min']}-{r_stats['reasoning_max']}")

        id_stats = stats["qa_id_range"]
        print(f"\n   🔢 QA ID Statistics:")
        print(f"      QA ID range: {id_stats['min_qa_id']}-{id_stats['max_qa_id']}")
        print(f"      Unique QA IDs: {id_stats['unique_qa_ids']}")


def main():
    """
    Main function to run the QA statistics analysis.
    """
    try:
        # Initialize the statistics analyzer
        qa_folder = "qa"
        analyzer = QAStatistics(qa_folder)

        # Generate comprehensive report
        analyzer.generate_comprehensive_report()

    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        raise


if __name__ == "__main__":
    main()
