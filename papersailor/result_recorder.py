#!/usr/bin/env python3
"""
Result Recorder for Paper QA System
Handles recording, persistence, and analysis of QA results
"""

import json
import os
import time
import threading
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from collections import defaultdict, Counter

from loguru import logger

from data_loader import QAItem, PaperInfo
from tools.scorer import LLMScorer


@dataclass
class QAResult:
    """Single QA result data structure"""

    # Paper metadata
    pdf_name: str
    paper_title: str
    status: str
    track: str
    conference: str
    ai_primary_category: str
    ai_research_type: str
    ai_eval_method: str

    # QA metadata
    qa_id: int
    question: str
    ground_truth_answer: str
    model_answer: str

    # Evaluation results
    is_correct: bool
    confidence: float
    evaluation_reason: str

    # Question metadata
    qa_type: str
    difficulty: str
    expected_tool_chain: List[str]
    reasoning: str

    # Timing
    timestamp: str
    processing_time: float


class ResultRecorder:
    """
    Handles recording and managing QA results

    Features:
    - Thread-safe JSONL writing
    - Checkpoint/resume functionality
    - Metrics calculation
    - Detailed reasoning logs
    """

    def __init__(self, model_name: str, results_dir: str = "results"):
        """
        Initialize result recorder

        Args:
            model_name: Name of the model being evaluated
            results_dir: Base directory for results
        """
        self.model_name = model_name
        self.results_dir = Path(results_dir)
        self.model_dir = self.results_dir / model_name

        # Create directories
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.reasoning_dir = self.model_dir / "reasoning_details"
        self.reasoning_dir.mkdir(exist_ok=True)

        # File paths
        self.jsonl_path = self.model_dir / "results.jsonl"
        self.metrics_path = self.model_dir / "metrics.txt"

        # Thread lock for file operations
        self._file_lock = threading.Lock()

        # Initialize scorer
        self.scorer = LLMScorer()

        logger.info(f"🔧 ResultRecorder initialized for model: {model_name}")
        logger.info(f"📁 Results directory: {self.model_dir}")

    def record_result(
        self,
        model: str,
        qa_item: QAItem,
        paper_info: PaperInfo,
        agent_response: str,
        intermediate_steps: List[Dict[str, Any]],
        start_time: float,
        processing_time: float
    ) -> QAResult:
        """
        Record a single QA result

        Args:
            model: str,
            qa_item: QAItem,
            paper_info: PaperInfo,
            agent_response: str,
            intermediate_steps: List[Dict[str, Any]],
            start_time: float,
            processing_time: float,
        Returns:
            QAResult: The recorded result
        """
        try:
            # Evaluate the answer using scorer
            evaluation = self._evaluate_answer(
                qa_item.question, qa_item.answer, agent_response, qa_item.qa_type
            )

            # Create result record
            result = QAResult(
                # Paper metadata
                pdf_name=qa_item.pdf_name,
                paper_title=paper_info.title,
                status=paper_info.status,
                track=paper_info.track,
                conference=paper_info.conference,
                ai_primary_category=paper_info.ai_primary_category,
                ai_research_type=paper_info.ai_research_type,
                ai_eval_method=paper_info.ai_eval_method,
                # QA metadata
                qa_id=qa_item.qa_id,
                question=qa_item.question,
                ground_truth_answer=qa_item.answer,
                model_answer=agent_response,
                # agent_reasoning_steps=intermediate_steps,
                # Evaluation results
                is_correct=evaluation["is_correct"],
                confidence=evaluation["confidence"],
                evaluation_reason=evaluation["reason"],
                # Question metadata
                qa_type=qa_item.qa_type,
                difficulty=qa_item.difficulty,
                expected_tool_chain=qa_item.tool_chain,
                reasoning=qa_item.reasoning,
                # Timing
                timestamp=datetime.now().isoformat(),
                processing_time=processing_time,
            )

            # Write to JSONL (thread-safe)
            self._write_jsonl_record(result)

            # Save detailed reasoning log
            self._save_reasoning_log(
                pdf_name=qa_item.pdf_name, 
                qa_id=qa_item.qa_id, 
                intermediate_steps=intermediate_steps
            )

            logger.info(
                f"✅ Recorded result for {qa_item.pdf_name}/QA-{qa_item.qa_id}: "
                f"{'✓' if result.is_correct else '✗'} (confidence: {result.confidence:.2f})"
            )

            return result

        except Exception as e:
            logger.error(f"❌ Failed to record result: {e}")
            raise

    def _evaluate_answer(
        self, question: str, ground_truth: str, model_answer: str, qa_type: str
    ) -> Dict[str, Any]:
        """
        Evaluate model answer using scorer tool

        Args:
            question: The question
            ground_truth: Ground truth answer
            model_answer: Model's answer
            qa_type: Type of QA

        Returns:
            Dict containing evaluation results
        """
        try:
            # Use LLM scorer to evaluate
            judge_result = self.scorer.score(
                question=question,
                response=model_answer,
                ground_truth=ground_truth,
            )

            # Parse scorer result
            return {
                "is_correct": judge_result.correct == "true",
                "confidence": judge_result.confidence
                / 10.0,  # Convert 0-10 scale to 0-1 scale
                "reason": judge_result.reasoning,
            }

        except Exception as e:
            logger.warning(f"⚠️ Scorer evaluation failed: {e}")
            return {
                "is_correct": False,
                "confidence": 0.0,
                "reason": f"Evaluation failed: {str(e)}",
            }

    def _write_jsonl_record(self, result: QAResult) -> None:
        """
        Write result to JSONL file (thread-safe)

        Args:
            result: QA result to write
        """
        with self._file_lock:
            try:
                with open(self.jsonl_path, "a", encoding="utf-8") as f:
                    json.dump(asdict(result), f, ensure_ascii=False)
                    f.write("\n")
            except Exception as e:
                logger.error(f"❌ Failed to write JSONL record: {e}")
                raise

    def _save_reasoning_log(
        self, pdf_name: str, qa_id: int, intermediate_steps: List
    ) -> None:
        """
        Save detailed reasoning log

        Args:
            pdf_name: PDF name
            qa_id: QA ID
            intermediate_steps: Detailed reasoning process
        """
        try:
            log_file = self.reasoning_dir / f"{pdf_name}_qa{qa_id}.txt"
            with open(log_file, "w", encoding="utf-8") as f:
                for step in intermediate_steps:
                    f.write(f"- {step}\n")

        except Exception as e:
            logger.warning(f"⚠️ Failed to save agent reasoning log: {e}")

    def get_completed_qa_ids(self) -> Dict[str, List[int]]:
        """
        Get completed QA IDs for checkpoint resume

        Returns:
            Dict mapping pdf_name to list of completed qa_ids
        """
        completed = defaultdict(list)

        if not self.jsonl_path.exists():
            return completed

        try:
            with open(self.jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        record = json.loads(line)
                        pdf_name = record["pdf_name"]
                        qa_id = record["qa_id"]
                        completed[pdf_name].append(qa_id)

        except Exception as e:
            logger.warning(f"⚠️ Failed to read completed QA IDs: {e}")

        return completed

    def calculate_and_save_metrics(self) -> Dict[str, Any]:
        """
        Calculate and save comprehensive metrics

        Returns:
            Dict containing all metrics
        """
        if not self.jsonl_path.exists():
            logger.warning("⚠️ No results file found for metrics calculation")
            return {}

        try:
            # Load all results
            results = []
            with open(self.jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        results.append(json.loads(line))

            if not results:
                logger.warning("⚠️ No results found for metrics calculation")
                return {}

            # Calculate metrics
            metrics = self._calculate_metrics(results)

            # Save metrics to file
            self._save_metrics_file(metrics)

            logger.info(f"📊 Calculated metrics for {len(results)} results")
            return metrics

        except Exception as e:
            logger.error(f"❌ Failed to calculate metrics: {e}")
            return {}

    def _calculate_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate comprehensive metrics from results

        Args:
            results: List of result records

        Returns:
            Dict containing calculated metrics
        """
        total_count = len(results)
        correct_count = sum(1 for r in results if r["is_correct"])

        # Overall metrics
        overall_accuracy = correct_count / total_count if total_count > 0 else 0.0
        avg_confidence = (
            sum(r["confidence"] for r in results) / total_count
            if total_count > 0
            else 0.0
        )
        avg_processing_time = (
            sum(r["processing_time"] for r in results) / total_count
            if total_count > 0
            else 0.0
        )

        # Breakdown by QA type and difficulty
        breakdown_metrics = {}

        # QA types: Multi-Choice Answer, Concise Answer, Open Answer
        qa_types = ["Multi-Choice Answer", "Concise Answer", "Open Answer"]
        difficulties = ["Easy", "Medium", "Hard"]

        for qa_type in qa_types:
            type_results = [r for r in results if r["qa_type"] == qa_type]
            breakdown_metrics[qa_type] = {}

            for difficulty in difficulties:
                diff_results = [
                    r for r in type_results if r["difficulty"] == difficulty
                ]

                if diff_results:
                    correct = sum(1 for r in diff_results if r["is_correct"])
                    accuracy = correct / len(diff_results)
                    breakdown_metrics[qa_type][difficulty] = {
                        "accuracy": accuracy,
                        "count": len(diff_results),
                        "correct": correct,
                    }
                else:
                    breakdown_metrics[qa_type][difficulty] = {
                        "accuracy": 0.0,
                        "count": 0,
                        "correct": 0,
                    }

        conference_accuracy = {}
        for conf in set(r["conference"] for r in results):
            conf_results = [r for r in results if r["conference"] == conf]
            conf_correct = sum(1 for r in conf_results if r["is_correct"])
            conference_accuracy[conf] = {
                "accuracy": conf_correct / len(conf_results) if conf_results else 0.0,
                "count": len(conf_results),
            }

        return {
            "model_name": self.model_name,
            "timestamp": datetime.now().isoformat(),
            "total_questions": total_count,
            "correct_answers": correct_count,
            "overall_accuracy": overall_accuracy,
            "average_confidence": avg_confidence,
            "average_processing_time": avg_processing_time,
            "breakdown_by_type_difficulty": breakdown_metrics,
            "conference_accuracy": conference_accuracy,
        }

    def _save_metrics_file(self, metrics: Dict[str, Any]) -> None:
        """
        Save metrics to text file

        Args:
            metrics: Calculated metrics
        """
        try:
            with open(self.metrics_path, "w", encoding="utf-8") as f:
                f.write(f"📊 Evaluation Metrics for {self.model_name}\n")
                f.write("=" * 60 + "\n")
                f.write(f"Generated: {metrics['timestamp']}\n\n")

                # Overall metrics
                f.write("🎯 Overall Performance:\n")
                f.write(f"  Total Questions: {metrics['total_questions']}\n")
                f.write(f"  Correct Answers: {metrics['correct_answers']}\n")
                f.write(f"  Overall Accuracy: {metrics['overall_accuracy']:.2%}\n")
                f.write(f"  Average Confidence: {metrics['average_confidence']:.3f}\n")
                f.write(
                    f"  Average Processing Time: {metrics['average_processing_time']:.2f}s\n\n"
                )

                # Breakdown by QA type and difficulty (10 combinations)
                f.write("📋 Breakdown by QA Type and Difficulty:\n")
                breakdown = metrics["breakdown_by_type_difficulty"]

                for qa_type in ["Multi-Choice Answer", "Concise Answer", "Open Answer"]:
                    f.write(f"\n  {qa_type}:\n")
                    for difficulty in ["Easy", "Medium", "Hard"]:
                        stats = breakdown[qa_type][difficulty]
                        f.write(
                            f"    {difficulty}: {stats['accuracy']:.2%} "
                            f"({stats['correct']}/{stats['count']})\n"
                        )

                # All questions combined by difficulty
                f.write(f"\n  All Questions Combined:\n")
                for difficulty in ["Easy", "Medium", "Hard"]:
                    total_count = sum(
                        breakdown[qt][difficulty]["count"] for qt in breakdown.keys()
                    )
                    total_correct = sum(
                        breakdown[qt][difficulty]["correct"] for qt in breakdown.keys()
                    )
                    accuracy = total_correct / total_count if total_count > 0 else 0.0
                    f.write(
                        f"    {difficulty}: {accuracy:.2%} ({total_correct}/{total_count})\n"
                    )

                # Conference breakdown
                f.write(f"\n🏛️ Performance by Conference:\n")
                for conf, stats in metrics["conference_accuracy"].items():
                    f.write(
                        f"  {conf}: {stats['accuracy']:.2%} ({stats['count']} questions)\n"
                    )


        except Exception as e:
            logger.error(f"❌ Failed to save metrics file: {e}")

    def print_progress_summary(self) -> None:
        """Print current progress summary"""
        completed = self.get_completed_qa_ids()
        total_completed = sum(len(qa_ids) for qa_ids in completed.values())

        print(f"\n📊 Progress Summary for {self.model_name}:")
        print(f"  Completed Questions: {total_completed}")
        print(f"  Completed Papers: {len(completed)}")

        if total_completed > 0:
            # Quick accuracy calculation
            try:
                with open(self.jsonl_path, "r", encoding="utf-8") as f:
                    results = [json.loads(line) for line in f if line.strip()]
                    correct = sum(1 for r in results if r["is_correct"])
                    accuracy = correct / len(results)
                    print(f"  Current Accuracy: {accuracy:.2%}")
            except:
                pass


if __name__ == "__main__":
    # Test the result recorder
    from data_loader import DataLoader

    # Initialize
    recorder = ResultRecorder("test-model")
    loader = DataLoader()

    success, error = loader.load_all_data()
    if not success:
        print(f"❌ Failed to load data: {error}")
        exit(1)

    # Test with first QA item
    if loader.qa_items and loader.papers:
        qa_item = loader.qa_items[0]
        paper_info = loader.get_paper_info(qa_item.pdf_name)

        # Mock result
        result = recorder.record_result(
            model="test-model",
            qa_item=qa_item,
            paper_info=paper_info,
            agent_response="This is a test answer",
            intermediate_steps=["Test reasoning process..."],
            start_time=0.0,
            processing_time=5.5,
        )

        print(f"✅ Test result recorded: {result.is_correct}")

        # Calculate metrics
        metrics = recorder.calculate_and_save_metrics()
        print(f"📊 Metrics calculated: {metrics.get('overall_accuracy', 0):.2%}")
