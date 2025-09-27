#!/usr/bin/env python3
"""
Batch evaluation script for Paper QA system
Run evaluations across multiple models and configurations
"""

import os
import subprocess
import time
from pathlib import Path
from typing import List, Dict, Any
from concurrent.futures import ProcessPoolExecutor, as_completed

from loguru import logger
from config_loader import ConfigLoader


class BatchRunner:
    """
    Batch runner for evaluating multiple models
    """

    def __init__(self, config_dir: str = "configs"):
        """
        Initialize batch runner

        Args:
            config_dir: Directory containing configuration files
        """
        self.config_loader = ConfigLoader(config_dir)
        self.results_summary = []

        logger.info("🚀 BatchRunner initialized")

    def run_single_evaluation(self, config_name: str) -> Dict[str, Any]:
        """
        Run evaluation for a single configuration

        Args:
            config_name: Name of configuration file

        Returns:
            Dict with evaluation results
        """
        start_time = time.time()

        try:
            logger.info(f"🔄 Starting evaluation: {config_name}")

            # Run the evaluation script
            cmd = ["python", "run_react.py", "--config", config_name]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=7200,  # 2 hour timeout
            )

            processing_time = time.time() - start_time

            if result.returncode == 0:
                logger.info(
                    f"✅ Completed evaluation: {config_name} ({processing_time:.1f}s)"
                )

                # Try to extract accuracy from output
                accuracy = self._extract_accuracy_from_output(result.stdout)

                return {
                    "config_name": config_name,
                    "success": True,
                    "processing_time": processing_time,
                    "accuracy": accuracy,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                }
            else:
                logger.error(f"❌ Failed evaluation: {config_name}")
                return {
                    "config_name": config_name,
                    "success": False,
                    "processing_time": processing_time,
                    "error": result.stderr,
                    "stdout": result.stdout,
                }

        except subprocess.TimeoutExpired:
            logger.error(f"⏰ Evaluation timeout: {config_name}")
            return {
                "config_name": config_name,
                "success": False,
                "processing_time": time.time() - start_time,
                "error": "Evaluation timeout (2 hours)",
            }
        except Exception as e:
            logger.error(f"❌ Evaluation error: {config_name} - {e}")
            return {
                "config_name": config_name,
                "success": False,
                "processing_time": time.time() - start_time,
                "error": str(e),
            }

    def _extract_accuracy_from_output(self, output: str) -> float:
        """
        Extract accuracy percentage from output

        Args:
            output: Command output string

        Returns:
            Accuracy as float (0.0 to 1.0)
        """
        try:
            # Look for "Overall Accuracy: XX.XX%"
            for line in output.split("\n"):
                if "Overall Accuracy:" in line:
                    # Extract percentage
                    parts = line.split("Overall Accuracy:")
                    if len(parts) > 1:
                        percent_str = parts[1].strip().replace("%", "")
                        return float(percent_str) / 100.0
        except:
            pass

        return 0.0

    def run_batch_sequential(self, config_names: List[str]) -> List[Dict[str, Any]]:
        """
        Run batch evaluation sequentially

        Args:
            config_names: List of configuration names to evaluate

        Returns:
            List of evaluation results
        """
        logger.info(
            f"🔄 Starting sequential batch evaluation of {len(config_names)} configs"
        )

        results = []

        for i, config_name in enumerate(config_names, 1):
            logger.info(f"📝 Processing {i}/{len(config_names)}: {config_name}")

            result = self.run_single_evaluation(config_name)
            results.append(result)

            # Log progress
            successful = sum(1 for r in results if r["success"])
            failed = len(results) - successful
            logger.info(
                f"📊 Progress: {i}/{len(config_names)} ({successful} ✅, {failed} ❌)"
            )

        return results

    def run_batch_parallel(
        self, config_names: List[str], max_workers: int = 2
    ) -> List[Dict[str, Any]]:
        """
        Run batch evaluation in parallel

        Args:
            config_names: List of configuration names to evaluate
            max_workers: Maximum number of parallel workers

        Returns:
            List of evaluation results
        """
        logger.info(f"🔄 Starting parallel batch evaluation with {max_workers} workers")

        results = []

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_config = {
                executor.submit(self.run_single_evaluation, config_name): config_name
                for config_name in config_names
            }

            # Process completed tasks
            for i, future in enumerate(as_completed(future_to_config), 1):
                config_name = future_to_config[future]

                try:
                    result = future.result()
                    results.append(result)

                    # Log progress
                    successful = sum(1 for r in results if r["success"])
                    failed = len(results) - successful
                    logger.info(
                        f"📊 Progress: {i}/{len(config_names)} ({successful} ✅, {failed} ❌)"
                    )

                except Exception as e:
                    logger.error(f"❌ Process execution failed for {config_name}: {e}")
                    results.append(
                        {
                            "config_name": config_name,
                            "success": False,
                            "processing_time": 0,
                            "error": str(e),
                        }
                    )

        return results

    def generate_summary_report(
        self, results: List[Dict[str, Any]], output_file: str = "batch_results.txt"
    ) -> None:
        """
        Generate summary report of batch evaluation

        Args:
            results: List of evaluation results
            output_file: Output file path
        """
        try:
            with open(output_file, "w", encoding="utf-8") as f:
                f.write("📊 Batch Evaluation Summary Report\n")
                f.write("=" * 50 + "\n")
                f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

                # Overall statistics
                total_configs = len(results)
                successful = sum(1 for r in results if r["success"])
                failed = total_configs - successful

                f.write(f"🎯 Overall Statistics:\n")
                f.write(f"  Total Configurations: {total_configs}\n")
                f.write(f"  Successful: {successful}\n")
                f.write(f"  Failed: {failed}\n")
                f.write(f"  Success Rate: {successful/total_configs:.1%}\n\n")

                # Results by configuration
                f.write(f"📋 Results by Configuration:\n")
                f.write("-" * 50 + "\n")

                # Sort by accuracy (successful first, then by accuracy descending)
                sorted_results = sorted(
                    results,
                    key=lambda x: (x["success"], x.get("accuracy", 0)),
                    reverse=True,
                )

                for result in sorted_results:
                    config_name = result["config_name"]

                    if result["success"]:
                        accuracy = result.get("accuracy", 0)
                        processing_time = result["processing_time"]
                        f.write(f"✅ {config_name}:\n")
                        f.write(f"   Accuracy: {accuracy:.2%}\n")
                        f.write(f"   Time: {processing_time:.1f}s\n")
                    else:
                        error = result.get("error", "Unknown error")
                        f.write(f"❌ {config_name}:\n")
                        f.write(f"   Error: {error}\n")
                    f.write("\n")

                # Top performers
                successful_results = [r for r in results if r["success"]]
                if successful_results:
                    f.write(f"🏆 Top Performers:\n")
                    f.write("-" * 30 + "\n")

                    top_results = sorted(
                        successful_results,
                        key=lambda x: x.get("accuracy", 0),
                        reverse=True,
                    )

                    for i, result in enumerate(top_results[:5], 1):
                        accuracy = result.get("accuracy", 0)
                        f.write(f"  {i}. {result['config_name']}: {accuracy:.2%}\n")

            logger.info(f"📁 Summary report saved: {output_file}")

        except Exception as e:
            logger.error(f"❌ Failed to generate summary report: {e}")


def main():
    """Main entry point for batch runner"""
    import argparse

    parser = argparse.ArgumentParser(description="Batch Paper QA Evaluation Runner")
    parser.add_argument(
        "--configs",
        "-c",
        nargs="+",
        help="Specific config names to run (default: all available)",
    )
    parser.add_argument(
        "--parallel", "-p", action="store_true", help="Run evaluations in parallel"
    )
    parser.add_argument(
        "--max-workers",
        "-w",
        type=int,
        default=2,
        help="Maximum number of parallel workers (default: 2)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="batch_results.txt",
        help="Output file for summary report",
    )

    args = parser.parse_args()

    # Initialize batch runner
    runner = BatchRunner()

    # Get configurations to run
    if args.configs:
        config_names = args.configs
    else:
        config_names = runner.config_loader.list_available_configs()

    if not config_names:
        print("❌ No configurations found to run")
        exit(1)

    print(f"🚀 Starting batch evaluation of {len(config_names)} configurations:")
    for config_name in config_names:
        print(f"  - {config_name}")

    # Run batch evaluation
    start_time = time.time()

    if args.parallel:
        results = runner.run_batch_parallel(config_names, args.max_workers)
    else:
        results = runner.run_batch_sequential(config_names)

    total_time = time.time() - start_time

    # Generate summary
    runner.generate_summary_report(results, args.output)

    # Print final summary
    successful = sum(1 for r in results if r["success"])
    failed = len(results) - successful

    print(f"\n🎉 Batch evaluation completed!")
    print(f"📊 Final Results:")
    print(f"  - Total Time: {total_time:.1f}s")
    print(f"  - Successful: {successful}/{len(results)}")
    print(f"  - Failed: {failed}/{len(results)}")
    print(f"  - Success Rate: {successful/len(results):.1%}")

    if successful > 0:
        # Show top performer
        successful_results = [r for r in results if r["success"]]
        best_result = max(successful_results, key=lambda x: x.get("accuracy", 0))
        print(
            f"🏆 Best Performance: {best_result['config_name']} ({best_result.get('accuracy', 0):.2%})"
        )

    print(f"📁 Detailed report saved: {args.output}")


if __name__ == "__main__":
    main()
