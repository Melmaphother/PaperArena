#!/usr/bin/env python3
"""
React Framework for Paper QA System
Main entry point for running evaluations with different models and configurations
"""

import os
import sys
import time
import threading
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from omegaconf import DictConfig, OmegaConf
from dotenv import load_dotenv
from loguru import logger

from smolagents import CodeAgent, OpenAIServerModel
from smolagents.memory import ActionStep, TaskStep, PlanningStep
from data_loader import DataLoader, QAItem, PaperInfo
from result_recorder import ResultRecorder
from template import AGENT_TEMPLATE

# Import all available tools
from tools.pdf_extractor import PDFExtractorTool, PDFExtractorRAGTool
from tools.data_analyzer import DataAnalyzer
from tools.cross_ref_lookup import CrossRefLookup
from tools.database_search import DatabaseSearch
from tools.async_web_crawler import SimpleCrawler, CrawlerSearchTool
from tools.mock_web_search import MockWebSearchTool
from tools.mock_local_web_search import MockLocalWebSearchTool
from tools.duckduckgo_search import DuckDuckGoSearchTool
from tools.code_executor import AUTHORIZED_IMPORTS


def load_config() -> DictConfig:
    """
    Load configuration using OmegaConf with command-line overrides

    Returns:
        DictConfig: Loaded configuration
    """
    try:
        # Parse command line arguments
        overrides = []
        config_file = "configs/base_config.yaml"

        # Parse sys.argv for key=value pairs
        for arg in sys.argv[1:]:
            if "=" in arg:
                key, value = arg.split("=", 1)
                if key == "config":
                    config_file = value
                else:
                    overrides.append(arg)

        # Load base config
        if Path(config_file).exists():
            cfg = OmegaConf.load(config_file)
        else:
            # Default configuration
            cfg = OmegaConf.create(
                {
                    "model_name": "gpt-4.1-2025-04-14",
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "max_tokens": 4096,
                    "max_steps": 15,
                    "verbosity_level": 1,
                    "num_threads": 1,
                    "results_dir": "results",
                    "filter_conference": None,
                    "filter_difficulty": None,
                    "filter_qa_type": None,
                    "data_dir": "/data/wdy/AgenticPaperQA/paper-prepare/qa",
                }
            )

        # Apply command-line overrides
        if overrides:
            override_cfg = OmegaConf.from_dotlist(overrides)
            cfg = OmegaConf.merge(cfg, override_cfg)

        logger.info(f"✅ Config loaded: {cfg.model_name}")
        return cfg

    except Exception as e:
        logger.warning(f"⚠️ Failed to load config: {e}, using defaults")
        return OmegaConf.create(
            {
                "model_name": "gpt-4.1-2025-04-14",
                "temperature": 0.7,
                "top_p": 0.9,
                "max_tokens": 4096,
                "max_steps": 15,
                "verbosity_level": 1,
                "num_threads": 1,
                "results_dir": "results",
                "data_dir": "/data/wdy/AgenticPaperQA/paper-prepare/qa",
            }
        )


def convert_response_to_str(obj):
    """
    Recursively converts numpy types in a dictionary or list to native python strings.
    """
    if isinstance(obj, dict):
        return str({key: convert_response_to_str(value) for key, value in obj.items()})
    elif isinstance(obj, list):
        return str([convert_response_to_str(element) for element in obj])
    # Check for a tuple of numpy types and convert them all to string
    elif isinstance(obj, (np.integer, np.floating, np.ndarray)):
        return str(obj)
    else:
        return str(obj)


class PaperQAFramework:
    """
    Main framework for running Paper QA evaluations
    """

    def __init__(self, config: DictConfig):
        """
        Initialize the framework

        Args:
            config: Evaluation configuration
        """
        self.config = config
        self.data_loader = None
        self.result_recorder = None
        self.agent = None

        # Threading lock for shared resources
        self._lock = threading.Lock()

        logger.info(f"🚀 PaperQA Framework initialized with model: {config.model_name}")

    def setup_environment(self) -> None:
        """Load environment variables and validate setup"""
        load_dotenv(override=True)

        # Check for required API keys
        required_keys = ["OPENAI_API_KEY", "OPENAI_BASE_URL"]
        missing_keys = []

        for key in required_keys:
            if not os.getenv(key):
                missing_keys.append(key)

        if missing_keys:
            logger.warning(
                f"⚠️ Missing environment variables: {', '.join(missing_keys)}"
            )
        else:
            logger.info("✅ Environment variables loaded successfully")

    def load_data(self) -> bool:
        """
        Load QA dataset

        Returns:
            bool: Success status
        """
        logger.info("📂 Loading QA dataset...")

        self.data_loader = DataLoader(data_dir=self.config.data_dir)
        success, error = self.data_loader.load_all_data()

        if not success:
            logger.error(f"❌ Failed to load dataset: {error}")
            return False

        logger.info(
            f"✅ Dataset loaded: {len(self.data_loader.qa_items)} QA items from {len(self.data_loader.papers)} papers"
        )
        return True

    def create_agent(self) -> CodeAgent:
        """Create and configure the CodeAgent with all tools"""
        logger.info("🔧 Creating CodeAgent...")

        # Create model using OpenAIServerModel
        api_key = os.getenv("OPENAI_API_KEY")
        api_base = os.getenv("OPENAI_BASE_URL")

        model = OpenAIServerModel(
            model_id=self.config.model_name,
            api_base=api_base,
            api_key=api_key,
            stop=None # o4-mini don't have stop parameter
        )

        # crawler = SimpleCrawler(serpapi_key=os.getenv("SERPAPI_API_KEY"))

        # Initialize tools
        tools = [
            PDFExtractorTool(pdf_base_dir=self.config.pdf_base_dir, output_base_dir=self.config.output_base_dir),
            PDFExtractorRAGTool(pdf_base_dir=self.config.pdf_base_dir, output_base_dir=self.config.output_base_dir),
            DataAnalyzer(output_base_dir=self.config.output_base_dir),
            CrossRefLookup(output_base_dir=self.config.output_base_dir),
            DatabaseSearch(database_dir=self.config.database_dir),
            # CrawlerSearchTool(crawler=crawler),
            DuckDuckGoSearchTool(output_directory=self.config.output_base_dir),
            # MockWebSearchTool(auth_token=os.getenv("MOCK_WEB_SEARCH_AUTH_TOKEN")),
        ]

        # Create agent
        agent = CodeAgent(
            tools=tools,
            model=model,
            max_steps=self.config.max_steps,
            verbosity_level=self.config.verbosity_level,
            additional_authorized_imports=AUTHORIZED_IMPORTS,
        )

        logger.info(f"✅ Agent created with {len(tools)} tools")
        return agent

    def initialize_result_recorder(self) -> None:
        """Initialize result recorder for the current model"""
        self.result_recorder = ResultRecorder(
            model_name=self.config.model_name, results_dir=self.config.results_dir
        )
        logger.info("✅ Result recorder initialized")

    def get_filtered_qa_items(self) -> List[Tuple[QAItem, PaperInfo]]:
        """
        Get filtered QA items based on configuration

        Returns:
            List of (QAItem, PaperInfo) tuples
        """
        qa_items = self.data_loader.get_qa_items(
            conference=self.config.filter_conference,
            difficulty=self.config.filter_difficulty,
            qa_type=self.config.filter_qa_type,
        )

        # Get corresponding paper info
        qa_with_papers = []
        for qa_item in qa_items:
            paper_info = self.data_loader.get_paper_info(qa_item.pdf_name)
            if paper_info:
                qa_with_papers.append((qa_item, paper_info))

        logger.info(f"🔍 Filtered to {len(qa_with_papers)} QA items")
        return qa_with_papers

    def get_remaining_qa_items(
        self, qa_with_papers: List[Tuple[QAItem, PaperInfo]]
    ) -> List[Tuple[QAItem, PaperInfo]]:
        """
        Get QA items that haven't been completed yet (checkpoint resume)

        Args:
            qa_with_papers: All QA items to process

        Returns:
            List of remaining QA items
        """
        completed = self.result_recorder.get_completed_qa_ids()

        remaining = []
        for qa_item, paper_info in qa_with_papers:
            if qa_item.qa_id not in completed.get(qa_item.pdf_name, []):
                remaining.append((qa_item, paper_info))

        completed_count = len(qa_with_papers) - len(remaining)
        logger.info(
            f"📊 Resume from checkpoint: {completed_count} completed, {len(remaining)} remaining"
        )

        return remaining

    def construct_question_prompt(self, qa_item: QAItem) -> str:
        """
        Construct the full question prompt using the template

        Args:
            qa_item: QA item

        Returns:
            str: Formatted question prompt
        """
        return AGENT_TEMPLATE.substitute(
            pdf_name=qa_item.pdf_name,
            qa_type=qa_item.qa_type,
            question=qa_item.question,
        )
    
    def extract_intermediate_steps(self):
        intermediate_steps = []
        for memory_step in self.agent.memory.steps:
            memory_step.model_input_messages = None
            step_dict = memory_step.dict()
            if isinstance(memory_step, ActionStep):
                step_dict["step_type"] = "action"
                step_dict.pop("model_output_message", None)
            elif isinstance(memory_step, TaskStep):
                step_dict["step_type"] = "task"
            elif isinstance(memory_step, PlanningStep):
                step_dict["step_type"] = "planning"
                step_dict.pop("model_output_message_facts", None)
                step_dict.pop("model_output_message_plan", None)
            else:
                step_dict["step_type"] = "unknown"
            intermediate_steps.append(step_dict)
        return intermediate_steps

    def process_single_qa(
        self, qa_item: QAItem, paper_info: PaperInfo
    ) -> Optional[Dict[str, Any]]:
        """
        Process a single QA item

        Args:
            qa_item: QA item to process
            paper_info: Paper information

        Returns:
            Dict with processing results or None if failed
        """
        start_time = time.time()

        try:
            logger.info(f"🤔 Processing {qa_item.pdf_name}/QA-{qa_item.qa_id}")

            # Construct question prompt
            question_prompt = self.construct_question_prompt(qa_item)

            # Run agent
            agent_response = self.agent.run(question_prompt)
            agent_response = convert_response_to_str(agent_response)

            processing_time = time.time() - start_time

            # Record result
            result = self.result_recorder.record_result(
                model=str(self.agent.model),
                qa_item=qa_item,
                paper_info=paper_info,
                agent_response=agent_response,
                intermediate_steps=self.extract_intermediate_steps(),
                start_time=start_time,
                processing_time=processing_time,
            )

            return {
                "success": True,
                "qa_id": qa_item.qa_id,
                "pdf_name": qa_item.pdf_name,
                "is_correct": result.is_correct,
                "processing_time": processing_time,
            }

        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"❌ Failed {qa_item.pdf_name}/QA-{qa_item.qa_id}: {e}")

            return {
                "success": False,
                "qa_id": qa_item.qa_id,
                "pdf_name": qa_item.pdf_name,
                "error": str(e),
                "processing_time": processing_time,
            }

    def run_single_threaded(
        self, qa_with_papers: List[Tuple[QAItem, PaperInfo]]
    ) -> Dict[str, Any]:
        """
        Run evaluation in single-threaded mode

        Args:
            qa_with_papers: List of QA items to process

        Returns:
            Dict with summary results
        """
        logger.info(
            f"🔄 Starting single-threaded evaluation of {len(qa_with_papers)} items"
        )

        results = []
        successful = 0
        failed = 0

        for i, (qa_item, paper_info) in enumerate(qa_with_papers, 1):
            logger.info(f"📝 Processing item {i}/{len(qa_with_papers)}")

            result = self.process_single_qa(qa_item, paper_info)
            results.append(result)

            if result and result["success"]:
                successful += 1
            else:
                failed += 1

            # Print progress every 10 items
            if i % 10 == 0:
                logger.info(
                    f"📊 Progress: {i}/{len(qa_with_papers)} ({successful} ✅, {failed} ❌)"
                )

        return {
            "total_processed": len(results),
            "successful": successful,
            "failed": failed,
            "results": results,
        }

    def run_multi_threaded(
        self, qa_with_papers: List[Tuple[QAItem, PaperInfo]]
    ) -> Dict[str, Any]:
        """
        Run evaluation in multi-threaded mode

        Args:
            qa_with_papers: List of QA items to process

        Returns:
            Dict with summary results
        """
        logger.info(
            f"🔄 Starting multi-threaded evaluation with {self.config.num_threads} threads"
        )

        results = []
        successful = 0
        failed = 0

        with ThreadPoolExecutor(max_workers=self.config.num_threads) as executor:
            # Submit all tasks
            future_to_qa = {
                executor.submit(self.process_single_qa, qa_item, paper_info): (
                    qa_item,
                    paper_info,
                )
                for qa_item, paper_info in qa_with_papers
            }

            # Process completed tasks
            for i, future in enumerate(as_completed(future_to_qa), 1):
                qa_item, paper_info = future_to_qa[future]

                try:
                    result = future.result()
                    results.append(result)

                    if result and result["success"]:
                        successful += 1
                    else:
                        failed += 1

                except Exception as e:
                    logger.error(
                        f"❌ Thread execution failed for {qa_item.pdf_name}/QA-{qa_item.qa_id}: {e}"
                    )
                    failed += 1
                    results.append(
                        {
                            "success": False,
                            "qa_id": qa_item.qa_id,
                            "pdf_name": qa_item.pdf_name,
                            "error": str(e),
                            "processing_time": 0,
                        }
                    )

                # Print progress
                if i % 10 == 0:
                    logger.info(
                        f"📊 Progress: {i}/{len(qa_with_papers)} ({successful} ✅, {failed} ❌)"
                    )

        return {
            "total_processed": len(results),
            "successful": successful,
            "failed": failed,
            "results": results,
        }

    def run_evaluation(self) -> Dict[str, Any]:
        """
        Run the complete evaluation pipeline

        Returns:
            Dict with evaluation summary
        """
        logger.info("🎯 Starting Paper QA Evaluation")

        try:
            # Setup
            self.setup_environment()

            if not self.load_data():
                return {"success": False, "error": "Failed to load data"}

            self.agent = self.create_agent()
            self.initialize_result_recorder()

            # Get QA items to process
            qa_with_papers = self.get_filtered_qa_items()
            if not qa_with_papers:
                logger.warning("⚠️ No QA items to process after filtering")
                return {"success": False, "error": "No QA items to process"}

            # Resume from checkpoint
            remaining_qa = self.get_remaining_qa_items(qa_with_papers)
            if not remaining_qa:
                logger.info("✅ All QA items already completed!")

            # Run evaluation
            if self.config.num_threads == 1:
                summary = self.run_single_threaded(remaining_qa)
            else:
                summary = self.run_multi_threaded(remaining_qa)

            # Calculate final metrics
            logger.info("📊 Calculating final metrics...")
            metrics = self.result_recorder.calculate_and_save_metrics()

            # Print summary
            logger.info("🎉 Evaluation completed!")
            logger.info(f"📈 Final Results:")
            logger.info(f"  - Total Processed: {summary['total_processed']}")
            logger.info(f"  - Successful: {summary['successful']}")
            logger.info(f"  - Failed: {summary['failed']}")
            if metrics:
                logger.info(
                    f"  - Overall Accuracy: {metrics.get('overall_accuracy', 0):.2%}"
                )

            return {
                "success": True,
                "summary": summary,
                "metrics": metrics,
            }

        except Exception as e:
            logger.error(f"❌ Evaluation failed: {e}")
            return {"success": False, "error": str(e)}


def main():
    """Main entry point"""
    # Load configuration
    config = load_config()

    print(f"🚀 Starting evaluation with model: {config.model_name}")

    # Run evaluation
    framework = PaperQAFramework(config)
    result = framework.run_evaluation()

    if result["success"]:
        print("\n🎉 Evaluation completed successfully!")
        if "metrics" in result and result["metrics"]:
            print(
                f"📊 Overall Accuracy: {result['metrics'].get('overall_accuracy', 0):.2%}"
            )
    else:
        print(f"\n❌ Evaluation failed: {result.get('error', 'Unknown error')}")
        exit(1)


if __name__ == "__main__":
    main()
