"""
Data Analyzer Tool for smolagents
Analyzes figures and tables from parsed PDF content using LLM
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

from smolagents import Tool
from loguru import logger

from .llm_chat import chat_with_llm


class DataAnalyzer(Tool):
    """
    Tool for analyzing figures and tables from parsed PDF content.

    This tool loads the content list JSON file from parsed PDF and analyzes
    specific figures or tables using multimodal LLM capabilities.
    """

    name = "data_analyzer"
    description = (
        "Analyze figures and tables from parsed PDF content. Takes a PDF name, "
        "content type (figure or table), index number, and question as input. "
        "Returns detailed analysis of the specified figure or table using GPT-4o."
    )
    inputs = {
        "pdf_name": {
            "type": "string",
            "description": "The name of the PDF file (without .pdf extension)",
        },
        "content_type": {
            "type": "string",
            "description": "Type of content to analyze: 'figure' or 'table'",
        },
        "index": {
            "type": "integer",
            "description": "Index number of the figure or table (e.g., 1 for Figure 1 or Table 1)",
        },
        "question": {
            "type": "string",
            "description": "Question or analysis request about the figure/table",
        },
    }
    output_type = "string"

    def __init__(self, output_base_dir: Path | str = None):
        """Initialize the data analyzer tool"""
        super().__init__()
        # Fixed paths for PDF processing
        if output_base_dir is None:
            self.output_base_dir = Path(
                "/data/wdy/AgenticPaperQA/paper-prepare/sampled_jsons"
            )
        else:
            self.output_base_dir = Path(output_base_dir)

    def setup(self):
        """Setup method called before first use"""
        logger.info("🔧 Setting up data analyzer tool...")
        self.is_initialized = True

    def _load_content_list(
        self, pdf_name: str
    ) -> tuple[List[Dict[str, Any]], Optional[str]]:
        """
        Load the content list JSON file for the given PDF

        Args:
            pdf_name: Name of the PDF file (without extension)

        Returns:
            Tuple of (content list, error message or None)
        """
        json_path = (
            self.output_base_dir / pdf_name / "auto" / f"{pdf_name}_content_list.json"
        )

        if not json_path.exists():
            return None, f"Content list file not found: {json_path}"

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                content_list = json.load(f)

            logger.info(f"📄 Loaded content list with {len(content_list)} blocks")
            return content_list, None

        except json.JSONDecodeError as e:
            error_msg = f"Failed to parse JSON file: {e}"
            logger.error(f"❌ {error_msg}")
            return None, error_msg
        except Exception as e:
            error_msg = f"Failed to read JSON file: {e}"
            logger.error(f"❌ {error_msg}")
            return None, error_msg

    def _find_figure_by_caption(
        self, content_list: List[Dict[str, Any]], index: int
    ) -> Optional[Dict[str, Any]]:
        """
        Find figure block by matching caption with Figure/Fig index

        Args:
            content_list: List of content blocks
            index: Figure index to search for

        Returns:
            Figure block dictionary or None if not found
        """
        # Pattern to match "Figure X" or "Fig X" or "Fig. X"
        patterns = [
            rf"Figure\s+{index}[\s:.]",
            rf"Fig\s+{index}[\s:.]",
            rf"Fig\.\s+{index}[\s:.]",
        ]

        image_blocks = [block for block in content_list if block.get("type") == "image"]

        for block in image_blocks:
            captions = block.get("image_caption", [])
            if not captions:
                continue

            for caption in captions:
                for pattern in patterns:
                    if re.search(pattern, caption, re.IGNORECASE):
                        logger.info(f"🎯 Found Figure {index} by caption matching")
                        return block

        return None

    def _find_table_by_caption(
        self, content_list: List[Dict[str, Any]], index: int
    ) -> Optional[Dict[str, Any]]:
        """
        Find table block by matching caption with Table index

        Args:
            content_list: List of content blocks
            index: Table index to search for

        Returns:
            Table block dictionary or None if not found
        """
        # Pattern to match "Table X", "Tab X", "Tab. X"
        patterns = [
            rf"Table\s+{index}[\s:.]",
            rf"Tab\s+{index}[\s:.]",
            rf"Tab\.\s+{index}[\s:.]",
        ]

        table_blocks = [block for block in content_list if block.get("type") == "table"]

        for block in table_blocks:
            captions = block.get("table_caption", [])
            if not captions:
                continue

            for caption in captions:
                for pattern in patterns:
                    if re.search(pattern, caption, re.IGNORECASE):
                        logger.info(f"🎯 Found Table {index} by caption matching")
                        break
                else:
                    continue
                return block
            else:
                continue

        return None

    def _get_block_by_index(
        self, content_list: List[Dict[str, Any]], content_type: str, index: int
    ) -> Optional[Dict[str, Any]]:
        """
        Get the nth block of specified type as fallback

        Args:
            content_list: List of content blocks
            content_type: Type of content ("image" or "table")
            index: Index of the block (1-based)

        Returns:
            Block dictionary or None if not found
        """
        blocks = [block for block in content_list if block.get("type") == content_type]

        if len(blocks) >= index:
            logger.info(f"📍 Using {index}th {content_type} block as fallback")
            return blocks[index - 1]  # Convert to 0-based index

        return None

    def _build_figure_prompt(
        self, block: Dict[str, Any], question: str, pdf_name: str
    ) -> tuple[str, Optional[Path]]:
        """
        Build prompt and image path for figure analysis

        Args:
            block: Figure block dictionary
            question: User's question about the figure
            pdf_name: Name of the PDF

        Returns:
            Tuple of (prompt_text, image_path)
        """
        # Get image path
        img_relative_path = block.get("img_path", "")
        if img_relative_path:
            img_path = self.output_base_dir / pdf_name / "auto" / img_relative_path
        else:
            img_path = None

        # Get caption
        captions = block.get("image_caption", [])
        caption_text = "\n".join(captions) if captions else "No caption available"

        # Build prompt
        prompt = f"""Please analyze the following figure and answer the question.

Figure Information:
- Caption: {caption_text}
- Source: {pdf_name}

Question: {question}

Please provide a detailed analysis based on the figure and any available caption information."""

        return prompt, img_path

    def _build_table_prompt(
        self, block: Dict[str, Any], question: str, pdf_name: str
    ) -> str:
        """
        Build prompt for table analysis (text-only)

        Args:
            block: Table block dictionary
            question: User's question about the table
            pdf_name: Name of the PDF

        Returns:
            Prompt text for LLM
        """
        # Get caption
        captions = block.get("table_caption", [])
        caption_text = "\n".join(captions) if captions else "No caption available"

        # Get table body
        table_body = block.get("table_body", "No table content available")

        # Build prompt
        prompt = f"""Please analyze the following table and answer the question.

Table Information:
- Caption: {caption_text}
- Source: {pdf_name}

Table Content (HTML format):
{table_body}

Question: {question}

Please provide a detailed analysis based on the table content and any available caption information."""

        return prompt

    def forward(
        self,
        pdf_name: str,
        content_type: str,
        index: int,
        question: str,
    ) -> str:
        """
        Analyze figure or table from PDF content

        Args:
            pdf_name: Name of the PDF file (without extension)
            content_type: Type of content ("figure" or "table")
            index: Index number of the figure/table
            question: Question about the content

        Returns:
            Analysis result from GPT-4o
        """
        try:
            logger.info(
                f"🔍 Analyzing {content_type} {index} from {pdf_name} with GPT-4o"
            )

            # Validate content type
            if content_type.lower() not in ["figure", "table"]:
                return f"Error: content_type must be 'figure' or 'table', got '{content_type}'"

            # Load content list
            content_list, load_error = self._load_content_list(pdf_name)
            if load_error:
                return f"Error: {load_error}"

            # Find the target block
            target_block = None

            if content_type.lower() == "figure":
                # Try to find by caption first
                target_block = self._find_figure_by_caption(content_list, index)

                # Fallback to index-based selection
                if target_block is None:
                    target_block = self._get_block_by_index(
                        content_list, "image", index
                    )

                if target_block is None:
                    return f"Error: Figure {index} not found in {pdf_name}"

                # Build prompt and analyze
                prompt, img_path = self._build_figure_prompt(
                    target_block, question, pdf_name
                )

                # Use multimodal LLM for figure analysis
                result = chat_with_llm(
                    text_prompt=prompt,
                    image_path=img_path,
                    system_prompt="You are an expert at analyzing scientific figures and diagrams. Provide detailed, accurate analysis based on visual content.",
                )

            else:  # table
                # Try to find by caption first
                target_block = self._find_table_by_caption(content_list, index)

                # Fallback to index-based selection
                if target_block is None:
                    target_block = self._get_block_by_index(
                        content_list, "table", index
                    )

                if target_block is None:
                    return f"Error: Table {index} not found in {pdf_name}"

                # Build prompt and analyze
                prompt = self._build_table_prompt(target_block, question, pdf_name)

                # Use multimodal LLM for table analysis (text only)
                result = chat_with_llm(
                    text_prompt=prompt,
                    system_prompt="You are an expert at analyzing data tables and extracting insights. Provide detailed, accurate analysis based on the table content.",
                )

            logger.info("✅ Analysis completed successfully")
            return result

        except Exception as e:
            error_msg = f"❌ Data analysis failed: {str(e)}"
            logger.error(error_msg)
            return f"Error: {str(e)}"


if __name__ == "__main__":
    # Test the data analyzer tool
    analyzer = DataAnalyzer()
    analyzer.setup()

    from dotenv import load_dotenv
    load_dotenv('/data/wdy/AgenticPaperQA/smolagents/paper_qa/.env', verbose=True)

    # Example usage for figure analysis
    test_pdf_name = "0cEZyhHEks_Taming Knowledge Conflicts in Language Models"  # Replace with actual PDF name

    try:
        result = analyzer.forward(
            pdf_name=test_pdf_name,
            content_type="figure",
            index=1,
            question="What does this figure show and what are the main findings?",
        )
        print("🔍 Figure Analysis Result:")
        print("=" * 50)
        print(result)
    except Exception as e:
        print(f"❌ Figure analysis test failed: {e}")

    # Example usage for table analysis
    try:
        result = analyzer.forward(
            pdf_name=test_pdf_name,
            content_type="table",
            index=1,
            question="What are the key statistics shown in this table?",
        )
        print("\n🔍 Table Analysis Result:")
        print("=" * 50)
        print(result)
    except Exception as e:
        print(f"❌ Table analysis test failed: {e}")
