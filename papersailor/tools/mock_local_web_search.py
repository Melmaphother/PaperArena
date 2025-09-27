"""
Mock Local Web Search Tool for Paper QA System
Provides web search functionality using local web crawler utilities
"""

import json
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from smolagents import Tool
from loguru import logger

# Import the web crawler utilities
from .web_crawler_utils import simulate_search_api


class MockLocalWebSearchTool(Tool):
    """
    Mock local web search tool that directly uses web crawler utilities
    
    This tool uses the local web_crawler_utils functions to perform web searches
    without requiring an external API service.
    """

    name = "web_search"
    description = "Perform a web search query (think a google search) and returns the search results."
    inputs = {
        "query": {"type": "string", "description": "The web search query to perform."},
        "filter_year": {
            "type": "string",
            "description": "[Optional parameter]: filter the search results to only include pages from a specific year. For example, '2020' will only include pages from 2020. Make sure to use this parameter if you're trying to search for articles from a specific date!",
            "nullable": True,
        },
    }
    output_type = "string"

    def __init__(
        self,
        top_k: int = 10,
        proxy: Optional[str] = None,
        use_concurrent: bool = True,
        max_workers: int = 3,
        output_directory: Optional[str] = None,
        save_results: bool = True,
    ):
        """
        Initialize the mock local web search tool

        Args:
            top_k: Number of search results to return
            proxy: Optional proxy server to use
            use_concurrent: Whether to use concurrent scraping
            max_workers: Maximum number of concurrent workers for scraping
            output_directory: Directory to save search results (defaults to ./search_outputs)
            save_results: Whether to save search results to JSONL files
        """
        super().__init__()
        self.top_k = top_k
        self.proxy = proxy
        self.use_concurrent = use_concurrent
        self.max_workers = max_workers
        self.save_results = save_results
        self.history = []  # Track search history like async_web_crawler
        
        # Set up output directory
        if output_directory is None:
            self.output_directory = Path.cwd() / "search_outputs"
        else:
            self.output_directory = Path(output_directory)
        
        # Create output directory if it doesn't exist
        if self.save_results:
            self.output_directory.mkdir(parents=True, exist_ok=True)
            logger.info(f"📁 Search results will be saved to: {self.output_directory}")

    def _check_history(self, query: str) -> str:
        """
        Check search history and add timestamp (similar to async_web_crawler)

        Args:
            query: Search query to check

        Returns:
            Header string with history information
        """
        header = ""
        for i in range(len(self.history) - 1, -1, -1):
            if self.history[i][0] == query:
                time_diff = round(time.time() - self.history[i][1])
                header += (
                    f"You previously searched for this query {time_diff} seconds ago.\n"
                )
                return header

        # Add current search to history
        self.history.append((query, time.time()))
        return header

    def _sanitize_filename(self, query: str) -> str:
        """
        Clean a string to be used as a valid filename
        
        Args:
            query: The search query string
            
        Returns:
            Sanitized filename string
        """
        # Remove invalid filename characters
        sanitized = re.sub(r'[\\/*?:"<>|]', "", query)
        # Replace spaces with underscores
        sanitized = sanitized.replace(" ", "_")
        # Limit length to avoid filesystem issues
        return sanitized[:100]

    def _save_results_to_jsonl(
        self, query: str, results: List[Dict[str, Any]], filter_year: Optional[str] = None
    ) -> Optional[str]:
        """
        Save search results to a JSONL file
        
        Args:
            query: The search query
            results: List of search result dictionaries
            filter_year: Optional year filter used in search
            
        Returns:
            Path to saved file or None if saving failed
        """
        if not self.save_results or not results:
            return None
            
        try:
            # Generate filename
            filename_base = self._sanitize_filename(query)
            if filter_year:
                filename_base += f"_year_{filter_year}"
            filename = f"{filename_base}.jsonl"
            
            # Create full file path
            file_path = self.output_directory / filename
            
            # Save results to JSONL file
            with open(file_path, 'w', encoding='utf-8') as f:
                for item in results:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
            logger.info(f"💾 Search results saved to: {file_path}")
            return str(file_path)
            
        except Exception as e:
            logger.error(f"❌ Failed to save search results: {e}")
            return None

    def _format_results_to_string(
        self, query: str, results: List[Dict[str, Any]]
    ) -> str:
        """
        Format search results into readable string (similar to async_web_crawler)

        Args:
            query: Original search query
            results: List of search result dictionaries

        Returns:
            Formatted string with search results
        """
        if not results:
            return (
                f"No results found for query: '{query}'. Try with a more general query."
            )

        web_snippets = []

        for result in results:
            idx = result.get("idx", "")
            title = result.get("title", "No title")
            date = result.get("date", "")
            google_snippet = result.get("google_snippet", "")
            subpage_snippet = result.get("subpage_snippet", "")
            source = result.get("source", "")
            link = result.get("link", "")

            # Format date
            date_str = f"\nDate published: {date}" if date else ""

            # Format source
            source_str = f"\nSource: {source}" if source else ""

            # Combine snippets - prioritize subpage_snippet if available, fallback to google_snippet
            snippet = subpage_snippet if subpage_snippet else google_snippet
            snippet_str = f"\n{snippet}" if snippet else ""

            # Format single result (similar to async_web_crawler format)
            formatted_result = (
                f"{idx}. [{title}]({link}){date_str}{source_str}{snippet_str}"
            )

            # Clean up any unwanted content
            formatted_result = formatted_result.replace(
                "Your browser can't play this video.", ""
            )

            web_snippets.append(formatted_result)

        # Create final content string
        content = (
            f"A web search for '{query}' found {len(web_snippets)} results:\n\n## Web Results\n"
            + "\n\n".join(web_snippets)
        )

        return content

    def forward(self, query: str, filter_year: Optional[str] = None) -> str:
        """
        Execute web search and return formatted results

        Args:
            query: Search query string
            filter_year: Optional year filter as string

        Returns:
            Formatted search results as string
        """
        try:
            logger.info(f"🔍 Starting local web search for query: '{query}'")
            
            # Check search history
            history_header = self._check_history(query)

            # Convert filter_year to int if provided
            year_filter = None
            if filter_year is not None:
                try:
                    year_filter = int(filter_year)
                    logger.info(f"📅 Using year filter: {year_filter}")
                except ValueError:
                    logger.warning(f"⚠️ Invalid year filter '{filter_year}', ignoring")

            # Use web_crawler_utils to perform search
            logger.info(f"🚀 Calling simulate_search_api with top_k={self.top_k}")
            results = simulate_search_api(
                query=query,
                top_k=self.top_k,
                proxy=self.proxy,
                filter_year=year_filter,
                use_concurrent=self.use_concurrent,
                max_workers=self.max_workers,
            )

            logger.info(f"📊 Retrieved {len(results)} search results")

            # Save results to JSONL file
            saved_file_path = self._save_results_to_jsonl(query, results, filter_year)

            # Format results to string
            formatted_results = self._format_results_to_string(query, results)

            # Add file save information to the output if results were saved
            if saved_file_path:
                save_info = f"\n💾 Search results have been saved to: {saved_file_path}\n"
                formatted_results = save_info + formatted_results

            # Combine history header with results
            return history_header + formatted_results

        except Exception as e:
            error_msg = f"❌ Local web search failed for query '{query}': {str(e)}"
            logger.error(error_msg)
            return error_msg


if __name__ == "__main__":
    """Test the mock local web search tool"""

    # Test cases
    test_cases = [
        {"query": "Python programming", "filter_year": "2024"},
        {"query": "machine learning", "filter_year": None},
        {"query": "transformer architecture deep learning", "filter_year": "2023"},
    ]

    try:
        print("🚀 Testing Mock Local Web Search Tool...")
        
        # Create tool with custom output directory for testing
        test_output_dir = Path.cwd() / "test_search_outputs"
        tool = MockLocalWebSearchTool(
            top_k=3,  # Use smaller number for testing
            use_concurrent=True,
            max_workers=2,
            output_directory=str(test_output_dir),
            save_results=True,
        )
        
        print(f"📁 Test results will be saved to: {test_output_dir}")

        for i, test_case in enumerate(test_cases, 1):
            print(f"\n📝 Test Case {i}:")
            print(f"Query: {test_case['query']}")
            print(f"Filter Year: {test_case['filter_year']}")

            try:
                result = tool.forward(
                    query=test_case["query"],
                    filter_year=test_case["filter_year"],
                )

                print("📊 Search Result:")
                print("=" * 50)
                print(result)
                print("=" * 50)

            except Exception as e:
                print(f"❌ Error in test case {i}: {e}")

            # Add delay between tests to be polite
            if i < len(test_cases):
                print("⏳ Waiting 5 seconds before next test...")
                time.sleep(5)

        # Check saved files
        print(f"\n📋 Checking saved files in {test_output_dir}:")
        if test_output_dir.exists():
            saved_files = list(test_output_dir.glob("*.jsonl"))
            if saved_files:
                print(f"✅ Found {len(saved_files)} saved JSONL files:")
                for file_path in saved_files:
                    file_size = file_path.stat().st_size
                    print(f"  - {file_path.name} ({file_size} bytes)")
            else:
                print("⚠️ No JSONL files found in output directory")
        else:
            print("⚠️ Output directory was not created")

    except Exception as e:
        print(f"❌ Failed to initialize mock local search tool: {e}")
        print("💡 Make sure Chrome is installed and accessible")
