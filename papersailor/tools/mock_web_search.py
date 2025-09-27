"""
Mock Web Search Tool for Paper QA System
Provides web search functionality via HTTP API to mock search service
"""

import json
import requests
import time
from typing import Dict, List, Any, Optional
from smolagents import Tool
from loguru import logger


class MockWebSearchTool(Tool):
    """
    Mock web search tool that communicates with a local search service

    This tool sends HTTP requests to a mock search API service and formats
    the response into a readable string format similar to async_web_crawler.
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
        api_url: str = "http://127.0.0.1:8765/api/enqueue",
        auth_token: str = "12345",
        top_k: int = 10,
        timeout: int = 100,
        base_remote_outputs: str = "/data/wdy/AgenticPaperQA/smolagents/paper_qa/Mock-Google-Search-API/remote_outputs",
    ):
        """
        Initialize the mock web search tool

        Args:
            api_url: URL of the mock search API service
            auth_token: Authentication token for the API
            timeout: Request timeout in seconds
        """
        super().__init__()
        self.api_url = api_url
        self.auth_token = auth_token
        self.top_k = top_k
        self.timeout = timeout
        self.history = []  # Track search history like async_web_crawler
        self.base_remote_outputs = base_remote_outputs

    def _wait_for_results(
        self, task_id: str, max_wait_time: int = 60, poll_interval: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Wait for task completion and read results from JSONL file

        Args:
            task_id: Task ID returned by the API
            max_wait_time: Maximum time to wait in seconds
            poll_interval: Time between polls in seconds

        Returns:
            List of search result dictionaries

        Raises:
            Exception: If task doesn't complete within max_wait_time or file read fails
        """
        import os
        from pathlib import Path

        result_file = Path(self.base_remote_outputs) / f"{task_id}.jsonl"
        start_time = time.time()

        logger.info(f"⏳ Waiting for task {task_id} to complete...")

        while time.time() - start_time < max_wait_time:
            if result_file.exists():
                try:
                    # Read JSONL file
                    results = []
                    with open(result_file, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                results.append(json.loads(line))

                    if results:
                        logger.info(
                            f"✅ Task {task_id} completed, found {len(results)} results"
                        )
                        return results
                    else:
                        logger.info(
                            f"📄 Task {task_id} file exists but empty, continuing to wait..."
                        )

                except (json.JSONDecodeError, IOError) as e:
                    logger.warning(
                        f"⚠️ Error reading result file: {e}, continuing to wait..."
                    )

            # Wait before next poll
            time.sleep(poll_interval)
            logger.info(
                f"⏳ Still waiting for task {task_id}... ({int(time.time() - start_time)}s elapsed)"
            )

        # Timeout
        raise Exception(
            f"Task {task_id} did not complete within {max_wait_time} seconds"
        )

    def _make_search_request(
        self, query: str, filter_year: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Make HTTP request to the mock search API

        Args:
            query: Search query string
            filter_year: Optional year filter as string

        Returns:
            List of search result dictionaries

        Raises:
            Exception: If API request fails or returns invalid response
        """
        # Prepare request payload
        payload = {"query": query, "top_k": self.top_k}

        if filter_year is not None:
            payload["filter_year"] = filter_year

        # Prepare headers
        headers = {"Content-Type": "application/json", "X-Auth-Token": self.auth_token}

        try:
            logger.info(
                f"🔍 Making search request: query='{query}', top_k={self.top_k}"
            )

            # Make HTTP POST request
            response = requests.post(
                self.api_url, json=payload, headers=headers, timeout=self.timeout
            )

            # Check response status
            if response.status_code != 200:
                raise Exception(
                    f"API request failed with status {response.status_code}: {response.text}"
                )

            # Parse JSON response to get task_id
            task_id_json = response.json()
            task_id = task_id_json.get("task_id")
            logger.info(f"📋 Received task ID: {task_id}")

            if not task_id:
                raise Exception("No task_id received from API")

            # Wait for task completion and read results from file
            results = self._wait_for_results(task_id)
            logger.info(f"📊 Retrieved {len(results)} search results")

            # Validate response format
            if not isinstance(results, list):
                # If it's a dict, maybe the results are nested
                if isinstance(results, dict):
                    print(f"Response is dict with keys: {list(results.keys())}")
                    # Try common key patterns
                    for key in ["results", "data", "items", "search_results"]:
                        if key in results:
                            print(f"Found '{key}' in response, using it as results")
                            results = results[key]
                            break
                    else:
                        raise Exception(
                            f"API response is dict but no expected keys found. Keys: {list(results.keys())}"
                        )
                else:
                    raise Exception(
                        f"API response should be a list of results, got {type(results)}: {results}"
                    )

            logger.info(f"✅ Search request successful: {len(results)} results")
            return results

        except requests.exceptions.Timeout:
            raise Exception(f"Search request timed out after {self.timeout} seconds")
        except requests.exceptions.ConnectionError:
            raise Exception(f"Could not connect to search API at {self.api_url}")
        except requests.exceptions.RequestException as e:
            raise Exception(f"Search request failed: {str(e)}")
        except Exception as e:
            raise Exception(f"Search API error: {str(e)}")

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
            # Check search history
            history_header = self._check_history(query)

            # Make search request
            results = self._make_search_request(query, filter_year)

            # Format results to string
            formatted_results = self._format_results_to_string(query, results)

            # Combine history header with results
            return history_header + formatted_results

        except Exception as e:
            error_msg = f"❌ Mock web search failed for query '{query}': {str(e)}"
            logger.error(error_msg)
            return error_msg


if __name__ == "__main__":
    """Test the mock web search tool"""

    # Test cases
    test_cases = [
        {"query": "Python programming", "filter_year": "2024"},
        {"query": "machine learning", "filter_year": None},
    ] * 10

    try:
        print("🚀 Testing Mock Web Search Tool...")
        tool = MockWebSearchTool()

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

    except Exception as e:
        print(f"❌ Failed to initialize mock search tool: {e}")
        print("💡 Make sure the mock search API is running at http://127.0.0.1:8765")
