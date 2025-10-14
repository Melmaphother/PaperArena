#!/usr/bin/env python3
"""
Simple QA system using SmolAgents CodeAgent with GoogleSearchTool.
This implementation demonstrates a basic ReAct loop for question answering.
"""

import os
from typing import Optional

from dotenv import load_dotenv

from smolagents import (
    CodeAgent, 
    LiteLLMModel, 
    PythonInterpreterTool,
)

from tools.async_web_crawler import CrawlerSearchTool

AUTHORIZED_IMPORTS = [
    "requests",
    "zipfile",
    "os",
    "pandas",
    "numpy",
    "sympy",
    "json",
    "bs4",
    "pubchempy",
    "xml",
    "yfinance",
    "Bio",
    "sklearn",
    "scipy",
    "pydub",
    "io",
    "PIL",
    "chess",
    "PyPDF2",
    "pptx",
    "torch",
    "datetime",
    "fractions",
    "csv",
    "random",
    "re",
    "sys",
    "shutil",
]


def setup_environment() -> None:
    """Load environment variables from .env file.

    Raises:
        ValueError: If required API keys are not found.
    """
    load_dotenv(override=True)

    # Check for required API keys
    required_keys = ["OPENAI_API_KEY", "SERPAPI_API_KEY"]

    missing_keys = []
    for key in required_keys:
        if not os.getenv(key):
            missing_keys.append(key)

    if missing_keys:
        print(f"⚠️ Warning: Missing environment variables: {', '.join(missing_keys)}")
        print("Make sure to set them in your .env file or environment.")


def create_qa_agent(
    model_id: str = "gpt-4o-mini",
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
    search_provider: str = "serpapi",
    max_steps: int = 10,
    verbose: bool = False,
) -> CodeAgent:
    """Create a CodeAgent configured for QA tasks.

    Args:
        model_id: The model ID to use for the LLM.
        api_base: The base URL for the API.
        api_key: The API key for the LLM.
        search_provider: The search provider for GoogleSearchTool.
        max_steps: Maximum number of steps for the agent.
        verbose: Whether to enable verbose output.

    Returns:
        CodeAgent: Configured agent for QA tasks.

    Raises:
        ValueError: If required API keys are missing.
    """
    print(f"🚀 Creating QA agent with model: {model_id}")

    # Create the model
    model_params = {
        "model_id": model_id,
        "max_completion_tokens": 4096,
    }

    if api_base:
        model_params["api_base"] = api_base
    if api_key:
        model_params["api_key"] = api_key

    model = LiteLLMModel(**model_params)

    # Create the search tool
    try:
        # search_tool = GoogleSearchTool(provider=search_provider)
        search_tool = CrawlerSearchTool()
        print(
            f"✅ Successfully initialized GoogleSearchTool with provider: {search_provider}"
        )
    except ValueError as e:
        print(f"❌ Failed to initialize GoogleSearchTool: {e}")
        raise

    # Create the CodeAgent with ReAct loop
    agent = CodeAgent(
        tools=[search_tool],
        model=model,
        max_steps=max_steps,
        verbosity_level=2 if verbose else 1,
        additional_authorized_imports=AUTHORIZED_IMPORTS,
    )

    print(f"✅ CodeAgent created successfully with {max_steps} max steps")
    return agent


def run_qa_session(agent: CodeAgent, question: str) -> str:
    """Run a QA session with the given agent and question.

    Args:
        agent: The CodeAgent to use for answering.
        question: The question to ask.

    Returns:
        str: The agent's answer.
    """
    print(f"🤔 Question: {question}")
    print("🔍 Starting QA session...")

    try:
        answer = agent.run(question)
        print(f"✅ QA session completed successfully")
        return answer
    except Exception as e:
        print(f"❌ Error during QA session: {e}")
        return f"Sorry, I encountered an error while trying to answer your question: {str(e)}"


def main() -> None:
    """Main function to run the QA system."""
    print("🎯 SmolAgents Simple QA System")
    print("=" * 50)

    # Setup environment
    setup_environment()

    # Configuration - modify these as needed
    model_id = "gpt-4o-mini"
    api_base = os.getenv("OPENAI_BASE_URL")
    api_key = os.getenv("OPENAI_API_KEY")
    search_provider = "serpapi"
    max_steps = 10
    verbose = True

    # Example question - change this to any question you want to ask
    question = "What are the latest developments in artificial intelligence in 2025?"

    try:
        # Create the agent
        agent = create_qa_agent(
            model_id=model_id,
            api_base=api_base,
            api_key=api_key,
            search_provider=search_provider,
            max_steps=max_steps,
            verbose=verbose,
        )

        # Run QA session
        answer = run_qa_session(agent, question)

        # Display results
        print("\n" + "=" * 50)
        print("📝 Final Answer:")
        print("-" * 20)
        print(answer)
        print("=" * 50)

    except Exception as e:
        print(f"❌ Fatal error: {e}")
        print("Please check your environment variables and try again.")
        exit(1)


if __name__ == "__main__":
    main()
