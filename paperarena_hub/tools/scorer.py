"""
LLM as Judge module for evaluating responses against ground truth.

This module provides functionality to evaluate the correctness of responses
by comparing them with ground truth answers using an LLM as a judge.
"""

import json
import os
from typing import Dict, Any, Optional
from dataclasses import dataclass

from openai import OpenAI


@dataclass
class JudgeResult:
    """Result of LLM judge evaluation.

    Args:
        correct: Whether the response is correct ("true" or "false")
        reasoning: Explanation of the judgment
        confidence: Confidence score from 0-10
    """

    correct: str
    reasoning: str
    confidence: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "correct": self.correct,
            "reasoning": self.reasoning,
            "confidence": self.confidence,
        }


class LLMScorer:
    """LLM as Judge scorer for evaluating response correctness.

    This class uses OpenAI's GPT-4o-mini model to evaluate whether a given
    response correctly answers a question when compared to ground truth.
    """

    def __init__(self, model: str = "gpt-4o-mini-2024-07-18"):
        """Initialize the LLM scorer.

        Args:
            model: OpenAI model to use for scoring

        Raises:
            ValueError: If required environment variables are not set
        """
        self.model = model
        self._init_openai_client()

    def _init_openai_client(self) -> None:
        """Initialize OpenAI client from environment variables."""
        base_url = os.getenv("OPENAI_BASE_URL")
        api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")

        # Initialize client with optional base_url
        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url

        self.client = OpenAI(**client_kwargs)

    def _create_judge_prompt(
        self, question: str, response: str, ground_truth: str
    ) -> str:
        """Create the judge prompt for evaluation.

        Args:
            question: The original question
            response: The response to evaluate
            ground_truth: The correct answer

        Returns:
            Formatted prompt string
        """
        return f"""You are an expert judge evaluating the correctness of responses to questions.

Your task is to determine if the given response correctly answers the question when compared to the ground truth answer.

**Question:** {question}

**Ground Truth Answer:** {ground_truth}

**Response to Evaluate:** {response}

Please evaluate whether the response correctly answers the question. Consider:
1. Factual accuracy compared to ground truth
2. Completeness of the answer
3. Relevance to the question asked

Provide your judgment in the following JSON format:
{{
    "correct": "true" or "false",
    "reasoning": "Detailed explanation of your judgment",
    "confidence": <integer from 0 to 10>
}}

Where:
- "correct": "true" if the response correctly answers the question, "false" otherwise
- "reasoning": Your detailed explanation for the judgment
- "confidence": Your confidence level (0=no confidence, 10=completely confident)

Respond with ONLY the JSON object, no additional text."""

    def _extract_json_response(self, response_text: str) -> Dict[str, Any]:
        """Extract and validate JSON from LLM response.

        Args:
            response_text: Raw response from LLM

        Returns:
            Parsed JSON dictionary

        Raises:
            ValueError: If JSON cannot be parsed or is invalid
        """
        try:
            # Try to find JSON in the response
            response_text = response_text.strip()

            # Handle cases where response might have extra text
            start_idx = response_text.find("{")
            end_idx = response_text.rfind("}")

            if start_idx == -1 or end_idx == -1:
                raise ValueError("No JSON object found in response")

            json_str = response_text[start_idx : end_idx + 1]
            result = json.loads(json_str)

            # Validate required fields
            required_fields = ["correct", "reasoning", "confidence"]
            for field in required_fields:
                if field not in result:
                    raise ValueError(f"Missing required field: {field}")

            # Validate field types and values
            if result["correct"] not in ["true", "false"]:
                raise ValueError("'correct' field must be 'true' or 'false'")

            if not isinstance(result["reasoning"], str):
                raise ValueError("'reasoning' field must be a string")

            if not isinstance(result["confidence"], int) or not (
                0 <= result["confidence"] <= 10
            ):
                raise ValueError(
                    "'confidence' field must be an integer between 0 and 10"
                )

            return result

        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON response: {e}")

    def score(
        self, question: str, response: str, ground_truth: str, temperature: float = 0.0
    ) -> JudgeResult:
        """Score a response against ground truth using LLM as judge.

        Args:
            question: The original question
            response: The response to evaluate
            ground_truth: The correct answer
            temperature: Temperature for LLM generation (default: 0.0 for deterministic)

        Returns:
            JudgeResult containing evaluation results

        Raises:
            Exception: If LLM call fails or response parsing fails
        """
        try:
            prompt = self._create_judge_prompt(question, response, ground_truth)

            # Make API call to OpenAI
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=1000,
            )

            response_text = completion.choices[0].message.content
            if not response_text:
                raise ValueError("Empty response from LLM")

            # Extract and validate JSON
            result_dict = self._extract_json_response(response_text)

            return JudgeResult(
                correct=result_dict["correct"],
                reasoning=result_dict["reasoning"],
                confidence=result_dict["confidence"],
            )

        except Exception as e:
            raise Exception(f"Failed to score response: {e}")


if __name__ == "__main__":
    """Test the LLM scorer with example data."""

    # Test data
    test_cases = [
        {
            "question": "What is the capital of France?",
            "response": "The capital of France is Paris.",
            "ground_truth": "Paris",
        },
        {
            "question": "What is 2 + 2?",
            "response": "2 + 2 equals 5.",
            "ground_truth": "4",
        },
        {
            "question": "Who wrote Romeo and Juliet?",
            "response": "William Shakespeare wrote Romeo and Juliet.",
            "ground_truth": "William Shakespeare",
        },
    ]

    try:
        print("🚀 Testing LLM Scorer...")
        scorer = LLMScorer()

        for i, test_case in enumerate(test_cases, 1):
            print(f"\n📝 Test Case {i}:")
            print(f"Question: {test_case['question']}")
            print(f"Response: {test_case['response']}")
            print(f"Ground Truth: {test_case['ground_truth']}")

            try:
                result = scorer.score(
                    question=test_case["question"],
                    response=test_case["response"],
                    ground_truth=test_case["ground_truth"],
                )

                print("📊 Evaluation Result:")
                print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))

            except Exception as e:
                print(f"❌ Error evaluating test case {i}: {e}")

    except Exception as e:
        print(f"❌ Failed to initialize scorer: {e}")
        print("💡 Make sure to set OPENAI_API_KEY environment variable")
        print("💡 Optionally set OPENAI_BASE_URL if using a custom endpoint")
