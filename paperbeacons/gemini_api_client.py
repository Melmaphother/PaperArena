"""
Gemini API Client for text and PDF processing using the OpenAI-compatible interface.

This module provides a function to interact with the gemini-2.5-pro-all model
through the API endpoint, supporting both text and PDF inputs.
"""

import base64
import json
import os
import requests
from typing import Optional, Union
from pathlib import Path
from dotenv import load_dotenv


def call_gemini_api(
    text_input: str,
    pdf_path: Optional[Union[str, Path]] = None,
    model: str = "gemini-2.5-pro-all",
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
) -> str:
    """
    Call the Gemini API with text and optional PDF input.

    This function sends a request to the gemini-2.5-pro-all model using the OpenAI-compatible
    interface. It supports both text-only queries and queries with PDF documents.

    Args:
        text_input (str): The text prompt/question to send to the model.
        pdf_path (Optional[Union[str, Path]]): Path to the PDF file to include in the request.
        model (str): Model name to use. Defaults to "gemini-2.5-pro-all".
        temperature (float): Sampling temperature (0.0 to 1.0). Defaults to 0.7.
        max_tokens (Optional[int]): Maximum number of tokens to generate.

    Returns:
        str: The generated response from the model.

    Raises:
        ValueError: If required parameters are missing or invalid.
        FileNotFoundError: If the specified PDF file doesn't exist.
        requests.RequestException: If the API request fails.

    Example:
        >>> # Text-only query
        >>> response = call_gemini_api(
        ...     text_input="What is machine learning?"
        ... )
        >>> print(response)

        >>> # Query with PDF
        >>> response = call_gemini_api(
        ...     text_input="Summarize this research paper",
        ...     pdf_path="paper.pdf"
        ... )
        >>> print(response)
    """
    # Load environment variables from .env file
    load_dotenv()

    # Get API configuration from environment variables
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")

    if not base_url:
        raise ValueError("❌ OPENAI_BASE_URL is required in .env file")

    if not api_key:
        raise ValueError("❌ OPENAI_API_KEY is required in .env file")

    if not text_input.strip():
        raise ValueError("❌ Text input cannot be empty")

    # Prepare the message content using the correct API format
    message_content = []

    # Add PDF content first if provided (using "file" type)
    if pdf_path:
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"❌ PDF file not found: {pdf_path}")

        try:
            print(f"📄 Reading PDF file: {pdf_path}")
            with open(pdf_path, "rb") as pdf_file:
                pdf_data = pdf_file.read()

            # Check file size and warn if too large
            file_size_mb = len(pdf_data) / (1024 * 1024)
            print(f"📏 PDF file size: {file_size_mb:.2f} MB")

            if file_size_mb > 10:
                print("⚠️ Warning: Large PDF file may cause timeout or API limits")
                print("   Consider using a smaller PDF for testing")

            pdf_base64 = base64.b64encode(pdf_data).decode("utf-8")
            base64_size_mb = len(pdf_base64) / (1024 * 1024)
            print(f"📦 Base64 encoded size: {base64_size_mb:.2f} MB")

            # Add PDF using the correct API format
            message_content.append(
                {
                    "type": "file",
                    "file": {
                        "filename": pdf_path.name,
                        "file_data": f"data:application/pdf;base64,{pdf_base64}",
                    },
                }
            )
            print("✅ PDF file successfully encoded")

        except Exception as e:
            raise ValueError(f"❌ Error reading PDF file: {str(e)}")

    # Add text content
    message_content.append({"type": "text", "text": text_input})

    print(f"📋 Message content structure: {[item['type'] for item in message_content]}")

    # Prepare the request payload
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": message_content}],
        "temperature": temperature,
    }

    if max_tokens:
        payload["max_tokens"] = max_tokens

    # Prepare headers
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    # Make the API request
    try:
        # Calculate estimated request size for PDF uploads
        if pdf_path:
            estimated_size_mb = len(json.dumps(payload).encode("utf-8")) / (1024 * 1024)
            print(f"📦 Estimated request size: {estimated_size_mb:.2f} MB")
            timeout = max(
                120, int(estimated_size_mb * 30)
            )  # Dynamic timeout based on size
            print(f"⏱️ Using timeout: {timeout} seconds")
        else:
            timeout = 60

        print(f"🚀 Sending request to {model}...")
        response = requests.post(
            f"{base_url}/chat/completions",
            headers=headers,
            json=payload,
            timeout=timeout,
        )

        # Check for HTTP errors
        response.raise_for_status()

        # Parse the response
        response_data = response.json()

        # Extract the generated text
        if "choices" in response_data and len(response_data["choices"]) > 0:
            generated_text = response_data["choices"][0]["message"]["content"]
            print("✅ Response received successfully")
            return generated_text.strip()
        else:
            raise ValueError("❌ Unexpected response format from API")

    except requests.exceptions.RequestException as e:
        raise requests.RequestException(f"❌ API request failed: {str(e)}")
    except json.JSONDecodeError as e:
        raise ValueError(f"❌ Failed to parse API response: {str(e)}")
    except KeyError as e:
        raise ValueError(f"❌ Missing expected field in API response: {str(e)}")


if __name__ == "__main__":
    """
    Example usage of the Gemini API client.
    """
    print("🚀 Gemini API Client Example")
    print("=" * 40)

    # Example 1: Text-only query
    # print("\n📝 Example 1: Simple text query")
    # try:
    #     response = call_gemini_api(
    #         text_input="What is the difference between machine learning and deep learning?",
    #         model="gemini-2.5-flash",
    #         temperature=0.7,
    #     )
    #     print(f"✅ Response: {response}")
    # except Exception as e:
    #     print(f"❌ Error: {e}")

    # Example 2: Query with PDF (if exists)
    print("\n📄 Example 2: Query with PDF")
    pdf_files = ["test_small.pdf"]

    if pdf_files:
        pdf_file = pdf_files[0]  # Use the first PDF found
        print(f"Using PDF: {pdf_file}")
        try:
            response = call_gemini_api(
                text_input="Please provide a brief summary of this paper.",
                pdf_path=pdf_file,
                temperature=0.8,
            )
            print(f"✅ Response: {response}")
        except Exception as e:
            print(f"❌ Error: {e}")
    else:
        print("📄 No PDF files found in current directory")
