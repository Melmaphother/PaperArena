"""
LLM Chat utilities for multimodal and text-only conversations
"""

from typing import List, Optional, Union, Dict, Any
from pathlib import Path
import base64
from io import BytesIO
from PIL import Image
import os
from smolagents import OpenAIServerModel
from loguru import logger


def encode_image_to_base64(image_path: Union[str, Path]) -> str:
    """
    Encode image file to base64 string for multimodal LLM input

    Args:
        image_path: Path to the image file

    Returns:
        Base64 encoded image string

    Raises:
        FileNotFoundError: If image file doesn't exist
        Exception: If image encoding fails
    """
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    try:
        with Image.open(image_path) as img:
            # Convert to RGB if necessary (for JPEG compatibility)
            if img.mode != "RGB":
                img = img.convert("RGB")

            # Save to bytes buffer
            buffer = BytesIO()
            img.save(buffer, format="JPEG", quality=85)
            buffer.seek(0)

            # Encode to base64
            image_bytes = buffer.getvalue()
            base64_string = base64.b64encode(image_bytes).decode("utf-8")

            return f"data:image/jpeg;base64,{base64_string}"

    except Exception as e:
        logger.error(f"❌ Failed to encode image {image_path}: {e}")
        raise


def chat_with_llm(
    text_prompt: str,
    model_name: str = "gpt-4o-2024-11-20",
    image_path: Optional[Union[str, Path]] = None,
    system_prompt: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 2000,
) -> str:
    """
    Chat with multimodal LLM that can process both text and images

    Args:
        text_prompt: Text prompt/question for the model
        image_path: Optional path to image file
        system_prompt: Optional system prompt to set context
        temperature: Sampling temperature (0.0 to 1.0)
        max_tokens: Maximum tokens in response

    Returns:
        Model's response text

    Raises:
        Exception: If LLM call fails
    """
    try:
        logger.info(f"🤖 Calling multimodal LLM: {model_name}")

        # Initialize the model
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL")
        model = OpenAIServerModel(
            model_id=model_name, 
            api_base=base_url,
            api_key=api_key,
        )

        # Prepare messages
        messages = []

        # Add system prompt if provided
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # Prepare user message content
        user_content = []

        # Add text content
        user_content.append({"type": "text", "text": text_prompt})

        # Add image content if provided
        if image_path:
            base64_image = encode_image_to_base64(image_path)
            user_content.append(
                {"type": "image_url", "image_url": {"url": base64_image}}
            )
            logger.info(f"📷 Added image: {image_path}")

        # Add user message
        messages.append({"role": "user", "content": user_content})

        # Make the API call
        response = model(
            messages=messages, temperature=temperature, max_tokens=max_tokens
        ).content

        logger.info("✅ LLM call completed successfully")
        return response

    except Exception as e:
        error_msg = f"❌ LLM call failed: {str(e)}"
        logger.error(error_msg)
        raise Exception(error_msg)


if __name__ == "__main__":
    # Test the multimodal LLM function

    try:
        response = chat_with_llm(
            text_prompt="What is machine learning?",
            model_name="gpt-4o-2024-11-20",
            system_prompt="You are a helpful AI assistant.",
        )
        print("🔍 Multimodal LLM Response (text only):")
        print("=" * 50)
        print(response)
    except Exception as e:
        print(f"❌ Multimodal LLM test failed: {e}")
