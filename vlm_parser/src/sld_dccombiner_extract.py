"""
Extracts DC Combiner metadata from SLD images using Anthropic Claude and a custom prompt.
Usage:
    python src/sld_dccombiner_extract.py
Requires: ANTHROPIC_API_KEY in environment or .env file.
"""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

import sys

# Add project root to sys.path so "schema" can be imported
sys.path.append(str(Path(__file__).parent.parent))

from pydantic import ValidationError
from anthropic import Anthropic
import instructor
import base64

# Import only the relevant model
from schema.SldDCCombiner import DCCombinerBox
from schema.SldDCCombiner import DCCombinerBoxExtractResponse


# Anthropic API setup (modular for reuse)
def get_anthropic_client() -> Anthropic:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError(
            "ANTHROPIC_API_KEY environment variable not set. "
            "Add it to your .env file or system environment."
        )
    return instructor.from_anthropic(Anthropic(api_key=api_key))


# Prompt loader (modular)
def load_prompt(prompt_path: Optional[Path] = None) -> str:
    default_path = Path(__file__).parent.parent / "prompts" / "sld_dc_combiner_prompt"
    path = prompt_path or default_path
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


# Image encoder
def _encode_image(image_path: str) -> tuple[str, str]:
    """Encode an image file to base64 and determine its media type."""
    media_type = "image/png"
    ext = Path(image_path).suffix.lower()
    if ext in (".jpg", ".jpeg"):
        media_type = "image/jpeg"
    elif ext == ".gif":
        media_type = "image/gif"
    elif ext == ".webp":
        media_type = "image/webp"

    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
    return encoded_string, media_type


def _call_llm(image_path: str, prompt_text: str, response_model):
    client = get_anthropic_client()
    encoded_string, media_type = _encode_image(image_path)
    return client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": encoded_string,
                        },
                    },
                ],
            }
        ],
        response_model=response_model,
    )


# Main parsing function
def parse_dc_combiner_metadata(
    image_path: str, prompt_path: Optional[Path] = None
) -> DCCombinerBoxExtractResponse:
    prompt = load_prompt(prompt_path)
    try:
        response = _call_llm(image_path, prompt, DCCombinerBoxExtractResponse)
        return DCCombinerBoxExtractResponse.model_validate(response)
    except ValidationError as e:
        print("Validation error:", e)
        raise


# CLI entrypoint for standalone use
if __name__ == "__main__":
    # Discover the first image in the input folder
    input_folder = Path(__file__).parent.parent / "data" / "input"
    image_extensions = (".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp")
    image_files = [
        f for f in input_folder.iterdir() if f.suffix.lower() in image_extensions
    ]

    if not image_files:
        print(f"No image files found in {input_folder}")
    else:
        image_path = image_files[0]
        print(f"Parsing SLD image: {image_path}")
        result = parse_dc_combiner_metadata(str(image_path))
        json_output = result.model_dump_json(indent=2)
        print(json_output)
