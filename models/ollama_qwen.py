"""
Ollama backend wrapper for Qwen 2.5-VL model.
Provides HTTP API interface for local inference.
"""

import base64
import requests
import json
import pathlib
from typing import Dict, Union, Optional
import logging

logger = logging.getLogger(__name__)

# Configuration
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5vl:7b"
TIMEOUT = 180  # seconds

# System prompt for early-fire detection
SYSTEM_PROMPT = (
    "You are an early-fire detection agent. "
    "Answer ONLY with a valid JSON matching this schema: "
    '{"has_smoke": bool, "bbox": [x_center, y_center, width, height]}. '
    'The bbox values should be normalized (0-1). '
    'If no smoke is detected, use bbox: [0, 0, 0, 0]. '
    'DO NOT include any other text in your response.'
)


def _encode_image(image_path: Union[str, pathlib.Path]) -> str:
    """Encode image to base64 string."""
    path = pathlib.Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    image_data = path.read_bytes()
    return base64.b64encode(image_data).decode('utf-8')


def infer(image_path: Union[str, pathlib.Path], 
          prompt_override: Optional[str] = None,
          temperature: float = 0.1) -> Dict:
    """
    Run inference on a single image for smoke detection.
    
    Args:
        image_path: Path to the image file
        prompt_override: Optional custom prompt (defaults to SYSTEM_PROMPT)
        temperature: Model temperature (default 0.1 for consistent outputs)
    
    Returns:
        Dict with format: {"has_smoke": bool, "bbox": [x, y, w, h]}
    
    Raises:
        Exception: If inference fails or response cannot be parsed
    """
    try:
        # Prepare the prompt
        prompt = prompt_override or SYSTEM_PROMPT
        full_prompt = f"<system>{prompt}</system>\n<image>Analyze this image for smoke or fire.</image>"
        
        # Encode image
        image_b64 = _encode_image(image_path)
        
        # Prepare request payload
        payload = {
            "model": MODEL_NAME,
            "prompt": full_prompt,
            "images": [image_b64],
            "stream": False,
            "temperature": temperature,
            "format": "json"  # Request JSON format
        }
        
        # Make request
        logger.debug(f"Sending request to Ollama for image: {image_path}")
        response = requests.post(OLLAMA_URL, json=payload, timeout=TIMEOUT)
        response.raise_for_status()
        
        # Extract response
        result = response.json()
        model_output = result.get("response", "")
        
        # Parse JSON response
        try:
            parsed = json.loads(model_output)
            
            # Validate schema
            if not isinstance(parsed.get("has_smoke"), bool):
                raise ValueError("Missing or invalid 'has_smoke' field")
            
            bbox = parsed.get("bbox", [0, 0, 0, 0])
            if not isinstance(bbox, list) or len(bbox) != 4:
                raise ValueError("Invalid bbox format")
            
            # Ensure bbox values are floats
            bbox = [float(x) for x in bbox]
            
            return {
                "has_smoke": parsed["has_smoke"],
                "bbox": bbox
            }
            
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Failed to parse model output: {model_output}")
            # Return safe default on parse error
            return {
                "has_smoke": False,
                "bbox": [0, 0, 0, 0],
                "error": f"Parse error: {str(e)}",
                "raw_output": model_output
            }
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Request failed: {e}")
        raise Exception(f"Ollama request failed: {e}")
    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise


def check_connection() -> bool:
    """Check if Ollama server is accessible."""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        return response.status_code == 200
    except:
        return False


def get_available_models() -> list:
    """Get list of available models from Ollama."""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            data = response.json()
            return [model["name"] for model in data.get("models", [])]
    except:
        pass
    return []


if __name__ == "__main__":
    # Test the module
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python ollama_qwen.py <image_path>")
        sys.exit(1)
    
    if not check_connection():
        print("Error: Cannot connect to Ollama. Make sure it's running.")
        sys.exit(1)
    
    models = get_available_models()
    if MODEL_NAME not in models:
        print(f"Error: Model {MODEL_NAME} not found. Available models: {models}")
        print(f"Run: ollama pull {MODEL_NAME}")
        sys.exit(1)
    
    try:
        result = infer(sys.argv[1])
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)