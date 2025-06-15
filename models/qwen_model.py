"""
Qwen 2.5-VL model implementation using Ollama backend.
Provides HTTP API interface for local inference.
"""

import base64
import requests
import json
import pathlib
from typing import Dict, Union, Optional
import logging

from .base import VisionModel, ModelConfig, register_model
from .registry import get_prompt_template

logger = logging.getLogger(__name__)

# Configuration
OLLAMA_URL = "http://localhost:11434/api/generate"
TIMEOUT = 180  # seconds


def _encode_image(image_path: Union[str, pathlib.Path]) -> str:
    """Encode image to base64 string."""
    path = pathlib.Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    image_data = path.read_bytes()
    return base64.b64encode(image_data).decode('utf-8')


@register_model("qwen2.5vl-7b")
@register_model("qwen2.5vl-3b")
class QwenVisionModel(VisionModel):
    """Qwen 2.5-VL model implementation using Ollama backend."""
    
    def infer(self, image_path: Union[str, pathlib.Path], 
              prompt_override: Optional[str] = None,
              temperature: Optional[float] = None) -> Dict:
        """
        Run inference on a single image for smoke detection.
        
        Args:
            image_path: Path to the image file
            prompt_override: Optional custom prompt
            temperature: Optional temperature override
        
        Returns:
            Dict with format: {"has_smoke": bool, "bbox": [x, y, w, h]}
        """
        try:
            # Prepare the prompt
            if prompt_override:
                prompt = prompt_override
            else:
                template_name = self.config.prompt_template
                prompt = get_prompt_template(template_name)
            
            full_prompt = f"<system>{prompt}</system>\\n<image>Analyze this image for smoke or fire.</image>"
            
            # Use config temperature or override
            temp = temperature if temperature is not None else self.config.temperature
            
            # Encode image
            image_b64 = _encode_image(image_path)
            
            # Prepare request payload
            payload = {
                "model": self.config.model_id,
                "prompt": full_prompt,
                "images": [image_b64],
                "stream": False,
                "temperature": temp,
                "format": "json"  # Request JSON format
            }
            
            # Make request
            self.logger.debug(f"Sending request to Ollama for image: {image_path}")
            response = requests.post(OLLAMA_URL, json=payload, timeout=TIMEOUT)
            response.raise_for_status()
            
            # Extract response
            result = response.json()
            model_output = result.get("response", "")
            
            # Parse JSON response
            try:
                parsed = json.loads(model_output)
                raw_response = {
                    "has_smoke": parsed.get("has_smoke", False),
                    "bbox": parsed.get("bbox", [0, 0, 0, 0])
                }
                
                # Add any additional fields from response
                for key, value in parsed.items():
                    if key not in raw_response:
                        raw_response[key] = value
                
                return self.validate_response(raw_response)
                
            except (json.JSONDecodeError, ValueError) as e:
                self.logger.error(f"Failed to parse model output: {model_output}")
                # Return safe default on parse error
                return self.validate_response({
                    "has_smoke": False,
                    "bbox": [0, 0, 0, 0],
                    "error": f"Parse error: {str(e)}",
                    "raw_output": model_output
                })
                
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Request failed: {e}")
            raise Exception(f"Ollama request failed: {e}")
        except Exception as e:
            self.logger.error(f"Inference error: {e}")
            raise
    
    def check_availability(self) -> bool:
        """Check if this model is available in Ollama."""
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code == 200:
                data = response.json()
                available_models = [model["name"] for model in data.get("models", [])]
                return self.config.model_id in available_models
        except:
            pass
        return False


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


# Legacy function for backward compatibility  
def infer(image_path: Union[str, pathlib.Path], 
          prompt_override: Optional[str] = None,
          temperature: float = 0.1) -> Dict:
    """Legacy function - creates a temporary Qwen model instance."""
    from .registry import get_model_config
    config = get_model_config("qwen2.5vl-7b")
    model = QwenVisionModel(config)
    return model.infer(image_path, prompt_override, temperature)