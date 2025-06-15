"""
Mistral Small 3.1 24B Vision model implementation using Ollama backend.
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
TIMEOUT = 300  # Increased timeout for larger model


def _encode_image(image_path: Union[str, pathlib.Path]) -> str:
    """Encode image to base64 string."""
    path = pathlib.Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    image_data = path.read_bytes()
    return base64.b64encode(image_data).decode('utf-8')


@register_model("mistral-small-3.1")
class MistralVisionModel(VisionModel):
    """Mistral Small 3.1 24B Vision model implementation using Ollama backend."""
    
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
            
            # Mistral uses a specific format with role-based messages
            full_prompt = f"[INST] {prompt}\\n\\n[IMAGE]\\n\\n[/INST]"
            
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
                "options": {
                    "num_predict": self.config.max_tokens,
                    "stop": ["[INST]", "[/INST]", "</s>"]
                }
            }
            
            # Make request
            self.logger.debug(f"Sending request to Ollama for image: {image_path}")
            response = requests.post(OLLAMA_URL, json=payload, timeout=TIMEOUT)
            response.raise_for_status()
            
            # Extract response
            result = response.json()
            model_output = result.get("response", "")
            
            # Parse JSON response (Mistral has good JSON compliance)
            try:
                # Try to extract JSON from the response
                start_idx = model_output.find('{')
                end_idx = model_output.rfind('}') + 1
                
                if start_idx >= 0 and end_idx > start_idx:
                    json_str = model_output[start_idx:end_idx]
                    parsed = json.loads(json_str)
                else:
                    # Fallback: try to parse the entire response
                    parsed = json.loads(model_output.strip())
                
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
                
                # Fallback: advanced text analysis for Mistral
                has_smoke = self._analyze_text_response(model_output)
                
                return self.validate_response({
                    "has_smoke": has_smoke,
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
    
    def _analyze_text_response(self, text: str) -> bool:
        """
        Analyze text response to determine if smoke/fire is detected.
        Mistral models tend to be quite precise and structured.
        """
        text_lower = text.lower()
        
        # Strong positive indicators
        strong_positive = [
            '"has_smoke": true', "'has_smoke': true", "has_smoke: true",
            "smoke detected", "fire detected", "smoke present", "fire present"
        ]
        
        # Strong negative indicators  
        strong_negative = [
            '"has_smoke": false', "'has_smoke': false", "has_smoke: false",
            "no smoke", "no fire", "smoke not detected", "fire not detected"
        ]
        
        # Check strong indicators first
        for indicator in strong_positive:
            if indicator in text_lower:
                return True
                
        for indicator in strong_negative:
            if indicator in text_lower:
                return False
        
        # Weaker positive indicators
        weak_positive = [
            'smoke', 'fire', 'burning', 'flames', 'wildfire', 'forest fire',
            'plume', 'smoldering', 'combustion', 'blaze', 'visible smoke'
        ]
        
        # Weaker negative indicators
        weak_negative = [
            'clear sky', 'normal landscape', 'no signs', 'appears normal',
            'peaceful', 'calm', 'clear air', 'clean environment'
        ]
        
        # Count weak indicators
        positive_count = sum(1 for indicator in weak_positive if indicator in text_lower)
        negative_count = sum(1 for indicator in weak_negative if indicator in text_lower)
        
        # Decision logic with conservative bias
        if positive_count > negative_count and positive_count >= 2:
            return True
        elif negative_count > positive_count:
            return False
        else:
            # Conservative default - require explicit confirmation
            return False
    
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