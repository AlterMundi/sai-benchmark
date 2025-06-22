"""
Ollama Engine Implementation

Interface for Ollama API supporting local inference of vision-language models.
Handles HTTP communication, image encoding, and response parsing.
"""

import requests
import json
import base64
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import time

from .base_engine import BaseEngine, EngineResponse


class OllamaEngine(BaseEngine):
    """Ollama inference engine for local models"""
    
    def __init__(self, model_id: str, base_url: str = "http://localhost:11434", **kwargs):
        self.base_url = base_url.rstrip('/')
        self.timeout = kwargs.get('timeout', 120)
        self.max_retries = kwargs.get('max_retries', 3)
        super().__init__(model_id, **kwargs)
    
    def _initialize(self):
        """Initialize Ollama client and verify model availability"""
        try:
            # Check if Ollama is running
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            response.raise_for_status()
            
            # Check if model is available
            models = response.json().get('models', [])
            model_names = [model['name'] for model in models]
            
            if self.model_id not in model_names:
                raise ValueError(f"Model '{self.model_id}' not found in Ollama. Available models: {model_names}")
                
        except requests.exceptions.ConnectionError:
            raise ConnectionError(f"Cannot connect to Ollama at {self.base_url}. Make sure Ollama is running.")
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Failed to initialize Ollama engine: {e}")
    
    def _encode_image(self, image_path: Union[str, Path]) -> str:
        """Encode image to base64 string"""
        try:
            with open(image_path, 'rb') as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            raise ValueError(f"Failed to encode image {image_path}: {e}")
    
    def generate(self, 
                prompt: str, 
                images: List[Union[str, Path]] = None,
                **kwargs) -> EngineResponse:
        """Generate response using Ollama API"""
        
        if not self.validate_inputs(prompt, images):
            return EngineResponse(
                content="",
                latency_ms=0,
                error="Invalid inputs provided"
            )
        
        # Prepare request payload
        payload = {
            "model": self.model_id,
            "prompt": prompt,
            "stream": False,
            "options": {}
        }
        
        # Add generation parameters
        if 'temperature' in kwargs:
            payload['options']['temperature'] = kwargs['temperature']
        if 'max_tokens' in kwargs:
            payload['options']['num_predict'] = kwargs['max_tokens']
        if 'top_p' in kwargs:
            payload['options']['top_p'] = kwargs['top_p']
        
        # Add images if provided
        if images:
            payload['images'] = []
            for img in images:
                if isinstance(img, str) and img.startswith('data:'):
                    # Already base64 encoded
                    payload['images'].append(img.split(',')[1])
                else:
                    # Encode image file
                    payload['images'].append(self._encode_image(img))
        
        # Make request with retries
        start_time = time.time()
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                    timeout=self.timeout
                )
                response.raise_for_status()
                
                result = response.json()
                latency_ms = (time.time() - start_time) * 1000
                
                # Extract response content
                content = result.get('response', '')
                
                # Extract metadata
                metadata = {
                    'model': result.get('model', self.model_id),
                    'total_duration': result.get('total_duration', 0),
                    'load_duration': result.get('load_duration', 0),
                    'prompt_eval_count': result.get('prompt_eval_count', 0),
                    'prompt_eval_duration': result.get('prompt_eval_duration', 0),
                    'eval_count': result.get('eval_count', 0),
                    'eval_duration': result.get('eval_duration', 0),
                    'attempt': attempt + 1
                }
                
                return EngineResponse(
                    content=self.postprocess_response(content),
                    latency_ms=latency_ms,
                    tokens_used=metadata.get('eval_count'),
                    cost=0.0,  # Local inference is free
                    metadata=metadata
                )
                
            except requests.exceptions.Timeout:
                last_error = f"Request timeout after {self.timeout}s"
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue
                    
            except requests.exceptions.RequestException as e:
                last_error = f"Request failed: {e}"
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                    
            except Exception as e:
                last_error = f"Unexpected error: {e}"
                break
        
        # Return error response if all retries failed
        return EngineResponse(
            content="",
            latency_ms=(time.time() - start_time) * 1000,
            error=last_error
        )
    
    def health_check(self) -> bool:
        """Check if Ollama service is healthy"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information from Ollama"""
        try:
            response = requests.post(
                f"{self.base_url}/api/show",
                json={"name": self.model_id},
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def list_available_models(self) -> List[str]:
        """List all available models in Ollama"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            response.raise_for_status()
            models = response.json().get('models', [])
            return [model['name'] for model in models]
        except Exception:
            return []
    
    def pull_model(self, model_name: str) -> bool:
        """Pull a model from Ollama registry"""
        try:
            response = requests.post(
                f"{self.base_url}/api/pull",
                json={"name": model_name},
                timeout=300  # 5 minutes for model download
            )
            return response.status_code == 200
        except Exception:
            return False
    
    def supports_batch(self) -> bool:
        """Ollama doesn't support batch processing"""
        return False
    
    def postprocess_response(self, raw_response: str) -> str:
        """Clean up Ollama response"""
        # Remove common artifacts from Ollama responses
        response = raw_response.strip()
        
        # Remove markdown code blocks if present
        if response.startswith('```'):
            lines = response.split('\n')
            if len(lines) > 2 and lines[-1].strip() == '```':
                response = '\n'.join(lines[1:-1])
        
        return response