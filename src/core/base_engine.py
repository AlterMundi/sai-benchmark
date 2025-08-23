"""
Base Engine Interface

Abstract base class defining unified interface for all inference engines.
Provides consistent API for model execution, response handling, and metrics collection.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
import time
from pathlib import Path


@dataclass
class EngineResponse:
    """Standardized response from any engine"""
    content: str
    latency_ms: float
    tokens_used: Optional[int] = None
    cost: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    
    @property
    def success(self) -> bool:
        """Check if response was successful"""
        return self.error is None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert response to dictionary"""
        return {
            "content": self.content,
            "latency_ms": self.latency_ms,
            "tokens_used": self.tokens_used,
            "cost": self.cost,
            "metadata": self.metadata or {},
            "error": self.error,
            "success": self.success
        }


class BaseEngine(ABC):
    """Abstract base class for all inference engines"""
    
    def __init__(self, model_id: str, **kwargs):
        self.model_id = model_id
        self.config = kwargs
        self._initialize()
    
    @abstractmethod
    def _initialize(self):
        """Initialize the engine (e.g., load model, setup API client)"""
        pass
    
    @abstractmethod
    def generate(self, 
                prompt: str, 
                images: List[Union[str, Path]] = None,
                **kwargs) -> EngineResponse:
        """
        Generate response from model
        
        Args:
            prompt: Text prompt to send to model
            images: List of image paths or base64 strings
            **kwargs: Additional generation parameters
            
        Returns:
            EngineResponse with generated content and metadata
        """
        pass
    
    def generate_with_timing(self, 
                           prompt: str,
                           images: List[Union[str, Path]] = None,
                           **kwargs) -> EngineResponse:
        """Generate response with automatic timing"""
        start_time = time.time()
        try:
            response = self.generate(prompt, images, **kwargs)
            if response.latency_ms is None:
                response.latency_ms = (time.time() - start_time) * 1000
            return response
        except Exception as e:
            return EngineResponse(
                content="",
                latency_ms=(time.time() - start_time) * 1000,
                error=str(e)
            )
    
    @abstractmethod
    def health_check(self) -> bool:
        """Check if engine is healthy and ready"""
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        pass
    
    def get_engine_name(self) -> str:
        """Get engine name (e.g., 'ollama', 'huggingface', 'openai')"""
        return self.__class__.__name__.lower().replace('engine', '')
    
    def supports_images(self) -> bool:
        """Check if engine supports image inputs"""
        return True  # Most vision-language models support images
    
    def supports_batch(self) -> bool:
        """Check if engine supports batch processing"""
        return False  # Override in subclasses that support batching
    
    def estimate_cost(self, prompt: str, images: List = None) -> float:
        """Estimate cost for a request (override in paid engines)"""
        return 0.0
    
    def validate_inputs(self, prompt: str, images: List = None) -> bool:
        """Validate inputs before sending to model"""
        if not prompt or not prompt.strip():
            return False
        
        if images:
            # Check if image files exist (for file paths)
            for img in images:
                if isinstance(img, (str, Path)) and not str(img).startswith('data:'):
                    if not Path(img).exists():
                        return False
        
        return True
    
    def preprocess_images(self, images: List[Union[str, Path]]) -> List[str]:
        """Preprocess images for the specific engine (override in subclasses)"""
        if not images:
            return []
        
        processed = []
        for img in images:
            if isinstance(img, Path):
                processed.append(str(img))
            else:
                processed.append(img)
        
        return processed
    
    def postprocess_response(self, raw_response: str) -> str:
        """Postprocess raw model response (override in subclasses)"""
        return raw_response.strip()
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model_id='{self.model_id}')"