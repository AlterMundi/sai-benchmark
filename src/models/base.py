"""
Base classes and interfaces for vision models in SAI-Benchmark.
Defines the common interface that all vision models must implement.
"""

from abc import ABC, abstractmethod
from typing import Dict, Union, Optional, List
from pathlib import Path
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ModelCapabilities:
    """Defines what capabilities a vision model supports."""
    supports_bbox: bool = False
    supports_multiimage: bool = False
    max_resolution: Optional[str] = None
    context_window: Optional[int] = None
    languages: List[str] = None
    

@dataclass
class ModelConfig:
    """Configuration for a specific vision model."""
    name: str
    display_name: str
    model_id: str  # Ollama model identifier
    prompt_template: str
    temperature: float = 0.1
    max_tokens: int = 2048
    capabilities: ModelCapabilities = None
    
    def __post_init__(self):
        if self.capabilities is None:
            self.capabilities = ModelCapabilities()


class VisionModel(ABC):
    """Abstract base class for all vision models."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
    @abstractmethod
    def infer(self, image_path: Union[str, Path], 
              prompt_override: Optional[str] = None,
              temperature: Optional[float] = None) -> Dict:
        """
        Run inference on a single image.
        
        Args:
            image_path: Path to the image file
            prompt_override: Optional custom prompt
            temperature: Optional temperature override
            
        Returns:
            Dict with format: {"has_smoke": bool, "bbox": [x, y, w, h], ...}
        """
        pass
    
    @abstractmethod
    def check_availability(self) -> bool:
        """Check if this model is available and ready to use."""
        pass
    
    def get_capabilities(self) -> ModelCapabilities:
        """Get the capabilities of this model."""
        return self.config.capabilities
    
    def get_display_name(self) -> str:
        """Get human-readable model name."""
        return self.config.display_name
        
    def validate_response(self, response: Dict) -> Dict:
        """
        Validate and normalize model response to standard format.
        
        Args:
            response: Raw model response
            
        Returns:
            Validated response in standard format
        """
        # Ensure required fields exist
        validated = {
            "has_smoke": response.get("has_smoke", False),
            "bbox": response.get("bbox", [0.0, 0.0, 0.0, 0.0])
        }
        
        # Validate bbox format
        bbox = validated["bbox"]
        if not isinstance(bbox, list) or len(bbox) != 4:
            validated["bbox"] = [0.0, 0.0, 0.0, 0.0]
            validated["bbox_error"] = "Invalid bbox format"
        else:
            # Ensure all bbox values are floats
            try:
                validated["bbox"] = [float(x) for x in bbox]
            except (ValueError, TypeError):
                validated["bbox"] = [0.0, 0.0, 0.0, 0.0]
                validated["bbox_error"] = "Non-numeric bbox values"
        
        # Copy over any additional fields (errors, raw output, etc.)
        for key, value in response.items():
            if key not in validated:
                validated[key] = value
                
        return validated


class ModelFactory:
    """Factory for creating vision model instances."""
    
    _registry = {}
    
    @classmethod
    def register(cls, model_name: str, model_class: type):
        """Register a new model class."""
        cls._registry[model_name] = model_class
        logger.info(f"Registered model: {model_name}")
    
    @classmethod
    def create(cls, config: ModelConfig) -> VisionModel:
        """Create a model instance from configuration."""
        model_class = cls._registry.get(config.name)
        if not model_class:
            raise ValueError(f"Unknown model: {config.name}. Available: {list(cls._registry.keys())}")
        
        return model_class(config)
    
    @classmethod
    def list_available(cls) -> List[str]:
        """List all registered model names."""
        return list(cls._registry.keys())


def register_model(name: str):
    """Decorator to register a model class."""
    def decorator(cls):
        ModelFactory.register(name, cls)
        return cls
    return decorator