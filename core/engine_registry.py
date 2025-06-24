"""
Engine Registry System

Manages engine configurations and provides unified interface for model execution.
Supports automatic engine selection based on capabilities and requirements.
"""

from typing import Dict, List, Any, Optional, Type, Union
from dataclasses import dataclass
from enum import Enum
import importlib
from pathlib import Path

from engines.base_engine import BaseEngine, EngineResponse
from core.model_registry import ModelRegistry, EngineType, ModelConfig


@dataclass
class EngineConfig:
    """Configuration for an inference engine"""
    engine_type: EngineType
    engine_class: Type[BaseEngine]
    default_config: Dict[str, Any]
    health_check_url: Optional[str] = None
    requirements: List[str] = None
    
    def __post_init__(self):
        if self.requirements is None:
            self.requirements = []


class EngineRegistry:
    """Registry for managing inference engines"""
    
    def __init__(self, model_registry: Optional[ModelRegistry] = None):
        self.model_registry = model_registry or ModelRegistry()
        self.engines: Dict[EngineType, EngineConfig] = {}
        self.active_engines: Dict[str, BaseEngine] = {}  # model_id -> engine instance
        self._register_builtin_engines()
    
    def _register_builtin_engines(self):
        """Register built-in engine configurations"""
        
        # Ollama Engine (requires external Ollama installation)
        from engines.ollama_engine import OllamaEngine
        self.register_engine(EngineConfig(
            engine_type=EngineType.OLLAMA,
            engine_class=OllamaEngine,
            default_config={
                "base_url": "http://localhost:11434",
                "timeout": 120,
                "max_retries": 3
            },
            health_check_url="http://localhost:11434/api/tags",
            requirements=["requests"]
        ))
        
        # Hugging Face Engine
        from engines.hf_engine import HuggingFaceEngine
        self.register_engine(EngineConfig(
            engine_type=EngineType.HUGGINGFACE,
            engine_class=HuggingFaceEngine,
            default_config={
                "device": "auto",
                "torch_dtype": "float16",
                "trust_remote_code": True
            },
            requirements=["transformers", "accelerate", "qwen-vl-utils", "torch"]
        ))
    
    def register_engine(self, config: EngineConfig):
        """Register an engine configuration"""
        self.engines[config.engine_type] = config
    
    def get_engine(self, model_id: str, **kwargs) -> BaseEngine:
        """Get or create engine instance for a model"""
        
        # Check if engine is already active
        if model_id in self.active_engines:
            return self.active_engines[model_id]
        
        # Get model configuration
        model_config = self.model_registry.get_model(model_id)
        engine_type = model_config.engine
        
        # Get engine configuration
        if engine_type not in self.engines:
            raise ValueError(f"Engine type '{engine_type}' not registered")
        
        engine_config = self.engines[engine_type]
        
        # Merge default config with user provided config
        final_config = {**engine_config.default_config, **kwargs}
        
        # Create engine instance
        try:
            engine = engine_config.engine_class(model_id, **final_config)
            self.active_engines[model_id] = engine
            return engine
        except Exception as e:
            raise RuntimeError(f"Failed to create engine for model '{model_id}': {e}")
    
    def execute_prompt(self, 
                      model_id: str, 
                      prompt: str, 
                      images: List[Union[str, Path]] = None,
                      **kwargs) -> EngineResponse:
        """Execute prompt on specified model"""
        engine = self.get_engine(model_id, **kwargs)
        return engine.generate_with_timing(prompt, images, **kwargs)
    
    def execute_batch(self,
                     model_id: str,
                     prompts: List[str],
                     image_lists: List[List[Union[str, Path]]] = None,
                     **kwargs) -> List[EngineResponse]:
        """Execute batch of prompts (if engine supports batching)"""
        engine = self.get_engine(model_id, **kwargs)
        
        if not engine.supports_batch():
            # Execute sequentially if batching not supported
            results = []
            image_lists = image_lists or [None] * len(prompts)
            
            for prompt, images in zip(prompts, image_lists):
                result = engine.generate_with_timing(prompt, images, **kwargs)
                results.append(result)
            
            return results
        else:
            # TODO: Implement batch processing for engines that support it
            raise NotImplementedError("Batch processing not yet implemented")
    
    def health_check(self, engine_type: EngineType) -> bool:
        """Check if an engine type is healthy"""
        if engine_type not in self.engines:
            return False
        
        engine_config = self.engines[engine_type]
        
        # Try URL-based health check first
        if engine_config.health_check_url:
            try:
                import requests
                response = requests.get(engine_config.health_check_url, timeout=5)
                return response.status_code == 200
            except:
                return False
        
        # Try creating a test engine instance
        try:
            # Get a test model for this engine type
            test_models = self.model_registry.get_engine_models(engine_type)
            if not test_models:
                return False
            
            test_model = test_models[0]
            engine = self.get_engine(test_model.id)
            return engine.health_check()
        except:
            return False
    
    def list_available_engines(self) -> List[EngineType]:
        """List all registered engine types"""
        return list(self.engines.keys())
    
    def list_healthy_engines(self) -> List[EngineType]:
        """List engine types that pass health checks"""
        healthy = []
        for engine_type in self.engines.keys():
            if self.health_check(engine_type):
                healthy.append(engine_type)
        return healthy
    
    def get_best_engine_for_use_case(self, 
                                   use_case: str,
                                   prefer_free: bool = True,
                                   prefer_fast: bool = True) -> Optional[ModelConfig]:
        """Get the best model/engine combination for a use case"""
        
        # Get models that support the use case
        suitable_models = self.model_registry.get_models_for_use_case(use_case)
        if not suitable_models:
            return None
        
        # Filter by healthy engines
        healthy_engines = self.list_healthy_engines()
        suitable_models = [m for m in suitable_models if m.engine in healthy_engines]
        
        if not suitable_models:
            return None
        
        # Apply preferences
        if prefer_free:
            free_models = [m for m in suitable_models if m.cost_per_1k_tokens == 0.0]
            if free_models:
                suitable_models = free_models
        
        if prefer_fast:
            # Sort by latency (lower is better)
            models_with_latency = [m for m in suitable_models if m.latency_ms is not None]
            if models_with_latency:
                suitable_models = sorted(models_with_latency, key=lambda m: m.latency_ms)
        
        return suitable_models[0] if suitable_models else None
    
    def check_requirements(self, engine_type: EngineType) -> Dict[str, bool]:
        """Check if requirements are met for an engine type"""
        if engine_type not in self.engines:
            return {"engine_registered": False}
        
        engine_config = self.engines[engine_type]
        results = {"engine_registered": True}
        
        # Check Python package requirements
        for requirement in engine_config.requirements:
            try:
                importlib.import_module(requirement)
                results[f"package_{requirement}"] = True
            except ImportError:
                results[f"package_{requirement}"] = False
        
        # Check service availability
        if engine_config.health_check_url:
            results["service_available"] = self.health_check(engine_type)
        
        return results
    
    def cleanup(self, model_id: Optional[str] = None):
        """Cleanup engine instances"""
        if model_id:
            if model_id in self.active_engines:
                del self.active_engines[model_id]
        else:
            # Clean up all engines
            self.active_engines.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get engine registry statistics"""
        engine_health = {}
        for engine_type in self.engines.keys():
            engine_health[engine_type.value] = self.health_check(engine_type)
        
        return {
            "total_engines": len(self.engines),
            "active_instances": len(self.active_engines),
            "engine_health": engine_health,
            "healthy_engines": len([h for h in engine_health.values() if h]),
            "supported_engine_types": [e.value for e in self.engines.keys()]
        }


# Global registry instance
engine_registry = EngineRegistry()