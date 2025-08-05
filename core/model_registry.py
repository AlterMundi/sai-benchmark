"""
Model Registry System

Manages model configurations, capabilities, and engine mappings.
Supports multi-engine deployment and capability detection.
"""

from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, asdict
from enum import Enum
import yaml
from pathlib import Path


class ModelCapability(Enum):
    """Model capability enumeration"""
    VISION = "vision"
    TEXT = "text"
    JSON_OUTPUT = "json_output"
    BBOX_DETECTION = "bbox_detection"
    SEQUENCE_ANALYSIS = "sequence_analysis"
    DYNAMIC_RESOLUTION = "dynamic_resolution"
    WINDOW_ATTENTION = "window_attention"


class EngineType(Enum):
    """Supported engine types"""
    OLLAMA = "ollama"
    HUGGINGFACE = "huggingface"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"


@dataclass
class ModelConfig:
    """Model configuration with capabilities and constraints"""
    id: str
    name: str
    engine: EngineType
    model_path: str  # Path/identifier for the engine
    capabilities: Set[ModelCapability]
    max_tokens: int = 4096
    max_images: int = 1
    supports_batch: bool = False
    cost_per_1k_tokens: float = 0.0
    latency_ms: Optional[int] = None
    gpu_memory_gb: Optional[float] = None
    description: Optional[str] = None
    version: str = "1.0"
    
    def __post_init__(self):
        # Convert capabilities to set if provided as list
        if isinstance(self.capabilities, list):
            self.capabilities = set(self.capabilities)
    
    def has_capability(self, capability: ModelCapability) -> bool:
        """Check if model has specific capability"""
        return capability in self.capabilities
    
    def supports_use_case(self, use_case: str) -> bool:
        """Check if model supports a specific use case"""
        use_case_requirements = {
            "wildfire_detection": {ModelCapability.VISION, ModelCapability.JSON_OUTPUT},
            "smoke_detection": {ModelCapability.VISION, ModelCapability.BBOX_DETECTION},
            "sequence_analysis": {ModelCapability.VISION, ModelCapability.SEQUENCE_ANALYSIS},
            "early_warning": {ModelCapability.VISION, ModelCapability.JSON_OUTPUT},
            "expert_analysis": {ModelCapability.VISION, ModelCapability.TEXT}
        }
        
        required_caps = use_case_requirements.get(use_case, set())
        return required_caps.issubset(self.capabilities)


class ModelRegistry:
    """Registry for managing model configurations and capabilities"""
    
    def __init__(self, registry_path: Optional[str] = None):
        self.registry_path = registry_path or "core/models.yaml"
        self.models: Dict[str, ModelConfig] = {}
        self._load_registry()
        self._register_builtin_models()
    
    def _load_registry(self):
        """Load models from registry file if it exists"""
        if Path(self.registry_path).exists():
            with open(self.registry_path, 'r') as f:
                data = yaml.safe_load(f)
                for model_data in data.get('models', []):
                    # Convert string capabilities to enum
                    if 'capabilities' in model_data:
                        model_data['capabilities'] = {
                            ModelCapability(cap) for cap in model_data['capabilities']
                        }
                    # Convert string engine to enum
                    if 'engine' in model_data:
                        model_data['engine'] = EngineType(model_data['engine'])
                    
                    model = ModelConfig(**model_data)
                    self.models[model.id] = model
    
    def _register_builtin_models(self):
        """Register built-in model configurations"""
        
        # Qwen 2.5-VL models
        self.register_model(ModelConfig(
            id="qwen2.5vl:7b",
            name="Qwen 2.5-VL 7B (Ollama)",
            engine=EngineType.OLLAMA,
            model_path="qwen2.5vl:7b",
            capabilities={
                ModelCapability.VISION,
                ModelCapability.TEXT,
                ModelCapability.JSON_OUTPUT,
                ModelCapability.BBOX_DETECTION,
                ModelCapability.DYNAMIC_RESOLUTION,
                ModelCapability.WINDOW_ATTENTION
            },
            max_tokens=1280,
            max_images=1,
            supports_batch=False,
            cost_per_1k_tokens=0.0,  # Local inference
            latency_ms=1000,
            gpu_memory_gb=0.0,  # CPU inference via Ollama
            description="Qwen 2.5-VL 7B model running locally via Ollama with quantization",
            version="7B-GGUF"
        ))
        
        self.register_model(ModelConfig(
            id="qwen2.5-vl-7b-hf",
            name="Qwen 2.5-VL 7B (Hugging Face)",
            engine=EngineType.HUGGINGFACE,
            model_path="Qwen/Qwen2.5-VL-7B-Instruct",
            capabilities={
                ModelCapability.VISION,
                ModelCapability.TEXT,
                ModelCapability.JSON_OUTPUT,
                ModelCapability.BBOX_DETECTION,
                ModelCapability.DYNAMIC_RESOLUTION,
                ModelCapability.WINDOW_ATTENTION
            },
            max_tokens=1280,
            max_images=1,
            supports_batch=True,
            cost_per_1k_tokens=0.0,  # Local inference
            latency_ms=250,
            gpu_memory_gb=16.0,
            description="Qwen 2.5-VL 7B model via Hugging Face Transformers",
            version="7B-Instruct"
        ))
        
        # OpenAI models
        self.register_model(ModelConfig(
            id="gpt-4o",
            name="GPT-4o",
            engine=EngineType.OPENAI,
            model_path="gpt-4o",
            capabilities={
                ModelCapability.VISION,
                ModelCapability.TEXT,
                ModelCapability.JSON_OUTPUT,
                ModelCapability.BBOX_DETECTION,
                ModelCapability.SEQUENCE_ANALYSIS
            },
            max_tokens=4096,
            max_images=10,
            supports_batch=True,
            cost_per_1k_tokens=5.0,  # Approximate
            latency_ms=2000,
            description="OpenAI's GPT-4o with vision capabilities",
            version="2024-05-13"
        ))
        
        self.register_model(ModelConfig(
            id="gpt-4-vision",
            name="GPT-4 Vision",
            engine=EngineType.OPENAI,
            model_path="gpt-4-vision-preview",
            capabilities={
                ModelCapability.VISION,
                ModelCapability.TEXT,
                ModelCapability.JSON_OUTPUT,
                ModelCapability.BBOX_DETECTION
            },
            max_tokens=4096,
            max_images=10,
            supports_batch=False,
            cost_per_1k_tokens=10.0,
            latency_ms=3000,
            description="OpenAI's GPT-4 with vision capabilities",
            version="vision-preview"
        ))
        
        # Legacy models for compatibility
        self._register_legacy_models()
    
    def _register_legacy_models(self):
        """Register legacy model configurations for backward compatibility"""
        
        legacy_models = [
            ("llava:13b", "LLaVA 13B", "llava:13b"),
            ("llava:7b", "LLaVA 7B", "llava:7b"),
            ("minicpm-v:8b", "MiniCPM-V 8B", "minicpm-v:8b"),
            ("minicpm-v:latest", "MiniCPM-V Latest", "minicpm-v:latest"),
            ("llama3.2-vision:latest", "LLaMA 3.2 Vision", "llama3.2-vision:latest"),
            ("bakllava:latest", "BakLLaVA", "bakllava:latest"),
            ("llava-phi3:latest", "LLaVA Phi3", "llava-phi3:latest"),
            ("gemma2:27b", "Gemma 2 27B", "gemma2:27b"),
            ("mistral-small:22b", "Mistral Small 22B", "mistral-small:22b")
        ]
        
        for model_id, name, path in legacy_models:
            self.register_model(ModelConfig(
                id=model_id,
                name=name,
                engine=EngineType.OLLAMA,
                model_path=path,
                capabilities={
                    ModelCapability.VISION,
                    ModelCapability.TEXT,
                    ModelCapability.JSON_OUTPUT
                },
                max_tokens=2048,
                max_images=1,
                supports_batch=False,
                cost_per_1k_tokens=0.0,
                latency_ms=1500,
                description=f"Legacy {name} model via Ollama",
                version="legacy"
            ))
    
    def register_model(self, model: ModelConfig):
        """Register a new model configuration"""
        self.models[model.id] = model
    
    def get_model(self, model_id: str) -> ModelConfig:
        """Get model configuration by ID"""
        if model_id not in self.models:
            raise ValueError(f"Model '{model_id}' not found in registry")
        return self.models[model_id]
    
    def list_models(self, 
                   engine: Optional[EngineType] = None,
                   capabilities: Optional[List[ModelCapability]] = None) -> List[ModelConfig]:
        """List models, optionally filtered by engine or capabilities"""
        models = list(self.models.values())
        
        if engine:
            models = [m for m in models if m.engine == engine]
        
        if capabilities:
            cap_set = set(capabilities)
            models = [m for m in models if cap_set.issubset(m.capabilities)]
        
        return models
    
    def get_models_for_use_case(self, use_case: str) -> List[ModelConfig]:
        """Get models that support a specific use case"""
        return [model for model in self.models.values() 
                if model.supports_use_case(use_case)]
    
    def get_cheapest_model(self, capabilities: List[ModelCapability]) -> ModelConfig:
        """Get the cheapest model with required capabilities"""
        compatible_models = self.list_models(capabilities=capabilities)
        if not compatible_models:
            raise ValueError(f"No models found with capabilities: {capabilities}")
        
        return min(compatible_models, key=lambda m: m.cost_per_1k_tokens)
    
    def get_fastest_model(self, capabilities: List[ModelCapability]) -> ModelConfig:
        """Get the fastest model with required capabilities"""
        compatible_models = self.list_models(capabilities=capabilities)
        if not compatible_models:
            raise ValueError(f"No models found with capabilities: {capabilities}")
        
        # Filter out models without latency data
        models_with_latency = [m for m in compatible_models if m.latency_ms is not None]
        if not models_with_latency:
            return compatible_models[0]  # Return first if no latency data
        
        return min(models_with_latency, key=lambda m: m.latency_ms)
    
    def validate_model_engine_compatibility(self, model_id: str, engine_type: EngineType) -> bool:
        """Validate if model is compatible with specified engine"""
        model = self.get_model(model_id)
        return model.engine == engine_type
    
    def get_engine_models(self, engine: EngineType) -> List[ModelConfig]:
        """Get all models available for a specific engine"""
        return [model for model in self.models.values() if model.engine == engine]
    
    def save_registry(self, path: Optional[str] = None):
        """Save registry to YAML file"""
        save_path = path or self.registry_path
        
        registry_data = {
            "version": "1.0",
            "models": []
        }
        
        for model in self.models.values():
            model_dict = asdict(model)
            # Convert enums to strings for YAML serialization
            model_dict['engine'] = model.engine.value
            model_dict['capabilities'] = [cap.value for cap in model.capabilities]
            registry_data["models"].append(model_dict)
        
        with open(save_path, 'w') as f:
            yaml.dump(registry_data, f, default_flow_style=False, indent=2)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get registry statistics"""
        engine_counts = {}
        capability_counts = {}
        
        for model in self.models.values():
            engine_counts[model.engine.value] = engine_counts.get(model.engine.value, 0) + 1
            
            for cap in model.capabilities:
                capability_counts[cap.value] = capability_counts.get(cap.value, 0) + 1
        
        return {
            "total_models": len(self.models),
            "engines": engine_counts,
            "capabilities": capability_counts,
            "avg_latency": sum(m.latency_ms for m in self.models.values() if m.latency_ms) / 
                         len([m for m in self.models.values() if m.latency_ms]),
            "free_models": len([m for m in self.models.values() if m.cost_per_1k_tokens == 0.0]),
            "paid_models": len([m for m in self.models.values() if m.cost_per_1k_tokens > 0.0])
        }


# Global registry instance
model_registry = ModelRegistry()