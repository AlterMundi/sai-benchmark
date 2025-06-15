"""
Vision models package for SAI-Benchmark.
Provides unified interface for different vision-language models.
"""

# Import all model implementations to trigger registration
from .qwen_model import QwenVisionModel
from .llama_model import LlamaVisionModel
from .minicpm_model import MiniCPMVisionModel
from .llava_model import LlavaVisionModel
from .gemma_model import Gemma3VisionModel
from .mistral_model import MistralVisionModel

# Import base classes and registry
from .base import VisionModel, ModelConfig, ModelCapabilities, ModelFactory
from .registry import (
    MODEL_CONFIGS, 
    get_model_config, 
    list_available_models,
    get_prompt_template,
    filter_models_by_capability,
    get_models_by_size
)

# Legacy imports for backward compatibility
from .qwen_model import check_connection, get_available_models
try:
    from .hf_qwen import infer as hf_infer, check_gpu_available as hf_check_gpu
except ImportError:
    def hf_infer(*args, **kwargs):
        raise NotImplementedError("HuggingFace backend not available")
    def hf_check_gpu():
        return False

# Legacy aliases
def ollama_infer(image_path, prompt_override=None, temperature=0.1):
    """Legacy function for backward compatibility."""
    config = get_model_config("qwen2.5vl-7b") 
    model = QwenVisionModel(config)
    return model.infer(image_path, prompt_override, temperature)

ollama_check = check_connection

__all__ = [
    # Base classes
    'VisionModel', 'ModelConfig', 'ModelCapabilities', 'ModelFactory',
    
    # Model implementations
    'QwenVisionModel', 'LlamaVisionModel', 'MiniCPMVisionModel', 'LlavaVisionModel',
    'Gemma3VisionModel', 'MistralVisionModel',
    
    # Registry functions
    'get_model_config', 'list_available_models', 'get_prompt_template',
    'filter_models_by_capability', 'get_models_by_size',
    
    # Legacy functions
    'check_connection', 'get_available_models', 'ollama_infer', 'ollama_check', 
    'hf_infer', 'hf_check_gpu'
]