"""
Engine Implementations

Unified interface for different inference backends including:
- Ollama (local inference)
- Hugging Face Transformers (GPU inference)  
- OpenAI API (cloud inference)
- Other vision-language model APIs
"""

from .base_engine import BaseEngine, EngineResponse
from .ollama_engine import OllamaEngine
from .hf_engine import HuggingFaceEngine

__all__ = [
    'BaseEngine',
    'EngineResponse', 
    'OllamaEngine',
    'HuggingFaceEngine'
]