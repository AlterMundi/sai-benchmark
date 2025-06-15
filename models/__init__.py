"""
Model wrappers for Qwen 2.5-VL inference.

This package provides unified interfaces for both Ollama and Hugging Face
backends to run Qwen 2.5-VL models for early-fire detection.
"""

from .ollama_qwen import infer as ollama_infer, check_connection as ollama_check
from .hf_qwen import infer as hf_infer, check_gpu_available as hf_check_gpu

__all__ = [
    'ollama_infer',
    'ollama_check', 
    'hf_infer',
    'hf_check_gpu'
]