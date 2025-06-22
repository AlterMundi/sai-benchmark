"""
SAI-Benchmark Core Framework

Unified multi-dimensional vision assessment framework supporting:
- Multi-prompt testing with centralized registry
- Multi-model support across different engines
- Automated test suite execution
- Structured metrics collection and analysis
"""

from .prompt_registry import PromptRegistry
from .model_registry import ModelRegistry
from .metrics_registry import MetricsRegistry

# Import engine registry conditionally
try:
    from .engine_registry import EngineRegistry
except ImportError:
    EngineRegistry = None

__all__ = [
    'PromptRegistry',
    'ModelRegistry', 
    'EngineRegistry',
    'MetricsRegistry'
]