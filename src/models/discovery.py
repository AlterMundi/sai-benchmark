"""
Model discovery and auto-configuration system.
Automatically detects available Ollama models and maps them to configurations.
"""

import requests
import logging
from typing import List, Dict, Set
from .base import ModelConfig
from .registry import MODEL_CONFIGS, get_model_config

logger = logging.getLogger(__name__)

OLLAMA_URL = "http://localhost:11434/api/tags"


def get_available_ollama_models() -> List[str]:
    """Get list of all models available in Ollama."""
    try:
        response = requests.get(OLLAMA_URL, timeout=5)
        if response.status_code == 200:
            data = response.json()
            return [model["name"] for model in data.get("models", [])]
    except Exception as e:
        logger.warning(f"Failed to fetch Ollama models: {e}")
    return []


def discover_vision_models() -> Dict[str, ModelConfig]:
    """
    Discover available vision models and return their configurations.
    
    Returns:
        Dict mapping model names to their configurations
    """
    available_ollama = set(get_available_ollama_models())
    discovered = {}
    
    logger.info(f"Found {len(available_ollama)} models in Ollama")
    
    # Check each configured model against available Ollama models
    for model_name, config in MODEL_CONFIGS.items():
        if config.model_id in available_ollama:
            discovered[model_name] = config
            logger.info(f"✓ Discovered: {config.display_name} ({config.model_id})")
        else:
            logger.debug(f"✗ Not available: {config.display_name} ({config.model_id})")
    
    return discovered


def auto_select_models(criteria: str = "balanced") -> List[str]:
    """
    Auto-select models based on criteria.
    
    Args:
        criteria: Selection criteria
            - "balanced": Mix of model sizes and capabilities
            - "fast": Prefer smaller, faster models  
            - "accurate": Prefer larger, more capable models
            - "all": All available models
    
    Returns:
        List of selected model names
    """
    discovered = discover_vision_models()
    
    if not discovered:
        logger.warning("No vision models discovered")
        return []
    
    if criteria == "all":
        return list(discovered.keys())
    
    # Get models by size categories
    from .registry import get_models_by_size
    size_groups = get_models_by_size()
    
    # Filter by available models
    available_small = [m for m in size_groups["small"] if m in discovered]
    available_medium = [m for m in size_groups["medium"] if m in discovered]
    available_large = [m for m in size_groups["large"] if m in discovered]
    
    if criteria == "fast":
        # Prefer small models, then medium
        selected = available_small[:2] + available_medium[:1]
        
    elif criteria == "accurate":
        # Prefer large models, then medium
        selected = available_large[:2] + available_medium[:2]
        
    elif criteria == "balanced":
        # Mix of all sizes
        selected = (available_small[:1] + 
                   available_medium[:2] + 
                   available_large[:1])
    else:
        raise ValueError(f"Unknown criteria: {criteria}")
    
    # Remove duplicates while preserving order
    result = []
    for model in selected:
        if model not in result:
            result.append(model)
    
    logger.info(f"Auto-selected {len(result)} models for '{criteria}' criteria: {result}")
    return result


def validate_model_availability(model_names: List[str]) -> List[str]:
    """
    Validate that specified models are available.
    
    Args:
        model_names: List of model names to validate
        
    Returns:
        List of validated (available) model names
    """
    discovered = discover_vision_models()
    validated = []
    
    for model_name in model_names:
        if model_name in discovered:
            validated.append(model_name)
            logger.info(f"✓ Validated: {model_name}")
        else:
            logger.warning(f"✗ Model not available: {model_name}")
    
    return validated


def get_model_recommendations() -> Dict[str, List[str]]:
    """
    Get model recommendations for different use cases.
    
    Returns:
        Dict mapping use cases to recommended model lists
    """
    discovered = discover_vision_models()
    
    recommendations = {
        "quick_test": [],
        "full_benchmark": [],
        "accuracy_focused": [],
        "speed_focused": []
    }
    
    # Quick test: fastest available model
    from .registry import get_models_by_size
    size_groups = get_models_by_size() 
    for model in size_groups["small"]:
        if model in discovered:
            recommendations["quick_test"].append(model)
            break
    
    # Speed focused: all small models
    recommendations["speed_focused"] = [m for m in size_groups["small"] if m in discovered]
    
    # Accuracy focused: medium and large models
    recommendations["accuracy_focused"] = (
        [m for m in size_groups["medium"] if m in discovered] +
        [m for m in size_groups["large"] if m in discovered]
    )
    
    # Full benchmark: all available models
    recommendations["full_benchmark"] = list(discovered.keys())
    
    return recommendations


def print_discovery_report():
    """Print a comprehensive discovery report."""
    print("\\n" + "="*60)
    print("SAI-BENCHMARK MODEL DISCOVERY REPORT")
    print("="*60)
    
    # Available models
    discovered = discover_vision_models()
    print(f"\\n🔍 DISCOVERED MODELS ({len(discovered)})")
    print("-" * 40)
    
    if not discovered:
        print("❌ No vision models found in Ollama")
        print("\\nTry installing models with:")
        print("  ollama pull qwen2.5vl:7b")
        print("  ollama pull llama3.2-vision:11b")
        return
    
    for name, config in discovered.items():
        caps = config.capabilities
        print(f"✓ {config.display_name}")
        print(f"  ID: {config.model_id}")
        print(f"  Context: {caps.context_window:,} tokens")
        print(f"  BBox: {'Yes' if caps.supports_bbox else 'No'}")
        print(f"  Multi-image: {'Yes' if caps.supports_multiimage else 'No'}")
        print()
    
    # Recommendations
    recommendations = get_model_recommendations()
    print("🎯 RECOMMENDATIONS")
    print("-" * 40)
    
    for use_case, models in recommendations.items():
        if models:
            print(f"{use_case.replace('_', ' ').title()}: {', '.join(models)}")
    
    print(f"\\n💡 USAGE EXAMPLES")
    print("-" * 40)
    print("# Test single model:")
    if discovered:
        first_model = list(discovered.keys())[0]
        print(f"python evaluate.py --models {first_model} --dataset ~/sequences")
    
    print("\\n# Compare multiple models:")
    balanced = auto_select_models("balanced")
    if balanced:
        print(f"python evaluate.py --models {','.join(balanced)} --dataset ~/sequences")
    
    print("\\n# Auto-select models:")
    print("python evaluate.py --models auto:balanced --dataset ~/sequences")
    print("python evaluate.py --models auto:fast --dataset ~/sequences")


if __name__ == "__main__":
    print_discovery_report()