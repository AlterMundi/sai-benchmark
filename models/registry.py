"""
Model registry and configuration definitions for all supported vision models.
Defines the available models and their specific configurations.
"""

from .base import ModelConfig, ModelCapabilities

# Define prompt templates for different model families
PROMPT_TEMPLATES = {
    "system_json": (
        "You are an early-fire detection agent. "
        "Answer ONLY with a valid JSON matching this schema: "
        '{"has_smoke": bool, "bbox": [x_center, y_center, width, height]}. '
        'The bbox values should be normalized (0-1). '
        'If no smoke is detected, use bbox: [0, 0, 0, 0]. '
        'DO NOT include any other text in your response.'
    ),
    
    "llama_format": (
        "Analyze this image for smoke or fire. "
        "Respond with JSON format: "
        '{"has_smoke": true/false, "bbox": [x, y, width, height]}. '
        'Bounding box coordinates should be normalized between 0 and 1. '
        'If no smoke/fire detected, set bbox to [0, 0, 0, 0].'
    ),
    
    "llava_format": (
        "Look at this image and detect any smoke or fire. "
        "Return your answer as JSON: "
        '{"has_smoke": boolean, "bbox": [center_x, center_y, width, height]}. '
        'Use normalized coordinates (0-1). No smoke means bbox [0,0,0,0].'
    ),
    
    "minicpm_format": (
        "Please analyze this image for smoke or fire detection. "
        "Output format must be JSON: "
        '{"has_smoke": true or false, "bbox": [x_center, y_center, w, h]}. '
        'Coordinates normalized 0-1. Empty bbox for no detection: [0,0,0,0].'
    ),
    
    "gemma_format": (
        "Analyze this image carefully for any signs of smoke or fire. "
        "Respond in JSON format only: "
        '{"has_smoke": boolean, "bbox": [x_center, y_center, width, height]}. '
        'Use normalized coordinates (0-1). If no smoke/fire is detected, set bbox to [0,0,0,0].'
    ),
    
    "mistral_format": (
        "You are an expert in wildfire detection. Examine this image for smoke or fire. "
        "Return only JSON in this exact format: "
        '{"has_smoke": true/false, "bbox": [x, y, w, h]}. '
        'Coordinates must be normalized between 0 and 1. Use [0,0,0,0] for no detection.'
    )
}

# Model configurations registry
MODEL_CONFIGS = {
    "qwen2.5vl-7b": ModelConfig(
        name="qwen2.5vl-7b",
        display_name="Qwen 2.5-VL 7B",
        model_id="qwen2.5vl:7b",
        prompt_template="system_json",
        temperature=0.1,
        max_tokens=1280,
        capabilities=ModelCapabilities(
            supports_bbox=True,
            supports_multiimage=True,
            max_resolution="1280x720",
            context_window=125000,
            languages=["en", "zh", "es", "fr", "de", "it", "ja", "ko"]
        )
    ),
    
    "qwen2.5vl-3b": ModelConfig(
        name="qwen2.5vl-3b",
        display_name="Qwen 2.5-VL 3B",
        model_id="qwen2.5vl:3b",
        prompt_template="system_json",
        temperature=0.1,
        max_tokens=1280,
        capabilities=ModelCapabilities(
            supports_bbox=True,
            supports_multiimage=True,
            max_resolution="1280x720",
            context_window=125000,
            languages=["en", "zh", "es", "fr", "de", "it", "ja", "ko"]
        )
    ),
    
    "llama3.2-vision": ModelConfig(
        name="llama3.2-vision",
        display_name="LLaMA 3.2 Vision 11B",
        model_id="llama3.2-vision:latest",
        prompt_template="llama_format",
        temperature=0.1,
        max_tokens=2048,
        capabilities=ModelCapabilities(
            supports_bbox=True,
            supports_multiimage=False,
            max_resolution="1664x1664",
            context_window=128000,
            languages=["en", "es", "fr", "de", "it", "pt", "hi", "th"]
        )
    ),
    
    "minicpm-v": ModelConfig(
        name="minicpm-v",
        display_name="MiniCPM-V 2.6",
        model_id="minicpm-v:latest",
        prompt_template="minicpm_format",
        temperature=0.1,
        max_tokens=2048,
        capabilities=ModelCapabilities(
            supports_bbox=True,
            supports_multiimage=True,
            max_resolution="1800x1800",
            context_window=32000,
            languages=["en", "zh", "de", "fr", "it", "ko"]
        )
    ),
    
    "bakllava": ModelConfig(
        name="bakllava",
        display_name="BakLLaVA",
        model_id="bakllava:latest",
        prompt_template="llava_format",
        temperature=0.1,
        max_tokens=2048,
        capabilities=ModelCapabilities(
            supports_bbox=True,
            supports_multiimage=False,
            max_resolution="336x336",
            context_window=4096,
            languages=["en"]
        )
    ),
    
    "llava-phi3": ModelConfig(
        name="llava-phi3",
        display_name="LLaVA-Phi3",
        model_id="llava-phi3:latest",
        prompt_template="llava_format",
        temperature=0.1,
        max_tokens=2048,
        capabilities=ModelCapabilities(
            supports_bbox=True,
            supports_multiimage=False,
            max_resolution="336x336",
            context_window=4096,
            languages=["en"]
        )
    ),
    
    "gemma3-27b": ModelConfig(
        name="gemma3-27b",
        display_name="Gemma 3 27B Vision",
        model_id="gemma3:27b-it-qat",
        prompt_template="gemma_format",
        temperature=0.1,
        max_tokens=8192,
        capabilities=ModelCapabilities(
            supports_bbox=True,
            supports_multiimage=False,
            max_resolution="1024x1024",
            context_window=131072,
            languages=["en", "es", "fr", "de", "it", "pt", "nl", "pl", "ru", "ja", "ko", "zh"]
        )
    ),
    
    "mistral-small-3.1": ModelConfig(
        name="mistral-small-3.1",
        display_name="Mistral Small 3.1 24B",
        model_id="mistral-small3.1:24b-instruct-2503-q4_K_M",
        prompt_template="mistral_format",
        temperature=0.1,
        max_tokens=8192,
        capabilities=ModelCapabilities(
            supports_bbox=True,
            supports_multiimage=False,
            max_resolution="1024x1024",
            context_window=131072,
            languages=["en", "fr", "es", "de", "it", "pt", "nl", "ru", "zh", "ja", "ko"]
        )
    )
}


def get_model_config(model_name: str) -> ModelConfig:
    """Get configuration for a specific model."""
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_CONFIGS.keys())}")
    return MODEL_CONFIGS[model_name]


def list_available_models() -> list:
    """List all available model configurations."""
    return list(MODEL_CONFIGS.keys())


def get_prompt_template(template_name: str) -> str:
    """Get a prompt template by name."""
    if template_name not in PROMPT_TEMPLATES:
        raise ValueError(f"Unknown template: {template_name}. Available: {list(PROMPT_TEMPLATES.keys())}")
    return PROMPT_TEMPLATES[template_name]


def filter_models_by_capability(capability: str, value: bool = True) -> list:
    """Filter models by a specific capability."""
    filtered = []
    for name, config in MODEL_CONFIGS.items():
        if hasattr(config.capabilities, capability):
            if getattr(config.capabilities, capability) == value:
                filtered.append(name)
    return filtered


def get_models_by_size() -> dict:
    """Group models by their approximate size categories."""
    size_groups = {
        "small": [],    # < 5B parameters
        "medium": [],   # 5-15B parameters  
        "large": []     # > 15B parameters
    }
    
    # Rough parameter count mapping based on model names
    size_mapping = {
        "qwen2.5vl-3b": "small",
        "llava-phi3": "small",
        "bakllava": "medium",
        "qwen2.5vl-7b": "medium", 
        "minicpm-v": "medium",
        "llama3.2-vision": "medium",
        "mistral-small-3.1": "large",
        "gemma3-27b": "large"
    }
    
    for model_name in MODEL_CONFIGS.keys():
        size_category = size_mapping.get(model_name, "medium")
        size_groups[size_category].append(model_name)
    
    return size_groups