"""
Hugging Face Transformers backend wrapper for Qwen 2.5-VL model.
Provides direct GPU inference using the transformers library.
"""

import json
import torch
import logging
from pathlib import Path
from typing import Dict, Union, Optional
from PIL import Image

# Lazy imports to avoid loading if not needed
_model = None
_processor = None
_tokenizer = None

logger = logging.getLogger(__name__)

# Configuration
MODEL_CHECKPOINT = "Qwen/Qwen2.5-VL-7B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

# System prompt for early-fire detection
SYSTEM_PROMPT = (
    "You are an early-fire detection agent. "
    "Return ONLY JSON: {\"has_smoke\": bool, \"bbox\": [x_center, y_center, width, height]}. "
    "The bbox values should be normalized (0-1). "
    "If no smoke is detected, use bbox: [0, 0, 0, 0]."
)


def _lazy_load_model():
    """Lazy load the model, processor, and tokenizer."""
    global _model, _processor, _tokenizer
    
    if _model is None:
        logger.info(f"Loading model {MODEL_CHECKPOINT}...")
        
        try:
            from transformers import AutoTokenizer, Qwen2VLForConditionalGeneration, AutoProcessor
        except ImportError:
            raise ImportError(
                "Please install transformers and qwen-vl-utils: "
                "pip install transformers accelerate qwen-vl-utils"
            )
        
        # Load tokenizer
        _tokenizer = AutoTokenizer.from_pretrained(
            MODEL_CHECKPOINT, 
            trust_remote_code=True
        )
        
        # Load processor
        _processor = AutoProcessor.from_pretrained(
            MODEL_CHECKPOINT, 
            trust_remote_code=True
        )
        
        # Load model
        _model = Qwen2VLForConditionalGeneration.from_pretrained(
            MODEL_CHECKPOINT,
            torch_dtype=TORCH_DTYPE,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True
        )
        
        if not torch.cuda.is_available():
            _model = _model.to(DEVICE)
        
        logger.info(f"Model loaded on {DEVICE}")
    
    return _model, _processor, _tokenizer


def infer(image_path: Union[str, Path],
          prompt_override: Optional[str] = None,
          max_new_tokens: int = 64,
          temperature: float = 0.1) -> Dict:
    """
    Run inference on a single image for smoke detection.
    
    Args:
        image_path: Path to the image file
        prompt_override: Optional custom prompt (defaults to SYSTEM_PROMPT)
        max_new_tokens: Maximum tokens to generate (default 64)
        temperature: Model temperature (default 0.1 for consistent outputs)
    
    Returns:
        Dict with format: {"has_smoke": bool, "bbox": [x, y, w, h]}
    
    Raises:
        Exception: If inference fails or response cannot be parsed
    """
    # Load model if not already loaded
    model, processor, tokenizer = _lazy_load_model()
    
    try:
        # Load and preprocess image
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        image = Image.open(image_path).convert("RGB")
        
        # Prepare prompt
        prompt = prompt_override or SYSTEM_PROMPT
        
        # Create conversation format expected by Qwen2-VL
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        # Process inputs
        text = processor.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        inputs = processor(
            text=[text],
            images=[image],
            return_tensors="pt",
            padding=True
        ).to(model.device)
        
        # Generate response
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # Decode output
        generated_ids = output_ids[0][inputs.input_ids.shape[1]:]
        output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # Parse JSON response
        try:
            # Clean output text (remove any markdown formatting)
            output_text = output_text.strip()
            if output_text.startswith("```json"):
                output_text = output_text[7:]
            if output_text.endswith("```"):
                output_text = output_text[:-3]
            output_text = output_text.strip()
            
            parsed = json.loads(output_text)
            
            # Validate schema
            if not isinstance(parsed.get("has_smoke"), bool):
                raise ValueError("Missing or invalid 'has_smoke' field")
            
            bbox = parsed.get("bbox", [0, 0, 0, 0])
            if not isinstance(bbox, list) or len(bbox) != 4:
                raise ValueError("Invalid bbox format")
            
            # Ensure bbox values are floats and normalized
            bbox = [float(x) for x in bbox]
            
            return {
                "has_smoke": parsed["has_smoke"],
                "bbox": bbox
            }
            
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Failed to parse model output: {output_text}")
            # Return safe default on parse error
            return {
                "has_smoke": False,
                "bbox": [0, 0, 0, 0],
                "error": f"Parse error: {str(e)}",
                "raw_output": output_text
            }
            
    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise


def check_gpu_available() -> bool:
    """Check if GPU is available for inference."""
    return torch.cuda.is_available()


def get_device_info() -> Dict:
    """Get information about the compute device."""
    info = {
        "device": DEVICE,
        "cuda_available": torch.cuda.is_available(),
    }
    
    if torch.cuda.is_available():
        info.update({
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_memory_total": torch.cuda.get_device_properties(0).total_memory / 1e9,  # GB
            "gpu_memory_allocated": torch.cuda.memory_allocated(0) / 1e9,  # GB
        })
    
    return info


def estimate_memory_usage() -> float:
    """Estimate memory usage for the model in GB."""
    # Rough estimates based on model size and precision
    model_params = {
        "3B": 3e9,
        "7B": 7e9,
        "72B": 72e9
    }
    
    # Extract model size from checkpoint name
    size = "7B"  # default
    for key in model_params:
        if key in MODEL_CHECKPOINT:
            size = key
            break
    
    params = model_params[size]
    
    # Calculate based on dtype
    bytes_per_param = 2 if TORCH_DTYPE == torch.float16 else 4
    
    # Add overhead for activations and optimizer states
    overhead_factor = 1.5
    
    return (params * bytes_per_param * overhead_factor) / 1e9  # GB


if __name__ == "__main__":
    # Test the module
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python hf_qwen.py <image_path>")
        sys.exit(1)
    
    print("Device info:", json.dumps(get_device_info(), indent=2))
    print(f"Estimated memory usage: {estimate_memory_usage():.1f} GB")
    
    if not check_gpu_available():
        print("Warning: No GPU available, inference will be slow")
    
    try:
        result = infer(sys.argv[1])
        print("\nResult:", json.dumps(result, indent=2))
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)