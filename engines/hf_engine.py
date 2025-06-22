"""
Hugging Face Engine Implementation

Interface for Hugging Face Transformers supporting GPU inference of vision-language models.
Handles model loading, image preprocessing, and generation with proper memory management.
"""

import torch
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import time
from PIL import Image
import gc

from .base_engine import BaseEngine, EngineResponse

try:
    from transformers import AutoProcessor, AutoModelForVision2Seq
    from qwen_vl_utils.vision_process import process_vision_info
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    AutoProcessor = None
    AutoModelForVision2Seq = None
    process_vision_info = None


class HuggingFaceEngine(BaseEngine):
    """Hugging Face Transformers engine for GPU inference"""
    
    def __init__(self, model_id: str, device: str = "auto", **kwargs):
        if not HF_AVAILABLE:
            raise ImportError(
                "Hugging Face Transformers not available. "
                "Install with: pip install transformers accelerate qwen-vl-utils"
            )
        
        self.device = device
        self.max_memory = kwargs.get('max_memory', None)
        self.torch_dtype = kwargs.get('torch_dtype', torch.float16)
        self.trust_remote_code = kwargs.get('trust_remote_code', True)
        self.model = None
        self.processor = None
        
        super().__init__(model_id, **kwargs)
    
    def _initialize(self):
        """Initialize Hugging Face model and processor"""
        try:
            # Determine device
            if self.device == "auto":
                if torch.cuda.is_available():
                    self.device = "cuda"
                else:
                    self.device = "cpu"
                    self.torch_dtype = torch.float32  # CPU inference uses float32
            
            print(f"Loading {self.model_id} on {self.device}...")
            
            # Load processor
            self.processor = AutoProcessor.from_pretrained(
                self.model_id,
                trust_remote_code=self.trust_remote_code
            )
            
            # Load model with appropriate settings
            model_kwargs = {
                "trust_remote_code": self.trust_remote_code,
                "torch_dtype": self.torch_dtype,
            }
            
            if self.device == "cuda":
                model_kwargs["device_map"] = "auto"
                if self.max_memory:
                    model_kwargs["max_memory"] = self.max_memory
            
            self.model = AutoModelForVision2Seq.from_pretrained(
                self.model_id,
                **model_kwargs
            )
            
            if self.device == "cpu":
                self.model = self.model.to(self.device)
            
            # Set to evaluation mode
            self.model.eval()
            
            print(f"Model {self.model_id} loaded successfully on {self.device}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Hugging Face model {self.model_id}: {e}")
    
    def _load_image(self, image_path: Union[str, Path]) -> Image.Image:
        """Load and preprocess image"""
        try:
            if isinstance(image_path, str) and image_path.startswith('data:'):
                # Handle base64 encoded images
                import base64
                import io
                header, data = image_path.split(',', 1)
                image_data = base64.b64decode(data)
                return Image.open(io.BytesIO(image_data)).convert('RGB')
            else:
                # Handle file paths
                return Image.open(image_path).convert('RGB')
        except Exception as e:
            raise ValueError(f"Failed to load image {image_path}: {e}")
    
    def generate(self, 
                prompt: str, 
                images: List[Union[str, Path]] = None,
                **kwargs) -> EngineResponse:
        """Generate response using Hugging Face model"""
        
        if not self.validate_inputs(prompt, images):
            return EngineResponse(
                content="",
                latency_ms=0,
                error="Invalid inputs provided"
            )
        
        if self.model is None or self.processor is None:
            return EngineResponse(
                content="",
                latency_ms=0,
                error="Model not initialized"
            )
        
        start_time = time.time()
        
        try:
            # Prepare messages for chat format
            messages = []
            
            if images:
                # For vision models, include images in the message
                content = []
                
                # Add images
                for img_path in images:
                    image = self._load_image(img_path)
                    content.append({"type": "image", "image": image})
                
                # Add text prompt
                content.append({"type": "text", "text": prompt})
                
                messages.append({
                    "role": "user",
                    "content": content
                })
            else:
                # Text-only message
                messages.append({
                    "role": "user", 
                    "content": prompt
                })
            
            # Process messages
            text = self.processor.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            # Process vision info for Qwen models
            if hasattr(self, '_is_qwen_model') or 'qwen' in self.model_id.lower():
                image_inputs, video_inputs = process_vision_info(messages)
                inputs = self.processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt"
                )
            else:
                # Standard processing for other models
                inputs = self.processor(text, return_tensors="pt")
            
            # Move inputs to device
            inputs = inputs.to(self.device)
            
            # Generation parameters
            generation_config = {
                "max_new_tokens": kwargs.get('max_tokens', 512),
                "do_sample": kwargs.get('do_sample', True),
                "temperature": kwargs.get('temperature', 0.7),
                "top_p": kwargs.get('top_p', 0.9),
                "pad_token_id": self.processor.tokenizer.pad_token_id,
            }
            
            # Generate response
            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, **generation_config)
            
            # Decode response
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False
            )[0]
            
            latency_ms = (time.time() - start_time) * 1000
            
            # Calculate tokens used
            tokens_used = sum(len(ids) for ids in generated_ids_trimmed)
            
            # Metadata
            metadata = {
                'model': self.model_id,
                'device': str(self.device),
                'torch_dtype': str(self.torch_dtype),
                'input_tokens': inputs.input_ids.shape[1] if hasattr(inputs, 'input_ids') else None,
                'output_tokens': tokens_used,
                'memory_allocated': torch.cuda.memory_allocated() if torch.cuda.is_available() else None,
                'memory_cached': torch.cuda.memory_reserved() if torch.cuda.is_available() else None
            }
            
            # Clean up GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return EngineResponse(
                content=self.postprocess_response(output_text),
                latency_ms=latency_ms,
                tokens_used=tokens_used,
                cost=0.0,  # Local inference is free
                metadata=metadata
            )
            
        except torch.cuda.OutOfMemoryError as e:
            # Handle GPU memory issues
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            return EngineResponse(
                content="",
                latency_ms=(time.time() - start_time) * 1000,
                error=f"GPU out of memory: {e}"
            )
            
        except Exception as e:
            return EngineResponse(
                content="",
                latency_ms=(time.time() - start_time) * 1000,
                error=f"Generation failed: {e}"
            )
    
    def health_check(self) -> bool:
        """Check if model is loaded and ready"""
        try:
            return self.model is not None and self.processor is not None
        except:
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        if self.model is None:
            return {"error": "Model not loaded"}
        
        info = {
            "model_id": self.model_id,
            "device": str(self.device),
            "torch_dtype": str(self.torch_dtype),
            "num_parameters": self.model.num_parameters() if hasattr(self.model, 'num_parameters') else None,
        }
        
        if torch.cuda.is_available() and self.device == "cuda":
            info.update({
                "cuda_device_count": torch.cuda.device_count(),
                "cuda_current_device": torch.cuda.current_device(),
                "cuda_memory_allocated": torch.cuda.memory_allocated(),
                "cuda_memory_reserved": torch.cuda.memory_reserved(),
            })
        
        return info
    
    def supports_batch(self) -> bool:
        """Hugging Face models can support batch processing"""
        return True
    
    def estimate_cost(self, prompt: str, images: List = None) -> float:
        """Local inference is free"""
        return 0.0
    
    def postprocess_response(self, raw_response: str) -> str:
        """Clean up Hugging Face model response"""
        response = raw_response.strip()
        
        # Remove common artifacts
        if response.startswith('```') and response.endswith('```'):
            lines = response.split('\n')
            if len(lines) > 2:
                response = '\n'.join(lines[1:-1])
        
        # Remove assistant prefixes if present
        prefixes_to_remove = ["Assistant:", "Response:", "Answer:"]
        for prefix in prefixes_to_remove:
            if response.startswith(prefix):
                response = response[len(prefix):].strip()
        
        return response
    
    def __del__(self):
        """Cleanup when engine is destroyed"""
        if hasattr(self, 'model') and self.model is not None:
            del self.model
        if hasattr(self, 'processor') and self.processor is not None:
            del self.processor
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()