"""
SAI Cascade Model Configuration

Model configuration for the SAI cascade inference system.
Integrates with the model registry to provide model metadata and capabilities.
"""

from typing import Set, Optional
from dataclasses import dataclass

from core.model_registry import ModelConfig, ModelCapability, EngineType


@dataclass
class SAICascadeModelConfig(ModelConfig):
    """Configuration for SAI cascade model"""
    
    def __init__(
        self,
        model_id: str = "sai-cascade-v1",
        detector_path: Optional[str] = None,
        verifier_path: Optional[str] = None,
        **kwargs
    ):
        # Set default values for SAI cascade model
        capabilities = {
            ModelCapability.VISION,
            ModelCapability.BBOX_DETECTION,
            ModelCapability.SEQUENCE_ANALYSIS,
            ModelCapability.JSON_OUTPUT
        }
        
        super().__init__(
            id=model_id,
            name="SAI Cascade Model v1",
            engine=EngineType.OLLAMA,  # Using OLLAMA placeholder, will be SAI_RNA
            model_path=detector_path or "RNA/weights/detector.pt",
            capabilities=capabilities,
            max_tokens=2048,
            max_images=10,
            supports_batch=True,
            cost_per_1k_tokens=0.0,  # Local inference, no cost
            latency_ms=500,  # Estimated latency
            gpu_memory_gb=4.0,  # Estimated GPU memory usage
            description="SAI cascade inference system with YOLOv8-s detector and SmokeyNet-Lite temporal verifier",
            version="1.0",
            **kwargs
        )
        
        # Additional SAI-specific configuration
        self.detector_path = detector_path
        self.verifier_path = verifier_path
        self.temporal_frames = kwargs.get('temporal_frames', 3)
        self.conf_threshold = kwargs.get('conf_threshold', 0.3)
    
    def get_engine_config(self) -> dict:
        """Get engine-specific configuration"""
        return {
            'detector_path': self.detector_path,
            'verifier_path': self.verifier_path,
            'temporal_frames': self.temporal_frames,
            'conf_threshold': self.conf_threshold,
            'device': 'auto'
        }


def register_sai_cascade_model():
    """Register SAI cascade model with the model registry"""
    try:
        from core.model_registry import model_registry
        
        # Create model configuration
        sai_cascade_config = SAICascadeModelConfig(
            model_id="sai-cascade-v1",
            detector_path="RNA/weights/detector.pt",
            verifier_path="RNA/weights/verifier.pt"
        )
        
        # Register with model registry
        model_registry.register_model(sai_cascade_config)
        
        print("SAI Cascade Model registered successfully")
        
    except ImportError:
        print("Warning: Could not register SAI Cascade Model - model registry not available")
    except Exception as e:
        print(f"Failed to register SAI Cascade Model: {e}")


# Auto-register when module is imported
register_sai_cascade_model()


if __name__ == "__main__":
    # Test model configuration
    config = SAICascadeModelConfig()
    
    print(f"Model ID: {config.id}")
    print(f"Model Name: {config.name}")
    print(f"Capabilities: {config.capabilities}")
    print(f"Engine Config: {config.get_engine_config()}")
    print(f"Supports wildfire detection: {config.supports_use_case('wildfire_detection')}")
    print(f"Supports sequence analysis: {config.supports_use_case('sequence_analysis')}")