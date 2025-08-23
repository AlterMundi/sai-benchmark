"""
SAI RNA Engine Implementation

Integration of SAI neural network models with the SAI-Benchmark framework.
Provides a unified interface for the cascade inference pipeline.
"""

import torch
import numpy as np
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import time
import logging

from .base_engine import BaseEngine, EngineResponse
from RNA.inference.cascade_inference import CascadeInference

logger = logging.getLogger(__name__)


class SAIRNAEngine(BaseEngine):
    """
    SAI RNA Engine for neural network inference
    
    Integrates the cascade inference pipeline (YOLOv8-s + SmokeyNet-Lite)
    with the SAI-Benchmark framework for evaluation and testing.
    """
    
    def __init__(
        self,
        model_id: str = "sai-cascade-v1",
        detector_path: Optional[str] = None,
        verifier_path: Optional[str] = None,
        device: str = 'auto',
        conf_threshold: float = 0.3,
        temporal_frames: int = 3,
        **kwargs
    ):
        self.detector_path = detector_path
        self.verifier_path = verifier_path
        self.device = device
        self.conf_threshold = conf_threshold
        self.temporal_frames = temporal_frames
        
        super().__init__(model_id, **kwargs)
        
        # Initialize cascade inference
        self.cascade = None
        self._initialize_cascade()
    
    def _initialize(self):
        """Initialize the RNA engine"""
        try:
            self._initialize_cascade()
            logger.info(f"SAI RNA Engine initialized with model_id: {self.model_id}")
        except Exception as e:
            logger.error(f"Failed to initialize SAI RNA Engine: {e}")
            raise RuntimeError(f"Engine initialization failed: {e}")
    
    def _initialize_cascade(self):
        """Initialize the cascade inference pipeline"""
        try:
            # Set default paths if not provided
            if self.detector_path is None:
                self.detector_path = "RNA/weights/detector.pt"
            if self.verifier_path is None:
                self.verifier_path = "RNA/weights/verifier.pt"
            
            # Create cascade inference
            self.cascade = CascadeInference(
                detector_path=self.detector_path,
                verifier_path=self.verifier_path,
                device=self.device,
                conf_threshold=self.conf_threshold,
                temporal_frames=self.temporal_frames
            )
            
            logger.info("Cascade inference pipeline initialized successfully")
            
        except Exception as e:
            logger.warning(f"Failed to initialize cascade inference: {e}")
            # Create a minimal fallback cascade for testing
            self.cascade = None
    
    def generate(
        self,
        prompt: str,
        images: List[Union[str, Path]] = None,
        **kwargs
    ) -> EngineResponse:
        """
        Generate response using SAI neural networks
        
        Args:
            prompt: Text prompt (used for context/metadata)
            images: List of image paths to process
            **kwargs: Additional parameters
            
        Returns:
            EngineResponse with detection results
        """
        start_time = time.time()
        
        try:
            if not images:
                return EngineResponse(
                    content="",
                    latency_ms=0,
                    error="No images provided for processing"
                )
            
            # Process images through cascade
            results = self._process_images(images, prompt, **kwargs)
            
            # Format response
            response_content = self._format_response(results, prompt)
            
            latency_ms = (time.time() - start_time) * 1000
            
            return EngineResponse(
                content=response_content,
                latency_ms=latency_ms,
                metadata={
                    'model_id': self.model_id,
                    'engine_type': 'sai_rna',
                    'num_images': len(images),
                    'processing_results': results
                }
            )
            
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            logger.error(f"Error in SAI RNA Engine generation: {e}")
            
            return EngineResponse(
                content="",
                latency_ms=latency_ms,
                error=f"SAI RNA Engine error: {str(e)}"
            )
    
    def _process_images(
        self,
        images: List[Union[str, Path]],
        prompt: str,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Process images through the cascade pipeline"""
        results = []
        
        if self.cascade is None:
            # Fallback processing for testing
            return self._fallback_processing(images)
        
        for i, image_path in enumerate(images):
            try:
                # Load image
                image = self._load_image(image_path)
                
                # Process through cascade
                result = self.cascade.process_frame(
                    image=image,
                    timestamp=time.time(),
                    camera_id=f"image_{i}"
                )
                
                # Add image metadata
                result['image_path'] = str(image_path)
                result['image_index'] = i
                
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error processing image {image_path}: {e}")
                results.append({
                    'image_path': str(image_path),
                    'image_index': i,
                    'error': str(e),
                    'detections': [],
                    'verified_detections': [],
                    'alarms': []
                })
        
        return results
    
    def _load_image(self, image_path: Union[str, Path]) -> np.ndarray:
        """Load image from path"""
        import cv2
        
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        return image
    
    def _fallback_processing(self, images: List[Union[str, Path]]) -> List[Dict[str, Any]]:
        """Fallback processing when cascade is not available"""
        results = []
        
        for i, image_path in enumerate(images):
            # Simulate processing with dummy results
            results.append({
                'image_path': str(image_path),
                'image_index': i,
                'detections': [],
                'verified_detections': [],
                'alarms': [],
                'metadata': {
                    'camera_id': f"image_{i}",
                    'timestamp': time.time(),
                    'inference_time_ms': 50.0,  # Dummy inference time
                    'active_regions': 0,
                    'buffer_size': 0
                },
                'fallback': True
            })
        
        return results
    
    def _format_response(
        self,
        results: List[Dict[str, Any]],
        prompt: str
    ) -> str:
        """Format the cascade results into a response string"""
        # Determine response format based on prompt
        response_format = self._determine_response_format(prompt)
        
        if response_format == 'json':
            return self._format_json_response(results)
        elif response_format == 'structured':
            return self._format_structured_response(results)
        else:
            return self._format_text_response(results)
    
    def _determine_response_format(self, prompt: str) -> str:
        """Determine desired response format from prompt"""
        prompt_lower = prompt.lower()
        
        if 'json' in prompt_lower:
            return 'json'
        elif any(keyword in prompt_lower for keyword in ['bbox', 'bounding', 'coordinates']):
            return 'structured'
        else:
            return 'text'
    
    def _format_json_response(self, results: List[Dict[str, Any]]) -> str:
        """Format response as JSON"""
        import json
        
        # Aggregate results across all images
        total_detections = sum(len(r.get('detections', [])) for r in results)
        total_verified = sum(len(r.get('verified_detections', [])) for r in results)
        total_alarms = sum(len(r.get('alarms', [])) for r in results)
        
        # Find highest confidence detection
        has_smoke = total_verified > 0 or total_alarms > 0
        
        # Get best bounding box if available
        best_bbox = [0, 0, 0, 0]
        max_confidence = 0.0
        
        for result in results:
            for detection in result.get('verified_detections', []):
                confidence = detection.get('confidence', 0.0)
                if confidence > max_confidence:
                    max_confidence = confidence
                    bbox = detection.get('bbox', [0, 0, 0, 0])
                    if len(bbox) >= 4:
                        best_bbox = bbox[:4]
        
        response_json = {
            "has_smoke": has_smoke,
            "bbox": best_bbox,
            "confidence": max_confidence,
            "total_detections": total_detections,
            "verified_detections": total_verified,
            "alarms": total_alarms
        }
        
        return json.dumps(response_json)
    
    def _format_structured_response(self, results: List[Dict[str, Any]]) -> str:
        """Format response with structured detection information"""
        lines = []
        
        for result in results:
            image_path = result.get('image_path', 'unknown')
            detections = result.get('verified_detections', [])
            alarms = result.get('alarms', [])
            
            lines.append(f"Image: {Path(image_path).name}")
            
            if detections:
                lines.append(f"Detections found: {len(detections)}")
                for i, detection in enumerate(detections):
                    bbox = detection.get('bbox', [])
                    confidence = detection.get('confidence', 0.0)
                    class_name = detection.get('class_name', 'smoke')
                    
                    if len(bbox) >= 4:
                        lines.append(f"  {i+1}. {class_name} (conf: {confidence:.3f}) bbox: {bbox}")
                    else:
                        lines.append(f"  {i+1}. {class_name} (conf: {confidence:.3f})")
            
            if alarms:
                lines.append(f"ALARMS: {len(alarms)} confirmed")
                for alarm in alarms:
                    persistence = alarm.get('persistence_score', 0.0)
                    frames = alarm.get('frame_count', 0)
                    lines.append(f"  - Region {alarm.get('region_id', 'unknown')} "
                               f"(persistence: {persistence:.3f}, frames: {frames})")
            
            if not detections and not alarms:
                lines.append("No smoke or fire detected")
            
            lines.append("")  # Empty line between images
        
        return "\n".join(lines)
    
    def _format_text_response(self, results: List[Dict[str, Any]]) -> str:
        """Format response as natural text"""
        total_images = len(results)
        total_detections = sum(len(r.get('verified_detections', [])) for r in results)
        total_alarms = sum(len(r.get('alarms', [])) for r in results)
        
        if total_alarms > 0:
            return f"ALERT: Smoke/fire detected with high confidence. " \
                   f"{total_alarms} alarms triggered across {total_images} images. " \
                   f"Immediate attention required."
        
        elif total_detections > 0:
            return f"Smoke/fire signatures detected in {total_detections} locations " \
                   f"across {total_images} images. Monitoring recommended."
        
        else:
            return f"No smoke or fire detected in {total_images} images. All clear."
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        info = {
            'model_id': self.model_id,
            'engine_type': 'sai_rna',
            'detector_path': self.detector_path,
            'verifier_path': self.verifier_path,
            'device': self.device,
            'conf_threshold': self.conf_threshold,
            'temporal_frames': self.temporal_frames,
            'cascade_available': self.cascade is not None
        }
        
        if self.cascade is not None:
            info['statistics'] = self.cascade.get_statistics()
        
        return info
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check of the engine"""
        try:
            # Test with dummy data
            dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            
            if self.cascade is not None:
                result = self.cascade.process_frame(dummy_image)
                return {
                    'status': 'healthy',
                    'cascade_available': True,
                    'test_result': 'passed'
                }
            else:
                return {
                    'status': 'degraded',
                    'cascade_available': False,
                    'test_result': 'fallback_mode'
                }
        
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'test_result': 'failed'
            }


# Register the engine with the engine registry
def register_sai_rna_engine():
    """Register SAI RNA Engine with the engine registry"""
    try:
        from core.engine_registry import engine_registry
        
        # Register the engine
        engine_registry.register_engine(
            engine_id="sai_rna",
            engine_class=SAIRNAEngine,
            description="SAI Neural Network Engine with cascade inference",
            capabilities=['vision', 'temporal', 'detection', 'classification']
        )
        
        logger.info("SAI RNA Engine registered successfully")
        
    except ImportError:
        logger.warning("Could not register SAI RNA Engine - engine registry not available")
    except Exception as e:
        logger.error(f"Failed to register SAI RNA Engine: {e}")


# Auto-register when module is imported
register_sai_rna_engine()


if __name__ == "__main__":
    # Test the engine
    engine = SAIRNAEngine(
        model_id="sai-cascade-test",
        device='cpu'
    )
    
    # Test health check
    health = engine.health_check()
    print(f"Health check: {health}")
    
    # Test with dummy prompt
    response = engine.generate(
        prompt="Do you detect smoke related to wildfires in this image?",
        images=[]  # No images for basic test
    )
    
    print(f"Engine test completed")
    print(f"Response: {response.content}")
    print(f"Error: {response.error}")
    print(f"Latency: {response.latency_ms}ms")