"""
YOLOv8-s Detector for SAI

Implementation of YOLOv8-s architecture optimized for early smoke detection.
Based on Ultralytics YOLOv8 with modifications for high recall performance.
"""

import torch
import torch.nn as nn
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from pathlib import Path

try:
    from ultralytics import YOLO
    from ultralytics.models.yolo.detect import DetectionModel
    from ultralytics.nn.modules import Detect
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    print("Warning: Ultralytics not available. Using custom implementation.")


class YOLOv8sDetector(nn.Module):
    """
    YOLOv8-s detector optimized for smoke detection with high recall.
    
    Features:
    - Low confidence threshold (0.3-0.4) for maximum recall
    - Anchor-free detection with C2f modules
    - Multi-scale feature pyramid network
    - Optimized for smoke/fire object detection
    """
    
    def __init__(
        self,
        num_classes: int = 2,  # smoke, fire
        conf_threshold: float = 0.3,
        iou_threshold: float = 0.45,
        max_detections: int = 300,
        input_size: Tuple[int, int] = (640, 640),
        pretrained: bool = True,
        model_path: Optional[str] = None
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.max_detections = max_detections
        self.input_size = input_size
        
        # Class names for SAI
        self.class_names = ['smoke', 'fire']
        
        if ULTRALYTICS_AVAILABLE and (pretrained or model_path):
            self._init_ultralytics_model(model_path)
        else:
            self._init_custom_model()
    
    def _init_ultralytics_model(self, model_path: Optional[str] = None):
        """Initialize using Ultralytics YOLO implementation"""
        if model_path and Path(model_path).exists():
            # Load custom trained model
            self.model = YOLO(model_path)
        else:
            # Load pretrained YOLOv8s and modify for our classes
            self.model = YOLO('yolov8s.pt')
            
            # Modify the model for our specific classes if needed
            if self.num_classes != 80:  # COCO has 80 classes
                self._modify_head_for_classes()
        
        # Set inference parameters
        self.model.conf = self.conf_threshold
        self.model.iou = self.iou_threshold
        self.model.max_det = self.max_detections
    
    def _modify_head_for_classes(self):
        """Modify detection head for smoke/fire classes"""
        # This will be implemented when we need to fine-tune from COCO
        # For now, we'll use transfer learning approach
        pass
    
    def _init_custom_model(self):
        """Initialize custom YOLOv8-s implementation"""
        # Custom implementation for when Ultralytics is not available
        # This would include the full YOLOv8 architecture implementation
        print("Using custom YOLOv8-s implementation")
        
        # Placeholder for custom implementation
        self.backbone = self._build_backbone()
        self.neck = self._build_neck()
        self.head = self._build_head()
    
    def _build_backbone(self):
        """Build YOLOv8-s backbone with C2f modules"""
        # Simplified backbone implementation
        # In practice, this would be the full CSPDarknet backbone
        return nn.Sequential(
            nn.Conv2d(3, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True),
            # ... rest of backbone layers
        )
    
    def _build_neck(self):
        """Build Feature Pyramid Network neck"""
        # Simplified FPN implementation
        return nn.Identity()  # Placeholder
    
    def _build_head(self):
        """Build detection head"""
        # Simplified detection head
        return nn.Identity()  # Placeholder
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training mode
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Dictionary with detection outputs
        """
        if ULTRALYTICS_AVAILABLE and hasattr(self, 'model'):
            # Use Ultralytics model for training
            return self._forward_ultralytics(x)
        else:
            # Use custom implementation
            return self._forward_custom(x)
    
    def _forward_ultralytics(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass using Ultralytics model"""
        # During training, we need to access the underlying model
        if self.training:
            # Access the underlying PyTorch model
            results = self.model.model(x)
            return {'predictions': results}
        else:
            # During inference, use the full pipeline
            return self.detect(x)
    
    def _forward_custom(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass using custom implementation"""
        features = self.backbone(x)
        neck_features = self.neck(features)
        predictions = self.head(neck_features)
        return {'predictions': predictions}
    
    def detect(self, images: torch.Tensor) -> List[Dict[str, Any]]:
        """
        Run inference on images
        
        Args:
            images: Batch of images (B, C, H, W)
            
        Returns:
            List of detection results for each image
        """
        if ULTRALYTICS_AVAILABLE and hasattr(self, 'model'):
            return self._detect_ultralytics(images)
        else:
            return self._detect_custom(images)
    
    def _detect_ultralytics(self, images: torch.Tensor) -> List[Dict[str, Any]]:
        """Detection using Ultralytics model"""
        # Convert tensor to numpy if needed
        if isinstance(images, torch.Tensor):
            if images.dim() == 4:  # Batch dimension
                images_np = images.cpu().numpy().transpose(0, 2, 3, 1)
            else:  # Single image
                images_np = images.cpu().numpy().transpose(1, 2, 0)
        
        # Run inference
        results = self.model(images_np)
        
        # Convert results to our format
        detections = []
        for result in results:
            boxes = []
            if result.boxes is not None:
                for box in result.boxes:
                    boxes.append({
                        'bbox': box.xyxy[0].cpu().numpy(),  # [x1, y1, x2, y2]
                        'confidence': float(box.conf[0]),
                        'class_id': int(box.cls[0]),
                        'class_name': self.class_names[int(box.cls[0])] if int(box.cls[0]) < len(self.class_names) else 'unknown'
                    })
            
            detections.append({
                'boxes': boxes,
                'image_shape': result.orig_shape if hasattr(result, 'orig_shape') else images_np.shape[:2]
            })
        
        return detections
    
    def _detect_custom(self, images: torch.Tensor) -> List[Dict[str, Any]]:
        """Detection using custom implementation"""
        # Placeholder for custom detection logic
        batch_size = images.shape[0]
        detections = []
        
        for i in range(batch_size):
            # Placeholder detection result
            detections.append({
                'boxes': [],
                'image_shape': self.input_size
            })
        
        return detections
    
    def filter_smoke_detections(
        self, 
        detections: List[Dict[str, Any]], 
        min_confidence: float = 0.3,
        min_area: float = 100.0
    ) -> List[Dict[str, Any]]:
        """
        Filter detections for smoke/fire objects with SAI-specific criteria
        
        Args:
            detections: Raw detection results
            min_confidence: Minimum confidence threshold
            min_area: Minimum bounding box area
            
        Returns:
            Filtered detections
        """
        filtered_detections = []
        
        for detection in detections:
            filtered_boxes = []
            
            for box in detection['boxes']:
                # Check confidence
                if box['confidence'] < min_confidence:
                    continue
                
                # Check if it's smoke or fire class
                if box['class_name'] not in ['smoke', 'fire']:
                    continue
                
                # Check minimum area
                bbox = box['bbox']
                area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                if area < min_area:
                    continue
                
                filtered_boxes.append(box)
            
            filtered_detections.append({
                'boxes': filtered_boxes,
                'image_shape': detection['image_shape']
            })
        
        return filtered_detections
    
    def get_roi_crops(
        self, 
        images: torch.Tensor, 
        detections: List[Dict[str, Any]],
        expand_ratio: float = 0.1
    ) -> List[torch.Tensor]:
        """
        Extract ROI crops from detections for temporal verification
        
        Args:
            images: Input images (B, C, H, W)
            detections: Detection results
            expand_ratio: Ratio to expand bounding boxes
            
        Returns:
            List of ROI crops for each image
        """
        roi_crops = []
        
        for i, detection in enumerate(detections):
            image = images[i]
            crops = []
            
            for box in detection['boxes']:
                bbox = box['bbox']
                h, w = image.shape[1], image.shape[2]
                
                # Expand bounding box
                x1, y1, x2, y2 = bbox
                dx = (x2 - x1) * expand_ratio
                dy = (y2 - y1) * expand_ratio
                
                x1 = max(0, int(x1 - dx))
                y1 = max(0, int(y1 - dy))
                x2 = min(w, int(x2 + dx))
                y2 = min(h, int(y2 + dy))
                
                # Extract crop
                crop = image[:, y1:y2, x1:x2]
                if crop.numel() > 0:  # Valid crop
                    crops.append(crop)
            
            roi_crops.append(crops)
        
        return roi_crops
    
    def save_model(self, path: str):
        """Save model weights"""
        if ULTRALYTICS_AVAILABLE and hasattr(self, 'model'):
            self.model.save(path)
        else:
            torch.save(self.state_dict(), path)
    
    def load_model(self, path: str):
        """Load model weights"""
        if ULTRALYTICS_AVAILABLE:
            self.model = YOLO(path)
        else:
            self.load_state_dict(torch.load(path))
    
    @property
    def device(self):
        """Get model device"""
        if ULTRALYTICS_AVAILABLE and hasattr(self, 'model'):
            return next(self.model.model.parameters()).device
        else:
            return next(self.parameters()).device


def create_yolov8s_detector(
    num_classes: int = 2,
    conf_threshold: float = 0.3,
    pretrained: bool = True,
    model_path: Optional[str] = None
) -> YOLOv8sDetector:
    """
    Factory function to create YOLOv8-s detector
    
    Args:
        num_classes: Number of classes (2 for smoke/fire)
        conf_threshold: Confidence threshold for detections
        pretrained: Use pretrained weights
        model_path: Path to custom model weights
        
    Returns:
        Configured YOLOv8-s detector
    """
    return YOLOv8sDetector(
        num_classes=num_classes,
        conf_threshold=conf_threshold,
        pretrained=pretrained,
        model_path=model_path
    )


if __name__ == "__main__":
    # Test the detector
    detector = create_yolov8s_detector()
    
    # Test with dummy input
    dummy_input = torch.randn(1, 3, 640, 640)
    detections = detector.detect(dummy_input)
    
    print(f"Detector created successfully")
    print(f"Input shape: {dummy_input.shape}")
    print(f"Number of detections: {len(detections[0]['boxes'])}")