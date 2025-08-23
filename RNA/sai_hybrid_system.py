#!/usr/bin/env python3
"""
SAI Hybrid System - Static Model with Temporal Voting Logic

Combines our trained models:
- YOLOv8-s detector (RNA/models/detector_best.pt)  
- EfficientNet-B0 verificator (RNA/training/runs/verificator_training/verificator_best.pt)

With temporal voting logic: 3 of 5 consecutive frames must be positive for alarm confirmation.

Author: SAI Development Team
Date: 2025-08-22
"""

import torch
import torch.nn.functional as F
import torch.nn as nn
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np
import cv2
from pathlib import Path
import time
from collections import deque
import logging
from dataclasses import dataclass

# Try to import YOLOv8
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: Ultralytics not available. Install with: pip install ultralytics")

# Try to import EfficientNet
try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    try:
        from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
        TORCHVISION_AVAILABLE = True
        TIMM_AVAILABLE = False
    except ImportError:
        TORCHVISION_AVAILABLE = False
        TIMM_AVAILABLE = False
        print("Warning: Neither timm nor torchvision available for EfficientNet")

logger = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    """Single detection result from detector"""
    bbox: List[float]  # [x1, y1, x2, y2]
    confidence: float
    class_id: int
    class_name: str


@dataclass
class VerificationResult:
    """Single verification result from verificator"""
    is_positive: bool
    confidence: float
    probabilities: List[float]  # [true_detection_prob, false_positive_prob]


@dataclass
class SAIResult:
    """Final SAI system result"""
    alert: bool
    confidence: float
    detections_count: int
    verified_detections: List[DetectionResult]
    temporal_votes: List[bool]
    reason: str


class EfficientNetVerificator(nn.Module):
    """
    EfficientNet-B0 verificator model matching our trained architecture
    """
    
    def __init__(self, num_classes: int = 2, dropout: float = 0.3):
        super().__init__()
        
        if TIMM_AVAILABLE:
            # Use timm version (preferred)
            self.backbone = timm.create_model('efficientnet_b0', pretrained=False, num_classes=0)
            feature_dim = 1280
        elif TORCHVISION_AVAILABLE:
            # Use torchvision version
            self.backbone = efficientnet_b0(weights=None)
            self.backbone.classifier = nn.Identity()
            feature_dim = 1280
        else:
            # Fallback to simple CNN
            print("Warning: Using simplified CNN backbone")
            self.backbone = self._create_simple_backbone()
            feature_dim = 512
        
        # Classifier head (matches our training script)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes)
        )
    
    def _create_simple_backbone(self):
        """Simple CNN when EfficientNet not available"""
        return nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 256, 3, 2, 1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 512, 3, 2, 1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits


class TemporalVotingBuffer:
    """
    Buffer for temporal voting logic
    Maintains sliding window of N frames and implements voting
    """
    
    def __init__(self, window_size: int = 5, majority_threshold: int = 3):
        self.window_size = window_size
        self.majority_threshold = majority_threshold
        self.votes_buffer = deque(maxlen=window_size)
        self.timestamps = deque(maxlen=window_size)
    
    def add_vote(self, vote: bool, timestamp: Optional[float] = None):
        """Add a vote to the buffer"""
        if timestamp is None:
            timestamp = time.time()
        
        self.votes_buffer.append(vote)
        self.timestamps.append(timestamp)
    
    def get_decision(self) -> Tuple[bool, List[bool], float]:
        """
        Get voting decision
        Returns: (decision, votes_list, confidence)
        """
        votes_list = list(self.votes_buffer)
        
        if len(votes_list) < self.window_size:
            # Not enough votes yet
            return False, votes_list, 0.0
        
        positive_votes = sum(votes_list)
        decision = positive_votes >= self.majority_threshold
        confidence = positive_votes / len(votes_list)
        
        return decision, votes_list, confidence
    
    def reset(self):
        """Reset the buffer"""
        self.votes_buffer.clear()
        self.timestamps.clear()
    
    def __len__(self):
        return len(self.votes_buffer)


class SAIHybridSystem:
    """
    Complete SAI Hybrid System
    
    Combines:
    - YOLOv8-s detector for high-recall detection
    - EfficientNet-B0 verificator for false positive reduction  
    - Temporal voting logic for robust decision making
    """
    
    def __init__(
        self,
        detector_path: str,
        verificator_path: str,
        device: str = 'auto',
        conf_threshold: float = 0.3,
        temporal_window: int = 5,
        majority_threshold: int = 3,
        crop_size: int = 224
    ):
        self.device = self._setup_device(device)
        self.conf_threshold = conf_threshold
        self.crop_size = crop_size
        
        # Load models
        self.detector = self._load_detector(detector_path)
        self.verificator = self._load_verificator(verificator_path)
        
        # Temporal voting system
        self.temporal_buffer = TemporalVotingBuffer(
            window_size=temporal_window,
            majority_threshold=majority_threshold
        )
        
        # Statistics
        self.stats = {
            'total_frames_processed': 0,
            'total_detections': 0,
            'total_verifications': 0,
            'confirmed_alerts': 0,
            'avg_inference_time_ms': 0.0
        }
        
        logger.info(f"SAI Hybrid System initialized on {self.device}")
        logger.info(f"Temporal voting: {majority_threshold}/{temporal_window} frames")
    
    def _setup_device(self, device: str) -> torch.device:
        """Setup computation device"""
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        return torch.device(device)
    
    def _load_detector(self, model_path: str) -> Optional[YOLO]:
        """Load YOLOv8 detector"""
        if not YOLO_AVAILABLE:
            logger.error("YOLO not available. Install ultralytics.")
            return None
        
        if not Path(model_path).exists():
            logger.error(f"Detector model not found: {model_path}")
            return None
        
        try:
            detector = YOLO(model_path)
            detector.conf = self.conf_threshold
            logger.info(f"Loaded detector from {model_path}")
            return detector
        except Exception as e:
            logger.error(f"Failed to load detector: {e}")
            return None
    
    def _load_verificator(self, model_path: str) -> Optional[EfficientNetVerificator]:
        """Load EfficientNet verificator"""
        if not Path(model_path).exists():
            logger.error(f"Verificator model not found: {model_path}")
            return None
        
        try:
            # Create model
            verificator = EfficientNetVerificator(num_classes=2)
            
            # Load weights
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Handle different checkpoint formats
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            # Remove 'module.' prefix if present (from DataParallel)
            cleaned_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('module.'):
                    cleaned_state_dict[key[7:]] = value
                else:
                    cleaned_state_dict[key] = value
            
            verificator.load_state_dict(cleaned_state_dict)
            verificator.to(self.device)
            verificator.eval()
            
            logger.info(f"Loaded verificator from {model_path}")
            return verificator
        except Exception as e:
            logger.error(f"Failed to load verificator: {e}")
            return None
    
    def _preprocess_image(self, image: Union[np.ndarray, str]) -> np.ndarray:
        """Preprocess input image"""
        if isinstance(image, str):
            # Load from path
            image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif isinstance(image, np.ndarray) and len(image.shape) == 3:
            # Already loaded image
            if image.shape[-1] == 3:  # RGB
                pass
            else:  # BGR
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        return image
    
    def _run_detector(self, image: np.ndarray) -> List[DetectionResult]:
        """Run YOLOv8 detector on image"""
        if self.detector is None:
            return []
        
        try:
            results = self.detector(image, verbose=False)
            detections = []
            
            if results[0].boxes is not None:
                boxes = results[0].boxes
                for i in range(len(boxes)):
                    bbox = boxes.xyxy[i].cpu().numpy()  # [x1, y1, x2, y2]
                    conf = float(boxes.conf[i])
                    cls = int(boxes.cls[i])
                    
                    # Filter by confidence
                    if conf >= self.conf_threshold:
                        class_names = ['fire', 'smoke']  # Assuming these are our classes
                        class_name = class_names[cls] if cls < len(class_names) else 'unknown'
                        
                        detections.append(DetectionResult(
                            bbox=bbox.tolist(),
                            confidence=conf,
                            class_id=cls,
                            class_name=class_name
                        ))
            
            return detections
        except Exception as e:
            logger.error(f"Detector error: {e}")
            return []
    
    def _extract_crop(self, image: np.ndarray, bbox: List[float]) -> Optional[np.ndarray]:
        """Extract crop from bounding box"""
        try:
            h, w = image.shape[:2]
            x1, y1, x2, y2 = [int(coord) for coord in bbox]
            
            # Clamp coordinates
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            if x2 <= x1 or y2 <= y1:
                return None
            
            # Extract crop
            crop = image[y1:y2, x1:x2]
            
            # Resize to model input size
            crop = cv2.resize(crop, (self.crop_size, self.crop_size))
            
            return crop
        except Exception as e:
            logger.error(f"Crop extraction error: {e}")
            return None
    
    def _run_verificator(self, crop: np.ndarray) -> Optional[VerificationResult]:
        """Run verificator on crop"""
        if self.verificator is None or crop is None:
            return None
        
        try:
            # Preprocess crop for model
            crop_tensor = torch.from_numpy(crop).float() / 255.0  # [0, 1]
            crop_tensor = crop_tensor.permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
            crop_tensor = crop_tensor.unsqueeze(0)  # Add batch dimension
            crop_tensor = crop_tensor.to(self.device)
            
            # Run inference
            with torch.no_grad():
                logits = self.verificator(crop_tensor)
                probabilities = torch.softmax(logits, dim=1)[0]  # [true_detection, false_positive]
                
                true_detection_prob = float(probabilities[0])
                false_positive_prob = float(probabilities[1])
                
                # Decision: true_detection if prob > 0.5
                is_positive = true_detection_prob > 0.5
                confidence = max(true_detection_prob, false_positive_prob)
                
                return VerificationResult(
                    is_positive=is_positive,
                    confidence=confidence,
                    probabilities=[true_detection_prob, false_positive_prob]
                )
        except Exception as e:
            logger.error(f"Verificator error: {e}")
            return None
    
    def process_frame(self, image: Union[np.ndarray, str]) -> SAIResult:
        """
        Process single frame through complete SAI system
        
        Args:
            image: Input image (numpy array or path)
            
        Returns:
            SAIResult with temporal voting decision
        """
        start_time = time.time()
        
        # Preprocess image
        processed_image = self._preprocess_image(image)
        
        # Stage 1: Detection
        detections = self._run_detector(processed_image)
        
        # Stage 2: Verification for each detection
        frame_vote = False  # Default vote for this frame
        verified_detections = []
        
        for detection in detections:
            crop = self._extract_crop(processed_image, detection.bbox)
            verification = self._run_verificator(crop)
            
            if verification and verification.is_positive:
                verified_detections.append(detection)
                frame_vote = True  # At least one positive verification
        
        # Stage 3: Temporal voting
        self.temporal_buffer.add_vote(frame_vote)
        final_decision, votes_history, temporal_confidence = self.temporal_buffer.get_decision()
        
        # Update statistics
        inference_time = (time.time() - start_time) * 1000  # ms
        self._update_stats(detections, verified_detections, final_decision, inference_time)
        
        # Determine reason
        if not detections:
            reason = "No detections found"
        elif not verified_detections:
            reason = "All detections rejected by verificator"
        elif len(self.temporal_buffer) < self.temporal_buffer.window_size:
            reason = f"Buffering frames ({len(self.temporal_buffer)}/{self.temporal_buffer.window_size})"
        elif final_decision:
            positive_votes = sum(votes_history)
            reason = f"FIRE CONFIRMED: {positive_votes}/{len(votes_history)} frames positive"
        else:
            positive_votes = sum(votes_history)
            reason = f"Insufficient evidence: {positive_votes}/{len(votes_history)} frames positive"
        
        return SAIResult(
            alert=final_decision,
            confidence=temporal_confidence,
            detections_count=len(detections),
            verified_detections=verified_detections,
            temporal_votes=votes_history,
            reason=reason
        )
    
    def _update_stats(self, detections, verified_detections, final_decision, inference_time):
        """Update system statistics"""
        self.stats['total_frames_processed'] += 1
        self.stats['total_detections'] += len(detections)
        self.stats['total_verifications'] += len(verified_detections)
        if final_decision:
            self.stats['confirmed_alerts'] += 1
        
        # Update average inference time (exponential moving average)
        alpha = 0.1
        self.stats['avg_inference_time_ms'] = (
            alpha * inference_time + 
            (1 - alpha) * self.stats['avg_inference_time_ms']
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get system performance statistics"""
        stats = self.stats.copy()
        
        if stats['total_frames_processed'] > 0:
            stats['alert_rate'] = stats['confirmed_alerts'] / stats['total_frames_processed']
            stats['detection_rate'] = stats['total_detections'] / stats['total_frames_processed']
            stats['verification_rate'] = (
                stats['total_verifications'] / max(1, stats['total_detections'])
            )
        
        return stats
    
    def reset_temporal_buffer(self):
        """Reset temporal voting buffer"""
        self.temporal_buffer.reset()
        logger.info("Temporal buffer reset")
    
    def reset_statistics(self):
        """Reset performance statistics"""
        self.stats = {
            'total_frames_processed': 0,
            'total_detections': 0,
            'total_verifications': 0,
            'confirmed_alerts': 0,
            'avg_inference_time_ms': 0.0
        }
        logger.info("Statistics reset")


def create_sai_system(
    detector_path: str = "RNA/models/detector_best.pt",
    verificator_path: str = "RNA/training/runs/verificator_training/verificator_best.pt",
    device: str = 'auto',
    **kwargs
) -> SAIHybridSystem:
    """
    Factory function to create SAI Hybrid System
    
    Args:
        detector_path: Path to YOLOv8 detector weights
        verificator_path: Path to EfficientNet verificator weights  
        device: Computation device ('auto', 'cuda', 'cpu')
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured SAI Hybrid System
    """
    return SAIHybridSystem(
        detector_path=detector_path,
        verificator_path=verificator_path,
        device=device,
        **kwargs
    )


def main():
    """Test the SAI Hybrid System"""
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create SAI system
    sai = create_sai_system(
        detector_path="RNA/models/detector_best.pt",
        verificator_path="RNA/training/runs/verificator_training/verificator_best.pt",
        device='auto',
        temporal_window=5,
        majority_threshold=3
    )
    
    print("🔥 SAI Hybrid System Test")
    print("=" * 50)
    print(f"Device: {sai.device}")
    print(f"Temporal voting: {sai.temporal_buffer.majority_threshold}/{sai.temporal_buffer.window_size}")
    print()
    
    # Test with dummy images (replace with real test images)
    test_images = []
    for i in range(7):  # Test 7 frames
        # Create dummy image
        dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        test_images.append(dummy_image)
    
    # Process frames sequentially
    for i, image in enumerate(test_images):
        print(f"Processing frame {i+1}/{len(test_images)}...")
        result = sai.process_frame(image)
        
        print(f"  Detections: {result.detections_count}")
        print(f"  Verified: {len(result.verified_detections)}")
        print(f"  Temporal votes: {result.temporal_votes}")
        print(f"  Alert: {result.alert}")
        print(f"  Confidence: {result.confidence:.3f}")
        print(f"  Reason: {result.reason}")
        print()
    
    # Show final statistics
    stats = sai.get_statistics()
    print("📊 Final Statistics:")
    print(f"  Frames processed: {stats['total_frames_processed']}")
    print(f"  Total detections: {stats['total_detections']}")
    print(f"  Confirmed alerts: {stats['confirmed_alerts']}")
    print(f"  Average inference time: {stats['avg_inference_time_ms']:.1f}ms")
    
    if 'alert_rate' in stats:
        print(f"  Alert rate: {stats['alert_rate']:.1%}")


if __name__ == "__main__":
    main()