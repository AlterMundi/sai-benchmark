"""
Cascade Inference Pipeline

Implements the full cascade inference pipeline:
1. YOLOv8-s detector for ROI extraction  
2. SmokeyNet-Lite temporal verifier for false positive reduction
3. Temporal persistence logic for final alarm decision
"""

import torch
import torch.nn.functional as F
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np
import cv2
from pathlib import Path
import time
from collections import deque
import logging

from ..models.detector.yolov8s_detector import YOLOv8sDetector
from ..models.verifier.smokeynet_lite import SmokeyNetLite

logger = logging.getLogger(__name__)


class TemporalBuffer:
    """Buffer for maintaining temporal context across frames"""
    
    def __init__(self, max_length: int = 5, retention_time: float = 60.0):
        self.max_length = max_length
        self.retention_time = retention_time
        self.buffer = deque(maxlen=max_length)
        self.timestamps = deque(maxlen=max_length)
    
    def add_frame(self, frame_data: Dict[str, Any], timestamp: Optional[float] = None):
        """Add frame data to buffer"""
        if timestamp is None:
            timestamp = time.time()
        
        self.buffer.append(frame_data)
        self.timestamps.append(timestamp)
        
        # Clean old frames
        self._clean_old_frames()
    
    def _clean_old_frames(self):
        """Remove frames older than retention time"""
        current_time = time.time()
        while (self.timestamps and 
               current_time - self.timestamps[0] > self.retention_time):
            self.buffer.popleft()
            self.timestamps.popleft()
    
    def get_recent_frames(self, count: int = None) -> List[Dict[str, Any]]:
        """Get recent frames from buffer"""
        if count is None:
            return list(self.buffer)
        else:
            return list(self.buffer)[-count:]
    
    def __len__(self):
        return len(self.buffer)


class PersistenceTracker:
    """Track detection persistence for alarm logic"""
    
    def __init__(
        self,
        min_frames: int = 2,
        time_window: float = 30.0,
        decay_factor: float = 0.8
    ):
        self.min_frames = min_frames
        self.time_window = time_window
        self.decay_factor = decay_factor
        
        # Track detections per ROI/region
        self.region_trackers = {}
    
    def update(
        self,
        detections: List[Dict[str, Any]],
        timestamp: Optional[float] = None
    ) -> Dict[str, Any]:
        """Update persistence tracking with new detections"""
        if timestamp is None:
            timestamp = time.time()
        
        # Clean old trackers
        self._clean_old_trackers(timestamp)
        
        # Update region trackers
        confirmed_alarms = []
        
        for detection in detections:
            region_id = self._get_region_id(detection)
            
            if region_id not in self.region_trackers:
                self.region_trackers[region_id] = {
                    'detections': [],
                    'confidence_history': [],
                    'last_seen': timestamp
                }
            
            tracker = self.region_trackers[region_id]
            tracker['detections'].append({
                'detection': detection,
                'timestamp': timestamp
            })
            tracker['confidence_history'].append(detection.get('confidence', 0.0))
            tracker['last_seen'] = timestamp
            
            # Check if alarm should be triggered
            if self._should_trigger_alarm(tracker, timestamp):
                confirmed_alarms.append({
                    'region_id': region_id,
                    'detection': detection,
                    'persistence_score': self._calculate_persistence_score(tracker),
                    'frame_count': len(tracker['detections'])
                })
        
        return {
            'confirmed_alarms': confirmed_alarms,
            'active_regions': len(self.region_trackers),
            'total_detections': sum(len(t['detections']) for t in self.region_trackers.values())
        }
    
    def _get_region_id(self, detection: Dict[str, Any]) -> str:
        """Generate region ID based on detection location"""
        bbox = detection.get('bbox', [0, 0, 0, 0])
        if len(bbox) >= 4:
            # Use center point for region identification
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2
            
            # Quantize to grid for region grouping
            grid_size = 0.1  # 10% of image
            grid_x = int(center_x / grid_size)
            grid_y = int(center_y / grid_size)
            
            return f"region_{grid_x}_{grid_y}"
        else:
            return "region_unknown"
    
    def _should_trigger_alarm(
        self,
        tracker: Dict[str, Any],
        current_time: float
    ) -> bool:
        """Check if alarm should be triggered for this region"""
        detections = tracker['detections']
        
        # Check minimum frame count
        if len(detections) < self.min_frames:
            return False
        
        # Check time window
        recent_detections = [
            d for d in detections 
            if current_time - d['timestamp'] <= self.time_window
        ]
        
        return len(recent_detections) >= self.min_frames
    
    def _calculate_persistence_score(self, tracker: Dict[str, Any]) -> float:
        """Calculate persistence score for alarm"""
        confidence_history = tracker['confidence_history']
        
        if not confidence_history:
            return 0.0
        
        # Weighted average with recent frames having higher weight
        weights = [self.decay_factor ** (len(confidence_history) - i - 1) 
                  for i in range(len(confidence_history))]
        
        weighted_sum = sum(w * c for w, c in zip(weights, confidence_history))
        weight_sum = sum(weights)
        
        return weighted_sum / weight_sum if weight_sum > 0 else 0.0
    
    def _clean_old_trackers(self, current_time: float):
        """Remove old region trackers"""
        expired_regions = []
        
        for region_id, tracker in self.region_trackers.items():
            if current_time - tracker['last_seen'] > self.time_window * 2:
                expired_regions.append(region_id)
        
        for region_id in expired_regions:
            del self.region_trackers[region_id]


class CascadeInference:
    """
    Main cascade inference pipeline for SAI
    
    Combines YOLOv8-s detector with SmokeyNet-Lite temporal verifier
    for robust smoke detection with low false positive rate.
    """
    
    def __init__(
        self,
        detector_path: str,
        verifier_path: str,
        device: str = 'auto',
        conf_threshold: float = 0.3,
        temporal_frames: int = 3,
        min_persistence_frames: int = 2,
        persistence_time_window: float = 30.0
    ):
        self.device = self._setup_device(device)
        self.conf_threshold = conf_threshold
        self.temporal_frames = temporal_frames
        
        # Load models
        self.detector = self._load_detector(detector_path)
        self.verifier = self._load_verifier(verifier_path)
        
        # Initialize temporal components
        self.temporal_buffer = TemporalBuffer(max_length=temporal_frames)
        self.persistence_tracker = PersistenceTracker(
            min_frames=min_persistence_frames,
            time_window=persistence_time_window
        )
        
        # Performance tracking
        self.stats = {
            'total_frames': 0,
            'total_detections': 0,
            'verified_detections': 0,
            'confirmed_alarms': 0,
            'avg_inference_time': 0.0
        }
    
    def _setup_device(self, device: str) -> torch.device:
        """Setup computation device"""
        if device == 'auto':
            if torch.cuda.is_available():
                device = 'cuda'
            else:
                device = 'cpu'
        
        return torch.device(device)
    
    def _load_detector(self, model_path: str) -> YOLOv8sDetector:
        """Load YOLOv8-s detector"""
        try:
            detector = YOLOv8sDetector(
                conf_threshold=self.conf_threshold,
                model_path=model_path
            )
            detector.to(self.device)
            detector.eval()
            
            logger.info(f"Loaded detector from {model_path}")
            return detector
        except Exception as e:
            logger.error(f"Failed to load detector: {e}")
            # Fallback to default detector
            detector = YOLOv8sDetector(conf_threshold=self.conf_threshold)
            detector.to(self.device)
            return detector
    
    def _load_verifier(self, model_path: str) -> SmokeyNetLite:
        """Load SmokeyNet-Lite verifier"""
        try:
            verifier = SmokeyNetLite(sequence_length=self.temporal_frames)
            
            if Path(model_path).exists():
                state_dict = torch.load(model_path, map_location=self.device)
                verifier.load_state_dict(state_dict)
                logger.info(f"Loaded verifier from {model_path}")
            else:
                logger.warning(f"Verifier weights not found at {model_path}, using random weights")
            
            verifier.to(self.device)
            verifier.eval()
            
            return verifier
        except Exception as e:
            logger.error(f"Failed to load verifier: {e}")
            # Fallback to default verifier
            verifier = SmokeyNetLite(sequence_length=self.temporal_frames)
            verifier.to(self.device)
            return verifier
    
    def process_frame(
        self,
        image: Union[np.ndarray, torch.Tensor],
        timestamp: Optional[float] = None,
        camera_id: str = "default"
    ) -> Dict[str, Any]:
        """
        Process single frame through cascade pipeline
        
        Args:
            image: Input image (H, W, C) or (C, H, W)
            timestamp: Frame timestamp
            camera_id: Camera identifier
            
        Returns:
            Processing results with detections and alarms
        """
        start_time = time.time()
        
        if timestamp is None:
            timestamp = start_time
        
        # Preprocess image
        processed_image = self._preprocess_image(image)
        
        # Stage 1: Detection
        detections = self._run_detector(processed_image)
        
        # Stage 2: Temporal verification (if we have enough frames)
        verified_detections = []
        if len(self.temporal_buffer) >= self.temporal_frames - 1:
            verified_detections = self._run_temporal_verification(
                processed_image, detections
            )
        
        # Update temporal buffer
        frame_data = {
            'image': processed_image,
            'detections': detections,
            'verified_detections': verified_detections,
            'camera_id': camera_id,
            'timestamp': timestamp
        }
        self.temporal_buffer.add_frame(frame_data, timestamp)
        
        # Stage 3: Persistence tracking and alarm logic
        alarm_result = self.persistence_tracker.update(verified_detections, timestamp)
        
        # Update statistics
        inference_time = time.time() - start_time
        self._update_stats(detections, verified_detections, alarm_result, inference_time)
        
        return {
            'detections': detections,
            'verified_detections': verified_detections,
            'alarms': alarm_result['confirmed_alarms'],
            'metadata': {
                'camera_id': camera_id,
                'timestamp': timestamp,
                'inference_time_ms': inference_time * 1000,
                'active_regions': alarm_result['active_regions'],
                'buffer_size': len(self.temporal_buffer)
            }
        }
    
    def _preprocess_image(self, image: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Preprocess image for model input"""
        if isinstance(image, np.ndarray):
            # Convert numpy to tensor
            if image.ndim == 3 and image.shape[-1] == 3:  # (H, W, C)
                image = torch.from_numpy(image).permute(2, 0, 1)  # (C, H, W)
            else:
                image = torch.from_numpy(image)
        
        # Ensure float and normalize
        if image.dtype != torch.float32:
            image = image.float() / 255.0
        
        # Add batch dimension if needed
        if image.dim() == 3:
            image = image.unsqueeze(0)  # (1, C, H, W)
        
        return image.to(self.device)
    
    def _run_detector(self, image: torch.Tensor) -> List[Dict[str, Any]]:
        """Run YOLOv8-s detector"""
        try:
            with torch.no_grad():
                detections = self.detector.detect(image)
            
            # Filter and format detections
            filtered_detections = []
            for detection in detections:
                for box in detection['boxes']:
                    if box['confidence'] >= self.conf_threshold:
                        filtered_detections.append({
                            'bbox': box['bbox'],
                            'confidence': box['confidence'],
                            'class_id': box['class_id'],
                            'class_name': box['class_name']
                        })
            
            return filtered_detections
        except Exception as e:
            logger.error(f"Detector error: {e}")
            return []
    
    def _run_temporal_verification(
        self,
        current_image: torch.Tensor,
        detections: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Run temporal verification on detected ROIs"""
        if not detections:
            return []
        
        verified_detections = []
        recent_frames = self.temporal_buffer.get_recent_frames(self.temporal_frames - 1)
        
        try:
            # Extract ROI sequences for each detection
            for detection in detections:
                roi_sequence = self._extract_roi_sequence(
                    current_image, detection, recent_frames
                )
                
                if roi_sequence is not None:
                    # Run temporal verifier
                    verification_result = self._verify_roi_sequence(roi_sequence)
                    
                    if verification_result['verified']:
                        # Add verification metadata to detection
                        verified_detection = detection.copy()
                        verified_detection.update({
                            'temporal_confidence': verification_result['smoke_probability'],
                            'verification_score': verification_result['confidence'],
                            'temporal_verified': True
                        })
                        verified_detections.append(verified_detection)
        
        except Exception as e:
            logger.error(f"Temporal verification error: {e}")
            # Fallback: return original detections
            return detections
        
        return verified_detections
    
    def _extract_roi_sequence(
        self,
        current_image: torch.Tensor,
        detection: Dict[str, Any],
        recent_frames: List[Dict[str, Any]]
    ) -> Optional[torch.Tensor]:
        """Extract ROI sequence for temporal verification"""
        try:
            sequence_rois = []
            bbox = detection['bbox']
            
            # Extract ROI from recent frames
            for frame_data in recent_frames:
                roi = self._extract_roi_from_image(frame_data['image'], bbox)
                if roi is not None:
                    sequence_rois.append(roi)
            
            # Extract ROI from current frame
            current_roi = self._extract_roi_from_image(current_image, bbox)
            if current_roi is not None:
                sequence_rois.append(current_roi)
            
            # Ensure we have enough ROIs
            if len(sequence_rois) < self.temporal_frames:
                # Pad with last available ROI
                while len(sequence_rois) < self.temporal_frames:
                    if sequence_rois:
                        sequence_rois.append(sequence_rois[-1])
                    else:
                        return None
            
            # Take last N frames
            sequence_rois = sequence_rois[-self.temporal_frames:]
            
            # Stack into sequence tensor
            roi_sequence = torch.stack(sequence_rois).unsqueeze(0)  # (1, T, C, H, W)
            
            return roi_sequence
        
        except Exception as e:
            logger.error(f"ROI extraction error: {e}")
            return None
    
    def _extract_roi_from_image(
        self,
        image: torch.Tensor,
        bbox: List[float],
        target_size: Tuple[int, int] = (224, 224)
    ) -> Optional[torch.Tensor]:
        """Extract ROI from image given bounding box"""
        try:
            if len(bbox) < 4:
                return None
            
            # Get image dimensions
            _, h, w = image.shape[-3:]
            
            # Convert normalized coordinates to pixel coordinates
            x1, y1, x2, y2 = bbox
            if max(bbox) <= 1.0:  # Normalized coordinates
                x1, x2 = int(x1 * w), int(x2 * w)
                y1, y2 = int(y1 * h), int(y2 * h)
            else:  # Pixel coordinates
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Clamp coordinates
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            if x2 <= x1 or y2 <= y1:
                return None
            
            # Extract ROI
            if image.dim() == 4:  # Batch dimension
                roi = image[0, :, y1:y2, x1:x2]
            else:  # No batch dimension
                roi = image[:, y1:y2, x1:x2]
            
            # Resize to target size
            roi = F.interpolate(
                roi.unsqueeze(0),
                size=target_size,
                mode='bilinear',
                align_corners=False
            ).squeeze(0)
            
            return roi
        
        except Exception as e:
            logger.error(f"ROI extraction error: {e}")
            return None
    
    def _verify_roi_sequence(self, roi_sequence: torch.Tensor) -> Dict[str, Any]:
        """Verify ROI sequence using temporal model"""
        try:
            with torch.no_grad():
                result = self.verifier.predict(roi_sequence)
            
            return {
                'verified': bool(result['predictions'][0]),
                'smoke_probability': float(result['smoke_probability'][0]),
                'confidence': float(result['confidence'][0])
            }
        
        except Exception as e:
            logger.error(f"Verification error: {e}")
            return {
                'verified': False,
                'smoke_probability': 0.0,
                'confidence': 0.0
            }
    
    def _update_stats(
        self,
        detections: List[Dict[str, Any]],
        verified_detections: List[Dict[str, Any]],
        alarm_result: Dict[str, Any],
        inference_time: float
    ):
        """Update performance statistics"""
        self.stats['total_frames'] += 1
        self.stats['total_detections'] += len(detections)
        self.stats['verified_detections'] += len(verified_detections)
        self.stats['confirmed_alarms'] += len(alarm_result['confirmed_alarms'])
        
        # Update average inference time
        alpha = 0.1  # Exponential moving average factor
        self.stats['avg_inference_time'] = (
            alpha * inference_time + 
            (1 - alpha) * self.stats['avg_inference_time']
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get pipeline performance statistics"""
        stats = self.stats.copy()
        
        if stats['total_frames'] > 0:
            stats['detection_rate'] = stats['total_detections'] / stats['total_frames']
            stats['verification_rate'] = (
                stats['verified_detections'] / max(1, stats['total_detections'])
            )
            stats['alarm_rate'] = stats['confirmed_alarms'] / stats['total_frames']
        
        return stats
    
    def reset_statistics(self):
        """Reset performance statistics"""
        self.stats = {
            'total_frames': 0,
            'total_detections': 0,
            'verified_detections': 0,
            'confirmed_alarms': 0,
            'avg_inference_time': 0.0
        }


def create_cascade_inference(
    detector_path: str,
    verifier_path: str,
    device: str = 'auto',
    **kwargs
) -> CascadeInference:
    """
    Factory function to create cascade inference pipeline
    
    Args:
        detector_path: Path to detector weights
        verifier_path: Path to verifier weights
        device: Computation device
        **kwargs: Additional configuration
        
    Returns:
        Configured cascade inference pipeline
    """
    return CascadeInference(
        detector_path=detector_path,
        verifier_path=verifier_path,
        device=device,
        **kwargs
    )


if __name__ == "__main__":
    # Test cascade inference
    cascade = create_cascade_inference(
        detector_path="detector.pt",
        verifier_path="verifier.pt",
        device='cpu'
    )
    
    # Test with dummy image
    dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    result = cascade.process_frame(dummy_image)
    
    print(f"Cascade inference test completed")
    print(f"Detections: {len(result['detections'])}")
    print(f"Verified: {len(result['verified_detections'])}")
    print(f"Alarms: {len(result['alarms'])}")
    print(f"Stats: {cascade.get_statistics()}")