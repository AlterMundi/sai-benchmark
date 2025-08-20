"""
YOLOv8-s Detector Module

High-recall smoke detection using YOLOv8-s architecture optimized for early fire detection.
Designed to work with low confidence thresholds (0.3-0.4) to maximize recall.
"""

from .yolov8s_detector import YOLOv8sDetector
from .yolo_head import YOLOv8DetectionHead
from .yolo_backbone import YOLOv8Backbone

__all__ = [
    'YOLOv8sDetector',
    'YOLOv8DetectionHead', 
    'YOLOv8Backbone'
]