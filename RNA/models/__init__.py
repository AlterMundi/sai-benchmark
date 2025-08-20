"""
SAI Neural Network Models

This module contains the implementation of neural network architectures for the SAI project:
- YOLOv8-s detector for high-recall smoke detection
- SmokeyNet-Lite temporal verifier for false positive reduction
- Integrated hybrid model (future implementation)
"""

from .detector.yolov8s_detector import YOLOv8sDetector
from .verifier.smokeynet_lite import SmokeyNetLite

__all__ = [
    'YOLOv8sDetector',
    'SmokeyNetLite'
]