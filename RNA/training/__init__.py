"""
Training Module for SAI Neural Networks

This module contains training pipelines for:
- YOLOv8-s detector training (Etapa A)
- SmokeyNet-Lite verifier training (Etapa B) 
- Cascade integration and fine-tuning (Etapa C)
"""

from .detector_trainer import DetectorTrainer
from .verifier_trainer import VerifierTrainer
from .cascade_trainer import CascadeTrainer

__all__ = [
    'DetectorTrainer',
    'VerifierTrainer', 
    'CascadeTrainer'
]