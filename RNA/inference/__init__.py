"""
Inference Module for SAI Neural Networks

This module provides inference pipelines for:
- Cascade inference (Detector + Temporal Verifier)
- Real-time processing for SAI-CAM integration
- Server API for production deployment
"""

from .cascade_inference import CascadeInference
from .server_api import SAIServerAPI

__all__ = [
    'CascadeInference',
    'SAIServerAPI'
]