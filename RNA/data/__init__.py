"""
Data Processing Pipeline for SAI

This module handles data loading, preprocessing, and augmentation for:
- FIgLib dataset (temporal sequences)
- FASDD dataset (bounding box annotations)
- D-Fire dataset (fire detection)
- Nemo dataset (smoke detection)
- PyroNear-2024 dataset (geographical diversity)
"""

from .dataset_loaders import (
    FIgLibDataset,
    FASSDDataset,
    SmokeDetectionDataset,
    TemporalSequenceDataset
)
from .augmentations import (
    SmokeAugmentations,
    TemporalAugmentations,
    get_training_transforms,
    get_validation_transforms
)
from .preprocessing import (
    ROIExtractor,
    SequenceProcessor,
    BoundingBoxProcessor
)

__all__ = [
    'FIgLibDataset',
    'FASSDDataset', 
    'SmokeDetectionDataset',
    'TemporalSequenceDataset',
    'SmokeAugmentations',
    'TemporalAugmentations',
    'get_training_transforms',
    'get_validation_transforms',
    'ROIExtractor',
    'SequenceProcessor',
    'BoundingBoxProcessor'
]