# SAI Neural Network Architectures (RNA)

This directory contains the implementation of neural network architectures for the SAI (Sistema de Alerta de Incendios) project.

## Architecture Overview

Based on the comprehensive evaluation in `modelo10.md`, we implement a two-phase approach:

### Phase 1: Cascade Architecture (Current Implementation)
- **Detector**: YOLOv8-s with high recall (low threshold 0.3-0.4)
- **Temporal Verifier**: SmokeyNet-Lite (EfficientNet-B0 + LSTM) over ROIs
- **Temporal Logic**: Persistence ≥2-3 frames for alarm confirmation

### Phase 2: Integrated Hybrid Model (Future)
- **Backbone**: ConvNeXt-T or EfficientNet-B4/B5
- **Temporal Module**: Bidirectional LSTM (3-5 frames)
- **Detection Head**: YOLOv8 anchor-free with spatiotemporal fusion

## Directory Structure

```
RNA/
├── README.md                    # This file
├── modelo10.md                  # Architecture evaluation document
├── models/                      # Model implementations
│   ├── __init__.py
│   ├── detector/               # YOLOv8-s detector implementation
│   ├── verifier/               # SmokeyNet-Lite temporal verifier
│   └── integrated/             # Future integrated model
├── training/                   # Training pipelines
│   ├── __init__.py
│   ├── detector_trainer.py
│   ├── verifier_trainer.py
│   └── cascade_trainer.py
├── data/                       # Data processing pipelines
│   ├── __init__.py
│   ├── dataset_loaders.py
│   ├── augmentations.py
│   └── preprocessing.py
├── evaluation/                 # Evaluation and metrics
│   ├── __init__.py
│   ├── cascade_evaluator.py
│   └── metrics.py
└── inference/                  # Inference pipelines
    ├── __init__.py
    ├── cascade_inference.py
    └── server_api.py
```

## Implementation Plan

### Etapa A: YOLOv8-s Detector
- [ ] Implement YOLOv8-s architecture
- [ ] Setup training pipeline with FASDD/D-Fire/Nemo datasets
- [ ] Optimize for high recall (low threshold)
- [ ] Integration with SAI-Benchmark

### Etapa B: SmokeyNet-Lite Temporal Verifier  
- [ ] Implement EfficientNet-B0 + LSTM architecture
- [ ] Setup training with FIgLib temporal sequences
- [ ] ROI-based processing pipeline
- [ ] Temporal persistence logic

### Etapa C: Integration and Fine-tuning
- [ ] Cascade inference pipeline
- [ ] Multi-task learning framework
- [ ] Threshold calibration system
- [ ] Performance optimization for RTX 3090

## Key Features

- **High Recall Detection**: YOLOv8-s optimized for early smoke detection
- **Temporal Verification**: LSTM-based verification to reduce false positives
- **Modular Design**: Separate training and inference for each component
- **SAI-Benchmark Integration**: Reproducible evaluation framework
- **GPU Optimization**: Designed for RTX 3090 deployment

## Dependencies

- PyTorch 2.7+
- Ultralytics YOLOv8
- OpenCV
- NumPy
- Albumentations (for augmentations)
- TensorBoard (for monitoring)

## Getting Started

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Prepare datasets (see `data/README.md`)

3. Train detector:
```bash
python training/detector_trainer.py --config configs/yolov8s_detector.yaml
```

4. Train temporal verifier:
```bash
python training/verifier_trainer.py --config configs/smokeynet_lite.yaml
```

5. Evaluate cascade:
```bash
python evaluation/cascade_evaluator.py --detector_weights detector.pt --verifier_weights verifier.pt
```

## References

- See `modelo10.md` for complete architecture evaluation and references
- SmokeyNet: Dewangan et al. (2022) - FIgLib & SmokeyNet for wildfire detection
- YOLOv8: Ultralytics anchor-free detection architecture