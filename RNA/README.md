# SAI Neural Network Architecture (RNA)

**Early Fire Detection with Cascade Inference Pipeline**

SAI RNA implements a production-ready cascade architecture for early smoke and fire detection using distributed camera systems with temporal analysis.

## 🏗️ Architecture Overview

### Cascade Pipeline
```
📸 Camera Feed (2880×1616) 
    ↓ [Scale to 1440×808]
🎯 YOLOv8-s Detector → ROI Detection
    ↓ [Extract 224×224 ROIs]
🧠 SmokeyNet-Lite Verifier → Temporal Analysis (3 frames)
    ↓ [Persistence Logic]
🚨 Alarm Decision → Final Alert
```

### Components
- **YOLOv8-s Detector**: Fast ROI detection (smoke/fire bounding boxes)
- **SmokeyNet-Lite Verifier**: Temporal consistency verification using LSTM
- **Temporal Buffer**: Multi-frame sequence management per camera
- **Persistence Tracker**: False positive reduction with alarm logic

## 🚀 Quick Start Training

### Prerequisites Check
```bash
# Verify system readiness
python3 check_training_readiness.py
```

### Start Training (Autonomous 15-20 hours)
```bash
# Launch autonomous training
./start_detector_training.sh

# Monitor progress (optional)
tail -f RNA/training/logs/detector_training.log
```

### Manual Training Steps
```bash
# Setup environment
source RNA/training/venv/bin/activate

# Train YOLOv8-s detector
python RNA/training/detector_trainer.py --config RNA/configs/sai_cascade_config.yaml

# Train SmokeyNet-Lite verifier (after detector)
python RNA/training/verifier_trainer.py --config RNA/configs/sai_cascade_config.yaml
```

## 📊 Performance Targets

| Metric | Target | Hardware | Resolution |
|--------|--------|----------|------------|
| **Inference Speed** | 6-10 FPS | RTX 3090 | 1440×808 |
| **Detection Recall** | ≥85% | Cloud | Early smoke |
| **False Positive Rate** | ≤20% | Cloud | Temporal verification |
| **Time to Detection** | ≤3 min | Cloud | From ignition |
| **Model Size** | ~60-75 MB | Both | Combined pipeline |

## 🔧 Configuration

### Training Configuration (`RNA/configs/sai_cascade_config.yaml`)
- **Resolution**: 1440×808 (native camera scale)
- **Batch Size**: 8 (RTX 3090 optimized)
- **Epochs**: 100 (early stopping enabled)
- **Mixed Precision**: FP16 for performance
- **Checkpoint**: Auto-save every 10 epochs

### Cascade Configuration
- **Temporal Frames**: 3 (5-second intervals)
- **Persistence Window**: 30 seconds
- **Buffer Retention**: 60 seconds
- **ROI Expansion**: 10% around detections

## 📁 Project Structure

```
RNA/
├── configs/                    # Training configurations
│   └── sai_cascade_config.yaml
├── data/                       # Dataset management
│   └── raw/                    # 173K images, 5 datasets
├── docs/                       # Documentation
│   ├── performance_estimates.md
│   ├── roadmap.md
│   └── modelo10.md
├── inference/                  # Inference pipeline
│   └── cascade_inference.py
├── models/                     # Model architectures
│   ├── detector/
│   │   └── yolov8s_detector.py
│   └── verifier/
│       └── smokeynet_lite.py
├── training/                   # Training infrastructure
│   ├── venv/                   # Virtual environment
│   ├── detector_trainer.py    # YOLOv8-s training
│   └── logs/                   # Training logs
└── weights/                    # Model weights (post-training)
```

## 📦 Datasets (Ready for Training)

| Dataset | Images | Size | Format | Status |
|---------|--------|------|--------|--------|
| **FASDD** | 95,314 | 11.4GB | COCO JSON | ✅ Ready |
| **PyroNear-2024** | 33,600 | 3.1GB | HuggingFace | ✅ Ready |
| **D-Fire** | 21,527 | 3.0GB | YOLO | ✅ Ready |
| **FIgLib** | 19,317 | 277MB | HuggingFace | ✅ Ready |
| **NEMO** | 3,493 | 1.42GB | COCO JSON | ✅ Ready |
| **Total** | **173,251** | **~27GB** | Mixed | **✅ Complete** |

## 🔄 Temporal Workflow

### Distributed Camera System
```python
# Each camera sends photos every 5 seconds
camera_schedule = {
    "camera_1": "T+0s, T+5s, T+10s, T+15s...",
    "camera_2": "T+2s, T+7s, T+12s, T+17s...",
    "camera_N": "T+xs, T+x+5s, T+x+10s..."
}

# Server maintains independent buffers
server_buffers = {
    "camera_1": TemporalBuffer(max_length=5, retention_time=60.0),
    "camera_2": TemporalBuffer(max_length=5, retention_time=60.0),
    # ... per camera
}
```

### Per-Photo Processing
1. **Photo Reception**: New image from any camera
2. **Buffer Update**: Add to camera-specific temporal buffer
3. **Detector Stage**: YOLOv8-s processes current photo only
4. **Verifier Stage**: SmokeyNet-Lite analyzes 3-frame ROI sequence
5. **Persistence Logic**: Track detections over time for alarm decision

## 🧪 Inference API

### Basic Usage
```python
from RNA.inference.cascade_inference import CascadeInference

# Initialize pipeline
cascade = CascadeInference(
    detector_path="RNA/weights/detector_best.pt",
    verifier_path="RNA/weights/verifier_best.pt",
    device="cuda"
)

# Process single frame
result = cascade.process_frame(
    image=camera_image,
    camera_id="camera_1",
    timestamp=time.time()
)

# Check for alarms
if result['alarms']:
    print(f"🚨 Fire detected! Confidence: {result['alarms'][0]['persistence_score']}")
```

### Advanced Configuration
```python
cascade = CascadeInference(
    detector_path="RNA/weights/detector_best.pt",
    verifier_path="RNA/weights/verifier_best.pt",
    device="cuda",
    conf_threshold=0.3,
    temporal_frames=3,
    min_persistence_frames=2,
    persistence_time_window=30.0
)
```

## 📈 Training Progress

### Current Status: 🚀 **Ready to Start Training**

- ✅ **Architecture Implemented**: Complete cascade pipeline
- ✅ **Datasets Prepared**: 173K images from 5 major datasets  
- ✅ **Training Infrastructure**: Autonomous pipeline with monitoring
- ✅ **Configuration Optimized**: RTX 3090, 1440×808 resolution
- 🚀 **Ready for Training**: Run `./start_detector_training.sh`

### Next Steps
1. **Start Training**: Execute autonomous 15-20 hour detector training
2. **Verifier Training**: SmokeyNet-Lite temporal model (2-3 hours)
3. **Integration Testing**: Complete cascade pipeline validation
4. **Performance Benchmarking**: Real-world accuracy testing
5. **Production Deployment**: TensorRT optimization and API setup

## 🔧 System Requirements

### Training Environment
- **GPU**: RTX 3090 (24GB VRAM) or equivalent
- **RAM**: 32GB+ recommended  
- **Storage**: 30GB+ for datasets and checkpoints
- **CUDA**: 11.8+ with cuDNN support

### Dependencies
- **PyTorch**: 2.8.0+ with CUDA support
- **Ultralytics**: 8.3.181+ (YOLOv8 framework)
- **OpenCV**: 4.6.0+ for image processing
- **NumPy**: 1.23.0+ for numerical operations

## 📄 Documentation

- **[Performance Estimates](docs/performance_estimates.md)**: Detailed benchmarks and timing
- **[Implementation Roadmap](docs/roadmap.md)**: Project progress and next steps  
- **[Architecture Analysis](docs/modelo10.md)**: Technical design decisions

## 🎯 Production Deployment

### Model Outputs (Post-Training)
- **Detector**: `RNA/weights/detector_best.pt` (~25-30 MB)
- **Verifier**: `RNA/weights/verifier_best.pt` (~35-45 MB)
- **Total Pipeline**: ~60-75 MB for complete system

### Optimization Features
- **TensorRT**: GPU inference acceleration
- **ONNX**: Cross-platform deployment
- **Half Precision**: Memory and speed optimization
- **Batch Processing**: Multi-camera efficiency

---

**SAI RNA** - Production-ready early fire detection with temporal intelligence and distributed camera support.

*Last updated: 2025-08-20*  
*Status: Training pipeline ready for execution*