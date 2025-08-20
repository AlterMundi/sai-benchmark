# SAI Performance Estimates and Benchmarks

This document provides detailed performance estimates for SAI neural network models across different resolutions, hardware configurations, and deployment scenarios.

## Executive Summary

| Configuration | Resolution | Hardware | Training Time | Inference Time | Throughput | Status |
|---------------|------------|----------|---------------|----------------|------------|--------|
| **Cloud MVP** | 1440×808 | RTX 3090 | **~15-20 hours** | ~100-150ms | 6-10 FPS | **🚀 Ready** |
| **Cloud Optimized** | 960×539 | RTX 3090 | **~10-14 hours** | ~60-80ms | 12-16 FPS | Available |
| **Edge Standard** | 480×270 | RPi 4B | N/A | ~2-5s | 0.2-0.5 FPS | Future |
| **Edge Lite** | 360×202 | RPi 4B | N/A | ~1-3s | 0.3-1 FPS | Future |

## Cloud Computing Performance (RTX 3090)

### YOLOv8-s Detector Training

| Resolution | Batch Size | Epochs | Est. Training Time | GPU Memory | Dataset Size |
|------------|------------|--------|-------------------|------------|--------------|
| **1440×808** | 8 | 100 | **15-20 hours** | ~20GB | **173K images** |
| 960×539 | 16 | 100 | 10-14 hours | ~16GB | 173K images |
| 720×404 | 24 | 100 | 8-12 hours | ~12GB | 173K images |
| 640×640 | 16 | 100 | 10-14 hours | ~14GB | 173K images |

**Assumptions:**
- Dataset: **173K images combined** (FASDD 95K + PyroNear 34K + D-Fire 22K + FIgLib 19K + NEMO 3K)
- RTX 3090: 24GB VRAM, 328 Tensor TFLOPS
- Mixed precision training (FP16)
- Data augmentation pipeline active

### SmokeyNet-Lite Verifier Training

| Input Resolution | Sequence Length | Batch Size | Epochs | Est. Training Time | GPU Memory |
|------------------|-----------------|------------|--------|-------------------|------------|
| **224×224** | 3 frames | 16 | 50 | **2-3 hours** | ~8GB |
| 224×224 | 5 frames | 12 | 50 | 3-4 hours | ~10GB |
| 320×180 | 3 frames | 12 | 50 | 3-4 hours | ~12GB |

**Assumptions:**
- Dataset: FIgLib sequences + synthetic temporal data (~10K sequences)
- EfficientNet-B0 backbone + LSTM processing
- Temporal sequence processing overhead

### Inference Performance (RTX 3090)

#### YOLOv8-s Detector Inference

| Resolution | Batch Size | Inference Time | Throughput | GPU Memory | Use Case |
|------------|------------|----------------|------------|------------|----------|
| **1440×808** | 1 | **80-120ms** | **8-12 FPS** | ~4GB | **Production MVP** |
| 1440×808 | 4 | 60-80ms/img | 12-16 FPS | ~12GB | Batch processing |
| 960×539 | 1 | 40-60ms | 16-25 FPS | ~2GB | Fast response |
| 720×404 | 1 | 25-35ms | 28-40 FPS | ~1.5GB | Real-time |
| 2880×1616 | 1 | 300-500ms | 2-3 FPS | ~16GB | Full resolution |

#### SmokeyNet-Lite Verifier Inference

| ROI Resolution | Sequence Length | Inference Time | Throughput | GPU Memory |
|----------------|-----------------|----------------|------------|------------|
| **224×224** | **3 frames** | **15-25ms** | **40-60 FPS** | ~1GB |
| 224×224 | 5 frames | 20-30ms | 30-50 FPS | ~1.5GB |
| 320×180 | 3 frames | 25-35ms | 28-40 FPS | ~1.5GB |

#### Cascade Pipeline (Combined)

| Configuration | Total Latency | End-to-End FPS | Comments |
|---------------|---------------|----------------|----------|
| **1440×808 + 224×224 ROI** | **100-150ms** | **6-10 FPS** | **Recommended MVP** |
| 960×539 + 224×224 ROI | 60-90ms | 11-16 FPS | Balanced performance |
| 720×404 + 224×224 ROI | 50-70ms | 14-20 FPS | Fast response |

**Pipeline Breakdown:**
- Detector: 70-80% of total time
- ROI extraction: ~5ms
- Verifier: 15-20% of total time
- Post-processing: ~5ms

## Edge Computing Performance (Raspberry Pi)

### Raspberry Pi 4B (4GB RAM, ARM Cortex-A72)

#### MobileNetV3-Small Classifier

| Resolution | Quantization | Inference Time | Throughput | RAM Usage | Power |
|------------|--------------|----------------|------------|-----------|-------|
| **480×270** | **INT8** | **2-4s** | **0.25-0.5 FPS** | ~1GB | ~3W |
| 360×202 | INT8 | 1-3s | 0.3-1 FPS | ~800MB | ~2.5W |
| 320×180 | INT8 | 1-2s | 0.5-1 FPS | ~600MB | ~2W |
| 240×135 | INT8 | 0.5-1s | 1-2 FPS | ~400MB | ~1.5W |

#### YOLOv8-Nano Detector (if needed)

| Resolution | Quantization | Inference Time | Throughput | RAM Usage | Power |
|------------|--------------|----------------|------------|-----------|-------|
| **360×202** | **INT8** | **5-8s** | **0.12-0.2 FPS** | ~1.5GB | ~4W |
| 320×180 | INT8 | 3-6s | 0.16-0.3 FPS | ~1.2GB | ~3.5W |
| 240×135 | INT8 | 2-4s | 0.25-0.5 FPS | ~800MB | ~3W |

**Edge Constraints:**
- CPU-only inference (no GPU acceleration)
- Limited to 4GB RAM total
- Thermal throttling under sustained load
- Network bandwidth limitations for uploading

## Memory Requirements

### Cloud (RTX 3090 - 24GB VRAM)

| Component | Resolution | Training Memory | Inference Memory | Notes |
|-----------|------------|-----------------|------------------|-------|
| YOLOv8-s | 1440×808 | 16-20GB | 4-6GB | With batch processing |
| SmokeyNet-Lite | 224×224 seq | 8-12GB | 1-2GB | Temporal sequences |
| **Combined Pipeline** | **Mixed** | **20-24GB** | **6-8GB** | **Safe operation** |

### Edge (Raspberry Pi 4B - 4GB RAM)

| Component | Resolution | Model Size | Runtime Memory | Available Buffer |
|-----------|------------|------------|-----------------|------------------|
| MobileNetV3 | 480×270 | ~5MB | 800MB-1GB | 2-3GB |
| YOLOv8-Nano | 360×202 | ~6MB | 1-1.5GB | 1.5-2GB |
| **System overhead** | - | - | **1GB** | **OS + services** |

## Dataset Preparation Times

| Dataset | Size | Format | Processing Time | Storage Required |
|---------|------|--------|-----------------|------------------|
| **FASDD** | **95K images** | COCO JSON | 4-8 hours | ~15GB |
| **PyroNear-2024** | **34K images** | HuggingFace | 2-4 hours | ~5GB |
| **D-Fire** | **22K images** | YOLO | 2-4 hours | ~4GB |
| **FIgLib** | **19K images** | HuggingFace | 2-4 hours | ~1GB |
| **NEMO** | **3K images** | COCO JSON | 1-2 hours | ~2GB |
| **Combined** | **~173K items** | **Mixed** | **12-20 hours** | **~27GB** |

**Processing includes:**
- Format conversion and validation
- Resolution scaling and cropping
- Augmentation pipeline setup
- Train/validation splits

## Production Deployment Estimates

### Cloud Server (RTX 3090)

| Metric | Conservative | Optimistic | Comments |
|--------|--------------|------------|----------|
| **Concurrent cameras** | **8-12** | **15-20** | 1440×808 resolution |
| **Detection latency** | 150ms | 100ms | 95th percentile |
| **Daily throughput** | 50M frames | 100M frames | 24/7 operation |
| **Storage per day** | 500GB | 1TB | Logs + results |

### Edge Network (Raspberry Pi fleet)

| Metric | Per Device | 100 Devices | Comments |
|--------|------------|-------------|----------|
| **Processing rate** | 0.3 FPS | 30 FPS total | Pre-filtering |
| **Upload reduction** | 80-90% | 8-9TB saved/day | vs raw stream |
| **Power consumption** | 3-5W | 300-500W total | Very efficient |
| **Maintenance** | 1 hour/month | 100 hours/month | Remote updates |

## Optimization Roadmap

### Phase 1: MVP (Current)
- **Target**: 1440×808 @ 6-10 FPS on RTX 3090
- **Timeline**: 2-3 weeks implementation + testing
- **Deliverable**: Working cascade pipeline

### Phase 2: Performance Optimization (Future)
- **TensorRT optimization**: +50% speed improvement
- **Model pruning**: -30% memory usage
- **Multi-GPU scaling**: 2-4x throughput

### Phase 3: Edge Deployment (Future)
- **Quantization**: INT8 models for Raspberry Pi
- **Federated learning**: Continuous improvement
- **5G integration**: Real-time streaming

## Validation Benchmarks

### Target Metrics (SAI Requirements)

| Metric | Cloud Target | Edge Target | Measurement |
|--------|--------------|-------------|-------------|
| **Recall** | ≥85% | ≥70% | Early smoke detection |
| **Precision** | ≥80% | ≥60% | False positive rate |
| **TTD (Time to Detect)** | ≤3 min | ≤5 min | From ignition |
| **Latency** | ≤200ms | ≤10s | End-to-end |
| **Availability** | 99.9% | 99% | Uptime target |

### Hardware Validation Tests

1. **Stress Test**: 24-hour continuous operation
2. **Thermal Test**: Performance under sustained load
3. **Memory Test**: Memory leaks and cleanup
4. **Network Test**: Bandwidth and reliability
5. **Power Test**: Consumption under various loads

## Cost Analysis

### Cloud Infrastructure (Monthly)

| Component | Specification | Cost (USD) | Notes |
|-----------|---------------|------------|-------|
| **Server** | RTX 3090 + 64GB RAM | $800-1200 | Amortized hardware |
| **Storage** | 2TB NVMe SSD | $50-100 | High-speed dataset storage |
| **Network** | 1Gbps upload | $100-200 | For multi-camera upload |
| **Power** | 500W average | $50-100 | 24/7 operation |
| **Total** | - | **$1000-1600** | **Per server** |

### Edge Deployment (Per Device)

| Component | Specification | Cost (USD) | Notes |
|-----------|---------------|------------|-------|
| **Raspberry Pi 4B** | 4GB RAM | $75 | One-time cost |
| **Storage** | 64GB microSD | $15 | Local processing |
| **Power** | 5V 3A supply | $10 | Continuous operation |
| **Case & cooling** | Protective housing | $20 | Outdoor deployment |
| **Total** | - | **$120** | **Per edge device** |

---

## Performance Testing Protocol

### Benchmark Suite
1. **Latency benchmarks**: Single image processing
2. **Throughput benchmarks**: Batch processing
3. **Memory benchmarks**: Peak and sustained usage
4. **Accuracy benchmarks**: Detection quality metrics
5. **Stress benchmarks**: Extended operation tests

### Testing Environment
- **Cloud**: Ubuntu 20.04, CUDA 11.8, Docker containers
- **Edge**: Raspberry Pi OS, ARM optimization
- **Network**: Realistic bandwidth and latency simulation

### Continuous Monitoring
- **Performance tracking**: Real-time metrics collection
- **Alert thresholds**: Automated performance degradation detection
- **Resource utilization**: GPU, CPU, memory, network monitoring

## Training Pipeline Implementation

### Autonomous Training System

The training pipeline is now **fully implemented and ready for execution**:

#### System Requirements Verified
- **GPU**: RTX 3090 (25.4GB VRAM) ✅
- **PyTorch**: 2.8.0+cu128 ✅  
- **Ultralytics**: 8.3.181 ✅
- **Datasets**: 173K images across 5 datasets ✅
- **Disk Space**: 2.4GB available ✅

#### Training Commands
```bash
# Quick readiness check
python3 check_training_readiness.py

# Start autonomous 15-20 hour training
./start_detector_training.sh

# Monitor progress
tail -f RNA/training/logs/detector_training.log
```

#### Key Features
- **Fully Autonomous**: No human intervention required for 15-20 hours
- **Early Stopping**: Automatic convergence detection with patience=50
- **Checkpoint Management**: Auto-save every 10 epochs with best model selection
- **Mixed Precision**: FP16 optimization for RTX 3090
- **Error Recovery**: Automatic restart on minor failures
- **Comprehensive Logging**: Real-time metrics and progress tracking

#### Training Configuration (Optimized)
- **Resolution**: 1440×808 (native camera resolution scaled)
- **Batch Size**: 8 (optimized for RTX 3090)
- **Epochs**: 100 (with early stopping)
- **Learning Rate**: 0.001 with AdamW optimizer
- **Augmentations**: Mosaic, MixUp, HSV, rotation, scaling
- **Loss Weights**: box=7.5, cls=0.5, dfl=1.5

#### Expected Outputs
- **Best Model**: `RNA/weights/detector_best.pt` (~25-30 MB)
- **Last Checkpoint**: `RNA/weights/detector_last.pt` (~25-30 MB)  
- **Training Logs**: `RNA/training/logs/detector_training.log`
- **Metrics**: JSON stats with performance data
- **Visualizations**: Training curves and validation plots

#### Post-Training Next Steps
1. **Verifier Training**: SmokeyNet-Lite temporal model (2-3 hours)
2. **Pipeline Integration**: Complete cascade inference testing
3. **Performance Validation**: Benchmark against target metrics
4. **Production Deployment**: TensorRT optimization and API setup

---

*Last updated: 2025-08-20*
*Training Status: Pipeline implemented and ready for execution*
*Dataset status: 173K images downloaded and ready for training*
*All 5 datasets completed: FASDD, PyroNear, D-Fire, FIgLib, NEMO*
*Next review: Training completion and performance validation*