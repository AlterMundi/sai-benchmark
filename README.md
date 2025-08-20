# SAI-Benchmark

**Unified Multi-Dimensional Vision Assessment Framework**

SAI-Benchmark provides comprehensive benchmarking capabilities for vision-language models with support for multiple prompting strategies, models, and inference backends. Now includes **SAI Neural Network Architecture (RNA)** for early fire detection with cascade inference pipeline.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-passing-green.svg)](tests/)

## 🔥 NEW: SAI Neural Network Implementation

The project now includes a complete neural network implementation for early fire detection with **production-ready datasets**:

- **🎯 Cascade Architecture**: YOLOv8-s detector + SmokeyNet-Lite temporal verifier
- **📊 Native Resolution**: 1440×808 optimized for camera feeds (2880×1616 scaled)
- **⚡ High Performance**: 6-10 FPS on RTX 3090, 100-150ms latency
- **🧠 Smart Verification**: Temporal consistency with 2-3 frame persistence
- **📈 Production Ready**: Integrated with SAI-Benchmark framework
- **📀 Complete Datasets**: 173K+ training images from 5 major fire detection datasets

### Dataset Status (Latest Update)
| Dataset | Status | Images | Size | Source |
|---------|--------|--------|------|--------|
| **FASDD** | ✅ Ready | 95,314 | 11.4GB | Kaggle |
| **PyroNear-2024** | ✅ Ready | 33,600 | 3.1GB | HuggingFace |
| **D-Fire** | ✅ Ready | 21,527 | 3.0GB | Manual |
| **FIgLib** | ✅ Ready | 19,317 | 277MB | HuggingFace |
| **NEMO** | ✅ Ready | 3,493 | 1.42GB | Kaggle |

**Total Available**: 173,251 training images ready for immediate use

### Quick Start with SAI RNA

#### Training Pipeline (Ready to Run)
```bash
# Check system readiness
python3 check_training_readiness.py

# Start autonomous training (15-20 hours)
./start_detector_training.sh

# Monitor training progress
tail -f RNA/training/logs/detector_training.log
```

#### Manual Training Steps
```bash
# Setup training environment
source RNA/training/venv/bin/activate

# Train YOLOv8-s detector
python RNA/training/detector_trainer.py --config RNA/configs/sai_cascade_config.yaml

# Train SmokeyNet-Lite verifier (after detector completes)
python RNA/training/verifier_trainer.py --config RNA/configs/sai_cascade_config.yaml

# Run cascade inference
python RNA/inference/cascade_inference.py --weights RNA/weights/
```

See [RNA Documentation](RNA/README.md) for detailed implementation guide.

## 🔄 SAI Temporal Workflow: Distributed Camera System

### Architecture Overview

The SAI system operates with a **distributed temporal architecture** where edge cameras capture images periodically and send them to a central server for cascade inference processing.

### 📸 Camera Capture Flow
```
Camera Network (every 5 seconds):
Camera 1: Photo @ T+0s, T+5s, T+10s, T+15s...
Camera 2: Photo @ T+2s, T+7s, T+12s, T+17s...
Camera N: Photo @ T+xs, T+x+5s, T+x+10s...
```

### 🖥️ Server-Side Temporal Processing

**The server maintains independent temporal buffers for each camera:**

```python
# Server Architecture
server_buffers = {
    "camera_1": TemporalBuffer(max_length=5, retention_time=60.0),
    "camera_2": TemporalBuffer(max_length=5, retention_time=60.0),
    "camera_N": TemporalBuffer(max_length=5, retention_time=60.0)
}
```

### 🔄 Per-Photo Inference Pipeline

**When a new photo arrives from any camera:**

```
1. Photo Reception
   📸 New photo from Camera_1 @ T=15s
   ↓
2. Buffer Update  
   🗂️ Buffer: [Photo_T-10s, Photo_T-5s, Photo_CURRENT]
   ↓
3. Detector Stage
   🎯 YOLOv8-s processes ONLY current photo → detects ROI(x,y,w,h)
   ↓
4. Temporal Verifier Stage
   🔍 SmokeyNet-Lite extracts ROI from last 3 buffered photos:
       • ROI(x,y,w,h) from Photo_T-10s → Frame 1
       • ROI(x,y,w,h) from Photo_T-5s  → Frame 2  
       • ROI(x,y,w,h) from Photo_CURRENT → Frame 3
   ↓
5. Temporal Analysis
   ⏱️ Analyzes 3-frame sequence → Smoke/No-Smoke verification
   ↓
6. Alarm Decision
   🚨 Persistence logic → Final alarm trigger
```

### ⏱️ Real-Time Timeline Example

```
T=0s:   Camera_1 sends Photo_1 
        → Buffer: [Photo_1] 
        → Detector: ✅ Detection 
        → Verifier: ❌ SKIP (only 1 frame)

T=5s:   Camera_1 sends Photo_2 
        → Buffer: [Photo_1, Photo_2] 
        → Detector: ✅ Detection 
        → Verifier: ⚠️ PARTIAL (padded sequence)

T=10s:  Camera_1 sends Photo_3 
        → Buffer: [Photo_1, Photo_2, Photo_3] 
        → Detector: ✅ Detection 
        → Verifier: ✅ FULL ANALYSIS (complete 3-frame sequence)

T=15s:  Camera_1 sends Photo_4 
        → Buffer: [Photo_2, Photo_3, Photo_4] 
        → Detector: ✅ Detection 
        → Verifier: ✅ FULL ANALYSIS → 🚨 ALARM TRIGGERED!
```

### 🎯 Cold Start Behavior

**System handles initial frames gracefully:**

- **Frame 1**: Detector only (no temporal verification)
- **Frame 2**: Verifier with padded sequence `[Frame1, Frame1, Frame2]`
- **Frame 3+**: Full temporal analysis with real sequences

### ⚡ Key Advantages

- **📊 One inference per photo**: Efficient processing
- **🔄 Independent camera buffers**: Scalable architecture  
- **⚱️ Low latency**: Immediate response per photo
- **🧠 Temporal intelligence**: Captures smoke evolution over time
- **🛡️ False positive reduction**: Multi-frame verification
- **🚀 Cold start ready**: Works from first frame

### 🔧 Configuration Parameters

```python
temporal_frames: int = 3                    # Analyze last 3 photos
min_persistence_frames: int = 2             # Minimum detections for alarm
persistence_time_window: float = 30.0       # 30-second alarm window
buffer_retention_time: float = 60.0         # Keep photos for 60 seconds
```

This distributed temporal architecture ensures **robust early fire detection** while maintaining **real-time performance** across multiple camera deployments.

## 🌟 Benchmark Framework Features

- **🎯 Multi-Prompt Testing**: Systematic evaluation across different prompting strategies
- **🤖 Multi-Model Support**: Qwen 2.5-VL, LLaMA 3.2 Vision, Gemma 3 27B, Mistral Small 3.1 24B Vision, and more
- **⚙️ Multi-Engine Architecture**: Ollama (local), HuggingFace (GPU), OpenAI (API) backends
- **🧪 Automated Test Suites**: YAML-based configurations for reproducible benchmarks
- **📊 Advanced Metrics**: Precision, recall, F1, IoU, latency with statistical analysis
- **🔧 Resource Management**: Intelligent GPU/CPU allocation and conflict resolution
- **📈 Results Analysis**: Comprehensive reporting and model comparison tools

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/AlterMundi/sai-benchmark.git
cd sai-benchmark

# Setup environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Verify installation
python validate_tests.py
```

### Basic Usage

```bash
# Setup Ollama (local inference)
ollama pull qwen2.5-vl:7b
ollama serve

# Run predefined test suite
python run_suite.py --suite suites/early_detection.yaml

# Matrix testing across models and prompts
python run_matrix.py \
  --prompts "early_fire_json,wildfire_confidence" \
  --models "qwen2.5-vl:7b,llama3.2-vision:11b" \
  --engines "ollama"

# Analyze results
python analyze_results.py \
  --results out/suite_*.json \
  --report comparison
```

## 📚 Documentation

### 🎓 Getting Started
- **[Quick Start Guide](docs/getting-started.md)** - Installation and first benchmark
- **[Basic Usage Tutorial](docs/tutorials/basic-usage.md)** - Step-by-step walkthrough
- **[Architecture Overview](docs/architecture.md)** - Framework design and components

### 📖 User Guides
- **[User Guide](docs/guides/user-guide.md)** - Comprehensive documentation
- **[Configuration Guide](docs/configuration.md)** - Test suite and model configuration
- **[CLI Reference](docs/guides/cli-reference.md)** - Command-line interface documentation

### 🎯 Tutorials
- **[Multi-Model Comparison](docs/tutorials/multi-model-comparison.md)** - Compare different models
- **[Custom Metrics](docs/tutorials/custom-metrics.md)** - Create custom evaluation metrics
- **[Advanced Benchmarking](docs/tutorials/advanced-benchmarking.md)** - Complex evaluation scenarios

### 🔧 Development
- **[Contributing Guide](docs/contributing/contributing.md)** - How to contribute
- **[Development Setup](docs/contributing/development.md)** - Development environment
- **[Testing Guide](docs/contributing/testing.md)** - Comprehensive test suite documentation

### 📋 Reference
- **[API Documentation](docs/api/)** - Complete API reference
- **[Examples](docs/examples/)** - Sample configurations and scripts

## 🏗️ Architecture

```
                    ┌─────────────────────┐
                    │   Test Suite Runner │
                    └─────────┬───────────┘
                              │
                    ┌─────────┴───────────┐
                    │   Registry System   │
                    │ • Prompts • Models  │
                    │ • Engines • Metrics │
                    └─────────┬───────────┘
                              │
          ┌───────────────────┼───────────────────┐
          │                   │                   │
    ┌─────▼─────┐       ┌─────▼─────┐       ┌─────▼─────┐
    │   Ollama  │       │    HF     │       │  OpenAI   │
    │  Engine   │       │  Engine   │       │  Engine   │
    └───────────┘       └───────────┘       └───────────┘
```

**Core Components:**
- **Registry System**: Centralized management of prompts, models, engines, and metrics
- **Test Suite Framework**: YAML-based test configuration and execution
- **Resource Manager**: GPU/CPU allocation and conflict resolution
- **Analysis Tools**: Statistical analysis and comparison reporting

## 🎯 Supported Models

### Vision-Language Models
- **Qwen 2.5-VL** (7B, 72B) - Ollama & HuggingFace
- **LLaMA 3.2 Vision** (11B, 90B) - Ollama & HuggingFace  
- **Gemma 3 27B Vision** - Ollama & HuggingFace
- **Mistral Small 3.1 24B Vision** - Ollama & HuggingFace
- **LLaVA** (7B, 13B) - Ollama
- **MiniCPM-V** (8B) - Ollama
- **GPT-4V / GPT-4O** - OpenAI API

### Inference Backends
- **Ollama**: Local quantized models (CPU/GPU)
- **HuggingFace**: Full-precision models (GPU required)
- **OpenAI**: API-based inference

## 📊 Use Cases

### Fire & Smoke Detection
```bash
python run_suite.py --suite suites/early_detection.yaml
```

### Model Performance Comparison
```bash
python run_matrix.py \
  --prompts "early_fire_json,wildfire_confidence" \
  --models "qwen2.5-vl:7b,llama3.2-vision:11b,gpt-4o" \
  --engines "ollama,hf,openai"
```

### Prompt Optimization
```bash
python run_suite.py --suite suites/prompt_optimization.yaml
```

## 🧪 Testing

The framework includes a comprehensive test suite with 4,400+ lines of test code:

```bash
# Run all tests
pytest tests/

# Quick validation
python validate_tests.py

# Run specific test categories
pytest tests/unit/          # Unit tests
pytest tests/integration/   # Integration tests
pytest -m "not slow"        # Exclude slow tests
```

**Test Coverage:**
- ✅ Unit tests for all core components
- ✅ Integration tests for end-to-end workflows
- ✅ Property-based testing with Hypothesis
- ✅ Thread safety and resource management tests
- ✅ Performance and memory usage validation

## 🔧 System Requirements

### Minimum Requirements
- Python 3.8+
- 8GB RAM
- 10GB disk space

### Recommended for GPU Inference
- NVIDIA GPU with 16GB+ VRAM
- 32GB+ RAM
- CUDA 11.8+ or ROCm 5.4+

### Supported Platforms
- Linux (Ubuntu 20.04+, CentOS 8+)
- macOS 12.0+
- Windows 10/11 (WSL2 recommended)

### External Dependencies
- **Ollama**: Required for local model inference. Must be installed separately - see [Ollama installation guide](https://ollama.ai/download)
- **NVIDIA drivers**: Required for GPU acceleration (when using CUDA models)

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](docs/contributing/contributing.md) for details.

### Ways to Contribute
- 🐛 **Bug Reports**: Report issues and bugs
- 💡 **Feature Requests**: Suggest new features
- 🔧 **Code Contributions**: Implement features, fix bugs
- 📖 **Documentation**: Improve docs, write tutorials
- 🧪 **Testing**: Add tests, improve coverage

### Development Setup
```bash
# Clone and setup
git clone https://github.com/YOUR_USERNAME/sai-benchmark.git
cd sai-benchmark

# Install development dependencies
pip install -r requirements.txt
pip install -e .

# Run tests
pytest tests/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Qwen Team** for the excellent Qwen 2.5-VL model
- **Meta** for LLaMA 3.2 Vision
- **Google** for Gemma 3 27B Vision
- **Mistral AI** for Mistral Small 3.1 24B Vision
- **Ollama** for the local inference platform
- **HuggingFace** for the Transformers library
- **OpenAI** for the vision API

## 📞 Support

- **Documentation**: [docs/](docs/)
- **Issues**: [GitHub Issues](https://github.com/AlterMundi/sai-benchmark/issues)
- **Discussions**: [GitHub Discussions](https://github.com/AlterMundi/sai-benchmark/discussions)

---

**SAI-Benchmark** - Making vision-language model evaluation systematic, reproducible, and accessible.