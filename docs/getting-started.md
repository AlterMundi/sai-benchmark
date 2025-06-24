# Getting Started with SAI-Benchmark

Welcome to SAI-Benchmark! This guide will help you get up and running quickly with the unified multi-dimensional vision assessment framework.

## What is SAI-Benchmark?

SAI-Benchmark is a comprehensive benchmarking framework for vision-language models that provides:

- **Multi-Prompt Testing**: Systematic evaluation across different prompting strategies
- **Multi-Model Support**: Unified interface for various vision models (Qwen 2.5-VL, LLaMA 3.2 Vision, etc.)
- **Multi-Engine Architecture**: Support for Ollama, Hugging Face, and OpenAI backends
- **Automated Test Suites**: YAML-based configurations for reproducible benchmarks
- **Advanced Metrics**: Precision, recall, F1, IoU, latency with statistical analysis

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/AlterMundi/sai-benchmark.git
cd sai-benchmark

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Verify Installation

```bash
# Quick validation test
python validate_tests.py

# Run basic tests
pytest tests/unit/ -v
```

### 3. Prerequisites

Before setting up model backends, ensure you have the required external dependencies:

- **Ollama** (for local inference): Must be installed separately from [ollama.ai](https://ollama.ai/download)
- **NVIDIA drivers** (for GPU acceleration): Required for CUDA-enabled models

### 4. Set Up Your First Model Backend

Choose one of the following backends:

#### Option A: Ollama (Local, CPU/GPU)
```bash
# Assuming Ollama is already installed (see Prerequisites above)
ollama pull qwen2.5-vl:7b
ollama serve
```

#### Option B: Hugging Face (GPU Required)
```bash
# Install additional dependencies
pip install transformers>=4.37.0 accelerate qwen-vl-utils torch>=2.0.0

# Optional: Set cache directory
export HF_HOME=/path/to/your/model/cache
```

#### Option C: OpenAI API
```bash
# Set your API key
export OPENAI_API_KEY=your_api_key_here
```

### 4. Run Your First Benchmark

```bash
# Run a predefined test suite
python run_suite.py --suite suites/early_detection.yaml

# Or run a quick matrix test
python run_matrix.py \
  --prompts "early_fire_json" \
  --models "qwen2.5-vl:7b" \
  --engines "ollama"
```

### 5. Analyze Results

```bash
# Generate a summary report
python analyze_results.py \
  --results out/suite_early_detection_*.json \
  --report summary
```

## Your First Custom Test

### Create a Simple Test Suite

Create `my_first_test.yaml`:

```yaml
name: "my_first_benchmark"
description: "My first SAI-Benchmark test"
version: "1.0"

prompts:
  - "early_fire_json"

models:
  - "qwen2.5-vl:7b"

datasets:
  - "path/to/your/test/images"

metrics:
  - "accuracy"
  - "latency"

engine_config:
  max_tokens: 512
  temperature: 0.1

test_config:
  max_workers: 2
  iou_threshold: 0.4
```

### Run Your Custom Test

```bash
python run_suite.py --suite my_first_test.yaml
```

## Understanding the Results

Results are saved as JSON files in the `out/` directory with timestamps:

```
out/
├── suite_my_first_benchmark_20241215_143052_results.json
├── suite_my_first_benchmark_20241215_143052_config.json
└── logs/
```

### Key Metrics Explained

- **Accuracy**: Overall correctness of predictions
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1 Score**: Harmonic mean of precision and recall
- **IoU**: Intersection over Union for bounding box accuracy
- **Latency**: Average response time per image

## Next Steps

### Explore Advanced Features

1. **[Multi-Model Comparison](tutorials/multi-model-comparison.md)** - Compare different models
2. **[Custom Metrics](tutorials/custom-metrics.md)** - Create your own evaluation metrics
3. **[Advanced Test Suites](guides/test-suite-guide.md)** - Complex testing scenarios

### Learn the Framework

1. **[Architecture Overview](architecture.md)** - Understand the framework design
2. **[Configuration Guide](configuration.md)** - Deep dive into configuration options
3. **[API Reference](api/core.md)** - Programmatic usage

### Get Help

- **[User Guide](guides/user-guide.md)** - Comprehensive documentation
- **[Troubleshooting](guides/troubleshooting.md)** - Common issues and solutions
- **[GitHub Issues](https://github.com/AlterMundi/sai-benchmark/issues)** - Report bugs or ask questions

## Common Use Cases

### Fire/Smoke Detection
```bash
python run_suite.py --suite suites/early_detection.yaml
```

### Model Performance Comparison
```bash
python run_matrix.py \
  --prompts "early_fire_json,wildfire_confidence" \
  --models "qwen2.5-vl:7b,llama3.2-vision:11b" \
  --engines "ollama,hf"
```

### Prompt Optimization
```bash
python run_suite.py --suite suites/prompt_optimization.yaml
```

## System Requirements

### Minimum Requirements
- Python 3.8+
- 8GB RAM
- 10GB disk space

### Recommended for GPU Inference
- NVIDIA GPU with 16GB+ VRAM
- CUDA 11.8+ or ROCm 5.4+
- 32GB+ RAM

### Supported Platforms
- Linux (Ubuntu 20.04+, CentOS 8+)
- macOS 12.0+
- Windows 10/11 (WSL2 recommended)

---

**Ready to dive deeper?** Check out the [User Guide](guides/user-guide.md) for comprehensive documentation, or jump into [tutorials](tutorials/) for hands-on learning!