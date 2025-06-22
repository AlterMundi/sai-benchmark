# SAI-Benchmark

**Unified Multi-Dimensional Vision Assessment Framework**

SAI-Benchmark provides comprehensive benchmarking capabilities for vision-language models with support for multiple prompting strategies, models, and inference backends.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-passing-green.svg)](tests/)

## 🌟 Key Features

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