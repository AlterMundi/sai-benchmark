# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SAI-Benchmark is a **unified multi-dimensional vision assessment framework** that provides comprehensive benchmarking capabilities for vision-language models. The framework supports:

- **Multi-Prompt Testing**: Centralized prompt registry with templating system for systematic prompt evaluation
- **Multi-Model Support**: Unified interface supporting Qwen 2.5-VL, LLaMA 3.2 Vision, Gemma 3 27B, Mistral Small 3.1 24B Vision, LLaVA, and MiniCPM models
- **Multi-Engine Architecture**: Integrated support for Ollama, Hugging Face Transformers, and extensible to other inference backends
- **Automated Test Suites**: YAML-based test configurations for reproducible, systematic evaluation
- **Unified Metrics Collection**: Structured metrics including precision, recall, F1, IOU, and latency with configurable aggregation

The framework enables systematic comparison of model performance across different prompting strategies and models, providing a robust foundation for data-driven optimization of vision detection systems.

## Commands

### Setup
```bash
# Install dependencies
pip3 install -r requirements.txt

# For Ollama backend
ollama pull qwen2.5-vl:7b
ollama pull llama3.2-vision:11b
ollama pull minicpm-v:8b
ollama serve

# For Hugging Face backend
pip3 install transformers>=4.37.0 accelerate qwen-vl-utils torch>=2.0.0
# Optional: Set cache directory
export HF_HOME=/mnt/models
# Optional: For memory-efficient inference
pip3 install bitsandbytes
```

### Running Benchmarks

#### Test Suite Framework
```bash
# Run predefined test suite
python3 run_suite.py --suite suites/early_detection.yaml

# Custom matrix testing across multiple dimensions
python3 run_matrix.py --prompts "early_fire_json,wildfire_confidence" --models "qwen2.5-vl:7b,llama3.2-vision:11b" --engines "ollama,hf"

# Analyze results and generate comparison reports
python3 analyze_results.py --results out/matrix_results.json --report comparison
```

#### Direct Model Evaluation
```bash
# Qwen 2.5-VL with Ollama backend
python3 evaluate.py --engine ollama --dataset <dataset_path>

# Qwen 2.5-VL with Hugging Face backend
python3 evaluate.py --engine hf --dataset <dataset_path>

# Compare both backends
python3 evaluate.py --engine both --dataset <dataset_path>
```

#### Resource Management Testing
```bash
# Test resource allocation and management
python3 test_resource_management.py
```

#### Legacy Tests (deprecated)
```bash
# Old smoke detection pipeline
python3 tests/test_server_base.py --sequence_folder <path_to_sequence> --model <model_name>

# Old parallel benchmark
python3 tests/benchmark-06.py --dataset <dataset_path> --model <model_name> --workers <num_workers>
```

### Common Arguments
- `--engine`: Backend to use ("ollama", "hf", or "both")
- `--dataset`: Path to dataset directory
- `--model`: Model to use (e.g., "qwen2.5-vl:7b", "llama3.2-vision:11b")
- `--max_tokens`: Maximum visual tokens (256-1280 recommended for Qwen 2.5-VL)
- `--iou_threshold`: IOU threshold for bbox evaluation (default 0.4)
- `--fps_sampling`: Frame sampling rate for sequences
- `--workers`: Number of parallel workers for processing
- `--output`: Output directory for results (default: "out/")

## Architecture

### Unified Multi-Dimensional Framework (Target Architecture)

```
                    +----------------------+
                    |   Test Suite Runner  |
                    +----------+-----------+
                               |
                    +----------v-----------+
                    |   Prompt Registry    |
                    |   Model Registry     |
                    |   Engine Registry    |
                    +----------+-----------+
                               |
          +--------------------+--------------------+
          |                    |                    |
    +-----v-----+        +-----v-----+        +-----v-----+
    |  Ollama   |        |    HF     |        |  OpenAI   |
    |  Engine   |        |  Engine   |        |  Engine   |
    +-----------+        +-----------+        +-----------+
          |                    |                    |
    qwen2.5-vl:7b    Qwen2.5-VL-7B-Instruct    gpt-4o
                               |
                    +----------v-----------+
                    |  Metrics Collector   |
                    |  Results Database    |
                    +----------------------+
```

### Current Qwen 2.5-VL Integration Architecture

```
          +------------------+
          | benchmark runner |
          +--------+---------+
                   |
         ┌─────────┴─────────┐
         |                   |
+---------v------+   +--------v---------+
|  local Ollama  |   |  HF Transformers |
|  (HTTP API)    |   |  (Python)        |
+---------+------+   +--------+---------+
         |                   |
   qwen2.5-vl:7b     Qwen2.5-VL-7B-Instruct
```

### Core Framework Components

1. **Core Registry System** (`core/`):
   - `prompt_registry.py`: Centralized prompt templates with metadata and versioning
   - `model_registry.py`: Multi-engine model configurations and capabilities
   - `engine_registry.py`: Unified interface for different inference backends
   - `metrics_registry.py`: Configurable evaluation metrics and aggregation functions

2. **Test Suite Framework** (`core/`):
   - `test_suite.py`: Template-based test configuration and execution
   - `matrix_runner.py`: Automated multi-dimensional testing across prompts/models/engines
   - `results_analyzer.py`: Statistical analysis and performance comparison tools

3. **Engine Implementations** (`engines/`):
   - `ollama_engine.py`: Local Ollama API integration
   - `hf_engine.py`: Hugging Face Transformers integration  
   - `openai_engine.py`: OpenAI API integration
   - `base_engine.py`: Abstract base class for consistent interface

4. **Suite Templates** (`suites/`):
   - YAML configuration files for different testing scenarios
   - Automated test execution with customizable parameters
   - Version-controlled test definitions for reproducibility

### Model Implementation Components

1. **Model Implementations** (`models/`):
   - `qwen_model.py`: Qwen 2.5-VL implementation with dynamic resolution support
   - `llama_model.py`: LLaMA 3.2 Vision support
   - `gemma_model.py`: Gemma 3 27B vision model
   - `mistral_model.py`: Mistral Small 3.1 24B Vision
   - `llava_model.py`: LLaVA implementation
   - `minicpm_model.py`: MiniCPM vision model
   - `discovery.py`: Automatic model discovery system
   - `registry.py`: Model factory and registration

2. **Evaluation Pipeline** (`evaluate.py`):
   - Unified interface for both backends
   - Handles image preprocessing and dynamic resolution
   - Enforces JSON output schema for smoke detection

3. **Benchmark Adjustments**:
   - Dynamic FPS sampling for long sequences (>1280 visual tokens)
   - IOU threshold relaxed to 0.4 for ViT-14 spatial granularity
   - Latency normalization based on backend (Ollama: 0.8-1.2s, HF: 0.25s)

### Legacy Components

1. **Smoke Detection Pipeline** (`test_server_base.py`):
   - Processes image sequences from folders
   - Converts images to base64 for LLM input
   - Uses structured prompts for smoke/fire detection
   - Calculates metrics against ground truth labels

2. **Parallel Processing** (`benchmark-06.py`):
   - Multi-threaded sequence processing using `concurrent.futures`
   - Caches ground truth data to avoid repeated file reads
   - Saves incremental results to prevent data loss
   - Optimizes performance with image resizing

3. **Dataset Structure**:
   - Sequences organized in folders: `dataset/sequence_name/`
   - Images: `*.jpg`, `*.jpeg`, `*.png`
   - Ground truth: `sequence_name.txt` (empty = no smoke, non-empty = smoke)
   - Bounding boxes in ground truth: `x_center,y_center,width,height` (normalized 0-1)

### Key Design Patterns

- **Engine Abstraction**: Unified interface across Ollama/HF/OpenAI backends
- **Model Discovery**: Automatic detection of available models via `discovery.py`
- **Prompt Templates**: Versioned prompts with metadata in registry
- **Response Parsing**: JSON schema validation with fallback strategies
- **Error Handling**: Graceful degradation with detailed error reporting
- **Progress Tracking**: Real-time progress with `tqdm` and incremental saves
- **Resource Management**: Queue-based GPU/CPU allocation for parallel processing

## Testing & Development

### Running Tests
```bash
# Run unit tests
python3 -m pytest tests/

# Test specific functionality
python3 tests/test.py

# Resource management tests
python3 test_resource_management.py
```

### Development Workflow
1. Make changes to model implementations in `models/`
2. Update registry entries in `core/model_registry.py` for new models
3. Add new prompts to `core/prompt_registry.py`
4. Create test suites in `suites/` for new scenarios
5. Run benchmarks using `run_suite.py` or `run_matrix.py`
6. Analyze results with `analyze_results.py`

## Important Notes

### Model-Specific Notes

#### Qwen 2.5-VL
- Uses Naive Dynamic Resolution with 256-1280 visual tokens recommended
- ViT-14 patch size affects spatial granularity (IOU threshold: 0.4)
- Supports both Ollama (quantized) and HF Transformers backends
- JSON schema enforcement: `{"has_smoke": bool, "bbox": [x_center, y_center, width, height]}`

#### LLaMA 3.2 Vision
- Available in 11B and 90B variants
- Requires explicit image tags in prompts
- Strong zero-shot performance on detection tasks

#### Performance Considerations
- Ollama backend: 0.8-1.2s per image (CPU/quantized)
- HF backend: ~0.25s per image (GPU)
- Dynamic FPS sampling for sequences >1280 visual tokens
- Results saved as timestamped JSON in `out/` directory

### Framework Capabilities
- **Multi-Dimensional Testing**: Test across prompts × models × engines in single run
- **Resource Management**: Automatic GPU/CPU allocation with queueing
- **Incremental Results**: Saves progress to prevent data loss
- **Statistical Analysis**: Built-in metrics aggregation and comparison
- **Reproducible Benchmarks**: YAML-based test definitions with versioning

## Permissions and Tools

### Git Operations
Claude has permission to:
- Create branches: `git checkout -b feature/branch-name`
- Stage changes: `git add -A` or `git add specific-file`
- Commit with descriptive messages: `git commit -m "feat: description"`
- View logs: `git log --oneline -n 10`
- Check status: `git status`
- View diffs: `git diff` or `git diff --staged`
- Stash changes: `git stash` / `git stash pop`

### VSCode Integration
Claude can execute VSCode commands:
- Open files: `code filename.py`
- Open folders: `code .`
- Install extensions: `code --install-extension ms-python.python`
- List extensions: `code --list-extensions`

### Useful Bash Commands
Claude is authorized to use:
- System monitoring: `htop`, `df -h`, `free -h`, `ps aux`
- File operations: `find`, `grep`, `sed`, `awk`, `sort`, `uniq`
- Network tools: `curl`, `wget`, `netstat`, `ss`
- Process management: `kill`, `killall`, `pgrep`, `pkill`
- Archive operations: `tar`, `zip`, `unzip`, `gzip`
- Performance tools: `time`, `watch`, `timeout`
- Python tools: `pip`, `python3`, `pytest`, `black`, `ruff`
- Docker: `docker ps`, `docker logs`, `docker exec`

## Code Style & Quality

### Linting and Formatting
```bash
# Format code with Black
black .

# Lint with ruff
ruff check .

# Type checking (if using type hints)
mypy .
```

### Monitoring and Debugging
```bash
# Monitor benchmark progress
python3 monitor_benchmark.py

# View logs
tail -f out/smoke_detector.log

# Check GPU usage during benchmarks
watch -n 1 nvidia-smi
```