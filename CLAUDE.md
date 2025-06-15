# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SAI-Benchmark is transitioning to focus exclusively on **Qwen 2.5-VL** for early-fire detection benchmarking. The project now supports dual execution paths: local inference via Ollama and cloud/GPU inference via Hugging Face Transformers. This pivot leverages Qwen 2.5-VL's native dynamic resolution capabilities and window attention mechanisms for improved smoke/fire detection performance.

## Commands

### Setup
```bash
# Install dependencies
pip3 install -r requirements.txt

# For Ollama path
ollama pull qwen2.5-vl:7b
ollama serve

# For Hugging Face path
pip3 install transformers accelerate qwen-vl-utils
# Optional: Set cache directory
export HF_HOME=/mnt/models
```

### Running Benchmarks

#### Qwen 2.5-VL Early-Fire Detection
```bash
# Run with Ollama backend
python3 evaluate.py --engine ollama --dataset <dataset_path>

# Run with Hugging Face backend
python3 evaluate.py --engine hf --dataset <dataset_path>

# Compare both backends
python3 evaluate.py --engine both --dataset <dataset_path>
```

#### Legacy Tests (for reference)
```bash
# Old smoke detection pipeline
python3 tests/test_server_base.py --sequence_folder <path_to_sequence> --model <model_name>

# Old parallel benchmark
python3 tests/benchmark-06.py --dataset <dataset_path> --model <model_name> --workers <num_workers>
```

### Common Arguments
- `--engine`: Backend to use ("ollama", "hf", or "both")
- `--dataset`: Path to dataset directory
- `--max_tokens`: Maximum visual tokens (256-1280 recommended for Qwen 2.5-VL)
- `--iou_threshold`: IOU threshold for bbox evaluation (default 0.4)
- `--fps_sampling`: Frame sampling rate for sequences

## Architecture

### Qwen 2.5-VL Integration Architecture

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

### Key Components

1. **Model Wrappers** (`models/`):
   - `ollama_qwen.py`: HTTP-based interface for local Ollama inference
   - `hf_qwen.py`: Direct Transformers integration for GPU inference

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

### Key Design Patterns

- **Ollama Integration**: All models run locally via Ollama API
- **Response Parsing**: Regex-based extraction of structured outputs
- **Error Handling**: Graceful failure with detailed error messages
- **Progress Tracking**: Uses `tqdm` for real-time progress visualization

## Important Notes

- Qwen 2.5-VL requires specific handling for its dynamic resolution system
- Ollama backend uses quantized GGUF models (int4) for CPU inference
- HF backend requires GPU with 16-24GB VRAM for 7B model
- JSON output schema: `{"has_smoke": bool, "bbox": [x_center, y_center, width, height]}`
- If JSON parsing fails, implement fallback with "BAD_JSON" detection
- Results are saved as JSON files in the `out/` directory

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