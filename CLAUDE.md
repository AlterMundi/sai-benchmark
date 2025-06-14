# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SAI-Benchmark has evolved from a smoke/fire detection benchmark to a comprehensive Multi-Dimensional Vision Assessment (MDVA) framework. The repository contains both legacy smoke detection tests and a new synthesized approach for definitively testing LLM vision capabilities across multiple dimensions.

## Commands

### Setup
```bash
# Install dependencies
pip3 install -r requirements.txt

# Ensure Ollama is running locally
ollama serve
```

### Running Benchmarks

#### Legacy Smoke Detection Tests
```bash
# Run smoke detection on a single sequence
python3 tests/test_server_base.py --sequence_folder <path_to_sequence> --model <model_name>

# Run parallel benchmark on multiple sequences
python3 tests/benchmark-06.py --dataset <dataset_path> --model <model_name> --workers <num_workers>
```

#### New MDVA Framework
```bash
# Run comprehensive vision benchmark
python3 vision_benchmark_prototype.py --model <model_name> --output results.json

# Run single test demonstration
python3 example_benchmark_run.py single

# Run model comparison
python3 example_benchmark_run.py compare
```

### Common Arguments
- `--model`: Ollama model name (e.g., "gemma-2b", "llama3.1")
- `--temperature`: Model temperature (0.0-1.0)
- `--max_images`: Maximum images per sequence
- `--resize`: Target resolution for image resizing
- `--workers`: Number of parallel workers

## Architecture

### New MDVA Framework (vision_benchmark_prototype.py)

The Multi-Dimensional Vision Assessment framework tests LLM vision capabilities across four levels:

1. **Basic Perception** (0-25): Colors, shapes, counting, object presence
2. **Spatial Understanding** (26-50): Relative positions, depth, motion, composition
3. **Semantic Understanding** (51-75): Object relationships, scene context, activities
4. **Abstract Reasoning** (76-100): Emotional content, symbolism, narratives

Key classes:
- `VisionBenchmark`: Main orchestrator
- `TestCase`: Individual test structure
- `VisionTest`: Abstract base for test categories
- `PerceptionTest`, `SpatialTest`: Concrete implementations

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

- The system expects Ollama to be running locally on default port
- Results are saved as JSON files in the `out/` directory
- Spanish comments in some files indicate international development
- The `old/` directory contains previous iterations for reference
- See `vision_test_synthesis.md` for detailed design rationale and framework evolution
- The new MDVA framework in `vision_benchmark_prototype.py` represents a comprehensive approach to LLM vision testing