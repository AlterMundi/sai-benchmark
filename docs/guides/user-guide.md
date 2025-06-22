# SAI-Benchmark User Guide

Comprehensive guide for using the SAI-Benchmark framework to evaluate vision-language models.

## Table of Contents

- [Overview](#overview)
- [Installation and Setup](#installation-and-setup)
- [Basic Usage](#basic-usage)
- [Test Suite Configuration](#test-suite-configuration)
- [Model Management](#model-management)
- [Results Analysis](#results-analysis)
- [Advanced Features](#advanced-features)
- [Troubleshooting](#troubleshooting)

## Overview

SAI-Benchmark provides a unified framework for evaluating vision-language models across multiple dimensions:

- **Multiple Models**: Qwen 2.5-VL, LLaMA 3.2 Vision, Gemma 3 27B, Mistral Small 3.1 24B Vision, and more
- **Multiple Engines**: Ollama (local), HuggingFace (GPU), OpenAI (API)
- **Multiple Prompts**: Systematic prompt evaluation with templating
- **Multiple Metrics**: Accuracy, precision, recall, F1, IoU, latency, and custom metrics

## Installation and Setup

### System Requirements

**Minimum:**
- Python 3.8+
- 8GB RAM
- 10GB disk space

**Recommended for GPU inference:**
- NVIDIA GPU with 16GB+ VRAM
- 32GB+ RAM
- CUDA 11.8+ or ROCm 5.4+

### Installation Steps

1. **Clone and Setup Environment**
   ```bash
   git clone https://github.com/AlterMundi/sai-benchmark.git
   cd sai-benchmark
   
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Verify Installation**
   ```bash
   python validate_tests.py
   pytest tests/unit/ -v
   ```

3. **Configure Backends**

   **For Ollama (Local Inference):**
   ```bash
   # Install Ollama
   curl -fsSL https://ollama.ai/install.sh | sh
   
   # Pull models
   ollama pull qwen2.5-vl:7b
   ollama pull llama3.2-vision:11b
   ollama serve
   ```

   **For HuggingFace (GPU Inference):**
   ```bash
   pip install transformers>=4.37.0 accelerate qwen-vl-utils torch>=2.0.0
   
   # Optional: Set model cache
   export HF_HOME=/path/to/model/cache
   ```

   **For OpenAI API:**
   ```bash
   export OPENAI_API_KEY=your_api_key_here
   ```

## Basic Usage

### Running Predefined Test Suites

```bash
# Run early fire detection benchmark
python run_suite.py --suite suites/early_detection.yaml

# Run model comparison benchmark
python run_suite.py --suite suites/model_comparison.yaml

# Run with custom settings
python run_suite.py \
  --suite suites/early_detection.yaml \
  --workers 4 \
  --output results/my_test/
```

### Matrix Testing

Test across multiple dimensions simultaneously:

```bash
# Compare models and prompts
python run_matrix.py \
  --prompts "early_fire_json,wildfire_confidence" \
  --models "qwen2.5-vl:7b,llama3.2-vision:11b" \
  --engines "ollama"

# Full matrix test
python run_matrix.py \
  --prompts "early_fire_json,wildfire_confidence,detailed_sequence_analysis" \
  --models "qwen2.5-vl:7b,llama3.2-vision:11b,minicpm-v:8b" \
  --engines "ollama,hf" \
  --workers 8
```

### Direct Model Evaluation

```bash
# Single model evaluation
python evaluate.py \
  --engine ollama \
  --model qwen2.5-vl:7b \
  --dataset /path/to/test/images

# Compare backends
python evaluate.py \
  --engine both \
  --model qwen2.5-vl:7b \
  --dataset /path/to/test/images
```

## Test Suite Configuration

### YAML Configuration Format

```yaml
name: "my_benchmark"
description: "Custom benchmark for fire detection"
version: "1.0"

# Test dimensions
prompts:
  - "early_fire_json"
  - "wildfire_confidence"

models:
  - "qwen2.5-vl:7b"
  - "llama3.2-vision:11b"

datasets:
  - "/path/to/fire/sequences"
  - "/path/to/smoke/sequences"

metrics:
  - "accuracy"
  - "precision"
  - "recall"
  - "f1_score"
  - "latency"
  - "bbox_iou"

# Engine configuration
engine_config:
  max_tokens: 512
  temperature: 0.1
  timeout: 30

# Test execution configuration
test_config:
  max_workers: 4
  iou_threshold: 0.4
  confidence_threshold: 0.5
  fps_sampling: 1.0
```

### Configuration Sections

#### Prompts
Define which prompt templates to use:
```yaml
prompts:
  - "early_fire_json"      # Built-in prompt for JSON output
  - "wildfire_confidence"  # Built-in prompt with confidence scores
  - "custom_prompt_id"     # Your custom prompt
```

#### Models
Specify which models to evaluate:
```yaml
models:
  - "qwen2.5-vl:7b"        # Ollama quantized model
  - "qwen2.5-vl-7b-hf"     # HuggingFace model
  - "gpt-4o"               # OpenAI model
```

#### Datasets
Point to your test data:
```yaml
datasets:
  - "/absolute/path/to/data"
  - "relative/path/to/data"
  - "~/home/user/data"
```

**Expected Dataset Structure:**
```
dataset/
├── sequence1/
│   ├── frame_001.jpg
│   ├── frame_002.jpg
│   └── ...
├── sequence2/
│   ├── image_001.png
│   ├── image_002.png
│   └── ...
└── ground_truth/
    ├── sequence1.txt     # Ground truth for sequence1
    └── sequence2.txt     # Ground truth for sequence2
```

#### Metrics
Choose evaluation metrics:
```yaml
metrics:
  - "accuracy"           # Overall correctness
  - "precision"          # True positive rate
  - "recall"             # Sensitivity
  - "f1_score"           # Harmonic mean of precision/recall
  - "bbox_iou"           # Bounding box intersection over union
  - "latency"            # Response time
  - "throughput"         # Images per second
  - "confidence"         # Average confidence scores
```

## Model Management

### Available Models

#### Qwen 2.5-VL
- **Ollama**: `qwen2.5-vl:7b` (quantized, CPU/GPU)
- **HuggingFace**: `qwen2.5-vl-7b-hf` (full precision, GPU required)
- **Capabilities**: Vision, JSON output, dynamic resolution, bbox detection

#### LLaMA 3.2 Vision
- **Ollama**: `llama3.2-vision:11b`, `llama3.2-vision:90b`
- **HuggingFace**: `llama3.2-vision-11b-hf`, `llama3.2-vision-90b-hf`
- **Capabilities**: Vision, structured output, strong zero-shot performance

#### Other Models
- **Gemma 3 27B**: Vision model with large context
- **Mistral Small 3.1 24B Vision**: Efficient vision-language model
- **LLaVA**: Open-source vision-language model
- **MiniCPM-V**: Compact vision model

### Model Selection Guidelines

**For Speed:**
- Ollama quantized models (CPU inference)
- Smaller models (7B vs 11B vs 27B)

**For Accuracy:**
- HuggingFace full-precision models
- Larger models with more parameters

**For Cost:**
- Local models (Ollama, HuggingFace)
- Avoid API-based models for large datasets

### Adding Custom Models

1. **Register Model Configuration:**
   ```python
   from core.model_registry import ModelRegistry, ModelConfig, ModelCapability, EngineType
   
   custom_model = ModelConfig(
       id="my-custom-model",
       name="My Custom Vision Model",
       engine=EngineType.HUGGINGFACE,
       model_path="username/model-name",
       capabilities={ModelCapability.VISION, ModelCapability.JSON_OUTPUT}
   )
   
   model_registry.register_model(custom_model)
   ```

2. **Create Model-Specific Prompts:**
   ```python
   from core.prompt_registry import PromptRegistry, PromptTemplate, OutputSchema
   
   custom_prompt = PromptTemplate(
       id="my_custom_prompt",
       name="Custom Detection Prompt",
       description="Optimized for my custom model",
       template="<image>\nAnalyze this image for {target_object}. Output: {format}",
       output_schema=OutputSchema(type="json", format={"detected": {"type": "boolean"}}),
       tags=["custom", "detection"]
   )
   
   prompt_registry.register_prompt(custom_prompt)
   ```

## Results Analysis

### Understanding Results

Results are saved as timestamped JSON files:
```
out/
├── suite_my_benchmark_20241215_143052_results.json
├── suite_my_benchmark_20241215_143052_config.json
└── logs/
    └── execution.log
```

### Result Structure

```json
{
  "suite_name": "my_benchmark",
  "config": { /* test configuration */ },
  "test_results": [
    {
      "test_case_id": "test_001",
      "prompt_id": "early_fire_json",
      "model_id": "qwen2.5-vl:7b",
      "engine_response": {
        "content": "{\"has_smoke\": true, \"bbox\": [0.5, 0.5, 0.2, 0.3]}",
        "latency": 1.25,
        "tokens_used": 150,
        "success": true
      },
      "metrics": {
        "accuracy": {"value": 1.0, "details": {}},
        "latency": {"value": 1.25, "details": {}}
      }
    }
  ],
  "aggregated_metrics": {
    "accuracy": {"value": 0.85, "details": {"std": 0.12, "count": 50}},
    "latency": {"value": 1.18, "details": {"std": 0.23, "count": 50}}
  },
  "execution_time": 125.4,
  "timestamp": "2024-12-15T14:30:52"
}
```

### Generating Reports

#### Summary Report
```bash
python analyze_results.py \
  --results out/suite_*.json \
  --report summary
```

#### Model Comparison
```bash
python analyze_results.py \
  --results out/model_comparison_*.json \
  --report comparison \
  --sort-by f1_score
```

#### Detailed Statistical Analysis
```bash
python analyze_results.py \
  --results out/*.json \
  --report detailed \
  --format html \
  --output analysis/
```

#### Custom Analysis
```bash
python analyze_results.py \
  --results out/*.json \
  --metrics "accuracy,precision,recall" \
  --filter-model "qwen2.5-vl:7b" \
  --min-success-rate 0.8
```

### Report Types

**Summary Report:**
- Overall performance statistics
- Best/worst performing models
- Execution time analysis
- Success/failure rates

**Comparison Report:**
- Side-by-side model comparison
- Statistical significance testing
- Performance ranking
- Cost-benefit analysis

**Detailed Report:**
- Individual test case results
- Error analysis and patterns
- Performance distribution
- Confidence intervals

**Statistical Report:**
- Hypothesis testing
- Correlation analysis
- Regression modeling
- Trend analysis

## Advanced Features

### Parallel Execution

Control parallelism with the `--workers` parameter:
```bash
# Conservative (good for limited resources)
python run_suite.py --suite my_test.yaml --workers 2

# Aggressive (good for powerful systems)
python run_suite.py --suite my_test.yaml --workers 8

# Auto-detect (recommended)
python run_suite.py --suite my_test.yaml  # Uses 4 workers by default
```

### Resource Management

Monitor resource usage:
```bash
# Test resource allocation
python test_resource_management.py

# Monitor during execution
python monitor_benchmark.py &
python run_suite.py --suite large_benchmark.yaml
```

### Custom Metrics

Create custom evaluation metrics:
```python
from core.metrics_registry import MetricsRegistry, MetricConfig, MetricType

def custom_metric(predictions, ground_truth, **kwargs):
    # Your custom calculation logic
    return custom_score

custom_config = MetricConfig(
    metric_type=MetricType.CUSTOM,
    function=custom_metric,
    description="My custom metric",
    higher_is_better=True
)

metrics_registry.register_metric(custom_config)
```

### Batch Processing

Process large datasets efficiently:
```bash
# Process multiple datasets
for dataset in datasets/*; do
  python run_suite.py \
    --suite production_test.yaml \
    --dataset "$dataset" \
    --output "results/$(basename $dataset)/"
done

# Aggregate results
python analyze_results.py \
  --results results/*/suite_*.json \
  --report comparison
```

## Troubleshooting

### Common Issues

**Ollama Connection Errors:**
```bash
# Check Ollama status
ollama list
systemctl status ollama  # Linux
brew services list | grep ollama  # macOS

# Restart Ollama
ollama serve
```

**GPU Memory Issues:**
```bash
# Check GPU usage
nvidia-smi

# Clear GPU memory
python -c "import torch; torch.cuda.empty_cache()"

# Use CPU fallback
export CUDA_VISIBLE_DEVICES=""
```

**Model Loading Failures:**
```bash
# Check model availability
ollama list
python -c "from transformers import AutoModel; AutoModel.from_pretrained('model-name')"

# Clear model cache
rm -rf ~/.cache/huggingface/
export HF_HOME=/tmp/hf_cache
```

**Permission Errors:**
```bash
# Fix permissions
chmod +x run_suite.py run_matrix.py analyze_results.py

# Install in user space
pip install --user -r requirements.txt
```

### Performance Optimization

**Speed Up Inference:**
- Use quantized models (Ollama)
- Reduce max_tokens
- Increase batch size
- Use GPU acceleration

**Reduce Memory Usage:**
- Use smaller models
- Reduce parallel workers
- Enable gradient checkpointing
- Use CPU offloading

**Optimize Accuracy:**
- Use full-precision models
- Increase max_tokens
- Optimize prompts
- Use ensemble methods

### Getting Help

- **Documentation**: Check other guides in `docs/`
- **Examples**: See `docs/examples/` for sample configurations
- **Issues**: Report bugs at [GitHub Issues](https://github.com/AlterMundi/sai-benchmark/issues)
- **Discussions**: Join community discussions

---

This user guide covers the essential aspects of using SAI-Benchmark. For specific use cases, check the [tutorials](../tutorials/) section or refer to the [API documentation](../api/) for programmatic usage.