# Configuration Guide

Complete reference for configuring SAI-Benchmark test suites, models, engines, and system settings.

## Table of Contents

- [Test Suite Configuration](#test-suite-configuration)
- [Model Configuration](#model-configuration)
- [Engine Configuration](#engine-configuration)
- [Metrics Configuration](#metrics-configuration)
- [System Configuration](#system-configuration)
- [Environment Variables](#environment-variables)
- [Configuration Examples](#configuration-examples)

## Test Suite Configuration

Test suites are defined in YAML files that specify what to test and how to test it.

### Basic Structure

```yaml
# Required fields
name: "test_suite_name"
description: "Description of what this test suite does"
version: "1.0"

# Test dimensions (required)
prompts: ["prompt1", "prompt2"]
models: ["model1", "model2"]
datasets: ["/path/to/data1", "/path/to/data2"]
metrics: ["accuracy", "precision", "recall"]

# Optional configuration sections
engine_config: {}
test_config: {}
metadata: {}
```

### Prompts Section

Define which prompt templates to use:

```yaml
prompts:
  # Built-in prompts
  - "early_fire_json"              # JSON output for fire detection
  - "wildfire_confidence"          # Confidence-based output
  - "detailed_sequence_analysis"   # Detailed analysis prompt
  
  # Model-specific prompts
  - "llama_format"                 # Optimized for LLaMA models
  - "llava_format"                 # Optimized for LLaVA models
  
  # Custom prompts (must be registered first)
  - "my_custom_prompt"
```

**Available Built-in Prompts:**
- `early_fire_json`: JSON output with bounding boxes
- `wildfire_confidence`: Structured text with confidence scores
- `detailed_sequence_analysis`: Comprehensive scene analysis
- `llama_format`: LLaMA 3.2 Vision optimized
- `llava_format`: LLaVA model optimized
- `minicpm_format`: MiniCPM-V optimized
- `gemma_format`: Gemma 3 27B optimized
- `mistral_format`: Mistral Small 3.1 24B optimized

### Models Section

Specify which models to evaluate:

```yaml
models:
  # Ollama models (local inference)
  - "qwen2.5-vl:7b"
  - "llama3.2-vision:11b"
  - "minicpm-v:8b"
  
  # HuggingFace models (GPU inference)
  - "qwen2.5-vl-7b-hf"
  - "llama3.2-vision-11b-hf"
  
  # OpenAI models (API)
  - "gpt-4o"
  - "gpt-4-vision"
  
  # Custom models
  - "my-custom-model"
```

### Datasets Section

Point to your test data directories:

```yaml
datasets:
  - "/absolute/path/to/dataset1"
  - "relative/path/to/dataset2"
  - "~/home/user/dataset3"
  - "s3://bucket/dataset4"          # Future: S3 support
```

**Dataset Requirements:**
- Directory contains image files (`.jpg`, `.jpeg`, `.png`)
- Ground truth files follow naming convention
- Bounding box format: `x_center,y_center,width,height` (normalized 0-1)

### Metrics Section

Choose evaluation metrics:

```yaml
metrics:
  # Classification metrics
  - "accuracy"
  - "precision"
  - "recall"
  - "f1_score"
  
  # Bounding box metrics
  - "bbox_iou"
  - "bbox_precision"
  - "bbox_recall"
  
  # Performance metrics
  - "latency"
  - "throughput"
  - "cost"
  
  # Quality metrics
  - "confidence"
  - "parse_success_rate"
  - "error_rate"
```

### Engine Configuration

Control model inference parameters:

```yaml
engine_config:
  # Token generation
  max_tokens: 512                   # Maximum output tokens
  temperature: 0.1                  # Randomness (0.0 = deterministic)
  top_p: 0.9                       # Nucleus sampling parameter
  top_k: 50                        # Top-k sampling parameter
  
  # Timeouts and retries
  timeout: 30                      # Request timeout in seconds
  max_retries: 3                   # Number of retry attempts
  retry_delay: 1.0                 # Delay between retries
  
  # Model-specific parameters
  max_image_size: 1024             # Maximum image dimension
  image_quality: "high"            # Image processing quality
  use_cache: true                  # Enable response caching
  
  # Ollama-specific
  num_ctx: 2048                    # Context window size
  num_gpu: 1                       # Number of GPUs to use
  
  # HuggingFace-specific
  device: "auto"                   # Device selection
  torch_dtype: "float16"           # Precision
  trust_remote_code: true          # Allow custom model code
  load_in_8bit: false             # Use 8-bit quantization
  load_in_4bit: false             # Use 4-bit quantization
  
  # OpenAI-specific
  max_concurrent_requests: 10      # Rate limiting
  request_delay: 0.1               # Delay between requests
```

### Test Configuration

Control test execution behavior:

```yaml
test_config:
  # Parallelism
  max_workers: 4                   # Number of parallel workers
  worker_timeout: 300              # Worker timeout in seconds
  
  # Data processing
  fps_sampling: 1.0                # Frame sampling rate for videos
  max_images_per_sequence: 100     # Limit images per test
  image_preprocessing: true        # Enable image preprocessing
  
  # Evaluation parameters
  iou_threshold: 0.4               # IoU threshold for bbox matching
  confidence_threshold: 0.5        # Confidence threshold for predictions
  nms_threshold: 0.5               # Non-maximum suppression threshold
  
  # Output control
  save_intermediate_results: true  # Save results during execution
  save_failed_cases: true          # Save information about failures
  include_raw_responses: false     # Include full model responses
  
  # Error handling
  continue_on_error: true          # Continue after individual failures
  max_failures: 10                 # Stop after this many failures
  failure_threshold: 0.2           # Stop if failure rate exceeds this
```

### Metadata Section

Optional metadata for documentation:

```yaml
metadata:
  author: "researcher_name"
  created: "2024-01-15"
  purpose: "fire_detection_comparison"
  dataset_version: "v2.1"
  tags: ["fire", "detection", "comparison"]
  notes: "Comparing different prompt strategies for fire detection"
  references:
    - "https://paper.url"
    - "doi:10.1000/example"
```

## Model Configuration

### Built-in Model Configurations

Models are pre-configured in the model registry. View available models:

```python
from core.model_registry import model_registry
models = model_registry.list_models()
for model in models:
    print(f"{model.id}: {model.name} ({model.engine.value})")
```

### Custom Model Registration

Add new models programmatically:

```python
from core.model_registry import ModelRegistry, ModelConfig, ModelCapability, EngineType

# Create custom model configuration
custom_model = ModelConfig(
    id="my-custom-model",
    name="My Custom Vision Model",
    engine=EngineType.HUGGINGFACE,
    model_path="username/model-name",
    capabilities={
        ModelCapability.VISION,
        ModelCapability.JSON_OUTPUT,
        ModelCapability.BBOX_DETECTION
    },
    max_tokens=2048,
    max_images=5,
    supports_batch=True,
    cost_per_1k_tokens=0.001,
    latency_ms=500,
    gpu_memory_gb=12.0,
    description="Custom vision model for specialized tasks",
    version="1.0",
    use_cases=["fire_detection", "object_detection"]
)

# Register the model
model_registry = ModelRegistry()
model_registry.register_model(custom_model)
```

### Model Capabilities

Models can have various capabilities:

```python
class ModelCapability(Enum):
    VISION = "vision"                    # Can process images
    TEXT = "text"                       # Text-only processing
    JSON_OUTPUT = "json_output"         # Can output structured JSON
    BBOX_DETECTION = "bbox_detection"   # Can detect bounding boxes
    SEQUENCE_ANALYSIS = "sequence_analysis"  # Can analyze video sequences
    DYNAMIC_RESOLUTION = "dynamic_resolution"  # Supports variable image sizes
    WINDOW_ATTENTION = "window_attention"      # Efficient attention mechanism
```

## Engine Configuration

### Engine Types

```python
class EngineType(Enum):
    OLLAMA = "ollama"           # Local Ollama service
    HUGGINGFACE = "huggingface" # HuggingFace Transformers
    OPENAI = "openai"           # OpenAI API
    ANTHROPIC = "anthropic"     # Anthropic API
    GOOGLE = "google"           # Google AI API
```

### Ollama Engine Configuration

```yaml
engine_config:
  # Ollama-specific parameters
  base_url: "http://localhost:11434"
  timeout: 120
  max_retries: 3
  
  # Model parameters
  num_ctx: 2048              # Context window
  num_gpu: 1                 # GPU layers
  num_thread: 8              # CPU threads
  
  # Generation parameters
  temperature: 0.1
  top_p: 0.9
  top_k: 50
  repeat_penalty: 1.1
  
  # System parameters
  mirostat: 0                # Mirostat sampling
  mirostat_eta: 0.1
  mirostat_tau: 5.0
```

### HuggingFace Engine Configuration

```yaml
engine_config:
  # Device and precision
  device: "auto"             # "cuda", "cpu", "auto"
  torch_dtype: "float16"     # "float32", "float16", "bfloat16"
  device_map: "auto"         # Device mapping strategy
  
  # Quantization
  load_in_8bit: false
  load_in_4bit: false
  bnb_4bit_compute_dtype: "float16"
  bnb_4bit_quant_type: "nf4"
  
  # Generation parameters
  max_new_tokens: 512
  temperature: 0.1
  do_sample: true
  top_p: 0.9
  top_k: 50
  
  # Model loading
  trust_remote_code: true
  use_auth_token: true       # Use HF token if required
  revision: "main"           # Model revision/branch
  
  # Performance optimization
  use_flash_attention_2: false
  torch_compile: false
  cache_dir: "/path/to/cache"
```

### OpenAI Engine Configuration

```yaml
engine_config:
  # API configuration
  api_key: "${OPENAI_API_KEY}"  # Environment variable
  base_url: "https://api.openai.com/v1"
  organization: "org-id"
  
  # Request parameters
  max_tokens: 512
  temperature: 0.1
  top_p: 0.9
  frequency_penalty: 0.0
  presence_penalty: 0.0
  
  # Rate limiting
  max_concurrent_requests: 10
  request_delay: 0.1
  max_retries: 3
  retry_delay: 1.0
  
  # Image parameters
  max_image_size: 2048
  image_quality: "high"      # "low", "high", "auto"
  image_detail: "auto"       # "low", "high", "auto"
```

## Metrics Configuration

### Built-in Metrics

Each metric has configurable parameters:

```python
# Accuracy metric
accuracy_config = MetricConfig(
    metric_type=MetricType.ACCURACY,
    function=calculate_accuracy,
    description="Overall prediction accuracy",
    higher_is_better=True,
    requires_ground_truth=True,
    aggregation_method="mean"
)

# Custom metric with parameters
custom_metric_config = MetricConfig(
    metric_type=MetricType.CUSTOM,
    function=lambda pred, gt, threshold=0.5: custom_calculation(pred, gt, threshold),
    description="Custom metric with threshold",
    parameters={"threshold": 0.5},
    higher_is_better=True
)
```

### Custom Metrics

Register custom evaluation metrics:

```python
from core.metrics_registry import MetricsRegistry, MetricConfig, MetricType

def weighted_accuracy(predictions, ground_truth, weights=None, **kwargs):
    """Custom weighted accuracy metric"""
    if weights is None:
        weights = [1.0] * len(predictions)
    
    correct = 0
    total_weight = 0
    
    for pred, gt, weight in zip(predictions, ground_truth, weights):
        if pred.get("has_smoke") == gt.get("has_smoke"):
            correct += weight
        total_weight += weight
    
    return correct / total_weight if total_weight > 0 else 0.0

# Register custom metric
metrics_registry = MetricsRegistry()
custom_config = MetricConfig(
    metric_type=MetricType.CUSTOM,
    function=weighted_accuracy,
    description="Weighted accuracy metric",
    higher_is_better=True,
    requires_ground_truth=True,
    parameters={"weights": None}
)

metrics_registry.register_metric("weighted_accuracy", custom_config)
```

## System Configuration

### Resource Management

Configure system resource allocation:

```python
# Resource manager configuration
resource_config = {
    "gpu_memory_limit": 16.0,      # GB
    "cpu_core_limit": 8,           # Number of cores
    "system_memory_limit": 32.0,   # GB
    "max_concurrent_models": 2,    # Simultaneous models
    "allocation_timeout": 60,      # Seconds
    "cleanup_interval": 300        # Seconds
}
```

### Logging Configuration

```python
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('sai_benchmark.log'),
        logging.StreamHandler()
    ]
)

# Component-specific logging levels
logging.getLogger('core.engine_registry').setLevel(logging.DEBUG)
logging.getLogger('engines.ollama_engine').setLevel(logging.WARNING)
```

## Environment Variables

### Authentication

```bash
# OpenAI
export OPENAI_API_KEY="your_api_key"
export OPENAI_ORG_ID="your_org_id"

# Anthropic
export ANTHROPIC_API_KEY="your_api_key"

# Google
export GOOGLE_API_KEY="your_api_key"
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/credentials.json"

# HuggingFace
export HF_TOKEN="your_hf_token"
export HF_HOME="/path/to/hf/cache"
```

### Service Configuration

```bash
# Ollama
export OLLAMA_HOST="http://localhost:11434"
export OLLAMA_MODELS="/path/to/ollama/models"

# CUDA
export CUDA_VISIBLE_DEVICES="0,1"  # Use specific GPUs
export CUDA_LAUNCH_BLOCKING=1      # For debugging

# Parallel processing
export OMP_NUM_THREADS=8           # OpenMP threads
export MKL_NUM_THREADS=8           # Intel MKL threads
```

### Framework Configuration

```bash
# SAI-Benchmark specific
export SAI_CONFIG_PATH="/path/to/config"
export SAI_CACHE_DIR="/path/to/cache"
export SAI_OUTPUT_DIR="/path/to/output"
export SAI_LOG_LEVEL="INFO"        # DEBUG, INFO, WARNING, ERROR
export SAI_MAX_WORKERS=4           # Default parallelism
```

## Configuration Examples

### Production Fire Detection

```yaml
name: "production_fire_detection"
description: "Production-ready fire detection benchmark"
version: "2.0"

prompts:
  - "early_fire_json"
  - "wildfire_confidence"

models:
  - "qwen2.5-vl:7b"
  - "llama3.2-vision:11b"

datasets:
  - "/data/fire_sequences_train"
  - "/data/fire_sequences_validation"
  - "/data/fire_sequences_test"

metrics:
  - "accuracy"
  - "precision"
  - "recall"
  - "f1_score"
  - "bbox_iou"
  - "latency"

engine_config:
  max_tokens: 512
  temperature: 0.0        # Deterministic for production
  timeout: 10             # Fast response required
  max_retries: 2

test_config:
  max_workers: 8
  iou_threshold: 0.5
  confidence_threshold: 0.7
  continue_on_error: false  # Strict for production
  save_failed_cases: true

metadata:
  author: "fire_safety_team"
  purpose: "production_validation"
  critical: true
```

### Research Comparison

```yaml
name: "research_model_comparison"
description: "Comprehensive model comparison for research"
version: "1.0"

prompts:
  - "early_fire_json"
  - "wildfire_confidence"
  - "detailed_sequence_analysis"
  - "llama_format"
  - "gemma_format"

models:
  - "qwen2.5-vl:7b"
  - "qwen2.5-vl-7b-hf"
  - "llama3.2-vision:11b"
  - "gemma3-27b-vision"
  - "gpt-4o"

datasets:
  - "/research/datasets/fire_detection_benchmark"

metrics:
  - "accuracy"
  - "precision"
  - "recall"
  - "f1_score"
  - "bbox_iou"
  - "bbox_precision"
  - "bbox_recall"
  - "latency"
  - "throughput"
  - "confidence"

engine_config:
  max_tokens: 1024
  temperature: 0.1
  timeout: 60
  max_retries: 3

test_config:
  max_workers: 4
  iou_threshold: 0.4
  save_intermediate_results: true
  include_raw_responses: true

metadata:
  author: "research_team"
  purpose: "model_comparison_study"
  paper: "Vision Models for Fire Detection: A Comparative Study"
```

### Quick Development Test

```yaml
name: "dev_quick_test"
description: "Quick test for development"
version: "dev"

prompts: ["early_fire_json"]
models: ["qwen2.5-vl:7b"]
datasets: ["sample_data"]
metrics: ["accuracy", "latency"]

engine_config:
  max_tokens: 256
  temperature: 0.1
  timeout: 30

test_config:
  max_workers: 1
  max_images_per_sequence: 5
  continue_on_error: true
```

---

This configuration guide covers all aspects of setting up and customizing SAI-Benchmark for your specific needs. For implementation details, see the [API documentation](api/core.md) or check out [configuration examples](examples/configurations/).