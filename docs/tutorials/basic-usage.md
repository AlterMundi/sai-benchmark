# Basic Usage Tutorial

This tutorial will walk you through your first SAI-Benchmark evaluation, from setup to results analysis.

## Prerequisites

- Python 3.8+ installed
- Basic familiarity with command line
- Test images for evaluation (we'll provide sample data)

## Step 1: Environment Setup

### Clone and Install

```bash
# Clone the repository
git clone https://github.com/AlterMundi/sai-benchmark.git
cd sai-benchmark

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Verify Installation

```bash
# Quick validation
python validate_tests.py
```

Expected output:
```
SAI-Benchmark Core Component Validation
==================================================
Testing PromptRegistry...
✅ OutputSchema creation works
✅ PromptTemplate creation works
...
🎉 All core component tests passed!
```

## Step 2: Setup a Model Backend

For this tutorial, we'll use Ollama (local inference) as it's the easiest to set up.

### Install and Configure Ollama

```bash
# Install Ollama (Linux/macOS)
curl -fsSL https://ollama.ai/install.sh | sh

# Pull a vision model
ollama pull qwen2.5-vl:7b

# Start Ollama service
ollama serve
```

### Verify Model Access

```bash
# Test model availability
ollama list
```

You should see `qwen2.5-vl:7b` in the list.

## Step 3: Prepare Test Data

### Create Sample Dataset

Let's create a simple test dataset:

```bash
# Create directory structure
mkdir -p sample_data/fire_test
mkdir -p sample_data/no_fire_test

# For this tutorial, you can use any images
# Copy some images to test directories
# (In practice, you'd use your actual fire/smoke detection dataset)
```

### Create Ground Truth Files

```bash
# Create ground truth for fire images (example with bounding box)
echo "0.5,0.5,0.2,0.3" > sample_data/fire_test.txt

# Create ground truth for no-fire images (empty file)
touch sample_data/no_fire_test.txt
```

**Ground Truth Format:**
- Empty file = no fire/smoke detected
- Non-empty file = fire/smoke detected with bounding box
- Bounding box format: `x_center,y_center,width,height` (normalized 0-1)

## Step 4: Create Your First Test Suite

Create a file called `my_first_test.yaml`:

```yaml
name: "my_first_fire_detection_test"
description: "Tutorial: Basic fire detection evaluation"
version: "1.0"

# Define what to test
prompts:
  - "early_fire_json"  # Built-in prompt for fire detection

models:
  - "qwen2.5-vl:7b"   # Ollama model we installed

datasets:
  - "sample_data"     # Our test data directory

# Metrics to calculate
metrics:
  - "accuracy"
  - "precision"
  - "recall"
  - "latency"

# Model configuration
engine_config:
  max_tokens: 512
  temperature: 0.1

# Test execution settings
test_config:
  max_workers: 1      # Start with single worker
  iou_threshold: 0.4  # Bounding box overlap threshold
```

## Step 5: Run Your First Test

Execute the test suite:

```bash
python run_suite.py --suite my_first_test.yaml --verbose
```

### Understanding the Output

You'll see output like:
```
Starting test suite execution: my_first_fire_detection_test
Loaded 1 prompts, 1 models, 1 datasets
Generating test cases...
Generated 1 test cases

Executing test cases with 1 workers...
Processing test case: test_001 (early_fire_json + qwen2.5-vl:7b)
✅ Test case test_001 completed successfully

Suite execution completed in 15.3 seconds
Results saved to: out/suite_my_first_fire_detection_test_20241215_143052_results.json
```

## Step 6: Analyze Results

### View Basic Results

```bash
python analyze_results.py \
  --results out/suite_my_first_fire_detection_test_*.json \
  --report summary
```

### Understanding the Report

The summary report will show:

```
SAI-Benchmark Results Summary
============================

Suite: my_first_fire_detection_test
Execution Time: 15.3 seconds
Total Test Cases: 1

Model Performance:
├── qwen2.5-vl:7b
    ├── Accuracy: 0.85 (85%)
    ├── Precision: 0.80
    ├── Recall: 0.90
    └── Average Latency: 1.25s

Success Rate: 100% (1/1 tests completed)
```

### Detailed Results

For more detailed analysis:

```bash
python analyze_results.py \
  --results out/suite_my_first_fire_detection_test_*.json \
  --report detailed \
  --verbose
```

## Step 7: Expand Your Testing

### Test Multiple Models

If you have multiple backends available:

```yaml
# Update my_first_test.yaml
models:
  - "qwen2.5-vl:7b"           # Ollama
  - "qwen2.5-vl-7b-hf"        # HuggingFace (if available)
  - "gpt-4o"                  # OpenAI (if API key set)
```

### Test Multiple Prompts

```yaml
# Update my_first_test.yaml
prompts:
  - "early_fire_json"         # JSON output format
  - "wildfire_confidence"     # Confidence-based output
```

### Matrix Testing

For systematic comparison:

```bash
python run_matrix.py \
  --prompts "early_fire_json,wildfire_confidence" \
  --models "qwen2.5-vl:7b" \
  --engines "ollama" \
  --dataset sample_data
```

## Step 8: Compare Different Configurations

### Run Comparative Tests

Create `comparison_test.yaml`:

```yaml
name: "prompt_comparison"
description: "Compare different prompt strategies"

prompts:
  - "early_fire_json"
  - "wildfire_confidence"
  - "detailed_sequence_analysis"

models:
  - "qwen2.5-vl:7b"

datasets:
  - "sample_data"

metrics:
  - "accuracy"
  - "precision"
  - "recall"
  - "f1_score"
  - "latency"
```

### Generate Comparison Report

```bash
python run_suite.py --suite comparison_test.yaml

python analyze_results.py \
  --results out/suite_prompt_comparison_*.json \
  --report comparison \
  --sort-by f1_score
```

## Step 9: Understanding Key Metrics

### Accuracy
- **What it means**: Overall correctness of predictions
- **Range**: 0.0 to 1.0 (higher is better)
- **Good value**: > 0.8 for most applications

### Precision
- **What it means**: Of all positive predictions, how many were correct?
- **Formula**: True Positives / (True Positives + False Positives)
- **Important when**: False positives are costly

### Recall
- **What it means**: Of all actual positives, how many were detected?
- **Formula**: True Positives / (True Positives + False Negatives)
- **Important when**: Missing positives is costly (fire safety!)

### F1 Score
- **What it means**: Harmonic mean of precision and recall
- **Formula**: 2 × (Precision × Recall) / (Precision + Recall)
- **Good balance**: When you need both precision and recall

### Latency
- **What it means**: Average time per inference
- **Units**: Seconds or milliseconds
- **Considerations**: Real-time applications need low latency

## Step 10: Next Steps

### Advanced Features to Explore

1. **Resource Management**
   ```bash
   python test_resource_management.py
   ```

2. **Custom Metrics**
   - Create application-specific evaluation metrics
   - See [Custom Metrics Tutorial](custom-metrics.md)

3. **Large-Scale Testing**
   ```bash
   python run_suite.py --suite large_test.yaml --workers 8
   ```

4. **Performance Optimization**
   - GPU acceleration with HuggingFace
   - Batch processing for efficiency

### Common Next Steps

1. **Add Real Data**: Replace sample data with your actual dataset
2. **Optimize Prompts**: Experiment with different prompt strategies
3. **Model Comparison**: Compare multiple models systematically
4. **Production Deployment**: Set up automated benchmarking

### Learning Resources

- **[Multi-Model Comparison Tutorial](multi-model-comparison.md)**
- **[Advanced Benchmarking Tutorial](advanced-benchmarking.md)**
- **[User Guide](../guides/user-guide.md)** - Comprehensive documentation
- **[API Reference](../api/core.md)** - Programmatic usage

## Troubleshooting

### Common Issues

**"Model not found" Error:**
```bash
# Check available models
ollama list

# Pull missing model
ollama pull qwen2.5-vl:7b
```

**"No test cases generated" Error:**
- Check that dataset directory exists
- Verify images are in supported formats (jpg, png, jpeg)
- Ensure ground truth files are properly formatted

**Performance Issues:**
- Start with `--workers 1` and increase gradually
- Monitor system resources with `htop` or Task Manager
- Use smaller models for initial testing

**Connection Errors:**
```bash
# Restart Ollama
ollama serve

# Check service status
curl http://localhost:11434/api/tags
```

---

**Congratulations!** You've completed your first SAI-Benchmark evaluation. You now know how to:

- Set up the framework
- Configure test suites
- Run evaluations
- Analyze results
- Compare different approaches

Ready for more advanced features? Check out the [Multi-Model Comparison Tutorial](multi-model-comparison.md) or explore the [User Guide](../guides/user-guide.md) for comprehensive documentation.