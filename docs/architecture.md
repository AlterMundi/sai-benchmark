# SAI-Benchmark Architecture Overview

This document provides a comprehensive overview of the SAI-Benchmark framework architecture, design principles, and component interactions.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    SAI-Benchmark Framework                      │
├─────────────────────────────────────────────────────────────────┤
│                     CLI & User Interface                       │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │
│  │ run_suite.py│  │run_matrix.py│  │  analyze_results.py     │ │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘ │
├─────────────────────────────────────────────────────────────────┤
│                    Test Suite Framework                        │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │              TestSuiteRunner                                │ │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────┐   │ │
│  │  │ Test Cases  │ │ Execution   │ │ Results Collection  │   │ │
│  │  │ Generation  │ │ Management  │ │ & Aggregation       │   │ │
│  │  └─────────────┘ └─────────────┘ └─────────────────────┘   │ │
│  └─────────────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────────────┤
│                      Core Registry System                      │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌───────────┐ │
│  │   Prompt    │ │    Model    │ │   Engine    │ │  Metrics  │ │
│  │  Registry   │ │  Registry   │ │  Registry   │ │ Registry  │ │
│  └─────────────┘ └─────────────┘ └─────────────┘ └───────────┘ │
├─────────────────────────────────────────────────────────────────┤
│                     Resource Management                        │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                  ResourceManager                            │ │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────┐   │ │
│  │  │ GPU/CPU     │ │ Memory      │ │ Service Coordination│   │ │
│  │  │ Allocation  │ │ Management  │ │ (Ollama, APIs)      │   │ │
│  │  └─────────────┘ └─────────────┘ └─────────────────────┘   │ │
│  └─────────────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────────────┤
│                      Engine Backends                           │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────────┐   │
│  │   Ollama    │ │ HuggingFace │ │        OpenAI           │   │
│  │   Engine    │ │   Engine    │ │        Engine           │   │
│  └─────────────┘ └─────────────┘ └─────────────────────────┘   │
├─────────────────────────────────────────────────────────────────┤
│                      Model Layer                               │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────────┐   │
│  │ Qwen 2.5-VL │ │ LLaMA 3.2   │ │ GPT-4V / Gemma / etc.   │   │
│  │             │ │ Vision      │ │                         │   │
│  └─────────────┘ └─────────────┘ └─────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## Core Design Principles

### 1. **Unified Interface**
- Single API for all models and engines
- Consistent data structures across components
- Standardized configuration format

### 2. **Modular Architecture**
- Loosely coupled components
- Plugin-based extension system
- Clear separation of concerns

### 3. **Multi-Dimensional Testing**
- Prompts × Models × Engines matrix evaluation
- Configurable test parameters
- Systematic comparison capabilities

### 4. **Resource Awareness**
- Intelligent resource allocation
- Conflict detection and resolution
- Performance optimization

### 5. **Extensibility**
- Easy addition of new models
- Custom metrics implementation
- Third-party engine integration

## Component Deep Dive

### Core Registry System

The registry system provides centralized management of framework components:

#### Prompt Registry (`core/prompt_registry.py`)
```python
class PromptRegistry:
    """Centralized prompt template management"""
    
    def register_prompt(self, prompt: PromptTemplate) -> None
    def get_prompt(self, prompt_id: str) -> PromptTemplate
    def validate_output(self, prompt_id: str, output: str) -> Dict
    def search_prompts(self, query: str) -> List[PromptTemplate]
```

**Key Features:**
- Template versioning and metadata
- Output schema validation
- Search and filtering capabilities
- JSON and structured text parsing

#### Model Registry (`core/model_registry.py`)
```python
class ModelRegistry:
    """Multi-engine model configuration management"""
    
    def register_model(self, model: ModelConfig) -> None
    def get_model(self, model_id: str) -> ModelConfig
    def list_models(self, **filters) -> List[ModelConfig]
    def get_models_for_use_case(self, use_case: str) -> List[ModelConfig]
```

**Key Features:**
- Capability-based model selection
- Performance characteristics tracking
- Engine compatibility validation
- Cost and latency optimization

#### Engine Registry (`core/engine_registry.py`)
```python
class EngineRegistry:
    """Unified inference backend management"""
    
    def register_engine(self, config: EngineConfig) -> None
    def get_engine(self, model_id: str) -> BaseEngine
    def execute_prompt(self, model_id: str, prompt: str, images: List) -> EngineResponse
    def health_check(self, engine_type: EngineType) -> bool
```

**Key Features:**
- Engine instance lifecycle management
- Health monitoring and service discovery
- Batch processing capabilities
- Error handling and retries

#### Metrics Registry (`core/metrics_registry.py`)
```python
class MetricsRegistry:
    """Configurable evaluation metrics system"""
    
    def register_metric(self, config: MetricConfig) -> None
    def calculate_metric(self, metric_name: str, predictions: List, ground_truth: List) -> MetricResult
    def calculate_all_metrics(self, predictions: List, ground_truth: List) -> Dict[str, MetricResult]
    def aggregate_results(self, results_list: List, method: str) -> Dict[str, MetricResult]
```

**Key Features:**
- Custom metric implementation
- Statistical aggregation methods
- Confidence interval calculation
- Model comparison utilities

### Test Suite Framework

#### Test Suite Runner (`core/test_suite.py`)
```python
class TestSuiteRunner:
    """Orchestrates test execution and result collection"""
    
    def generate_test_cases(self, config: TestSuiteConfig) -> List[TestCase]
    def execute_test_case(self, test_case: TestCase, config: TestSuiteConfig) -> TestResult
    def run_suite(self, suite_path: str) -> TestSuiteResult
```

**Execution Flow:**
1. **Configuration Loading**: Parse YAML test suite definitions
2. **Test Case Generation**: Create test cases from prompt×model×dataset combinations
3. **Resource Allocation**: Acquire necessary system resources
4. **Parallel Execution**: Execute test cases with configurable concurrency
5. **Result Collection**: Aggregate metrics and generate reports
6. **Cleanup**: Release resources and save results

### Resource Management

#### Resource Manager (`core/resource_manager.py`)
```python
class ResourceManager:
    """Hardware resource allocation and conflict management"""
    
    def can_allocate_model(self, model_config: ModelConfig) -> bool
    def allocate_model(self, model_config: ModelConfig) -> ContextManager
    def get_resource_stats(self) -> Dict[str, Any]
```

**Resource Types:**
- **GPU Memory**: VRAM allocation for model loading
- **CPU Cores**: Parallel processing capacity
- **System Memory**: RAM requirements for data processing
- **Service Slots**: Exclusive access to external services (Ollama)
- **Network Bandwidth**: API rate limiting and throughput

**Conflict Resolution:**
- Queue-based allocation for exclusive resources
- Timeout handling for resource acquisition
- Automatic cleanup on process termination

### Engine Implementations

#### Base Engine Interface (`engines/base_engine.py`)
```python
class BaseEngine(ABC):
    """Abstract base class for all inference backends"""
    
    @abstractmethod
    def generate(self, prompt: str, images: List = None, **kwargs) -> str
    
    def generate_with_timing(self, prompt: str, images: List = None, **kwargs) -> EngineResponse
    def health_check(self) -> bool
    def supports_batch(self) -> bool
```

#### Ollama Engine (`engines/ollama_engine.py`)
- HTTP API integration for local models
- Automatic service discovery and health checks
- GGUF quantized model support
- Streaming response handling

#### HuggingFace Engine (`engines/hf_engine.py`)
- Direct Transformers integration
- GPU memory optimization
- Dynamic model loading
- Batch processing support

#### OpenAI Engine (`engines/openai_engine.py`)
- REST API integration
- Rate limiting and retry logic
- Token usage tracking
- Error classification

## Data Flow Architecture

### 1. **Test Execution Flow**
```
YAML Config → TestSuiteRunner → TestCase Generation → Resource Allocation
     ↓
Parallel Execution → Engine Selection → Model Inference → Response Parsing
     ↓
Metric Calculation → Result Aggregation → Report Generation → Persistent Storage
```

### 2. **Model Inference Flow**
```
Input (Prompt + Images) → Engine Registry → Engine Instance → Model Backend
     ↓
Raw Response → Response Parsing → Schema Validation → Structured Output
     ↓
Timing Metadata → Error Handling → EngineResponse Object
```

### 3. **Metrics Calculation Flow**
```
Predictions + Ground Truth → Metrics Registry → Individual Metric Calculation
     ↓
Statistical Analysis → Confidence Intervals → Result Aggregation
     ↓
Comparison Analysis → Report Generation → Visualization Data
```

## Scalability Considerations

### Horizontal Scaling
- **Multi-Process Execution**: Parallel test case processing
- **Distributed Resources**: Multiple GPU allocation
- **Service Coordination**: Multiple Ollama instances

### Vertical Scaling
- **Memory Optimization**: Lazy loading and caching
- **GPU Utilization**: Batch processing and model sharing
- **CPU Efficiency**: Asynchronous I/O and threading

### Performance Monitoring
- Resource utilization tracking
- Execution time profiling
- Memory usage analysis
- Throughput optimization

## Extension Points

### Adding New Models
1. Implement model-specific configuration in `ModelRegistry`
2. Add engine support if needed
3. Create prompt templates optimized for the model
4. Define model-specific capabilities and constraints

### Adding New Engines
1. Implement `BaseEngine` interface
2. Register engine configuration in `EngineRegistry`
3. Handle engine-specific authentication and setup
4. Implement health checks and error handling

### Adding New Metrics
1. Define metric calculation function
2. Register with `MetricsRegistry`
3. Specify aggregation method and properties
4. Add visualization and reporting support

### Adding New Test Types
1. Define test configuration schema
2. Implement test case generation logic
3. Add specialized execution handling
4. Create custom result analysis

## Security Considerations

### Data Protection
- Secure handling of API keys and credentials
- Input sanitization and validation
- Output redaction for sensitive content

### Resource Isolation
- Process-level isolation for model execution
- Memory protection and cleanup
- Network access control

### Audit and Logging
- Comprehensive execution logging
- Performance metrics tracking
- Error reporting and classification

---

This architecture enables SAI-Benchmark to provide a flexible, scalable, and extensible framework for vision-language model evaluation while maintaining simplicity and ease of use.