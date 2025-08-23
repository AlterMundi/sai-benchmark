"""
Test Suite Framework

Template-based test configuration and execution system.
Supports YAML-defined test suites with multi-dimensional testing capabilities.
"""

import yaml
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from pathlib import Path
import json
import time
from datetime import datetime
import itertools
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from .prompt_registry import PromptRegistry, prompt_registry
from .model_registry import ModelRegistry, model_registry  
from .engine_registry import EngineRegistry, engine_registry
from .metrics_registry import MetricsRegistry, metrics_registry, MetricResult
from .resource_manager import ResourceManager, resource_manager, ResourceError
from engines.base_engine import EngineResponse


@dataclass
class TestCase:
    """Individual test case configuration"""
    id: str
    prompt_id: str
    model_id: str
    dataset_path: str
    images: List[str] = field(default_factory=list)
    expected_output: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TestSuiteConfig:
    """Test suite configuration loaded from YAML"""
    name: str
    description: str
    prompts: List[str]
    models: List[str]  
    datasets: List[str]
    metrics: List[str]
    engine_config: Dict[str, Any] = field(default_factory=dict)
    test_config: Dict[str, Any] = field(default_factory=dict)
    version: str = "1.0"
    
    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> 'TestSuiteConfig':
        """Load test suite configuration from YAML file"""
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        
        return cls(**data)


@dataclass
class TestResult:
    """Result of a single test execution"""
    test_case_id: str
    prompt_id: str
    model_id: str
    engine_response: EngineResponse
    images: List[str] = field(default_factory=list)
    dataset_path: str = ""
    sequence_id: str = ""
    parsed_output: Optional[Dict[str, Any]] = None
    validation_result: Optional[Dict[str, Any]] = None
    ground_truth: Optional[Dict[str, Any]] = None
    metrics: Dict[str, MetricResult] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "test_case_id": self.test_case_id,
            "prompt_id": self.prompt_id,
            "model_id": self.model_id,
            "dataset_path": self.dataset_path,
            "sequence_id": self.sequence_id,
            "images": self.images,
            "engine_response": self.engine_response.to_dict(),
            "parsed_output": self.parsed_output,
            "validation_result": self.validation_result,
            "ground_truth": self.ground_truth,
            "metrics": {k: v.to_dict() for k, v in self.metrics.items()},
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class TestSuiteResult:
    """Results of a complete test suite execution"""
    suite_name: str
    config: TestSuiteConfig
    test_results: List[TestResult]
    aggregated_metrics: Dict[str, MetricResult]
    execution_time: float
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "suite_name": self.suite_name,
            "config": {
                "name": self.config.name,
                "description": self.config.description,
                "prompts": self.config.prompts,
                "models": self.config.models,
                "datasets": self.config.datasets,
                "metrics": self.config.metrics,
                "version": self.config.version
            },
            "test_results": [r.to_dict() for r in self.test_results],
            "aggregated_metrics": {k: v.to_dict() for k, v in self.aggregated_metrics.items()},
            "execution_time": self.execution_time,
            "timestamp": self.timestamp.isoformat(),
            "summary": {
                "total_tests": len(self.test_results),
                "successful_tests": len([r for r in self.test_results if r.engine_response.success]),
                "failed_tests": len([r for r in self.test_results if not r.engine_response.success]),
                "avg_latency": sum(r.engine_response.latency_ms for r in self.test_results) / len(self.test_results) if self.test_results else 0
            }
        }


class TestSuiteRunner:
    """Main test suite execution engine"""
    
    def __init__(self,
                 prompt_registry: PromptRegistry = None,
                 model_registry: ModelRegistry = None,
                 engine_registry: EngineRegistry = None,
                 metrics_registry: MetricsRegistry = None):
        
        self.prompt_registry = prompt_registry or globals()['prompt_registry']
        self.model_registry = model_registry or globals()['model_registry']
        self.engine_registry = engine_registry or globals()['engine_registry']  
        self.metrics_registry = metrics_registry or globals()['metrics_registry']
    
    def load_dataset(self, dataset_path: str) -> List[Dict[str, Any]]:
        """Load dataset from path (supports various formats)"""
        dataset_path = Path(dataset_path)
        
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")
        
        if dataset_path.is_file():
            # Single file dataset (JSON)
            if dataset_path.suffix.lower() == '.json':
                with open(dataset_path, 'r') as f:
                    return json.load(f)
            else:
                raise ValueError(f"Unsupported dataset format: {dataset_path.suffix}")
        
        elif dataset_path.is_dir():
            # Directory-based dataset (legacy format)
            return self._load_directory_dataset(dataset_path)
        
        else:
            raise ValueError(f"Invalid dataset path: {dataset_path}")
    
    def _load_directory_dataset(self, dataset_dir: Path) -> List[Dict[str, Any]]:
        """Load legacy directory-based dataset"""
        dataset = []
        
        # Find all subdirectories (sequences)
        for sequence_dir in dataset_dir.iterdir():
            if not sequence_dir.is_dir():
                continue
            
            # Load images from sequence
            images = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                images.extend(sequence_dir.glob(ext))
            
            if not images:
                continue
            
            # Sort images by name
            images.sort()
            
            # Load ground truth
            gt_file = dataset_dir / f"{sequence_dir.name}.txt"
            has_smoke = False
            bbox = [0, 0, 0, 0]
            
            if gt_file.exists():
                with open(gt_file, 'r') as f:
                    content = f.read().strip()
                    has_smoke = len(content) > 0
                    
                    # Try to parse bbox if available
                    if content and ',' in content:
                        try:
                            bbox = [float(x) for x in content.split(',')]
                        except:
                            pass
            
            dataset.append({
                "sequence_id": sequence_dir.name,
                "images": [str(img) for img in images],
                "ground_truth": {
                    "has_smoke": has_smoke,
                    "bbox": bbox
                }
            })
        
        return dataset
    
    def create_test_cases(self, config: TestSuiteConfig) -> List[TestCase]:
        """Generate test cases from configuration"""
        test_cases = []
        
        # Load datasets
        datasets = {}
        for dataset_path in config.datasets:
            datasets[dataset_path] = self.load_dataset(dataset_path)
        
        # Generate all combinations
        test_id = 0
        for prompt_id, model_id, dataset_path in itertools.product(
            config.prompts, config.models, config.datasets
        ):
            dataset = datasets[dataset_path]
            
            for item in dataset:
                test_cases.append(TestCase(
                    id=f"test_{test_id:04d}",
                    prompt_id=prompt_id,
                    model_id=model_id,
                    dataset_path=dataset_path,
                    images=item.get('images', []),
                    expected_output=item.get('ground_truth'),
                    metadata={
                        "sequence_id": item.get('sequence_id', ''),
                        "dataset_path": dataset_path
                    }
                ))
                test_id += 1
        
        return test_cases
    
    def execute_test_case(self, test_case: TestCase, config: TestSuiteConfig) -> TestResult:
        """Execute a single test case with resource management"""
        
        # Get prompt template
        prompt_template = self.prompt_registry.get_prompt(test_case.prompt_id)
        
        # Get model configuration for resource management
        model_config = self.model_registry.get_model(test_case.model_id)
        
        # Execute with resource management
        try:
            with resource_manager.acquire_resources(model_config, test_case.id):
                # Execute prompt on engine
                engine_response = self.engine_registry.execute_prompt(
                    model_id=test_case.model_id,
                    prompt=prompt_template.template,
                    images=test_case.images,
                    **config.engine_config
                )
        except ResourceError as e:
            # Create error response if resources can't be acquired
            engine_response = EngineResponse(
                content="",
                latency_ms=0,
                error=f"Resource allocation failed: {e}"
            )
        
        # Validate and parse response
        parsed_output = None
        validation_result = None
        
        if engine_response.success:
            validation_result = self.prompt_registry.validate_output(
                test_case.prompt_id, 
                engine_response.content
            )
            parsed_output = validation_result.get('parsed_output')
        
        # Calculate metrics
        metrics = {}
        if parsed_output and test_case.expected_output:
            for metric_name in config.metrics:
                try:
                    metric_result = self.metrics_registry.calculate_metric(
                        metric_name=metric_name,
                        predictions=[parsed_output],
                        ground_truth=[test_case.expected_output]
                    )
                    metrics[metric_name] = metric_result
                except Exception as e:
                    # Skip metrics that can't be calculated
                    pass
        
        return TestResult(
            test_case_id=test_case.id,
            prompt_id=test_case.prompt_id,
            model_id=test_case.model_id,
            images=test_case.images,
            dataset_path=test_case.dataset_path,
            sequence_id=test_case.metadata.get('sequence_id', ''),
            engine_response=engine_response,
            parsed_output=parsed_output,
            validation_result=validation_result,
            ground_truth=test_case.expected_output,
            metrics=metrics
        )
    
    def run_suite(self, 
                 config: TestSuiteConfig,
                 max_workers: int = 4,
                 save_results: bool = True,
                 output_dir: str = "out") -> TestSuiteResult:
        """Execute complete test suite with smart resource management"""
        
        print(f"Running test suite: {config.name}")
        print(f"Description: {config.description}")
        
        start_time = time.time()
        
        # Generate test cases
        test_cases = self.create_test_cases(config)
        print(f"Generated {len(test_cases)} test cases")
        
        # Analyze models used in test cases for resource planning
        models_used = set(tc.model_id for tc in test_cases)
        model_configs = [self.model_registry.get_model(mid) for mid in models_used]
        
        # Determine optimal parallelism based on resource constraints
        optimal_workers = resource_manager.get_optimal_parallelism(model_configs)
        can_parallelize = resource_manager.can_run_parallel(model_configs)
        
        # Override max_workers if resource constraints require it
        if not can_parallelize:
            actual_workers = 1
            print(f"Resource constraints detected: forcing sequential execution")
        else:
            actual_workers = min(max_workers, optimal_workers)
            if actual_workers != max_workers:
                print(f"Resource optimization: using {actual_workers} workers instead of {max_workers}")
        
        # Show resource status
        resource_status = resource_manager.get_resource_status()
        print(f"Resource status: GPU={resource_status['can_use_gpu']}, "
              f"Ollama={resource_status['ollama_available']}, "
              f"Workers={actual_workers}")
        
        # Execute test cases
        test_results = []
        
        if actual_workers > 1:
            # Smart parallel execution with resource awareness
            test_results = self._execute_parallel_with_resources(test_cases, config, actual_workers)
        else:
            # Sequential execution
            for test_case in tqdm(test_cases, desc="Running tests"):
                try:
                    result = self.execute_test_case(test_case, config)
                    test_results.append(result)
                except Exception as e:
                    print(f"Test case {test_case.id} failed: {e}")
        
        # Aggregate metrics
        aggregated_metrics = self._aggregate_metrics(test_results, config.metrics)
        
        execution_time = time.time() - start_time
        
        # Create suite result
        suite_result = TestSuiteResult(
            suite_name=config.name,
            config=config,
            test_results=test_results,
            aggregated_metrics=aggregated_metrics,
            execution_time=execution_time
        )
        
        # Save results
        if save_results:
            self._save_results(suite_result, output_dir)
        
        return suite_result
    
    def _execute_parallel_with_resources(self, 
                                       test_cases: List[TestCase], 
                                       config: TestSuiteConfig, 
                                       max_workers: int) -> List[TestResult]:
        """Execute test cases in parallel with resource-aware scheduling"""
        test_results = []
        
        # Group test cases by model to enable better resource scheduling
        model_groups = {}
        for test_case in test_cases:
            if test_case.model_id not in model_groups:
                model_groups[test_case.model_id] = []
            model_groups[test_case.model_id].append(test_case)
        
        # Check if all models can run in parallel
        model_configs = [self.model_registry.get_model(mid) for mid in model_groups.keys()]
        
        if len(model_configs) == 1 or resource_manager.can_run_parallel(model_configs):
            # All models are compatible - use standard parallel execution
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_test = {
                    executor.submit(self.execute_test_case, test_case, config): test_case
                    for test_case in test_cases
                }
                
                for future in tqdm(as_completed(future_to_test), total=len(test_cases), desc="Running tests"):
                    test_case = future_to_test[future]
                    try:
                        result = future.result()
                        test_results.append(result)
                    except Exception as e:
                        print(f"Test case {test_case.id} failed: {e}")
        else:
            # Models have resource conflicts - execute model groups sequentially, tests within groups in parallel
            print("Resource conflicts detected: using hybrid sequential/parallel execution")
            
            for model_id, group_test_cases in model_groups.items():
                print(f"Processing {len(group_test_cases)} test cases for model {model_id}")
                
                # Check if this model can run multiple tests in parallel
                model_config = self.model_registry.get_model(model_id)
                if resource_manager.can_run_parallel([model_config]):
                    # This model type can handle parallel execution
                    with ThreadPoolExecutor(max_workers=max_workers) as executor:
                        future_to_test = {
                            executor.submit(self.execute_test_case, test_case, config): test_case
                            for test_case in group_test_cases
                        }
                        
                        for future in tqdm(as_completed(future_to_test), 
                                         total=len(group_test_cases), 
                                         desc=f"Model {model_id}"):
                            test_case = future_to_test[future]
                            try:
                                result = future.result()
                                test_results.append(result)
                            except Exception as e:
                                print(f"Test case {test_case.id} failed: {e}")
                else:
                    # This model needs sequential execution
                    for test_case in tqdm(group_test_cases, desc=f"Model {model_id} (sequential)"):
                        try:
                            result = self.execute_test_case(test_case, config)
                            test_results.append(result)
                        except Exception as e:
                            print(f"Test case {test_case.id} failed: {e}")
        
        return test_results
    
    def _aggregate_metrics(self, 
                          test_results: List[TestResult],
                          metric_names: List[str]) -> Dict[str, MetricResult]:
        """Aggregate metrics across all test results"""
        
        aggregated = {}
        
        for metric_name in metric_names:
            # Collect all values for this metric
            metric_results = []
            for result in test_results:
                if metric_name in result.metrics:
                    metric_results.append({metric_name: result.metrics[metric_name]})
            
            if metric_results:
                # Use metrics registry to aggregate
                agg_result = self.metrics_registry.aggregate_results(metric_results)
                if metric_name in agg_result:
                    aggregated[metric_name] = agg_result[metric_name]
        
        return aggregated
    
    def _save_results(self, suite_result: TestSuiteResult, output_dir: str):
        """Save test suite results to files"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        timestamp = suite_result.timestamp.strftime("%Y%m%d_%H%M%S")
        base_name = f"suite_{suite_result.suite_name}_{timestamp}"
        
        # Save full results as JSON
        results_file = output_path / f"{base_name}_results.json"
        with open(results_file, 'w') as f:
            json.dump(suite_result.to_dict(), f, indent=2)
        
        # Save summary report
        summary_file = output_path / f"{base_name}_summary.txt"
        with open(summary_file, 'w') as f:
            self._write_summary_report(suite_result, f)
        
        print(f"Results saved to: {results_file}")
        print(f"Summary saved to: {summary_file}")
    
    def _write_summary_report(self, suite_result: TestSuiteResult, file):
        """Write human-readable summary report"""
        file.write(f"TEST SUITE REPORT\n")
        file.write(f"================\n\n")
        file.write(f"Suite: {suite_result.suite_name}\n")
        file.write(f"Description: {suite_result.config.description}\n")
        file.write(f"Executed: {suite_result.timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n")
        file.write(f"Duration: {suite_result.execution_time:.2f} seconds\n\n")
        
        # Test summary
        summary = suite_result.to_dict()['summary']
        file.write(f"TEST SUMMARY\n")
        file.write(f"------------\n")
        file.write(f"Total tests: {summary['total_tests']}\n")
        file.write(f"Successful: {summary['successful_tests']}\n")
        file.write(f"Failed: {summary['failed_tests']}\n")
        file.write(f"Success rate: {summary['successful_tests']/summary['total_tests']*100:.1f}%\n")
        file.write(f"Average latency: {summary['avg_latency']:.1f}ms\n\n")
        
        # Metrics
        file.write(f"AGGREGATED METRICS\n")
        file.write(f"------------------\n")
        for metric_name, metric_result in suite_result.aggregated_metrics.items():
            file.write(f"{metric_name}: {metric_result.value:.4f}\n")
            if 'std' in metric_result.details:
                file.write(f"  (std: {metric_result.details['std']:.4f})\n")
        
        file.write(f"\nCONFIGURATION\n")
        file.write(f"-------------\n")
        file.write(f"Prompts: {', '.join(suite_result.config.prompts)}\n")
        file.write(f"Models: {', '.join(suite_result.config.models)}\n")
        file.write(f"Datasets: {', '.join(suite_result.config.datasets)}\n")
        file.write(f"Metrics: {', '.join(suite_result.config.metrics)}\n")


def load_suite_config(config_path: Union[str, Path]) -> TestSuiteConfig:
    """Load test suite configuration from YAML file"""
    return TestSuiteConfig.from_yaml(config_path)


def run_test_suite(config_path: Union[str, Path], 
                  max_workers: int = 4,
                  output_dir: str = "out") -> TestSuiteResult:
    """Convenience function to run a test suite from config file"""
    config = load_suite_config(config_path)
    runner = TestSuiteRunner()
    return runner.run_suite(config, max_workers=max_workers, output_dir=output_dir)