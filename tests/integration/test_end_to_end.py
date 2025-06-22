"""
End-to-end integration tests for SAI-Benchmark.

Tests cover:
- Complete workflow from test suite configuration to results analysis
- Multi-component interaction testing
- Real-world scenario simulation
- Performance and resource management integration
- Error propagation through the full stack
"""

import pytest
import sys
import json
import yaml
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import time

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.test_suite import TestSuiteRunner, TestSuiteConfig
from core.prompt_registry import PromptRegistry
from core.model_registry import ModelRegistry
from core.engine_registry import EngineRegistry
from core.metrics_registry import MetricsRegistry
from core.resource_manager import ResourceManager
from engines.base_engine import EngineResponse, BaseEngine


class MockVisionEngine(BaseEngine):
    """Mock vision model engine for testing."""
    
    def __init__(self, model_id: str, **kwargs):
        super().__init__(model_id, **kwargs)
        self.response_patterns = {
            "has_smoke_true": '{"has_smoke": true, "bbox": [0.5, 0.5, 0.2, 0.3], "confidence": 0.85}',
            "has_smoke_false": '{"has_smoke": false, "bbox": [0, 0, 0, 0], "confidence": 0.9}',
            "error": "ERROR: Model failed to process image"
        }
        self.call_count = 0
    
    def generate(self, prompt: str, images=None, **kwargs) -> str:
        """Generate mock response based on patterns."""
        self.call_count += 1
        
        # Simulate different response patterns
        if "early_fire" in prompt.lower():
            return self.response_patterns["has_smoke_true"]
        elif "no smoke" in prompt.lower():
            return self.response_patterns["has_smoke_false"]
        else:
            # Default to smoke detection
            return self.response_patterns["has_smoke_true"]
    
    def health_check(self) -> bool:
        return True
    
    def supports_batch(self) -> bool:
        return False


@pytest.mark.integration
class TestEndToEndWorkflow:
    """Test complete end-to-end workflows."""
    
    @pytest.fixture
    def temp_workspace(self):
        """Create temporary workspace with test data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            
            # Create test suite directory structure
            suites_dir = workspace / "suites"
            suites_dir.mkdir()
            
            data_dir = workspace / "test_data"
            data_dir.mkdir()
            
            results_dir = workspace / "results"
            results_dir.mkdir()
            
            # Create test images
            for i in range(5):
                (data_dir / f"test_image_{i:03d}.jpg").touch()
            
            # Create ground truth files
            for i in range(5):
                gt_file = data_dir / f"test_image_{i:03d}.txt"
                if i < 3:  # First 3 have smoke
                    gt_file.write_text("0.5,0.5,0.2,0.3\n")
                else:  # Last 2 don't have smoke
                    gt_file.write_text("")
            
            yield {
                "workspace": workspace,
                "suites_dir": suites_dir,
                "data_dir": data_dir,
                "results_dir": results_dir
            }
    
    @pytest.fixture
    def sample_test_suite(self, temp_workspace):
        """Create a sample test suite configuration."""
        suite_config = {
            "name": "integration_test_suite",
            "description": "End-to-end integration test suite",
            "version": "1.0",
            "prompts": [
                "early_fire_json",
                "wildfire_confidence"
            ],
            "models": [
                "mock-qwen:7b",
                "mock-llama:11b"
            ],
            "datasets": [
                str(temp_workspace["data_dir"])
            ],
            "metrics": [
                "accuracy",
                "precision",
                "recall",
                "f1_score",
                "latency",
                "bbox_iou"
            ],
            "engine_config": {
                "max_tokens": 512,
                "temperature": 0.1,
                "timeout": 30
            },
            "test_config": {
                "max_workers": 2,
                "iou_threshold": 0.4,
                "confidence_threshold": 0.5
            }
        }
        
        suite_file = temp_workspace["suites_dir"] / "integration_test.yaml"
        with open(suite_file, 'w') as f:
            yaml.dump(suite_config, f)
        
        return suite_file, suite_config
    
    @pytest.fixture
    def mock_registries(self):
        """Create and configure mock registries."""
        # Prompt Registry
        prompt_registry = PromptRegistry()
        
        # Model Registry  
        model_registry = ModelRegistry()
        
        # Add mock models
        from core.model_registry import ModelConfig, ModelCapability, EngineType
        
        mock_models = [
            ModelConfig(
                id="mock-qwen:7b",
                name="Mock Qwen 2.5 VL 7B",
                engine=EngineType.OLLAMA,
                model_path="mock-qwen:7b",
                capabilities={ModelCapability.VISION, ModelCapability.JSON_OUTPUT, ModelCapability.BBOX_DETECTION},
                latency_ms=1000,
                cost_per_1k_tokens=0.0
            ),
            ModelConfig(
                id="mock-llama:11b",
                name="Mock LLaMA 3.2 Vision 11B",
                engine=EngineType.OLLAMA,
                model_path="mock-llama:11b",
                capabilities={ModelCapability.VISION, ModelCapability.JSON_OUTPUT},
                latency_ms=1500,
                cost_per_1k_tokens=0.0
            )
        ]
        
        for model in mock_models:
            model_registry.register_model(model)
        
        # Engine Registry
        engine_registry = EngineRegistry(model_registry)
        
        # Override engines with mock implementations
        from core.engine_registry import EngineConfig
        
        mock_engine_config = EngineConfig(
            engine_type=EngineType.OLLAMA,
            engine_class=MockVisionEngine,
            default_config={"temperature": 0.1},
            health_check_url=None
        )
        
        engine_registry.register_engine(mock_engine_config)
        
        # Metrics Registry
        metrics_registry = MetricsRegistry()
        
        # Resource Manager
        resource_manager = ResourceManager()
        
        return {
            "prompt_registry": prompt_registry,
            "model_registry": model_registry,
            "engine_registry": engine_registry,
            "metrics_registry": metrics_registry,
            "resource_manager": resource_manager
        }
    
    def test_complete_workflow_success(self, temp_workspace, sample_test_suite, mock_registries):
        """Test complete successful workflow from configuration to results."""
        suite_file, suite_config = sample_test_suite
        
        # Create test suite runner
        runner = TestSuiteRunner(
            prompt_registry=mock_registries["prompt_registry"],
            model_registry=mock_registries["model_registry"],
            engine_registry=mock_registries["engine_registry"],
            metrics_registry=mock_registries["metrics_registry"],
            resource_manager=mock_registries["resource_manager"]
        )
        
        # Execute test suite
        start_time = time.time()
        result = runner.run_suite(str(suite_file))
        execution_time = time.time() - start_time
        
        # Verify results structure
        assert result.suite_name == "integration_test_suite"
        assert len(result.test_results) > 0
        assert result.execution_time > 0
        assert result.execution_time < execution_time + 1  # Should be close to actual time
        
        # Verify test results
        expected_test_count = len(suite_config["prompts"]) * len(suite_config["models"])
        assert len(result.test_results) == expected_test_count
        
        # Verify each test result has required fields
        for test_result in result.test_results:
            assert test_result.test_case_id is not None
            assert test_result.prompt_id in suite_config["prompts"]
            assert test_result.model_id in suite_config["models"]
            assert test_result.engine_response is not None
            assert test_result.engine_response.is_success()
            assert len(test_result.metrics) > 0
        
        # Verify aggregated metrics
        assert len(result.aggregated_metrics) > 0
        assert "accuracy" in result.aggregated_metrics
        assert "latency" in result.aggregated_metrics
        
        # Verify metric values are reasonable
        accuracy = result.aggregated_metrics["accuracy"].value
        assert 0.0 <= accuracy <= 1.0
        
        latency = result.aggregated_metrics["latency"].value
        assert latency > 0.0
    
    def test_workflow_with_mixed_results(self, temp_workspace, sample_test_suite, mock_registries):
        """Test workflow with mixed success/failure results."""
        suite_file, suite_config = sample_test_suite
        
        # Modify engine to simulate some failures
        original_generate = MockVisionEngine.generate
        
        def mixed_generate(self, prompt, images=None, **kwargs):
            if self.call_count >= 2:  # Fail after 2 successful calls
                raise Exception("Simulated engine failure")
            return original_generate(self, prompt, images, **kwargs)
        
        with patch.object(MockVisionEngine, 'generate', mixed_generate):
            runner = TestSuiteRunner(**mock_registries)
            result = runner.run_suite(str(suite_file))
            
            # Should still complete with partial results
            assert result.suite_name == "integration_test_suite"
            assert len(result.test_results) > 0
            
            # Check that some results succeeded and some failed
            successful_results = [r for r in result.test_results if r.engine_response.is_success()]
            failed_results = [r for r in result.test_results if not r.engine_response.is_success()]
            
            assert len(successful_results) >= 2  # At least 2 should succeed
            assert len(failed_results) >= 1     # At least 1 should fail
    
    def test_workflow_with_resource_constraints(self, temp_workspace, sample_test_suite, mock_registries):
        """Test workflow with resource management constraints."""
        suite_file, suite_config = sample_test_suite
        
        # Mock resource manager to simulate constraints
        resource_manager = mock_registries["resource_manager"]
        original_can_allocate = resource_manager.can_allocate_model
        
        call_count = 0
        def constrained_allocation(model_config, timeout=60):
            nonlocal call_count
            call_count += 1
            
            # Allow first allocation, deny second to simulate resource conflict
            if call_count <= 1:
                return original_can_allocate(model_config, timeout)
            else:
                return False
        
        with patch.object(resource_manager, 'can_allocate_model', constrained_allocation):
            runner = TestSuiteRunner(**mock_registries)
            result = runner.run_suite(str(suite_file))
            
            # Should still complete, but might have different execution patterns
            assert result.suite_name == "integration_test_suite"
            assert result.execution_time > 0
    
    def test_workflow_performance_characteristics(self, temp_workspace, sample_test_suite, mock_registries):
        """Test workflow performance and timing characteristics."""
        suite_file, suite_config = sample_test_suite
        
        # Add latency simulation to mock engine
        original_generate = MockVisionEngine.generate
        
        def timed_generate(self, prompt, images=None, **kwargs):
            # Simulate processing time
            time.sleep(0.05)  # 50ms processing time
            return original_generate(self, prompt, images, **kwargs)
        
        with patch.object(MockVisionEngine, 'generate', timed_generate):
            runner = TestSuiteRunner(**mock_registries)
            
            start_time = time.time()
            result = runner.run_suite(str(suite_file))
            total_time = time.time() - start_time
            
            # Verify timing characteristics
            assert result.execution_time > 0
            assert result.execution_time <= total_time
            
            # With 2 workers and simulated latency, should be faster than sequential
            expected_sequential_time = len(result.test_results) * 0.05
            assert result.execution_time < expected_sequential_time * 0.8  # Should be at least 20% faster
            
            # Verify latency metrics
            latency_results = [r for r in result.test_results if "latency" in r.metrics]
            assert len(latency_results) > 0
            
            for test_result in latency_results:
                recorded_latency = test_result.metrics["latency"].value
                assert recorded_latency >= 0.04  # Should be at least 40ms (close to simulated time)
    
    def test_workflow_error_propagation(self, temp_workspace, sample_test_suite, mock_registries):
        """Test error propagation through the complete workflow."""
        suite_file, suite_config = sample_test_suite
        
        # Test various error scenarios
        runner = TestSuiteRunner(**mock_registries)
        
        # 1. Test invalid suite configuration
        invalid_suite_data = {"name": "invalid", "missing": "required_fields"}
        invalid_suite_file = temp_workspace["suites_dir"] / "invalid.yaml"
        with open(invalid_suite_file, 'w') as f:
            yaml.dump(invalid_suite_data, f)
        
        with pytest.raises(TypeError):  # Should raise error for missing required fields
            runner.run_suite(str(invalid_suite_file))
        
        # 2. Test missing dataset directory
        missing_data_config = suite_config.copy()
        missing_data_config["datasets"] = ["/nonexistent/directory"]
        
        missing_data_file = temp_workspace["suites_dir"] / "missing_data.yaml"
        with open(missing_data_file, 'w') as f:
            yaml.dump(missing_data_config, f)
        
        # Should handle gracefully and continue with available data
        result = runner.run_suite(str(missing_data_file))
        assert result.suite_name == missing_data_config["name"]
    
    def test_workflow_with_real_metrics_calculation(self, temp_workspace, sample_test_suite, mock_registries):
        """Test workflow with realistic metrics calculation."""
        suite_file, suite_config = sample_test_suite
        
        # Configure mock engine to return varied, realistic responses
        response_patterns = [
            '{"has_smoke": true, "bbox": [0.5, 0.5, 0.2, 0.3], "confidence": 0.85}',   # True positive
            '{"has_smoke": false, "bbox": [0, 0, 0, 0], "confidence": 0.9}',            # True negative
            '{"has_smoke": true, "bbox": [0.3, 0.4, 0.15, 0.25], "confidence": 0.75}', # False positive
            '{"has_smoke": false, "bbox": [0, 0, 0, 0], "confidence": 0.8}',            # False negative
            '{"has_smoke": true, "bbox": [0.48, 0.52, 0.22, 0.28], "confidence": 0.9}'  # True positive with good IOU
        ]
        
        call_count = 0
        def realistic_generate(self, prompt, images=None, **kwargs):
            nonlocal call_count
            response = response_patterns[call_count % len(response_patterns)]
            call_count += 1
            return response
        
        with patch.object(MockVisionEngine, 'generate', realistic_generate):
            runner = TestSuiteRunner(**mock_registries)
            result = runner.run_suite(str(suite_file))
            
            # Verify realistic metric calculations
            assert "accuracy" in result.aggregated_metrics
            assert "precision" in result.aggregated_metrics
            assert "recall" in result.aggregated_metrics
            assert "f1_score" in result.aggregated_metrics
            
            # Values should be reasonable for mixed predictions
            accuracy = result.aggregated_metrics["accuracy"].value
            precision = result.aggregated_metrics["precision"].value
            recall = result.aggregated_metrics["recall"].value
            f1_score = result.aggregated_metrics["f1_score"].value
            
            assert 0.0 <= accuracy <= 1.0
            assert 0.0 <= precision <= 1.0
            assert 0.0 <= recall <= 1.0
            assert 0.0 <= f1_score <= 1.0
            
            # F1 should be harmonic mean of precision and recall
            if precision + recall > 0:
                expected_f1 = 2 * (precision * recall) / (precision + recall)
                assert abs(f1_score - expected_f1) < 0.01
    
    def test_workflow_results_serialization(self, temp_workspace, sample_test_suite, mock_registries):
        """Test that workflow results can be properly serialized and analyzed."""
        suite_file, suite_config = sample_test_suite
        
        runner = TestSuiteRunner(**mock_registries)
        result = runner.run_suite(str(suite_file))
        
        # Serialize to JSON
        result_dict = result.to_dict()
        
        # Verify JSON serialization
        json_str = json.dumps(result_dict, indent=2)
        assert len(json_str) > 0
        
        # Verify deserialization
        loaded_data = json.loads(json_str)
        
        assert loaded_data["suite_name"] == result.suite_name
        assert len(loaded_data["test_results"]) == len(result.test_results)
        assert "aggregated_metrics" in loaded_data
        assert "execution_time" in loaded_data
        
        # Save to file
        results_file = temp_workspace["results_dir"] / "integration_test_results.json"
        with open(results_file, 'w') as f:
            json.dump(result_dict, f, indent=2)
        
        assert results_file.exists()
        assert results_file.stat().st_size > 0
        
        # Verify file can be loaded for analysis
        with open(results_file, 'r') as f:
            file_data = json.load(f)
        
        assert file_data["suite_name"] == result.suite_name


@pytest.mark.integration
@pytest.mark.slow
class TestWorkflowScalability:
    """Test workflow scalability with larger datasets."""
    
    def test_workflow_with_large_dataset(self, temp_workspace, mock_registries):
        """Test workflow with larger number of test cases."""
        # Create larger dataset
        large_data_dir = temp_workspace["workspace"] / "large_data"
        large_data_dir.mkdir()
        
        # Create 50 test images
        for i in range(50):
            (large_data_dir / f"image_{i:03d}.jpg").touch()
            
            # Create ground truth
            gt_file = large_data_dir / f"image_{i:03d}.txt"
            if i % 3 == 0:  # Every 3rd image has smoke
                gt_file.write_text("0.5,0.5,0.2,0.3\n")
            else:
                gt_file.write_text("")
        
        # Create test suite with multiple prompts and models
        large_suite_config = {
            "name": "large_scale_test",
            "description": "Large scale test suite",
            "prompts": ["early_fire_json", "wildfire_confidence", "detailed_sequence_analysis"],
            "models": ["mock-qwen:7b", "mock-llama:11b"],
            "datasets": [str(large_data_dir)],
            "metrics": ["accuracy", "precision", "recall", "latency"],
            "test_config": {"max_workers": 4}
        }
        
        suite_file = temp_workspace["workspace"] / "large_suite.yaml"
        with open(suite_file, 'w') as f:
            yaml.dump(large_suite_config, f)
        
        # Execute with timing
        runner = TestSuiteRunner(**mock_registries)
        
        start_time = time.time()
        result = runner.run_suite(str(suite_file))
        execution_time = time.time() - start_time
        
        # Should handle large dataset efficiently
        expected_test_count = 3 * 2  # 3 prompts × 2 models
        assert len(result.test_results) == expected_test_count
        
        # Should complete in reasonable time (parallel execution)
        assert execution_time < 30  # Should complete in under 30 seconds
        
        # Verify all results have metrics
        for test_result in result.test_results:
            assert len(test_result.metrics) > 0
    
    def test_workflow_memory_usage(self, temp_workspace, mock_registries):
        """Test workflow memory usage patterns."""
        import psutil
        import os
        
        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create moderate-sized test suite
        medium_data_dir = temp_workspace["workspace"] / "medium_data"
        medium_data_dir.mkdir()
        
        for i in range(20):
            (medium_data_dir / f"test_{i}.jpg").touch()
            (medium_data_dir / f"test_{i}.txt").write_text("0.5,0.5,0.2,0.3\n" if i % 2 == 0 else "")
        
        suite_config = {
            "name": "memory_test",
            "description": "Memory usage test",
            "prompts": ["early_fire_json"],
            "models": ["mock-qwen:7b"],
            "datasets": [str(medium_data_dir)],
            "metrics": ["accuracy", "latency"]
        }
        
        suite_file = temp_workspace["workspace"] / "memory_test.yaml"
        with open(suite_file, 'w') as f:
            yaml.dump(suite_config, f)
        
        # Execute and monitor memory
        runner = TestSuiteRunner(**mock_registries)
        result = runner.run_suite(str(suite_file))
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_growth = final_memory - initial_memory
        
        # Memory growth should be reasonable (< 100MB for this test size)
        assert memory_growth < 100
        
        # Should successfully complete
        assert len(result.test_results) > 0
        assert result.execution_time > 0


@pytest.mark.integration
@pytest.mark.external
class TestWorkflowWithExternalServices:
    """Test workflows that require external services (optional)."""
    
    @pytest.mark.skipif(True, reason="Requires external Ollama service")
    def test_workflow_with_real_ollama(self):
        """Test workflow with real Ollama service (when available)."""
        # This test would run against real Ollama if available
        # Implementation would be similar to mock tests but with real service
        pass
    
    @pytest.mark.skipif(True, reason="Requires GPU and model weights")
    def test_workflow_with_real_hf_models(self):
        """Test workflow with real HuggingFace models (when available)."""
        # This test would run against real HF models if available
        # Implementation would test actual model loading and inference
        pass