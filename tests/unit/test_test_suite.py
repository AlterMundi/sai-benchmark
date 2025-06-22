"""
Unit tests for Test Suite Framework.

Tests cover:
- TestCase, TestSuiteConfig, and TestResult dataclasses
- YAML configuration loading and validation
- Test suite execution and orchestration
- Result aggregation and metrics collection
- Error handling in test execution
- Resource-aware parallel execution
"""

import pytest
import sys
import yaml
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.test_suite import (
    TestCase, TestSuiteConfig, TestResult, TestSuiteResult, 
    TestSuiteRunner
)
from core.metrics_registry import MetricResult
from engines.base_engine import EngineResponse


class TestTestCase:
    """Test TestCase dataclass."""
    
    def test_test_case_creation(self):
        """Test creating TestCase with all fields."""
        test_case = TestCase(
            id="test_001",
            prompt_id="early_fire_json",
            model_id="qwen2.5-vl:7b",
            dataset_path="/data/sequences/fire_001",
            images=["frame_001.jpg", "frame_002.jpg"],
            expected_output={"has_smoke": True, "bbox": [0.5, 0.5, 0.2, 0.3]},
            metadata={"sequence_type": "wildfire", "confidence": 0.9}
        )
        
        assert test_case.id == "test_001"
        assert test_case.prompt_id == "early_fire_json"
        assert test_case.model_id == "qwen2.5-vl:7b"
        assert len(test_case.images) == 2
        assert test_case.expected_output["has_smoke"] is True
        assert test_case.metadata["sequence_type"] == "wildfire"
    
    def test_test_case_minimal(self):
        """Test TestCase with minimal required fields."""
        test_case = TestCase(
            id="minimal_test",
            prompt_id="basic_prompt",
            model_id="test_model",
            dataset_path="/data/test"
        )
        
        assert test_case.id == "minimal_test"
        assert test_case.images == []
        assert test_case.expected_output is None
        assert test_case.metadata == {}


class TestTestSuiteConfig:
    """Test TestSuiteConfig dataclass."""
    
    @pytest.fixture
    def sample_yaml_config(self, temp_dir):
        """Create sample YAML configuration file."""
        config_data = {
            "name": "early_detection_suite",
            "description": "Test suite for early fire detection",
            "version": "2.0",
            "prompts": ["early_fire_json", "wildfire_confidence"],
            "models": ["qwen2.5-vl:7b", "llama3.2-vision:11b"],
            "datasets": ["/data/fire_sequences", "/data/smoke_sequences"],
            "metrics": ["accuracy", "precision", "recall", "f1_score", "latency"],
            "engine_config": {
                "max_tokens": 512,
                "temperature": 0.1,
                "timeout": 30
            },
            "test_config": {
                "iou_threshold": 0.4,
                "confidence_threshold": 0.5,
                "max_workers": 4
            }
        }
        
        yaml_path = temp_dir / "test_suite.yaml"
        with open(yaml_path, 'w') as f:
            yaml.dump(config_data, f)
        
        return yaml_path, config_data
    
    def test_test_suite_config_creation(self):
        """Test creating TestSuiteConfig directly."""
        config = TestSuiteConfig(
            name="test_suite",
            description="Test description",
            prompts=["prompt1", "prompt2"],
            models=["model1", "model2"],
            datasets=["/data/test1", "/data/test2"],
            metrics=["accuracy", "latency"]
        )
        
        assert config.name == "test_suite"
        assert len(config.prompts) == 2
        assert len(config.models) == 2
        assert len(config.datasets) == 2
        assert len(config.metrics) == 2
        assert config.version == "1.0"  # Default
        assert config.engine_config == {}  # Default
        assert config.test_config == {}  # Default
    
    def test_from_yaml_success(self, sample_yaml_config):
        """Test loading configuration from YAML file."""
        yaml_path, expected_data = sample_yaml_config
        
        config = TestSuiteConfig.from_yaml(yaml_path)
        
        assert config.name == expected_data["name"]
        assert config.description == expected_data["description"]
        assert config.version == expected_data["version"]
        assert config.prompts == expected_data["prompts"]
        assert config.models == expected_data["models"]
        assert config.datasets == expected_data["datasets"]
        assert config.metrics == expected_data["metrics"]
        assert config.engine_config == expected_data["engine_config"]
        assert config.test_config == expected_data["test_config"]
    
    def test_from_yaml_missing_file(self, temp_dir):
        """Test loading configuration from non-existent file."""
        non_existent_path = temp_dir / "missing.yaml"
        
        with pytest.raises(FileNotFoundError):
            TestSuiteConfig.from_yaml(non_existent_path)
    
    def test_from_yaml_invalid_yaml(self, temp_dir):
        """Test loading configuration from invalid YAML."""
        invalid_yaml_path = temp_dir / "invalid.yaml"
        with open(invalid_yaml_path, 'w') as f:
            f.write("invalid: yaml: content: [unclosed")
        
        with pytest.raises(yaml.YAMLError):
            TestSuiteConfig.from_yaml(invalid_yaml_path)
    
    def test_from_yaml_missing_required_fields(self, temp_dir):
        """Test loading configuration with missing required fields."""
        incomplete_config = {
            "name": "incomplete",
            # Missing description, prompts, models, datasets, metrics
        }
        
        yaml_path = temp_dir / "incomplete.yaml"
        with open(yaml_path, 'w') as f:
            yaml.dump(incomplete_config, f)
        
        with pytest.raises(TypeError):  # Missing required arguments
            TestSuiteConfig.from_yaml(yaml_path)


class TestTestResult:
    """Test TestResult dataclass."""
    
    @pytest.fixture
    def sample_engine_response(self):
        """Create sample EngineResponse."""
        return EngineResponse(
            content='{"has_smoke": true, "bbox": [0.5, 0.5, 0.2, 0.3]}',
            model="qwen2.5-vl:7b",
            latency=1.25,
            tokens_used=150,
            raw_response={"test": "data"}
        )
    
    def test_test_result_creation(self, sample_engine_response):
        """Test creating TestResult with all fields."""
        test_result = TestResult(
            test_case_id="test_001",
            prompt_id="early_fire_json",
            model_id="qwen2.5-vl:7b",
            engine_response=sample_engine_response,
            parsed_output={"has_smoke": True, "bbox": [0.5, 0.5, 0.2, 0.3]},
            validation_result={"valid": True, "errors": []},
            ground_truth={"has_smoke": True, "bbox": [0.48, 0.52, 0.22, 0.28]},
            metrics={
                "accuracy": MetricResult("accuracy", 1.0),
                "latency": MetricResult("latency", 1.25)
            }
        )
        
        assert test_result.test_case_id == "test_001"
        assert test_result.prompt_id == "early_fire_json"
        assert test_result.model_id == "qwen2.5-vl:7b"
        assert test_result.engine_response == sample_engine_response
        assert test_result.parsed_output["has_smoke"] is True
        assert test_result.validation_result["valid"] is True
        assert test_result.ground_truth["has_smoke"] is True
        assert len(test_result.metrics) == 2
        assert isinstance(test_result.timestamp, datetime)
    
    def test_test_result_to_dict(self, sample_engine_response):
        """Test converting TestResult to dictionary."""
        test_result = TestResult(
            test_case_id="test_002",
            prompt_id="wildfire_confidence",
            model_id="llama3.2-vision:11b",
            engine_response=sample_engine_response,
            parsed_output={"judgment": "Yes", "confidence": 0.85},
            metrics={"accuracy": MetricResult("accuracy", 0.85)}
        )
        
        result_dict = test_result.to_dict()
        
        assert result_dict["test_case_id"] == "test_002"
        assert result_dict["prompt_id"] == "wildfire_confidence"
        assert result_dict["model_id"] == "llama3.2-vision:11b"
        assert "engine_response" in result_dict
        assert result_dict["parsed_output"]["judgment"] == "Yes"
        assert "accuracy" in result_dict["metrics"]
        assert isinstance(result_dict["timestamp"], str)
    
    def test_test_result_minimal(self, sample_engine_response):
        """Test TestResult with minimal fields."""
        test_result = TestResult(
            test_case_id="minimal",
            prompt_id="test_prompt",
            model_id="test_model",
            engine_response=sample_engine_response
        )
        
        assert test_result.parsed_output is None
        assert test_result.validation_result is None
        assert test_result.ground_truth is None
        assert test_result.metrics == {}


class TestTestSuiteResult:
    """Test TestSuiteResult dataclass."""
    
    @pytest.fixture
    def sample_test_suite_config(self):
        """Create sample TestSuiteConfig."""
        return TestSuiteConfig(
            name="sample_suite",
            description="Sample test suite",
            prompts=["prompt1"],
            models=["model1"],
            datasets=["/data/test"],
            metrics=["accuracy"]
        )
    
    @pytest.fixture
    def sample_test_results(self):
        """Create sample TestResult list."""
        engine_response = EngineResponse(
            content='{"result": "test"}',
            model="test_model",
            latency=1.0
        )
        
        return [
            TestResult(
                test_case_id="test_001",
                prompt_id="prompt1",
                model_id="model1",
                engine_response=engine_response,
                metrics={"accuracy": MetricResult("accuracy", 0.9)}
            ),
            TestResult(
                test_case_id="test_002",
                prompt_id="prompt1",
                model_id="model1",
                engine_response=engine_response,
                metrics={"accuracy": MetricResult("accuracy", 0.8)}
            )
        ]
    
    def test_test_suite_result_creation(self, sample_test_suite_config, sample_test_results):
        """Test creating TestSuiteResult."""
        suite_result = TestSuiteResult(
            suite_name="test_suite",
            config=sample_test_suite_config,
            test_results=sample_test_results,
            aggregated_metrics={"accuracy": MetricResult("accuracy", 0.85)},
            execution_time=125.5
        )
        
        assert suite_result.suite_name == "test_suite"
        assert suite_result.config == sample_test_suite_config
        assert len(suite_result.test_results) == 2
        assert suite_result.aggregated_metrics["accuracy"].value == 0.85
        assert suite_result.execution_time == 125.5
        assert isinstance(suite_result.timestamp, datetime)
    
    def test_test_suite_result_to_dict(self, sample_test_suite_config, sample_test_results):
        """Test converting TestSuiteResult to dictionary."""
        suite_result = TestSuiteResult(
            suite_name="dict_test",
            config=sample_test_suite_config,
            test_results=sample_test_results,
            aggregated_metrics={"accuracy": MetricResult("accuracy", 0.75)},
            execution_time=89.2
        )
        
        result_dict = suite_result.to_dict()
        
        assert result_dict["suite_name"] == "dict_test"
        assert "config" in result_dict
        assert len(result_dict["test_results"]) == 2
        assert "aggregated_metrics" in result_dict
        assert result_dict["execution_time"] == 89.2
        assert isinstance(result_dict["timestamp"], str)


class TestTestSuiteRunner:
    """Test TestSuiteRunner functionality."""
    
    @pytest.fixture
    def mock_registries(self):
        """Create mock registries."""
        prompt_registry = Mock()
        model_registry = Mock()
        engine_registry = Mock()
        metrics_registry = Mock()
        resource_manager = Mock()
        
        # Configure prompt registry
        prompt_registry.get_prompt.return_value = Mock(
            id="test_prompt",
            template="Test template",
            output_schema=Mock(type="json")
        )
        prompt_registry.validate_output.return_value = {
            "valid": True,
            "parsed_output": {"has_smoke": True},
            "errors": []
        }
        
        # Configure model registry
        prompt_registry.list_prompts.return_value = [Mock(id="test_prompt")]
        model_registry.list_models.return_value = [Mock(id="test_model")]
        
        # Configure engine registry
        engine_registry.execute_prompt.return_value = EngineResponse(
            content='{"has_smoke": true}',
            model="test_model",
            latency=1.0
        )
        
        # Configure metrics registry
        metrics_registry.calculate_all_metrics.return_value = {
            "accuracy": MetricResult("accuracy", 0.9)
        }
        metrics_registry.aggregate_results.return_value = {
            "accuracy": MetricResult("accuracy", 0.85)
        }
        
        # Configure resource manager
        resource_manager.can_allocate.return_value = True
        resource_manager.allocate.return_value = Mock()  # Context manager
        resource_manager.allocate.__enter__ = Mock(return_value=None)
        resource_manager.allocate.__exit__ = Mock(return_value=None)
        
        return {
            "prompt_registry": prompt_registry,
            "model_registry": model_registry,
            "engine_registry": engine_registry,
            "metrics_registry": metrics_registry,
            "resource_manager": resource_manager
        }
    
    def test_test_suite_runner_initialization(self, mock_registries):
        """Test TestSuiteRunner initialization."""
        runner = TestSuiteRunner(
            prompt_registry=mock_registries["prompt_registry"],
            model_registry=mock_registries["model_registry"],
            engine_registry=mock_registries["engine_registry"],
            metrics_registry=mock_registries["metrics_registry"],
            resource_manager=mock_registries["resource_manager"]
        )
        
        assert runner.prompt_registry == mock_registries["prompt_registry"]
        assert runner.model_registry == mock_registries["model_registry"]
        assert runner.engine_registry == mock_registries["engine_registry"]
        assert runner.metrics_registry == mock_registries["metrics_registry"]
        assert runner.resource_manager == mock_registries["resource_manager"]
    
    @patch('core.test_suite.Path.glob')
    @patch('core.test_suite.Path.exists')
    def test_generate_test_cases(self, mock_exists, mock_glob, mock_registries):
        """Test generating test cases from configuration."""
        # Mock dataset files
        mock_exists.return_value = True
        mock_glob.return_value = [
            Path("/data/test/image1.jpg"),
            Path("/data/test/image2.jpg")
        ]
        
        config = TestSuiteConfig(
            name="test_suite",
            description="Test",
            prompts=["prompt1", "prompt2"],
            models=["model1"],
            datasets=["/data/test"],
            metrics=["accuracy"]
        )
        
        runner = TestSuiteRunner(**mock_registries)
        test_cases = runner.generate_test_cases(config)
        
        # Should generate 2 prompts × 1 model × 1 dataset = 2 test cases
        assert len(test_cases) == 2
        assert all(isinstance(tc, TestCase) for tc in test_cases)
        assert test_cases[0].prompt_id == "prompt1"
        assert test_cases[1].prompt_id == "prompt2"
        assert all(tc.model_id == "model1" for tc in test_cases)
    
    def test_execute_test_case_success(self, mock_registries):
        """Test successful test case execution."""
        config = TestSuiteConfig(
            name="test",
            description="Test",
            prompts=["test_prompt"],
            models=["test_model"],
            datasets=["/data/test"],
            metrics=["accuracy"]
        )
        
        test_case = TestCase(
            id="test_001",
            prompt_id="test_prompt",
            model_id="test_model",
            dataset_path="/data/test",
            images=["test.jpg"]
        )
        
        runner = TestSuiteRunner(**mock_registries)
        result = runner.execute_test_case(test_case, config)
        
        assert isinstance(result, TestResult)
        assert result.test_case_id == "test_001"
        assert result.prompt_id == "test_prompt"
        assert result.model_id == "test_model"
        assert result.validation_result["valid"] is True
        assert "accuracy" in result.metrics
    
    def test_execute_test_case_engine_failure(self, mock_registries):
        """Test test case execution with engine failure."""
        # Configure engine to fail
        mock_registries["engine_registry"].execute_prompt.return_value = EngineResponse(
            content="",
            model="test_model",
            latency=0.0,
            error="Engine failed"
        )
        
        config = TestSuiteConfig(
            name="test",
            description="Test",
            prompts=["test_prompt"],
            models=["test_model"],
            datasets=["/data/test"],
            metrics=["accuracy"]
        )
        
        test_case = TestCase(
            id="test_fail",
            prompt_id="test_prompt",
            model_id="test_model",
            dataset_path="/data/test"
        )
        
        runner = TestSuiteRunner(**mock_registries)
        result = runner.execute_test_case(test_case, config)
        
        assert isinstance(result, TestResult)
        assert not result.engine_response.is_success()
        assert result.engine_response.error == "Engine failed"
    
    def test_run_suite_success(self, mock_registries, temp_dir):
        """Test successful test suite execution."""
        # Create test suite config
        config_data = {
            "name": "integration_test",
            "description": "Integration test suite",
            "prompts": ["test_prompt"],
            "models": ["test_model"],
            "datasets": [str(temp_dir)],
            "metrics": ["accuracy"],
            "test_config": {"max_workers": 1}
        }
        
        config_path = temp_dir / "test_suite.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f)
        
        # Create test images
        (temp_dir / "test1.jpg").touch()
        (temp_dir / "test2.jpg").touch()
        
        runner = TestSuiteRunner(**mock_registries)
        
        with patch('core.test_suite.Path.glob') as mock_glob:
            mock_glob.return_value = [temp_dir / "test1.jpg", temp_dir / "test2.jpg"]
            
            result = runner.run_suite(str(config_path))
        
        assert isinstance(result, TestSuiteResult)
        assert result.suite_name == "integration_test"
        assert len(result.test_results) > 0
        assert "accuracy" in result.aggregated_metrics
        assert result.execution_time > 0
    
    def test_parallel_execution(self, mock_registries, temp_dir):
        """Test parallel test execution."""
        config_data = {
            "name": "parallel_test",
            "description": "Parallel execution test",
            "prompts": ["prompt1", "prompt2"],
            "models": ["model1", "model2"],
            "datasets": [str(temp_dir)],
            "metrics": ["accuracy"],
            "test_config": {"max_workers": 2}
        }
        
        config_path = temp_dir / "parallel_suite.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f)
        
        # Create test images
        (temp_dir / "test.jpg").touch()
        
        runner = TestSuiteRunner(**mock_registries)
        
        with patch('core.test_suite.Path.glob') as mock_glob:
            mock_glob.return_value = [temp_dir / "test.jpg"]
            
            result = runner.run_suite(str(config_path))
        
        # Should have 2 prompts × 2 models = 4 test cases
        assert len(result.test_results) == 4
        assert result.execution_time > 0
    
    def test_error_handling_invalid_config(self, mock_registries, temp_dir):
        """Test error handling with invalid configuration."""
        # Create invalid config (missing required fields)
        invalid_config = {"name": "invalid"}
        
        config_path = temp_dir / "invalid.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(invalid_config, f)
        
        runner = TestSuiteRunner(**mock_registries)
        
        with pytest.raises(TypeError):
            runner.run_suite(str(config_path))
    
    def test_resource_management_integration(self, mock_registries):
        """Test integration with resource manager."""
        config = TestSuiteConfig(
            name="resource_test",
            description="Resource management test",
            prompts=["test_prompt"],
            models=["test_model"],
            datasets=["/data/test"],
            metrics=["accuracy"]
        )
        
        test_case = TestCase(
            id="resource_test",
            prompt_id="test_prompt",
            model_id="test_model",
            dataset_path="/data/test"
        )
        
        # Configure resource manager to deny allocation
        mock_registries["resource_manager"].can_allocate.return_value = False
        
        runner = TestSuiteRunner(**mock_registries)
        
        # Execution should still work but might be queued
        result = runner.execute_test_case(test_case, config)
        assert isinstance(result, TestResult)
        
        # Verify resource manager was consulted
        mock_registries["resource_manager"].can_allocate.assert_called()