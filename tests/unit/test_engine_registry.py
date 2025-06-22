"""
Unit tests for EngineRegistry system.

Tests cover:
- Engine registration and configuration
- Engine instance creation and management
- Prompt execution (single and batch)
- Health checks and requirement validation
- Engine selection for use cases
- Error handling and cleanup
"""

import pytest
from pathlib import Path
from typing import List, Union, Optional
from unittest.mock import Mock, patch, MagicMock
from dataclasses import dataclass

# Since the codebase structure uses absolute imports from root
import sys
sys.path.insert(0, '/root/sai-benchmark')

from core.engine_registry import EngineRegistry, EngineConfig
from core.model_registry import ModelRegistry, ModelConfig, ModelCapability, EngineType
from engines.base_engine import BaseEngine, EngineResponse


class MockEngine(BaseEngine):
    """Mock engine for testing."""
    
    def __init__(self, model_id: str, **kwargs):
        super().__init__(model_id, **kwargs)
        self.generate_called = False
        self.health_check_called = False
        self.batch_support = kwargs.get('supports_batch', False)
    
    def generate(self, prompt: str, images: List[Union[str, Path]] = None, **kwargs) -> str:
        self.generate_called = True
        return '{"has_smoke": true, "bbox": [0.5, 0.5, 0.2, 0.3]}'
    
    def health_check(self) -> bool:
        self.health_check_called = True
        return True
    
    def supports_batch(self) -> bool:
        return self.batch_support


class MockBatchEngine(MockEngine):
    """Mock engine with batch support."""
    
    def __init__(self, model_id: str, **kwargs):
        super().__init__(model_id, **kwargs)
        self.batch_support = True


class TestEngineResponse:
    """Test EngineResponse dataclass functionality."""
    
    def test_engine_response_creation(self):
        """Test creating EngineResponse with all fields."""
        response = EngineResponse(
            content='{"result": "test"}',
            model="test-model",
            latency=0.5,
            tokens_used=100,
            raw_response={"test": "data"},
            error=None
        )
        
        assert response.content == '{"result": "test"}'
        assert response.latency == 0.5
        assert response.is_success() is True
    
    def test_engine_response_with_error(self):
        """Test EngineResponse with error."""
        response = EngineResponse(
            content="",
            model="test-model",
            latency=0.0,
            tokens_used=0,
            raw_response=None,
            error="Model failed to generate"
        )
        
        assert response.is_success() is False
        assert response.error == "Model failed to generate"
    
    def test_engine_response_json_parsing(self):
        """Test JSON parsing from response."""
        response = EngineResponse(
            content='{"has_smoke": true, "confidence": 0.85}',
            model="test-model",
            latency=1.0
        )
        
        parsed = response.get_json()
        assert parsed is not None
        assert parsed["has_smoke"] is True
        assert parsed["confidence"] == 0.85
    
    def test_engine_response_invalid_json(self):
        """Test handling invalid JSON in response."""
        response = EngineResponse(
            content="This is not JSON",
            model="test-model",
            latency=1.0
        )
        
        parsed = response.get_json()
        assert parsed is None


class TestBaseEngine:
    """Test BaseEngine abstract class."""
    
    def test_base_engine_initialization(self):
        """Test BaseEngine initialization."""
        engine = MockEngine("test-model", temperature=0.1, max_tokens=512)
        
        assert engine.model_id == "test-model"
        assert engine.config["temperature"] == 0.1
        assert engine.config["max_tokens"] == 512
    
    def test_generate_with_timing(self):
        """Test generate_with_timing method."""
        engine = MockEngine("test-model")
        
        response = engine.generate_with_timing("Test prompt")
        
        assert isinstance(response, EngineResponse)
        assert response.content == '{"has_smoke": true, "bbox": [0.5, 0.5, 0.2, 0.3]}'
        assert response.model == "test-model"
        assert response.latency > 0
        assert engine.generate_called is True
    
    def test_generate_with_timing_error_handling(self):
        """Test error handling in generate_with_timing."""
        engine = MockEngine("test-model")
        engine.generate = Mock(side_effect=Exception("Generation failed"))
        
        response = engine.generate_with_timing("Test prompt")
        
        assert response.is_success() is False
        assert "Generation failed" in response.error
        assert response.content == ""
    
    def test_validate_inputs(self):
        """Test input validation."""
        engine = MockEngine("test-model")
        
        # Valid inputs
        result = engine.validate_inputs("Valid prompt", ["image1.jpg", "image2.png"])
        assert result is True
        
        # Empty prompt
        result = engine.validate_inputs("", ["image.jpg"])
        assert result is False
        
        # Invalid image extensions
        result = engine.validate_inputs("Prompt", ["document.pdf", "image.bmp"])
        assert result is False


class TestEngineConfig:
    """Test EngineConfig dataclass."""
    
    def test_engine_config_creation(self):
        """Test creating EngineConfig with all fields."""
        config = EngineConfig(
            engine_type=EngineType.OLLAMA,
            engine_class=MockEngine,
            default_config={"base_url": "http://localhost:11434"},
            health_check_url="http://localhost:11434/api/tags",
            requirements=["requests", "pillow"]
        )
        
        assert config.engine_type == EngineType.OLLAMA
        assert config.engine_class == MockEngine
        assert "base_url" in config.default_config
        assert len(config.requirements) == 2
    
    def test_engine_config_defaults(self):
        """Test EngineConfig with default values."""
        config = EngineConfig(
            engine_type=EngineType.HUGGINGFACE,
            engine_class=MockEngine,
            default_config={}
        )
        
        assert config.health_check_url is None
        assert config.requirements == []


class TestEngineRegistry:
    """Test EngineRegistry functionality."""
    
    @pytest.fixture
    def mock_model_registry(self):
        """Create a mock model registry."""
        registry = Mock(spec=ModelRegistry)
        
        # Create test models
        test_model = ModelConfig(
            id="test-model",
            name="Test Model",
            engine=EngineType.OLLAMA,
            model_path="test/path",
            capabilities={ModelCapability.VISION, ModelCapability.JSON_OUTPUT},
            latency_ms=1000,
            cost_per_1k_tokens=0.0
        )
        
        batch_model = ModelConfig(
            id="batch-model",
            name="Batch Model",
            engine=EngineType.HUGGINGFACE,
            model_path="batch/path",
            capabilities={ModelCapability.VISION},
            supports_batch=True
        )
        
        registry.get_model.side_effect = lambda model_id: {
            "test-model": test_model,
            "batch-model": batch_model
        }.get(model_id)
        
        registry.get_engine_models.return_value = [test_model]
        registry.get_models_for_use_case.return_value = [test_model, batch_model]
        
        return registry
    
    @pytest.fixture
    def engine_registry(self, mock_model_registry):
        """Create an engine registry with mock model registry."""
        with patch('core.engine_registry.ModelRegistry', return_value=mock_model_registry):
            registry = EngineRegistry(model_registry=mock_model_registry)
            
            # Register mock engines
            registry.register_engine(EngineConfig(
                engine_type=EngineType.OLLAMA,
                engine_class=MockEngine,
                default_config={"temperature": 0.1}
            ))
            
            registry.register_engine(EngineConfig(
                engine_type=EngineType.HUGGINGFACE,
                engine_class=MockBatchEngine,
                default_config={"device": "cuda"}
            ))
            
            return registry
    
    def test_registry_initialization(self, mock_model_registry):
        """Test engine registry initialization."""
        registry = EngineRegistry(model_registry=mock_model_registry)
        assert registry.model_registry == mock_model_registry
        assert len(registry.engines) >= 2  # At least builtin engines
        assert len(registry.active_engines) == 0
    
    def test_register_engine(self, engine_registry):
        """Test registering a new engine."""
        config = EngineConfig(
            engine_type=EngineType.OPENAI,
            engine_class=MockEngine,
            default_config={"api_key": "test"}
        )
        
        engine_registry.register_engine(config)
        
        assert EngineType.OPENAI in engine_registry.engines
        assert engine_registry.engines[EngineType.OPENAI] == config
    
    def test_get_engine_new_instance(self, engine_registry):
        """Test getting a new engine instance."""
        engine = engine_registry.get_engine("test-model")
        
        assert isinstance(engine, MockEngine)
        assert engine.model_id == "test-model"
        assert "test-model" in engine_registry.active_engines
    
    def test_get_engine_cached_instance(self, engine_registry):
        """Test getting cached engine instance."""
        engine1 = engine_registry.get_engine("test-model")
        engine2 = engine_registry.get_engine("test-model")
        
        assert engine1 is engine2  # Same instance
        assert len(engine_registry.active_engines) == 1
    
    def test_get_engine_with_custom_config(self, engine_registry):
        """Test getting engine with custom configuration."""
        engine = engine_registry.get_engine("test-model", temperature=0.5, custom_param="value")
        
        assert engine.config["temperature"] == 0.5
        assert engine.config["custom_param"] == "value"
    
    def test_get_engine_unregistered_type(self, engine_registry, mock_model_registry):
        """Test getting engine for unregistered engine type."""
        # Create a model with unregistered engine type
        unregistered_model = ModelConfig(
            id="unregistered",
            name="Unregistered",
            engine=EngineType.ANTHROPIC,
            model_path="test",
            capabilities={ModelCapability.TEXT}
        )
        mock_model_registry.get_model.return_value = unregistered_model
        
        with pytest.raises(ValueError, match="Engine type .* not registered"):
            engine_registry.get_engine("unregistered")
    
    def test_execute_prompt(self, engine_registry):
        """Test executing a single prompt."""
        response = engine_registry.execute_prompt(
            "test-model",
            "Analyze this image",
            images=["test.jpg"]
        )
        
        assert isinstance(response, EngineResponse)
        assert response.is_success()
        assert response.model == "test-model"
    
    def test_execute_batch_without_support(self, engine_registry):
        """Test batch execution on engine without batch support."""
        prompts = ["Prompt 1", "Prompt 2", "Prompt 3"]
        image_lists = [["img1.jpg"], ["img2.jpg"], ["img3.jpg"]]
        
        responses = engine_registry.execute_batch(
            "test-model",
            prompts,
            image_lists
        )
        
        assert len(responses) == 3
        assert all(isinstance(r, EngineResponse) for r in responses)
        assert all(r.is_success() for r in responses)
    
    def test_execute_batch_with_support(self, engine_registry):
        """Test batch execution on engine with batch support."""
        prompts = ["Prompt 1", "Prompt 2"]
        
        # Should raise NotImplementedError as batch processing is not implemented
        with pytest.raises(NotImplementedError, match="Batch processing not yet implemented"):
            engine_registry.execute_batch("batch-model", prompts)
    
    def test_health_check_url_based(self, engine_registry):
        """Test URL-based health check."""
        # Add health check URL to engine config
        engine_registry.engines[EngineType.OLLAMA].health_check_url = "http://localhost:11434/api/tags"
        
        with patch('requests.get') as mock_get:
            mock_get.return_value.status_code = 200
            
            result = engine_registry.health_check(EngineType.OLLAMA)
            assert result is True
            
            mock_get.return_value.status_code = 500
            result = engine_registry.health_check(EngineType.OLLAMA)
            assert result is False
    
    def test_health_check_instance_based(self, engine_registry):
        """Test instance-based health check."""
        # Remove health check URL to force instance-based check
        engine_registry.engines[EngineType.OLLAMA].health_check_url = None
        
        result = engine_registry.health_check(EngineType.OLLAMA)
        assert result is True  # MockEngine always returns True
    
    def test_list_available_engines(self, engine_registry):
        """Test listing available engines."""
        engines = engine_registry.list_available_engines()
        
        assert EngineType.OLLAMA in engines
        assert EngineType.HUGGINGFACE in engines
        assert len(engines) >= 2
    
    def test_list_healthy_engines(self, engine_registry):
        """Test listing healthy engines."""
        with patch.object(engine_registry, 'health_check') as mock_health:
            mock_health.side_effect = lambda e: e == EngineType.OLLAMA
            
            healthy = engine_registry.list_healthy_engines()
            
            assert EngineType.OLLAMA in healthy
            assert EngineType.HUGGINGFACE not in healthy
    
    def test_get_best_engine_for_use_case(self, engine_registry):
        """Test getting best engine for use case."""
        with patch.object(engine_registry, 'list_healthy_engines') as mock_healthy:
            mock_healthy.return_value = [EngineType.OLLAMA, EngineType.HUGGINGFACE]
            
            # Test prefer free and fast
            best = engine_registry.get_best_engine_for_use_case(
                "wildfire_detection",
                prefer_free=True,
                prefer_fast=True
            )
            
            assert best is not None
            assert best.cost_per_1k_tokens == 0.0
    
    def test_get_best_engine_no_suitable_models(self, engine_registry, mock_model_registry):
        """Test getting best engine when no models support use case."""
        mock_model_registry.get_models_for_use_case.return_value = []
        
        best = engine_registry.get_best_engine_for_use_case("unknown_use_case")
        assert best is None
    
    def test_check_requirements(self, engine_registry):
        """Test checking engine requirements."""
        # Mock importlib to simulate package availability
        with patch('importlib.import_module') as mock_import:
            mock_import.side_effect = lambda m: None if m == "requests" else ImportError()
            
            engine_registry.engines[EngineType.OLLAMA].requirements = ["requests", "nonexistent"]
            
            results = engine_registry.check_requirements(EngineType.OLLAMA)
            
            assert results["engine_registered"] is True
            assert results["package_requests"] is True
            assert results["package_nonexistent"] is False
    
    def test_cleanup_single_model(self, engine_registry):
        """Test cleaning up single model engine."""
        engine_registry.get_engine("test-model")
        assert "test-model" in engine_registry.active_engines
        
        engine_registry.cleanup("test-model")
        assert "test-model" not in engine_registry.active_engines
    
    def test_cleanup_all(self, engine_registry):
        """Test cleaning up all engines."""
        engine_registry.get_engine("test-model")
        engine_registry.get_engine("batch-model")
        assert len(engine_registry.active_engines) == 2
        
        engine_registry.cleanup()
        assert len(engine_registry.active_engines) == 0
    
    def test_get_stats(self, engine_registry):
        """Test getting registry statistics."""
        # Create some active engines
        engine_registry.get_engine("test-model")
        
        with patch.object(engine_registry, 'health_check') as mock_health:
            mock_health.return_value = True
            
            stats = engine_registry.get_stats()
            
            assert stats["total_engines"] >= 2
            assert stats["active_instances"] == 1
            assert "engine_health" in stats
            assert stats["healthy_engines"] >= 0
            assert "supported_engine_types" in stats
    
    def test_error_handling_engine_creation(self, engine_registry):
        """Test error handling during engine creation."""
        # Make engine class raise exception
        engine_registry.engines[EngineType.OLLAMA].engine_class = Mock(
            side_effect=Exception("Engine init failed")
        )
        
        with pytest.raises(RuntimeError, match="Failed to create engine"):
            engine_registry.get_engine("test-model")
    
    def test_execute_prompt_with_no_images(self, engine_registry):
        """Test executing prompt without images."""
        response = engine_registry.execute_prompt(
            "test-model",
            "Analyze this text"
        )
        
        assert response.is_success()
        assert response.content is not None


# Integration tests (when real components are available)
@pytest.mark.integration
class TestEngineRegistryIntegration:
    """Integration tests with real components."""
    
    @pytest.mark.skipif(not Path("/root/sai-benchmark/engines/ollama_engine.py").exists(),
                        reason="Ollama engine not available")
    def test_real_ollama_engine_registration(self):
        """Test registration with real Ollama engine."""
        from engines.ollama_engine import OllamaEngine
        
        registry = EngineRegistry()
        assert EngineType.OLLAMA in registry.engines
        assert registry.engines[EngineType.OLLAMA].engine_class == OllamaEngine
    
    @pytest.mark.skipif(not Path("/root/sai-benchmark/engines/hf_engine.py").exists(),
                        reason="HF engine not available")
    def test_real_hf_engine_registration(self):
        """Test registration with real HuggingFace engine."""
        from engines.hf_engine import HuggingFaceEngine
        
        registry = EngineRegistry()
        assert EngineType.HUGGINGFACE in registry.engines
        assert registry.engines[EngineType.HUGGINGFACE].engine_class == HuggingFaceEngine