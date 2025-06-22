"""
Unit tests for ModelRegistry system.

Tests cover:
- Model registration and retrieval
- Capability checking and use case validation
- Engine compatibility
- Filtering and searching functionality
- Stats generation
- Error handling
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import patch, mock_open

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.model_registry import (
    ModelRegistry, ModelConfig, ModelCapability, EngineType
)


class TestModelCapability:
    """Test ModelCapability enum."""
    
    def test_capability_values(self):
        """Test that all expected capabilities are defined."""
        expected_capabilities = [
            "vision", "text", "json_output", "bbox_detection",
            "sequence_analysis", "dynamic_resolution", "window_attention"
        ]
        
        actual_values = [cap.value for cap in ModelCapability]
        for expected in expected_capabilities:
            assert expected in actual_values


class TestEngineType:
    """Test EngineType enum."""
    
    def test_engine_values(self):
        """Test that all expected engines are defined."""
        expected_engines = ["ollama", "huggingface", "openai", "anthropic", "google"]
        
        actual_values = [engine.value for engine in EngineType]
        for expected in expected_engines:
            assert expected in actual_values


class TestModelConfig:
    """Test ModelConfig dataclass functionality."""
    
    def test_model_config_creation(self):
        """Test creating ModelConfig with all fields."""
        config = ModelConfig(
            id="test_model",
            name="Test Model",
            engine=EngineType.OLLAMA,
            model_path="test/path",
            capabilities={ModelCapability.VISION, ModelCapability.TEXT},
            max_tokens=2048,
            max_images=5,
            supports_batch=True,
            cost_per_1k_tokens=1.5,
            latency_ms=500,
            gpu_memory_gb=8.0,
            description="A test model",
            version="2.0"
        )
        
        assert config.id == "test_model"
        assert config.engine == EngineType.OLLAMA
        assert ModelCapability.VISION in config.capabilities
        assert config.cost_per_1k_tokens == 1.5
    
    def test_model_config_defaults(self):
        """Test ModelConfig with default values."""
        config = ModelConfig(
            id="minimal",
            name="Minimal Model",
            engine=EngineType.HUGGINGFACE,
            model_path="minimal/path",
            capabilities={ModelCapability.TEXT}
        )
        
        assert config.max_tokens == 4096
        assert config.max_images == 1
        assert config.supports_batch is False
        assert config.cost_per_1k_tokens == 0.0
        assert config.latency_ms is None
        assert config.gpu_memory_gb is None
        assert config.version == "1.0"
    
    def test_capabilities_list_to_set_conversion(self):
        """Test that capabilities list is converted to set."""
        config = ModelConfig(
            id="test",
            name="Test",
            engine=EngineType.OLLAMA,
            model_path="test",
            capabilities=[ModelCapability.VISION, ModelCapability.TEXT]
        )
        
        assert isinstance(config.capabilities, set)
        assert len(config.capabilities) == 2
    
    def test_has_capability(self):
        """Test checking for specific capabilities."""
        config = ModelConfig(
            id="test",
            name="Test",
            engine=EngineType.OLLAMA,
            model_path="test",
            capabilities={ModelCapability.VISION, ModelCapability.JSON_OUTPUT}
        )
        
        assert config.has_capability(ModelCapability.VISION) is True
        assert config.has_capability(ModelCapability.TEXT) is False
    
    @pytest.mark.parametrize("use_case,expected", [
        ("wildfire_detection", True),  # Has VISION and JSON_OUTPUT
        ("smoke_detection", False),     # Missing BBOX_DETECTION
        ("early_warning", True),        # Has VISION and JSON_OUTPUT
        ("expert_analysis", False),     # Missing TEXT
        ("unknown_use_case", True),     # Unknown use case has no requirements
    ])
    def test_supports_use_case(self, use_case, expected):
        """Test use case support validation."""
        config = ModelConfig(
            id="test",
            name="Test",
            engine=EngineType.OLLAMA,
            model_path="test",
            capabilities={ModelCapability.VISION, ModelCapability.JSON_OUTPUT}
        )
        
        assert config.supports_use_case(use_case) == expected


class TestModelRegistry:
    """Test ModelRegistry functionality."""
    
    @pytest.fixture
    def empty_registry(self, temp_dir):
        """Create an empty registry without builtin models."""
        with patch.object(ModelRegistry, '_register_builtin_models'):
            registry = ModelRegistry(registry_path=str(temp_dir / "models.yaml"))
        return registry
    
    @pytest.fixture
    def populated_registry(self):
        """Create a registry with builtin models."""
        with patch('pathlib.Path.exists', return_value=False):
            registry = ModelRegistry()
        return registry
    
    def test_registry_initialization(self, temp_dir):
        """Test registry initialization."""
        # Test without existing file
        with patch('pathlib.Path.exists', return_value=False):
            registry = ModelRegistry()
            assert len(registry.models) > 0  # Should have builtin models
        
        # Test with custom path
        custom_path = str(temp_dir / "custom_models.yaml")
        with patch('pathlib.Path.exists', return_value=False):
            registry = ModelRegistry(registry_path=custom_path)
            assert registry.registry_path == custom_path
    
    def test_register_model(self, empty_registry):
        """Test registering a new model."""
        model = ModelConfig(
            id="custom_model",
            name="Custom Model",
            engine=EngineType.OPENAI,
            model_path="custom/path",
            capabilities={ModelCapability.VISION}
        )
        
        empty_registry.register_model(model)
        
        assert "custom_model" in empty_registry.models
        assert empty_registry.models["custom_model"].name == "Custom Model"
    
    def test_get_model_success(self, populated_registry):
        """Test retrieving an existing model."""
        model = populated_registry.get_model("qwen2.5-vl:7b")
        
        assert model.id == "qwen2.5-vl:7b"
        assert model.engine == EngineType.OLLAMA
        assert ModelCapability.DYNAMIC_RESOLUTION in model.capabilities
    
    def test_get_model_not_found(self, empty_registry):
        """Test retrieving non-existent model."""
        with pytest.raises(ValueError, match="Model 'nonexistent' not found"):
            empty_registry.get_model("nonexistent")
    
    def test_list_models_all(self, populated_registry):
        """Test listing all models."""
        models = populated_registry.list_models()
        
        assert len(models) > 0
        assert all(isinstance(m, ModelConfig) for m in models)
    
    def test_list_models_by_engine(self, populated_registry):
        """Test filtering models by engine."""
        # Test OLLAMA models
        ollama_models = populated_registry.list_models(engine=EngineType.OLLAMA)
        assert len(ollama_models) > 0
        assert all(m.engine == EngineType.OLLAMA for m in ollama_models)
        
        # Test HUGGINGFACE models
        hf_models = populated_registry.list_models(engine=EngineType.HUGGINGFACE)
        assert len(hf_models) > 0
        assert all(m.engine == EngineType.HUGGINGFACE for m in hf_models)
    
    def test_list_models_by_capabilities(self, populated_registry):
        """Test filtering models by capabilities."""
        # Single capability
        vision_models = populated_registry.list_models(
            capabilities=[ModelCapability.VISION]
        )
        assert len(vision_models) > 0
        assert all(ModelCapability.VISION in m.capabilities for m in vision_models)
        
        # Multiple capabilities
        json_bbox_models = populated_registry.list_models(
            capabilities=[ModelCapability.JSON_OUTPUT, ModelCapability.BBOX_DETECTION]
        )
        assert len(json_bbox_models) > 0
        for model in json_bbox_models:
            assert ModelCapability.JSON_OUTPUT in model.capabilities
            assert ModelCapability.BBOX_DETECTION in model.capabilities
    
    def test_get_models_for_use_case(self, populated_registry):
        """Test getting models for specific use cases."""
        # Wildfire detection models
        wildfire_models = populated_registry.get_models_for_use_case("wildfire_detection")
        assert len(wildfire_models) > 0
        assert all(m.supports_use_case("wildfire_detection") for m in wildfire_models)
        
        # Sequence analysis models
        seq_models = populated_registry.get_models_for_use_case("sequence_analysis")
        # Some models should support this
        for model in seq_models:
            assert ModelCapability.SEQUENCE_ANALYSIS in model.capabilities
    
    def test_get_cheapest_model(self, populated_registry):
        """Test finding cheapest model with capabilities."""
        # Free models (local)
        cheapest = populated_registry.get_cheapest_model([ModelCapability.VISION])
        assert cheapest.cost_per_1k_tokens == 0.0
        
        # With specific capabilities
        cheapest_json = populated_registry.get_cheapest_model(
            [ModelCapability.VISION, ModelCapability.JSON_OUTPUT]
        )
        assert ModelCapability.JSON_OUTPUT in cheapest_json.capabilities
    
    def test_get_cheapest_model_no_match(self, empty_registry):
        """Test getting cheapest model when none match."""
        with pytest.raises(ValueError, match="No models found with capabilities"):
            empty_registry.get_cheapest_model([ModelCapability.VISION])
    
    def test_get_fastest_model(self, populated_registry):
        """Test finding fastest model with capabilities."""
        fastest = populated_registry.get_fastest_model([ModelCapability.VISION])
        assert fastest.latency_ms is not None
        
        # Verify it's actually the fastest
        compatible_models = populated_registry.list_models(
            capabilities=[ModelCapability.VISION]
        )
        models_with_latency = [m for m in compatible_models if m.latency_ms is not None]
        min_latency = min(m.latency_ms for m in models_with_latency)
        assert fastest.latency_ms == min_latency
    
    def test_get_fastest_model_no_latency_data(self, empty_registry):
        """Test getting fastest model when no latency data available."""
        model = ModelConfig(
            id="no_latency",
            name="No Latency",
            engine=EngineType.OLLAMA,
            model_path="test",
            capabilities={ModelCapability.VISION},
            latency_ms=None
        )
        empty_registry.register_model(model)
        
        # Should return the model even without latency data
        fastest = empty_registry.get_fastest_model([ModelCapability.VISION])
        assert fastest.id == "no_latency"
    
    def test_validate_model_engine_compatibility(self, populated_registry):
        """Test validating model-engine compatibility."""
        # Compatible
        assert populated_registry.validate_model_engine_compatibility(
            "qwen2.5-vl:7b", EngineType.OLLAMA
        ) is True
        
        # Incompatible
        assert populated_registry.validate_model_engine_compatibility(
            "qwen2.5-vl:7b", EngineType.OPENAI
        ) is False
    
    def test_get_engine_models(self, populated_registry):
        """Test getting all models for a specific engine."""
        ollama_models = populated_registry.get_engine_models(EngineType.OLLAMA)
        assert len(ollama_models) > 0
        assert all(m.engine == EngineType.OLLAMA for m in ollama_models)
        
        openai_models = populated_registry.get_engine_models(EngineType.OPENAI)
        assert len(openai_models) > 0
        assert all(m.engine == EngineType.OPENAI for m in openai_models)
    
    def test_save_and_load_registry(self, temp_dir, empty_registry):
        """Test saving and loading registry from YAML."""
        # Add custom models
        model1 = ModelConfig(
            id="save_test_1",
            name="Save Test 1",
            engine=EngineType.HUGGINGFACE,
            model_path="test/path1",
            capabilities={ModelCapability.VISION, ModelCapability.TEXT}
        )
        model2 = ModelConfig(
            id="save_test_2",
            name="Save Test 2",
            engine=EngineType.OPENAI,
            model_path="test/path2",
            capabilities={ModelCapability.JSON_OUTPUT},
            cost_per_1k_tokens=2.5
        )
        
        empty_registry.register_model(model1)
        empty_registry.register_model(model2)
        
        # Save registry
        save_path = str(temp_dir / "test_models.yaml")
        empty_registry.save_registry(save_path)
        
        # Load in new registry
        with patch.object(ModelRegistry, '_register_builtin_models'):
            new_registry = ModelRegistry(registry_path=save_path)
        
        assert "save_test_1" in new_registry.models
        assert "save_test_2" in new_registry.models
        
        loaded_model1 = new_registry.models["save_test_1"]
        assert loaded_model1.name == "Save Test 1"
        assert loaded_model1.engine == EngineType.HUGGINGFACE
        assert ModelCapability.VISION in loaded_model1.capabilities
        
        loaded_model2 = new_registry.models["save_test_2"]
        assert loaded_model2.cost_per_1k_tokens == 2.5
    
    def test_get_stats(self, populated_registry):
        """Test registry statistics generation."""
        stats = populated_registry.get_stats()
        
        assert "total_models" in stats
        assert stats["total_models"] > 0
        
        assert "engines" in stats
        assert "ollama" in stats["engines"]
        assert stats["engines"]["ollama"] > 0
        
        assert "capabilities" in stats
        assert "vision" in stats["capabilities"]
        
        assert "avg_latency" in stats
        assert isinstance(stats["avg_latency"], (int, float))
        
        assert "free_models" in stats
        assert "paid_models" in stats
        assert stats["free_models"] + stats["paid_models"] == stats["total_models"]
    
    def test_builtin_models_completeness(self, populated_registry):
        """Test that expected builtin models are registered."""
        expected_models = [
            "qwen2.5-vl:7b",
            "qwen2.5-vl-7b-hf",
            "gpt-4o",
            "gpt-4-vision"
        ]
        
        for model_id in expected_models:
            assert model_id in populated_registry.models
            model = populated_registry.models[model_id]
            assert model.engine is not None
            assert len(model.capabilities) > 0
    
    def test_legacy_models(self, populated_registry):
        """Test that legacy models are properly registered."""
        legacy_models = ["llava:13b", "llava:7b", "minicpm-v:8b", 
                        "gemma2:27b", "mistral-small:22b"]
        
        for model_id in legacy_models:
            assert model_id in populated_registry.models
            model = populated_registry.models[model_id]
            assert model.engine == EngineType.OLLAMA
            assert model.version == "legacy"
            assert ModelCapability.VISION in model.capabilities
    
    def test_qwen_models_capabilities(self, populated_registry):
        """Test Qwen models have correct capabilities."""
        qwen_ollama = populated_registry.get_model("qwen2.5-vl:7b")
        qwen_hf = populated_registry.get_model("qwen2.5-vl-7b-hf")
        
        # Both should have same capabilities
        assert qwen_ollama.capabilities == qwen_hf.capabilities
        
        # Check specific capabilities
        expected_caps = {
            ModelCapability.VISION,
            ModelCapability.DYNAMIC_RESOLUTION,
            ModelCapability.WINDOW_ATTENTION,
            ModelCapability.JSON_OUTPUT,
            ModelCapability.BBOX_DETECTION
        }
        assert expected_caps.issubset(qwen_ollama.capabilities)
        
        # Different engines
        assert qwen_ollama.engine == EngineType.OLLAMA
        assert qwen_hf.engine == EngineType.HUGGINGFACE
        
        # Different performance characteristics
        assert qwen_hf.latency_ms < qwen_ollama.latency_ms  # GPU faster
        assert qwen_hf.gpu_memory_gb > qwen_ollama.gpu_memory_gb
    
    def test_concurrent_access(self, empty_registry):
        """Test thread-safe access to registry."""
        import threading
        
        def add_model(registry, i):
            model = ModelConfig(
                id=f"concurrent_{i}",
                name=f"Concurrent {i}",
                engine=EngineType.OLLAMA,
                model_path=f"concurrent/{i}",
                capabilities={ModelCapability.TEXT}
            )
            registry.register_model(model)
        
        threads = []
        for i in range(10):
            t = threading.Thread(target=add_model, args=(empty_registry, i))
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        # All models should be registered
        assert len([m for m in empty_registry.models if m.startswith("concurrent_")]) == 10