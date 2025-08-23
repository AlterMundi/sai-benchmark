#!/usr/bin/env python3
"""
Simple test validation script that tests core components without external dependencies.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def test_prompt_registry():
    """Test basic prompt registry functionality."""
    # Import directly to avoid heavy dependencies in __init__.py
    from core.prompt_registry import PromptRegistry, PromptTemplate, OutputSchema
    
    print("Testing PromptRegistry...")
    
    # Test OutputSchema creation
    schema = OutputSchema(type="json", format={"test": "value"})
    assert schema.type == "json"
    print("✅ OutputSchema creation works")
    
    # Test PromptTemplate creation
    template = PromptTemplate(
        id="test",
        name="Test",
        description="Test prompt",
        template="Test template",
        output_schema=schema,
        tags=["test"]
    )
    assert template.id == "test"
    print("✅ PromptTemplate creation works")
    
    # Test PromptRegistry creation and builtin prompts
    registry = PromptRegistry()
    assert len(registry.prompts) > 0
    print("✅ PromptRegistry initialization works")
    
    # Test getting a builtin prompt
    prompt = registry.get_prompt("early_fire_json")
    assert prompt.id == "early_fire_json"
    print("✅ Prompt retrieval works")
    
    # Test validation
    valid_output = '{"has_smoke": true, "bbox": [0.5, 0.5, 0.2, 0.3]}'
    result = registry.validate_output("early_fire_json", valid_output)
    assert result["valid"] is True
    print("✅ Output validation works")
    
    print("✅ All PromptRegistry tests passed!\n")


def test_model_registry():
    """Test basic model registry functionality."""
    from core.model_registry import ModelRegistry, ModelConfig, ModelCapability, EngineType
    
    print("Testing ModelRegistry...")
    
    # Test ModelConfig creation
    config = ModelConfig(
        id="test-model",
        name="Test Model",
        engine=EngineType.OLLAMA,
        model_path="test/path",
        capabilities={ModelCapability.VISION}
    )
    assert config.id == "test-model"
    print("✅ ModelConfig creation works")
    
    # Test capability checking
    assert config.has_capability(ModelCapability.VISION) is True
    assert config.has_capability(ModelCapability.TEXT) is False
    print("✅ Capability checking works")
    
    # Test ModelRegistry creation
    registry = ModelRegistry()
    assert len(registry.models) > 0
    print("✅ ModelRegistry initialization works")
    
    # Test getting a builtin model
    model = registry.get_model("qwen2.5-vl:7b")
    assert model.id == "qwen2.5-vl:7b"
    print("✅ Model retrieval works")
    
    # Test filtering
    ollama_models = registry.list_models(engine=EngineType.OLLAMA)
    assert len(ollama_models) > 0
    print("✅ Model filtering works")
    
    print("✅ All ModelRegistry tests passed!\n")


def test_metrics_registry():
    """Test basic metrics registry functionality."""
    from core.metrics_registry import MetricsRegistry, MetricResult, MetricType
    
    print("Testing MetricsRegistry...")
    
    # Test MetricResult creation
    result = MetricResult("accuracy", 0.95)
    assert result.metric_name == "accuracy"
    assert result.value == 0.95
    print("✅ MetricResult creation works")
    
    # Test MetricsRegistry creation
    registry = MetricsRegistry()
    assert len(registry.metrics) > 0
    print("✅ MetricsRegistry initialization works")
    
    # Test basic metric calculation
    predictions = [
        {"has_smoke": True},
        {"has_smoke": False},
        {"has_smoke": True},
        {"has_smoke": False}
    ]
    ground_truth = [
        {"has_smoke": True},
        {"has_smoke": False},
        {"has_smoke": False},  # Wrong prediction
        {"has_smoke": False}
    ]
    
    accuracy_result = registry.calculate_metric("accuracy", predictions, ground_truth)
    assert accuracy_result.value == 0.75  # 3 out of 4 correct
    print("✅ Accuracy calculation works")
    
    # Test latency calculation (no ground truth needed)
    predictions_with_latency = [
        {"latency_ms": 100},
        {"latency_ms": 200},
        {"latency_ms": 150}
    ]
    latency_result = registry.calculate_metric("latency", predictions_with_latency)
    assert latency_result.value == 150.0  # Average
    print("✅ Latency calculation works")
    
    print("✅ All MetricsRegistry tests passed!\n")


def main():
    """Run all validation tests."""
    print("SAI-Benchmark Core Component Validation")
    print("=" * 50)
    
    try:
        test_prompt_registry()
        test_model_registry()
        test_metrics_registry()
        
        print("🎉 All core component tests passed!")
        print("\nNext steps:")
        print("- Install full dependencies: pip install -r requirements.txt")
        print("- Run full test suite: pytest tests/")
        print("- Add more comprehensive tests")
        
        return 0
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())