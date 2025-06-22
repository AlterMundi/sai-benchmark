"""
Unit tests for PromptRegistry system.

Tests cover:
- Prompt registration and retrieval
- Template rendering and validation
- Output schema validation
- Search and filtering functionality
- Error handling and edge cases
"""

import json
import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch, mock_open

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.prompt_registry import PromptRegistry, PromptTemplate, OutputSchema


class TestOutputSchema:
    """Test OutputSchema dataclass functionality."""
    
    def test_output_schema_creation(self):
        """Test creating OutputSchema with all fields."""
        schema = OutputSchema(
            type="json",
            format={"has_smoke": {"type": "boolean"}},
            validation_regex=r'\{"has_smoke":\s*(true|false)\}',
            fallback_detection=["ERROR", "INVALID"]
        )
        
        assert schema.type == "json"
        assert "has_smoke" in schema.format
        assert schema.validation_regex is not None
        assert len(schema.fallback_detection) == 2
    
    def test_output_schema_optional_fields(self):
        """Test OutputSchema with optional fields."""
        schema = OutputSchema(
            type="text",
            format={}
        )
        
        assert schema.validation_regex is None
        assert schema.fallback_detection is None


class TestPromptTemplate:
    """Test PromptTemplate dataclass functionality."""
    
    def test_prompt_template_creation(self):
        """Test creating PromptTemplate with all fields."""
        schema = OutputSchema(type="json", format={})
        template = PromptTemplate(
            id="test_prompt",
            name="Test Prompt",
            description="A test prompt",
            template="Analyze this: {input}",
            output_schema=schema,
            tags=["test", "unit"],
            version="2.0",
            created_by="tester",
            use_cases=["testing"],
            performance_notes="Fast execution"
        )
        
        assert template.id == "test_prompt"
        assert template.version == "2.0"
        assert "test" in template.tags
        assert len(template.use_cases) == 1
    
    def test_prompt_template_defaults(self):
        """Test PromptTemplate with default values."""
        schema = OutputSchema(type="text", format={})
        template = PromptTemplate(
            id="minimal",
            name="Minimal",
            description="Minimal prompt",
            template="Simple template",
            output_schema=schema,
            tags=[]
        )
        
        assert template.version == "1.0"
        assert template.created_by == "system"
        assert template.use_cases == []
        assert template.performance_notes is None


class TestPromptRegistry:
    """Test PromptRegistry functionality."""
    
    @pytest.fixture
    def empty_registry(self, temp_dir):
        """Create an empty registry without builtin prompts."""
        with patch.object(PromptRegistry, '_register_builtin_prompts'):
            registry = PromptRegistry(registry_path=str(temp_dir / "prompts.yaml"))
        return registry
    
    @pytest.fixture
    def populated_registry(self):
        """Create a registry with builtin prompts."""
        with patch('pathlib.Path.exists', return_value=False):
            registry = PromptRegistry()
        return registry
    
    def test_registry_initialization(self, temp_dir):
        """Test registry initialization with and without file."""
        # Test without existing file
        with patch('pathlib.Path.exists', return_value=False):
            registry = PromptRegistry()
            assert len(registry.prompts) > 0  # Should have builtin prompts
        
        # Test with custom path
        custom_path = str(temp_dir / "custom_prompts.yaml")
        with patch('pathlib.Path.exists', return_value=False):
            registry = PromptRegistry(registry_path=custom_path)
            assert registry.registry_path == custom_path
    
    def test_register_prompt(self, empty_registry):
        """Test registering a new prompt."""
        schema = OutputSchema(type="json", format={"result": {"type": "string"}})
        prompt = PromptTemplate(
            id="custom_prompt",
            name="Custom Prompt",
            description="A custom prompt",
            template="Do something",
            output_schema=schema,
            tags=["custom"]
        )
        
        empty_registry.register_prompt(prompt)
        
        assert "custom_prompt" in empty_registry.prompts
        assert empty_registry.prompts["custom_prompt"].name == "Custom Prompt"
    
    def test_get_prompt_success(self, populated_registry):
        """Test retrieving an existing prompt."""
        prompt = populated_registry.get_prompt("early_fire_json")
        
        assert prompt.id == "early_fire_json"
        assert "fire" in prompt.tags
        assert prompt.output_schema.type == "json"
    
    def test_get_prompt_not_found(self, empty_registry):
        """Test retrieving non-existent prompt."""
        with pytest.raises(ValueError, match="Prompt 'nonexistent' not found"):
            empty_registry.get_prompt("nonexistent")
    
    def test_list_prompts_all(self, populated_registry):
        """Test listing all prompts."""
        prompts = populated_registry.list_prompts()
        
        assert len(prompts) > 0
        assert all(isinstance(p, PromptTemplate) for p in prompts)
    
    def test_list_prompts_by_tags(self, populated_registry):
        """Test filtering prompts by tags."""
        # Test single tag
        fire_prompts = populated_registry.list_prompts(tags=["fire"])
        assert len(fire_prompts) > 0
        assert all("fire" in p.tags for p in fire_prompts)
        
        # Test multiple tags (OR logic)
        json_or_bbox_prompts = populated_registry.list_prompts(tags=["json", "bbox"])
        assert len(json_or_bbox_prompts) > 0
        
        # Test non-existent tag
        empty_prompts = populated_registry.list_prompts(tags=["nonexistent_tag"])
        assert len(empty_prompts) == 0
    
    def test_search_prompts(self, populated_registry):
        """Test searching prompts by query."""
        # Search by name
        results = populated_registry.search_prompts("Early Fire")
        assert len(results) > 0
        assert any("Early Fire" in p.name for p in results)
        
        # Search by description
        results = populated_registry.search_prompts("qwen")
        assert len(results) > 0
        
        # Search by tag
        results = populated_registry.search_prompts("wildfire")
        assert len(results) > 0
        assert any("wildfire" in p.tags for p in results)
        
        # Case insensitive search
        results = populated_registry.search_prompts("FIRE")
        assert len(results) > 0
    
    def test_validate_json_output_success(self, populated_registry):
        """Test validating correct JSON output."""
        output = '{"has_smoke": true, "bbox": [0.5, 0.5, 0.2, 0.3]}'
        result = populated_registry.validate_output("early_fire_json", output)
        
        assert result["valid"] is True
        assert result["parsed_output"]["has_smoke"] is True
        assert len(result["parsed_output"]["bbox"]) == 4
        assert len(result["errors"]) == 0
    
    def test_validate_json_output_with_extra_text(self, populated_registry):
        """Test JSON extraction from output with extra text."""
        output = 'Sure, here is the analysis: {"has_smoke": false, "bbox": [0, 0, 0, 0]}'
        result = populated_registry.validate_output("early_fire_json", output)
        
        # Should fail direct parsing but might succeed with regex
        assert "JSON parsing failed" in str(result["errors"])
    
    def test_validate_structured_output(self, populated_registry):
        """Test validating structured text output."""
        output = """Judgment: Yes
Confidence: 0.85
Justification: Clear smoke visible in the upper right quadrant"""
        
        result = populated_registry.validate_output("wildfire_confidence", output)
        
        assert result["valid"] is True
        assert result["parsed_output"]["judgment"] == "Yes"
        assert result["parsed_output"]["confidence"] == 0.85
        assert "Clear smoke visible" in result["parsed_output"]["justification"]
    
    def test_validate_output_fallback_detection(self, populated_registry):
        """Test fallback pattern detection in invalid output."""
        output = "ERROR: Model failed to generate valid response. BAD_JSON format."
        result = populated_registry.validate_output("early_fire_json", output)
        
        assert result["valid"] is False
        assert any("Fallback pattern detected" in error for error in result["errors"])
    
    def test_validate_output_malformed(self, populated_registry):
        """Test handling completely malformed output."""
        output = "This is not valid JSON or structured output at all"
        result = populated_registry.validate_output("early_fire_json", output)
        
        assert result["valid"] is False
        assert len(result["errors"]) > 0
    
    def test_save_and_load_registry(self, temp_dir, empty_registry):
        """Test saving and loading registry from YAML."""
        # Add a custom prompt
        schema = OutputSchema(
            type="json",
            format={"test": {"type": "string"}},
            validation_regex=r'\{"test":\s*"[^"]+"\}'
        )
        prompt = PromptTemplate(
            id="save_test",
            name="Save Test",
            description="Test saving",
            template="Test template",
            output_schema=schema,
            tags=["test", "save"],
            use_cases=["testing"]
        )
        empty_registry.register_prompt(prompt)
        
        # Save registry
        save_path = str(temp_dir / "test_prompts.yaml")
        empty_registry.save_registry(save_path)
        
        # Load in new registry
        with patch.object(PromptRegistry, '_register_builtin_prompts'):
            new_registry = PromptRegistry(registry_path=save_path)
        
        assert "save_test" in new_registry.prompts
        loaded_prompt = new_registry.prompts["save_test"]
        assert loaded_prompt.name == "Save Test"
        assert loaded_prompt.tags == ["test", "save"]
    
    def test_get_stats(self, populated_registry):
        """Test registry statistics generation."""
        stats = populated_registry.get_stats()
        
        assert "total_prompts" in stats
        assert stats["total_prompts"] > 0
        
        assert "unique_tags" in stats
        assert stats["unique_tags"] > 0
        
        assert "all_tags" in stats
        assert isinstance(stats["all_tags"], list)
        assert "fire" in stats["all_tags"]
        
        assert "use_cases" in stats
        assert "wildfire_detection" in stats["use_cases"]
        
        assert "output_types" in stats
        assert "json" in stats["output_types"]
    
    def test_builtin_prompts_completeness(self, populated_registry):
        """Test that all expected builtin prompts are registered."""
        expected_prompts = [
            "early_fire_json",
            "wildfire_confidence",
            "detailed_sequence_analysis",
            "llama_format",
            "llava_format",
            "minicpm_format",
            "gemma_format",
            "mistral_format"
        ]
        
        for prompt_id in expected_prompts:
            assert prompt_id in populated_registry.prompts
            prompt = populated_registry.prompts[prompt_id]
            assert prompt.template != ""
            assert prompt.output_schema is not None
            assert len(prompt.tags) > 0
    
    def test_model_specific_prompts(self, populated_registry):
        """Test model-specific prompt variations."""
        model_prompts = ["llama_format", "llava_format", "minicpm_format", 
                        "gemma_format", "mistral_format"]
        
        for prompt_id in model_prompts:
            prompt = populated_registry.get_prompt(prompt_id)
            model_name = prompt_id.split('_')[0]
            
            assert model_name in prompt.tags
            assert prompt.output_schema.type == "json"
            assert "model_specific" in prompt.use_cases
    
    @pytest.mark.parametrize("prompt_id,expected_type", [
        ("early_fire_json", "json"),
        ("wildfire_confidence", "structured"),
        ("detailed_sequence_analysis", "json")
    ])
    def test_output_schema_types(self, populated_registry, prompt_id, expected_type):
        """Test that prompts have correct output schema types."""
        prompt = populated_registry.get_prompt(prompt_id)
        assert prompt.output_schema.type == expected_type
    
    def test_concurrent_access(self, empty_registry):
        """Test thread-safe access to registry (basic test)."""
        import threading
        
        def add_prompt(registry, i):
            schema = OutputSchema(type="text", format={})
            prompt = PromptTemplate(
                id=f"concurrent_{i}",
                name=f"Concurrent {i}",
                description="Test concurrent access",
                template="Test",
                output_schema=schema,
                tags=["concurrent"]
            )
            registry.register_prompt(prompt)
        
        threads = []
        for i in range(10):
            t = threading.Thread(target=add_prompt, args=(empty_registry, i))
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        # All prompts should be registered
        assert len([p for p in empty_registry.prompts if p.startswith("concurrent_")]) == 10