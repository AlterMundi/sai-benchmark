"""
Centralized Prompt Registry System

Manages prompt templates with metadata, versioning, and validation.
Supports dynamic templating and multi-format output schemas.
"""

import json
import yaml
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import re


@dataclass
class OutputSchema:
    """Defines expected output format for a prompt"""
    type: str  # "json", "text", "structured"
    format: Dict[str, Any]  # Schema definition
    validation_regex: Optional[str] = None
    fallback_detection: Optional[List[str]] = None


@dataclass
class PromptTemplate:
    """Prompt template with metadata and configuration"""
    id: str
    name: str
    description: str
    template: str
    output_schema: OutputSchema
    tags: List[str]
    version: str = "1.0"
    created_by: str = "system"
    use_cases: List[str] = None
    performance_notes: Optional[str] = None
    
    def __post_init__(self):
        if self.use_cases is None:
            self.use_cases = []


class PromptRegistry:
    """Centralized registry for managing prompt templates"""
    
    def __init__(self, registry_path: Optional[str] = None):
        self.registry_path = registry_path or "core/prompts.yaml"
        self.prompts: Dict[str, PromptTemplate] = {}
        self._load_registry()
        self._register_builtin_prompts()
    
    def _load_registry(self):
        """Load prompts from registry file if it exists"""
        if Path(self.registry_path).exists():
            with open(self.registry_path, 'r') as f:
                data = yaml.safe_load(f)
                for prompt_data in data.get('prompts', []):
                    schema_data = prompt_data.pop('output_schema')
                    output_schema = OutputSchema(**schema_data)
                    prompt = PromptTemplate(output_schema=output_schema, **prompt_data)
                    self.prompts[prompt.id] = prompt
    
    def _register_builtin_prompts(self):
        """Register built-in prompts extracted from existing codebase"""
        
        # Early fire detection with JSON output (current main prompt)
        self.register_prompt(PromptTemplate(
            id="early_fire_json",
            name="Early Fire Detection JSON",
            description="Qwen 2.5-VL optimized prompt for early fire detection with strict JSON output",
            template=(
                "You are an early-fire detection agent. "
                "Answer ONLY with a valid JSON matching this schema: "
                '{"has_smoke": bool, "bbox": [x_center, y_center, width, height]}. '
                "The bbox values should be normalized (0-1). "
                "If no smoke is detected, use bbox: [0, 0, 0, 0]. "
                "DO NOT include any other text in your response."
            ),
            output_schema=OutputSchema(
                type="json",
                format={
                    "has_smoke": {"type": "boolean"},
                    "bbox": {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4}
                },
                validation_regex=r'\{"has_smoke":\s*(true|false),\s*"bbox":\s*\[[0-9.,\s]+\]\}',
                fallback_detection=["BAD_JSON", "ERROR", "INVALID"]
            ),
            tags=["fire", "json", "bbox", "qwen", "strict"],
            use_cases=["wildfire_detection", "smoke_detection", "early_warning"],
            performance_notes="Optimized for Qwen 2.5-VL dynamic resolution system"
        ))
        
        # Legacy confidence-based detection
        self.register_prompt(PromptTemplate(
            id="wildfire_confidence", 
            name="Wildfire Detection with Confidence",
            description="Legacy prompt format with judgment/confidence/justification output",
            template=(
                "Do you detect smoke related to wildfires in this image? Consider that smoke can vary "
                "in density, color (gray, white, black), and shape (columns, diffuse layers), and may be "
                "distant or nearby. Distinguish smoke from fog, mist, or clouds based on its origin (fire) "
                "and visual characteristics. Confidence represents your certainty in the judgment: 1.0 means "
                "absolute certainty (no doubt), 0.5 means complete uncertainty (equal chance of Yes/No), "
                "and values in between reflect your level of certainty based on the evidence. "
                "Respond in this format:\n"
                "Judgment: [Yes/No]\n"
                "Confidence: [number between 0.0 and 1.0]\n"
                "Justification: [brief text]"
            ),
            output_schema=OutputSchema(
                type="structured",
                format={
                    "judgment": {"type": "string", "enum": ["Yes", "No"]},
                    "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "justification": {"type": "string"}
                },
                validation_regex=r'Judgment:\s*(Yes|No)\s*\nConfidence:\s*([0-1]\.?[0-9]*)\s*\nJustification:\s*(.+)',
                fallback_detection=["Judgment:", "Confidence:", "ERROR"]
            ),
            tags=["wildfire", "confidence", "legacy", "structured"],
            use_cases=["wildfire_detection", "expert_analysis"],
            performance_notes="Works well with LLaMA and older vision models"
        ))
        
        # Comprehensive sequence analysis
        self.register_prompt(PromptTemplate(
            id="detailed_sequence_analysis",
            name="Detailed Fire Sequence Analysis", 
            description="Comprehensive multi-image analysis with detailed instructions",
            template=(
                "# ROLE AND OBJECTIVE\n\n"
                "You are a highly sensitive expert AI system specialized in the early and subtle detection "
                "of forest fires through intelligent visual analysis of image sequences. Your primary objective "
                "is to identify incipient or developing signs of smoke in images captured sequentially by "
                "fixed-position cameras in natural environments, ideally before a fire becomes obvious or "
                "while confirming its initial development. Accuracy and early detection are critical.\n\n"
                "# STRICT OUTPUT FORMAT (JSON)\n\n"
                "```json\n"
                "{\n"
                '  "smoke_detected": "Yes" | "No",\n'
                '  "justification": "string",\n'
                '  "confidence": float,\n'
                '  "images_discarded": boolean,\n'
                '  "number_of_images": integer\n'
                "}\n"
                "```"
            ),
            output_schema=OutputSchema(
                type="json",
                format={
                    "smoke_detected": {"type": "string", "enum": ["Yes", "No"]},
                    "justification": {"type": "string"},
                    "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "images_discarded": {"type": "boolean"},
                    "number_of_images": {"type": "integer", "minimum": 1}
                },
                validation_regex=r'\{\s*"smoke_detected":\s*"(Yes|No)",\s*"justification":\s*"[^"]+",\s*"confidence":\s*[0-9.]+,\s*"images_discarded":\s*(true|false),\s*"number_of_images":\s*[0-9]+\s*\}',
                fallback_detection=["smoke_detected", "BAD_JSON"]
            ),
            tags=["sequence", "detailed", "json", "comprehensive"],
            use_cases=["sequence_analysis", "detailed_assessment", "research"],
            performance_notes="Requires models with strong instruction following capabilities"
        ))
        
        # Model-specific format templates
        self._register_model_specific_prompts()
    
    def _register_model_specific_prompts(self):
        """Register model-specific prompt variations"""
        
        base_instruction = (
            "Analyze this image for smoke or fire. Respond with JSON format: "
            '{"has_smoke": true/false, "bbox": [x, y, width, height]}. '
            "Bounding box coordinates should be normalized between 0 and 1. "
            "If no smoke/fire detected, set bbox to [0, 0, 0, 0]."
        )
        
        model_prompts = [
            ("llama_format", "LLaMA Format", "LLaMA-optimized fire detection prompt"),
            ("llava_format", "LLaVA Format", "LLaVA-optimized fire detection prompt"), 
            ("minicpm_format", "MiniCPM Format", "MiniCPM-optimized fire detection prompt"),
            ("gemma_format", "Gemma Format", "Gemma-optimized fire detection prompt"),
            ("mistral_format", "Mistral Format", "Mistral-optimized fire detection prompt")
        ]
        
        for prompt_id, name, description in model_prompts:
            self.register_prompt(PromptTemplate(
                id=prompt_id,
                name=name,
                description=description,
                template=base_instruction,
                output_schema=OutputSchema(
                    type="json",
                    format={
                        "has_smoke": {"type": "boolean"},
                        "bbox": {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4}
                    },
                    validation_regex=r'\{"has_smoke":\s*(true|false),\s*"bbox":\s*\[[0-9.,\s]+\]\}',
                    fallback_detection=["BAD_JSON"]
                ),
                tags=["fire", "json", "bbox", prompt_id.split('_')[0]],
                use_cases=["wildfire_detection", "model_specific"]
            ))
    
    def register_prompt(self, prompt: PromptTemplate):
        """Register a new prompt template"""
        self.prompts[prompt.id] = prompt
    
    def get_prompt(self, prompt_id: str) -> PromptTemplate:
        """Get prompt template by ID"""
        if prompt_id not in self.prompts:
            raise ValueError(f"Prompt '{prompt_id}' not found in registry")
        return self.prompts[prompt_id]
    
    def list_prompts(self, tags: Optional[List[str]] = None) -> List[PromptTemplate]:
        """List prompts, optionally filtered by tags"""
        prompts = list(self.prompts.values())
        if tags:
            prompts = [p for p in prompts if any(tag in p.tags for tag in tags)]
        return prompts
    
    def search_prompts(self, query: str) -> List[PromptTemplate]:
        """Search prompts by name, description, or tags"""
        query = query.lower()
        results = []
        for prompt in self.prompts.values():
            if (query in prompt.name.lower() or 
                query in prompt.description.lower() or
                any(query in tag.lower() for tag in prompt.tags)):
                results.append(prompt)
        return results
    
    def validate_output(self, prompt_id: str, output: str) -> Dict[str, Any]:
        """Validate model output against prompt's expected schema"""
        prompt = self.get_prompt(prompt_id)
        schema = prompt.output_schema
        
        result = {
            "valid": False,
            "parsed_output": None,
            "errors": []
        }
        
        try:
            if schema.type == "json":
                # Try direct JSON parsing
                try:
                    parsed = json.loads(output)
                    result["parsed_output"] = parsed
                    result["valid"] = True
                except json.JSONDecodeError:
                    # Try regex extraction if direct parsing fails
                    if schema.validation_regex:
                        match = re.search(schema.validation_regex, output, re.DOTALL)
                        if match:
                            result["errors"].append("JSON parsing failed, used regex extraction")
                            # Handle different regex group patterns
                            if prompt_id == "wildfire_confidence":
                                result["parsed_output"] = {
                                    "judgment": match.group(1),
                                    "confidence": float(match.group(2)),
                                    "justification": match.group(3).strip()
                                }
                                result["valid"] = True
                    
                    if not result["valid"]:
                        result["errors"].append(f"JSON parsing failed: {output}")
                        
            elif schema.type == "structured":
                if schema.validation_regex:
                    match = re.search(schema.validation_regex, output, re.DOTALL)
                    if match:
                        if prompt_id == "wildfire_confidence":
                            result["parsed_output"] = {
                                "judgment": match.group(1),
                                "confidence": float(match.group(2)), 
                                "justification": match.group(3).strip()
                            }
                            result["valid"] = True
                    else:
                        result["errors"].append("Structured format validation failed")
            
            # Check for fallback detection patterns
            if schema.fallback_detection and not result["valid"]:
                for pattern in schema.fallback_detection:
                    if pattern in output:
                        result["errors"].append(f"Fallback pattern detected: {pattern}")
                        break
                        
        except Exception as e:
            result["errors"].append(f"Validation error: {str(e)}")
        
        return result
    
    def save_registry(self, path: Optional[str] = None):
        """Save registry to YAML file"""
        save_path = path or self.registry_path
        
        registry_data = {
            "version": "1.0",
            "prompts": []
        }
        
        for prompt in self.prompts.values():
            prompt_dict = asdict(prompt)
            registry_data["prompts"].append(prompt_dict)
        
        with open(save_path, 'w') as f:
            yaml.dump(registry_data, f, default_flow_style=False, indent=2)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get registry statistics"""
        all_tags = set()
        use_case_counts = {}
        
        for prompt in self.prompts.values():
            all_tags.update(prompt.tags)
            for use_case in prompt.use_cases:
                use_case_counts[use_case] = use_case_counts.get(use_case, 0) + 1
        
        return {
            "total_prompts": len(self.prompts),
            "unique_tags": len(all_tags),
            "all_tags": sorted(list(all_tags)),
            "use_cases": use_case_counts,
            "output_types": {
                output_type: len([p for p in self.prompts.values() if p.output_schema.type == output_type])
                for output_type in set(p.output_schema.type for p in self.prompts.values())
            }
        }


# Global registry instance
prompt_registry = PromptRegistry()