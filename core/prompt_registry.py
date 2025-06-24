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
        
        # Vision capability testing prompts
        self._register_vision_capability_prompts()
    
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
    
    def _register_vision_capability_prompts(self):
        """Register comprehensive vision capability testing prompts"""
        
        # Object Detection Prompt
        self.register_prompt(PromptTemplate(
            id="vision_object_detection",
            name="Vision Object Detection",
            description="Comprehensive object detection and identification in images",
            template=(
                "Analyze this image and identify all visible objects. For each object, provide: "
                "name, category, and position. Respond in JSON format: "
                '{"objects": [{"name": "string", "category": "string", "position": "string"}]}. '
                "Categories should be: electronics, stationery, clothing, tableware, organic, accessory, decoration, or other."
            ),
            output_schema=OutputSchema(
                type="json",
                format={
                    "objects": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "category": {"type": "string"},
                                "position": {"type": "string"}
                            }
                        }
                    }
                },
                validation_regex=r'\{"objects":\s*\[.*\]\s*\}',
                fallback_detection=["objects", "BAD_JSON"]
            ),
            tags=["vision", "object_detection", "json"],
            use_cases=["vision_testing", "object_recognition"]
        ))
        
        # Object Counting Prompt
        self.register_prompt(PromptTemplate(
            id="vision_object_counting",
            name="Vision Object Counting",
            description="Count specific types of objects in images",
            template=(
                "Count the number of objects in each category in this image. "
                "Respond in JSON format: "
                '{"total_objects": number, "categories": {"electronics": number, "stationery": number, '
                '"clothing": number, "tableware": number, "organic": number, "accessory": number, '
                '"decoration": number}}. Set category counts to 0 if none are present.'
            ),
            output_schema=OutputSchema(
                type="json", 
                format={
                    "total_objects": {"type": "integer"},
                    "categories": {
                        "type": "object",
                        "properties": {
                            "electronics": {"type": "integer"},
                            "stationery": {"type": "integer"},
                            "clothing": {"type": "integer"},
                            "tableware": {"type": "integer"},
                            "organic": {"type": "integer"},
                            "accessory": {"type": "integer"},
                            "decoration": {"type": "integer"}
                        }
                    }
                },
                validation_regex=r'\{"total_objects":\s*\d+,\s*"categories":\s*\{.*\}\s*\}',
                fallback_detection=["total_objects", "categories", "BAD_JSON"]
            ),
            tags=["vision", "counting", "json"],
            use_cases=["vision_testing", "object_counting"]
        ))
        
        # Color Recognition Prompt
        self.register_prompt(PromptTemplate(
            id="vision_color_recognition",
            name="Vision Color Recognition",
            description="Identify and count colors of objects in images",
            template=(
                "Identify the primary colors of objects in this image. "
                "Respond in JSON format: "
                '{"colors": {"black": number, "white": number, "green": number, "red": number, '
                '"silver": number, "blue": number, "yellow": number, "mixed": number}}. '
                "Count how many objects have each color as their primary color."
            ),
            output_schema=OutputSchema(
                type="json",
                format={
                    "colors": {
                        "type": "object",
                        "properties": {
                            "black": {"type": "integer"},
                            "white": {"type": "integer"},
                            "green": {"type": "integer"},
                            "red": {"type": "integer"},
                            "silver": {"type": "integer"},
                            "blue": {"type": "integer"},
                            "yellow": {"type": "integer"},
                            "mixed": {"type": "integer"}
                        }
                    }
                },
                validation_regex=r'\{"colors":\s*\{.*\}\s*\}',
                fallback_detection=["colors", "BAD_JSON"]
            ),
            tags=["vision", "color", "json"],
            use_cases=["vision_testing", "color_recognition"]
        ))
        
        # Spatial Relationships Prompt
        self.register_prompt(PromptTemplate(
            id="vision_spatial_relationships",
            name="Vision Spatial Relationships",
            description="Analyze spatial relationships between objects",
            template=(
                "Describe the spatial relationships between objects in this image. "
                "Focus on relative positions like 'left_of', 'right_of', 'above', 'below', 'near', 'overlaps'. "
                "Respond in JSON format: "
                '{"relationships": [{"object1": "string", "relationship": "string", "object2": "string"}]}.'
            ),
            output_schema=OutputSchema(
                type="json",
                format={
                    "relationships": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "object1": {"type": "string"},
                                "relationship": {"type": "string"},
                                "object2": {"type": "string"}
                            }
                        }
                    }
                },
                validation_regex=r'\{"relationships":\s*\[.*\]\s*\}',
                fallback_detection=["relationships", "BAD_JSON"]
            ),
            tags=["vision", "spatial", "json"],
            use_cases=["vision_testing", "spatial_analysis"]
        ))
        
        # Size Estimation Prompt
        self.register_prompt(PromptTemplate(
            id="vision_size_estimation",
            name="Vision Size Estimation",
            description="Estimate relative sizes of objects",
            template=(
                "Categorize objects in this image by their relative sizes: small, medium, or large. "
                "Respond in JSON format: "
                '{"sizes": {"small": number, "medium": number, "large": number}}. '
                "Count how many objects fall into each size category."
            ),
            output_schema=OutputSchema(
                type="json",
                format={
                    "sizes": {
                        "type": "object",
                        "properties": {
                            "small": {"type": "integer"},
                            "medium": {"type": "integer"},
                            "large": {"type": "integer"}
                        }
                    }
                },
                validation_regex=r'\{"sizes":\s*\{.*\}\s*\}',
                fallback_detection=["sizes", "BAD_JSON"]
            ),
            tags=["vision", "size", "json"],
            use_cases=["vision_testing", "size_estimation"]
        ))
        
        # Text Recognition Prompt
        self.register_prompt(PromptTemplate(
            id="vision_text_recognition",
            name="Vision Text Recognition",
            description="Identify any visible text in images",
            template=(
                "Identify any text, logos, or readable content in this image. "
                "Respond in JSON format: "
                '{"text_found": boolean, "text_items": [{"location": "object_name", "text": "text_content"}]}.'
            ),
            output_schema=OutputSchema(
                type="json",
                format={
                    "text_found": {"type": "boolean"},
                    "text_items": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "location": {"type": "string"},
                                "text": {"type": "string"}
                            }
                        }
                    }
                },
                validation_regex=r'\{"text_found":\s*(true|false),\s*"text_items":\s*\[.*\]\s*\}',
                fallback_detection=["text_found", "text_items", "BAD_JSON"]
            ),
            tags=["vision", "text", "ocr", "json"],
            use_cases=["vision_testing", "text_recognition"]
        ))
        
        # Material Recognition Prompt
        self.register_prompt(PromptTemplate(
            id="vision_material_recognition",
            name="Vision Material Recognition", 
            description="Identify materials and textures of objects",
            template=(
                "Identify the materials of objects in this image (e.g., plastic, metal, fabric, glass, wood, paper, ceramic). "
                "Respond in JSON format: "
                '{"materials": {"plastic": number, "metal": number, "fabric": number, "glass": number, '
                '"wood": number, "paper": number, "ceramic": number, "organic": number}}. '
                "Count how many objects are primarily made of each material."
            ),
            output_schema=OutputSchema(
                type="json",
                format={
                    "materials": {
                        "type": "object",
                        "properties": {
                            "plastic": {"type": "integer"},
                            "metal": {"type": "integer"},
                            "fabric": {"type": "integer"},
                            "glass": {"type": "integer"},
                            "wood": {"type": "integer"},
                            "paper": {"type": "integer"},
                            "ceramic": {"type": "integer"},
                            "organic": {"type": "integer"}
                        }
                    }
                },
                validation_regex=r'\{"materials":\s*\{.*\}\s*\}',
                fallback_detection=["materials", "BAD_JSON"]
            ),
            tags=["vision", "material", "texture", "json"],
            use_cases=["vision_testing", "material_recognition"]
        ))
        
        # Shape Analysis Prompt
        self.register_prompt(PromptTemplate(
            id="vision_shape_analysis",
            name="Vision Shape Analysis",
            description="Analyze geometric shapes of objects",
            template=(
                "Identify the basic geometric shapes of objects in this image. "
                "Respond in JSON format: "
                '{"shapes": {"rectangular": number, "circular": number, "cylindrical": number, '
                '"curved": number, "irregular": number}}. '
                "Count objects by their primary geometric shape."
            ),
            output_schema=OutputSchema(
                type="json",
                format={
                    "shapes": {
                        "type": "object",
                        "properties": {
                            "rectangular": {"type": "integer"},
                            "circular": {"type": "integer"},
                            "cylindrical": {"type": "integer"},
                            "curved": {"type": "integer"},
                            "irregular": {"type": "integer"}
                        }
                    }
                },
                validation_regex=r'\{"shapes":\s*\{.*\}\s*\}',
                fallback_detection=["shapes", "BAD_JSON"]
            ),
            tags=["vision", "shape", "geometry", "json"],
            use_cases=["vision_testing", "shape_analysis"]
        ))
        
        # Comprehensive Analysis Prompt
        self.register_prompt(PromptTemplate(
            id="vision_comprehensive_analysis",
            name="Vision Comprehensive Analysis",
            description="Complete analysis combining all vision capabilities",
            template=(
                "Provide a comprehensive analysis of this image including: "
                "1) List all objects with names and categories "
                "2) Count total objects and objects per category "
                "3) Identify primary colors "
                "4) Describe spatial relationships "
                "5) Identify materials and shapes "
                "6) Note any visible text "
                "Respond in structured JSON format with all requested information."
            ),
            output_schema=OutputSchema(
                type="json",
                format={
                    "comprehensive_analysis": {
                        "type": "object",
                        "properties": {
                            "objects": {"type": "array"},
                            "total_objects": {"type": "integer"},
                            "categories": {"type": "object"},
                            "colors": {"type": "object"},
                            "materials": {"type": "object"},
                            "shapes": {"type": "object"},
                            "text_found": {"type": "boolean"},
                            "spatial_relationships": {"type": "array"}
                        }
                    }
                },
                validation_regex=r'\{"comprehensive_analysis":\s*\{.*\}\s*\}',
                fallback_detection=["comprehensive_analysis", "BAD_JSON"]
            ),
            tags=["vision", "comprehensive", "analysis", "json"],
            use_cases=["vision_testing", "complete_analysis"]
        ))
        
        # Resolution Comparison Prompt
        self.register_prompt(PromptTemplate(
            id="vision_resolution_comparison",
            name="Vision Resolution Comparison",
            description="Analyze how image resolution affects object detection",
            template=(
                "Analyze this image and report on the level of detail visible. "
                "Consider: Can you see fine details? Are small objects clearly visible? "
                "Is text readable? Are materials and textures distinguishable? "
                "Respond in JSON format: "
                '{"resolution_analysis": {"detail_level": "high|medium|low", "small_objects_visible": boolean, '
                '"text_readable": boolean, "textures_clear": boolean, "confidence_score": 0.0-1.0}}.'
            ),
            output_schema=OutputSchema(
                type="json",
                format={
                    "resolution_analysis": {
                        "type": "object",
                        "properties": {
                            "detail_level": {"type": "string", "enum": ["high", "medium", "low"]},
                            "small_objects_visible": {"type": "boolean"},
                            "text_readable": {"type": "boolean"},
                            "textures_clear": {"type": "boolean"},
                            "confidence_score": {"type": "number", "minimum": 0.0, "maximum": 1.0}
                        }
                    }
                },
                validation_regex=r'\{"resolution_analysis":\s*\{.*\}\s*\}',
                fallback_detection=["resolution_analysis", "BAD_JSON"]
            ),
            tags=["vision", "resolution", "quality", "json"],
            use_cases=["vision_testing", "resolution_analysis"]
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