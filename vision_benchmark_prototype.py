#!/usr/bin/env python3
"""
Multi-Dimensional Vision Assessment (MDVA) Framework
A comprehensive benchmark for testing LLM vision capabilities
"""

import json
import base64
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple
from abc import ABC, abstractmethod
from pathlib import Path
from PIL import Image
from io import BytesIO
import ollama


@dataclass
class TestCase:
    """Individual test case structure"""
    test_id: str
    category: str
    difficulty: int  # 1-10
    image_path: str
    question: str
    expected_output: Dict[str, Any]
    scoring_rubric: Dict[str, float]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'test_id': self.test_id,
            'category': self.category,
            'difficulty': self.difficulty,
            'question': self.question,
            'expected': self.expected_output,
            'rubric': self.scoring_rubric
        }


@dataclass
class TestResult:
    """Result from a single test"""
    test_id: str
    raw_output: str
    parsed_output: Dict[str, Any]
    score: float
    sub_scores: Dict[str, float]
    success: bool
    error: Optional[str] = None


class VisionTest(ABC):
    """Abstract base class for vision tests"""
    
    @abstractmethod
    def generate_prompt(self, test_case: TestCase) -> str:
        """Generate model prompt for test case"""
        pass
    
    @abstractmethod
    def parse_response(self, response: str, test_case: TestCase) -> Dict[str, Any]:
        """Parse model response into structured output"""
        pass
    
    @abstractmethod
    def score_response(self, parsed_output: Dict[str, Any], test_case: TestCase) -> Tuple[float, Dict[str, float]]:
        """Score the response against expected output"""
        pass


class PerceptionTest(VisionTest):
    """Tests for basic visual perception"""
    
    def generate_prompt(self, test_case: TestCase) -> str:
        return f"""Analyze this image and answer: {test_case.question}

Provide your response in JSON format with these fields:
- answer: Your direct answer to the question
- confidence: A number between 0 and 1
- reasoning: Brief explanation

```json
{{
  "answer": "your answer",
  "confidence": 0.X,
  "reasoning": "explanation"
}}
```"""
    
    def parse_response(self, response: str, test_case: TestCase) -> Dict[str, Any]:
        try:
            # Extract JSON from response
            import re
            json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(1))
            else:
                # Try parsing the whole response as JSON
                return json.loads(response)
        except:
            return {"answer": response, "confidence": 0.5, "reasoning": "Parse error"}
    
    def score_response(self, parsed_output: Dict[str, Any], test_case: TestCase) -> Tuple[float, Dict[str, float]]:
        sub_scores = {}
        total_weight = 0
        total_score = 0
        
        # Score each component based on rubric
        for component, weight in test_case.scoring_rubric.items():
            if component in test_case.expected_output:
                expected = test_case.expected_output[component]
                actual = parsed_output.get('answer', {})
                
                if isinstance(expected, (int, float)):
                    # Numeric comparison with tolerance
                    if isinstance(actual, (int, float)):
                        error = abs(expected - actual) / max(expected, 1)
                        component_score = max(0, 1 - error)
                    else:
                        component_score = 0
                elif isinstance(expected, str):
                    # String comparison (exact match)
                    component_score = 1.0 if str(actual).lower() == expected.lower() else 0.0
                else:
                    # Complex comparison - simplified for prototype
                    component_score = 0.5
                
                sub_scores[component] = component_score * weight
                total_score += sub_scores[component]
                total_weight += weight
        
        final_score = total_score / total_weight if total_weight > 0 else 0
        return final_score, sub_scores


class SpatialTest(VisionTest):
    """Tests for spatial understanding"""
    
    def generate_prompt(self, test_case: TestCase) -> str:
        return f"""Analyze the spatial relationships in this image.
{test_case.question}

Respond in JSON format:
```json
{{
  "objects": ["list of objects"],
  "relationships": [
    {{"object1": "name", "relation": "type", "object2": "name"}}
  ],
  "overall_structure": "description"
}}
```"""
    
    def parse_response(self, response: str, test_case: TestCase) -> Dict[str, Any]:
        try:
            import re
            json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(1))
            return json.loads(response)
        except:
            return {"objects": [], "relationships": [], "overall_structure": ""}
    
    def score_response(self, parsed_output: Dict[str, Any], test_case: TestCase) -> Tuple[float, Dict[str, float]]:
        sub_scores = {}
        
        # Score object detection
        if 'objects' in test_case.expected_output:
            expected_objects = set(test_case.expected_output['objects'])
            actual_objects = set(parsed_output.get('objects', []))
            
            precision = len(expected_objects & actual_objects) / len(actual_objects) if actual_objects else 0
            recall = len(expected_objects & actual_objects) / len(expected_objects) if expected_objects else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            sub_scores['object_detection'] = f1
        
        # Score spatial relationships
        if 'relationships' in test_case.expected_output:
            expected_rels = test_case.expected_output['relationships']
            actual_rels = parsed_output.get('relationships', [])
            
            # Simplified scoring - count matching relationships
            matches = 0
            for exp_rel in expected_rels:
                for act_rel in actual_rels:
                    if (exp_rel.get('object1') == act_rel.get('object1') and
                        exp_rel.get('relation') == act_rel.get('relation') and
                        exp_rel.get('object2') == act_rel.get('object2')):
                        matches += 1
                        break
            
            sub_scores['relationships'] = matches / len(expected_rels) if expected_rels else 0
        
        # Calculate weighted total
        total_score = sum(sub_scores.values()) / len(sub_scores) if sub_scores else 0
        return total_score, sub_scores


class VisionBenchmark:
    """Main benchmark orchestrator"""
    
    def __init__(self, model_name: str = "llava"):
        self.model_name = model_name
        self.test_suites = {
            'perception': PerceptionTest(),
            'spatial': SpatialTest(),
            # Add more test types as needed
        }
        self.results: List[TestResult] = []
    
    def image_to_base64(self, image_path: str) -> str:
        """Convert image to base64 for model input"""
        with Image.open(image_path) as img:
            buffered = BytesIO()
            img.save(buffered, format="JPEG")
            return base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    def run_test(self, test_case: TestCase) -> TestResult:
        """Run a single test case"""
        try:
            # Get appropriate test suite
            test_suite = self.test_suites.get(test_case.category)
            if not test_suite:
                return TestResult(
                    test_id=test_case.test_id,
                    raw_output="",
                    parsed_output={},
                    score=0,
                    sub_scores={},
                    success=False,
                    error=f"Unknown test category: {test_case.category}"
                )
            
            # Generate prompt
            prompt = test_suite.generate_prompt(test_case)
            
            # Prepare image
            image_base64 = self.image_to_base64(test_case.image_path)
            
            # Call model
            response = ollama.generate(
                model=self.model_name,
                prompt=prompt,
                images=[image_base64],
                options={"temperature": 0.1}
            )
            
            raw_output = response['response']
            
            # Parse response
            parsed_output = test_suite.parse_response(raw_output, test_case)
            
            # Score response
            score, sub_scores = test_suite.score_response(parsed_output, test_case)
            
            return TestResult(
                test_id=test_case.test_id,
                raw_output=raw_output,
                parsed_output=parsed_output,
                score=score,
                sub_scores=sub_scores,
                success=True
            )
            
        except Exception as e:
            return TestResult(
                test_id=test_case.test_id,
                raw_output="",
                parsed_output={},
                score=0,
                sub_scores={},
                success=False,
                error=str(e)
            )
    
    def run_benchmark(self, test_cases: List[TestCase]) -> Dict[str, Any]:
        """Run full benchmark suite"""
        self.results = []
        category_scores = {}
        
        print(f"Running {len(test_cases)} tests on model: {self.model_name}")
        
        for i, test_case in enumerate(test_cases):
            print(f"Test {i+1}/{len(test_cases)}: {test_case.test_id}")
            result = self.run_test(test_case)
            self.results.append(result)
            
            # Aggregate by category
            if test_case.category not in category_scores:
                category_scores[test_case.category] = []
            category_scores[test_case.category].append(result.score)
        
        # Calculate summary statistics
        summary = {
            'model': self.model_name,
            'total_tests': len(test_cases),
            'successful_tests': sum(1 for r in self.results if r.success),
            'category_scores': {},
            'overall_score': 0
        }
        
        # Category averages
        for category, scores in category_scores.items():
            summary['category_scores'][category] = {
                'average': sum(scores) / len(scores) if scores else 0,
                'count': len(scores)
            }
        
        # Overall score (weighted by category)
        if summary['category_scores']:
            summary['overall_score'] = sum(
                cat_data['average'] for cat_data in summary['category_scores'].values()
            ) / len(summary['category_scores'])
        
        # Map to 0-100 scale
        summary['overall_score'] = int(summary['overall_score'] * 100)
        
        return summary
    
    def save_results(self, output_path: str):
        """Save detailed results to file"""
        results_data = {
            'summary': self.run_benchmark(test_cases),
            'detailed_results': [
                {
                    'test_id': r.test_id,
                    'score': r.score,
                    'sub_scores': r.sub_scores,
                    'success': r.success,
                    'error': r.error
                }
                for r in self.results
            ]
        }
        
        with open(output_path, 'w') as f:
            json.dump(results_data, f, indent=2)


# Example test cases
def create_sample_tests() -> List[TestCase]:
    """Create sample test cases for demonstration"""
    return [
        TestCase(
            test_id="perception_001",
            category="perception",
            difficulty=2,
            image_path="images/sample1.jpg",
            question="How many red objects are in this image?",
            expected_output={"count": 3},
            scoring_rubric={"count": 1.0}
        ),
        TestCase(
            test_id="spatial_001",
            category="spatial",
            difficulty=4,
            image_path="images/sample2.jpg",
            question="Describe the spatial arrangement of the objects",
            expected_output={
                "objects": ["box", "ball", "cup"],
                "relationships": [
                    {"object1": "ball", "relation": "inside", "object2": "box"},
                    {"object1": "cup", "relation": "left_of", "object2": "box"}
                ]
            },
            scoring_rubric={"object_detection": 0.5, "relationships": 0.5}
        )
    ]


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run vision benchmark")
    parser.add_argument("--model", default="llava", help="Model name")
    parser.add_argument("--output", default="benchmark_results.json", help="Output file")
    args = parser.parse_args()
    
    # Initialize benchmark
    benchmark = VisionBenchmark(model_name=args.model)
    
    # Load or create test cases
    test_cases = create_sample_tests()
    
    # Run benchmark
    summary = benchmark.run_benchmark(test_cases)
    
    # Display results
    print("\n=== Benchmark Results ===")
    print(f"Model: {summary['model']}")
    print(f"Overall Score: {summary['overall_score']}/100")
    print("\nCategory Scores:")
    for category, data in summary['category_scores'].items():
        print(f"  {category}: {data['average']*100:.1f}% ({data['count']} tests)")
    
    # Save detailed results
    benchmark.save_results(args.output)
    print(f"\nDetailed results saved to: {args.output}")