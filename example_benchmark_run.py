#!/usr/bin/env python3
"""
Example: Running the Vision Benchmark with concrete test cases
"""

import json
from vision_benchmark_prototype import VisionBenchmark, TestCase


def create_comprehensive_test_suite():
    """Create a comprehensive set of test cases across all levels"""
    
    test_cases = []
    
    # Level 1: Basic Perception Tests
    test_cases.extend([
        TestCase(
            test_id="color_001",
            category="perception",
            difficulty=1,
            image_path="images/rio-de-janeiro-2113836_1920.jpg",
            question="What is the dominant color of the sky in this image?",
            expected_output={"answer": "blue"},
            scoring_rubric={"answer": 1.0}
        ),
        TestCase(
            test_id="count_001",
            category="perception",
            difficulty=2,
            image_path="images/camera-1284459_640.jpg",
            question="How many cameras or camera-like objects can you see?",
            expected_output={"answer": 1},
            scoring_rubric={"answer": 1.0}
        ),
        TestCase(
            test_id="shape_001",
            category="perception",
            difficulty=2,
            image_path="images/camera-1284459_640.jpg",
            question="What is the primary shape of the camera lens (circular, square, rectangular)?",
            expected_output={"answer": "circular"},
            scoring_rubric={"answer": 1.0}
        ),
    ])
    
    # Level 2: Spatial Understanding Tests
    test_cases.extend([
        TestCase(
            test_id="spatial_002",
            category="spatial",
            difficulty=3,
            image_path="images/rio-de-janeiro-2113836_1920.jpg",
            question="Describe the spatial relationship between the Christ statue and the city below",
            expected_output={
                "objects": ["statue", "city", "mountain"],
                "relationships": [
                    {"object1": "statue", "relation": "above", "object2": "city"},
                    {"object1": "statue", "relation": "on", "object2": "mountain"}
                ]
            },
            scoring_rubric={"object_detection": 0.4, "relationships": 0.6}
        ),
    ])
    
    # Add more sophisticated tests as needed...
    
    return test_cases


def run_comparison_benchmark():
    """Run benchmark on multiple models for comparison"""
    
    # Models to test (ensure they're available in Ollama)
    models = ["llava", "bakllava", "llava-llama3"]  # Adjust based on available models
    
    # Create test suite
    test_cases = create_comprehensive_test_suite()
    
    # Results storage
    all_results = {}
    
    for model in models:
        print(f"\n{'='*50}")
        print(f"Testing model: {model}")
        print('='*50)
        
        try:
            benchmark = VisionBenchmark(model_name=model)
            summary = benchmark.run_benchmark(test_cases)
            all_results[model] = summary
            
            # Save model-specific results
            benchmark.save_results(f"results_{model}.json")
            
        except Exception as e:
            print(f"Error testing {model}: {e}")
            all_results[model] = {"error": str(e)}
    
    # Create comparison report
    print("\n\n" + "="*60)
    print("COMPARATIVE RESULTS")
    print("="*60)
    
    # Header
    print(f"{'Model':<20} {'Overall':<10} {'Perception':<15} {'Spatial':<15}")
    print("-"*60)
    
    # Results
    for model, results in all_results.items():
        if "error" in results:
            print(f"{model:<20} {'ERROR':<10}")
        else:
            overall = f"{results['overall_score']}/100"
            perception = results['category_scores'].get('perception', {}).get('average', 0)
            spatial = results['category_scores'].get('spatial', {}).get('average', 0)
            
            print(f"{model:<20} {overall:<10} {perception*100:<15.1f} {spatial*100:<15.1f}")
    
    # Save comparison
    with open("benchmark_comparison.json", 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print("\nDetailed comparison saved to: benchmark_comparison.json")


def demonstrate_single_test():
    """Demonstrate running a single test with detailed output"""
    
    print("=== Single Test Demonstration ===\n")
    
    # Create a single test case
    test = TestCase(
        test_id="demo_001",
        category="perception",
        difficulty=1,
        image_path="images/camera-1284459_640.jpg",
        question="Is there a camera in this image? Answer with just 'yes' or 'no'.",
        expected_output={"answer": "yes"},
        scoring_rubric={"answer": 1.0}
    )
    
    # Run test
    benchmark = VisionBenchmark(model_name="llava")
    result = benchmark.run_test(test)
    
    # Display results
    print(f"Test ID: {result.test_id}")
    print(f"Success: {result.success}")
    print(f"Score: {result.score}")
    print(f"\nModel Response:")
    print(result.raw_output)
    print(f"\nParsed Output:")
    print(json.dumps(result.parsed_output, indent=2))
    
    if result.error:
        print(f"\nError: {result.error}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "single":
        # Run single test demonstration
        demonstrate_single_test()
    elif len(sys.argv) > 1 and sys.argv[1] == "compare":
        # Run comparison benchmark
        run_comparison_benchmark()
    else:
        # Run standard benchmark
        test_suite = create_comprehensive_test_suite()
        benchmark = VisionBenchmark()
        summary = benchmark.run_benchmark(test_suite)
        
        print("\n=== Benchmark Complete ===")
        print(f"Overall Score: {summary['overall_score']}/100")
        print("\nRun with 'single' argument for single test demo")
        print("Run with 'compare' argument for model comparison")