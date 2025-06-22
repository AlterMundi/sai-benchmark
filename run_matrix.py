#!/usr/bin/env python3
"""
Matrix Testing Runner

Command-line interface for running custom matrix tests across multiple dimensions:
prompts × models × engines × datasets. Provides flexible testing without predefined suites.
"""

import argparse
import sys
from pathlib import Path
import itertools
from datetime import datetime

from core.test_suite import TestSuiteRunner, TestSuiteConfig
from core.prompt_registry import prompt_registry
from core.model_registry import model_registry
from core.engine_registry import engine_registry


def main():
    parser = argparse.ArgumentParser(
        description="Run matrix tests across multiple dimensions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test multiple prompts on single model
  python run_matrix.py --prompts "early_fire_json,wildfire_confidence" --models "qwen2.5-vl:7b" --datasets "data/test_set"
  
  # Compare models with single prompt
  python run_matrix.py --prompts "early_fire_json" --models "qwen2.5-vl:7b,gpt-4o" --datasets "data/benchmark"
  
  # Full matrix test
  python run_matrix.py --prompts "early_fire_json,wildfire_confidence" --models "qwen2.5-vl:7b,qwen2.5-vl-7b-hf" --datasets "data/fire_test,data/smoke_test"
        """
    )
    
    parser.add_argument(
        "--prompts", "-p",
        required=True,
        help="Comma-separated list of prompt IDs"
    )
    
    parser.add_argument(
        "--models", "-m", 
        required=True,
        help="Comma-separated list of model IDs"
    )
    
    parser.add_argument(
        "--datasets", "-d",
        required=True,
        help="Comma-separated list of dataset paths"
    )
    
    parser.add_argument(
        "--metrics",
        default="accuracy,precision,recall,f1_score,latency,parse_success_rate",
        help="Comma-separated list of metrics to calculate"
    )
    
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4)"
    )
    
    parser.add_argument(
        "--output", "-o",
        default="out",
        help="Output directory for results (default: out)"
    )
    
    parser.add_argument(
        "--name",
        help="Custom name for the test run (default: auto-generated)"
    )
    
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Model temperature (default: 0.1)"
    )
    
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Request timeout in seconds (default: 120)"
    )
    
    parser.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help="Maximum retries per request (default: 2)"
    )
    
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.4,
        help="IoU threshold for bbox evaluation (default: 0.4)"
    )
    
    parser.add_argument(
        "--show-combinations",
        action="store_true",
        help="Show all test combinations before running"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true", 
        help="Show what would be executed without running tests"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )

    args = parser.parse_args()
    
    # Parse input lists
    prompts = [p.strip() for p in args.prompts.split(',')]
    models = [m.strip() for m in args.models.split(',')]
    datasets = [d.strip() for d in args.datasets.split(',')]
    metrics = [m.strip() for m in args.metrics.split(',')]
    
    # Validate inputs
    try:
        validate_inputs(prompts, models, datasets, metrics)
    except ValueError as e:
        print(f"Validation error: {e}")
        sys.exit(1)
    
    # Generate test name
    test_name = args.name or generate_test_name(prompts, models, datasets)
    
    # Show combinations if requested
    if args.show_combinations or args.dry_run:
        show_test_combinations(prompts, models, datasets, metrics, test_name)
        
        if args.dry_run:
            return
    
    # Create dynamic test suite configuration
    config = TestSuiteConfig(
        name=test_name,
        description=f"Matrix test: {len(prompts)} prompts × {len(models)} models × {len(datasets)} datasets",
        prompts=prompts,
        models=models,
        datasets=datasets,
        metrics=metrics,
        engine_config={
            "temperature": args.temperature,
            "timeout": args.timeout,
            "max_retries": args.max_retries
        },
        test_config={
            "iou_threshold": args.iou_threshold
        }
    )
    
    try:
        print(f"Starting matrix test: {test_name}")
        print(f"Configuration: {len(prompts)}×{len(models)}×{len(datasets)}")
        print(f"Workers: {args.workers}")
        print(f"Output directory: {args.output}")
        print("-" * 50)
        
        # Run the test suite
        runner = TestSuiteRunner()
        result = runner.run_suite(
            config=config,
            max_workers=args.workers,
            output_dir=args.output
        )
        
        # Print results
        print_matrix_results(result, args.verbose)
        
    except KeyboardInterrupt:
        print("\nMatrix test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error running matrix test: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def validate_inputs(prompts, models, datasets, metrics):
    """Validate that all inputs exist in registries"""
    
    # Validate prompts
    for prompt_id in prompts:
        try:
            prompt_registry.get_prompt(prompt_id)
        except ValueError:
            raise ValueError(f"Prompt '{prompt_id}' not found in registry")
    
    # Validate models
    for model_id in models:
        try:
            model_registry.get_model(model_id)
        except ValueError:
            raise ValueError(f"Model '{model_id}' not found in registry")
    
    # Validate datasets
    for dataset_path in datasets:
        if not Path(dataset_path).exists():
            raise ValueError(f"Dataset '{dataset_path}' not found")
    
    # Validate metrics
    for metric_name in metrics:
        if metric_name not in metrics_registry.metrics:
            raise ValueError(f"Metric '{metric_name}' not found in registry")


def generate_test_name(prompts, models, datasets):
    """Generate a test name based on inputs"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create abbreviated components
    prompt_abbrev = "_".join(p[:8] for p in prompts[:2])
    if len(prompts) > 2:
        prompt_abbrev += f"_plus{len(prompts)-2}"
    
    model_abbrev = "_".join(m.split(':')[0] for m in models[:2])
    if len(models) > 2:
        model_abbrev += f"_plus{len(models)-2}"
    
    return f"matrix_{prompt_abbrev}_{model_abbrev}_{timestamp}"


def show_test_combinations(prompts, models, datasets, metrics, test_name):
    """Show all test combinations that will be executed"""
    print("MATRIX TEST CONFIGURATION")
    print("=" * 50)
    print(f"Test name: {test_name}")
    print()
    
    print(f"Prompts ({len(prompts)}):")
    for prompt_id in prompts:
        try:
            prompt = prompt_registry.get_prompt(prompt_id)
            print(f"  ✓ {prompt_id}: {prompt.name}")
        except ValueError:
            print(f"  ✗ {prompt_id}: NOT FOUND")
    print()
    
    print(f"Models ({len(models)}):")
    for model_id in models:
        try:
            model = model_registry.get_model(model_id)
            print(f"  ✓ {model_id}: {model.name} ({model.engine.value})")
        except ValueError:
            print(f"  ✗ {model_id}: NOT FOUND")
    print()
    
    print(f"Datasets ({len(datasets)}):")
    for dataset_path in datasets:
        exists = Path(dataset_path).exists()
        status = "✓" if exists else "✗"
        print(f"  {status} {dataset_path}")
    print()
    
    print(f"Metrics ({len(metrics)}):")
    for metric_name in metrics:
        print(f"  • {metric_name}")
    print()
    
    # Calculate combinations
    total_combinations = len(prompts) * len(models) * len(datasets)
    print(f"Total combinations: {total_combinations}")
    
    # Show matrix
    print("\nCOMBINATION MATRIX:")
    print("-" * 30)
    for i, (prompt, model, dataset) in enumerate(itertools.product(prompts, models, datasets)):
        print(f"{i+1:3d}. {prompt} × {model} × {Path(dataset).name}")
    
    print("-" * 50)


def print_matrix_results(result, verbose=False):
    """Print matrix test results with dimensional analysis"""
    print("\nMATRIX TEST COMPLETED")
    print("=" * 50)
    
    summary = result.to_dict()['summary']
    
    print(f"Test: {result.suite_name}")
    print(f"Duration: {result.execution_time:.2f} seconds")
    print(f"Total tests: {summary['total_tests']}")
    print(f"Success rate: {summary['successful_tests']/summary['total_tests']*100:.1f}%")
    print()
    
    # Show aggregated metrics
    if result.aggregated_metrics:
        print("OVERALL METRICS")
        print("-" * 20)
        for metric_name, metric_result in result.aggregated_metrics.items():
            print(f"{metric_name}: {metric_result.value:.4f}")
            if verbose and 'std' in metric_result.details:
                print(f"  (±{metric_result.details['std']:.4f})")
        print()
    
    # Dimensional analysis
    if verbose:
        print_dimensional_analysis(result)


def print_dimensional_analysis(result):
    """Print analysis broken down by each dimension"""
    test_results = result.test_results
    
    # By Prompt
    prompts = set(r.prompt_id for r in test_results)
    if len(prompts) > 1:
        print("BY PROMPT")
        print("-" * 15)
        for prompt_id in sorted(prompts):
            prompt_results = [r for r in test_results if r.prompt_id == prompt_id]
            successful = len([r for r in prompt_results if r.engine_response.success])
            avg_latency = sum(r.engine_response.latency_ms for r in prompt_results) / len(prompt_results)
            
            print(f"{prompt_id}:")
            print(f"  Success: {successful}/{len(prompt_results)} ({successful/len(prompt_results)*100:.1f}%)")
            print(f"  Avg latency: {avg_latency:.1f}ms")
            
            # Show best accuracy metric if available
            if 'accuracy' in result.aggregated_metrics:
                accuracies = [r.metrics.get('accuracy', {}).get('value', 0) for r in prompt_results if 'accuracy' in r.metrics]
                if accuracies:
                    print(f"  Avg accuracy: {sum(accuracies)/len(accuracies):.3f}")
        print()
    
    # By Model
    models = set(r.model_id for r in test_results)
    if len(models) > 1:
        print("BY MODEL")
        print("-" * 15)
        for model_id in sorted(models):
            model_results = [r for r in test_results if r.model_id == model_id]
            successful = len([r for r in model_results if r.engine_response.success])
            avg_latency = sum(r.engine_response.latency_ms for r in model_results) / len(model_results)
            
            print(f"{model_id}:")
            print(f"  Success: {successful}/{len(model_results)} ({successful/len(model_results)*100:.1f}%)")
            print(f"  Avg latency: {avg_latency:.1f}ms")
            
            # Show best accuracy metric if available
            if 'accuracy' in result.aggregated_metrics:
                accuracies = [r.metrics.get('accuracy', {}).get('value', 0) for r in model_results if 'accuracy' in r.metrics]
                if accuracies:
                    print(f"  Avg accuracy: {sum(accuracies)/len(accuracies):.3f}")
        print()


if __name__ == "__main__":
    # Need to import here to avoid circular imports
    from core.metrics_registry import metrics_registry
    main()