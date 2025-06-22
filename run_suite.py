#!/usr/bin/env python3
"""
Test Suite Runner

Command-line interface for executing test suites defined in YAML configuration files.
Supports single suite execution with configurable parallelism and output options.
"""

import argparse
import sys
from pathlib import Path
import json

from core.test_suite import run_test_suite, TestSuiteRunner
from core.prompt_registry import prompt_registry
from core.model_registry import model_registry
from core.engine_registry import engine_registry
from core.metrics_registry import metrics_registry
from core.resource_manager import resource_manager


def main():
    parser = argparse.ArgumentParser(
        description="Run SAI-Benchmark test suites",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run early detection benchmark
  python run_suite.py --suite suites/early_detection.yaml
  
  # Run with custom parallelism and output directory
  python run_suite.py --suite suites/model_comparison.yaml --workers 8 --output results/
  
  # Run with registry information
  python run_suite.py --suite suites/prompt_optimization.yaml --show-registries
        """
    )
    
    parser.add_argument(
        "--suite", "-s",
        required=True,
        help="Path to test suite YAML configuration file"
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
        "--no-save",
        action="store_true",
        help="Don't save results to files"
    )
    
    parser.add_argument(
        "--show-registries",
        action="store_true",
        help="Show registry information before running"
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
    
    # Validate suite file exists
    suite_path = Path(args.suite)
    if not suite_path.exists():
        print(f"Error: Suite file not found: {args.suite}")
        sys.exit(1)
    
    # Show registry information if requested
    if args.show_registries:
        print_registry_info()
    
    try:
        if args.dry_run:
            # Load config and show what would be executed
            from core.test_suite import load_suite_config
            config = load_suite_config(suite_path)
            show_dry_run_info(config)
        else:
            # Run the test suite
            print(f"Starting test suite: {args.suite}")
            print(f"Workers: {args.workers}")
            print(f"Output directory: {args.output}")
            print("-" * 50)
            
            result = run_test_suite(
                config_path=suite_path,
                max_workers=args.workers,
                output_dir=args.output
            )
            
            # Print summary
            print_summary(result, args.verbose)
            
    except KeyboardInterrupt:
        print("\nTest suite interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error running test suite: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def print_registry_info():
    """Print information about loaded registries"""
    print("REGISTRY INFORMATION")
    print("=" * 50)
    
    # Prompt registry
    prompt_stats = prompt_registry.get_stats()
    print(f"Prompts: {prompt_stats['total_prompts']} registered")
    print(f"  Tags: {', '.join(prompt_stats['all_tags'][:10])}{'...' if len(prompt_stats['all_tags']) > 10 else ''}")
    print(f"  Output types: {prompt_stats['output_types']}")
    
    # Model registry  
    model_stats = model_registry.get_stats()
    print(f"Models: {model_stats['total_models']} registered")
    print(f"  Engines: {model_stats['engines']}")
    print(f"  Free models: {model_stats['free_models']}, Paid: {model_stats['paid_models']}")
    
    # Engine registry
    engine_stats = engine_registry.get_stats()
    print(f"Engines: {engine_stats['total_engines']} registered, {engine_stats['healthy_engines']} healthy")
    print(f"  Health: {engine_stats['engine_health']}")
    
    # Metrics registry
    metrics_stats = metrics_registry.get_stats() if hasattr(metrics_registry, 'get_stats') else {}
    print(f"Metrics: {len(metrics_registry.metrics)} registered")
    
    # Resource manager
    resource_stats = resource_manager.get_resource_status()
    print(f"Resources: GPU={resource_stats['can_use_gpu']}, Ollama={resource_stats['ollama_available']}")
    print(f"  Available: {resource_stats['available_resources']['gpu_memory_gb']:.1f}GB GPU, "
          f"{resource_stats['available_resources']['cpu_cores']} CPU cores")
    
    print("-" * 50)


def show_dry_run_info(config):
    """Show what would be executed in dry run mode"""
    print("DRY RUN - Test Suite Configuration")
    print("=" * 50)
    print(f"Suite: {config.name}")
    print(f"Description: {config.description}")
    print(f"Version: {config.version}")
    print()
    
    print(f"Prompts ({len(config.prompts)}):")
    for prompt_id in config.prompts:
        try:
            prompt = prompt_registry.get_prompt(prompt_id)
            print(f"  ✓ {prompt_id}: {prompt.name}")
        except ValueError:
            print(f"  ✗ {prompt_id}: NOT FOUND")
    print()
    
    print(f"Models ({len(config.models)}):")
    for model_id in config.models:
        try:
            model = model_registry.get_model(model_id)
            print(f"  ✓ {model_id}: {model.name} ({model.engine.value})")
        except ValueError:
            print(f"  ✗ {model_id}: NOT FOUND")
    print()
    
    print(f"Datasets ({len(config.datasets)}):")
    for dataset_path in config.datasets:
        exists = Path(dataset_path).exists()
        status = "✓" if exists else "✗"
        print(f"  {status} {dataset_path}")
    print()
    
    print(f"Metrics ({len(config.metrics)}):")
    for metric_name in config.metrics:
        exists = metric_name in metrics_registry.metrics
        status = "✓" if exists else "✗"
        print(f"  {status} {metric_name}")
    print()
    
    # Calculate total test cases
    total_combinations = len(config.prompts) * len(config.models) * len(config.datasets)
    print(f"Total test combinations: {total_combinations}")
    
    # Estimate dataset size
    total_items = 0
    for dataset_path in config.datasets:
        if Path(dataset_path).exists():
            try:
                runner = TestSuiteRunner()
                dataset = runner.load_dataset(dataset_path)
                total_items += len(dataset)
                print(f"  {dataset_path}: {len(dataset)} items")
            except Exception as e:
                print(f"  {dataset_path}: Error loading - {e}")
    
    total_tests = total_combinations * total_items
    print(f"Estimated total test cases: {total_tests}")


def print_summary(result, verbose=False):
    """Print test suite execution summary"""
    print("\nTEST SUITE COMPLETED")
    print("=" * 50)
    
    summary = result.to_dict()['summary']
    
    print(f"Suite: {result.suite_name}")
    print(f"Duration: {result.execution_time:.2f} seconds")
    print(f"Total tests: {summary['total_tests']}")
    print(f"Successful: {summary['successful_tests']}")
    print(f"Failed: {summary['failed_tests']}")
    print(f"Success rate: {summary['successful_tests']/summary['total_tests']*100:.1f}%")
    print(f"Average latency: {summary['avg_latency']:.1f}ms")
    print()
    
    # Show aggregated metrics
    if result.aggregated_metrics:
        print("AGGREGATED METRICS")
        print("-" * 30)
        for metric_name, metric_result in result.aggregated_metrics.items():
            print(f"{metric_name}: {metric_result.value:.4f}")
            if verbose and 'std' in metric_result.details:
                print(f"  (±{metric_result.details['std']:.4f})")
        print()
    
    # Show per-model breakdown if verbose
    if verbose and len(set(r.model_id for r in result.test_results)) > 1:
        print("PER-MODEL BREAKDOWN")
        print("-" * 30)
        
        models = set(r.model_id for r in result.test_results)
        for model_id in sorted(models):
            model_results = [r for r in result.test_results if r.model_id == model_id]
            successful = len([r for r in model_results if r.engine_response.success])
            avg_latency = sum(r.engine_response.latency_ms for r in model_results) / len(model_results)
            
            print(f"{model_id}:")
            print(f"  Tests: {len(model_results)}")
            print(f"  Success: {successful}/{len(model_results)} ({successful/len(model_results)*100:.1f}%)")
            print(f"  Avg latency: {avg_latency:.1f}ms")


if __name__ == "__main__":
    main()