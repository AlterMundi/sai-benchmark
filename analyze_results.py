#!/usr/bin/env python3
"""
Results Analysis Tool

Command-line interface for analyzing test suite results and generating comparison reports.
Supports multiple result files, statistical analysis, and visualization exports.
"""

import argparse
import sys
import json
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
from datetime import datetime

from core.metrics_registry import metrics_registry


def main():
    parser = argparse.ArgumentParser(
        description="Analyze test suite results and generate reports",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze single result file
  python analyze_results.py --results out/suite_early_detection_20241215_143052_results.json
  
  # Compare multiple results
  python analyze_results.py --results out/result1.json out/result2.json --report comparison
  
  # Generate detailed statistical report
  python analyze_results.py --results out/*.json --report detailed --output analysis/
        """
    )
    
    parser.add_argument(
        "--results", "-r",
        nargs="+",
        required=True,
        help="One or more result JSON files to analyze"
    )
    
    parser.add_argument(
        "--report",
        choices=["summary", "comparison", "detailed", "statistical"],
        default="summary",
        help="Type of report to generate (default: summary)"
    )
    
    parser.add_argument(
        "--output", "-o",
        help="Output directory for generated reports"
    )
    
    parser.add_argument(
        "--format",
        choices=["text", "json", "csv", "html"],
        default="text",
        help="Output format (default: text)"
    )
    
    parser.add_argument(
        "--metrics",
        help="Comma-separated list of metrics to focus on (default: all)"
    )
    
    parser.add_argument(
        "--filter-model",
        help="Filter results to specific model ID"
    )
    
    parser.add_argument(
        "--filter-prompt", 
        help="Filter results to specific prompt ID"
    )
    
    parser.add_argument(
        "--min-success-rate",
        type=float,
        help="Filter out results below this success rate (0.0-1.0)"
    )
    
    parser.add_argument(
        "--sort-by",
        default="accuracy",
        help="Metric to sort results by (default: accuracy)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )

    args = parser.parse_args()
    
    # Load all result files
    try:
        results = load_results(args.results)
        print(f"Loaded {len(results)} result files")
    except Exception as e:
        print(f"Error loading results: {e}")
        sys.exit(1)
    
    # Apply filters
    if args.filter_model or args.filter_prompt or args.min_success_rate:
        results = apply_filters(results, args)
        print(f"Filtered to {len(results)} results")
    
    # Focus on specific metrics if specified
    focus_metrics = None
    if args.metrics:
        focus_metrics = [m.strip() for m in args.metrics.split(',')]
    
    try:
        # Generate report based on type
        if args.report == "summary":
            report = generate_summary_report(results, focus_metrics)
        elif args.report == "comparison":
            report = generate_comparison_report(results, focus_metrics, args.sort_by)
        elif args.report == "detailed":
            report = generate_detailed_report(results, focus_metrics)
        elif args.report == "statistical":
            report = generate_statistical_report(results, focus_metrics)
        
        # Output report
        output_report(report, args)
        
    except Exception as e:
        print(f"Error generating report: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def load_results(result_paths: List[str]) -> List[Dict[str, Any]]:
    """Load results from JSON files"""
    results = []
    
    for path_pattern in result_paths:
        # Handle glob patterns
        if '*' in path_pattern:
            paths = list(Path('.').glob(path_pattern))
        else:
            paths = [Path(path_pattern)]
        
        for path in paths:
            if not path.exists():
                print(f"Warning: Result file not found: {path}")
                continue
            
            try:
                with open(path, 'r') as f:
                    result = json.load(f)
                    result['_file_path'] = str(path)
                    results.append(result)
            except Exception as e:
                print(f"Warning: Failed to load {path}: {e}")
    
    return results


def apply_filters(results: List[Dict[str, Any]], args) -> List[Dict[str, Any]]:
    """Apply filters to results"""
    filtered = []
    
    for result in results:
        # Filter by success rate
        if args.min_success_rate:
            summary = result.get('summary', {})
            success_rate = summary.get('successful_tests', 0) / max(summary.get('total_tests', 1), 1)
            if success_rate < args.min_success_rate:
                continue
        
        # Filter test results by model/prompt
        if args.filter_model or args.filter_prompt:
            test_results = result.get('test_results', [])
            filtered_tests = []
            
            for test in test_results:
                if args.filter_model and test.get('model_id') != args.filter_model:
                    continue
                if args.filter_prompt and test.get('prompt_id') != args.filter_prompt:
                    continue
                filtered_tests.append(test)
            
            if filtered_tests:
                result = result.copy()
                result['test_results'] = filtered_tests
                # Recalculate summary
                result['summary'] = {
                    'total_tests': len(filtered_tests),
                    'successful_tests': len([t for t in filtered_tests if t.get('engine_response', {}).get('success', False)]),
                    'failed_tests': len([t for t in filtered_tests if not t.get('engine_response', {}).get('success', False)]),
                    'avg_latency': sum(t.get('engine_response', {}).get('latency_ms', 0) for t in filtered_tests) / len(filtered_tests) if filtered_tests else 0
                }
        
        filtered.append(result)
    
    return filtered


def generate_summary_report(results: List[Dict[str, Any]], focus_metrics: List[str] = None) -> Dict[str, Any]:
    """Generate summary report for all results"""
    report = {
        "report_type": "summary",
        "generated_at": datetime.now().isoformat(),
        "total_result_files": len(results),
        "results": []
    }
    
    for result in results:
        suite_summary = {
            "file_path": result.get('_file_path', 'unknown'),
            "suite_name": result.get('suite_name', 'unknown'),
            "timestamp": result.get('timestamp', ''),
            "execution_time": result.get('execution_time', 0),
            "summary": result.get('summary', {}),
            "aggregated_metrics": {}
        }
        
        # Extract relevant metrics
        agg_metrics = result.get('aggregated_metrics', {})
        if focus_metrics:
            for metric in focus_metrics:
                if metric in agg_metrics:
                    suite_summary["aggregated_metrics"][metric] = agg_metrics[metric]
        else:
            suite_summary["aggregated_metrics"] = agg_metrics
        
        report["results"].append(suite_summary)
    
    return report


def generate_comparison_report(results: List[Dict[str, Any]], focus_metrics: List[str] = None, sort_by: str = "accuracy") -> Dict[str, Any]:
    """Generate comparison report between multiple results"""
    if len(results) < 2:
        raise ValueError("Comparison report requires at least 2 result files")
    
    report = {
        "report_type": "comparison",
        "generated_at": datetime.now().isoformat(),
        "compared_suites": len(results),
        "sort_by": sort_by,
        "comparisons": []
    }
    
    # Extract metrics for comparison
    suite_metrics = []
    for result in results:
        metrics = result.get('aggregated_metrics', {})
        suite_info = {
            "suite_name": result.get('suite_name', 'unknown'),
            "file_path": result.get('_file_path', 'unknown'),
            "metrics": {}
        }
        
        if focus_metrics:
            for metric in focus_metrics:
                if metric in metrics:
                    suite_info["metrics"][metric] = metrics[metric].get('value', 0)
        else:
            for metric, data in metrics.items():
                suite_info["metrics"][metric] = data.get('value', 0)
        
        suite_metrics.append(suite_info)
    
    # Sort by specified metric
    if sort_by in suite_metrics[0]["metrics"]:
        suite_metrics.sort(key=lambda x: x["metrics"].get(sort_by, 0), reverse=True)
    
    # Generate pairwise comparisons
    for i in range(len(suite_metrics)):
        for j in range(i + 1, len(suite_metrics)):
            suite1 = suite_metrics[i]
            suite2 = suite_metrics[j]
            
            comparison = compare_suites(suite1, suite2)
            report["comparisons"].append(comparison)
    
    # Add ranking
    report["ranking"] = suite_metrics
    
    return report


def generate_detailed_report(results: List[Dict[str, Any]], focus_metrics: List[str] = None) -> Dict[str, Any]:
    """Generate detailed analysis report"""
    report = {
        "report_type": "detailed",
        "generated_at": datetime.now().isoformat(),
        "analysis": {}
    }
    
    # Analyze by dimensions
    all_test_results = []
    for result in results:
        suite_name = result.get('suite_name', 'unknown')
        for test in result.get('test_results', []):
            test['suite_name'] = suite_name
            all_test_results.append(test)
    
    if all_test_results:
        report["analysis"]["by_prompt"] = analyze_by_dimension(all_test_results, "prompt_id", focus_metrics)
        report["analysis"]["by_model"] = analyze_by_dimension(all_test_results, "model_id", focus_metrics)
        report["analysis"]["by_suite"] = analyze_by_dimension(all_test_results, "suite_name", focus_metrics)
        
        # Performance analysis
        report["analysis"]["performance"] = analyze_performance(all_test_results)
        
        # Error analysis
        report["analysis"]["errors"] = analyze_errors(all_test_results)
    
    return report


def generate_statistical_report(results: List[Dict[str, Any]], focus_metrics: List[str] = None) -> Dict[str, Any]:
    """Generate statistical analysis report"""
    report = {
        "report_type": "statistical",
        "generated_at": datetime.now().isoformat(),
        "statistics": {}
    }
    
    # Extract all metric values
    all_metrics = {}
    for result in results:
        for metric, data in result.get('aggregated_metrics', {}).items():
            if focus_metrics and metric not in focus_metrics:
                continue
            
            if metric not in all_metrics:
                all_metrics[metric] = []
            all_metrics[metric].append(data.get('value', 0))
    
    # Calculate statistics for each metric
    for metric, values in all_metrics.items():
        if len(values) > 0:
            import numpy as np
            report["statistics"][metric] = {
                "count": len(values),
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "median": float(np.median(values)),
                "q1": float(np.percentile(values, 25)),
                "q3": float(np.percentile(values, 75))
            }
    
    return report


def compare_suites(suite1: Dict[str, Any], suite2: Dict[str, Any]) -> Dict[str, Any]:
    """Compare two test suites"""
    comparison = {
        "suite1": suite1["suite_name"],
        "suite2": suite2["suite_name"],
        "metrics": {},
        "summary": {}
    }
    
    common_metrics = set(suite1["metrics"].keys()) & set(suite2["metrics"].keys())
    wins1 = 0
    wins2 = 0
    
    for metric in common_metrics:
        value1 = suite1["metrics"][metric]
        value2 = suite2["metrics"][metric]
        
        # Determine if higher is better (simplified heuristic)
        higher_is_better = metric in ["accuracy", "precision", "recall", "f1_score", "throughput", "parse_success_rate"]
        
        if higher_is_better:
            better = suite1["suite_name"] if value1 > value2 else suite2["suite_name"] if value2 > value1 else "tie"
            improvement = ((value1 - value2) / value2 * 100) if value2 != 0 else 0
        else:
            better = suite1["suite_name"] if value1 < value2 else suite2["suite_name"] if value2 < value1 else "tie"
            improvement = ((value2 - value1) / value1 * 100) if value1 != 0 else 0
        
        comparison["metrics"][metric] = {
            "value1": value1,
            "value2": value2,
            "better": better,
            "improvement_percent": improvement
        }
        
        if better == suite1["suite_name"]:
            wins1 += 1
        elif better == suite2["suite_name"]:
            wins2 += 1
    
    comparison["summary"] = {
        "total_metrics": len(common_metrics),
        f"{suite1['suite_name']}_wins": wins1,
        f"{suite2['suite_name']}_wins": wins2,
        "ties": len(common_metrics) - wins1 - wins2
    }
    
    return comparison


def analyze_by_dimension(test_results: List[Dict], dimension: str, focus_metrics: List[str] = None) -> Dict[str, Any]:
    """Analyze test results by a specific dimension (prompt_id, model_id, etc.)"""
    analysis = {}
    
    # Group by dimension
    groups = {}
    for test in test_results:
        key = test.get(dimension, 'unknown')
        if key not in groups:
            groups[key] = []
        groups[key].append(test)
    
    # Analyze each group
    for key, tests in groups.items():
        successful = len([t for t in tests if t.get('engine_response', {}).get('success', False)])
        total = len(tests)
        avg_latency = sum(t.get('engine_response', {}).get('latency_ms', 0) for t in tests) / total if total > 0 else 0
        
        group_analysis = {
            "total_tests": total,
            "successful_tests": successful,
            "success_rate": successful / total if total > 0 else 0,
            "avg_latency": avg_latency,
            "metrics": {}
        }
        
        # Calculate average metrics
        metric_sums = {}
        metric_counts = {}
        
        for test in tests:
            for metric, data in test.get('metrics', {}).items():
                if focus_metrics and metric not in focus_metrics:
                    continue
                
                value = data.get('value', 0)
                metric_sums[metric] = metric_sums.get(metric, 0) + value
                metric_counts[metric] = metric_counts.get(metric, 0) + 1
        
        for metric, total_value in metric_sums.items():
            group_analysis["metrics"][metric] = total_value / metric_counts[metric]
        
        analysis[key] = group_analysis
    
    return analysis


def analyze_performance(test_results: List[Dict]) -> Dict[str, Any]:
    """Analyze performance characteristics"""
    latencies = [t.get('engine_response', {}).get('latency_ms', 0) for t in test_results]
    successes = [t.get('engine_response', {}).get('success', False) for t in test_results]
    
    import numpy as np
    
    return {
        "latency": {
            "mean": float(np.mean(latencies)),
            "median": float(np.median(latencies)),
            "p95": float(np.percentile(latencies, 95)),
            "p99": float(np.percentile(latencies, 99))
        },
        "success_rate": sum(successes) / len(successes) if successes else 0,
        "total_tests": len(test_results)
    }


def analyze_errors(test_results: List[Dict]) -> Dict[str, Any]:
    """Analyze error patterns"""
    errors = []
    parse_errors = []
    
    for test in test_results:
        engine_response = test.get('engine_response', {})
        if engine_response.get('error'):
            errors.append(engine_response['error'])
        
        validation = test.get('validation_result', {})
        if not validation.get('valid', True):
            parse_errors.extend(validation.get('errors', []))
    
    # Count error types
    error_counts = {}
    for error in errors:
        error_type = error.split(':')[0] if ':' in error else error
        error_counts[error_type] = error_counts.get(error_type, 0) + 1
    
    parse_error_counts = {}
    for error in parse_errors:
        parse_error_counts[error] = parse_error_counts.get(error, 0) + 1
    
    return {
        "total_errors": len(errors),
        "total_parse_errors": len(parse_errors),
        "error_types": error_counts,
        "parse_error_types": parse_error_counts
    }


def output_report(report: Dict[str, Any], args):
    """Output report in specified format"""
    if args.format == "json":
        output_json_report(report, args.output)
    elif args.format == "csv":
        output_csv_report(report, args.output)
    elif args.format == "html":
        output_html_report(report, args.output)
    else:
        output_text_report(report, args.output)


def output_text_report(report: Dict[str, Any], output_dir: str = None):
    """Output report in text format"""
    lines = []
    
    report_type = report.get("report_type", "unknown").upper()
    lines.append(f"{report_type} REPORT")
    lines.append("=" * (len(report_type) + 7))
    lines.append(f"Generated: {report.get('generated_at', 'unknown')}")
    lines.append("")
    
    if report_type == "SUMMARY":
        lines.append(f"Total result files: {report.get('total_result_files', 0)}")
        lines.append("")
        
        for result in report.get("results", []):
            lines.append(f"Suite: {result['suite_name']}")
            lines.append(f"File: {Path(result['file_path']).name}")
            summary = result.get('summary', {})
            lines.append(f"  Tests: {summary.get('total_tests', 0)} (success: {summary.get('successful_tests', 0)})")
            lines.append(f"  Success rate: {summary.get('successful_tests', 0)/max(summary.get('total_tests', 1), 1)*100:.1f}%")
            lines.append(f"  Avg latency: {summary.get('avg_latency', 0):.1f}ms")
            
            for metric, data in result.get('aggregated_metrics', {}).items():
                value = data.get('value', 0) if isinstance(data, dict) else data
                lines.append(f"  {metric}: {value:.4f}")
            lines.append("")
    
    elif report_type == "COMPARISON":
        lines.append(f"Compared suites: {report.get('compared_suites', 0)}")
        lines.append(f"Sorted by: {report.get('sort_by', 'unknown')}")
        lines.append("")
        
        # Show ranking
        lines.append("RANKING:")
        for i, suite in enumerate(report.get('ranking', [])):
            lines.append(f"{i+1}. {suite['suite_name']}")
            for metric, value in suite['metrics'].items():
                lines.append(f"   {metric}: {value:.4f}")
        lines.append("")
        
        # Show comparisons
        lines.append("PAIRWISE COMPARISONS:")
        for comp in report.get('comparisons', []):
            lines.append(f"{comp['suite1']} vs {comp['suite2']}")
            summary = comp.get('summary', {})
            lines.append(f"  {comp['suite1']} wins: {summary.get(f\"{comp['suite1']}_wins\", 0)}")
            lines.append(f"  {comp['suite2']} wins: {summary.get(f\"{comp['suite2']}_wins\", 0)}")
            lines.append(f"  Ties: {summary.get('ties', 0)}")
            lines.append("")
    
    # Output to file or console
    content = "\n".join(lines)
    
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"analysis_{report_type.lower()}_{timestamp}.txt"
        
        with open(output_path / filename, 'w') as f:
            f.write(content)
        
        print(f"Report saved to: {output_path / filename}")
    else:
        print(content)


def output_json_report(report: Dict[str, Any], output_dir: str = None):
    """Output report in JSON format"""
    content = json.dumps(report, indent=2)
    
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"analysis_{report['report_type']}_{timestamp}.json"
        
        with open(output_path / filename, 'w') as f:
            f.write(content)
        
        print(f"Report saved to: {output_path / filename}")
    else:
        print(content)


def output_csv_report(report: Dict[str, Any], output_dir: str = None):
    """Output report in CSV format (simplified)"""
    try:
        import pandas as pd
        
        if report["report_type"] == "summary":
            # Create DataFrame from summary data
            data = []
            for result in report.get("results", []):
                row = {
                    "suite_name": result["suite_name"],
                    "execution_time": result.get("execution_time", 0),
                    **result.get("summary", {}),
                    **{k: v.get("value", 0) if isinstance(v, dict) else v 
                       for k, v in result.get("aggregated_metrics", {}).items()}
                }
                data.append(row)
            
            df = pd.DataFrame(data)
        else:
            print("CSV output only supported for summary reports")
            return
        
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"analysis_{report['report_type']}_{timestamp}.csv"
            
            df.to_csv(output_path / filename, index=False)
            print(f"Report saved to: {output_path / filename}")
        else:
            print(df.to_string(index=False))
    
    except ImportError:
        print("pandas not available for CSV output")


def output_html_report(report: Dict[str, Any], output_dir: str = None):
    """Output report in HTML format"""
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>{report['report_type'].title()} Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            .metric {{ font-weight: bold; }}
        </style>
    </head>
    <body>
        <h1>{report['report_type'].title()} Report</h1>
        <p>Generated: {report.get('generated_at', 'unknown')}</p>
        
        <pre>{json.dumps(report, indent=2)}</pre>
    </body>
    </html>
    """
    
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"analysis_{report['report_type']}_{timestamp}.html"
        
        with open(output_path / filename, 'w') as f:
            f.write(html_content)
        
        print(f"Report saved to: {output_path / filename}")
    else:
        print(html_content)


if __name__ == "__main__":
    main()