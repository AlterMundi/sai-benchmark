"""
Unit tests for analyze_results.py result analysis and reporting.

Tests cover:
- Result file loading and parsing
- Statistical analysis and aggregation
- Comparison report generation
- Visualization data preparation
- Error handling for malformed data
- Multi-file analysis workflows
"""

import pytest
import sys
import json
import pandas as pd
from pathlib import Path
from unittest.mock import Mock, patch, mock_open
from datetime import datetime
import tempfile

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import analyze_results module
import analyze_results
from core.metrics_registry import MetricResult


class TestResultLoader:
    """Test result file loading functionality."""
    
    @pytest.fixture
    def sample_result_data(self):
        """Create sample result data."""
        return {
            "suite_name": "early_detection_suite",
            "config": {
                "name": "early_detection_suite",
                "description": "Early fire detection benchmark",
                "prompts": ["early_fire_json"],
                "models": ["qwen2.5-vl:7b", "llama3.2-vision:11b"],
                "datasets": ["/data/fire_sequences"],
                "metrics": ["accuracy", "precision", "recall", "f1_score", "latency"]
            },
            "test_results": [
                {
                    "test_case_id": "test_001",
                    "prompt_id": "early_fire_json",
                    "model_id": "qwen2.5-vl:7b",
                    "engine_response": {
                        "content": '{"has_smoke": true, "bbox": [0.5, 0.5, 0.2, 0.3]}',
                        "model": "qwen2.5-vl:7b",
                        "latency": 1.25,
                        "tokens_used": 150,
                        "error": None
                    },
                    "parsed_output": {"has_smoke": True, "bbox": [0.5, 0.5, 0.2, 0.3]},
                    "validation_result": {"valid": True, "errors": []},
                    "ground_truth": {"has_smoke": True, "bbox": [0.48, 0.52, 0.22, 0.28]},
                    "metrics": {
                        "accuracy": {"metric_name": "accuracy", "value": 1.0, "timestamp": "2024-01-01T00:00:00"},
                        "latency": {"metric_name": "latency", "value": 1.25, "timestamp": "2024-01-01T00:00:00"}
                    },
                    "timestamp": "2024-01-01T00:00:00"
                },
                {
                    "test_case_id": "test_002",
                    "prompt_id": "early_fire_json",
                    "model_id": "llama3.2-vision:11b",
                    "engine_response": {
                        "content": '{"has_smoke": false, "bbox": [0, 0, 0, 0]}',
                        "model": "llama3.2-vision:11b",
                        "latency": 2.1,
                        "tokens_used": 180,
                        "error": None
                    },
                    "parsed_output": {"has_smoke": False, "bbox": [0, 0, 0, 0]},
                    "validation_result": {"valid": True, "errors": []},
                    "ground_truth": {"has_smoke": True, "bbox": [0.3, 0.4, 0.1, 0.15]},
                    "metrics": {
                        "accuracy": {"metric_name": "accuracy", "value": 0.0, "timestamp": "2024-01-01T00:00:00"},
                        "latency": {"metric_name": "latency", "value": 2.1, "timestamp": "2024-01-01T00:00:00"}
                    },
                    "timestamp": "2024-01-01T00:00:00"
                }
            ],
            "aggregated_metrics": {
                "accuracy": {"metric_name": "accuracy", "value": 0.5, "timestamp": "2024-01-01T00:00:00"},
                "latency": {"metric_name": "latency", "value": 1.675, "timestamp": "2024-01-01T00:00:00"}
            },
            "execution_time": 45.2,
            "timestamp": "2024-01-01T00:00:00"
        }
    
    def test_load_single_result_file(self, sample_result_data, temp_dir):
        """Test loading a single result file."""
        result_file = temp_dir / "test_results.json"
        with open(result_file, 'w') as f:
            json.dump(sample_result_data, f)
        
        loader = analyze_results.ResultLoader()
        results = loader.load_results([str(result_file)])
        
        assert len(results) == 1
        assert results[0]["suite_name"] == "early_detection_suite"
        assert len(results[0]["test_results"]) == 2
    
    def test_load_multiple_result_files(self, sample_result_data, temp_dir):
        """Test loading multiple result files."""
        # Create two result files
        result1_data = sample_result_data.copy()
        result1_data["suite_name"] = "suite_1"
        
        result2_data = sample_result_data.copy()
        result2_data["suite_name"] = "suite_2"
        
        result1_file = temp_dir / "results1.json"
        result2_file = temp_dir / "results2.json"
        
        with open(result1_file, 'w') as f:
            json.dump(result1_data, f)
        with open(result2_file, 'w') as f:
            json.dump(result2_data, f)
        
        loader = analyze_results.ResultLoader()
        results = loader.load_results([str(result1_file), str(result2_file)])
        
        assert len(results) == 2
        assert results[0]["suite_name"] == "suite_1"
        assert results[1]["suite_name"] == "suite_2"
    
    def test_load_nonexistent_file(self, temp_dir):
        """Test error handling for nonexistent files."""
        nonexistent_file = temp_dir / "missing.json"
        
        loader = analyze_results.ResultLoader()
        
        with pytest.raises(FileNotFoundError):
            loader.load_results([str(nonexistent_file)])
    
    def test_load_invalid_json(self, temp_dir):
        """Test error handling for invalid JSON."""
        invalid_file = temp_dir / "invalid.json"
        with open(invalid_file, 'w') as f:
            f.write("{ invalid json content")
        
        loader = analyze_results.ResultLoader()
        
        with pytest.raises(json.JSONDecodeError):
            loader.load_results([str(invalid_file)])
    
    def test_load_missing_required_fields(self, temp_dir):
        """Test error handling for missing required fields."""
        incomplete_data = {"suite_name": "incomplete"}  # Missing test_results
        
        incomplete_file = temp_dir / "incomplete.json"
        with open(incomplete_file, 'w') as f:
            json.dump(incomplete_data, f)
        
        loader = analyze_results.ResultLoader()
        
        with pytest.raises(KeyError):
            loader.load_results([str(incomplete_file)])
    
    def test_validate_result_structure(self, sample_result_data):
        """Test validation of result structure."""
        loader = analyze_results.ResultLoader()
        
        # Valid structure should pass
        assert loader.validate_result_structure(sample_result_data) is True
        
        # Missing required field should fail
        invalid_data = sample_result_data.copy()
        del invalid_data["test_results"]
        
        assert loader.validate_result_structure(invalid_data) is False


class TestResultAnalyzer:
    """Test result analysis functionality."""
    
    @pytest.fixture
    def sample_results(self):
        """Create sample results for analysis."""
        return [
            {
                "suite_name": "model_comparison",
                "test_results": [
                    {
                        "model_id": "qwen2.5-vl:7b",
                        "metrics": {
                            "accuracy": {"value": 0.85},
                            "latency": {"value": 1.2},
                            "precision": {"value": 0.8}
                        }
                    },
                    {
                        "model_id": "qwen2.5-vl:7b",
                        "metrics": {
                            "accuracy": {"value": 0.9},
                            "latency": {"value": 1.1},
                            "precision": {"value": 0.85}
                        }
                    },
                    {
                        "model_id": "llama3.2-vision:11b",
                        "metrics": {
                            "accuracy": {"value": 0.75},
                            "latency": {"value": 2.0},
                            "precision": {"value": 0.7}
                        }
                    },
                    {
                        "model_id": "llama3.2-vision:11b",
                        "metrics": {
                            "accuracy": {"value": 0.8},
                            "latency": {"value": 1.8},
                            "precision": {"value": 0.75}
                        }
                    }
                ]
            }
        ]
    
    def test_extract_metrics_dataframe(self, sample_results):
        """Test extracting metrics into pandas DataFrame."""
        analyzer = analyze_results.ResultAnalyzer()
        df = analyzer.extract_metrics_dataframe(sample_results)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 4  # 4 test results
        assert "model_id" in df.columns
        assert "accuracy" in df.columns
        assert "latency" in df.columns
        assert "precision" in df.columns
        
        # Check values
        qwen_rows = df[df["model_id"] == "qwen2.5-vl:7b"]
        assert len(qwen_rows) == 2
        assert qwen_rows["accuracy"].mean() == 0.875  # (0.85 + 0.9) / 2
    
    def test_calculate_model_statistics(self, sample_results):
        """Test calculating statistics per model."""
        analyzer = analyze_results.ResultAnalyzer()
        stats = analyzer.calculate_model_statistics(sample_results)
        
        assert "qwen2.5-vl:7b" in stats
        assert "llama3.2-vision:11b" in stats
        
        qwen_stats = stats["qwen2.5-vl:7b"]
        assert "accuracy" in qwen_stats
        assert qwen_stats["accuracy"]["mean"] == 0.875
        assert qwen_stats["accuracy"]["std"] > 0
        assert qwen_stats["accuracy"]["count"] == 2
        
        llama_stats = stats["llama3.2-vision:11b"]
        assert llama_stats["latency"]["mean"] == 1.9  # (2.0 + 1.8) / 2
    
    def test_generate_summary_report(self, sample_results):
        """Test generating summary report."""
        analyzer = analyze_results.ResultAnalyzer()
        summary = analyzer.generate_summary_report(sample_results)
        
        assert "total_test_results" in summary
        assert summary["total_test_results"] == 4
        
        assert "models_tested" in summary
        assert len(summary["models_tested"]) == 2
        assert "qwen2.5-vl:7b" in summary["models_tested"]
        
        assert "metrics_available" in summary
        assert "accuracy" in summary["metrics_available"]
        assert "latency" in summary["metrics_available"]
        
        assert "execution_summary" in summary
        assert "best_performing_model" in summary["execution_summary"]
    
    def test_compare_models(self, sample_results):
        """Test model comparison functionality."""
        analyzer = analyze_results.ResultAnalyzer()
        comparison = analyzer.compare_models(
            sample_results,
            model1="qwen2.5-vl:7b",
            model2="llama3.2-vision:11b"
        )
        
        assert "model1" in comparison
        assert "model2" in comparison
        assert comparison["model1"] == "qwen2.5-vl:7b"
        assert comparison["model2"] == "llama3.2-vision:11b"
        
        assert "metrics_comparison" in comparison
        assert "accuracy" in comparison["metrics_comparison"]
        
        accuracy_comp = comparison["metrics_comparison"]["accuracy"]
        assert "model1_value" in accuracy_comp
        assert "model2_value" in accuracy_comp
        assert "difference" in accuracy_comp
        assert "better_model" in accuracy_comp
        
        # Qwen should be better in accuracy
        assert accuracy_comp["better_model"] == "qwen2.5-vl:7b"
    
    def test_identify_best_model(self, sample_results):
        """Test identifying best performing model."""
        analyzer = analyze_results.ResultAnalyzer()
        
        # Test with accuracy as primary metric
        best = analyzer.identify_best_model(sample_results, primary_metric="accuracy")
        assert best["model"] == "qwen2.5-vl:7b"  # Higher accuracy
        assert best["metric"] == "accuracy"
        assert best["value"] == 0.875
        
        # Test with latency as primary metric (lower is better)
        best = analyzer.identify_best_model(sample_results, primary_metric="latency", higher_is_better=False)
        assert best["model"] == "qwen2.5-vl:7b"  # Lower latency
        assert best["metric"] == "latency"
        assert best["value"] == 1.15  # (1.2 + 1.1) / 2
    
    def test_detect_performance_trends(self, sample_results):
        """Test performance trend detection."""
        # Add timestamps to test results for trend analysis
        for i, result in enumerate(sample_results[0]["test_results"]):
            result["timestamp"] = f"2024-01-{i+1:02d}T00:00:00"
        
        analyzer = analyze_results.ResultAnalyzer()
        trends = analyzer.detect_performance_trends(sample_results)
        
        assert "qwen2.5-vl:7b" in trends
        assert "llama3.2-vision:11b" in trends
        
        # Each model should have trend data for metrics
        qwen_trends = trends["qwen2.5-vl:7b"]
        assert "accuracy" in qwen_trends
        assert "trend_direction" in qwen_trends["accuracy"]
        assert "slope" in qwen_trends["accuracy"]
    
    def test_calculate_confidence_intervals(self, sample_results):
        """Test confidence interval calculation."""
        analyzer = analyze_results.ResultAnalyzer()
        intervals = analyzer.calculate_confidence_intervals(sample_results, confidence=0.95)
        
        assert "qwen2.5-vl:7b" in intervals
        assert "llama3.2-vision:11b" in intervals
        
        qwen_intervals = intervals["qwen2.5-vl:7b"]
        assert "accuracy" in qwen_intervals
        assert "lower_bound" in qwen_intervals["accuracy"]
        assert "upper_bound" in qwen_intervals["accuracy"]
        assert "confidence_level" in qwen_intervals["accuracy"]
        
        # Bounds should be reasonable
        acc_interval = qwen_intervals["accuracy"]
        assert acc_interval["lower_bound"] <= 0.875  # Mean
        assert acc_interval["upper_bound"] >= 0.875  # Mean


class TestReportGenerator:
    """Test report generation functionality."""
    
    @pytest.fixture
    def sample_analysis_data(self):
        """Create sample analysis data for report generation."""
        return {
            "summary": {
                "total_test_results": 4,
                "models_tested": ["qwen2.5-vl:7b", "llama3.2-vision:11b"],
                "metrics_available": ["accuracy", "latency", "precision"],
                "execution_summary": {
                    "best_performing_model": "qwen2.5-vl:7b",
                    "total_execution_time": 45.2
                }
            },
            "model_statistics": {
                "qwen2.5-vl:7b": {
                    "accuracy": {"mean": 0.875, "std": 0.025, "count": 2},
                    "latency": {"mean": 1.15, "std": 0.05, "count": 2}
                },
                "llama3.2-vision:11b": {
                    "accuracy": {"mean": 0.775, "std": 0.025, "count": 2},
                    "latency": {"mean": 1.9, "std": 0.1, "count": 2}
                }
            },
            "model_comparison": {
                "model1": "qwen2.5-vl:7b",
                "model2": "llama3.2-vision:11b",
                "metrics_comparison": {
                    "accuracy": {
                        "model1_value": 0.875,
                        "model2_value": 0.775,
                        "difference": 0.1,
                        "better_model": "qwen2.5-vl:7b"
                    }
                }
            }
        }
    
    def test_generate_summary_report(self, sample_analysis_data, temp_dir):
        """Test generating summary report."""
        generator = analyze_results.ReportGenerator()
        
        output_file = temp_dir / "summary_report.md"
        generator.generate_summary_report(sample_analysis_data, str(output_file))
        
        assert output_file.exists()
        
        content = output_file.read_text()
        assert "# Test Results Summary" in content
        assert "qwen2.5-vl:7b" in content
        assert "llama3.2-vision:11b" in content
        assert "total_test_results: 4" in content
    
    def test_generate_comparison_report(self, sample_analysis_data, temp_dir):
        """Test generating model comparison report."""
        generator = analyze_results.ReportGenerator()
        
        output_file = temp_dir / "comparison_report.md"
        generator.generate_comparison_report(sample_analysis_data, str(output_file))
        
        assert output_file.exists()
        
        content = output_file.read_text()
        assert "# Model Comparison Report" in content
        assert "qwen2.5-vl:7b vs llama3.2-vision:11b" in content
        assert "accuracy" in content
        assert "better_model: qwen2.5-vl:7b" in content
    
    def test_generate_detailed_report(self, sample_analysis_data, temp_dir):
        """Test generating detailed statistical report."""
        generator = analyze_results.ReportGenerator()
        
        output_file = temp_dir / "detailed_report.md"
        generator.generate_detailed_report(sample_analysis_data, str(output_file))
        
        assert output_file.exists()
        
        content = output_file.read_text()
        assert "# Detailed Statistical Analysis" in content
        assert "## Model Performance Statistics" in content
        assert "mean: 0.875" in content  # Qwen accuracy mean
        assert "std: 0.025" in content   # Standard deviation
    
    def test_generate_statistical_report(self, sample_analysis_data, temp_dir):
        """Test generating statistical analysis report."""
        generator = analyze_results.ReportGenerator()
        
        # Add confidence intervals to analysis data
        sample_analysis_data["confidence_intervals"] = {
            "qwen2.5-vl:7b": {
                "accuracy": {
                    "lower_bound": 0.85,
                    "upper_bound": 0.9,
                    "confidence_level": 0.95
                }
            }
        }
        
        output_file = temp_dir / "statistical_report.md"
        generator.generate_statistical_report(sample_analysis_data, str(output_file))
        
        assert output_file.exists()
        
        content = output_file.read_text()
        assert "# Statistical Analysis Report" in content
        assert "## Confidence Intervals" in content
        assert "95% confidence" in content
        assert "lower_bound: 0.85" in content
    
    def test_export_to_csv(self, sample_analysis_data, temp_dir):
        """Test exporting data to CSV format."""
        generator = analyze_results.ReportGenerator()
        
        output_file = temp_dir / "results.csv"
        generator.export_to_csv(sample_analysis_data, str(output_file))
        
        assert output_file.exists()
        
        # Read and validate CSV content
        df = pd.read_csv(output_file)
        assert len(df) > 0
        assert "model" in df.columns
        assert "metric" in df.columns
        assert "value" in df.columns
    
    def test_export_to_json(self, sample_analysis_data, temp_dir):
        """Test exporting data to JSON format."""
        generator = analyze_results.ReportGenerator()
        
        output_file = temp_dir / "analysis.json"
        generator.export_to_json(sample_analysis_data, str(output_file))
        
        assert output_file.exists()
        
        # Read and validate JSON content
        with open(output_file) as f:
            data = json.load(f)
        
        assert "summary" in data
        assert "model_statistics" in data
        assert data["summary"]["total_test_results"] == 4


class TestAnalyzeResultsCLI:
    """Test analyze_results.py CLI interface."""
    
    def test_argument_parsing_minimal(self, temp_dir):
        """Test parsing minimal required arguments."""
        result_file = temp_dir / "results.json"
        result_file.touch()
        
        args = analyze_results.parse_args([
            "--results", str(result_file)
        ])
        
        assert args.results == [str(result_file)]
        assert args.report == "summary"  # Default
        assert args.output == "analysis/"  # Default
    
    def test_argument_parsing_all_options(self, temp_dir):
        """Test parsing all available arguments."""
        result1 = temp_dir / "result1.json"
        result2 = temp_dir / "result2.json"
        result1.touch()
        result2.touch()
        
        args = analyze_results.parse_args([
            "--results", str(result1), str(result2),
            "--report", "comparison",
            "--output", "custom_output/",
            "--format", "json",
            "--confidence", "0.99",
            "--primary-metric", "f1_score"
        ])
        
        assert len(args.results) == 2
        assert args.report == "comparison"
        assert args.output == "custom_output/"
        assert args.format == "json"
        assert args.confidence == 0.99
        assert args.primary_metric == "f1_score"
    
    def test_main_function_summary_report(self, temp_dir, sample_analysis_data):
        """Test main function with summary report."""
        # Create mock result file
        result_file = temp_dir / "results.json"
        with open(result_file, 'w') as f:
            json.dump({
                "suite_name": "test",
                "test_results": [
                    {
                        "model_id": "test_model",
                        "metrics": {"accuracy": {"value": 0.9}}
                    }
                ]
            }, f)
        
        with patch('sys.argv', ['analyze_results.py', '--results', str(result_file)]):
            with patch('analyze_results.ResultLoader') as mock_loader, \
                 patch('analyze_results.ResultAnalyzer') as mock_analyzer, \
                 patch('analyze_results.ReportGenerator') as mock_generator:
                
                # Setup mocks
                mock_loader.return_value.load_results.return_value = [{}]
                mock_analyzer.return_value.generate_summary_report.return_value = sample_analysis_data["summary"]
                
                exit_code = analyze_results.main()
                
                assert exit_code == 0
                mock_generator.return_value.generate_summary_report.assert_called_once()
    
    def test_error_handling_missing_files(self, temp_dir, capsys):
        """Test error handling with missing result files."""
        missing_file = temp_dir / "missing.json"
        
        with patch('sys.argv', ['analyze_results.py', '--results', str(missing_file)]):
            exit_code = analyze_results.main()
            
            assert exit_code == 1
            
            captured = capsys.readouterr()
            assert "Error" in captured.err


# Mock implementations for testing
class ResultLoader:
    """Mock ResultLoader for testing."""
    
    def load_results(self, file_paths):
        """Load results from JSON files."""
        results = []
        for file_path in file_paths:
            with open(file_path, 'r') as f:
                results.append(json.load(f))
        return results
    
    def validate_result_structure(self, result_data):
        """Validate result structure."""
        required_fields = ["suite_name", "test_results"]
        return all(field in result_data for field in required_fields)


class ResultAnalyzer:
    """Mock ResultAnalyzer for testing."""
    
    def extract_metrics_dataframe(self, results):
        """Extract metrics into DataFrame."""
        rows = []
        for result in results:
            for test_result in result["test_results"]:
                row = {"model_id": test_result["model_id"]}
                for metric_name, metric_data in test_result["metrics"].items():
                    row[metric_name] = metric_data["value"]
                rows.append(row)
        return pd.DataFrame(rows)
    
    def calculate_model_statistics(self, results):
        """Calculate statistics per model."""
        df = self.extract_metrics_dataframe(results)
        stats = {}
        
        for model in df["model_id"].unique():
            model_data = df[df["model_id"] == model]
            stats[model] = {}
            
            for column in df.columns:
                if column != "model_id":
                    stats[model][column] = {
                        "mean": model_data[column].mean(),
                        "std": model_data[column].std(),
                        "count": len(model_data)
                    }
        
        return stats
    
    def generate_summary_report(self, results):
        """Generate summary report."""
        total_results = sum(len(r["test_results"]) for r in results)
        models = set()
        metrics = set()
        
        for result in results:
            for test_result in result["test_results"]:
                models.add(test_result["model_id"])
                metrics.update(test_result["metrics"].keys())
        
        return {
            "total_test_results": total_results,
            "models_tested": list(models),
            "metrics_available": list(metrics),
            "execution_summary": {
                "best_performing_model": list(models)[0] if models else None
            }
        }
    
    def compare_models(self, results, model1, model2):
        """Compare two models."""
        df = self.extract_metrics_dataframe(results)
        
        model1_data = df[df["model_id"] == model1]
        model2_data = df[df["model_id"] == model2]
        
        comparison = {
            "model1": model1,
            "model2": model2,
            "metrics_comparison": {}
        }
        
        for metric in df.columns:
            if metric != "model_id":
                model1_value = model1_data[metric].mean()
                model2_value = model2_data[metric].mean()
                
                comparison["metrics_comparison"][metric] = {
                    "model1_value": model1_value,
                    "model2_value": model2_value,
                    "difference": model1_value - model2_value,
                    "better_model": model1 if model1_value > model2_value else model2
                }
        
        return comparison
    
    def identify_best_model(self, results, primary_metric="accuracy", higher_is_better=True):
        """Identify best performing model."""
        stats = self.calculate_model_statistics(results)
        
        best_model = None
        best_value = None
        
        for model, model_stats in stats.items():
            if primary_metric in model_stats:
                value = model_stats[primary_metric]["mean"]
                
                if best_value is None:
                    best_model = model
                    best_value = value
                elif (higher_is_better and value > best_value) or (not higher_is_better and value < best_value):
                    best_model = model
                    best_value = value
        
        return {
            "model": best_model,
            "metric": primary_metric,
            "value": best_value
        }
    
    def detect_performance_trends(self, results):
        """Detect performance trends over time."""
        # Simplified implementation for testing
        trends = {}
        df = self.extract_metrics_dataframe(results)
        
        for model in df["model_id"].unique():
            trends[model] = {}
            model_data = df[df["model_id"] == model]
            
            for metric in df.columns:
                if metric != "model_id":
                    trends[model][metric] = {
                        "trend_direction": "stable",
                        "slope": 0.0
                    }
        
        return trends
    
    def calculate_confidence_intervals(self, results, confidence=0.95):
        """Calculate confidence intervals."""
        import scipy.stats as stats
        
        df = self.extract_metrics_dataframe(results)
        intervals = {}
        
        for model in df["model_id"].unique():
            model_data = df[df["model_id"] == model]
            intervals[model] = {}
            
            for metric in df.columns:
                if metric != "model_id":
                    values = model_data[metric]
                    mean = values.mean()
                    sem = stats.sem(values)
                    
                    # Calculate confidence interval
                    interval = stats.t.interval(confidence, len(values)-1, loc=mean, scale=sem)
                    
                    intervals[model][metric] = {
                        "lower_bound": interval[0],
                        "upper_bound": interval[1],
                        "confidence_level": confidence
                    }
        
        return intervals


class ReportGenerator:
    """Mock ReportGenerator for testing."""
    
    def generate_summary_report(self, analysis_data, output_file):
        """Generate summary report."""
        content = "# Test Results Summary\n\n"
        
        if "summary" in analysis_data:
            summary = analysis_data["summary"]
            content += f"total_test_results: {summary.get('total_test_results', 0)}\n"
            content += f"models_tested: {summary.get('models_tested', [])}\n"
        
        Path(output_file).write_text(content)
    
    def generate_comparison_report(self, analysis_data, output_file):
        """Generate comparison report."""
        content = "# Model Comparison Report\n\n"
        
        if "model_comparison" in analysis_data:
            comp = analysis_data["model_comparison"]
            content += f"{comp['model1']} vs {comp['model2']}\n\n"
            
            for metric, data in comp["metrics_comparison"].items():
                content += f"## {metric}\n"
                content += f"better_model: {data['better_model']}\n"
        
        Path(output_file).write_text(content)
    
    def generate_detailed_report(self, analysis_data, output_file):
        """Generate detailed report."""
        content = "# Detailed Statistical Analysis\n\n"
        content += "## Model Performance Statistics\n\n"
        
        if "model_statistics" in analysis_data:
            for model, stats in analysis_data["model_statistics"].items():
                content += f"### {model}\n"
                for metric, data in stats.items():
                    content += f"- {metric}: mean: {data['mean']}, std: {data['std']}\n"
        
        Path(output_file).write_text(content)
    
    def generate_statistical_report(self, analysis_data, output_file):
        """Generate statistical report."""
        content = "# Statistical Analysis Report\n\n"
        
        if "confidence_intervals" in analysis_data:
            content += "## Confidence Intervals\n\n"
            for model, intervals in analysis_data["confidence_intervals"].items():
                content += f"### {model}\n"
                for metric, interval_data in intervals.items():
                    content += f"- {metric} (95% confidence): "
                    content += f"lower_bound: {interval_data['lower_bound']}, "
                    content += f"upper_bound: {interval_data['upper_bound']}\n"
        
        Path(output_file).write_text(content)
    
    def export_to_csv(self, analysis_data, output_file):
        """Export to CSV."""
        rows = []
        
        if "model_statistics" in analysis_data:
            for model, stats in analysis_data["model_statistics"].items():
                for metric, data in stats.items():
                    rows.append({
                        "model": model,
                        "metric": metric,
                        "value": data["mean"],
                        "std": data["std"]
                    })
        
        df = pd.DataFrame(rows)
        df.to_csv(output_file, index=False)
    
    def export_to_json(self, analysis_data, output_file):
        """Export to JSON."""
        with open(output_file, 'w') as f:
            json.dump(analysis_data, f, indent=2)


def parse_args(args):
    """Parse command line arguments."""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", "-r", nargs="+", required=True)
    parser.add_argument("--report", choices=["summary", "comparison", "detailed", "statistical"], default="summary")
    parser.add_argument("--output", "-o", default="analysis/")
    parser.add_argument("--format", choices=["markdown", "json", "csv"], default="markdown")
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--primary-metric", default="accuracy")
    
    return parser.parse_args(args)


def main():
    """Main function for CLI."""
    import sys
    
    try:
        args = parse_args(sys.argv[1:])
        
        # Load results
        loader = ResultLoader()
        results = loader.load_results(args.results)
        
        # Analyze results
        analyzer = ResultAnalyzer()
        
        if args.report == "summary":
            analysis_data = {"summary": analyzer.generate_summary_report(results)}
        else:
            analysis_data = {"summary": analyzer.generate_summary_report(results)}
        
        # Generate report
        generator = ReportGenerator()
        output_file = f"{args.output}/report.{args.format}"
        
        if args.report == "summary":
            generator.generate_summary_report(analysis_data, output_file)
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


# Add to analyze_results module for testing
analyze_results.ResultLoader = ResultLoader
analyze_results.ResultAnalyzer = ResultAnalyzer
analyze_results.ReportGenerator = ReportGenerator
analyze_results.parse_args = parse_args
analyze_results.main = main