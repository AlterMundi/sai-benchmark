"""
Unit tests for MetricsRegistry system.

Tests cover:
- Metric registration and configuration
- Metric calculation functions
- Result aggregation and comparison
- Error handling and edge cases
- IOU calculations for bounding boxes
"""

import pytest
import sys
import numpy as np
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, patch
from hypothesis import given, strategies as st, assume

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.metrics_registry import (
    MetricsRegistry, MetricConfig, MetricResult, MetricType
)


class TestMetricType:
    """Test MetricType enum."""
    
    def test_metric_type_values(self):
        """Test that all expected metric types are defined."""
        expected_metrics = [
            "accuracy", "precision", "recall", "f1_score",
            "latency", "throughput", "cost",
            "bbox_iou", "bbox_precision", "bbox_recall",
            "confidence", "error_rate", "parse_success_rate"
        ]
        
        actual_values = [mt.value for mt in MetricType]
        for expected in expected_metrics:
            assert expected in actual_values


class TestMetricResult:
    """Test MetricResult dataclass functionality."""
    
    def test_metric_result_creation(self):
        """Test creating MetricResult with all fields."""
        result = MetricResult(
            metric_name="accuracy",
            value=0.95,
            details={"threshold": 0.5, "samples": 100}
        )
        
        assert result.metric_name == "accuracy"
        assert result.value == 0.95
        assert result.details["samples"] == 100
        assert isinstance(result.timestamp, datetime)
    
    def test_metric_result_to_dict(self):
        """Test converting MetricResult to dictionary."""
        result = MetricResult(
            metric_name="precision",
            value=0.87,
            details={"tp": 87, "fp": 13}
        )
        
        result_dict = result.to_dict()
        
        assert result_dict["metric_name"] == "precision"
        assert result_dict["value"] == 0.87
        assert result_dict["details"]["tp"] == 87
        assert "timestamp" in result_dict
        assert isinstance(result_dict["timestamp"], str)


class TestMetricConfig:
    """Test MetricConfig dataclass functionality."""
    
    def test_metric_config_creation(self):
        """Test creating MetricConfig with all fields."""
        def dummy_function(predictions, ground_truth, **kwargs):
            return 0.5
        
        config = MetricConfig(
            metric_type=MetricType.ACCURACY,
            function=dummy_function,
            description="Test metric",
            higher_is_better=True,
            requires_ground_truth=True,
            requires_bbox=False,
            aggregation_method="mean"
        )
        
        assert config.metric_type == MetricType.ACCURACY
        assert config.function == dummy_function
        assert config.higher_is_better is True
        assert config.aggregation_method == "mean"
    
    def test_metric_config_defaults(self):
        """Test MetricConfig with default values."""
        config = MetricConfig(
            metric_type=MetricType.LATENCY,
            function=lambda x, y: 0.0,
            description="Latency metric"
        )
        
        assert config.higher_is_better is True  # Default
        assert config.requires_ground_truth is True  # Default
        assert config.requires_bbox is False  # Default
        assert config.aggregation_method == "mean"  # Default


class TestMetricsRegistry:
    """Test MetricsRegistry functionality."""
    
    @pytest.fixture
    def empty_registry(self):
        """Create an empty registry without builtin metrics."""
        with patch.object(MetricsRegistry, '_register_builtin_metrics'):
            registry = MetricsRegistry()
        return registry
    
    @pytest.fixture
    def populated_registry(self):
        """Create a registry with builtin metrics."""
        registry = MetricsRegistry()
        return registry
    
    @pytest.fixture
    def sample_predictions(self):
        """Create sample predictions for testing."""
        return [
            {"has_smoke": True, "bbox": [0.5, 0.5, 0.2, 0.3], "latency_ms": 100, "confidence": 0.9},
            {"has_smoke": False, "bbox": [0, 0, 0, 0], "latency_ms": 95, "confidence": 0.8},
            {"has_smoke": True, "bbox": [0.3, 0.4, 0.15, 0.25], "latency_ms": 105, "confidence": 0.85},
            {"has_smoke": False, "bbox": [0, 0, 0, 0], "latency_ms": 98, "confidence": 0.75},
        ]
    
    @pytest.fixture
    def sample_ground_truth(self):
        """Create sample ground truth for testing."""
        return [
            {"has_smoke": True, "bbox": [0.48, 0.52, 0.22, 0.28]},
            {"has_smoke": False, "bbox": [0, 0, 0, 0]},
            {"has_smoke": False, "bbox": [0, 0, 0, 0]},  # False negative case
            {"has_smoke": False, "bbox": [0, 0, 0, 0]},
        ]
    
    def test_registry_initialization(self):
        """Test metrics registry initialization."""
        registry = MetricsRegistry()
        assert len(registry.metrics) > 0  # Should have builtin metrics
        assert MetricType.ACCURACY.value in registry.metrics
        assert MetricType.PRECISION.value in registry.metrics
    
    def test_register_metric(self, empty_registry):
        """Test registering a custom metric."""
        def custom_metric(predictions, ground_truth, **kwargs):
            return len(predictions)
        
        config = MetricConfig(
            metric_type=MetricType.CONFIDENCE,
            function=custom_metric,
            description="Custom confidence metric"
        )
        
        empty_registry.register_metric(config)
        
        assert MetricType.CONFIDENCE.value in empty_registry.metrics
        assert empty_registry.metrics[MetricType.CONFIDENCE.value].function == custom_metric
    
    def test_calculate_metric_success(self, populated_registry, sample_predictions, sample_ground_truth):
        """Test calculating a single metric."""
        result = populated_registry.calculate_metric(
            MetricType.ACCURACY.value,
            sample_predictions,
            sample_ground_truth
        )
        
        assert isinstance(result, MetricResult)
        assert result.metric_name == MetricType.ACCURACY.value
        assert 0 <= result.value <= 1
        assert result.value == 0.75  # 3 correct out of 4
    
    def test_calculate_metric_not_found(self, populated_registry):
        """Test calculating non-existent metric."""
        with pytest.raises(ValueError, match="Metric 'nonexistent' not registered"):
            populated_registry.calculate_metric("nonexistent", [], [])
    
    def test_calculate_metric_missing_ground_truth(self, populated_registry, sample_predictions):
        """Test calculating metric that requires ground truth without providing it."""
        with pytest.raises(ValueError, match="requires ground truth data"):
            populated_registry.calculate_metric(
                MetricType.ACCURACY.value,
                sample_predictions
            )
    
    def test_calculate_all_metrics(self, populated_registry, sample_predictions, sample_ground_truth):
        """Test calculating all applicable metrics."""
        results = populated_registry.calculate_all_metrics(
            sample_predictions,
            sample_ground_truth,
            iou_threshold=0.5
        )
        
        assert isinstance(results, dict)
        assert MetricType.ACCURACY.value in results
        assert MetricType.PRECISION.value in results
        assert MetricType.RECALL.value in results
        assert MetricType.F1_SCORE.value in results
        assert MetricType.LATENCY.value in results
        
        # All results should be MetricResult objects
        assert all(isinstance(r, MetricResult) for r in results.values())
    
    def test_calculate_all_metrics_no_ground_truth(self, populated_registry, sample_predictions):
        """Test calculating metrics without ground truth (only performance metrics)."""
        results = populated_registry.calculate_all_metrics(sample_predictions)
        
        # Should only have metrics that don't require ground truth
        assert MetricType.LATENCY.value in results
        assert MetricType.THROUGHPUT.value in results
        assert MetricType.PARSE_SUCCESS_RATE.value in results
        
        # Should not have metrics that require ground truth
        assert MetricType.ACCURACY.value not in results
        assert MetricType.PRECISION.value not in results
    
    # Test individual metric calculations
    def test_calculate_accuracy(self, populated_registry, sample_predictions, sample_ground_truth):
        """Test accuracy calculation."""
        accuracy = populated_registry._calculate_accuracy(sample_predictions, sample_ground_truth)
        assert accuracy == 0.75  # 3 correct out of 4
    
    def test_calculate_precision(self, populated_registry, sample_predictions, sample_ground_truth):
        """Test precision calculation."""
        precision = populated_registry._calculate_precision(sample_predictions, sample_ground_truth)
        # TP=1 (first), FP=1 (third), precision = 1/(1+1) = 0.5
        assert precision == 0.5
    
    def test_calculate_recall(self, populated_registry, sample_predictions, sample_ground_truth):
        """Test recall calculation."""
        recall = populated_registry._calculate_recall(sample_predictions, sample_ground_truth)
        # TP=1 (first), FN=0, recall = 1/(1+0) = 1.0
        assert recall == 1.0
    
    def test_calculate_f1_score(self, populated_registry, sample_predictions, sample_ground_truth):
        """Test F1 score calculation."""
        f1 = populated_registry._calculate_f1_score(sample_predictions, sample_ground_truth)
        # precision=0.5, recall=1.0, f1 = 2*(0.5*1.0)/(0.5+1.0) = 0.667
        assert 0.66 <= f1 <= 0.67
    
    def test_calculate_latency(self, populated_registry, sample_predictions):
        """Test latency calculation."""
        latency = populated_registry._calculate_latency(sample_predictions)
        expected = np.mean([100, 95, 105, 98])
        assert latency == expected
    
    def test_calculate_throughput(self, populated_registry, sample_predictions):
        """Test throughput calculation."""
        throughput = populated_registry._calculate_throughput(sample_predictions)
        total_time_s = sum([100, 95, 105, 98]) / 1000
        expected = 4 / total_time_s
        assert abs(throughput - expected) < 0.01
    
    def test_calculate_parse_success_rate(self, populated_registry):
        """Test parse success rate calculation."""
        predictions_with_errors = [
            {"has_smoke": True},
            {"has_smoke": False, "error": "Parse failed"},
            {"has_smoke": True},
            {"has_smoke": False}
        ]
        
        success_rate = populated_registry._calculate_parse_success_rate(predictions_with_errors)
        assert success_rate == 0.75  # 3 out of 4 successful
    
    def test_calculate_error_rate(self, populated_registry):
        """Test error rate calculation."""
        predictions_with_errors = [
            {"has_smoke": True},
            {"has_smoke": False, "error": "Model failed"},
            {"has_smoke": True, "error": "Timeout"},
            {"has_smoke": False}
        ]
        
        error_rate = populated_registry._calculate_error_rate(predictions_with_errors)
        assert error_rate == 0.5  # 2 out of 4 have errors
    
    def test_compute_iou(self, populated_registry):
        """Test IOU computation for bounding boxes."""
        # Test perfect overlap
        bbox1 = [0.5, 0.5, 0.2, 0.2]
        iou = populated_registry._compute_iou(bbox1, bbox1)
        assert iou == 1.0
        
        # Test no overlap
        bbox2 = [0.9, 0.9, 0.1, 0.1]
        iou = populated_registry._compute_iou(bbox1, bbox2)
        assert iou == 0.0
        
        # Test partial overlap
        bbox3 = [0.55, 0.55, 0.2, 0.2]
        iou = populated_registry._compute_iou(bbox1, bbox3)
        assert 0 < iou < 1
    
    def test_calculate_bbox_iou(self, populated_registry, sample_predictions, sample_ground_truth):
        """Test average bbox IOU calculation."""
        avg_iou = populated_registry._calculate_bbox_iou(sample_predictions, sample_ground_truth)
        assert 0 <= avg_iou <= 1
    
    def test_calculate_bbox_precision(self, populated_registry, sample_predictions, sample_ground_truth):
        """Test bbox precision calculation."""
        bbox_precision = populated_registry._calculate_bbox_precision(
            sample_predictions, 
            sample_ground_truth,
            iou_threshold=0.5
        )
        assert 0 <= bbox_precision <= 1
    
    def test_aggregate_results_mean(self, populated_registry):
        """Test aggregating results with mean method."""
        results_list = [
            {
                "accuracy": MetricResult("accuracy", 0.8),
                "precision": MetricResult("precision", 0.7)
            },
            {
                "accuracy": MetricResult("accuracy", 0.9),
                "precision": MetricResult("precision", 0.8)
            },
            {
                "accuracy": MetricResult("accuracy", 0.85),
                "precision": MetricResult("precision", 0.75)
            }
        ]
        
        aggregated = populated_registry.aggregate_results(results_list, "mean")
        
        assert aggregated["accuracy"].value == 0.85
        assert aggregated["precision"].value == 0.75
        assert aggregated["accuracy"].details["num_samples"] == 3
        assert "std" in aggregated["accuracy"].details
    
    def test_aggregate_results_other_methods(self, populated_registry):
        """Test aggregating results with different methods."""
        results_list = [
            {"metric": MetricResult("metric", 1.0)},
            {"metric": MetricResult("metric", 2.0)},
            {"metric": MetricResult("metric", 3.0)}
        ]
        
        # Test median
        agg_median = populated_registry.aggregate_results(results_list, "median")
        assert agg_median["metric"].value == 2.0
        
        # Test max
        agg_max = populated_registry.aggregate_results(results_list, "max")
        assert agg_max["metric"].value == 3.0
        
        # Test min
        agg_min = populated_registry.aggregate_results(results_list, "min")
        assert agg_min["metric"].value == 1.0
        
        # Test sum
        agg_sum = populated_registry.aggregate_results(results_list, "sum")
        assert agg_sum["metric"].value == 6.0
    
    def test_aggregate_results_empty(self, populated_registry):
        """Test aggregating empty results."""
        aggregated = populated_registry.aggregate_results([])
        assert aggregated == {}
    
    def test_compare_results(self, populated_registry):
        """Test comparing results between two models."""
        results1 = {
            "accuracy": MetricResult("accuracy", 0.85),
            "precision": MetricResult("precision", 0.8),
            "latency": MetricResult("latency", 100),  # Lower is better
            "f1_score": MetricResult("f1_score", 0.82)
        }
        
        results2 = {
            "accuracy": MetricResult("accuracy", 0.90),
            "precision": MetricResult("precision", 0.75),
            "latency": MetricResult("latency", 120),
            "f1_score": MetricResult("f1_score", 0.82)  # Tie
        }
        
        comparison = populated_registry.compare_results(
            results1, results2,
            "Model A", "Model B"
        )
        
        assert comparison["model1"] == "Model A"
        assert comparison["model2"] == "Model B"
        
        # Check individual comparisons
        assert comparison["comparisons"]["accuracy"]["better"] == "Model B"
        assert comparison["comparisons"]["precision"]["better"] == "Model A"
        assert comparison["comparisons"]["latency"]["better"] == "Model A"
        assert comparison["comparisons"]["f1_score"]["better"] == "tie"
        
        # Check summary
        assert comparison["summary"]["metrics_compared"] == 4
        assert comparison["summary"]["Model A_wins"] == 2
        assert comparison["summary"]["Model B_wins"] == 2
    
    def test_metric_calculation_error_handling(self, populated_registry):
        """Test error handling in metric calculation."""
        # Create a metric that will raise an exception
        def error_metric(predictions, ground_truth, **kwargs):
            raise ValueError("Test error")
        
        populated_registry.register_metric(MetricConfig(
            metric_type=MetricType.CONFIDENCE,
            function=error_metric,
            description="Error metric"
        ))
        
        result = populated_registry.calculate_metric(
            MetricType.CONFIDENCE.value,
            [], []
        )
        
        assert result.value == 0.0
        assert "error" in result.details
        assert "Test error" in result.details["error"]
    
    def test_mismatched_lengths(self, populated_registry):
        """Test handling mismatched prediction and ground truth lengths."""
        predictions = [{"has_smoke": True}, {"has_smoke": False}]
        ground_truth = [{"has_smoke": True}]  # Different length
        
        with pytest.raises(ValueError, match="same length"):
            populated_registry._calculate_accuracy(predictions, ground_truth)
    
    def test_empty_predictions(self, populated_registry):
        """Test handling empty predictions."""
        accuracy = populated_registry._calculate_accuracy([], [])
        assert accuracy == 0.0
        
        latency = populated_registry._calculate_latency([])
        assert latency == 0.0
    
    def test_missing_bbox_data(self, populated_registry):
        """Test handling missing bbox data."""
        predictions = [{"has_smoke": True}]  # No bbox
        ground_truth = [{"has_smoke": True, "bbox": [0.5, 0.5, 0.2, 0.2]}]
        
        results = populated_registry.calculate_all_metrics(predictions, ground_truth)
        
        # Should calculate non-bbox metrics
        assert MetricType.ACCURACY.value in results
        
        # Should skip bbox metrics
        assert MetricType.BBOX_IOU.value not in results
        assert MetricType.BBOX_PRECISION.value not in results


# Property-based tests using Hypothesis
class TestMetricsRegistryProperties:
    """Property-based tests for robust edge case coverage."""
    
    @given(st.lists(st.booleans(), min_size=1, max_size=100))
    def test_accuracy_calculation_properties(self, predictions_bool, populated_registry):
        """Test accuracy calculation with property-based testing."""
        # Generate predictions and ground truth from boolean list
        predictions = [{"has_smoke": pred} for pred in predictions_bool]
        ground_truth = [{"has_smoke": True} for _ in predictions_bool]
        
        accuracy = populated_registry._calculate_accuracy(predictions, ground_truth)
        
        # Property: accuracy must be between 0 and 1
        assert 0.0 <= accuracy <= 1.0
        
        # Property: accuracy equals the proportion of True predictions
        expected_accuracy = sum(predictions_bool) / len(predictions_bool)
        assert abs(accuracy - expected_accuracy) < 1e-10
    
    @given(st.lists(st.floats(min_value=0.0, max_value=1000.0), min_size=1, max_size=50))
    def test_latency_calculation_properties(self, latencies, populated_registry):
        """Test latency calculation properties."""
        predictions = [{"latency_ms": latency} for latency in latencies]
        
        calculated_latency = populated_registry._calculate_latency(predictions)
        
        # Property: calculated latency should equal the mean
        expected_latency = np.mean(latencies)
        assert abs(calculated_latency - expected_latency) < 1e-10
        
        # Property: latency should be non-negative
        assert calculated_latency >= 0.0
        
        # Property: latency should be within min/max bounds of input
        assert min(latencies) <= calculated_latency <= max(latencies)
    
    @given(st.lists(
        st.tuples(st.booleans(), st.booleans()),
        min_size=1,
        max_size=100
    ))
    def test_precision_recall_properties(self, pred_gt_pairs, populated_registry):
        """Test precision and recall calculation properties."""
        predictions = [{"has_smoke": pred} for pred, _ in pred_gt_pairs]
        ground_truth = [{"has_smoke": gt} for _, gt in pred_gt_pairs]
        
        precision = populated_registry._calculate_precision(predictions, ground_truth)
        recall = populated_registry._calculate_recall(predictions, ground_truth)
        
        # Property: precision and recall must be between 0 and 1
        assert 0.0 <= precision <= 1.0
        assert 0.0 <= recall <= 1.0
        
        # Calculate expected values manually
        tp = sum(1 for (pred, gt) in pred_gt_pairs if pred and gt)
        fp = sum(1 for (pred, gt) in pred_gt_pairs if pred and not gt)
        fn = sum(1 for (pred, gt) in pred_gt_pairs if not pred and gt)
        
        if tp + fp == 0:
            expected_precision = 0.0
        else:
            expected_precision = tp / (tp + fp)
        
        if tp + fn == 0:
            expected_recall = 0.0
        else:
            expected_recall = tp / (tp + fn)
        
        assert abs(precision - expected_precision) < 1e-10
        assert abs(recall - expected_recall) < 1e-10
    
    @given(st.lists(
        st.tuples(
            st.floats(min_value=0.0, max_value=1.0),  # x_center
            st.floats(min_value=0.0, max_value=1.0),  # y_center
            st.floats(min_value=0.01, max_value=1.0), # width
            st.floats(min_value=0.01, max_value=1.0)  # height
        ),
        min_size=1,
        max_size=20
    ))
    def test_iou_calculation_properties(self, bbox_coords, populated_registry):
        """Test IOU calculation properties with random bounding boxes."""
        # Filter out invalid bounding boxes
        valid_bboxes = []
        for x, y, w, h in bbox_coords:
            # Ensure bbox is within image bounds
            if x + w/2 <= 1.0 and x - w/2 >= 0.0 and y + h/2 <= 1.0 and y - h/2 >= 0.0:
                valid_bboxes.append([x, y, w, h])
        
        assume(len(valid_bboxes) >= 2)  # Need at least 2 boxes for comparison
        
        # Test IOU properties
        bbox1 = valid_bboxes[0]
        bbox2 = valid_bboxes[1]
        
        iou = populated_registry._compute_iou(bbox1, bbox2)
        
        # Property: IOU must be between 0 and 1
        assert 0.0 <= iou <= 1.0
        
        # Property: IOU is symmetric
        iou_reverse = populated_registry._compute_iou(bbox2, bbox1)
        assert abs(iou - iou_reverse) < 1e-10
        
        # Property: IOU with itself should be 1.0
        self_iou = populated_registry._compute_iou(bbox1, bbox1)
        assert abs(self_iou - 1.0) < 1e-10
    
    @given(st.lists(
        st.dictionaries(
            st.sampled_from(["accuracy", "precision", "recall", "f1_score", "latency"]),
            st.floats(min_value=0.0, max_value=1.0),
            min_size=1
        ),
        min_size=2,
        max_size=10
    ))
    def test_aggregation_properties(self, results_dicts, populated_registry):
        """Test metric aggregation properties."""
        # Convert to MetricResult format
        results_list = []
        for result_dict in results_dicts:
            result = {}
            for metric_name, value in result_dict.items():
                result[metric_name] = MetricResult(metric_name, value)
            results_list.append(result)
        
        # Test different aggregation methods
        for method in ["mean", "median", "min", "max"]:
            aggregated = populated_registry.aggregate_results(results_list, method)
            
            # Property: aggregated results should have same metrics as input
            input_metrics = set(results_list[0].keys())
            output_metrics = set(aggregated.keys())
            assert input_metrics == output_metrics
            
            # Property: aggregated values should be within reasonable bounds
            for metric_name in input_metrics:
                values = [result[metric_name].value for result in results_list]
                aggregated_value = aggregated[metric_name].value
                
                if method == "mean":
                    expected = np.mean(values)
                elif method == "median":
                    expected = np.median(values)
                elif method == "min":
                    expected = min(values)
                elif method == "max":
                    expected = max(values)
                
                assert abs(aggregated_value - expected) < 1e-10
                assert min(values) <= aggregated_value <= max(values)
    
    @given(st.lists(st.floats(min_value=0.0, max_value=10.0), min_size=1, max_size=100))
    def test_throughput_properties(self, latencies_ms, populated_registry):
        """Test throughput calculation properties."""
        predictions = [{"latency_ms": latency} for latency in latencies_ms]
        
        throughput = populated_registry._calculate_throughput(predictions)
        
        # Property: throughput should be positive
        assert throughput > 0.0
        
        # Property: throughput should be inversely related to average latency
        avg_latency_s = np.mean(latencies_ms) / 1000.0
        expected_throughput = len(latencies_ms) / (sum(latencies_ms) / 1000.0)
        
        assert abs(throughput - expected_throughput) < 1e-6
    
    @given(st.lists(
        st.dictionaries(
            st.just("error"),
            st.one_of(st.none(), st.text(min_size=1)),
            min_size=0,
            max_size=1
        ),
        min_size=1,
        max_size=50
    ))
    def test_error_rate_properties(self, predictions_with_errors, populated_registry):
        """Test error rate calculation properties."""
        error_rate = populated_registry._calculate_error_rate(predictions_with_errors)
        
        # Property: error rate must be between 0 and 1
        assert 0.0 <= error_rate <= 1.0
        
        # Property: error rate should match manual calculation
        errors = sum(1 for pred in predictions_with_errors if pred.get("error") is not None)
        expected_rate = errors / len(predictions_with_errors)
        
        assert abs(error_rate - expected_rate) < 1e-10
    
    @given(st.lists(st.floats(min_value=0.0, max_value=1.0), min_size=1, max_size=100))
    def test_confidence_calculation_properties(self, confidence_values, populated_registry):
        """Test confidence metric calculation properties."""
        predictions = [{"confidence": conf} for conf in confidence_values]
        
        # Calculate average confidence
        avg_confidence = populated_registry._calculate_confidence(predictions)
        
        # Property: average confidence should be between 0 and 1
        assert 0.0 <= avg_confidence <= 1.0
        
        # Property: should equal numpy mean
        expected = np.mean(confidence_values)
        assert abs(avg_confidence - expected) < 1e-10
        
        # Property: should be within min/max bounds
        assert min(confidence_values) <= avg_confidence <= max(confidence_values)