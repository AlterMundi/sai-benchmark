"""
Metrics Registry System

Manages evaluation metrics, aggregation functions, and performance analysis.
Supports customizable metrics for different evaluation scenarios.
"""

from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import numpy as np
from pathlib import Path
import time
from datetime import datetime


class MetricType(Enum):
    """Types of metrics supported"""
    ACCURACY = "accuracy"
    PRECISION = "precision"  
    RECALL = "recall"
    F1_SCORE = "f1_score"
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    COST = "cost"
    BBOX_IOU = "bbox_iou"
    BBOX_PRECISION = "bbox_precision"
    BBOX_RECALL = "bbox_recall"
    CONFIDENCE = "confidence"
    ERROR_RATE = "error_rate"
    PARSE_SUCCESS_RATE = "parse_success_rate"
    # Vision-specific metrics
    OBJECT_DETECTION_ACCURACY = "object_detection_accuracy"
    COUNTING_ACCURACY = "counting_accuracy"
    COLOR_ACCURACY = "color_accuracy"
    SPATIAL_ACCURACY = "spatial_accuracy"
    TEXT_RECOGNITION_ACCURACY = "text_recognition_accuracy"
    MATERIAL_RECOGNITION_ACCURACY = "material_recognition_accuracy"
    COMPREHENSIVE_SCORE = "comprehensive_score"
    RESOLUTION_CONSISTENCY = "resolution_consistency"


@dataclass
class MetricResult:
    """Result of a metric calculation"""
    metric_name: str
    value: float
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric_name": self.metric_name,
            "value": self.value,
            "details": self.details,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass 
class MetricConfig:
    """Configuration for a metric calculation"""
    metric_type: MetricType
    function: Callable
    description: str
    higher_is_better: bool = True
    requires_ground_truth: bool = True
    requires_bbox: bool = False
    aggregation_method: str = "mean"  # mean, median, sum, max, min


class MetricsRegistry:
    """Registry for managing evaluation metrics"""
    
    def __init__(self):
        self.metrics: Dict[str, MetricConfig] = {}
        self._register_builtin_metrics()
    
    def _register_builtin_metrics(self):
        """Register built-in metric functions"""
        
        # Classification metrics
        self.register_metric(MetricConfig(
            metric_type=MetricType.ACCURACY,
            function=self._calculate_accuracy,
            description="Binary classification accuracy for smoke detection",
            higher_is_better=True,
            requires_ground_truth=True
        ))
        
        self.register_metric(MetricConfig(
            metric_type=MetricType.PRECISION,
            function=self._calculate_precision,
            description="Precision (positive predictive value) for smoke detection",
            higher_is_better=True,
            requires_ground_truth=True
        ))
        
        self.register_metric(MetricConfig(
            metric_type=MetricType.RECALL,
            function=self._calculate_recall,
            description="Recall (sensitivity) for smoke detection",
            higher_is_better=True,
            requires_ground_truth=True
        ))
        
        self.register_metric(MetricConfig(
            metric_type=MetricType.F1_SCORE,
            function=self._calculate_f1_score,
            description="F1 score (harmonic mean of precision and recall)",
            higher_is_better=True,
            requires_ground_truth=True
        ))
        
        # Performance metrics
        self.register_metric(MetricConfig(
            metric_type=MetricType.LATENCY,
            function=self._calculate_latency,
            description="Average response time in milliseconds",
            higher_is_better=False,
            requires_ground_truth=False
        ))
        
        self.register_metric(MetricConfig(
            metric_type=MetricType.THROUGHPUT,
            function=self._calculate_throughput,
            description="Requests per second",
            higher_is_better=True,
            requires_ground_truth=False
        ))
        
        self.register_metric(MetricConfig(
            metric_type=MetricType.COST,
            function=self._calculate_cost,
            description="Cost per inference in USD",
            higher_is_better=False,
            requires_ground_truth=False
        ))
        
        # Bounding box metrics
        self.register_metric(MetricConfig(
            metric_type=MetricType.BBOX_IOU,
            function=self._calculate_bbox_iou,
            description="Intersection over Union for bounding boxes",
            higher_is_better=True,
            requires_ground_truth=True,
            requires_bbox=True
        ))
        
        self.register_metric(MetricConfig(
            metric_type=MetricType.BBOX_PRECISION,
            function=self._calculate_bbox_precision,
            description="Bounding box precision at IoU threshold",
            higher_is_better=True,
            requires_ground_truth=True,
            requires_bbox=True
        ))
        
        # Quality metrics
        self.register_metric(MetricConfig(
            metric_type=MetricType.PARSE_SUCCESS_RATE,
            function=self._calculate_parse_success_rate,
            description="Rate of successful response parsing",
            higher_is_better=True,
            requires_ground_truth=False
        ))
        
        self.register_metric(MetricConfig(
            metric_type=MetricType.ERROR_RATE,
            function=self._calculate_error_rate,
            description="Rate of errors during inference",
            higher_is_better=False,
            requires_ground_truth=False
        ))
        
        # Vision-specific metrics
        self.register_metric(MetricConfig(
            metric_type=MetricType.OBJECT_DETECTION_ACCURACY,
            function=self._calculate_object_detection_accuracy,
            description="Accuracy of object detection and identification",
            higher_is_better=True,
            requires_ground_truth=True
        ))
        
        self.register_metric(MetricConfig(
            metric_type=MetricType.COUNTING_ACCURACY,
            function=self._calculate_counting_accuracy,
            description="Accuracy of object counting",
            higher_is_better=True,
            requires_ground_truth=True
        ))
        
        self.register_metric(MetricConfig(
            metric_type=MetricType.COLOR_ACCURACY,
            function=self._calculate_color_accuracy,
            description="Accuracy of color recognition",
            higher_is_better=True,
            requires_ground_truth=True
        ))
        
        self.register_metric(MetricConfig(
            metric_type=MetricType.SPATIAL_ACCURACY,
            function=self._calculate_spatial_accuracy,
            description="Accuracy of spatial relationship detection",
            higher_is_better=True,
            requires_ground_truth=True
        ))
        
        self.register_metric(MetricConfig(
            metric_type=MetricType.TEXT_RECOGNITION_ACCURACY,
            function=self._calculate_text_recognition_accuracy,
            description="Accuracy of text recognition",
            higher_is_better=True,
            requires_ground_truth=True
        ))
        
        self.register_metric(MetricConfig(
            metric_type=MetricType.MATERIAL_RECOGNITION_ACCURACY,
            function=self._calculate_material_recognition_accuracy,
            description="Accuracy of material identification",
            higher_is_better=True,
            requires_ground_truth=True
        ))
        
        self.register_metric(MetricConfig(
            metric_type=MetricType.COMPREHENSIVE_SCORE,
            function=self._calculate_comprehensive_score,
            description="Combined score across all vision capabilities",
            higher_is_better=True,
            requires_ground_truth=True
        ))
        
        self.register_metric(MetricConfig(
            metric_type=MetricType.RESOLUTION_CONSISTENCY,
            function=self._calculate_resolution_consistency,
            description="Consistency of results across different image resolutions",
            higher_is_better=True,
            requires_ground_truth=True
        ))
    
    def register_metric(self, config: MetricConfig):
        """Register a new metric configuration"""
        self.metrics[config.metric_type.value] = config
    
    def calculate_metric(self, 
                        metric_name: str,
                        predictions: List[Dict[str, Any]],
                        ground_truth: List[Dict[str, Any]] = None,
                        **kwargs) -> MetricResult:
        """Calculate a specific metric"""
        
        if metric_name not in self.metrics:
            raise ValueError(f"Metric '{metric_name}' not registered")
        
        config = self.metrics[metric_name]
        
        if config.requires_ground_truth and ground_truth is None:
            raise ValueError(f"Metric '{metric_name}' requires ground truth data")
        
        try:
            value = config.function(predictions, ground_truth, **kwargs)
            
            return MetricResult(
                metric_name=metric_name,
                value=float(value),
                details=kwargs
            )
        except Exception as e:
            return MetricResult(
                metric_name=metric_name,
                value=0.0,
                details={"error": str(e)}
            )
    
    def calculate_all_metrics(self,
                            predictions: List[Dict[str, Any]],
                            ground_truth: List[Dict[str, Any]] = None,
                            **kwargs) -> Dict[str, MetricResult]:
        """Calculate all applicable metrics"""
        
        results = {}
        
        for metric_name, config in self.metrics.items():
            # Skip metrics that require ground truth if not provided
            if config.requires_ground_truth and ground_truth is None:
                continue
            
            # Skip bbox metrics if bbox data not available
            if config.requires_bbox:
                has_bbox_pred = any('bbox' in pred for pred in predictions)
                has_bbox_gt = ground_truth and any('bbox' in gt for gt in ground_truth)
                if not (has_bbox_pred and has_bbox_gt):
                    continue
            
            result = self.calculate_metric(metric_name, predictions, ground_truth, **kwargs)
            results[metric_name] = result
        
        return results
    
    # Metric calculation functions
    def _calculate_accuracy(self, predictions: List[Dict], ground_truth: List[Dict], **kwargs) -> float:
        """Calculate binary classification accuracy"""
        if len(predictions) != len(ground_truth):
            raise ValueError("Predictions and ground truth must have same length")
        
        correct = 0
        total = len(predictions)
        
        for pred, gt in zip(predictions, ground_truth):
            pred_has_smoke = pred.get('has_smoke', False)
            gt_has_smoke = gt.get('has_smoke', False)
            
            if pred_has_smoke == gt_has_smoke:
                correct += 1
        
        return correct / total if total > 0 else 0.0
    
    def _calculate_precision(self, predictions: List[Dict], ground_truth: List[Dict], **kwargs) -> float:
        """Calculate precision (true positives / (true positives + false positives))"""
        tp = fp = 0
        
        for pred, gt in zip(predictions, ground_truth):
            pred_has_smoke = pred.get('has_smoke', False)
            gt_has_smoke = gt.get('has_smoke', False)
            
            if pred_has_smoke and gt_has_smoke:
                tp += 1
            elif pred_has_smoke and not gt_has_smoke:
                fp += 1
        
        return tp / (tp + fp) if (tp + fp) > 0 else 0.0
    
    def _calculate_recall(self, predictions: List[Dict], ground_truth: List[Dict], **kwargs) -> float:
        """Calculate recall (true positives / (true positives + false negatives))"""
        tp = fn = 0
        
        for pred, gt in zip(predictions, ground_truth):
            pred_has_smoke = pred.get('has_smoke', False)
            gt_has_smoke = gt.get('has_smoke', False)
            
            if pred_has_smoke and gt_has_smoke:
                tp += 1
            elif not pred_has_smoke and gt_has_smoke:
                fn += 1
        
        return tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    def _calculate_f1_score(self, predictions: List[Dict], ground_truth: List[Dict], **kwargs) -> float:
        """Calculate F1 score (harmonic mean of precision and recall)"""
        precision = self._calculate_precision(predictions, ground_truth)
        recall = self._calculate_recall(predictions, ground_truth)
        
        return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    def _calculate_latency(self, predictions: List[Dict], ground_truth: List[Dict] = None, **kwargs) -> float:
        """Calculate average latency"""
        latencies = [pred.get('latency_ms', 0) for pred in predictions]
        return np.mean(latencies) if latencies else 0.0
    
    def _calculate_throughput(self, predictions: List[Dict], ground_truth: List[Dict] = None, **kwargs) -> float:
        """Calculate throughput (requests per second)"""
        total_time_s = sum(pred.get('latency_ms', 0) for pred in predictions) / 1000
        return len(predictions) / total_time_s if total_time_s > 0 else 0.0
    
    def _calculate_cost(self, predictions: List[Dict], ground_truth: List[Dict] = None, **kwargs) -> float:
        """Calculate average cost per inference"""
        costs = [pred.get('cost', 0) for pred in predictions]
        return np.mean(costs) if costs else 0.0
    
    def _calculate_bbox_iou(self, predictions: List[Dict], ground_truth: List[Dict], **kwargs) -> float:
        """Calculate average IoU for bounding boxes"""
        ious = []
        
        for pred, gt in zip(predictions, ground_truth):
            pred_bbox = pred.get('bbox', [0, 0, 0, 0])
            gt_bbox = gt.get('bbox', [0, 0, 0, 0])
            
            iou = self._compute_iou(pred_bbox, gt_bbox)
            ious.append(iou)
        
        return np.mean(ious) if ious else 0.0
    
    def _calculate_bbox_precision(self, predictions: List[Dict], ground_truth: List[Dict], **kwargs) -> float:
        """Calculate bounding box precision at IoU threshold"""
        iou_threshold = kwargs.get('iou_threshold', 0.5)
        correct = 0
        total_predictions = 0
        
        for pred, gt in zip(predictions, ground_truth):
            pred_has_smoke = pred.get('has_smoke', False)
            gt_has_smoke = gt.get('has_smoke', False)
            
            if pred_has_smoke:
                total_predictions += 1
                
                if gt_has_smoke:
                    pred_bbox = pred.get('bbox', [0, 0, 0, 0])
                    gt_bbox = gt.get('bbox', [0, 0, 0, 0])
                    
                    iou = self._compute_iou(pred_bbox, gt_bbox)
                    if iou >= iou_threshold:
                        correct += 1
        
        return correct / total_predictions if total_predictions > 0 else 0.0
    
    def _calculate_parse_success_rate(self, predictions: List[Dict], ground_truth: List[Dict] = None, **kwargs) -> float:
        """Calculate rate of successful response parsing"""
        successful = sum(1 for pred in predictions if not pred.get('error'))
        return successful / len(predictions) if predictions else 0.0
    
    def _calculate_error_rate(self, predictions: List[Dict], ground_truth: List[Dict] = None, **kwargs) -> float:
        """Calculate error rate during inference"""
        errors = sum(1 for pred in predictions if pred.get('error'))
        return errors / len(predictions) if predictions else 0.0
    
    def _compute_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        """Compute Intersection over Union for two bounding boxes"""
        # Convert from center format to corner format if needed
        if len(bbox1) == 4 and len(bbox2) == 4:
            # Assume format: [x_center, y_center, width, height]
            x1_min = bbox1[0] - bbox1[2] / 2
            y1_min = bbox1[1] - bbox1[3] / 2
            x1_max = bbox1[0] + bbox1[2] / 2
            y1_max = bbox1[1] + bbox1[3] / 2
            
            x2_min = bbox2[0] - bbox2[2] / 2
            y2_min = bbox2[1] - bbox2[3] / 2
            x2_max = bbox2[0] + bbox2[2] / 2
            y2_max = bbox2[1] + bbox2[3] / 2
            
            # Calculate intersection
            inter_x_min = max(x1_min, x2_min)
            inter_y_min = max(y1_min, y2_min)
            inter_x_max = min(x1_max, x2_max)
            inter_y_max = min(y1_max, y2_max)
            
            if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
                return 0.0
            
            intersection = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
            
            # Calculate union
            area1 = bbox1[2] * bbox1[3]
            area2 = bbox2[2] * bbox2[3]
            union = area1 + area2 - intersection
            
            return intersection / union if union > 0 else 0.0
        
        return 0.0
    
    def aggregate_results(self, 
                         results_list: List[Dict[str, MetricResult]],
                         aggregation_method: str = "mean") -> Dict[str, MetricResult]:
        """Aggregate results from multiple test runs"""
        
        if not results_list:
            return {}
        
        # Get all metric names
        all_metrics = set()
        for results in results_list:
            all_metrics.update(results.keys())
        
        aggregated = {}
        
        for metric_name in all_metrics:
            values = []
            for results in results_list:
                if metric_name in results:
                    values.append(results[metric_name].value)
            
            if values:
                if aggregation_method == "mean":
                    agg_value = np.mean(values)
                elif aggregation_method == "median":
                    agg_value = np.median(values)
                elif aggregation_method == "max":
                    agg_value = np.max(values)
                elif aggregation_method == "min":
                    agg_value = np.min(values)
                elif aggregation_method == "sum":
                    agg_value = np.sum(values)
                else:
                    agg_value = np.mean(values)  # Default to mean
                
                aggregated[metric_name] = MetricResult(
                    metric_name=metric_name,
                    value=agg_value,
                    details={
                        "aggregation_method": aggregation_method,
                        "num_samples": len(values),
                        "std": np.std(values),
                        "min": np.min(values),
                        "max": np.max(values)
                    }
                )
        
        return aggregated
    
    def compare_results(self, 
                       results1: Dict[str, MetricResult],
                       results2: Dict[str, MetricResult],
                       name1: str = "Model 1",
                       name2: str = "Model 2") -> Dict[str, Any]:
        """Compare results between two models"""
        
        comparison = {
            "model1": name1,
            "model2": name2,
            "comparisons": {},
            "summary": {}
        }
        
        common_metrics = set(results1.keys()) & set(results2.keys())
        
        better_count = 0
        total_comparisons = 0
        
        for metric_name in common_metrics:
            if metric_name not in self.metrics:
                continue
                
            config = self.metrics[metric_name]
            
            value1 = results1[metric_name].value
            value2 = results2[metric_name].value
            
            # Determine which is better
            if config.higher_is_better:
                better = name1 if value1 > value2 else name2 if value2 > value1 else "tie"
                improvement = ((value1 - value2) / value2 * 100) if value2 != 0 else 0
            else:
                better = name1 if value1 < value2 else name2 if value2 < value1 else "tie"
                improvement = ((value2 - value1) / value1 * 100) if value1 != 0 else 0
            
            comparison["comparisons"][metric_name] = {
                "value1": value1,
                "value2": value2,
                "better": better,
                "improvement_percent": improvement,
                "higher_is_better": config.higher_is_better
            }
            
            if better == name1:
                better_count += 1
            total_comparisons += 1
        
        comparison["summary"] = {
            "metrics_compared": len(common_metrics),
            f"{name1}_wins": better_count,
            f"{name2}_wins": total_comparisons - better_count,
            "win_rate": better_count / total_comparisons if total_comparisons > 0 else 0.0
        }
        
        return comparison
    
    # Vision-specific metric calculation functions
    def _calculate_object_detection_accuracy(self, predictions: List[Dict], ground_truth: List[Dict], **kwargs) -> float:
        """Calculate object detection accuracy based on detected objects"""
        if len(predictions) != len(ground_truth):
            return 0.0
        
        total_score = 0.0
        for pred, gt in zip(predictions, ground_truth):
            pred_objects = set()
            gt_objects = set()
            
            # Extract object names from predictions
            if 'objects' in pred:
                pred_objects = {obj.get('name', '').lower() for obj in pred['objects']}
            elif 'comprehensive_analysis' in pred and 'objects' in pred['comprehensive_analysis']:
                pred_objects = {obj.get('name', '').lower() for obj in pred['comprehensive_analysis']['objects']}
            
            # Extract object names from ground truth
            if 'objects' in gt:
                gt_objects = {obj.get('name', '').lower() for obj in gt['objects']}
            
            # Calculate Jaccard similarity (intersection over union)
            if gt_objects:
                intersection = len(pred_objects & gt_objects)
                union = len(pred_objects | gt_objects)
                score = intersection / union if union > 0 else 0.0
                total_score += score
        
        return total_score / len(predictions) if predictions else 0.0
    
    def _calculate_counting_accuracy(self, predictions: List[Dict], ground_truth: List[Dict], **kwargs) -> float:
        """Calculate accuracy of object counting"""
        if len(predictions) != len(ground_truth):
            return 0.0
        
        total_score = 0.0
        for pred, gt in zip(predictions, ground_truth):
            pred_total = pred.get('total_objects', 0)
            gt_total = gt.get('total_objects', 0)
            
            # Also try comprehensive analysis format
            if pred_total == 0 and 'comprehensive_analysis' in pred:
                pred_total = pred['comprehensive_analysis'].get('total_objects', 0)
            
            if gt_total > 0:
                # Calculate relative error (1 - |error|/gt_total), capped at 0
                relative_error = abs(pred_total - gt_total) / gt_total
                score = max(0.0, 1.0 - relative_error)
                total_score += score
        
        return total_score / len(predictions) if predictions else 0.0
    
    def _calculate_color_accuracy(self, predictions: List[Dict], ground_truth: List[Dict], **kwargs) -> float:
        """Calculate color recognition accuracy"""
        if len(predictions) != len(ground_truth):
            return 0.0
        
        total_score = 0.0
        for pred, gt in zip(predictions, ground_truth):
            pred_colors = pred.get('colors', {})
            gt_colors = gt.get('colors', {})
            
            # Also try comprehensive analysis format
            if not pred_colors and 'comprehensive_analysis' in pred:
                pred_colors = pred['comprehensive_analysis'].get('colors', {})
            
            if gt_colors:
                # Calculate accuracy for each color category
                color_scores = []
                for color, gt_count in gt_colors.items():
                    pred_count = pred_colors.get(color, 0)
                    if gt_count > 0:
                        relative_error = abs(pred_count - gt_count) / gt_count
                        color_score = max(0.0, 1.0 - relative_error)
                    else:
                        color_score = 1.0 if pred_count == 0 else 0.0
                    color_scores.append(color_score)
                
                total_score += np.mean(color_scores) if color_scores else 0.0
        
        return total_score / len(predictions) if predictions else 0.0
    
    def _calculate_spatial_accuracy(self, predictions: List[Dict], ground_truth: List[Dict], **kwargs) -> float:
        """Calculate spatial relationship accuracy"""
        if len(predictions) != len(ground_truth):
            return 0.0
        
        total_score = 0.0
        for pred, gt in zip(predictions, ground_truth):
            pred_relationships = pred.get('relationships', [])
            gt_relationships = gt.get('spatial_relationships', [])
            
            # Also try comprehensive analysis format
            if not pred_relationships and 'comprehensive_analysis' in pred:
                pred_relationships = pred['comprehensive_analysis'].get('spatial_relationships', [])
            
            if gt_relationships:
                # Convert to comparable format
                pred_set = set()
                for rel in pred_relationships:
                    if all(key in rel for key in ['object1', 'relationship', 'object2']):
                        pred_set.add((rel['object1'].lower(), rel['relationship'].lower(), rel['object2'].lower()))
                
                gt_set = set()
                for rel in gt_relationships:
                    if all(key in rel for key in ['object1', 'relationship', 'object2']):
                        gt_set.add((rel['object1'].lower(), rel['relationship'].lower(), rel['object2'].lower()))
                
                # Calculate Jaccard similarity
                if gt_set:
                    intersection = len(pred_set & gt_set)
                    union = len(pred_set | gt_set)
                    score = intersection / union if union > 0 else 0.0
                    total_score += score
        
        return total_score / len(predictions) if predictions else 0.0
    
    def _calculate_text_recognition_accuracy(self, predictions: List[Dict], ground_truth: List[Dict], **kwargs) -> float:
        """Calculate text recognition accuracy"""
        if len(predictions) != len(ground_truth):
            return 0.0
        
        total_score = 0.0
        for pred, gt in zip(predictions, ground_truth):
            pred_text_found = pred.get('text_found', False)
            gt_text_items = gt.get('text_visible', [])
            
            # Also try comprehensive analysis format  
            if not pred_text_found and 'comprehensive_analysis' in pred:
                pred_text_found = pred['comprehensive_analysis'].get('text_found', False)
            
            gt_has_text = len(gt_text_items) > 0
            
            # Simple binary accuracy for text detection
            if pred_text_found == gt_has_text:
                total_score += 1.0
        
        return total_score / len(predictions) if predictions else 0.0
    
    def _calculate_material_recognition_accuracy(self, predictions: List[Dict], ground_truth: List[Dict], **kwargs) -> float:
        """Calculate material recognition accuracy"""
        if len(predictions) != len(ground_truth):
            return 0.0
        
        total_score = 0.0
        for pred, gt in zip(predictions, ground_truth):
            pred_materials = pred.get('materials', {})
            gt_materials = gt.get('materials', {})
            
            # Also try comprehensive analysis format
            if not pred_materials and 'comprehensive_analysis' in pred:
                pred_materials = pred['comprehensive_analysis'].get('materials', {})
            
            if gt_materials:
                # Calculate accuracy for each material category
                material_scores = []
                for material, gt_count in gt_materials.items():
                    pred_count = pred_materials.get(material, 0)
                    if gt_count > 0:
                        relative_error = abs(pred_count - gt_count) / gt_count
                        material_score = max(0.0, 1.0 - relative_error)
                    else:
                        material_score = 1.0 if pred_count == 0 else 0.0
                    material_scores.append(material_score)
                
                total_score += np.mean(material_scores) if material_scores else 0.0
        
        return total_score / len(predictions) if predictions else 0.0
    
    def _calculate_comprehensive_score(self, predictions: List[Dict], ground_truth: List[Dict], **kwargs) -> float:
        """Calculate comprehensive score combining multiple vision capabilities"""
        if len(predictions) != len(ground_truth):
            return 0.0
        
        # Calculate individual capability scores
        object_score = self._calculate_object_detection_accuracy(predictions, ground_truth)
        counting_score = self._calculate_counting_accuracy(predictions, ground_truth)
        color_score = self._calculate_color_accuracy(predictions, ground_truth)
        spatial_score = self._calculate_spatial_accuracy(predictions, ground_truth)
        text_score = self._calculate_text_recognition_accuracy(predictions, ground_truth)
        material_score = self._calculate_material_recognition_accuracy(predictions, ground_truth)
        
        # Weighted average (equal weights for now)
        weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        scores = [object_score, counting_score, color_score, spatial_score, text_score, material_score]
        
        weighted_sum = sum(w * s for w, s in zip(weights, scores))
        total_weight = sum(weights)
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0
    
    def _calculate_resolution_consistency(self, predictions: List[Dict], ground_truth: List[Dict], **kwargs) -> float:
        """Calculate consistency of results across different image resolutions"""
        if len(predictions) < 2:  # Need at least 2 resolutions to compare
            return 1.0
        
        # Group predictions by image content (assuming same objects across resolutions)
        resolution_groups = {}
        for pred in predictions:
            # Extract key identifying information
            total_objects = pred.get('total_objects', 0)
            if total_objects == 0 and 'comprehensive_analysis' in pred:
                total_objects = pred['comprehensive_analysis'].get('total_objects', 0)
            
            # Group by total object count (simple heuristic)
            if total_objects not in resolution_groups:
                resolution_groups[total_objects] = []
            resolution_groups[total_objects].append(pred)
        
        # Calculate consistency within each group
        consistency_scores = []
        for group in resolution_groups.values():
            if len(group) >= 2:
                # Compare first prediction with others in group
                base_pred = group[0]
                group_scores = []
                
                for other_pred in group[1:]:
                    # Calculate similarity in object detection
                    base_objects = set()
                    other_objects = set()
                    
                    if 'objects' in base_pred:
                        base_objects = {obj.get('name', '').lower() for obj in base_pred['objects']}
                    elif 'comprehensive_analysis' in base_pred:
                        base_objects = {obj.get('name', '').lower() for obj in base_pred['comprehensive_analysis'].get('objects', [])}
                    
                    if 'objects' in other_pred:
                        other_objects = {obj.get('name', '').lower() for obj in other_pred['objects']}
                    elif 'comprehensive_analysis' in other_pred:
                        other_objects = {obj.get('name', '').lower() for obj in other_pred['comprehensive_analysis'].get('objects', [])}
                    
                    # Jaccard similarity
                    if base_objects or other_objects:
                        intersection = len(base_objects & other_objects)
                        union = len(base_objects | other_objects)
                        similarity = intersection / union if union > 0 else 0.0
                        group_scores.append(similarity)
                
                if group_scores:
                    consistency_scores.append(np.mean(group_scores))
        
        return np.mean(consistency_scores) if consistency_scores else 1.0


# Global registry instance
metrics_registry = MetricsRegistry()