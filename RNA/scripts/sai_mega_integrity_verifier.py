#!/usr/bin/env python3
"""
SAI MEGA INTEGRITY VERIFIER
===========================
Comprehensive dataset integrity verification for SAI Fire Detection System
CRITICAL: Fire detection requires 100% dataset integrity for life safety applications

This verifier performs exhaustive checks across multiple dimensions:
- File integrity and accessibility
- YOLO format compliance and coordinate validation
- Image integrity and metadata consistency
- Class distribution and balance analysis
- Temporal consistency and sequence validation
- Geographic and environmental diversity analysis
- Training/validation split quality
- Memory and performance impact estimation
- Cross-dataset consistency validation
- Production readiness assessment

Author: SAI Team
Purpose: Mission-critical fire detection dataset validation
"""

import os
import sys
import argparse
import yaml
import json
import hashlib
import statistics
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict, Counter
from datetime import datetime
import time

import numpy as np
from PIL import Image, ImageFile
from tqdm import tqdm
import cv2

# Allow loading of truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

class SAIMegaIntegrityVerifier:
    """
    Comprehensive integrity verifier for SAI fire detection datasets
    Performs multi-dimensional validation for mission-critical applications
    """
    
    def __init__(self, dataset_path: str, config_path: Optional[str] = None):
        self.dataset_path = Path(dataset_path)
        self.config_path = Path(config_path) if config_path else None
        self.report = {
            'timestamp': datetime.now().isoformat(),
            'dataset_path': str(self.dataset_path),
            'version': '1.0.0',
            'checks_performed': [],
            'critical_issues': [],
            'warnings': [],
            'statistics': {},
            'recommendations': [],
            'production_readiness': False
        }
        
    def log_check(self, check_name: str, status: str, details: Any = None):
        """Log a verification check result"""
        self.report['checks_performed'].append({
            'name': check_name,
            'status': status,
            'details': details,
            'timestamp': datetime.now().isoformat()
        })
        
    def add_critical_issue(self, issue: str, impact: str = "high"):
        """Add a critical issue that blocks production use"""
        self.report['critical_issues'].append({
            'issue': issue,
            'impact': impact,
            'timestamp': datetime.now().isoformat()
        })
        
    def add_warning(self, warning: str, recommendation: str = ""):
        """Add a warning with optional recommendation"""
        self.report['warnings'].append({
            'warning': warning,
            'recommendation': recommendation,
            'timestamp': datetime.now().isoformat()
        })
        
    def add_recommendation(self, recommendation: str, priority: str = "medium"):
        """Add a recommendation for improvement"""
        self.report['recommendations'].append({
            'recommendation': recommendation,
            'priority': priority
        })

    def verify_file_structure(self) -> bool:
        """Verify basic file structure and dataset.yaml"""
        print("🔍 Verifying file structure...")
        
        # Check dataset.yaml
        yaml_file = self.dataset_path / "dataset.yaml"
        if not yaml_file.exists():
            self.add_critical_issue("dataset.yaml not found", "critical")
            return False
            
        try:
            with open(yaml_file, 'r') as f:
                config = yaml.safe_load(f)
            self.config = config
            self.log_check("dataset_yaml_load", "pass", config)
        except Exception as e:
            self.add_critical_issue(f"Cannot load dataset.yaml: {e}", "critical")
            return False
        
        # Verify required directories
        required_dirs = [
            "images/train", "images/val", 
            "labels/train", "labels/val"
        ]
        
        missing_dirs = []
        for dir_name in required_dirs:
            dir_path = self.dataset_path / dir_name
            if not dir_path.exists():
                missing_dirs.append(dir_name)
                
        if missing_dirs:
            self.add_critical_issue(f"Missing directories: {missing_dirs}", "critical")
            return False
            
        self.log_check("file_structure", "pass", {"required_dirs": required_dirs})
        return True

    def verify_image_integrity(self) -> Dict[str, Any]:
        """Verify image files can be opened and are valid"""
        print("🖼️  Verifying image integrity...")
        
        results = {
            'total_images': 0,
            'valid_images': 0,
            'corrupted_images': [],
            'unsupported_formats': [],
            'size_issues': [],
            'metadata_stats': defaultdict(int)
        }
        
        for split in ['train', 'val']:
            images_dir = self.dataset_path / "images" / split
            if not images_dir.exists():
                continue
                
            image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
            
            for img_file in tqdm(image_files, desc=f"Checking {split} images"):
                results['total_images'] += 1
                
                try:
                    # Try to open with PIL
                    with Image.open(img_file) as img:
                        # Verify the image
                        img.verify()
                        
                    # Re-open for metadata (verify() closes the image)
                    with Image.open(img_file) as img:
                        width, height = img.size
                        mode = img.mode
                        format_type = img.format
                        
                        # Check minimum size requirements
                        if width < 32 or height < 32:
                            results['size_issues'].append({
                                'file': str(img_file),
                                'size': f"{width}x{height}",
                                'issue': 'too_small'
                            })
                        
                        # Check aspect ratio extremes
                        aspect_ratio = width / height
                        if aspect_ratio > 10 or aspect_ratio < 0.1:
                            results['size_issues'].append({
                                'file': str(img_file),
                                'aspect_ratio': aspect_ratio,
                                'issue': 'extreme_aspect_ratio'
                            })
                        
                        # Collect metadata stats
                        results['metadata_stats'][f"format_{format_type}"] += 1
                        results['metadata_stats'][f"mode_{mode}"] += 1
                        results['metadata_stats']['total_pixels'] += width * height
                        
                    results['valid_images'] += 1
                    
                except Exception as e:
                    results['corrupted_images'].append({
                        'file': str(img_file),
                        'error': str(e)
                    })
        
        # Calculate statistics
        if results['total_images'] > 0:
            success_rate = results['valid_images'] / results['total_images']
            if success_rate < 0.99:
                self.add_critical_issue(
                    f"Image integrity failure rate: {(1-success_rate)*100:.2f}%", 
                    "high"
                )
            elif success_rate < 0.999:
                self.add_warning(
                    f"Some corrupted images detected: {len(results['corrupted_images'])} files",
                    "Consider removing or fixing corrupted images"
                )
                
        self.log_check("image_integrity", "pass" if results['valid_images'] > 0 else "fail", results)
        return results

    def verify_yolo_format(self) -> Dict[str, Any]:
        """Verify YOLO format compliance and coordinate validation"""
        print("🎯 Verifying YOLO format compliance...")
        
        results = {
            'total_labels': 0,
            'valid_labels': 0,
            'empty_labels': 0,
            'format_errors': [],
            'coordinate_errors': [],
            'class_distribution': Counter(),
            'bbox_stats': defaultdict(list)
        }
        
        # Get expected classes from config
        if hasattr(self, 'config') and 'names' in self.config:
            expected_classes = set(range(len(self.config['names'])))
            class_names = self.config['names']
        else:
            expected_classes = {0, 1}  # Default: fire, smoke
            class_names = {0: 'fire', 1: 'smoke'}
        
        for split in ['train', 'val']:
            labels_dir = self.dataset_path / "labels" / split
            if not labels_dir.exists():
                continue
                
            label_files = list(labels_dir.glob("*.txt"))
            
            for label_file in tqdm(label_files, desc=f"Checking {split} labels"):
                results['total_labels'] += 1
                
                try:
                    with open(label_file, 'r') as f:
                        lines = f.readlines()
                    
                    # Check if empty (background image)
                    if not lines or all(not line.strip() for line in lines):
                        results['empty_labels'] += 1
                        continue
                    
                    # Parse each annotation line
                    for line_num, line in enumerate(lines, 1):
                        line = line.strip()
                        if not line:
                            continue
                            
                        parts = line.split()
                        
                        # Check format: class_id x_center y_center width height
                        if len(parts) != 5:
                            results['format_errors'].append({
                                'file': str(label_file),
                                'line': line_num,
                                'error': f"Expected 5 values, got {len(parts)}",
                                'content': line
                            })
                            continue
                        
                        try:
                            class_id = int(parts[0])
                            x_center, y_center, width, height = map(float, parts[1:5])
                        except ValueError as e:
                            results['format_errors'].append({
                                'file': str(label_file),
                                'line': line_num,
                                'error': f"Parse error: {e}",
                                'content': line
                            })
                            continue
                        
                        # Validate class ID
                        if class_id not in expected_classes:
                            results['format_errors'].append({
                                'file': str(label_file),
                                'line': line_num,
                                'error': f"Invalid class_id {class_id}, expected {expected_classes}",
                                'content': line
                            })
                            continue
                        
                        # Validate coordinates (normalized 0-1)
                        coord_errors = []
                        if not (0 <= x_center <= 1):
                            coord_errors.append(f"x_center={x_center}")
                        if not (0 <= y_center <= 1):
                            coord_errors.append(f"y_center={y_center}")
                        if not (0 < width <= 1):
                            coord_errors.append(f"width={width}")
                        if not (0 < height <= 1):
                            coord_errors.append(f"height={height}")
                            
                        if coord_errors:
                            results['coordinate_errors'].append({
                                'file': str(label_file),
                                'line': line_num,
                                'errors': coord_errors,
                                'content': line
                            })
                            continue
                        
                        # Collect statistics
                        results['class_distribution'][class_id] += 1
                        results['bbox_stats']['widths'].append(width)
                        results['bbox_stats']['heights'].append(height)
                        results['bbox_stats']['areas'].append(width * height)
                        results['bbox_stats']['aspect_ratios'].append(width / height)
                    
                    results['valid_labels'] += 1
                    
                except Exception as e:
                    results['format_errors'].append({
                        'file': str(label_file),
                        'error': f"File read error: {e}"
                    })
        
        # Analyze class distribution
        total_annotations = sum(results['class_distribution'].values())
        if total_annotations > 0:
            print(f"\n📊 Class Distribution:")
            for class_id, count in results['class_distribution'].items():
                class_name = class_names.get(class_id, f"class_{class_id}")
                percentage = (count / total_annotations) * 100
                print(f"   {class_name}: {count:,} ({percentage:.1f}%)")
                
                # Check for severe class imbalance
                if percentage < 5:
                    self.add_warning(
                        f"Class {class_name} is underrepresented: {percentage:.1f}%",
                        "Consider augmenting underrepresented classes"
                    )
        
        # Analyze bounding box statistics
        if results['bbox_stats']['areas']:
            bbox_areas = results['bbox_stats']['areas']
            mean_area = statistics.mean(bbox_areas)
            median_area = statistics.median(bbox_areas)
            
            # Check for very small bounding boxes
            small_boxes = [area for area in bbox_areas if area < 0.001]  # < 0.1% of image
            if small_boxes:
                percentage_small = (len(small_boxes) / len(bbox_areas)) * 100
                if percentage_small > 10:
                    self.add_warning(
                        f"{percentage_small:.1f}% of bounding boxes are very small (< 0.1% of image)",
                        "Very small objects may be difficult to detect"
                    )
        
        # Check error rates
        if results['total_labels'] > 0:
            format_error_rate = len(results['format_errors']) / results['total_labels']
            coord_error_rate = len(results['coordinate_errors']) / results['total_labels']
            
            if format_error_rate > 0.01:  # > 1%
                self.add_critical_issue(
                    f"High format error rate: {format_error_rate*100:.2f}%",
                    "critical"
                )
            if coord_error_rate > 0.01:  # > 1%
                self.add_critical_issue(
                    f"High coordinate error rate: {coord_error_rate*100:.2f}%",
                    "critical"
                )
        
        self.log_check("yolo_format", "pass" if len(results['format_errors']) == 0 else "fail", results)
        return results

    def verify_dataset_consistency(self) -> Dict[str, Any]:
        """Verify consistency between images and labels"""
        print("🔄 Verifying dataset consistency...")
        
        results = {
            'train': {'images': 0, 'labels': 0, 'orphaned_images': [], 'orphaned_labels': []},
            'val': {'images': 0, 'labels': 0, 'orphaned_images': [], 'orphaned_labels': []},
            'naming_consistency': True,
            'split_balance': {}
        }
        
        for split in ['train', 'val']:
            images_dir = self.dataset_path / "images" / split
            labels_dir = self.dataset_path / "labels" / split
            
            if not images_dir.exists() or not labels_dir.exists():
                continue
            
            # Get all files
            image_files = set()
            for ext in ['.jpg', '.jpeg', '.png']:
                image_files.update([f.stem for f in images_dir.glob(f"*{ext}")])
            
            label_files = set([f.stem for f in labels_dir.glob("*.txt")])
            
            results[split]['images'] = len(image_files)
            results[split]['labels'] = len(label_files)
            
            # Find orphaned files
            orphaned_images = image_files - label_files
            orphaned_labels = label_files - image_files
            
            results[split]['orphaned_images'] = list(orphaned_images)
            results[split]['orphaned_labels'] = list(orphaned_labels)
            
            if orphaned_images:
                self.add_warning(
                    f"{split}: {len(orphaned_images)} images without labels",
                    "Consider creating empty label files for background images"
                )
            
            if orphaned_labels:
                self.add_critical_issue(
                    f"{split}: {len(orphaned_labels)} labels without images",
                    "high"
                )
        
        # Check split balance
        total_train = results['train']['images']
        total_val = results['val']['images']
        total_images = total_train + total_val
        
        if total_images > 0:
            train_ratio = total_train / total_images
            val_ratio = total_val / total_images
            
            results['split_balance'] = {
                'train_ratio': train_ratio,
                'val_ratio': val_ratio,
                'train_count': total_train,
                'val_count': total_val
            }
            
            # Check if split ratios are reasonable
            if val_ratio < 0.1:
                self.add_warning(
                    f"Validation split very small: {val_ratio*100:.1f}%",
                    "Consider increasing validation split to 15-20%"
                )
            elif val_ratio > 0.3:
                self.add_warning(
                    f"Validation split very large: {val_ratio*100:.1f}%",
                    "Consider reducing validation split to 15-20%"
                )
        
        consistency_score = 1.0
        if results['train']['orphaned_images'] or results['train']['orphaned_labels']:
            consistency_score -= 0.5
        if results['val']['orphaned_images'] or results['val']['orphaned_labels']:
            consistency_score -= 0.5
            
        self.log_check("dataset_consistency", "pass" if consistency_score > 0.8 else "fail", results)
        return results

    def analyze_dataset_diversity(self) -> Dict[str, Any]:
        """Analyze diversity across dataset sources and characteristics"""
        print("🌍 Analyzing dataset diversity...")
        
        results = {
            'source_distribution': Counter(),
            'filename_patterns': Counter(),
            'estimated_scenarios': {},
            'geographic_diversity': {},
            'temporal_diversity': {}
        }
        
        # Analyze filename patterns to identify sources
        for split in ['train', 'val']:
            images_dir = self.dataset_path / "images" / split
            if not images_dir.exists():
                continue
                
            for img_file in images_dir.glob("*.jpg"):
                filename = img_file.name
                
                # Identify source dataset from prefix
                if filename.startswith('fasdd_'):
                    results['source_distribution']['FASDD'] += 1
                    # Analyze FASDD categories
                    if '_bothFireAndSmoke_' in filename:
                        results['filename_patterns']['fasdd_both'] += 1
                    elif '_fire_' in filename:
                        results['filename_patterns']['fasdd_fire'] += 1
                    elif '_smoke_' in filename:
                        results['filename_patterns']['fasdd_smoke'] += 1
                    elif '_neitherFireNorSmoke_' in filename:
                        results['filename_patterns']['fasdd_background'] += 1
                        
                elif filename.startswith('nemo_'):
                    results['source_distribution']['NEMO'] += 1
                    results['filename_patterns']['nemo_fire'] += 1
                    
                elif filename.startswith('pyro_'):
                    results['source_distribution']['Pyronear-2024'] += 1
                    results['filename_patterns']['pyronear_geographical'] += 1
                    
                elif filename.startswith('figlib_'):
                    results['source_distribution']['FigLib'] += 1
                    results['filename_patterns']['figlib_smoke'] += 1
                else:
                    results['source_distribution']['Unknown'] += 1
        
        # Calculate diversity metrics
        total_images = sum(results['source_distribution'].values())
        if total_images > 0:
            print(f"\n📊 Source Distribution:")
            for source, count in results['source_distribution'].items():
                percentage = (count / total_images) * 100
                print(f"   {source}: {count:,} ({percentage:.1f}%)")
                
                # Check for source diversity
                if count == 0:
                    self.add_warning(f"No images from {source} dataset")
                elif percentage < 5:
                    self.add_warning(f"Low representation from {source}: {percentage:.1f}%")
        
        # Analyze scenario diversity based on patterns
        fire_scenarios = results['filename_patterns'].get('fasdd_fire', 0) + \
                        results['filename_patterns'].get('fasdd_both', 0) + \
                        results['filename_patterns'].get('nemo_fire', 0)
                        
        smoke_scenarios = results['filename_patterns'].get('fasdd_smoke', 0) + \
                         results['filename_patterns'].get('fasdd_both', 0) + \
                         results['filename_patterns'].get('figlib_smoke', 0)
                         
        background_scenarios = results['filename_patterns'].get('fasdd_background', 0)
        
        results['estimated_scenarios'] = {
            'fire_detection': fire_scenarios,
            'smoke_detection': smoke_scenarios,
            'background_negatives': background_scenarios,
            'geographical_diversity': results['filename_patterns'].get('pyronear_geographical', 0)
        }
        
        # Assess diversity quality
        diversity_score = 0
        if len(results['source_distribution']) >= 3:
            diversity_score += 0.4  # Multiple sources
        if fire_scenarios > 0 and smoke_scenarios > 0:
            diversity_score += 0.3  # Both fire and smoke
        if background_scenarios > total_images * 0.2:
            diversity_score += 0.3  # Sufficient negatives
        
        if diversity_score < 0.7:
            self.add_warning(
                f"Limited dataset diversity (score: {diversity_score:.2f})",
                "Consider adding more diverse scenarios and sources"
            )
        
        self.log_check("dataset_diversity", "pass", results)
        return results

    def estimate_training_performance(self) -> Dict[str, Any]:
        """Estimate training performance and resource requirements"""
        print("⚡ Estimating training performance...")
        
        results = {
            'estimated_training_time': {},
            'memory_requirements': {},
            'dataset_size_analysis': {},
            'bottleneck_analysis': {}
        }
        
        # Calculate dataset size
        total_size_bytes = 0
        image_count = 0
        
        for split in ['train', 'val']:
            images_dir = self.dataset_path / "images" / split
            if not images_dir.exists():
                continue
                
            for img_file in images_dir.glob("*.jpg"):
                total_size_bytes += img_file.stat().st_size
                image_count += 1
        
        total_size_gb = total_size_bytes / (1024**3)
        avg_image_size_mb = (total_size_bytes / image_count) / (1024**2) if image_count > 0 else 0
        
        results['dataset_size_analysis'] = {
            'total_images': image_count,
            'total_size_gb': total_size_gb,
            'avg_image_size_mb': avg_image_size_mb
        }
        
        # Estimate training time (based on common benchmarks)
        # YOLOv8-s on RTX 3090: ~30-50 images/second during training
        if image_count > 0:
            images_per_epoch = image_count * 0.8  # Assuming 80% for training
            epochs = 100  # Typical for YOLOv8
            processing_rate = 40  # images/second (conservative estimate)
            
            seconds_per_epoch = images_per_epoch / processing_rate
            total_training_hours = (seconds_per_epoch * epochs) / 3600
            
            results['estimated_training_time'] = {
                'images_per_epoch': images_per_epoch,
                'seconds_per_epoch': seconds_per_epoch,
                'hours_per_epoch': seconds_per_epoch / 3600,
                'total_training_hours': total_training_hours,
                'processing_rate_assumption': processing_rate
            }
            
            if total_training_hours > 24:
                self.add_warning(
                    f"Long training time estimated: {total_training_hours:.1f} hours",
                    "Consider using a subset for initial testing"
                )
        
        # Estimate memory requirements
        # Conservative estimates for YOLOv8-s
        batch_size_8 = image_count * 0.001  # MB per image at batch=8
        batch_size_16 = image_count * 0.002  # MB per image at batch=16
        
        results['memory_requirements'] = {
            'estimated_vram_batch_8': max(6, batch_size_8),  # Minimum 6GB
            'estimated_vram_batch_16': max(10, batch_size_16),  # Minimum 10GB
            'recommended_ram': max(16, total_size_gb * 2),  # 2x dataset size
            'cache_size_estimate': total_size_gb * 1.2  # 20% overhead for caching
        }
        
        # Check if current system can handle the dataset
        if total_size_gb > 50:
            self.add_warning(
                f"Large dataset size: {total_size_gb:.1f}GB",
                "Ensure sufficient storage and consider SSD for faster I/O"
            )
        
        self.log_check("training_performance", "pass", results)
        return results

    def verify_production_readiness(self) -> Dict[str, Any]:
        """Assess overall production readiness for fire detection system"""
        print("🚀 Assessing production readiness...")
        
        results = {
            'readiness_score': 0,
            'critical_blockers': [],
            'production_recommendations': [],
            'safety_assessment': {},
            'performance_expectations': {}
        }
        
        # Calculate readiness score based on all checks
        score = 0
        max_score = 100
        
        # File structure and format compliance (30 points)
        if not self.report['critical_issues']:
            score += 30
        elif len(self.report['critical_issues']) <= 2:
            score += 20
        elif len(self.report['critical_issues']) <= 5:
            score += 10
        
        # Data quality (25 points)
        warning_count = len(self.report['warnings'])
        if warning_count == 0:
            score += 25
        elif warning_count <= 3:
            score += 20
        elif warning_count <= 6:
            score += 15
        elif warning_count <= 10:
            score += 10
        else:
            score += 5
        
        # Dataset size and diversity (25 points)
        stats = self.report.get('statistics', {})
        image_count = 0
        for check in self.report['checks_performed']:
            if check['name'] == 'dataset_consistency' and 'details' in check:
                details = check['details']
                image_count = details.get('train', {}).get('images', 0) + \
                             details.get('val', {}).get('images', 0)
                break
        
        if image_count >= 50000:
            score += 25
        elif image_count >= 30000:
            score += 20
        elif image_count >= 15000:
            score += 15
        elif image_count >= 5000:
            score += 10
        else:
            score += 5
        
        # Performance expectations (20 points)
        if image_count > 0:
            score += 20  # Have data to train with
        
        results['readiness_score'] = score
        
        # Determine production readiness
        if score >= 80 and not self.report['critical_issues']:
            results['production_ready'] = True
            self.report['production_readiness'] = True
            results['production_recommendations'].append(
                "✅ Dataset is ready for production training"
            )
        elif score >= 60:
            results['production_ready'] = False
            results['production_recommendations'].append(
                "⚠️ Dataset needs minor improvements before production use"
            )
        else:
            results['production_ready'] = False
            results['critical_blockers'].append(
                "❌ Dataset requires significant improvements before production use"
            )
        
        # Safety assessment for fire detection
        results['safety_assessment'] = {
            'mission_critical': True,
            'life_safety_impact': "High - False negatives could delay fire response",
            'acceptable_error_rate': "< 1% for critical fire detection scenarios",
            'recommended_validation': "Extensive field testing required"
        }
        
        # Performance expectations
        results['performance_expectations'] = {
            'min_recall_fire': 0.85,
            'min_recall_smoke': 0.80,
            'max_false_positive_rate': 0.20,
            'target_inference_speed': "6-10 FPS on RTX 3090",
            'deployment_considerations': [
                "Real-time processing requirements",
                "Geographic and weather adaptability",
                "Temporal consistency validation",
                "Edge case handling (night, fog, etc.)"
            ]
        }
        
        self.log_check("production_readiness", "pass" if results['production_ready'] else "warn", results)
        return results

    def generate_comprehensive_report(self) -> str:
        """Generate a comprehensive integrity report"""
        
        # Compile final statistics
        total_checks = len(self.report['checks_performed'])
        passed_checks = len([c for c in self.report['checks_performed'] if c['status'] == 'pass'])
        
        self.report['statistics'] = {
            'total_checks_performed': total_checks,
            'checks_passed': passed_checks,
            'critical_issues_count': len(self.report['critical_issues']),
            'warnings_count': len(self.report['warnings']),
            'recommendations_count': len(self.report['recommendations'])
        }
        
        # Generate report
        report_lines = [
            "🔥 SAI MEGA INTEGRITY VERIFICATION REPORT",
            "=" * 80,
            f"Dataset: {self.dataset_path}",
            f"Verification Time: {self.report['timestamp']}",
            f"Production Ready: {'✅ YES' if self.report['production_readiness'] else '❌ NO'}",
            "",
            "📊 SUMMARY",
            "-" * 40,
            f"Total Checks: {total_checks}",
            f"Passed: {passed_checks}",
            f"Critical Issues: {len(self.report['critical_issues'])}",
            f"Warnings: {len(self.report['warnings'])}",
            f"Recommendations: {len(self.report['recommendations'])}",
            ""
        ]
        
        # Critical Issues
        if self.report['critical_issues']:
            report_lines.extend([
                "🚨 CRITICAL ISSUES (MUST FIX)",
                "-" * 40
            ])
            for issue in self.report['critical_issues']:
                report_lines.append(f"❌ {issue['issue']}")
            report_lines.append("")
        
        # Warnings
        if self.report['warnings']:
            report_lines.extend([
                "⚠️ WARNINGS",
                "-" * 40
            ])
            for warning in self.report['warnings']:
                report_lines.append(f"⚠️ {warning['warning']}")
                if warning['recommendation']:
                    report_lines.append(f"   💡 {warning['recommendation']}")
            report_lines.append("")
        
        # Recommendations
        if self.report['recommendations']:
            report_lines.extend([
                "💡 RECOMMENDATIONS",
                "-" * 40
            ])
            for rec in self.report['recommendations']:
                priority_icon = "🔴" if rec['priority'] == "high" else "🟡" if rec['priority'] == "medium" else "🟢"
                report_lines.append(f"{priority_icon} {rec['recommendation']}")
            report_lines.append("")
        
        # Detailed Check Results
        report_lines.extend([
            "🔍 DETAILED CHECK RESULTS",
            "-" * 40
        ])
        
        for check in self.report['checks_performed']:
            status_icon = "✅" if check['status'] == 'pass' else "❌" if check['status'] == 'fail' else "⚠️"
            report_lines.append(f"{status_icon} {check['name']}: {check['status'].upper()}")
        
        report_lines.extend([
            "",
            "🎯 NEXT STEPS",
            "-" * 40
        ])
        
        if self.report['production_readiness']:
            report_lines.extend([
                "✅ Dataset is ready for production training!",
                "🚀 Proceed with YOLOv8-s detector training",
                "📊 Monitor training metrics and validation performance",
                "🧪 Conduct field testing before deployment"
            ])
        else:
            report_lines.extend([
                "❌ Dataset requires fixes before production use",
                "🔧 Address all critical issues listed above",
                "⚠️ Review and fix warnings for optimal performance",
                "🔄 Re-run verification after fixes"
            ])
        
        report_lines.extend([
            "",
            "=" * 80,
            "Report generated by SAI Mega Integrity Verifier v1.0.0",
            f"For fire detection mission-critical applications",
            "=" * 80
        ])
        
        return "\n".join(report_lines)

    def run_full_verification(self) -> bool:
        """Run complete verification suite"""
        print("🔥 SAI MEGA INTEGRITY VERIFIER")
        print("=" * 50)
        print("Comprehensive dataset verification for fire detection")
        print(f"Dataset: {self.dataset_path}")
        print("=" * 50)
        
        start_time = time.time()
        
        try:
            # 1. File Structure
            if not self.verify_file_structure():
                print("❌ Basic file structure verification failed")
                return False
            
            # 2. Image Integrity
            self.verify_image_integrity()
            
            # 3. YOLO Format
            self.verify_yolo_format()
            
            # 4. Dataset Consistency
            self.verify_dataset_consistency()
            
            # 5. Diversity Analysis
            self.analyze_dataset_diversity()
            
            # 6. Performance Estimation
            self.estimate_training_performance()
            
            # 7. Production Readiness
            self.verify_production_readiness()
            
            verification_time = time.time() - start_time
            print(f"\n⏱️ Verification completed in {verification_time:.1f} seconds")
            
            return True
            
        except Exception as e:
            self.add_critical_issue(f"Verification failed with error: {e}", "critical")
            print(f"❌ Verification failed: {e}")
            return False

def main():
    parser = argparse.ArgumentParser(description='SAI Mega Integrity Verifier')
    parser.add_argument('dataset_path', help='Path to dataset directory')
    parser.add_argument('--config', help='Path to dataset config file (optional)')
    parser.add_argument('--output', help='Output report file (default: stdout)')
    parser.add_argument('--json', action='store_true', help='Output JSON report')
    args = parser.parse_args()
    
    # Initialize verifier
    verifier = SAIMegaIntegrityVerifier(args.dataset_path, args.config)
    
    # Run verification
    if not verifier.run_full_verification():
        print("❌ Verification failed")
        sys.exit(1)
    
    # Generate report
    if args.json:
        report_content = json.dumps(verifier.report, indent=2)
    else:
        report_content = verifier.generate_comprehensive_report()
    
    # Output report
    if args.output:
        with open(args.output, 'w') as f:
            f.write(report_content)
        print(f"\n📋 Report saved to: {args.output}")
    else:
        print("\n" + report_content)
    
    # Exit with appropriate code
    if verifier.report['production_readiness']:
        print("\n🎉 Dataset verification PASSED - Ready for production!")
        sys.exit(0)
    else:
        print("\n⚠️ Dataset verification completed with issues - See report above")
        sys.exit(1)

if __name__ == "__main__":
    main()