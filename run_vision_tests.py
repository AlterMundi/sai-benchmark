#!/usr/bin/env python3
"""
Vision Capability Test Runner

Demonstration script for running comprehensive vision capability tests
using the table-objects images with different resolutions.

Usage:
    python run_vision_tests.py
"""

from core.test_suite import run_test_suite
from core.prompt_registry import prompt_registry
from core.metrics_registry import metrics_registry
import sys
from pathlib import Path

def main():
    """Run the vision capability test suite"""
    
    print("=" * 60)
    print("VISION CAPABILITY TEST SUITE")
    print("=" * 60)
    print()
    
    # Show available vision prompts
    print("Available Vision Prompts:")
    vision_prompts = prompt_registry.list_prompts(tags=["vision"])
    for prompt in vision_prompts:
        print(f"  - {prompt.id}: {prompt.description}")
    print()
    
    # Show available vision metrics
    print("Available Vision Metrics:")
    vision_metrics = [
        "object_detection_accuracy",
        "counting_accuracy", 
        "color_accuracy",
        "spatial_accuracy",
        "text_recognition_accuracy",
        "material_recognition_accuracy",
        "comprehensive_score",
        "resolution_consistency"
    ]
    for metric in vision_metrics:
        if metric in metrics_registry.metrics:
            config = metrics_registry.metrics[metric]
            print(f"  - {metric}: {config.description}")
    print()
    
    # Check if test suite and dataset exist
    suite_path = Path("suites/vision_capability_tests.yaml")
    dataset_path = Path("datasets/vision_table_objects.json")
    
    if not suite_path.exists():
        print(f"❌ Test suite not found: {suite_path}")
        return 1
    
    if not dataset_path.exists():
        print(f"❌ Dataset not found: {dataset_path}")
        return 1
    
    # Check if images exist
    image_paths = [
        Path("images/table-objects_640.jpg"),
        Path("images/table-objects_1920.jpg"),
        Path("images/table-objects_3840.jpg")
    ]
    
    missing_images = [img for img in image_paths if not img.exists()]
    if missing_images:
        print("❌ Missing images:")
        for img in missing_images:
            print(f"   {img}")
        return 1
    
    print("✅ All required files found!")
    print()
    
    # Information about the test suite
    print("Test Suite Information:")
    print("  - 3 table-objects images (640px, 1920px, 3840px)")
    print("  - 10 different vision capability prompts")
    print("  - 8 vision-specific metrics")
    print("  - Tests object detection, counting, colors, spatial relationships, etc.")
    print("  - Evaluates resolution sensitivity and consistency")
    print()
    
    print("To run the vision tests (requires configured models):")
    print(f"  python -m core.test_suite {suite_path}")
    print()
    print("Or use the test suite runner:")
    print("  python run_suite.py suites/vision_capability_tests.yaml")
    print()
    
    print("Dataset Structure:")
    print("  - Each image has detailed ground truth annotations")
    print("  - 15+ objects per image with categories, colors, materials, positions")
    print("  - Spatial relationships defined between objects")
    print("  - Text and material information included")
    print()
    
    print("Vision Test Categories:")
    print("  1. Object Detection & Classification")
    print("  2. Object Counting by Category")
    print("  3. Color Recognition")
    print("  4. Spatial Relationship Analysis")
    print("  5. Size Estimation")
    print("  6. Text Recognition (OCR)")
    print("  7. Material Recognition")
    print("  8. Shape Analysis")
    print("  9. Comprehensive Multi-capability Analysis") 
    print("  10. Resolution Sensitivity Testing")
    print()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())