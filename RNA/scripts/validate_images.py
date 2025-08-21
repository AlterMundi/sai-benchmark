#!/usr/bin/env python3
"""
Image validation script for all datasets
Checks for corrupted, invalid, or problematic images
"""

import os
import sys
from pathlib import Path
from PIL import Image
import json
from typing import List, Dict, Any
import argparse
from tqdm import tqdm

def validate_image(image_path: Path) -> Dict[str, Any]:
    """Validate a single image file"""
    result = {
        'path': str(image_path),
        'valid': True,
        'errors': [],
        'width': None,
        'height': None,
        'format': None,
        'size_mb': None
    }
    
    try:
        # Check if file exists and get size
        if not image_path.exists():
            result['valid'] = False
            result['errors'].append('File does not exist')
            return result
            
        size_bytes = image_path.stat().st_size
        result['size_mb'] = round(size_bytes / (1024*1024), 2)
        
        # Check if file is too small (likely corrupted)
        if size_bytes < 1024:  # Less than 1KB
            result['valid'] = False
            result['errors'].append(f'File too small: {size_bytes} bytes')
            return result
            
        # Try to open and verify image
        with Image.open(image_path) as img:
            result['width'] = img.width
            result['height'] = img.height
            result['format'] = img.format
            
            # Check minimum dimensions
            if img.width < 32 or img.height < 32:
                result['valid'] = False
                result['errors'].append(f'Image too small: {img.width}x{img.height}')
                
            # Check maximum dimensions (reasonable limit)
            if img.width > 8000 or img.height > 8000:
                result['valid'] = False
                result['errors'].append(f'Image too large: {img.width}x{img.height}')
                
            # Try to load the image data (catches some corruption)
            img.load()
            
            # Verify image
            img.verify()
            
    except Exception as e:
        result['valid'] = False
        result['errors'].append(f'PIL Error: {str(e)}')
        
    return result

def validate_dataset_images(dataset_path: Path, extensions: List[str] = None) -> Dict[str, Any]:
    """Validate all images in a dataset directory"""
    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
        
    print(f"🔍 Validating images in: {dataset_path}")
    
    # Find all image files
    image_files = []
    for ext in extensions:
        image_files.extend(dataset_path.rglob(f"*{ext}"))
        image_files.extend(dataset_path.rglob(f"*{ext.upper()}"))
    
    if not image_files:
        return {
            'dataset': str(dataset_path),
            'total_images': 0,
            'valid_images': 0,
            'corrupted_images': 0,
            'errors': ['No image files found'],
            'details': []
        }
    
    print(f"Found {len(image_files)} image files")
    
    results = []
    valid_count = 0
    corrupted_count = 0
    
    # Validate each image with progress bar
    for image_path in tqdm(image_files, desc="Validating"):
        result = validate_image(image_path)
        results.append(result)
        
        if result['valid']:
            valid_count += 1
        else:
            corrupted_count += 1
            
    return {
        'dataset': str(dataset_path),
        'total_images': len(image_files),
        'valid_images': valid_count,
        'corrupted_images': corrupted_count,
        'validation_rate': round(valid_count / len(image_files) * 100, 2),
        'details': results
    }

def save_validation_report(results: Dict[str, Any], output_file: Path):
    """Save validation report to JSON file"""
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"📄 Validation report saved: {output_file}")

def print_summary(results: Dict[str, Any]):
    """Print validation summary"""
    print(f"\n📊 VALIDATION SUMMARY")
    print(f"Dataset: {results['dataset']}")
    print(f"Total images: {results['total_images']:,}")
    print(f"✅ Valid images: {results['valid_images']:,} ({results['validation_rate']:.1f}%)")
    print(f"❌ Corrupted images: {results['corrupted_images']:,}")
    
    if results['corrupted_images'] > 0:
        print(f"\n⚠️  CORRUPTED FILES:")
        for detail in results['details']:
            if not detail['valid']:
                print(f"  - {detail['path']}")
                for error in detail['errors']:
                    print(f"    Error: {error}")

def main():
    parser = argparse.ArgumentParser(description="Validate images in dataset directories")
    parser.add_argument("dataset_path", help="Path to dataset directory")
    parser.add_argument("--output", "-o", help="Output JSON report file")
    parser.add_argument("--summary-only", action="store_true", help="Only show summary")
    
    args = parser.parse_args()
    
    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        print(f"❌ Dataset path does not exist: {dataset_path}")
        sys.exit(1)
        
    # Validate images
    results = validate_dataset_images(dataset_path)
    
    # Print summary
    print_summary(results)
    
    # Save detailed report if requested
    if args.output:
        output_file = Path(args.output)
        if not args.summary_only:
            save_validation_report(results, output_file)
        else:
            # Save only summary
            summary = {k: v for k, v in results.items() if k != 'details'}
            save_validation_report(summary, output_file)
    
    # Exit with error code if there are corrupted images
    if results['corrupted_images'] > 0:
        print(f"\n⚠️  Found {results['corrupted_images']} corrupted images")
        sys.exit(1)
    else:
        print(f"\n✅ All images are valid!")

if __name__ == "__main__":
    main()