#!/usr/bin/env python3
"""
Utility script to create test datasets from image directories
"""

import json
import os
from pathlib import Path
import argparse

def create_dataset_from_directory(image_dir, output_file, has_smoke=False, bbox=None):
    """Create a JSON dataset from a directory of images"""
    
    image_dir = Path(image_dir)
    dataset = []
    
    # Find all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    images = []
    
    for ext in image_extensions:
        images.extend(image_dir.glob(f"*{ext}"))
        images.extend(image_dir.glob(f"*{ext.upper()}"))
    
    if not images:
        print(f"No images found in {image_dir}")
        return
    
    # Sort images
    images.sort()
    
    # Create dataset entry
    dataset_entry = {
        "sequence_id": image_dir.name,
        "images": [str(img) for img in images],
        "ground_truth": {
            "has_smoke": has_smoke,
            "bbox": bbox or [0, 0, 0, 0]
        }
    }
    
    dataset.append(dataset_entry)
    
    # Write to JSON file
    with open(output_file, 'w') as f:
        json.dump(dataset, f, indent=2)
    
    print(f"Created dataset with {len(images)} images: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create test dataset from image directory")
    parser.add_argument("image_dir", help="Directory containing images")
    parser.add_argument("output_file", help="Output JSON file")
    parser.add_argument("--has-smoke", action="store_true", help="Images contain smoke/fire")
    parser.add_argument("--bbox", nargs=4, type=float, help="Bounding box: x_center y_center width height")
    
    args = parser.parse_args()
    
    create_dataset_from_directory(
        args.image_dir, 
        args.output_file, 
        args.has_smoke, 
        args.bbox
    )