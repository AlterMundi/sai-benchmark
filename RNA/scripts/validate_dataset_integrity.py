#!/usr/bin/env python3
"""
Validate dataset integrity before training
Checks for corrupted images, missing labels, and format issues
"""

import os
from pathlib import Path
from PIL import Image, ImageFile
from tqdm import tqdm
import argparse
import yaml
import shutil

# Allow loading of truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

def validate_image(img_path):
    """Validate a single image file"""
    try:
        with Image.open(img_path) as img:
            img.verify()  # Verify image integrity
            return True, None
    except Exception as e:
        return False, str(e)

def validate_label(label_path):
    """Validate YOLO format label file"""
    if not label_path.exists():
        return False, "Missing label file"
    
    try:
        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:  # Empty line is OK
                continue
                
            parts = line.split()
            if len(parts) != 5:
                return False, f"Line {i+1}: Expected 5 values, got {len(parts)}"
            
            try:
                class_id = int(parts[0])
                x_center, y_center, width, height = map(float, parts[1:5])
                
                # Validate class ID
                if class_id not in [0, 1]:  # fire=0, smoke=1
                    return False, f"Line {i+1}: Invalid class_id {class_id}, expected 0 or 1"
                
                # Validate coordinates (should be normalized 0-1)
                if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 
                       0 <= width <= 1 and 0 <= height <= 1):
                    return False, f"Line {i+1}: Coordinates not normalized: {x_center}, {y_center}, {width}, {height}"
                       
            except ValueError as e:
                return False, f"Line {i+1}: Parse error: {e}"
        
        return True, None
        
    except Exception as e:
        return False, f"File read error: {e}"

def validate_dataset_split(images_dir, labels_dir, split_name):
    """Validate a dataset split (train/val)"""
    
    print(f"\n🔍 Validating {split_name} split...")
    print(f"Images: {images_dir}")
    print(f"Labels: {labels_dir}")
    
    if not images_dir.exists():
        print(f"❌ Images directory not found: {images_dir}")
        return 0, 0, []
    
    if not labels_dir.exists():
        print(f"❌ Labels directory not found: {labels_dir}")
        return 0, 0, []
    
    # Find all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(images_dir.glob(f"*{ext}"))
        image_files.extend(images_dir.glob(f"*{ext.upper()}"))
    
    print(f"📊 Found {len(image_files)} images to validate")
    
    valid_count = 0
    invalid_files = []
    
    for img_file in tqdm(image_files, desc=f"Validating {split_name}"):
        # Validate image
        img_valid, img_error = validate_image(img_file)
        
        # Validate corresponding label
        label_file = labels_dir / (img_file.stem + '.txt')
        label_valid, label_error = validate_label(label_file)
        
        if img_valid and label_valid:
            valid_count += 1
        else:
            error_info = {
                'file': str(img_file),
                'image_error': img_error if not img_valid else None,
                'label_error': label_error if not label_valid else None
            }
            invalid_files.append(error_info)
    
    print(f"✅ Valid: {valid_count}/{len(image_files)} files")
    if invalid_files:
        print(f"❌ Invalid: {len(invalid_files)} files")
    
    return len(image_files), valid_count, invalid_files

def fix_corrupted_files(invalid_files, backup_dir=None):
    """Fix or remove corrupted files"""
    
    if not invalid_files:
        return 0
    
    print(f"\n🔧 Fixing {len(invalid_files)} corrupted files...")
    
    if backup_dir:
        backup_dir = Path(backup_dir)
        backup_dir.mkdir(parents=True, exist_ok=True)
    
    fixed_count = 0
    
    for file_info in tqdm(invalid_files, desc="Fixing files"):
        img_path = Path(file_info['file'])
        label_path = img_path.parent.parent / "labels" / img_path.parent.name / (img_path.stem + '.txt')
        
        try:
            # Backup files if requested
            if backup_dir:
                backup_img = backup_dir / img_path.name
                backup_label = backup_dir / (img_path.stem + '.txt')
                
                if img_path.exists():
                    shutil.copy2(img_path, backup_img)
                if label_path.exists():
                    shutil.copy2(label_path, backup_label)
            
            # Try to fix image by re-saving it
            if file_info['image_error'] and img_path.exists():
                try:
                    with Image.open(img_path) as img:
                        # Convert to RGB to ensure compatibility
                        if img.mode != 'RGB':
                            img = img.convert('RGB')
                        # Re-save with high quality
                        img.save(img_path, 'JPEG', quality=95)
                        print(f"🔧 Fixed image: {img_path.name}")
                        fixed_count += 1
                except:
                    # If can't fix, remove both image and label
                    print(f"🗑️ Removing unfixable image: {img_path.name}")
                    if img_path.exists():
                        img_path.unlink()
                    if label_path.exists():
                        label_path.unlink()
            
            # Fix empty or missing labels
            if file_info['label_error'] and 'Missing label file' in file_info['label_error']:
                # Create empty label file
                label_path.touch()
                print(f"🔧 Created empty label: {label_path.name}")
                fixed_count += 1
                
        except Exception as e:
            print(f"⚠️ Error fixing {img_path.name}: {e}")
    
    print(f"✅ Fixed: {fixed_count} files")
    return fixed_count

def main():
    parser = argparse.ArgumentParser(description='Validate dataset integrity')
    parser.add_argument('--dataset_dir', required=True, help='Path to dataset directory')
    parser.add_argument('--fix', action='store_true', help='Attempt to fix corrupted files')
    parser.add_argument('--backup_dir', help='Backup directory for corrupted files')
    args = parser.parse_args()
    
    dataset_dir = Path(args.dataset_dir)
    
    print("🔍 Dataset Integrity Validation")
    print(f"Dataset: {dataset_dir}")
    
    # Check if dataset.yaml exists
    yaml_file = dataset_dir / "dataset.yaml"
    if not yaml_file.exists():
        print(f"❌ dataset.yaml not found: {yaml_file}")
        return 1
    
    # Load dataset config
    try:
        with open(yaml_file, 'r') as f:
            config = yaml.safe_load(f)
        print(f"✅ Dataset config loaded")
        print(f"   Classes: {config.get('nc', 'unknown')}")
        print(f"   Names: {config.get('names', 'unknown')}")
    except Exception as e:
        print(f"❌ Error loading dataset.yaml: {e}")
        return 1
    
    # Validate splits
    splits = ['train', 'val']
    total_files = 0
    total_valid = 0
    all_invalid_files = []
    
    for split in splits:
        images_dir = dataset_dir / "images" / split
        labels_dir = dataset_dir / "labels" / split
        
        files_count, valid_count, invalid_files = validate_dataset_split(
            images_dir, labels_dir, split
        )
        
        total_files += files_count
        total_valid += valid_count
        all_invalid_files.extend(invalid_files)
    
    # Summary
    print(f"\n📊 Validation Summary:")
    print(f"   Total files: {total_files}")
    print(f"   Valid files: {total_valid}")
    print(f"   Invalid files: {len(all_invalid_files)}")
    print(f"   Success rate: {total_valid/total_files*100:.2f}%" if total_files > 0 else "   Success rate: 0%")
    
    # Fix corrupted files if requested
    if args.fix and all_invalid_files:
        fixed_count = fix_corrupted_files(all_invalid_files, args.backup_dir)
        print(f"\n🔧 Fixed {fixed_count} files")
        
        # Re-validate after fixing
        print(f"\n🔍 Re-validating after fixes...")
        total_files_after = 0
        total_valid_after = 0
        
        for split in splits:
            images_dir = dataset_dir / "images" / split
            labels_dir = dataset_dir / "labels" / split
            
            files_count, valid_count, _ = validate_dataset_split(
                images_dir, labels_dir, split
            )
            
            total_files_after += files_count
            total_valid_after += valid_count
        
        print(f"\n📊 After Fixes:")
        print(f"   Total files: {total_files_after}")
        print(f"   Valid files: {total_valid_after}")
        print(f"   Success rate: {total_valid_after/total_files_after*100:.2f}%" if total_files_after > 0 else "   Success rate: 0%")
    
    # Show some invalid files for debugging
    if all_invalid_files:
        print(f"\n❌ Sample of invalid files:")
        for i, file_info in enumerate(all_invalid_files[:5]):  # Show first 5
            print(f"   {i+1}. {Path(file_info['file']).name}")
            if file_info['image_error']:
                print(f"      Image error: {file_info['image_error']}")
            if file_info['label_error']:
                print(f"      Label error: {file_info['label_error']}")
        
        if len(all_invalid_files) > 5:
            print(f"   ... and {len(all_invalid_files)-5} more")
    
    # Recommendations
    if len(all_invalid_files) > 0:
        error_rate = len(all_invalid_files) / total_files * 100
        print(f"\n💡 Recommendations:")
        
        if error_rate < 1:
            print(f"   ✅ Low error rate ({error_rate:.2f}%) - Safe to proceed with training")
            print(f"   💡 Use --fix flag to clean up corrupted files")
        elif error_rate < 5:
            print(f"   ⚠️  Moderate error rate ({error_rate:.2f}%) - Recommend fixing before training")
            print(f"   🔧 Run with --fix flag to attempt repairs")
        else:
            print(f"   ❌ High error rate ({error_rate:.2f}%) - Must fix before training")
            print(f"   🔧 Run with --fix --backup_dir ./corrupted_backup")
            return 1
    else:
        print(f"\n✅ Dataset validation passed! Ready for training.")
    
    return 0

if __name__ == "__main__":
    exit(main())