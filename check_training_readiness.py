#!/usr/bin/env python3
"""
Quick verification script to check training readiness
"""

import sys
import os
from pathlib import Path

def check_dependencies():
    """Check if all required dependencies are available"""
    print("🔍 Checking dependencies...")
    
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
        
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"✅ GPU: {gpu_name} ({gpu_memory:.1f}GB)")
        else:
            print("⚠️  CUDA not available - training will be slow")
            
    except ImportError:
        print("❌ PyTorch not installed")
        return False
    
    try:
        import ultralytics
        print(f"✅ Ultralytics: {ultralytics.__version__}")
    except ImportError:
        print("❌ Ultralytics not installed")
        return False
        
    return True

def check_config():
    """Check if configuration files exist"""
    print("\n📋 Checking configuration...")
    
    config_path = Path("RNA/configs/sai_cascade_config.yaml")
    if config_path.exists():
        print(f"✅ Config found: {config_path}")
        return True
    else:
        print(f"❌ Config not found: {config_path}")
        return False

def check_datasets():
    """Check dataset availability"""
    print("\n📊 Checking datasets...")
    
    data_root = Path("RNA/data/raw")
    datasets = ["FASDD", "PyroNear-2024", "D-Fire", "FIgLib", "NEMO"]
    found_datasets = 0
    
    for dataset in datasets:
        dataset_path = data_root / dataset
        if dataset_path.exists():
            print(f"✅ {dataset}: Found")
            found_datasets += 1
        else:
            print(f"⚠️  {dataset}: Not found at {dataset_path}")
    
    print(f"\n📈 Total datasets available: {found_datasets}/5")
    return found_datasets >= 3  # Need at least 3 datasets

def check_disk_space():
    """Check available disk space"""
    print("\n💾 Checking disk space...")
    
    import shutil
    total, used, free = shutil.disk_usage("/")
    
    free_gb = free / (1024**3)
    print(f"📊 Available disk space: {free_gb:.1f}GB")
    
    # Need ~2GB for training outputs
    if free_gb > 2:
        print("✅ Sufficient disk space")
        return True
    else:
        print("❌ Insufficient disk space (need ~2GB)")
        return False

def estimate_training():
    """Provide training estimates"""
    print("\n⏱️  Training Estimates:")
    print("   📊 Dataset: ~173K images")
    print("   🎯 Resolution: 1440×808")
    print("   ⏰ Estimated time: 15-20 hours")
    print("   💾 Model size: ~25-30 MB")
    print("   📈 Checkpoints: ~200 MB")
    print("   🔄 Auto early stopping: enabled")

def main():
    print("🔥 SAI Training Readiness Check")
    print("=" * 40)
    
    checks = [
        check_dependencies(),
        check_config(),
        check_datasets(),
        check_disk_space()
    ]
    
    print("\n" + "=" * 40)
    if all(checks):
        print("🎉 ALL CHECKS PASSED - READY FOR TRAINING!")
        print("\nTo start training:")
        print("   ./start_detector_training.sh")
        estimate_training()
        return 0
    else:
        print("❌ SOME CHECKS FAILED")
        print("Please resolve issues before training")
        return 1

if __name__ == "__main__":
    sys.exit(main())