#!/bin/bash

# SAI Test Training Script
# Quick 15-minute validation before full training
# Estimated time: 10-15 minutes

set -e

echo "🔥 SAI Test Training - Quick Validation"
echo "========================================"
echo "⏱️  ESTIMATED TIME: 10-15 minutes"
echo "🎯 PURPOSE: Validate training pipeline before full run"
echo "📊 DATASET: 10% subset (~6,400 images)"
echo "🏃 EPOCHS: 10 (vs 100 for full training)"
echo "========================================"

# Check if we're in the right directory
if [ ! -f "RNA/configs/sai_test_config.yaml" ]; then
    echo "❌ Error: Must run from sai-benchmark root directory"
    echo "   Current: $(pwd)"
    echo "   Expected: /mnt/raid1/sai-benchmark/"
    exit 1
fi

# Check GPU availability
echo "🔍 Checking GPU availability..."
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ nvidia-smi not found. GPU training may not work."
    exit 1
fi

gpu_info=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits)
echo "✅ GPU detected: $gpu_info"

# Check mega dataset
echo "🔍 Checking mega dataset..."
if [ ! -d "RNA/data/mega_fire_dataset" ]; then
    echo "❌ Error: Mega dataset not found at RNA/data/mega_fire_dataset"
    echo "   Run dataset creation first"
    exit 1
fi

train_images=$(find RNA/data/mega_fire_dataset/images/train -name "*.jpg" -o -name "*.png" 2>/dev/null | wc -l)
val_images=$(find RNA/data/mega_fire_dataset/images/val -name "*.jpg" -o -name "*.png" 2>/dev/null | wc -l)
echo "✅ Dataset found: $train_images train, $val_images val images"

# Create test output directory
mkdir -p RNA/training/test_runs
echo "📁 Created test output directory"

# Activate virtual environment
cd RNA/training
echo "🔄 Activating virtual environment..."

if [ ! -d "venv" ]; then
    echo "❌ Error: Virtual environment not found"
    echo "   Run setup_environment.py first"
    exit 1
fi

source venv/bin/activate
echo "✅ Virtual environment activated"

# Check required packages
echo "🔍 Checking required packages..."
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import ultralytics; print(f'Ultralytics: {ultralytics.__version__}')"

# Start test training with time estimation
echo ""
echo "🚀 STARTING TEST TRAINING"
echo "=========================="
echo "🕒 Start time: $(date)"
echo "⏱️  Estimated completion: $(date -d '+15 minutes')"
echo "📊 Monitor progress: tail -f RNA/training/test_runs/train/*/logs/train.log"
echo ""

# Run training with test config
python -c "
import sys
import time
from pathlib import Path
from ultralytics import YOLO
import yaml

print('🔥 Initializing YOLOv8-s for test training...')

# Load test configuration
config_path = '../configs/sai_test_config.yaml'
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

print(f'📊 Test configuration loaded')
print(f'   - Epochs: {config[\"training\"][\"detector\"][\"epochs\"]}')
print(f'   - Batch size: {config[\"training\"][\"detector\"][\"batch_size\"]}')
print(f'   - Image size: {config[\"training\"][\"detector\"][\"image_size\"]}')
print(f'   - Learning rate: {config[\"training\"][\"detector\"][\"learning_rate\"]}')

# Initialize model
model = YOLO('yolov8s.pt')  # Start with pretrained weights
print('✅ Model initialized with YOLOv8-s pretrained weights')

# Training parameters
train_params = {
    'data': '../data/mega_fire_dataset/dataset.yaml',
    'epochs': config['training']['detector']['epochs'],
    'batch': config['training']['detector']['batch_size'],
    'imgsz': config['training']['detector']['image_size'][0],
    'lr0': config['training']['detector']['learning_rate'],
    'weight_decay': config['training']['detector']['weight_decay'],
    'warmup_epochs': config['training']['detector']['warmup_epochs'],
    'patience': config['training']['detector']['patience'],
    'save_period': config['training']['detector']['save_period'],
    'cache': config['training']['detector']['cache'],
    'amp': config['training']['detector']['mixed_precision'],
    'workers': config['training']['detector']['workers'],
    'optimizer': config['training']['detector']['optimizer'],
    'box': config['training']['detector']['box_loss_gain'],
    'cls': config['training']['detector']['cls_loss_gain'],
    'dfl': config['training']['detector']['dfl_loss_gain'],
    'project': 'test_runs',
    'name': 'test_detector',
    'exist_ok': True,
    'verbose': True,
    'fraction': 0.1  # Use only 10% of dataset for speed
}

print(f'🚀 Starting test training with {train_params[\"fraction\"]*100:.0f}% of dataset...')
print(f'⏱️  Expected duration: ~{config[\"training\"][\"detector\"][\"epochs\"]} epochs × ~1 min/epoch = {config[\"training\"][\"detector\"][\"epochs\"]} minutes')

start_time = time.time()

# Start training
try:
    results = model.train(**train_params)
    
    training_time = time.time() - start_time
    print(f'')
    print(f'🎉 TEST TRAINING COMPLETED SUCCESSFULLY!')
    print(f'⏱️  Actual training time: {training_time/60:.1f} minutes')
    print(f'📊 Results saved to: test_runs/test_detector/')
    print(f'')
    print(f'📈 Quick metrics:')
    if hasattr(results, 'results_dict'):
        for key, value in results.results_dict.items():
            if 'map' in key.lower() or 'precision' in key.lower() or 'recall' in key.lower():
                print(f'   {key}: {value:.3f}')
    
    # Save test results
    with open('test_runs/test_results.txt', 'w') as f:
        f.write(f'SAI Test Training Results\\n')
        f.write(f'Training time: {training_time/60:.1f} minutes\\n')
        f.write(f'Epochs completed: {config[\"training\"][\"detector\"][\"epochs\"]}\\n')
        f.write(f'Dataset fraction: {train_params[\"fraction\"]*100:.0f}%\\n')
        if hasattr(results, 'results_dict'):
            for key, value in results.results_dict.items():
                f.write(f'{key}: {value}\\n')
    
    print(f'✅ Test results saved to: test_runs/test_results.txt')
    print(f'')
    print(f'🚀 SYSTEM VALIDATION: SUCCESS!')
    print(f'   ✅ GPU training works')
    print(f'   ✅ Dataset loading works') 
    print(f'   ✅ Model training works')
    print(f'   ✅ Ready for full training!')
    
except Exception as e:
    print(f'')
    print(f'❌ TEST TRAINING FAILED!')
    print(f'Error: {e}')
    print(f'')
    print(f'🔍 Troubleshooting:')
    print(f'   1. Check GPU memory usage')
    print(f'   2. Verify dataset integrity')
    print(f'   3. Check virtual environment')
    print(f'   4. Review error logs above')
    sys.exit(1)
"

echo ""
echo "🏁 Test training script completed"
echo "🕒 End time: $(date)"

# Deactivate virtual environment
deactivate

echo ""
echo "📋 Next steps if test successful:"
echo "   1. Review test results in RNA/training/test_runs/"
echo "   2. If satisfied, run full training: ./start_detector_training.sh"
echo "   3. Full training estimated time: 15-20 hours"