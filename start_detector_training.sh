#!/bin/bash
# SAI YOLOv8-s Detector Training Launcher
# Optimized for 15-20 hour autonomous training on RTX 3090

set -e

echo "🔥 SAI Detector Training Launcher"
echo "=================================="
echo "Estimated training time: 15-20 hours"
echo "Target resolution: 1440×808"
echo "Dataset: 173K images from 5 sources"
echo ""

# Check if training is already running
if pgrep -f "detector_trainer.py" > /dev/null; then
    echo "❌ Training already running!"
    echo "Current training process:"
    ps aux | grep detector_trainer.py | grep -v grep
    exit 1
fi

# Create logs directory
mkdir -p RNA/training/logs

# Setup virtual environment
if [ ! -d "RNA/training/venv" ]; then
    echo "🔧 Creating virtual environment..."
    python3 -m venv RNA/training/venv
fi

# Activate virtual environment
source RNA/training/venv/bin/activate

# Check dependencies
echo "🔍 Checking dependencies..."
python -c "import torch; print(f'PyTorch: {torch.__version__}')" 2>/dev/null || {
    echo "❌ PyTorch not installed. Installing..."
    pip install torch torchvision torchaudio
}
python -c "import ultralytics; print(f'Ultralytics: {ultralytics.__version__}')" 2>/dev/null || {
    echo "❌ Ultralytics not installed. Installing..."
    pip install ultralytics
}

# Check GPU
echo "🖥️  GPU Status:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader || echo "⚠️  nvidia-smi not available"

echo ""
echo "🚀 Starting autonomous training process..."
echo "Training will run for ~15-20 hours"
echo "Logs: RNA/training/logs/detector_training.log"
echo "Monitor with: tail -f RNA/training/logs/detector_training.log"
echo ""

# Start training in background with comprehensive logging
nohup python RNA/training/detector_trainer.py \
    --config RNA/configs/sai_cascade_config.yaml \
    --prepare-data \
    --auto-resume \
    > RNA/training/logs/detector_training.log 2>&1 &

TRAINING_PID=$!
echo "✅ Training started successfully!"
echo "Process ID: $TRAINING_PID"
echo "Log file: RNA/training/logs/detector_training.log"
echo ""
echo "Commands for monitoring:"
echo "  Check progress: tail -f RNA/training/logs/detector_training.log"
echo "  Check GPU usage: watch nvidia-smi"
echo "  Stop training: kill $TRAINING_PID"
echo ""
echo "The training will run autonomously for 15-20 hours."
echo "Early stopping is enabled - it may finish sooner if convergence is reached."
echo ""
echo "🏁 Training launched! Check logs for progress."