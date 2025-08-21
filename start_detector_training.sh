#!/bin/bash
# SAI YOLOv8-s Detector Training Launcher
# Optimized for 15-20 hour autonomous training on RTX 3090

set -e

echo "🔥 SAI Detector Training Launcher"
echo "=================================="
echo "Estimated training time: 3-5 hours"
echo "Target resolution: 1440×808"
echo "Dataset: D-Fire (21K images) - YOLO format ready"
echo ""

# Check if training is already running
if pgrep -f "yolo.*train" > /dev/null; then
    echo "❌ Training already running!"
    echo "Current training process:"
    ps aux | grep "yolo.*train" | grep -v grep
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
echo "Training will run for ~3-5 hours"
echo "Logs: RNA/training/logs/detector_training.log"
echo "Monitor with: tail -f RNA/training/logs/detector_training.log"
echo ""

# Start training in background with comprehensive logging using D-Fire dataset
nohup yolo detect train \
    data=RNA/data/processed/dfire_dataset.yaml \
    model=yolov8s.pt \
    epochs=100 \
    imgsz=1440 \
    batch=8 \
    patience=50 \
    save_period=10 \
    cache=disk \
    amp=true \
    optimizer=AdamW \
    lr0=0.001 \
    weight_decay=0.0005 \
    warmup_epochs=3 \
    box=7.5 \
    cls=0.5 \
    dfl=1.5 \
    project=RNA/training/runs \
    name=sai_detector_training \
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
echo "The training will run autonomously for 3-5 hours."
echo "Early stopping is enabled - it may finish sooner if convergence is reached."
echo ""
echo "🏁 Training launched! Check logs for progress."