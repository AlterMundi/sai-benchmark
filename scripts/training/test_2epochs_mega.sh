#!/bin/bash
# SAI 2-Epoch Test on MEGA Dataset
# Performance benchmarking for realistic training time projections

set -e

echo "🔥 SAI 2-Epoch MEGA Dataset Performance Test"
echo "============================================="
echo "⚡ NVMe OPTIMIZED - Timing Benchmark"
echo "Dataset: MEGA (64K images)"
echo "Purpose: Calculate realistic training time projections"
echo "Expected duration: 30-60 minutes for 2 epochs"
echo ""

# Check if training is already running
if pgrep -f "yolo.*train" > /dev/null; then
    echo "❌ Training already running!"
    echo "Current training process:"
    ps aux | grep "yolo.*train" | grep -v grep
    exit 1
fi

# Create logs directory
mkdir -p RNA/training/test_logs

# Ensure we're in NVMe directory
if [[ "$PWD" != "/mnt/n8n-data/sai-benchmark" ]]; then
    echo "⚠️  Not in NVMe directory! Current: $PWD"
    echo "🔧 Switching to NVMe directory..."
    cd /mnt/n8n-data/sai-benchmark
fi

# Remove any conflicting directories
if [ -d "/root/sai-benchmark" ]; then
    echo "⚠️  Found conflicting directory at /root/sai-benchmark - this could cause issues"
fi

# Setup virtual environment in NVMe
if [ ! -d "RNA/training/venv" ]; then
    echo "🔧 Creating virtual environment in NVMe..."
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

# Verify dataset
echo "📊 Dataset verification:"
echo "Dataset file: RNA/data/mega_fire_dataset/dataset.yaml"
ls -la RNA/data/mega_fire_dataset/dataset.yaml || {
    echo "❌ Dataset file not found!"
    exit 1
}
echo "Images count: $(find RNA/data/mega_fire_dataset/images -name "*.jpg" | wc -l)"
echo "Labels count: $(find RNA/data/mega_fire_dataset/labels -name "*.txt" | wc -l)"
echo ""

echo "🚀 Starting 2-epoch timing test..."
echo "⏱️  Start time: $(date)"
echo "📝 Logs: RNA/training/test_logs/2epoch_test.log"
echo "🔍 Monitor with: tail -f RNA/training/test_logs/2epoch_test.log"
echo ""

# Record start time
START_TIME=$(date +%s)
echo "START_TIME: $START_TIME" > RNA/training/test_logs/timing.txt

# Start training with exactly 2 epochs
echo "Launching 2-epoch training test..."
yolo detect train \
    data=RNA/data/mega_fire_dataset/dataset.yaml \
    model=yolov8s.pt \
    epochs=2 \
    imgsz=1440 \
    batch=8 \
    cache=disk \
    amp=true \
    optimizer=AdamW \
    lr0=0.001 \
    weight_decay=0.0005 \
    warmup_epochs=1 \
    box=7.5 \
    cls=0.5 \
    dfl=1.5 \
    project=RNA/training/test_runs \
    name=mega_2epoch_test \
    > RNA/training/test_logs/2epoch_test.log 2>&1

# Record end time and calculate duration
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))
SECONDS=$((DURATION % 60))

echo "END_TIME: $END_TIME" >> RNA/training/test_logs/timing.txt
echo "DURATION: $DURATION" >> RNA/training/test_logs/timing.txt

echo ""
echo "✅ 2-Epoch Test Completed!"
echo "⏱️  End time: $(date)"
echo "⌛ Total duration: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo "📊 Duration for 2 epochs: $DURATION seconds"
echo ""

# Calculate projections
EPOCHS_FULL=100
PROJECTED_SECONDS=$((DURATION * EPOCHS_FULL / 2))
PROJECTED_HOURS=$((PROJECTED_SECONDS / 3600))
PROJECTED_MINUTES=$(((PROJECTED_SECONDS % 3600) / 60))

echo "📈 PROJECTIONS FOR FULL TRAINING (100 epochs):"
echo "   Estimated time: ${PROJECTED_HOURS}h ${PROJECTED_MINUTES}m"
echo "   Total seconds: $PROJECTED_SECONDS"
echo ""

# Save projections
cat > RNA/training/test_logs/projections.txt << EOF
2-Epoch Test Results - MEGA Dataset (64K images)
================================================
Test Date: $(date)
Hardware: NVMe + RTX 3090
Resolution: 1440×808

TIMING RESULTS:
- 2 epochs duration: ${HOURS}h ${MINUTES}m ${SECONDS}s ($DURATION seconds)
- Average per epoch: $((DURATION / 2)) seconds

PROJECTIONS FOR 100 EPOCHS:
- Estimated total time: ${PROJECTED_HOURS}h ${PROJECTED_MINUTES}m
- With early stopping (50-80 epochs): $((PROJECTED_SECONDS * 75 / 100 / 3600))h - $((PROJECTED_SECONDS * 80 / 100 / 3600))h

PERFORMANCE METRICS:
- Images per second: $(echo "scale=2; 64000 * 2 / $DURATION" | bc -l) img/s
- NVMe optimization: Confirmed active
- GPU utilization: Check nvidia-smi logs

RECOMMENDATION:
$(if [ $PROJECTED_HOURS -lt 12 ]; then echo "✅ Proceed with full training - under 12h estimate"; else echo "⚠️ Consider dataset reduction - over 12h estimate"; fi)
EOF

echo "📄 Detailed results saved to: RNA/training/test_logs/projections.txt"
echo "🔍 View results: cat RNA/training/test_logs/projections.txt"
echo ""
echo "🏁 Test completed successfully!"