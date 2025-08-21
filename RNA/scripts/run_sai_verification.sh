#!/bin/bash
# SAI Mega Dataset Verification Runner
# Executes comprehensive integrity verification after dataset recreation

echo "🔥 SAI MEGA DATASET VERIFICATION"
echo "================================"

# Configuration
DATASET_PATH="../data/mega_fire_dataset"
VERIFIER_SCRIPT="./sai_mega_integrity_verifier.py"
REPORT_PATH="../data/mega_fire_dataset/sai_integrity_report.txt"

# Check if dataset exists
if [ ! -d "$DATASET_PATH" ]; then
    echo "❌ ERROR: Dataset not found at $DATASET_PATH"
    exit 1
fi

# Check if verifier exists
if [ ! -f "$VERIFIER_SCRIPT" ]; then
    echo "❌ ERROR: Verifier script not found at $VERIFIER_SCRIPT"
    exit 1
fi

echo "📁 Dataset path: $DATASET_PATH"
echo "🔍 Verifier: $VERIFIER_SCRIPT"
echo "📋 Report output: $REPORT_PATH"
echo ""

# Run comprehensive verification
echo "🚀 Starting comprehensive SAI integrity verification..."
echo "⏱️ This may take several minutes for large datasets..."
echo ""

python3 "$VERIFIER_SCRIPT" \
    --dataset_path "$DATASET_PATH" \
    --output "$REPORT_PATH" \
    --verbose

# Check result
if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 SAI VERIFICATION COMPLETED SUCCESSFULLY!"
    echo "✅ Dataset is ready for fire detection training"
    echo "📋 Full report available at: $REPORT_PATH"
else
    echo ""
    echo "⚠️ SAI VERIFICATION COMPLETED WITH ISSUES"
    echo "🔍 Please review the report for details: $REPORT_PATH"
fi

echo ""
echo "📊 Next step: Start YOLOv8-s detector training"
echo "🔥 Mission: Critical fire detection system deployment"