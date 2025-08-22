#!/bin/bash
# Script para transferir configuración A100 REAL al servidor
# Generado: 2025-08-22T17:08:32.064169

echo "🚀 Transfiriendo configuración A100 REAL al servidor..."

# Transferir scripts de entrenamiento
scp -P 31939 -i ~/.ssh/sai-n8n-deploy \
    /mnt/n8n-data/sai-benchmark/RNA/deployment_a100_real/train_detector_a100_real.py \
    /mnt/n8n-data/sai-benchmark/RNA/deployment_a100_real/train_verificator_a100_real.py \
    /mnt/n8n-data/sai-benchmark/RNA/deployment_a100_real/start_a100_monitoring.py \
    root@88.207.86.56:/data/sai-benchmark/RNA/scripts/

# Transferir configuraciones
scp -P 31939 -i ~/.ssh/sai-n8n-deploy \
    /mnt/n8n-data/sai-benchmark/RNA/deployment_a100_real/*.yaml \
    root@88.207.86.56:/data/sai-benchmark/RNA/configs/

# Transferir monitor robusto
scp -P 31939 -i ~/.ssh/sai-n8n-deploy \
    /mnt/n8n-data/sai-benchmark/RNA/scripts/robust_training_monitor.py \
    /mnt/n8n-data/sai-benchmark/RNA/deployment_a100/validation_suite.py \
    root@88.207.86.56:/data/sai-benchmark/RNA/scripts/

echo "✅ Transferencia completada"
echo "🎯 Para usar en A100:"
echo "   cd /data/sai-benchmark"
echo "   python3 RNA/scripts/train_detector_a100_real.py"
echo "   # o"
echo "   python3 RNA/scripts/train_verificator_a100_real.py"
