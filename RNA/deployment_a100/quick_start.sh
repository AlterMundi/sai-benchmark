#!/bin/bash
# SAI A100 Quick Start Script
# Generado: 2025-08-22T17:02:10.777929

echo "🚀 SAI A100 Training Quick Start"
echo "================================"

# Verificar entorno
echo "🔍 Verificando entorno..."
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "RAM disponible: $(free -h | grep Mem | awk '{print $7}')"
echo "Espacio en disco: $(df -h /data | tail -1 | awk '{print $4}')"

# Opciones de entrenamiento
echo ""
echo "Opciones disponibles:"
echo "1) Entrenar detector YOLOv8"
echo "2) Entrenar verificator CNN"
echo "3) Iniciar solo monitoreo"
echo "4) Ver configuraciones"

read -p "Selecciona opción [1-4]: " option

case $option in
    1)
        echo "🔥 Iniciando entrenamiento detector..."
        python3 train_yolo_detector_a100.py
        ;;
    2)
        echo "🔥 Iniciando entrenamiento verificator..."
        python3 train_verificator_a100.py
        ;;
    3)
        echo "🔍 Iniciando monitoreo..."
        python3 start_monitoring.py --training-dir /data/sai-benchmark/RNA/training/runs --interval 15
        ;;
    4)
        echo "📋 Configuraciones disponibles:"
        ls -la *.yaml
        ;;
    *)
        echo "❌ Opción inválida"
        exit 1
        ;;
esac
