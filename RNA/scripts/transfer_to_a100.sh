#!/bin/bash
# Script para transferir y ejecutar completado del dataset en servidor A100

echo "🚀 TRANSFERENCIA Y EJECUCIÓN EN SERVIDOR A100"
echo "=============================================="

# Configuración del servidor A100
A100_HOST="88.207.86.56"
A100_PORT="31939"
A100_USER="root"
A100_KEY="~/.ssh/sai-n8n-deploy"
A100_PATH="/mnt/n8n-data/sai-benchmark"
SSH_CMD="ssh -i $A100_KEY -p $A100_PORT"
SCP_CMD="scp -i $A100_KEY -P $A100_PORT"
RSYNC_CMD="rsync -avP -e 'ssh -i $A100_KEY -p $A100_PORT'"

echo "📋 Configuración:"
echo "   Host: $A100_HOST"
echo "   Path: $A100_PATH"
echo "   Local progress: $(ls RNA/data/verificator_dataset/images/train/false_positive/ 2>/dev/null | wc -l) false positives ya generados"
echo

# Función para transferir script
transfer_script() {
    echo "📤 Transfiriendo script de completado..."
    $SCP_CMD RNA/scripts/complete_verificator_dataset.py $A100_USER@$A100_HOST:$A100_PATH/RNA/scripts/
    
    if [ $? -eq 0 ]; then
        echo "✅ Script transferido exitosamente"
        return 0
    else
        echo "❌ Error transfiriendo script"
        return 1
    fi
}

# Función para ejecutar en A100
execute_on_a100() {
    echo "🔥 Ejecutando completado del dataset en A100..."
    echo "⏱️  Tiempo estimado: 35-45 minutos (vs 4-5 horas local)"
    
    $SSH_CMD $A100_USER@$A100_HOST "cd $A100_PATH && \
        source RNA/verification_env/bin/activate && \
        python3 RNA/scripts/complete_verificator_dataset.py \
            --detector-path RNA/models/detector_best.pt \
            --mega-dataset RNA/data/mega_fire_dataset \
            --verificator-dataset RNA/data/verificator_dataset \
            --balance-ratio 1.0 \
            --fp-conf-min 0.3 \
            --fp-conf-max 0.8"
}

# Función para descargar dataset completo
download_completed() {
    echo "📥 Descargando dataset completado..."
    eval "$RSYNC_CMD $A100_USER@$A100_HOST:$A100_PATH/RNA/data/verificator_dataset/ RNA/data/verificator_dataset_completed/"
    
    if [ $? -eq 0 ]; then
        echo "✅ Dataset descargado exitosamente"
        echo "📊 Verificando integridad..."
        echo "   Falsos positivos: $(ls RNA/data/verificator_dataset_completed/images/train/false_positive/ 2>/dev/null | wc -l)"
        echo "   Split validación: $(ls RNA/data/verificator_dataset_completed/images/val/ 2>/dev/null | wc -l)"
    else
        echo "❌ Error descargando dataset"
        return 1
    fi
}

# Ejecutar workflow completo
main() {
    if transfer_script; then
        echo "🎯 Ejecutando en A100 (8x más rápido)..."
        if execute_on_a100; then
            echo "✅ Completado exitosamente en A100"
            download_completed
        else
            echo "❌ Error ejecutando en A100"
            exit 1
        fi
    else
        echo "❌ Error en transferencia inicial"
        exit 1
    fi
}

# Verificar si tenemos SSH configurado
if timeout 5 $SSH_CMD $A100_USER@$A100_HOST 'echo "test"' >/dev/null 2>&1; then
    echo "🔑 SSH configurado correctamente"
    main
else
    echo "❌ SSH no configurado. Configurar primero la conexión al servidor A100"
    exit 1
fi