#!/usr/bin/env python3
"""
SAI Verificator CNN - Optimizado para A100-SXM4-40GB REAL
Hardware: A100 40GB VRAM, 243GB RAM, 128 CPU cores
Generado: 2025-08-22T17:08:32.062409
"""

import os
import sys
import torch
from pathlib import Path
from datetime import datetime

# Configuración REAL A100
HARDWARE_CONFIG = {
    "gpu_name": "NVIDIA A100-SXM4-40GB",
    "gpu_memory_gb": 40.96,
    "ram_total_gb": 251,
    "ram_available_gb": 243,
    "cpu_cores": 128,
    "disk_total_gb": 300,
    "disk_available_gb": 285,
    "cuda_driver": "550.107.02"
}

TRAINING_CONFIG = {
    "model_type": "verificator",
    "backbone": "efficientnet_b0",
    "num_classes": 2,
    "temporal_frames": 1,
    "dropout": 0.3,
    "dataset_path": "/data/sai-benchmark/RNA/data/verificator_dataset",
    "epochs": 50,
    "batch_size": 256,
    "learning_rate": 0.0003,
    "weight_decay": 1e-05,
    "device": "auto",
    "num_workers": 64,
    "pin_memory": true,
    "persistent_workers": true,
    "prefetch_factor": 8,
    "optimizer": "adamw",
    "scheduler": "reduce_on_plateau",
    "scheduler_patience": 5,
    "patience": 15,
    "grad_clip": 1.0,
    "use_class_weights": true,
    "output_dir": "/data/sai-benchmark/RNA/training/runs/verificator_a100_real",
    "save_every": 5,
    "seed": 42,
    "deterministic": true,
    "mixed_precision": true,
    "compile_model": true,
    "tensor_cores": true,
    "channels_last": true,
    "torch_compile": true,
    "cudnn_benchmark": true
}

def setup_a100_environment():
    """Configurar entorno específico para A100"""
    
    print("🔥 Configurando entorno CNN para A100-SXM4-40GB")
    
    # Configuraciones específicas A100
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.set_float32_matmul_precision('medium')
    
    # Configuraciones optimizadas para CNN en A100
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:2048,expandable_segments:True'
    
    # Threading optimizado para 128 cores
    torch.set_num_threads(64)
    os.environ['OMP_NUM_THREADS'] = '64'
    
    print(f"✅ Configurado para CNN en {torch.cuda.get_device_name(0)}")

def start_verificator_training():
    """Iniciar entrenamiento del verificator"""
    
    try:
        # Importar trainer del verificator
        sys.path.append('/data/sai-benchmark')
        from RNA.scripts.train_verificator import VerificatorTrainer
        
        print("🚀 INICIANDO ENTRENAMIENTO VERIFICATOR A100")
        print("=" * 50)
        print(f"⏰ Inicio: {datetime.now()}")
        print(f"🎯 Configuración A100 optimizada")
        print(f"📊 Batch size: {TRAINING_CONFIG['batch_size']}")
        print(f"🔢 Workers: {TRAINING_CONFIG['num_workers']}")
        print(f"🧠 Backbone: {TRAINING_CONFIG['backbone']}")
        print()
        
        # Crear trainer con configuración A100
        trainer = VerificatorTrainer(TRAINING_CONFIG)
        
        # Entrenar
        best_f1, final_metrics = trainer.train()
        
        print("\n🎉 ENTRENAMIENTO VERIFICATOR A100 COMPLETADO")
        print(f"⏰ Finalizado: {datetime.now()}")
        print(f"🏆 Mejor F1: {best_f1:.3f}")
        
        return best_f1, final_metrics
        
    except Exception as e:
        print(f"❌ Error en entrenamiento verificator: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def main():
    """Función principal"""
    
    # Configurar entorno A100
    setup_a100_environment()
    
    # Verificar hardware
    if not torch.cuda.is_available():
        print("❌ CUDA no disponible")
        sys.exit(1)
    
    # Iniciar entrenamiento
    best_f1, metrics = start_verificator_training()
    
    if best_f1:
        print(f"✅ Entrenamiento exitoso - F1: {best_f1:.3f}")
        return 0
    else:
        print("❌ Entrenamiento fallido")
        return 1

if __name__ == "__main__":
    sys.exit(main())
