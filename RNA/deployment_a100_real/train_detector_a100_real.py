#!/usr/bin/env python3
"""
SAI Detector YOLOv8 - Optimizado para A100-SXM4-40GB REAL
Hardware: A100 40GB VRAM, 243GB RAM, 128 CPU cores
Generado: 2025-08-22T17:08:32.062305
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
    "model_type": "yolo",
    "model_name": "yolov8s.pt",
    "task": "detect",
    "data": "/data/sai-benchmark/RNA/data/mega_fire_dataset/dataset.yaml",
    "imgsz": 1440,
    "epochs": 100,
    "batch": 32,
    "patience": 20,
    "save_period": 5,
    "val": true,
    "plots": true,
    "save_json": true,
    "device": "auto",
    "amp": true,
    "cache": true,
    "compile": true,
    "workers": 32,
    "pin_memory": true,
    "persistent_workers": true,
    "prefetch_factor": 8,
    "optimizer": "AdamW",
    "lr0": 0.001,
    "lrf": 0.01,
    "momentum": 0.937,
    "weight_decay": 0.0005,
    "warmup_epochs": 3,
    "warmup_momentum": 0.8,
    "warmup_bias_lr": 0.1,
    "cos_lr": true,
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "degrees": 0.0,
    "translate": 0.1,
    "scale": 0.5,
    "shear": 0.0,
    "perspective": 0.0,
    "flipud": 0.0,
    "fliplr": 0.5,
    "mosaic": 1.0,
    "mixup": 0.0,
    "copy_paste": 0.0,
    "project": "/data/sai-benchmark/RNA/training/runs",
    "name": "sai_detector_a100_real",
    "exist_ok": true,
    "verbose": true,
    "seed": 42,
    "deterministic": true,
    "rect": false,
    "overlap_mask": true,
    "mask_ratio": 4,
    "dropout": 0.0,
    "val_split": 0.0,
    "half": false,
    "dnn": false,
    "auto_augment": "randaugment",
    "erasing": 0.4,
    "crop_fraction": 1.0
}

def setup_a100_environment():
    """Configurar entorno específico para A100"""
    
    print("🔥 Configurando entorno para A100-SXM4-40GB")
    
    # Configuraciones específicas A100
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.set_float32_matmul_precision('medium')  # Tensor Cores
    
    # Configuraciones de memoria optimizadas para 40GB VRAM
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:1024,expandable_segments:True'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
    os.environ['CUDA_CACHE_DISABLE'] = '0'
    
    # Threading para 128 cores
    torch.set_num_threads(64)  # Usar mitad para balance
    os.environ['OMP_NUM_THREADS'] = '64'
    os.environ['MKL_NUM_THREADS'] = '64'
    
    # A100 específico
    os.environ['NVIDIA_TF32_OVERRIDE'] = '1'  # Habilitar TF32
    
    print(f"✅ Configurado para {torch.cuda.get_device_name(0)}")
    print(f"✅ VRAM disponible: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")

def validate_a100_hardware():
    """Validar que estamos en A100 correcto"""
    
    if not torch.cuda.is_available():
        print("❌ CUDA no disponible")
        return False
    
    gpu_name = torch.cuda.get_device_name(0)
    if 'A100' not in gpu_name:
        print(f"⚠️ GPU detectado: {gpu_name}")
        print("⚠️ Esperado: A100-SXM4-40GB")
        response = input("¿Continuar de todas formas? (y/N): ")
        if response.lower() != 'y':
            return False
    
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    expected_vram = HARDWARE_CONFIG['gpu_memory_gb']
    
    if abs(vram_gb - expected_vram) > 5:  # 5GB tolerance
        print(f"⚠️ VRAM detectado: {vram_gb:.1f}GB")
        print(f"⚠️ Esperado: {expected_vram:.1f}GB")
        response = input("¿Continuar de todas formas? (y/N): ")
        if response.lower() != 'y':
            return False
    
    print(f"✅ Hardware validado: {gpu_name} ({vram_gb:.1f}GB)")
    return True

def start_training():
    """Iniciar entrenamiento YOLOv8"""
    
    try:
        from ultralytics import YOLO
        
        print("🚀 INICIANDO ENTRENAMIENTO YOLO A100")
        print("=" * 50)
        print(f"⏰ Inicio: {datetime.now()}")
        print(f"🎯 Configuración A100 optimizada")
        print(f"📊 Batch size: {TRAINING_CONFIG['batch']}")
        print(f"🔢 Workers: {TRAINING_CONFIG['workers']}")
        print(f"💾 Cache: {TRAINING_CONFIG['cache']}")
        print()
        
        # Crear modelo
        model = YOLO(TRAINING_CONFIG['model_name'])
        
        # Entrenar con configuración A100
        results = model.train(**TRAINING_CONFIG)
        
        print("\n🎉 ENTRENAMIENTO A100 COMPLETADO")
        print(f"⏰ Finalizado: {datetime.now()}")
        
        return results
        
    except Exception as e:
        print(f"❌ Error en entrenamiento: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Función principal"""
    
    # Configurar entorno A100
    setup_a100_environment()
    
    # Validar hardware
    if not validate_a100_hardware():
        print("❌ Validación de hardware fallida")
        sys.exit(1)
    
    # Iniciar entrenamiento
    results = start_training()
    
    if results:
        print("✅ Entrenamiento exitoso")
        return 0
    else:
        print("❌ Entrenamiento fallido")
        return 1

if __name__ == "__main__":
    sys.exit(main())
