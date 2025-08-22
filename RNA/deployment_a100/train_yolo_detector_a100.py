#!/usr/bin/env python3
"""
Script de entrenamiento SAI generado automáticamente
Configurado para A100 con monitoreo robusto
Generado: 2025-08-22T17:02:10.775932
"""

import os
import sys
import torch
import yaml
import json
from pathlib import Path
from datetime import datetime

# Configuración optimizada
CONFIG = {
    "model_type": "yolo",
    "dataset_path": "/data/sai-benchmark/RNA/data/mega_fire_dataset",
    "device": "auto",
    "seed": 42,
    "deterministic": true,
    "epochs": 100,
    "patience": 20,
    "save_period": 5,
    "val_period": 1,
    "optimizer": "AdamW",
    "lr0": 0.001,
    "weight_decay": 0.0005,
    "momentum": 0.937,
    "warmup_epochs": 3,
    "warmup_momentum": 0.8,
    "warmup_bias_lr": 0.1,
    "lrf": 0.01,
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
    "val_split": 0.2,
    "shuffle": true,
    "rect": false,
    "auto_augment": "randaugment",
    "verbose": true,
    "save_json": true,
    "save_hybrid": true,
    "conf": null,
    "iou": 0.7,
    "max_det": 300,
    "half": false,
    "dnn": false,
    "robust_validation": true,
    "validation_frequency": 1,
    "checkpoint_validation": true,
    "nan_detection": true,
    "automatic_recovery": true,
    "emergency_stop": true,
    "num_workers": 16,
    "save_dir": "/data/sai-benchmark/RNA/training/runs",
    "cleanup_old_checkpoints": true,
    "max_checkpoints": 10
}

def setup_environment():
    """Configurar entorno de entrenamiento"""
    
    # Configuraciones de PyTorch para A100
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False  # Para velocidad máxima
    torch.set_float32_matmul_precision('medium')  # Tensor cores
    
    # Configuraciones de memoria
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
    
    # Configuraciones de threading
    torch.set_num_threads(16)
    
    print("🔧 Entorno configurado para A100")

def validate_config():
    """Validar configuración antes del entrenamiento"""
    
    errors = []
    
    # Verificar dataset
    if not Path(CONFIG['dataset_path']).exists():
        errors.append(f"Dataset no encontrado: {CONFIG['dataset_path']}")
    
    # Verificar GPU
    if not torch.cuda.is_available():
        errors.append("CUDA no disponible")
    
    # Verificar espacio en disco
    import shutil
    free_gb = shutil.disk_usage('/data')[2] / (1024**3)
    if free_gb < 50:
        errors.append(f"Poco espacio en disco: {free_gb:.1f}GB")
    
    if errors:
        print("❌ Errores de configuración:")
        for error in errors:
            print(f"   - {error}")
        sys.exit(1)
    
    print("✅ Configuración validada")

def start_monitoring():
    """Iniciar monitoreo en background"""
    
    try:
        from RNA.scripts.robust_training_monitor import TrainingMonitor, create_monitoring_config
        
        monitor_config = create_monitoring_config(
            training_dir=CONFIG['save_dir'],
            log_dir=CONFIG['save_dir'] + '/monitoring'
        )
        
        monitor = TrainingMonitor(monitor_config)
        monitor.start_monitoring()
        
        print("🔍 Monitor de entrenamiento iniciado")
        return monitor
        
    except ImportError:
        print("⚠️ Monitor no disponible, continuando sin monitoreo")
        return None

def train_model():
    """Ejecutar entrenamiento principal"""
    
    if CONFIG['model_type'] == 'yolo':
        from ultralytics import YOLO
        
        # Crear modelo
        model = YOLO('yolov8s.pt')
        
        # Entrenar con configuración robusta
        results = model.train(
            data=CONFIG['dataset_path'] + '/dataset.yaml',
            epochs=CONFIG['epochs'],
            imgsz=1440,
            batch=CONFIG['batch_size'],
            device='auto',
            amp=CONFIG.get('amp', True),
            cache=CONFIG.get('cache', True),
            patience=CONFIG['patience'],
            save_period=CONFIG['save_period'],
            project=CONFIG['save_dir'],
            name='sai_detector_robust',
            exist_ok=True,
            verbose=True,
            seed=CONFIG['seed'],
            deterministic=CONFIG.get('deterministic', True),
            val=True,
            plots=True,
            save_json=True
        )
        
    elif CONFIG['model_type'] == 'verificator':
        sys.path.append(str(Path(__file__).parent))
        from train_verificator import VerificatorTrainer
        
        trainer = VerificatorTrainer(CONFIG)
        results = trainer.train()
    
    return results

def main():
    """Función principal"""
    
    print("🚀 SAI ENTRENAMIENTO ROBUSTO A100")
    print("=" * 50)
    print(f"⏰ Inicio: {datetime.now()}")
    print(f"🎯 Modelo: {CONFIG['model_type']}")
    print(f"📊 Épocas: {CONFIG['epochs']}")
    print(f"🔢 Batch size: {CONFIG['batch_size']}")
    print()
    
    try:
        # Configurar entorno
        setup_environment()
        
        # Validar configuración
        validate_config()
        
        # Iniciar monitoreo
        monitor = start_monitoring()
        
        # Entrenar modelo
        print("🔥 Iniciando entrenamiento...")
        results = train_model()
        
        print("\n🎉 ENTRENAMIENTO COMPLETADO")
        print(f"⏰ Finalizado: {datetime.now()}")
        
        # Detener monitoreo
        if monitor:
            monitor.stop_monitoring()
        
        return results
        
    except KeyboardInterrupt:
        print("\n⏹️ Entrenamiento interrumpido por usuario")
        sys.exit(0)
        
    except Exception as e:
        print(f"\n❌ Error en entrenamiento: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
