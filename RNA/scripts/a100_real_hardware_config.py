#!/usr/bin/env python3
"""
SAI Configuración REAL para A100-SXM4-40GB
Configuración corregida para el hardware real del servidor alquilado

HARDWARE VERIFICADO:
- GPU: A100-SXM4-40GB (40.96GB VRAM)
- RAM: 251GB (243GB disponibles)
- CPU: 128 cores
- Disco: 300GB (285GB libres)
- Driver: 550.107.02
"""

import json
import yaml
from pathlib import Path
from datetime import datetime

class A100RealHardwareConfig:
    """Configuración específica para A100 real del servidor alquilado"""
    
    def __init__(self):
        # Hardware REAL verificado del servidor A100
        self.hardware_specs = {
            'gpu_name': 'NVIDIA A100-SXM4-40GB',
            'gpu_memory_gb': 40.96,
            'ram_total_gb': 251,
            'ram_available_gb': 243,
            'cpu_cores': 128,
            'disk_total_gb': 300,
            'disk_available_gb': 285,
            'cuda_driver': '550.107.02'
        }
        
        print("🔧 Configurando para A100 REAL:")
        print(f"   GPU: {self.hardware_specs['gpu_name']}")
        print(f"   VRAM: {self.hardware_specs['gpu_memory_gb']:.1f}GB")
        print(f"   RAM: {self.hardware_specs['ram_available_gb']:.0f}GB disponible")
        print(f"   CPU: {self.hardware_specs['cpu_cores']} cores")
        print(f"   Disco: {self.hardware_specs['disk_available_gb']:.0f}GB libres")
    
    def get_yolo_detector_config(self):
        """Configuración optimizada para YOLOv8 en A100 real"""
        
        config = {
            # Configuración del modelo
            'model_type': 'yolo',
            'model_name': 'yolov8s.pt',
            'task': 'detect',
            
            # Dataset
            'data': '/data/sai-benchmark/RNA/data/mega_fire_dataset/dataset.yaml',
            'imgsz': 1440,  # Resolución nativa del dataset
            
            # Configuración de entrenamiento optimizada para A100
            'epochs': 100,
            'batch': 32,  # Aumentado para A100 40GB (vs 24 para RTX 3090)
            'patience': 20,  # Conservador para evitar early stopping prematuro
            'save_period': 5,
            'val': True,
            'plots': True,
            'save_json': True,
            
            # Configuración de dispositivo A100
            'device': 'auto',
            'amp': True,  # Mixed precision para A100
            'cache': True,  # Cache completo en RAM (243GB disponibles)
            'compile': True,  # PyTorch 2.0 compilation para A100
            
            # Optimizaciones específicas A100
            'workers': 32,  # Aumentado para 128 cores (vs 16 para local)
            'pin_memory': True,
            'persistent_workers': True,
            'prefetch_factor': 8,  # Aumentado para RAM abundante
            
            # Configuración de optimizador
            'optimizer': 'AdamW',
            'lr0': 0.001,
            'lrf': 0.01,
            'momentum': 0.937,
            'weight_decay': 0.0005,
            'warmup_epochs': 3,
            'warmup_momentum': 0.8,
            'warmup_bias_lr': 0.1,
            'cos_lr': True,
            
            # Data augmentation
            'hsv_h': 0.015,
            'hsv_s': 0.7,
            'hsv_v': 0.4,
            'degrees': 0.0,
            'translate': 0.1,
            'scale': 0.5,
            'shear': 0.0,
            'perspective': 0.0,
            'flipud': 0.0,
            'fliplr': 0.5,
            'mosaic': 1.0,
            'mixup': 0.0,
            'copy_paste': 0.0,
            
            # Configuración de directorios
            'project': '/data/sai-benchmark/RNA/training/runs',
            'name': 'sai_detector_a100_real',
            'exist_ok': True,
            
            # Configuración de robustez
            'verbose': True,
            'seed': 42,
            'deterministic': True,
            'rect': False,
            'overlap_mask': True,
            'mask_ratio': 4,
            'dropout': 0.0,
            'val_split': 0.0,  # Usar split predefinido del dataset
            
            # Configuración específica para prevenir corrupción
            'half': False,  # Evitar FP16 en validación
            'dnn': False,
            'auto_augment': 'randaugment',
            'erasing': 0.4,
            'crop_fraction': 1.0
        }
        
        return config
    
    def get_verificator_config(self):
        """Configuración optimizada para Verificator CNN en A100 real"""
        
        config = {
            # Configuración del modelo
            'model_type': 'verificator',
            'backbone': 'efficientnet_b0',
            'num_classes': 2,
            'temporal_frames': 1,
            'dropout': 0.3,
            
            # Dataset
            'dataset_path': '/data/sai-benchmark/RNA/data/verificator_dataset',
            
            # Configuración de entrenamiento optimizada para A100
            'epochs': 50,
            'batch_size': 256,  # Significativamente aumentado para A100 (vs 128)
            'learning_rate': 3e-4,  # Aumentado para batch size mayor
            'weight_decay': 1e-5,
            
            # Configuración de dispositivo A100
            'device': 'auto',
            'num_workers': 64,  # Aumentado para 128 cores (vs 16)
            'pin_memory': True,
            'persistent_workers': True,
            'prefetch_factor': 8,
            
            # Configuración de optimizador
            'optimizer': 'adamw',
            'scheduler': 'reduce_on_plateau',
            'scheduler_patience': 5,
            'patience': 15,  # Early stopping
            'grad_clip': 1.0,
            'use_class_weights': True,
            
            # Configuración de directorios
            'output_dir': '/data/sai-benchmark/RNA/training/runs/verificator_a100_real',
            'save_every': 5,
            
            # Configuración de robustez
            'seed': 42,
            'deterministic': True,
            'mixed_precision': True,
            'compile_model': True,
            
            # Configuraciones específicas A100
            'tensor_cores': True,
            'channels_last': True,
            'torch_compile': True,
            'cudnn_benchmark': True
        }
        
        return config
    
    def get_monitoring_config(self):
        """Configuración de monitoreo para A100 real"""
        
        config = {
            'enabled': True,
            'training_dir': '/data/sai-benchmark/RNA/training/runs',
            'log_dir': '/data/sai-benchmark/RNA/training/monitoring',
            'monitor_interval': 10,  # Más frecuente para A100 rápido
            
            # Thresholds ajustados para A100
            'alert_thresholds': {
                'nan_consecutive': 2,
                'memory_usage': 0.90,  # A100 puede usar más VRAM
                'ram_usage': 0.85,     # 243GB disponibles
                'disk_usage': 0.80,    # 285GB libres, threshold más conservador
                'gpu_memory': 0.85,    # 40GB VRAM
                'gpu_utilization_min': 0.70,  # A100 debe estar bien utilizado
                'validation_loss_spike': 5.0,
                'no_improvement_epochs': 15
            },
            
            # Configuración específica A100
            'monitor_gpu_stats': True,
            'monitor_nvlink': True,  # A100 tiene NVLink
            'monitor_tensor_cores': True,
            'track_memory_fragmentation': True,
            
            'emergency_actions': {
                'create_checkpoint': True,
                'send_alerts': True,
                'pause_training': False,
                'notify_user': True,
                'save_system_state': True
            },
            
            'validation_checks': {
                'check_nan_values': True,
                'check_inf_values': True,
                'check_metric_ranges': True,
                'check_gradient_norms': True,
                'check_learning_rate': True,
                'check_gpu_memory_leaks': True,
                'validate_tensor_shapes': True
            }
        }
        
        return config
    
    def create_optimized_training_scripts(self, output_dir: str):
        """Crear scripts de entrenamiento optimizados para A100 real"""
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Script para detector YOLOv8
        yolo_config = self.get_yolo_detector_config()
        yolo_script = self.create_yolo_script(yolo_config)
        
        yolo_script_path = output_path / 'train_detector_a100_real.py'
        with open(yolo_script_path, 'w') as f:
            f.write(yolo_script)
        
        # Script para verificator
        verificator_config = self.get_verificator_config()
        verificator_script = self.create_verificator_script(verificator_config)
        
        verificator_script_path = output_path / 'train_verificator_a100_real.py'
        with open(verificator_script_path, 'w') as f:
            f.write(verificator_script)
        
        # Configuraciones YAML
        yolo_yaml_path = output_path / 'yolo_detector_a100_real.yaml'
        with open(yolo_yaml_path, 'w') as f:
            yaml.dump(yolo_config, f, default_flow_style=False)
        
        verificator_yaml_path = output_path / 'verificator_a100_real.yaml'
        with open(verificator_yaml_path, 'w') as f:
            yaml.dump(verificator_config, f, default_flow_style=False)
        
        # Script de monitoreo
        monitoring_config = self.get_monitoring_config()
        monitoring_script = self.create_monitoring_script(monitoring_config)
        
        monitoring_script_path = output_path / 'start_a100_monitoring.py'
        with open(monitoring_script_path, 'w') as f:
            f.write(monitoring_script)
        
        # Hacer ejecutables
        import os
        for script_path in [yolo_script_path, verificator_script_path, monitoring_script_path]:
            os.chmod(script_path, 0o755)
        
        return {
            'yolo_script': str(yolo_script_path),
            'verificator_script': str(verificator_script_path),
            'monitoring_script': str(monitoring_script_path),
            'yolo_config': str(yolo_yaml_path),
            'verificator_config': str(verificator_yaml_path)
        }
    
    def create_yolo_script(self, config):
        """Crear script optimizado para YOLOv8 en A100"""
        
        return f'''#!/usr/bin/env python3
"""
SAI Detector YOLOv8 - Optimizado para A100-SXM4-40GB REAL
Hardware: A100 40GB VRAM, 243GB RAM, 128 CPU cores
Generado: {datetime.now().isoformat()}
"""

import os
import sys
import torch
from pathlib import Path
from datetime import datetime

# Configuración REAL A100
HARDWARE_CONFIG = {json.dumps(self.hardware_specs, indent=4)}

TRAINING_CONFIG = {json.dumps(config, indent=4)}

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
    
    print(f"✅ Configurado para {{torch.cuda.get_device_name(0)}}")
    print(f"✅ VRAM disponible: {{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}}GB")

def validate_a100_hardware():
    """Validar que estamos en A100 correcto"""
    
    if not torch.cuda.is_available():
        print("❌ CUDA no disponible")
        return False
    
    gpu_name = torch.cuda.get_device_name(0)
    if 'A100' not in gpu_name:
        print(f"⚠️ GPU detectado: {{gpu_name}}")
        print("⚠️ Esperado: A100-SXM4-40GB")
        response = input("¿Continuar de todas formas? (y/N): ")
        if response.lower() != 'y':
            return False
    
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    expected_vram = HARDWARE_CONFIG['gpu_memory_gb']
    
    if abs(vram_gb - expected_vram) > 5:  # 5GB tolerance
        print(f"⚠️ VRAM detectado: {{vram_gb:.1f}}GB")
        print(f"⚠️ Esperado: {{expected_vram:.1f}}GB")
        response = input("¿Continuar de todas formas? (y/N): ")
        if response.lower() != 'y':
            return False
    
    print(f"✅ Hardware validado: {{gpu_name}} ({{vram_gb:.1f}}GB)")
    return True

def start_training():
    """Iniciar entrenamiento YOLOv8"""
    
    try:
        from ultralytics import YOLO
        
        print("🚀 INICIANDO ENTRENAMIENTO YOLO A100")
        print("=" * 50)
        print(f"⏰ Inicio: {{datetime.now()}}")
        print(f"🎯 Configuración A100 optimizada")
        print(f"📊 Batch size: {{TRAINING_CONFIG['batch']}}")
        print(f"🔢 Workers: {{TRAINING_CONFIG['workers']}}")
        print(f"💾 Cache: {{TRAINING_CONFIG['cache']}}")
        print()
        
        # Crear modelo
        model = YOLO(TRAINING_CONFIG['model_name'])
        
        # Entrenar con configuración A100
        results = model.train(**TRAINING_CONFIG)
        
        print("\\n🎉 ENTRENAMIENTO A100 COMPLETADO")
        print(f"⏰ Finalizado: {{datetime.now()}}")
        
        return results
        
    except Exception as e:
        print(f"❌ Error en entrenamiento: {{e}}")
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
'''
    
    def create_verificator_script(self, config):
        """Crear script optimizado para Verificator en A100"""
        
        return f'''#!/usr/bin/env python3
"""
SAI Verificator CNN - Optimizado para A100-SXM4-40GB REAL
Hardware: A100 40GB VRAM, 243GB RAM, 128 CPU cores
Generado: {datetime.now().isoformat()}
"""

import os
import sys
import torch
from pathlib import Path
from datetime import datetime

# Configuración REAL A100
HARDWARE_CONFIG = {json.dumps(self.hardware_specs, indent=4)}

TRAINING_CONFIG = {json.dumps(config, indent=4)}

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
    
    print(f"✅ Configurado para CNN en {{torch.cuda.get_device_name(0)}}")

def start_verificator_training():
    """Iniciar entrenamiento del verificator"""
    
    try:
        # Importar trainer del verificator
        sys.path.append('/data/sai-benchmark')
        from RNA.scripts.train_verificator import VerificatorTrainer
        
        print("🚀 INICIANDO ENTRENAMIENTO VERIFICATOR A100")
        print("=" * 50)
        print(f"⏰ Inicio: {{datetime.now()}}")
        print(f"🎯 Configuración A100 optimizada")
        print(f"📊 Batch size: {{TRAINING_CONFIG['batch_size']}}")
        print(f"🔢 Workers: {{TRAINING_CONFIG['num_workers']}}")
        print(f"🧠 Backbone: {{TRAINING_CONFIG['backbone']}}")
        print()
        
        # Crear trainer con configuración A100
        trainer = VerificatorTrainer(TRAINING_CONFIG)
        
        # Entrenar
        best_f1, final_metrics = trainer.train()
        
        print("\\n🎉 ENTRENAMIENTO VERIFICATOR A100 COMPLETADO")
        print(f"⏰ Finalizado: {{datetime.now()}}")
        print(f"🏆 Mejor F1: {{best_f1:.3f}}")
        
        return best_f1, final_metrics
        
    except Exception as e:
        print(f"❌ Error en entrenamiento verificator: {{e}}")
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
        print(f"✅ Entrenamiento exitoso - F1: {{best_f1:.3f}}")
        return 0
    else:
        print("❌ Entrenamiento fallido")
        return 1

if __name__ == "__main__":
    sys.exit(main())
'''
    
    def create_monitoring_script(self, config):
        """Crear script de monitoreo específico A100"""
        
        return f'''#!/usr/bin/env python3
"""
SAI Monitoring - Específico para A100-SXM4-40GB
Monitoreo optimizado para hardware del servidor alquilado
"""

import sys
import os
sys.path.append('/data/sai-benchmark')

from RNA.scripts.robust_training_monitor import TrainingMonitor
import json
from datetime import datetime

# Configuración REAL A100
HARDWARE_CONFIG = {json.dumps(self.hardware_specs, indent=4)}

MONITORING_CONFIG = {json.dumps(config, indent=4)}

def main():
    """Iniciar monitoreo A100"""
    
    print("🔍 SAI A100 MONITORING")
    print("=" * 30)
    print(f"🖥️ Hardware: {{HARDWARE_CONFIG['gpu_name']}}")
    print(f"💾 VRAM: {{HARDWARE_CONFIG['gpu_memory_gb']}}GB")
    print(f"🔢 RAM: {{HARDWARE_CONFIG['ram_available_gb']}}GB")
    print(f"⚙️ CPU: {{HARDWARE_CONFIG['cpu_cores']}} cores")
    print(f"💿 Disco: {{HARDWARE_CONFIG['disk_available_gb']}}GB")
    print()
    
    # Crear monitor con configuración A100
    monitor = TrainingMonitor(MONITORING_CONFIG)
    
    try:
        # Iniciar monitoreo
        monitor.start_monitoring()
        
        print("🔍 Monitoreo A100 iniciado")
        print("📊 Presiona Ctrl+C para detener")
        
        # Mantener corriendo
        import time
        while True:
            time.sleep(60)
            status = monitor.get_status_report()
            print(f"📈 Estado: Época {{status.get('last_epoch', '?')}}, "
                  f"Anomalías: {{status.get('nan_count', 0)}}")
    
    except KeyboardInterrupt:
        print("\\n⏹️ Deteniendo monitoreo A100...")
        monitor.stop_monitoring()
        print("✅ Monitoreo detenido")

if __name__ == "__main__":
    main()
'''


def main():
    """Función principal"""
    
    print("🔧 RECONFIGURANDO PARA A100 REAL")
    print("=" * 40)
    
    # Crear configurador A100 real
    a100_config = A100RealHardwareConfig()
    
    # Crear scripts corregidos
    output_dir = '/mnt/n8n-data/sai-benchmark/RNA/deployment_a100_real'
    scripts = a100_config.create_optimized_training_scripts(output_dir)
    
    print(f"\\n📦 Scripts A100 REAL creados en: {output_dir}")
    print("\\n📋 Archivos generados:")
    for name, path in scripts.items():
        print(f"   ✅ {name}: {path}")
    
    # Crear script de transferencia a A100
    transfer_script = Path(output_dir) / 'transfer_to_a100.sh'
    with open(transfer_script, 'w') as f:
        f.write(f'''#!/bin/bash
# Script para transferir configuración A100 REAL al servidor
# Generado: {datetime.now().isoformat()}

echo "🚀 Transfiriendo configuración A100 REAL al servidor..."

# Transferir scripts de entrenamiento
scp -P 31939 -i ~/.ssh/sai-n8n-deploy \\
    {output_dir}/train_detector_a100_real.py \\
    {output_dir}/train_verificator_a100_real.py \\
    {output_dir}/start_a100_monitoring.py \\
    root@88.207.86.56:/data/sai-benchmark/RNA/scripts/

# Transferir configuraciones
scp -P 31939 -i ~/.ssh/sai-n8n-deploy \\
    {output_dir}/*.yaml \\
    root@88.207.86.56:/data/sai-benchmark/RNA/configs/

# Transferir monitor robusto
scp -P 31939 -i ~/.ssh/sai-n8n-deploy \\
    /mnt/n8n-data/sai-benchmark/RNA/scripts/robust_training_monitor.py \\
    /mnt/n8n-data/sai-benchmark/RNA/deployment_a100/validation_suite.py \\
    root@88.207.86.56:/data/sai-benchmark/RNA/scripts/

echo "✅ Transferencia completada"
echo "🎯 Para usar en A100:"
echo "   cd /data/sai-benchmark"
echo "   python3 RNA/scripts/train_detector_a100_real.py"
echo "   # o"
echo "   python3 RNA/scripts/train_verificator_a100_real.py"
''')
    
    import os
    os.chmod(transfer_script, 0o755)
    
    print(f"\\n🔄 Script de transferencia: {transfer_script}")
    print(f"\\n🎯 PRÓXIMO PASO:")
    print(f"   cd {output_dir}")
    print(f"   ./transfer_to_a100.sh")


if __name__ == "__main__":
    main()