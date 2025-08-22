#!/usr/bin/env python3
"""
SAI Configuración Optimizada para Entrenamiento en A100
Script que configura entrenamiento robusto con monitoreo completo

Recursos disponibles:
- A100 40GB VRAM
- 252GB RAM sistema  
- 300GB almacenamiento NVMe

Características:
- Configuración optimizada para hardware disponible
- Monitoreo robusto con detección temprana de corrupción
- Checkpoints frecuentes y de emergencia
- Validación continua de métricas
- Logs estructurados y alertas
"""

import os
import sys
import yaml
import torch
import subprocess
import psutil
from pathlib import Path
from datetime import datetime
import json
import shutil
from typing import Dict, List

class A100OptimizedTrainer:
    """Configurador de entrenamiento optimizado para A100"""
    
    def __init__(self, base_config: Dict):
        self.base_config = base_config
        self.hardware_info = self.detect_hardware()
        self.training_config = self.optimize_for_hardware()
        
        print("🚀 SAI A100 Optimized Trainer Initialized")
        print(f"🖥️ Hardware: {self.hardware_info}")
    
    def detect_hardware(self) -> Dict:
        """Detectar configuración de hardware"""
        
        hardware = {
            'cpu_cores': psutil.cpu_count(),
            'ram_gb': psutil.virtual_memory().total / (1024**3),
            'gpu_available': torch.cuda.is_available(),
            'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0
        }
        
        if hardware['gpu_available']:
            gpu_properties = torch.cuda.get_device_properties(0)
            hardware.update({
                'gpu_name': gpu_properties.name,
                'gpu_memory_gb': gpu_properties.total_memory / (1024**3),
                'gpu_compute_capability': f"{gpu_properties.major}.{gpu_properties.minor}"
            })
        
        # Detectar almacenamiento disponible
        disk_usage = psutil.disk_usage('/')
        hardware['disk_total_gb'] = disk_usage.total / (1024**3)
        hardware['disk_free_gb'] = disk_usage.free / (1024**3)
        
        return hardware
    
    def optimize_for_hardware(self) -> Dict:
        """Optimizar configuración según hardware disponible"""
        
        config = self.base_config.copy()
        
        # Optimizaciones específicas para A100
        if 'A100' in self.hardware_info.get('gpu_name', ''):
            print("🔥 A100 detectado - Aplicando optimizaciones específicas")
            
            # Batch size optimizado para 40GB VRAM
            if self.base_config.get('model_type') == 'yolo':
                config.update({
                    'batch_size': 24,  # Óptimo para YOLOv8-s en 1440x808
                    'amp': True,  # Mixed precision para A100
                    'compile': True,  # PyTorch 2.0 compilation
                    'cache': True,  # Cache dataset en RAM
                })
            elif self.base_config.get('model_type') == 'verificator':
                config.update({
                    'batch_size': 128,  # CNN más pequeño, batch más grande
                    'num_workers': 16,  # Aprovechar 252GB RAM
                })
            
            # Optimizaciones de memoria y computación
            config.update({
                'pin_memory': True,
                'persistent_workers': True,
                'prefetch_factor': 4,
                'gradient_accumulation_steps': 1,  # No necesario con A100
                'torch_compile': True,
                'channels_last': True,  # Optimización de memoria
            })
        
        # Configuración de workers basada en CPU
        cpu_workers = min(self.hardware_info['cpu_cores'], 16)
        config['num_workers'] = cpu_workers
        
        # Configuración de memoria
        ram_gb = self.hardware_info['ram_gb']
        if ram_gb > 200:  # 252GB disponibles
            config.update({
                'cache_ram_limit': '200GB',
                'dataset_cache': 'ram',
                'cache_images': True
            })
        
        # Configuración de almacenamiento
        config.update({
            'save_period': 5,  # Checkpoint cada 5 épocas
            'save_dir': '/data/sai-benchmark/RNA/training/runs',
            'cleanup_old_checkpoints': True,
            'max_checkpoints': 10
        })
        
        return config
    
    def create_robust_config(self, model_type: str, dataset_path: str) -> Dict:
        """Crear configuración robusta con monitoreo"""
        
        base_config = {
            'model_type': model_type,
            'dataset_path': dataset_path,
            'device': 'auto',
            'seed': 42,
            'deterministic': True,
            
            # Configuración de entrenamiento
            'epochs': 100 if model_type == 'yolo' else 50,
            'patience': 20,  # Early stopping más conservador
            'save_period': 5,
            'val_period': 1,  # Validar cada época
            
            # Configuración de optimizador
            'optimizer': 'AdamW',
            'lr0': 0.001,
            'weight_decay': 0.0005,
            'momentum': 0.937,
            'warmup_epochs': 3,
            'warmup_momentum': 0.8,
            'warmup_bias_lr': 0.1,
            
            # Configuración de scheduler
            'lrf': 0.01,  # Final learning rate factor
            'cos_lr': True,  # Cosine learning rate schedule
            
            # Configuración de data augmentation
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
            
            # Configuración de validación robusta
            'val_split': 0.2,
            'shuffle': True,
            'rect': False,  # Rectangular training deshabilitado
            'auto_augment': 'randaugment',
            
            # Configuración de logging y monitoreo
            'verbose': True,
            'save_json': True,
            'save_hybrid': True,
            'conf': None,
            'iou': 0.7,
            'max_det': 300,
            'half': False,  # Evitar FP16 en validación
            'dnn': False,
            
            # Configuración específica de robustez
            'robust_validation': True,
            'validation_frequency': 1,
            'checkpoint_validation': True,
            'nan_detection': True,
            'automatic_recovery': True,
            'emergency_stop': True
        }
        
        # Aplicar optimizaciones de hardware
        self.base_config = base_config
        optimized_config = self.optimize_for_hardware()
        
        return optimized_config
    
    def setup_monitoring(self, training_dir: str) -> Dict:
        """Configurar sistema de monitoreo robusto"""
        
        monitoring_dir = Path(training_dir) / 'monitoring'
        monitoring_dir.mkdir(parents=True, exist_ok=True)
        
        monitor_config = {
            'enabled': True,
            'training_dir': training_dir,
            'log_dir': str(monitoring_dir),
            'monitor_interval': 15,  # Verificar cada 15 segundos
            'alert_thresholds': {
                'nan_consecutive': 2,  # Más agresivo: 2 épocas con NaN
                'memory_usage': 0.95,
                'disk_usage': 0.85,  # Conservador para 300GB
                'gpu_memory': 0.90,
                'validation_loss_spike': 5.0,  # Detectar picos anómalos
                'no_improvement_epochs': 15
            },
            'emergency_actions': {
                'create_checkpoint': True,
                'send_alerts': True,
                'pause_training': False,  # No parar automáticamente
                'notify_user': True
            },
            'validation_checks': {
                'check_nan_values': True,
                'check_inf_values': True,
                'check_metric_ranges': True,
                'check_gradient_norms': True,
                'check_learning_rate': True
            }
        }
        
        return monitor_config
    
    def create_training_script(self, config: Dict, output_path: str) -> str:
        """Crear script de entrenamiento optimizado"""
        
        script_content = f'''#!/usr/bin/env python3
"""
Script de entrenamiento SAI generado automáticamente
Configurado para A100 con monitoreo robusto
Generado: {datetime.now().isoformat()}
"""

import os
import sys
import torch
import yaml
import json
from pathlib import Path
from datetime import datetime

# Configuración optimizada
CONFIG = {json.dumps(config, indent=4)}

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
    torch.set_num_threads({self.hardware_info['cpu_cores']})
    
    print("🔧 Entorno configurado para A100")

def validate_config():
    """Validar configuración antes del entrenamiento"""
    
    errors = []
    
    # Verificar dataset
    if not Path(CONFIG['dataset_path']).exists():
        errors.append(f"Dataset no encontrado: {{CONFIG['dataset_path']}}")
    
    # Verificar GPU
    if not torch.cuda.is_available():
        errors.append("CUDA no disponible")
    
    # Verificar espacio en disco
    import shutil
    free_gb = shutil.disk_usage('/data')[2] / (1024**3)
    if free_gb < 50:
        errors.append(f"Poco espacio en disco: {{free_gb:.1f}}GB")
    
    if errors:
        print("❌ Errores de configuración:")
        for error in errors:
            print(f"   - {{error}}")
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
            imgsz={config.get('imgsz', 1440)},
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
    print(f"⏰ Inicio: {{datetime.now()}}")
    print(f"🎯 Modelo: {{CONFIG['model_type']}}")
    print(f"📊 Épocas: {{CONFIG['epochs']}}")
    print(f"🔢 Batch size: {{CONFIG['batch_size']}}")
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
        
        print("\\n🎉 ENTRENAMIENTO COMPLETADO")
        print(f"⏰ Finalizado: {{datetime.now()}}")
        
        # Detener monitoreo
        if monitor:
            monitor.stop_monitoring()
        
        return results
        
    except KeyboardInterrupt:
        print("\\n⏹️ Entrenamiento interrumpido por usuario")
        sys.exit(0)
        
    except Exception as e:
        print(f"\\n❌ Error en entrenamiento: {{e}}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
'''
        
        # Guardar script
        script_path = Path(output_path)
        script_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        # Hacer ejecutable
        os.chmod(script_path, 0o755)
        
        print(f"📝 Script de entrenamiento creado: {script_path}")
        return str(script_path)
    
    def create_deployment_package(self, output_dir: str) -> str:
        """Crear paquete completo de deployment"""
        
        deployment_dir = Path(output_dir)
        deployment_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Crear configuraciones para diferentes modelos
        configs = {}
        
        # Configuración para YOLOv8 detector
        configs['yolo_detector'] = self.create_robust_config(
            model_type='yolo',
            dataset_path='/data/sai-benchmark/RNA/data/mega_fire_dataset'
        )
        
        # Configuración para verificator CNN
        configs['verificator'] = self.create_robust_config(
            model_type='verificator', 
            dataset_path='/data/sai-benchmark/RNA/data/verificator_dataset'
        )
        
        # 2. Crear scripts de entrenamiento
        scripts = {}
        for name, config in configs.items():
            script_path = deployment_dir / f'train_{name}_a100.py'
            scripts[name] = self.create_training_script(config, script_path)
        
        # 3. Crear script de monitoreo independiente
        monitor_script = deployment_dir / 'start_monitoring.py'
        with open(monitor_script, 'w') as f:
            f.write(f'''#!/usr/bin/env python3
import sys
sys.path.append('/data/sai-benchmark')
from RNA.scripts.robust_training_monitor import main
main()
''')
        os.chmod(monitor_script, 0o755)
        
        # 4. Crear configuraciones YAML
        for name, config in configs.items():
            yaml_path = deployment_dir / f'{name}_config.yaml'
            with open(yaml_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
        
        # 5. Crear script de inicio rápido
        quick_start = deployment_dir / 'quick_start.sh'
        with open(quick_start, 'w') as f:
            f.write(f'''#!/bin/bash
# SAI A100 Quick Start Script
# Generado: {datetime.now().isoformat()}

echo "🚀 SAI A100 Training Quick Start"
echo "================================"

# Verificar entorno
echo "🔍 Verificando entorno..."
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "RAM disponible: $(free -h | grep Mem | awk '{{print $7}}')"
echo "Espacio en disco: $(df -h /data | tail -1 | awk '{{print $4}}')"

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
''')
        os.chmod(quick_start, 0o755)
        
        # 6. Crear README
        readme_path = deployment_dir / 'README.md'
        with open(readme_path, 'w') as f:
            f.write(f'''# SAI A100 Optimized Training Package

Generado: {datetime.now().isoformat()}

## Hardware Detectado
- **GPU**: {self.hardware_info.get('gpu_name', 'N/A')}
- **VRAM**: {self.hardware_info.get('gpu_memory_gb', 0):.1f} GB
- **RAM**: {self.hardware_info.get('ram_gb', 0):.1f} GB
- **CPU Cores**: {self.hardware_info.get('cpu_cores', 0)}
- **Disco disponible**: {self.hardware_info.get('disk_free_gb', 0):.1f} GB

## Scripts Disponibles

### Entrenamiento Principal
- `train_yolo_detector_a100.py` - Detector YOLOv8 optimizado
- `train_verificator_a100.py` - Verificator CNN optimizado

### Monitoreo
- `start_monitoring.py` - Monitor independiente
- `RNA/scripts/robust_training_monitor.py` - Monitor completo

### Configuración
- `yolo_detector_config.yaml` - Config detector
- `verificator_config.yaml` - Config verificator

### Inicio Rápido
```bash
./quick_start.sh
```

## Características Optimizadas

### Para A100 40GB
- Batch size optimizado según modelo
- Mixed precision (AMP) habilitado
- PyTorch compilation activado
- Cache en RAM aprovechando 252GB

### Monitoreo Robusto
- Detección temprana de NaN (2 épocas)
- Checkpoints de emergencia automáticos
- Alertas en tiempo real
- Logs estructurados

### Configuración Robusta
- Early stopping conservador (patience=20)
- Validación cada época
- Checkpoints cada 5 épocas
- Configuración determinística

## Uso Recomendado

1. **Verificar recursos**:
   ```bash
   nvidia-smi
   free -h
   df -h /data
   ```

2. **Ejecutar entrenamiento**:
   ```bash
   ./quick_start.sh
   ```

3. **Monitorear progreso**:
   ```bash
   tail -f RNA/training/runs/*/monitoring/training_monitor_*.log
   ```

## Estructura de Outputs

```
/data/sai-benchmark/RNA/training/runs/
├── sai_detector_robust/
│   ├── weights/
│   │   ├── best.pt
│   │   ├── last.pt
│   │   └── epoch_*.pt
│   ├── monitoring/
│   │   ├── training_monitor_*.log
│   │   ├── alerts.json
│   │   └── monitoring_state.json
│   └── results.csv
└── verificator/
    ├── verificator_best.pt
    ├── training_curves.png
    └── confusion_matrix.png
```

## Alertas y Monitoreo

El sistema monitoreará automáticamente:
- ✅ Valores NaN/Inf en métricas
- ✅ Uso de memoria GPU/RAM
- ✅ Espacio en disco
- ✅ Procesos de entrenamiento activos
- ✅ Gradientes anómalos
- ✅ Picos en loss de validación

Los logs se guardan en tiempo real y se pueden consultar:
```bash
# Ver estado actual
cat RNA/training/runs/*/monitoring/monitoring_state.json | jq

# Ver alertas
cat RNA/training/runs/*/monitoring/alerts.json | jq

# Seguir logs en tiempo real  
tail -f RNA/training/runs/*/monitoring/training_monitor_*.log
```
''')
        
        print(f"📦 Paquete de deployment creado en: {deployment_dir}")
        print(f"🚀 Para iniciar: cd {deployment_dir} && ./quick_start.sh")
        
        return str(deployment_dir)


def main():
    """Función principal"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Crear configuración optimizada para A100')
    parser.add_argument('--output-dir', 
                       default='/data/sai-benchmark/RNA/deployment_a100',
                       help='Directorio de salida')
    parser.add_argument('--model-type',
                       choices=['yolo', 'verificator', 'both'],
                       default='both',
                       help='Tipo de modelo a configurar')
    
    args = parser.parse_args()
    
    # Crear configurador
    trainer = A100OptimizedTrainer({})
    
    # Crear paquete completo
    deployment_path = trainer.create_deployment_package(args.output_dir)
    
    print("\\n🎉 CONFIGURACIÓN A100 COMPLETADA")
    print("=" * 40)
    print(f"📁 Deployment package: {deployment_path}")
    print(f"🚀 Inicio rápido: cd {deployment_path} && ./quick_start.sh")
    print("\\n📋 Próximos pasos:")
    print("   1. Revisar configuraciones en *.yaml")
    print("   2. Ejecutar ./quick_start.sh")
    print("   3. Monitorear logs en tiempo real")
    print("   4. Verificar checkpoints cada 5 épocas")


if __name__ == "__main__":
    main()