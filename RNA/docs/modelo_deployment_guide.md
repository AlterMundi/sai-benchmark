# SAI - Guía de Despliegue y Manejo de Modelos

## 🎯 **Visión General**

Esta guía explica cómo manejar, transferir y versionar los modelos entrenados del sistema SAI, incluyendo estrategias para desarrollo y producción.

## 📁 **Estructura de Modelos SAI**

### **Modelos del Sistema Completo**
```bash
# Archivos de modelos entrenados
detector_best.pt        67MB    # YOLOv8-s Detector (Stage A)
verificator_best.pt     25MB    # SmokeyNet-Lite Verificator (Stage B)
total_system           92MB     # Sistema completo SAI
```

### **Contenido de los Archivos .pt**
Los archivos `.pt` de PyTorch contienen **todo lo necesario** en un solo archivo:

```python
# Estructura del checkpoint completo
checkpoint_data = {
    'model': model,                    # Arquitectura completa del modelo
    'state_dict': model.state_dict(),  # Pesos entrenados
    'optimizer': optimizer.state_dict(), # Estado del optimizador
    'training_results': results,       # Métricas de entrenamiento
    'epoch': 100,                     # Época final
    'config': training_config,        # Configuración de entrenamiento
    'date': datetime.now(),           # Timestamp de creación
    'version': '1.0.0',               # Versión del modelo
}

# Para usar directamente:
from ultralytics import YOLO
model = YOLO('detector_best.pt')  # ¡Listo para inferencia!
```

## 🔄 **Estrategias de Transferencia**

### **Transferencia A100 → Servidor Local**

#### **Método Recomendado: SCP Directo**
```bash
# Transferir modelos principales desde A100
scp -P 31939 -i ~/.ssh/sai-n8n-deploy \
  root@88.207.86.56:/data/sai-benchmark/RNA/training/runs/sai_detector_training/weights/best.pt \
  /mnt/n8n-data/sai-benchmark/RNA/models/detector_best.pt

scp -P 31939 -i ~/.ssh/sai-n8n-deploy \
  root@88.207.86.56:/data/sai-benchmark/RNA/training/runs/verificator/verificator_best.pt \
  /mnt/n8n-data/sai-benchmark/RNA/models/verificator_best.pt

# Tiempo estimado: ~30 segundos para 92MB total
```

#### **Verificación Post-Transferencia**
```bash
# Verificar integridad de modelos transferidos
python3 -c "
import torch
from ultralytics import YOLO

# Verificar detector
detector = YOLO('RNA/models/detector_best.pt')
print(f'✅ Detector cargado: {len(detector.model.parameters())} parámetros')

# Verificar verificator
verificator = torch.load('RNA/models/verificator_best.pt')
print(f'✅ Verificator cargado: época {verificator[\"epoch\"]}')
print('🔥 Modelos SAI transferidos exitosamente')
"
```

## 📚 **Manejo de Archivos Grandes con Git**

### **¿Qué es Git LFS?**
Git LFS (Large File Storage) es una extensión que maneja archivos grandes eficientemente:

```bash
# Problema sin Git LFS
git add detector_best.pt  # 67MB van al repositorio
git clone                # Descarga TODO el historial de modelos
# Resultado: repositorio de varios GB

# Solución con Git LFS
git lfs track "*.pt"      # Configurar tracking
git add detector_best.pt  # Solo se guarda un "pointer" pequeño
git clone                # Repo liviano, modelos por separado
```

### **Configuración Git LFS (Opcional)**
```bash
# Instalar y configurar Git LFS
git lfs install

# Configurar tracking para modelos SAI
git lfs track "*.pt"
git lfs track "*.pth"
git lfs track "*.onnx"
git lfs track "RNA/models/*.pt"

# Confirmar configuración
git add .gitattributes
git commit -m "Configure Git LFS for model files"
```

## 🎯 **Estrategia de Versionado SAI**

### **Para Desarrollo (Actual - Recomendado)**
```bash
# Estructura simple sin Git LFS
RNA/models/
├── detector_best.pt           # Modelo actual del detector
├── verificator_best.pt        # Modelo actual del verificator
├── detector_backup.pt         # Backup del detector
└── verificator_backup.pt      # Backup del verificator

# Manejo:
# 1. Transferencia directa via SCP
# 2. Backups locales manuales
# 3. Sin versionado en Git (archivos en .gitignore)
```

**Ventajas del método actual:**
- ✅ **Simple**: No requiere configuración extra
- ✅ **Rápido**: Transferencia directa sin overhead
- ✅ **Flexible**: Fácil reemplazar modelos
- ✅ **Sin dependencias**: No requiere Git LFS

### **Para Producción (Futuro - Con Versionado)**
```bash
# Estructura con versionado semántico
RNA/models/
├── detector/
│   ├── v1.0.0_detector_best.pt      # Primera versión estable
│   ├── v1.1.0_detector_best.pt      # Mejora menor
│   ├── v2.0.0_detector_best.pt      # Cambio mayor
│   └── latest -> v2.0.0_detector_best.pt  # Symlink a actual
├── verificator/
│   ├── v1.0.0_verificator_best.pt
│   ├── v1.1.0_verificator_best.pt
│   └── latest -> v1.1.0_verificator_best.pt
└── metadata/
    ├── model_registry.json          # Registro de versiones
    └── performance_benchmarks.json  # Métricas por versión
```

### **Script de Versionado Automático**
```python
#!/usr/bin/env python3
# RNA/scripts/version_models.py
"""
Script para versionar modelos SAI automáticamente
"""

import torch
import json
import shutil
from pathlib import Path
from datetime import datetime

def version_model(model_path, model_type, version="auto"):
    """
    Versiona un modelo entrenado con metadatos
    
    Args:
        model_path: Ruta al modelo .pt
        model_type: "detector" o "verificator"  
        version: Versión (auto-detecta si es "auto")
    """
    
    models_dir = Path("RNA/models") / model_type
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Auto-detectar versión si es necesario
    if version == "auto":
        existing = list(models_dir.glob("v*.pt"))
        if not existing:
            version = "v1.0.0"
        else:
            # Incrementar versión minor
            last_version = sorted(existing)[-1].stem.split('_')[0]
            major, minor, patch = last_version[1:].split('.')
            version = f"v{major}.{int(minor)+1}.0"
    
    # Copiar modelo con nueva versión
    new_path = models_dir / f"{version}_{model_type}_best.pt"
    shutil.copy2(model_path, new_path)
    
    # Actualizar symlink latest
    latest_link = models_dir / "latest"
    if latest_link.exists():
        latest_link.unlink()
    latest_link.symlink_to(new_path.name)
    
    # Cargar modelo para extraer metadatos
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Actualizar registro de modelos
    registry_path = Path("RNA/models/metadata/model_registry.json")
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    
    if registry_path.exists():
        with open(registry_path, 'r') as f:
            registry = json.load(f)
    else:
        registry = {"models": []}
    
    # Añadir nueva versión al registro
    model_info = {
        "version": version,
        "type": model_type,
        "path": str(new_path),
        "created": datetime.now().isoformat(),
        "size_mb": new_path.stat().st_size / (1024*1024),
        "training_epochs": checkpoint.get('epoch', 'unknown'),
        "performance": checkpoint.get('metrics', {}),
        "config": checkpoint.get('config', {})
    }
    
    registry["models"].append(model_info)
    
    with open(registry_path, 'w') as f:
        json.dump(registry, f, indent=2)
    
    print(f"✅ Modelo versionado como {version}")
    print(f"📁 Guardado en: {new_path}")
    print(f"🔗 Latest actualizado: {latest_link}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', help='Ruta al modelo .pt')
    parser.add_argument('model_type', choices=['detector', 'verificator'])
    parser.add_argument('--version', default='auto', help='Versión del modelo')
    
    args = parser.parse_args()
    version_model(args.model_path, args.model_type, args.version)
```

## 🚀 **Flujo de Despliegue Completo**

### **1. Post-Entrenamiento A100**
```bash
# En servidor A100 (cuando termine entrenamiento)
cd /data/sai-benchmark

# Verificar modelos generados
ls -la RNA/training/runs/sai_detector_training/weights/
ls -la RNA/training/runs/verificator/

# Opcional: crear tar.gz para backup
tar -czf sai_models_$(date +%Y%m%d).tar.gz \
  RNA/training/runs/*/weights/best.pt \
  RNA/training/runs/verificator/verificator_best.pt
```

### **2. Transferencia a Servidor Local**
```bash
# En servidor local
cd /mnt/n8n-data/sai-benchmark

# Crear directorio de modelos
mkdir -p RNA/models

# Transferir modelos principales
scp -P 31939 -i ~/.ssh/sai-n8n-deploy \
  root@88.207.86.56:/data/sai-benchmark/RNA/training/runs/sai_detector_training/weights/best.pt \
  RNA/models/detector_best.pt

scp -P 31939 -i ~/.ssh/sai-n8n-deploy \
  root@88.207.86.56:/data/sai-benchmark/RNA/training/runs/verificator/verificator_best.pt \
  RNA/models/verificator_best.pt

# Verificar transferencia exitosa
python3 -c "
import torch
from ultralytics import YOLO

detector = YOLO('RNA/models/detector_best.pt')
verificator = torch.load('RNA/models/verificator_best.pt')

print('🔥 SAI Models Ready:')
print(f'   Detector: {detector.model.__class__.__name__}')
print(f'   Verificator: Epoch {verificator[\"epoch\"]}')
"
```

### **3. Integración y Testing**
```bash
# Test del sistema completo
python3 -c "
from RNA.scripts.train_verificator import SmokeyNetLite
from ultralytics import YOLO
import torch

# Cargar ambos modelos
detector = YOLO('RNA/models/detector_best.pt')
verificator_checkpoint = torch.load('RNA/models/verificator_best.pt')

verificator = SmokeyNetLite()
verificator.load_state_dict(verificator_checkpoint['model_state_dict'])
verificator.eval()

print('✅ Sistema SAI completo cargado y listo')
print('🔥 Detector + Verificator operacional')
"

# Test de inferencia (opcional)
python3 RNA/scripts/sai_inference.py --test
```

## 📊 **Monitoreo de Performance**

### **Benchmarking de Modelos**
```python
# Script para comparar performance de versiones
def benchmark_sai_models():
    """Benchmark completo del sistema SAI"""
    
    import time
    import torch
    from ultralytics import YOLO
    
    # Cargar modelos
    detector = YOLO('RNA/models/detector_best.pt')
    verificator = torch.load('RNA/models/verificator_best.pt')
    
    # Test images
    test_images = ["test1.jpg", "test2.jpg", "test3.jpg"]
    
    results = {
        'detector_inference_ms': [],
        'verificator_inference_ms': [],
        'total_pipeline_ms': [],
        'memory_usage_mb': []
    }
    
    for img in test_images:
        start_time = time.time()
        
        # Stage A: Detection
        det_start = time.time()
        detections = detector(img)
        det_time = (time.time() - det_start) * 1000
        
        # Stage B: Verification (simulado)
        ver_start = time.time()
        # verification_logic here
        ver_time = (time.time() - ver_start) * 1000
        
        total_time = (time.time() - start_time) * 1000
        
        results['detector_inference_ms'].append(det_time)
        results['verificator_inference_ms'].append(ver_time)
        results['total_pipeline_ms'].append(total_time)
    
    # Calcular promedios
    avg_results = {k: sum(v)/len(v) for k, v in results.items()}
    
    print("📊 SAI Performance Benchmark:")
    print(f"   Detector: {avg_results['detector_inference_ms']:.1f}ms")
    print(f"   Verificator: {avg_results['verificator_inference_ms']:.1f}ms") 
    print(f"   Total Pipeline: {avg_results['total_pipeline_ms']:.1f}ms")
    
    return avg_results
```

## 🎯 **Recomendaciones Actuales**

### **Para Desarrollo Inmediato (Ahora)**
1. ✅ **Usar transferencia SCP directa** - Simple y eficiente
2. ✅ **Mantener modelos fuera de Git** - Usar `.gitignore` actual  
3. ✅ **Crear backups manuales** - Copias locales de seguridad
4. ✅ **Testing básico** - Verificar carga de modelos post-transferencia

### **Para Producción Futura (Cuando escalemos)**
1. 📋 **Implementar Git LFS** - Para versionado de modelos
2. 📋 **Estructura semántica** - Versionado v1.0.0, v1.1.0, etc.
3. 📋 **Automatizar deployment** - Scripts de versionado automático
4. 📋 **Monitoreo continuo** - Benchmarks de performance por versión

---

**💡 Resumen: Para el desarrollo actual del SAI, la transferencia directa via SCP es la estrategia más práctica y eficiente. El versionado con Git LFS se implementará cuando el sistema esté en producción y requiera control de versiones más estricto.**

## 🔗 **Enlaces Relacionados**

- **[SAI Complete Architecture](sai_arquitectura_completa.md)** - Arquitectura completa del sistema
- **[Stage B Verificator Guide](etapa_b_verificador.md)** - Documentación del verificador
- **[A100 Migration Plan](a100_migration_plan.md)** - Optimizaciones de cloud training
- **[Performance Estimates](performance_estimates.md)** - Métricas de rendimiento esperadas