# SAI - Arquitectura Completa del Sistema de Detección de Incendios

## 🎯 **VISIÓN GENERAL**

El Sistema de Alerta de Incendios (SAI) implementa una arquitectura híbrida de dos etapas para lograr detección temprana con máxima precisión y mínimos falsos positivos.

```
🔥 FLUJO COMPLETO DEL SISTEMA SAI
=====================================

Imagen de Entrada (1440x808) 
        ↓
┌─────────────────────────────┐
│    ETAPA A: DETECTOR        │
│    YOLOv8-s                 │
│    • Alto Recall            │
│    • Detección Rápida       │
└─────────────────────────────┘
        ↓
   Bounding Boxes + Conf.
        ↓
┌─────────────────────────────┐
│    ETAPA B: VERIFICADOR     │  
│    CNN SmokeyNet-Lite       │
│    • Reduce Falsos Positivos│
│    • Confirma Detecciones   │
└─────────────────────────────┘
        ↓
   Decisión Final: ALERTA/NO_ALERTA
        ↓
┌─────────────────────────────┐
│    SISTEMA DE NOTIFICACIÓN  │
│    • Telegram Bot           │
│    • Dashboard Web          │
│    • Logs de Eventos        │
└─────────────────────────────┘
```

## 📊 **ESPECIFICACIONES TÉCNICAS**

### **Hardware Recomendado**

#### **Servidor Central (Entrenamiento + Inferencia)**
- **GPU**: NVIDIA RTX 3090 (24GB VRAM) o A100 (40GB VRAM)
- **CPU**: AMD Ryzen 9 5900X o Intel i9-11900K
- **RAM**: 64GB DDR4
- **Storage**: 2TB NVMe SSD
- **Conectividad**: Gigabit Ethernet

#### **Nodos Edge (SAI-CAM)**
- **SBC**: Raspberry Pi 4B (8GB RAM) 
- **Cámara**: IP Camera 1080p con visión nocturna
- **Storage**: 128GB microSD (Clase 10)
- **Conectividad**: WiFi 802.11ac / Ethernet
- **Alimentación**: PoE+ (25W) o fuente 5V/3A

### **Software Stack**

#### **Servidor**
- **OS**: Ubuntu 22.04 LTS
- **Python**: 3.11+
- **Deep Learning**: PyTorch 2.0+, Ultralytics YOLOv8
- **Inference**: TensorRT, ONNX Runtime  
- **Database**: PostgreSQL para logs, Redis para cache
- **API**: FastAPI para endpoints REST

#### **Edge Devices**
- **OS**: Raspberry Pi OS Lite (64-bit)
- **Python**: 3.11+ 
- **Inference**: TFLite, ONNX Runtime ARM
- **Camera**: OpenCV, PiCamera
- **Communication**: MQTT, HTTP requests

## 🏗️ **ARQUITECTURA DETALLADA**

### **ETAPA A: DETECTOR YOLOv8-s**

#### **Objetivo**
Detectar regiones sospechosas de fuego/humo con **máximo recall** (no perder ningún incendio real).

#### **Especificaciones**
```python
# Configuración del Detector
Model: YOLOv8-s
Input Size: 1440x808 (native camera format)
Classes: [fire, smoke] 
Batch Size: 20 (A100) / 8 (RTX 3090)
Inference Time: ~15ms per image (A100)
Confidence Threshold: 0.3 (bajo para alto recall)
NMS Threshold: 0.45
```

#### **Dataset de Entrenamiento**
- **Total**: 64,000 imágenes balanceadas
- **Fuentes**: FASDD, D-Fire, NEMO, PyroNear-2024, FigLib
- **Split**: 80% train / 20% validation
- **Augmentación**: Mosaic, MixUp, CutMix, HSV, Blur

#### **Métricas Objetivo**
- **Recall**: ≥95% (crítico - no perder incendios reales)
- **mAP@0.5**: ≥85%
- **Inference Speed**: <20ms por imagen
- **False Positive Rate**: ~40% (aceptable, será corregido por verificador)

#### **Salidas del Detector**
```python
detection_output = {
    'bounding_boxes': [(x1, y1, x2, y2), ...],
    'confidences': [0.85, 0.72, ...],
    'classes': ['fire', 'smoke', ...],
    'inference_time_ms': 15.2
}
```

### **ETAPA B: VERIFICADOR SmokeyNet-Lite**

#### **Objetivo**
Confirmar o rechazar detecciones del YOLOv8, reduciendo **falsos positivos en 70%+**.

#### **Especificaciones**
```python
# Configuración del Verificador
Model: SmokeyNet-Lite (EfficientNet-B0 + LSTM)
Input Size: 224x224 (crops de bounding boxes)
Classes: [true_detection, false_positive]
Batch Size: 64 (A100) / 32 (RTX 3090) 
Inference Time: ~5ms per crop (A100)
Confidence Threshold: 0.5
```

#### **Arquitectura del Verificador**
```python
class SmokeyNetLite(nn.Module):
    def __init__(self):
        # Backbone pre-entrenado
        self.backbone = EfficientNet-B0(pretrained=True)  # 1280 features
        
        # Módulo temporal (opcional)
        self.lstm = LSTM(1280, 256, bidirectional=True)  # 512 features
        
        # Clasificador binario
        self.classifier = Sequential(
            Linear(512, 256),
            ReLU(), Dropout(0.3),
            Linear(256, 64),
            ReLU(), Dropout(0.3), 
            Linear(64, 2)  # [true_detection, false_positive]
        )
```

#### **Dataset del Verificador**
```
Estructura del Dataset:
RNA/data/verificator_dataset/
├── images/
│   ├── train/ (40,000 crops)
│   │   ├── true_fire/      (10,000 - 25%)
│   │   ├── true_smoke/     (10,000 - 25%) 
│   │   └── false_positive/ (20,000 - 50%)
│   └── val/ (10,000 crops)
│       ├── true_fire/      (2,500)
│       ├── true_smoke/     (2,500)
│       └── false_positive/ (5,000)
```

#### **Estrategia de Falsos Positivos**
- **Nubes**: Detectadas como humo → rechazar
- **Vapores**: Vapor agua, niebla → rechazar  
- **Reflejos**: Reflejos solares → rechazar
- **Polvo**: Polvo en movimiento → rechazar
- **Objetos**: Chimeneas, estructuras → rechazar

#### **Métricas Objetivo**
- **Precisión Sistema**: ≥85% (vs 60% solo detector)
- **Reducción FP**: ≥70% 
- **Retención TP**: ≥95%
- **F1-Score**: ≥0.80
- **Inference Speed**: <10ms por crop

### **LÓGICA DE DECISIÓN COMBINADA**

```python
def sai_decision_engine(image):
    """
    Motor de decisión SAI completo
    Combina Detector + Verificador para decisión final
    """
    
    # ETAPA A: Detección
    detections = yolo_detector(image, conf_threshold=0.3)
    
    if not detections:
        return {
            'alert': False,
            'reason': 'No detections found',
            'confidence': 0.0
        }
    
    # ETAPA B: Verificación
    verified_detections = []
    
    for detection in detections:
        # Extraer crop de la detección
        crop = extract_crop(image, detection.bbox, padding=0.2)
        
        # Verificar con CNN
        verification_result = verificator_cnn(crop)
        
        # Decisión por detección individual
        detector_conf = detection.confidence
        verificator_conf = verification_result.probability
        
        # Reglas de decisión combinada
        if verificator_conf > 0.7:  # Alta confianza verificador
            verified_detections.append(detection)
        elif detector_conf > 0.8 and verificator_conf > 0.3:  # Alta det, media ver
            verified_detections.append(detection)
        elif detector_conf > 0.9:  # Muy alta confianza detector (bypass)
            verified_detections.append(detection)
        # Else: rechazar detección
    
    # Decisión final del sistema
    if verified_detections:
        max_confidence = max(d.confidence for d in verified_detections)
        return {
            'alert': True,
            'detections_count': len(verified_detections),
            'max_confidence': max_confidence,
            'verified_detections': verified_detections,
            'reason': f'{len(verified_detections)} detection(s) verified'
        }
    else:
        return {
            'alert': False,
            'reason': 'All detections rejected by verificator',
            'rejected_count': len(detections)
        }
```

## ⚡ **PIPELINE DE INFERENCIA COMPLETO**

### **Modo Servidor (Centralizado)**
```python
# Pipeline optimizado para servidor con GPU
async def server_inference_pipeline(image_batch):
    """
    Pipeline de inferencia optimizado para servidor
    Procesa múltiples imágenes en paralelo
    """
    
    # Batch processing del detector
    batch_detections = await yolo_detector.predict_batch(
        image_batch, 
        conf=0.3,
        batch_size=20
    )
    
    # Extraer todos los crops para verificación
    all_crops = []
    crop_metadata = []
    
    for img_idx, detections in enumerate(batch_detections):
        for det_idx, detection in enumerate(detections):
            crop = extract_crop(image_batch[img_idx], detection.bbox)
            all_crops.append(crop)
            crop_metadata.append({
                'image_idx': img_idx,
                'detection_idx': det_idx,
                'original_detection': detection
            })
    
    # Batch verification de todos los crops
    if all_crops:
        verification_results = await verificator_cnn.predict_batch(
            all_crops,
            batch_size=64
        )
        
        # Combinar resultados
        final_results = combine_results(
            crop_metadata, 
            verification_results
        )
    else:
        final_results = [{'alert': False}] * len(image_batch)
    
    return final_results

# Performance esperado:
# - 20 imágenes/batch en A100
# - ~400ms tiempo total por batch
# - ~20 imágenes/segundo throughput
```

### **Modo Edge (Distribuido)**
```python
# Pipeline simplificado para Raspberry Pi
def edge_inference_pipeline(image):
    """
    Pipeline liviano para edge devices
    Solo clasificación, detección en servidor
    """
    
    # Pre-filtro ligero local
    edge_classification = mobilenet_classifier(image)
    
    if edge_classification.probability > 0.4:  # Threshold conservativo
        # Enviar al servidor para procesamiento completo
        server_result = await send_to_server(image)
        return server_result
    else:
        return {'alert': False, 'source': 'edge_filter'}

# Performance esperado:
# - ~200ms por imagen en RPi 4
# - Reducción 60-80% tráfico a servidor
# - Autonomía local para filtrado básico
```

## 📈 **MÉTRICAS DE PERFORMANCE ESPERADAS**

### **Métricas del Sistema Completo**

#### **Precisión y Recall**
```
Configuración Actual (Solo Detector):
├── Recall: 95% ✅ (excelente)
├── Precision: 60% ⚠️ (mejorable)
├── F1-Score: 0.74
└── False Positive Rate: 40%

Configuración Objetivo (Detector + Verificador):
├── Recall: 92% ✅ (mantener alto)
├── Precision: 85% ✅ (mejora +25 puntos)
├── F1-Score: 0.88 ✅ (+0.14 mejora)
└── False Positive Rate: 8% ✅ (-32 puntos)
```

#### **Latencia y Throughput**
```
Servidor A100:
├── Detector: 15ms/imagen
├── Verificador: 5ms/crop (promedio 2 crops/imagen)
├── Latencia Total: ~25ms/imagen
├── Throughput: 40 imágenes/segundo
└── Capacidad: 144,000 imágenes/hora

Servidor RTX 3090:
├── Detector: 25ms/imagen  
├── Verificador: 8ms/crop
├── Latencia Total: ~40ms/imagen
├── Throughput: 25 imágenes/segundo
└── Capacidad: 90,000 imágenes/hora

Edge (Raspberry Pi 4):
├── Clasificador: 200ms/imagen
├── Throughput: 5 imágenes/segundo
├── Capacidad: 18,000 imágenes/hora
└── Tráfico reducido: 70% menos envíos
```

## 🚀 **PLAN DE IMPLEMENTACIÓN**

### **Fase 1: MVP Servidor (ACTUAL)**
- ✅ **Detector YOLOv8-s entrenado y desplegado**
- ⏳ **Verificador CNN en entrenamiento** 
- ⏳ **Integración Detector + Verificador**
- ⏳ **Sistema de notificaciones básico**

### **Fase 2: Optimización Servidor**
- 📋 **Optimización de throughput batch**
- 📋 **Sistema de caching inteligente**  
- 📋 **API REST completa**
- 📋 **Dashboard de monitoreo**

### **Fase 3: Despliegue Edge** 
- 📋 **Modelo edge optimizado (TFLite)**
- 📋 **Sistema de comunicación MQTT**
- 📋 **Sincronización automática**
- 📋 **Monitoreo distribuido**

### **Fase 4: Producción Completa**
- 📋 **Balanceador de carga multi-GPU**
- 📋 **Sistema de alertas escalable**  
- 📋 **Analytics y reportes**
- 📋 **Mantenimiento automático**

## 🛠️ **SCRIPTS DE DESPLIEGUE**

### **Script de Entrenamiento Completo**
```bash
#!/bin/bash
# train_sai_complete.sh - Entrena todo el sistema SAI

echo "🔥 Entrenando Sistema SAI Completo"

# Etapa A: Detector (Si no está entrenado)
if [ ! -f "RNA/training/runs/sai_detector_training/weights/best.pt" ]; then
    echo "🎯 Entrenando Detector YOLOv8-s..."
    ./start_detector_training_optimized.sh
fi

# Crear dataset del verificador
echo "📊 Creando dataset del verificador..."
python3 RNA/scripts/create_verificator_dataset.py \
    --yolo-dataset RNA/data/mega_fire_dataset \
    --output RNA/data/verificator_dataset

# Etapa B: Verificador  
echo "🔍 Entrenando Verificador CNN..."
python3 RNA/scripts/train_verificator.py \
    --dataset RNA/data/verificator_dataset \
    --output-dir RNA/training/runs/verificator \
    --gpu-optimized

echo "✅ Sistema SAI entrenado completamente"
```

### **Script de Inferencia Integrada**
```python
#!/usr/bin/env python3
# sai_inference.py - Inferencia completa del sistema SAI

import torch
from ultralytics import YOLO
from RNA.scripts.train_verificator import SmokeyNetLite

class SAISystem:
    def __init__(self, detector_path, verificator_path):
        # Cargar modelos
        self.detector = YOLO(detector_path)
        
        # Cargar verificador
        checkpoint = torch.load(verificator_path)
        self.verificator = SmokeyNetLite(
            backbone=checkpoint['config']['backbone']
        )
        self.verificator.load_state_dict(checkpoint['model_state_dict'])
        self.verificator.eval()
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.verificator.to(self.device)
    
    def predict(self, image_path):
        """Predicción completa SAI"""
        
        # Etapa A: Detectar
        results = self.detector(image_path, conf=0.3)
        
        if not results[0].boxes:
            return {'alert': False, 'reason': 'No detections'}
        
        # Etapa B: Verificar cada detección
        verified_detections = []
        
        for box in results[0].boxes:
            # Extraer crop
            crop = self.extract_crop(image_path, box)
            
            # Verificar
            with torch.no_grad():
                crop_tensor = self.preprocess_crop(crop)
                output = self.verificator(crop_tensor)
                prob = torch.softmax(output, dim=1)[0][0]  # true_detection prob
                
                if prob > 0.5:
                    verified_detections.append({
                        'bbox': box.xyxy[0].tolist(),
                        'confidence': float(box.conf[0]),
                        'verification_prob': float(prob),
                        'class': int(box.cls[0])
                    })
        
        # Decisión final
        if verified_detections:
            return {
                'alert': True,
                'detections': verified_detections,
                'count': len(verified_detections)
            }
        else:
            return {
                'alert': False, 
                'reason': 'All detections rejected by verificator'
            }

if __name__ == "__main__":
    # Inicializar sistema
    sai = SAISystem(
        detector_path="RNA/training/runs/sai_detector_training/weights/best.pt",
        verificator_path="RNA/training/runs/verificator/verificator_best.pt"
    )
    
    # Predecir
    result = sai.predict("test_image.jpg")
    print(f"🔥 SAI Result: {result}")
```

## 📊 **MONITOREO Y ANÁLISIS**

### **Métricas de Producción**
```python
# Métricas que se monitorean en producción
production_metrics = {
    'performance': {
        'detector_inference_time_ms': [],
        'verificator_inference_time_ms': [],
        'total_pipeline_time_ms': [],
        'throughput_images_per_second': [],
        'gpu_utilization_percent': []
    },
    'accuracy': {
        'true_positives_count': 0,
        'false_positives_count': 0,
        'false_negatives_count': 0,
        'precision': 0.0,
        'recall': 0.0,
        'f1_score': 0.0
    },
    'system': {
        'memory_usage_gb': [],
        'cpu_usage_percent': [],
        'disk_usage_gb': [],
        'network_bandwidth_mbps': []
    }
}
```

### **Dashboard de Monitoreo**
- **Grafana** para visualización de métricas en tiempo real
- **Prometheus** para recolección de métricas
- **Alertmanager** para notificaciones de sistema
- **Logs centralizados** con ELK Stack

## 🔒 **CONSIDERACIONES DE SEGURIDAD**

### **Seguridad del Modelo**
- **Encriptación** de modelos en disco
- **Autenticación** para APIs de inferencia  
- **Rate limiting** para prevenir ataques
- **Validación** de inputs para evitar adversarial attacks

### **Seguridad de Datos**
- **Encriptación** de comunicaciones (TLS 1.3)
- **Anonimización** de metadatos de imágenes
- **Retention policies** para logs y datos
- **Backup** automático de modelos y configuraciones

## 🎯 **OBJETIVOS DE RENDIMIENTO FINAL**

```
🏆 OBJETIVOS SAI SISTEMA COMPLETO
===================================

Precisión:
├── Recall: ≥92% (mantener detección incendios reales)
├── Precision: ≥85% (reducir falsos positivos críticos)  
├── F1-Score: ≥0.88 (balance óptimo)
└── Uptime: ≥99.5% (alta disponibilidad)

Performance:
├── Latencia: <50ms por imagen (servidor)
├── Throughput: ≥25 img/s (RTX 3090) / ≥40 img/s (A100)
├── Escalabilidad: 100+ cámaras simultáneas
└── Eficiencia: <300W consumo total

Operación:
├── Tiempo de respuesta: <2 minutos alerta
├── Falsos positivos: <5 por día por cámara
├── Mantenimiento: <4 horas/mes downtime
└── Actualizaciones: Sin interrución servicio
```

---

**🔥 Esta arquitectura completa establece el framework técnico para implementar el sistema SAI más avanzado de detección de incendios con IA, combinando máxima precisión con eficiencia operacional.**

## 📞 **PRÓXIMOS PASOS INMEDIATOS**

1. **Completar entrenamiento detector A100** (en progreso - ~2 horas restantes)
2. **Crear dataset verificador** usando modelo entrenado
3. **Entrenar verificador CNN** (estimado 2-4 horas A100)
4. **Integrar ambos modelos** en pipeline unificado  
5. **Desplegar sistema completo** en servidor de producción

**Estado Actual**: ✅ Detector 95% completo, Verificador 100% preparado para entrenar