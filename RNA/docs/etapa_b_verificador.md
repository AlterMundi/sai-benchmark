# SAI Etapa B - Verificador CNN: Documentación Técnica

## 🎯 **OBJETIVO**
Entrenar el segundo modelo del sistema SAI: un verificador CNN que confirma o rechaza las detecciones del modelo detector YOLOv8-s, reduciendo falsos positivos y aumentando la confianza del sistema.

## 📋 **ARQUITECTURA DEL SISTEMA SAI COMPLETO**

### **Flujo de Procesamiento**
```
Imagen → [ETAPA A: Detector YOLOv8-s] → Bounding Boxes
                    ↓
Crops de ROIs → [ETAPA B: Verificador CNN] → Confianza de Verificación
                    ↓
         Decisión Final: ALERTA/NO ALERTA
```

### **Etapa A - Detector (ACTUAL)**
- **Modelo**: YOLOv8-s
- **Input**: Imagen completa 1440x808
- **Output**: Bounding boxes + clasificación [fire, smoke]
- **Función**: Detectar regiones sospechosas con alto recall
- **Estado**: ✅ **ENTRENANDO EN A100**

### **Etapa B - Verificador (SIGUIENTE)**
- **Modelo**: CNN Clasificador
- **Input**: Crops de bounding boxes (224x224)
- **Output**: Probabilidad [true_fire, false_positive]
- **Función**: Verificar detecciones y reducir falsos positivos
- **Estado**: ⏳ **PENDIENTE**

## 🏗️ **ARQUITECTURA DEL VERIFICADOR**

### **Diseño Propuesto: SmokeyNet-Lite**
Basado en la arquitectura probada de SmokeyNet pero simplificada:

```python
# Arquitectura del Verificador
Input: Crop 224x224 (de bounding box del detector)
↓
Backbone: EfficientNet-B0 o ResNet-34
↓
Módulo Temporal: LSTM (2-3 frames consecutivos)
↓
Head: Clasificación binaria [true_detection, false_positive]
↓
Output: Confianza de verificación [0.0 - 1.0]
```

### **Especificaciones Técnicas**
- **Input Size**: 224x224x3 (crops de ROIs del detector)
- **Backbone**: EfficientNet-B0 (7.8M params) o ResNet-34 (21M params)
- **Temporal Window**: 2-3 frames consecutivos para robustez
- **Output**: Probabilidad sigmoid [0-1]
- **Threshold**: 0.5 (ajustable según métricas de validación)

### **Criterio de Decisión Final**
```python
# Lógica de decisión combinada
detector_conf = yolo_detection.confidence  # De Etapa A
verificator_conf = cnn_verification.probability  # De Etapa B

# Reglas de combinación
if detector_conf > 0.4 and verificator_conf > 0.5:
    decision = "FIRE_ALERT"
elif detector_conf > 0.6:  # Alta confianza del detector
    decision = "FIRE_ALERT"  # Bypass verificador
else:
    decision = "NO_ALERT"
```

## 📊 **DATASET PARA EL VERIFICADOR**

### **Fuentes de Datos**

#### **Positivos Verdaderos**
1. **Crops del dataset original**: Extraer ROIs de fire/smoke de FASDD, D-Fire, NEMO, PyroNear
2. **Detecciones del YOLOv8**: Usar crops confirmados como verdaderos positivos
3. **FIgLib temporal**: Crops de secuencias verificadas de humo real

#### **Falsos Positivos (CRÍTICO)**
1. **Nubes**: Crops de nubes confundidas con humo
2. **Vapores**: Vapor de agua, niebla, bruma
3. **Reflejos**: Reflejos solares, iluminación artificial
4. **Polvo**: Polvo en movimiento, tierra levantada
5. **Objetos estáticos**: Chimeneas, estructuras, vegetación

### **Estructura del Dataset Verificador**
```
RNA/data/verificador_dataset/
├── images/
│   ├── train/
│   │   ├── true_fire/        # Crops reales de fuego
│   │   ├── true_smoke/       # Crops reales de humo  
│   │   └── false_positive/   # Crops de falsos positivos
│   └── val/
│       ├── true_fire/
│       ├── true_smoke/
│       └── false_positive/
└── dataset.yaml             # Configuración para entrenamiento
```

### **Estrategia de Balanceado**
- **True Fire**: 25%
- **True Smoke**: 25%
- **False Positives**: 50% (crítico para reducir falsos positivos)

## 🛠️ **PIPELINE DE CREACIÓN DEL DATASET**

### **Paso 1: Extraer Crops del Dataset Detector**
```python
# Script: extract_crops_for_verificator.py
def extract_crops_from_yolo_dataset():
    """
    Extraer crops de 224x224 de las imágenes del dataset YOLO
    usando las anotaciones de bounding boxes
    """
    # Leer annotations YOLO format
    # Extraer crops con padding
    # Guardar en estructura clasificación
    # Balancear clases true/false
```

### **Paso 2: Generar Falsos Positivos**
```python
# Script: generate_false_positives.py
def generate_hard_negatives():
    """
    Ejecutar YOLOv8-s entrenado sobre dataset de imágenes sin fuego
    para capturar falsos positivos típicos (nubes, vapores, etc.)
    """
    # Dataset de imágenes landscape sin fuego
    # Ejecutar detector entrenado
    # Extraer crops de detecciones falsas
    # Etiquetar como false_positive
```

### **Paso 3: Augmentación Específica**
```python
# Augmentaciones para el verificador
augmentations = {
    'color_jitter': True,      # Variaciones de iluminación
    'gaussian_blur': True,     # Simular humo difuso
    'motion_blur': True,       # Simular movimiento
    'weather_effects': True,   # Lluvia, niebla artificial
    'contrast_variation': True # Condiciones de luz variables
}
```

## 🚀 **SCRIPTS DE ENTRENAMIENTO**

### **Script Principal: train_verificator.py**
```python
#!/usr/bin/env python3
"""
SAI Etapa B - Entrenamiento del Verificador CNN
Entrena el modelo de verificación que confirma/rechaza detecciones del YOLOv8
"""

import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader
import timm  # Para EfficientNet

class SmokeyNetLite(nn.Module):
    """Verificador CNN basado en SmokeyNet simplificado"""
    
    def __init__(self, backbone='efficientnet_b0', num_classes=1):
        super().__init__()
        
        # Backbone pre-entrenado
        if backbone == 'efficientnet_b0':
            self.backbone = timm.create_model('efficientnet_b0', 
                                            pretrained=True, 
                                            num_classes=0)  # Remove classifier
            backbone_features = 1280
        elif backbone == 'resnet34':
            self.backbone = models.resnet34(pretrained=True)
            backbone_features = 512
            
        # Módulo temporal (LSTM simple)
        self.lstm = nn.LSTM(backbone_features, 256, 
                           batch_first=True, bidirectional=True)
        
        # Clasificador final
        self.classifier = nn.Sequential(
            nn.Linear(512, 128),  # 256*2 (bidirectional)
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # x shape: (batch, frames, channels, height, width)
        batch_size, frames = x.size(0), x.size(1)
        
        # Procesar cada frame
        x = x.view(-1, *x.shape[2:])  # (batch*frames, C, H, W)
        features = self.backbone(x)   # (batch*frames, features)
        
        # Reshape para LSTM
        features = features.view(batch_size, frames, -1)  # (batch, frames, features)
        
        # Módulo temporal
        lstm_out, _ = self.lstm(features)  # (batch, frames, 512)
        
        # Usar último frame para decisión
        final_features = lstm_out[:, -1, :]  # (batch, 512)
        
        # Clasificación
        output = self.classifier(final_features)
        return output

def train_verificator():
    """Función principal de entrenamiento"""
    
    # Configuración
    config = {
        'batch_size': 32,
        'epochs': 50,
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'patience': 10,  # Early stopping
        'backbone': 'efficientnet_b0',
        'num_frames': 2  # Ventana temporal
    }
    
    # Data loaders
    train_loader = create_train_dataloader(config)
    val_loader = create_val_dataloader(config)
    
    # Modelo
    model = SmokeyNetLite(backbone=config['backbone'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Optimización
    criterion = nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), 
                                 lr=config['learning_rate'],
                                 weight_decay=config['weight_decay'])
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5, verbose=True
    )
    
    # Training loop
    best_f1 = 0.0
    patience_counter = 0
    
    for epoch in range(config['epochs']):
        # Entrenamiento
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validación
        val_loss, val_metrics = validate_epoch(model, val_loader, criterion, device)
        
        # Métricas
        precision = val_metrics['precision']
        recall = val_metrics['recall'] 
        f1_score = val_metrics['f1']
        
        print(f"Epoch {epoch+1}/{config['epochs']}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1_score:.4f}")
        
        # Early stopping y checkpoint
        if f1_score > best_f1:
            best_f1 = f1_score
            patience_counter = 0
            
            # Guardar mejor modelo
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': config,
                'metrics': val_metrics,
                'epoch': epoch
            }, 'RNA/training/runs/verificator_best.pt')
            
        else:
            patience_counter += 1
            
        if patience_counter >= config['patience']:
            print(f"Early stopping at epoch {epoch+1}")
            break
            
        scheduler.step(val_loss)

if __name__ == "__main__":
    train_verificator()
```

### **Script de Evaluación: evaluate_verificator.py**
```python
#!/usr/bin/env python3
"""
Evaluación del verificador CNN con métricas específicas
"""

def evaluate_verificator_performance():
    """
    Evaluar el verificador con métricas críticas:
    - Reducción de falsos positivos
    - Mantenimiento de recall para verdaderos positivos
    - Análisis por tipo de falso positivo
    """
    
    metrics = {
        'false_positive_reduction': 0.0,  # % reducción FP
        'true_positive_retention': 0.0,   # % mantenimiento TP  
        'precision_improvement': 0.0,     # Mejora en precisión
        'inference_time': 0.0,            # ms por crop
        'memory_usage': 0.0               # MB de VRAM
    }
    
    # Análisis detallado por categorías
    category_analysis = {
        'clouds_rejection': 0.0,      # % nubes rechazadas
        'vapor_rejection': 0.0,       # % vapores rechazados  
        'glare_rejection': 0.0,       # % reflejos rechazados
        'fire_retention': 0.0,        # % fuegos mantenidos
        'smoke_retention': 0.0        # % humo mantenido
    }
    
    return metrics, category_analysis
```

## ⚡ **OPTIMIZACIONES PARA A100**

### **Configuración Optimizada**
```python
# Configuración específica para A100
a100_config = {
    'batch_size': 64,           # Aprovechar VRAM de 40GB
    'mixed_precision': True,    # AMP para velocidad  
    'num_workers': 16,          # Paralelización I/O
    'pin_memory': True,         # Transferencia GPU optimizada
    'compile_model': True,      # torch.compile() en PyTorch 2.0+
    'gradient_accumulation': 4, # Batch efectivo de 256
}
```

### **Estimaciones de Performance A100**
- **Tiempo de entrenamiento**: 2-4 horas (vs 15-20 horas RTX 3090)
- **Throughput**: ~500 crops/segundo en inferencia
- **VRAM usage**: ~15GB para batch_size=64
- **Dataset size**: ~50K crops (balanceados)

## 📈 **MÉTRICAS DE ÉXITO**

### **Objetivos del Verificador**
1. **Reducción FP**: ≥70% reducción en falsos positivos
2. **Retención TP**: ≥95% mantenimiento de verdaderos positivos  
3. **Precisión sistema**: Subir de ~60% a ≥85%
4. **Latencia**: <50ms por crop en A100
5. **F1-Score**: ≥0.80 en dataset de validación

### **Métricas de Monitoreo**
```python
verification_metrics = {
    'detector_precision': 0.60,      # Baseline YOLOv8-s solo
    'verificator_precision': 0.85,   # Target con verificador
    'system_recall': 0.92,           # Mantener alto recall
    'false_positive_rate': 0.05,     # <5% FP rate final
    'processing_latency': 45.0,      # ms total (detector + verificator)
}
```

## 🎯 **PRÓXIMOS PASOS**

### **Inmediatos (mientras termina Etapa A)**
1. **Crear script de extracción de crops** del dataset actual
2. **Generar dataset de falsos positivos** ejecutando YOLOv8 en imágenes sin fuego
3. **Preparar pipeline de entrenamiento** del verificador
4. **Configurar métricas de evaluación** específicas

### **Al completar Etapa A**
1. **Usar modelo YOLOv8 entrenado** para generar crops de entrenamiento
2. **Entrenar verificador** en A100 (2-4 horas estimadas)
3. **Evaluar sistema completo** (Detector + Verificador)
4. **Optimizar umbrales** de decisión combinada
5. **Desplegar sistema híbrido** en producción

## 🔄 **INTEGRACIÓN CON ETAPA A**

### **Flujo de Inferencia Completo**
```python
# Pseudocódigo del sistema completo
def sai_fire_detection(image):
    # Etapa A: Detección
    detections = yolo_detector(image)
    
    if not detections:
        return "NO_FIRE"
    
    # Etapa B: Verificación
    verified_detections = []
    for detection in detections:
        crop = extract_crop(image, detection.bbox)
        confidence = cnn_verificator(crop)
        
        if confidence > VERIFICATION_THRESHOLD:
            verified_detections.append(detection)
    
    # Decisión final
    if verified_detections:
        return f"FIRE_ALERT: {len(verified_detections)} detections verified"
    else:
        return "NO_FIRE: Detections rejected by verificator"
```

---

**🔥 Esta documentación establece la base completa para entrenar la Etapa B del sistema SAI una vez que termine el entrenamiento actual del detector YOLOv8-s en el A100.**