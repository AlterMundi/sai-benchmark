#!/usr/bin/env python3
"""
SAI Completar Dataset Verificator
Generar falsos positivos y split de validación usando el detector entrenado

Proceso:
1. Cargar detector best.pt 
2. Ejecutar sobre imágenes negativas
3. Extraer crops de falsos positivos
4. Crear validation split balanceado
5. Generar dataset.yaml
"""

import os
import sys
import cv2
import numpy as np
import yaml
import json
import shutil
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from ultralytics import YOLO
import random
from typing import Dict, List, Tuple

class VerificatorDatasetCompleter:
    """Completar dataset del verificator con falsos positivos y validation split"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.detector = None
        self.stats = {
            'false_positives_generated': 0,
            'validation_split_created': 0,
            'total_samples': 0
        }
        
        print("🔧 Iniciando completado del dataset verificator")
    
    def load_detector(self):
        """Cargar detector entrenado para generar falsos positivos"""
        
        detector_path = self.config['detector_path']
        
        if not Path(detector_path).exists():
            raise FileNotFoundError(f"Detector no encontrado: {detector_path}")
        
        print(f"🔥 Cargando detector: {detector_path}")
        self.detector = YOLO(detector_path)
        print("✅ Detector cargado exitosamente")
    
    def find_background_images(self) -> List[Path]:
        """Encontrar imágenes sin fuego para generar falsos positivos"""
        
        mega_dataset = Path(self.config['mega_dataset_path'])
        images_dir = mega_dataset / 'images' / 'train'
        labels_dir = mega_dataset / 'labels' / 'train'
        
        background_images = []
        image_files = list(images_dir.glob('*.jpg'))
        
        print(f"🔍 Buscando imágenes sin fuego en {len(image_files)} total...")
        
        for img_path in tqdm(image_files, desc="Filtrando backgrounds"):
            label_path = labels_dir / f"{img_path.stem}.txt"
            
            # Si no tiene archivo de labels, es background
            if not label_path.exists():
                background_images.append(img_path)
                continue
            
            # Si tiene labels pero están vacías, también es background
            if label_path.stat().st_size == 0:
                background_images.append(img_path)
                continue
            
            # Verificar si tiene muy pocas detecciones (potencial background)
            with open(label_path, 'r') as f:
                lines = f.readlines()
                if len(lines) <= 1:  # 0 o 1 detección
                    background_images.append(img_path)
        
        print(f"✅ Encontradas {len(background_images)} imágenes de background")
        return background_images
    
    def generate_false_positives(self, background_images: List[Path], 
                                 target_count: int) -> int:
        """Generar falsos positivos ejecutando detector sobre backgrounds"""
        
        output_dir = Path(self.config['verificator_dataset_path']) / 'images' / 'train' / 'false_positive'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        false_positives_generated = 0
        confidence_range = self.config.get('fp_confidence_range', (0.3, 0.8))
        crop_size = self.config.get('crop_size', 224)
        
        print(f"🎯 Generando {target_count} falsos positivos...")
        print(f"📊 Confianza objetivo: {confidence_range[0]}-{confidence_range[1]}")
        
        # Mezclar imágenes para variety
        random.shuffle(background_images)
        
        for img_path in tqdm(background_images, desc="Generando FPs"):
            if false_positives_generated >= target_count:
                break
            
            try:
                # Ejecutar detector
                results = self.detector(str(img_path), verbose=False, conf=confidence_range[0])
                
                if len(results) == 0:
                    continue
                
                result = results[0]
                
                if result.boxes is None or len(result.boxes) == 0:
                    continue
                
                # Cargar imagen original
                image = cv2.imread(str(img_path))
                if image is None:
                    continue
                
                img_h, img_w = image.shape[:2]
                
                # Procesar detecciones como falsos positivos
                for i, box in enumerate(result.boxes):
                    if false_positives_generated >= target_count:
                        break
                    
                    confidence = float(box.conf[0])
                    
                    # Filtrar por rango de confianza (FPs realistas)
                    if not (confidence_range[0] <= confidence <= confidence_range[1]):
                        continue
                    
                    # Extraer coordenadas de la box
                    x1, y1, x2, y2 = box.xyxy[0].int().tolist()
                    
                    # Añadir padding
                    padding = int(max(x2-x1, y2-y1) * 0.2)
                    x1_pad = max(0, x1 - padding)
                    y1_pad = max(0, y1 - padding) 
                    x2_pad = min(img_w, x2 + padding)
                    y2_pad = min(img_h, y2 + padding)
                    
                    # Extraer crop
                    crop = image[y1_pad:y2_pad, x1_pad:x2_pad]
                    
                    if crop.shape[0] < 32 or crop.shape[1] < 32:
                        continue
                    
                    # Redimensionar
                    crop_resized = cv2.resize(crop, (crop_size, crop_size))
                    
                    # Guardar falso positivo
                    fp_filename = f"{img_path.stem}_fp_{i}_conf{confidence:.3f}.jpg"
                    fp_path = output_dir / fp_filename
                    
                    cv2.imwrite(str(fp_path), crop_resized)
                    false_positives_generated += 1
                    
            except Exception as e:
                print(f"⚠️ Error procesando {img_path}: {e}")
                continue
        
        print(f"✅ Generados {false_positives_generated} falsos positivos")
        self.stats['false_positives_generated'] = false_positives_generated
        return false_positives_generated
    
    def create_validation_split(self, validation_ratio: float = 0.2) -> int:
        """Crear split de validación balanceado"""
        
        dataset_root = Path(self.config['verificator_dataset_path'])
        train_dir = dataset_root / 'images' / 'train'
        val_dir = dataset_root / 'images' / 'val'
        val_dir.mkdir(parents=True, exist_ok=True)
        
        moved_samples = 0
        
        # Crear subdirectorios de validación
        for category in ['true_fire', 'true_smoke', 'false_positive']:
            (val_dir / category).mkdir(exist_ok=True)
            
            # Obtener archivos de entrenamiento
            train_category_dir = train_dir / category
            if not train_category_dir.exists():
                continue
                
            train_files = list(train_category_dir.glob('*.jpg'))
            
            if len(train_files) == 0:
                continue
            
            # Calcular número de samples para validación
            val_count = int(len(train_files) * validation_ratio)
            
            if val_count == 0:
                continue
            
            # Seleccionar archivos aleatoriamente
            random.shuffle(train_files)
            val_files = train_files[:val_count]
            
            # Mover a validación
            val_category_dir = val_dir / category
            
            for file_path in val_files:
                dest_path = val_category_dir / file_path.name
                shutil.move(str(file_path), str(dest_path))
                moved_samples += 1
            
            print(f"📊 {category}: {len(val_files)} samples → validation")
        
        print(f"✅ Split de validación creado: {moved_samples} samples")
        self.stats['validation_split_created'] = moved_samples
        return moved_samples
    
    def create_dataset_yaml(self) -> str:
        """Crear archivo de configuración dataset.yaml"""
        
        dataset_root = Path(self.config['verificator_dataset_path'])
        train_dir = dataset_root / 'images' / 'train'
        val_dir = dataset_root / 'images' / 'val'
        
        # Contar samples por categoría
        train_counts = {}
        val_counts = {}
        
        for category in ['true_fire', 'true_smoke', 'false_positive']:
            train_counts[category] = len(list((train_dir / category).glob('*.jpg')))
            val_counts[category] = len(list((val_dir / category).glob('*.jpg')))
        
        # Crear configuración
        dataset_config = {
            'path': str(dataset_root.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            
            'nc': 2,  # Número de clases: true_detection, false_positive
            'names': {
                0: 'true_detection',   # true_fire + true_smoke
                1: 'false_positive'    # false detections
            },
            
            'dataset_info': {
                'created_date': datetime.now().isoformat(),
                'version': '1.0.0',
                'purpose': 'SAI Verificator CNN Training',
                'description': 'Balanced dataset for fire detection verification',
                
                'statistics': {
                    'train': {
                        'true_fire': train_counts.get('true_fire', 0),
                        'true_smoke': train_counts.get('true_smoke', 0), 
                        'false_positive': train_counts.get('false_positive', 0),
                        'total': sum(train_counts.values())
                    },
                    'val': {
                        'true_fire': val_counts.get('true_fire', 0),
                        'true_smoke': val_counts.get('true_smoke', 0),
                        'false_positive': val_counts.get('false_positive', 0),
                        'total': sum(val_counts.values())
                    }
                },
                
                'class_balance': {
                    'true_detection_total': train_counts.get('true_fire', 0) + train_counts.get('true_smoke', 0) + val_counts.get('true_fire', 0) + val_counts.get('true_smoke', 0),
                    'false_positive_total': train_counts.get('false_positive', 0) + val_counts.get('false_positive', 0)
                },
                
                'generation_config': {
                    'detector_used': self.config['detector_path'],
                    'fp_confidence_range': self.config.get('fp_confidence_range', (0.3, 0.8)),
                    'crop_size': self.config.get('crop_size', 224),
                    'validation_ratio': 0.2
                }
            }
        }
        
        # Guardar YAML
        yaml_path = dataset_root / 'dataset.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(dataset_config, f, default_flow_style=False, indent=2)
        
        print(f"✅ Dataset config creado: {yaml_path}")
        
        # Mostrar resumen
        print("\n📊 RESUMEN FINAL DEL DATASET:")
        print("=" * 50)
        print(f"📁 Train samples: {sum(train_counts.values())}")
        print(f"📁 Val samples: {sum(val_counts.values())}")
        print(f"📁 Total samples: {sum(train_counts.values()) + sum(val_counts.values())}")
        print()
        print("Por categoría:")
        for category in ['true_fire', 'true_smoke', 'false_positive']:
            train_c = train_counts.get(category, 0)
            val_c = val_counts.get(category, 0)
            total_c = train_c + val_c
            print(f"  {category}: {total_c} (train: {train_c}, val: {val_c})")
        
        return str(yaml_path)
    
    def run_complete_pipeline(self) -> Dict:
        """Ejecutar pipeline completo de completado del dataset"""
        
        print("🚀 INICIANDO COMPLETADO DEL DATASET VERIFICATOR")
        print("=" * 60)
        
        start_time = datetime.now()
        
        # 1. Cargar detector
        self.load_detector()
        
        # 2. Encontrar imágenes de background
        background_images = self.find_background_images()
        
        if len(background_images) == 0:
            raise ValueError("No se encontraron imágenes de background")
        
        # 3. Calcular cuántos falsos positivos necesitamos
        dataset_root = Path(self.config['verificator_dataset_path'])
        train_dir = dataset_root / 'images' / 'train'
        
        current_positives = 0
        for category in ['true_fire', 'true_smoke']:
            category_dir = train_dir / category
            if category_dir.exists():
                current_positives += len(list(category_dir.glob('*.jpg')))
        
        # Target: mismo número de FPs que positivos para balance 50/50
        target_fps = int(current_positives * self.config.get('balance_ratio', 1.0))
        
        print(f"🎯 Samples positivos actuales: {current_positives}")
        print(f"🎯 Falsos positivos objetivo: {target_fps}")
        
        # 4. Generar falsos positivos
        fps_generated = self.generate_false_positives(background_images, target_fps)
        
        # 5. Crear validation split
        val_samples = self.create_validation_split()
        
        # 6. Crear dataset.yaml
        yaml_path = self.create_dataset_yaml()
        
        # 7. Reporte final
        end_time = datetime.now()
        duration = end_time - start_time
        
        result = {
            'success': True,
            'duration': str(duration),
            'stats': {
                'false_positives_generated': fps_generated,
                'validation_samples_created': val_samples,
                'background_images_available': len(background_images)
            },
            'dataset_yaml': yaml_path,
            'timestamp': end_time.isoformat()
        }
        
        print(f"\n🎉 COMPLETADO EXITOSO EN {duration}")
        print(f"💾 Dataset listo en: {self.config['verificator_dataset_path']}")
        
        return result


def main():
    """Función principal"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Completar dataset verificator SAI')
    parser.add_argument('--detector-path',
                       default='RNA/models/detector_best.pt',
                       help='Ruta al detector entrenado')
    parser.add_argument('--mega-dataset', 
                       default='RNA/data/mega_fire_dataset',
                       help='Ruta al dataset MEGA (source)')
    parser.add_argument('--verificator-dataset',
                       default='RNA/data/verificator_dataset', 
                       help='Ruta al dataset verificator')
    parser.add_argument('--balance-ratio', type=float, default=1.0,
                       help='Ratio FPs vs positivos (1.0 = 50/50)')
    parser.add_argument('--fp-conf-min', type=float, default=0.3,
                       help='Confianza mínima para FPs')
    parser.add_argument('--fp-conf-max', type=float, default=0.8,
                       help='Confianza máxima para FPs')
    
    args = parser.parse_args()
    
    # Configuración
    config = {
        'detector_path': args.detector_path,
        'mega_dataset_path': args.mega_dataset,
        'verificator_dataset_path': args.verificator_dataset,
        'balance_ratio': args.balance_ratio,
        'fp_confidence_range': (args.fp_conf_min, args.fp_conf_max),
        'crop_size': 224
    }
    
    print("⚙️ CONFIGURACIÓN:")
    for key, value in config.items():
        print(f"   {key}: {value}")
    print()
    
    # Ejecutar completado
    try:
        completer = VerificatorDatasetCompleter(config)
        result = completer.run_complete_pipeline()
        
        if result['success']:
            print("✅ Dataset verificator completado exitosamente")
            return 0
        else:
            print("❌ Error completando dataset")
            return 1
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())