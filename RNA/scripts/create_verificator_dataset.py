#!/usr/bin/env python3
"""
SAI Etapa B - Creación del Dataset del Verificador CNN
Extrae crops de bounding boxes del dataset YOLO para entrenar el verificador

Uso:
    python3 RNA/scripts/create_verificator_dataset.py
"""

import os
import sys
import cv2
import yaml
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
import shutil
from datetime import datetime
import argparse

class VerificatorDatasetCreator:
    def __init__(self, config):
        self.config = config
        self.yolo_dataset_path = Path(config['yolo_dataset_path'])
        self.output_path = Path(config['output_path'])
        self.crop_size = config['crop_size']
        self.padding_ratio = config['padding_ratio']
        
        # Estadísticas
        self.stats = {
            'true_fire_crops': 0,
            'true_smoke_crops': 0,
            'false_positive_crops': 0,
            'total_processed_images': 0,
            'skipped_small_boxes': 0,
            'processing_errors': 0
        }
        
    def create_directory_structure(self):
        """Crear estructura de directorios para el dataset del verificador"""
        directories = [
            self.output_path / 'images' / 'train' / 'true_fire',
            self.output_path / 'images' / 'train' / 'true_smoke', 
            self.output_path / 'images' / 'train' / 'false_positive',
            self.output_path / 'images' / 'val' / 'true_fire',
            self.output_path / 'images' / 'val' / 'true_smoke',
            self.output_path / 'images' / 'val' / 'false_positive',
            self.output_path / 'logs'
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            
        print(f"✅ Estructura de directorios creada en: {self.output_path}")
    
    def load_yolo_dataset_config(self):
        """Cargar configuración del dataset YOLO original"""
        dataset_yaml = self.yolo_dataset_path / 'dataset.yaml'
        
        if not dataset_yaml.exists():
            raise FileNotFoundError(f"No se encuentra dataset.yaml en {dataset_yaml}")
            
        with open(dataset_yaml, 'r') as f:
            dataset_config = yaml.safe_load(f)
            
        return dataset_config
    
    def extract_crop_from_yolo_box(self, image, yolo_box, img_h, img_w):
        """
        Extraer crop de imagen usando coordenadas YOLO
        
        Args:
            image: Imagen OpenCV (numpy array)
            yolo_box: [class, x_center, y_center, width, height] en formato YOLO (0-1)
            img_h, img_w: Dimensiones de la imagen original
            
        Returns:
            crop: Imagen recortada de tamaño crop_size x crop_size
            success: Boolean indicando si el crop fue exitoso
        """
        try:
            class_id, x_center, y_center, box_width, box_height = yolo_box
            
            # Convertir a coordenadas absolutas
            x_center_abs = int(x_center * img_w)
            y_center_abs = int(y_center * img_h)
            box_width_abs = int(box_width * img_w)
            box_height_abs = int(box_height * img_h)
            
            # Calcular coordenadas de la bounding box
            x1 = x_center_abs - box_width_abs // 2
            y1 = y_center_abs - box_height_abs // 2
            x2 = x_center_abs + box_width_abs // 2
            y2 = y_center_abs + box_height_abs // 2
            
            # Añadir padding
            padding = int(max(box_width_abs, box_height_abs) * self.padding_ratio)
            x1_padded = max(0, x1 - padding)
            y1_padded = max(0, y1 - padding)
            x2_padded = min(img_w, x2 + padding)
            y2_padded = min(img_h, y2 + padding)
            
            # Extraer crop
            crop = image[y1_padded:y2_padded, x1_padded:x2_padded]
            
            # Verificar tamaño mínimo
            if crop.shape[0] < 32 or crop.shape[1] < 32:
                return None, False
                
            # Redimensionar a tamaño objetivo
            crop_resized = cv2.resize(crop, (self.crop_size, self.crop_size))
            
            return crop_resized, True
            
        except Exception as e:
            print(f"Error extrayendo crop: {e}")
            return None, False
    
    def process_yolo_annotations(self, split='train'):
        """
        Procesar anotaciones YOLO y extraer crops
        
        Args:
            split: 'train' o 'val'
        """
        print(f"\n🔥 Procesando split: {split}")
        
        # Rutas
        images_path = self.yolo_dataset_path / 'images' / split
        labels_path = self.yolo_dataset_path / 'labels' / split
        
        if not images_path.exists() or not labels_path.exists():
            print(f"⚠️ Split {split} no encontrado, saltando...")
            return
        
        # Obtener lista de imágenes
        image_files = list(images_path.glob('*.jpg')) + list(images_path.glob('*.png'))
        
        print(f"📊 Encontradas {len(image_files)} imágenes en {split}")
        
        for image_file in tqdm(image_files, desc=f"Procesando {split}"):
            try:
                # Cargar imagen
                image = cv2.imread(str(image_file))
                if image is None:
                    print(f"❌ Error cargando imagen: {image_file}")
                    self.stats['processing_errors'] += 1
                    continue
                    
                img_h, img_w = image.shape[:2]
                self.stats['total_processed_images'] += 1
                
                # Buscar archivo de anotaciones correspondiente
                label_file = labels_path / f"{image_file.stem}.txt"
                
                if not label_file.exists():
                    # Imagen sin anotaciones - potencial falso positivo
                    continue
                
                # Leer anotaciones YOLO
                with open(label_file, 'r') as f:
                    annotations = f.readlines()
                
                # Procesar cada anotación
                for i, annotation in enumerate(annotations):
                    parts = list(map(float, annotation.strip().split()))
                    if len(parts) != 5:
                        continue
                        
                    class_id, x_center, y_center, width, height = parts
                    
                    # Extraer crop
                    crop, success = self.extract_crop_from_yolo_box(
                        image, parts, img_h, img_w
                    )
                    
                    if not success:
                        self.stats['skipped_small_boxes'] += 1
                        continue
                    
                    # Determinar categoría según clase YOLO
                    if int(class_id) == 0:  # fire
                        category = 'true_fire'
                        self.stats['true_fire_crops'] += 1
                    elif int(class_id) == 1:  # smoke  
                        category = 'true_smoke'
                        self.stats['true_smoke_crops'] += 1
                    else:
                        continue  # Clase no reconocida
                    
                    # Guardar crop
                    crop_filename = f"{image_file.stem}_crop_{i}_{category}.jpg"
                    output_dir = self.output_path / 'images' / split / category
                    crop_path = output_dir / crop_filename
                    
                    cv2.imwrite(str(crop_path), crop)
                    
            except Exception as e:
                print(f"❌ Error procesando {image_file}: {e}")
                self.stats['processing_errors'] += 1
                continue
    
    def generate_false_positives_from_negatives(self, split='train'):
        """
        Generar falsos positivos ejecutando el detector en imágenes sin fuego
        
        Para implementar después del entrenamiento del detector
        """
        print(f"\n🚫 Generación de falsos positivos para {split}")
        print("⏳ Esta función se implementará después del entrenamiento del detector YOLOv8")
        
        # TODO: Implementar después del entrenamiento
        # 1. Cargar modelo YOLOv8 entrenado
        # 2. Ejecutar sobre dataset de imágenes landscape sin fuego
        # 3. Extraer crops de detecciones falsas
        # 4. Guardar como false_positive
        
        # Por ahora, crear algunos falsos positivos sintéticos para testing
        self.create_synthetic_false_positives(split)
    
    def create_synthetic_false_positives(self, split='train'):
        """Crear falsos positivos sintéticos para testing inicial"""
        
        # Buscar imágenes que no tengan anotaciones (backgrounds)
        images_path = self.yolo_dataset_path / 'images' / split
        labels_path = self.yolo_dataset_path / 'labels' / split
        
        image_files = list(images_path.glob('*.jpg')) + list(images_path.glob('*.png'))
        
        fp_count = 0
        target_fp_count = min(1000, len(image_files) // 4)  # 25% de falsos positivos
        
        for image_file in tqdm(image_files[:target_fp_count * 4], desc=f"Creando FP sintéticos {split}"):
            try:
                label_file = labels_path / f"{image_file.stem}.txt"
                
                # Solo usar imágenes sin anotaciones o con pocas
                if label_file.exists():
                    with open(label_file, 'r') as f:
                        annotations = f.readlines()
                    if len(annotations) > 2:  # Saltar imágenes con muchas detecciones
                        continue
                
                # Cargar imagen
                image = cv2.imread(str(image_file))
                if image is None:
                    continue
                    
                img_h, img_w = image.shape[:2]
                
                # Extraer crops aleatorios que podrían ser falsos positivos
                for _ in range(np.random.randint(1, 4)):  # 1-3 crops por imagen
                    # Coordenadas aleatorias
                    crop_size_px = np.random.randint(100, 300)
                    x = np.random.randint(0, max(1, img_w - crop_size_px))
                    y = np.random.randint(0, max(1, img_h - crop_size_px))
                    
                    # Extraer crop
                    crop = image[y:y+crop_size_px, x:x+crop_size_px]
                    
                    if crop.shape[0] < 50 or crop.shape[1] < 50:
                        continue
                        
                    # Redimensionar
                    crop_resized = cv2.resize(crop, (self.crop_size, self.crop_size))
                    
                    # Guardar como falso positivo
                    fp_filename = f"{image_file.stem}_fp_{fp_count}.jpg"
                    output_dir = self.output_path / 'images' / split / 'false_positive'
                    fp_path = output_dir / fp_filename
                    
                    cv2.imwrite(str(fp_path), crop_resized)
                    
                    fp_count += 1
                    self.stats['false_positive_crops'] += 1
                    
                    if fp_count >= target_fp_count:
                        break
                        
                if fp_count >= target_fp_count:
                    break
                    
            except Exception as e:
                continue
        
        print(f"✅ Creados {fp_count} falsos positivos sintéticos para {split}")
    
    def balance_dataset(self):
        """Balancear el dataset según la estrategia definida"""
        print("\n⚖️ Balanceando dataset...")
        
        for split in ['train', 'val']:
            split_path = self.output_path / 'images' / split
            
            # Contar archivos por categoría
            true_fire_count = len(list((split_path / 'true_fire').glob('*.jpg')))
            true_smoke_count = len(list((split_path / 'true_smoke').glob('*.jpg')))
            false_positive_count = len(list((split_path / 'false_positive').glob('*.jpg')))
            
            print(f"\n📊 {split.upper()} - Conteos actuales:")
            print(f"   True Fire: {true_fire_count}")
            print(f"   True Smoke: {true_smoke_count}")
            print(f"   False Positive: {false_positive_count}")
            
            # Calcular target según estrategia (25%, 25%, 50%)
            total_positives = true_fire_count + true_smoke_count
            target_fp = total_positives  # 50% del total serán FP
            
            print(f"\n🎯 Target balance:")
            print(f"   True Fire: {true_fire_count} (25%)")
            print(f"   True Smoke: {true_smoke_count} (25%)")
            print(f"   False Positive: {target_fp} (50%)")
            
            # Si necesitamos más falsos positivos, crear más sintéticos
            if false_positive_count < target_fp:
                needed_fp = target_fp - false_positive_count
                print(f"⚠️ Necesitamos {needed_fp} falsos positivos más en {split}")
    
    def create_dataset_yaml(self):
        """Crear archivo de configuración del dataset del verificador"""
        dataset_config = {
            'path': str(self.output_path.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'nc': 3,  # Número de clases
            'names': {
                0: 'true_fire',
                1: 'true_smoke', 
                2: 'false_positive'
            },
            'dataset_info': {
                'created_date': datetime.now().isoformat(),
                'source_dataset': str(self.yolo_dataset_path),
                'crop_size': self.crop_size,
                'padding_ratio': self.padding_ratio,
                'purpose': 'SAI Verificator CNN training',
                'statistics': self.stats
            }
        }
        
        yaml_path = self.output_path / 'dataset.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(dataset_config, f, default_flow_style=False)
        
        print(f"✅ Dataset config guardado: {yaml_path}")
    
    def create_statistics_report(self):
        """Crear reporte de estadísticas del dataset creado"""
        report = {
            'dataset_creation_report': {
                'timestamp': datetime.now().isoformat(),
                'source_dataset': str(self.yolo_dataset_path),
                'output_dataset': str(self.output_path),
                'configuration': self.config,
                'statistics': self.stats
            }
        }
        
        # Contar archivos finales
        final_counts = {}
        for split in ['train', 'val']:
            final_counts[split] = {}
            split_path = self.output_path / 'images' / split
            
            for category in ['true_fire', 'true_smoke', 'false_positive']:
                count = len(list((split_path / category).glob('*.jpg')))
                final_counts[split][category] = count
        
        report['dataset_creation_report']['final_counts'] = final_counts
        
        # Guardar reporte
        report_path = self.output_path / 'logs' / 'dataset_creation_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Mostrar resumen
        print("\n📊 REPORTE FINAL DEL DATASET VERIFICADOR")
        print("=" * 50)
        print(f"📁 Dataset creado en: {self.output_path}")
        print(f"📈 Total imágenes procesadas: {self.stats['total_processed_images']}")
        print(f"🔥 Crops true fire: {self.stats['true_fire_crops']}")
        print(f"💨 Crops true smoke: {self.stats['true_smoke_crops']}")
        print(f"🚫 Crops false positive: {self.stats['false_positive_crops']}")
        print(f"⚠️ Boxes pequeñas saltadas: {self.stats['skipped_small_boxes']}")
        print(f"❌ Errores de procesamiento: {self.stats['processing_errors']}")
        
        print("\n📋 CONTEOS FINALES POR SPLIT:")
        for split, counts in final_counts.items():
            total = sum(counts.values())
            print(f"\n{split.upper()}:")
            for category, count in counts.items():
                percentage = (count / total * 100) if total > 0 else 0
                print(f"   {category}: {count} ({percentage:.1f}%)")
        
        print(f"\n📄 Reporte completo: {report_path}")
        return report_path
    
    def run(self):
        """Ejecutar el proceso completo de creación del dataset"""
        print("🚀 INICIANDO CREACIÓN DEL DATASET VERIFICADOR SAI")
        print("=" * 60)
        
        # Validar dataset YOLO original
        dataset_config = self.load_yolo_dataset_config()
        print(f"✅ Dataset YOLO cargado: {dataset_config['path']}")
        print(f"📊 Clases: {dataset_config['names']}")
        
        # Crear estructura
        self.create_directory_structure()
        
        # Procesar splits
        for split in ['train', 'val']:
            if (self.yolo_dataset_path / 'images' / split).exists():
                self.process_yolo_annotations(split)
                self.generate_false_positives_from_negatives(split)
        
        # Balance y configuración final
        self.balance_dataset()
        self.create_dataset_yaml()
        
        # Reporte final
        report_path = self.create_statistics_report()
        
        print("\n🎉 DATASET VERIFICADOR CREADO EXITOSAMENTE")
        print("=" * 50)
        print("🔥 Listo para entrenar la Etapa B del sistema SAI")
        
        return report_path


def main():
    parser = argparse.ArgumentParser(description='Crear dataset del verificador SAI')
    parser.add_argument('--yolo-dataset', 
                       default='RNA/data/mega_fire_dataset',
                       help='Ruta al dataset YOLO original')
    parser.add_argument('--output', 
                       default='RNA/data/verificator_dataset',
                       help='Ruta de salida del dataset verificador')
    parser.add_argument('--crop-size', type=int, default=224,
                       help='Tamaño de los crops cuadrados')
    parser.add_argument('--padding-ratio', type=float, default=0.2,
                       help='Ratio de padding alrededor de bounding boxes')
    
    args = parser.parse_args()
    
    # Configuración
    config = {
        'yolo_dataset_path': args.yolo_dataset,
        'output_path': args.output,
        'crop_size': args.crop_size,
        'padding_ratio': args.padding_ratio
    }
    
    # Crear dataset
    creator = VerificatorDatasetCreator(config)
    report_path = creator.run()
    
    print(f"\n✅ Proceso completado. Reporte: {report_path}")


if __name__ == "__main__":
    main()