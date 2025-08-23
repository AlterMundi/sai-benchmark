#!/usr/bin/env python3
"""
SAI Threshold Optimization Suite
Encuentra el threshold óptimo del verificador para maximizar recall manteniendo precision

Uso:
    python optimize_sai_threshold.py --dataset RNA/data/mega_fire_dataset --subset_size 2000
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import timm
import cv2
from pathlib import Path
import argparse
from tqdm import tqdm
from ultralytics import YOLO
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import random

class ProductionVerificator(nn.Module):
    """Verificador CNN que coincide EXACTAMENTE con la arquitectura de entrenamiento"""
    
    def __init__(self, num_classes: int = 2, dropout: float = 0.3):
        super().__init__()
        
        # Backbone pre-entrenado (EXACTO como entrenamiento)
        self.backbone = timm.create_model('efficientnet_b0', pretrained=False, num_classes=0)
        backbone_features = 1280
        
        # Clasificador - ARQUITECTURA EXACTA del script de entrenamiento
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),                    # 0.3
            nn.Linear(backbone_features, 256),      # 256 neurons (NOT 128)
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),                    # 0.3
            nn.Linear(256, 64),                     # 64 neurons
            nn.ReLU(inplace=True),
            nn.Dropout(dropout // 2),               # 0.15 (dropout // 2)
            nn.Linear(64, num_classes)              # Binary classification
        )
    
    def forward(self, x):
        features = self.backbone(x)
        output = self.classifier(features)
        return output

class MegaDatasetSubset(Dataset):
    """Dataset subset para optimización rápida"""
    
    def __init__(self, dataset_path: str, subset_size: int = 2000, seed: int = 42):
        self.dataset_path = Path(dataset_path)
        self.subset_size = subset_size
        
        # Cargar todas las imágenes de validación
        val_images_dir = self.dataset_path / 'images' / 'val'
        all_images = list(val_images_dir.glob('*.jpg'))
        
        # Crear subset balanceado
        random.seed(seed)
        selected_images = random.sample(all_images, min(subset_size, len(all_images)))
        
        self.samples = []
        for img_path in selected_images:
            # Determinar ground truth basado en el nombre del archivo
            has_fire = self._determine_ground_truth(img_path.name)
            self.samples.append((str(img_path), has_fire))
        
        # Estadísticas del subset
        fire_count = sum(1 for _, has_fire in self.samples if has_fire)
        no_fire_count = len(self.samples) - fire_count
        
        print(f"📊 Subset creado: {len(self.samples)} imágenes")
        print(f"   - Con fuego: {fire_count} ({fire_count/len(self.samples)*100:.1f}%)")
        print(f"   - Sin fuego: {no_fire_count} ({no_fire_count/len(self.samples)*100:.1f}%)")
    
    def _determine_ground_truth(self, filename: str) -> bool:
        """Determinar si la imagen contiene fuego basado en el nombre del archivo"""
        fire_indicators = [
            'fire', 'smoke', 'pyro', 'figlib', 'nemo'  # Datasets que contienen fuegos
        ]
        no_fire_indicators = [
            'neitherFireNorSmoke', 'CV0'  # Indicadores específicos de no-fuego
        ]
        
        filename_lower = filename.lower()
        
        # Verificar indicadores explícitos de no-fuego
        if any(indicator in filename_lower for indicator in no_fire_indicators):
            return False
            
        # Verificar indicadores de fuego
        if any(indicator in filename_lower for indicator in fire_indicators):
            return True
            
        # Por defecto, asumir no-fuego si no hay indicadores claros
        return False
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, has_fire = self.samples[idx]
        return img_path, has_fire

class SAIThresholdOptimizer:
    """Optimizador de threshold para el sistema SAI"""
    
    def __init__(self, 
                 detector_path: str = "RNA/models/detector_best.pt",
                 verificator_path: str = "RNA/training/runs/verificator_training/verificator_best.pt",
                 device: str = "cuda"):
        
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        print(f"🔧 Usando device: {self.device}")
        
        # Cargar detector
        print("📥 Cargando detector YOLOv8...")
        self.detector = YOLO(detector_path)
        print("✅ Detector cargado")
        
        # Cargar verificador con arquitectura correcta
        print("📥 Cargando verificador...")
        self.verificator = ProductionVerificator(num_classes=2, dropout=0.3)
        
        # Cargar pesos del verificador
        checkpoint = torch.load(verificator_path, map_location='cpu', weights_only=False)
        if 'model_state_dict' in checkpoint:
            self.verificator.load_state_dict(checkpoint['model_state_dict'])
            print("✅ Verificator pesos cargados correctamente")
        else:
            raise ValueError("No se encontró 'model_state_dict' en el checkpoint")
        
        self.verificator.to(self.device)
        self.verificator.eval()
        
        print("✅ Sistema SAI cargado y listo para optimización")
    
    def evaluate_threshold(self, dataset: MegaDatasetSubset, verificator_threshold: float) -> dict:
        """Evalúa el sistema SAI con un threshold específico"""
        
        print(f"🔍 Evaluando threshold {verificator_threshold:.3f}...")
        
        detector_threshold = 0.3  # Fijo, ya optimizado
        
        true_labels = []
        predicted_labels = []
        
        # Métricas detalladas
        metrics = {
            'threshold': verificator_threshold,
            'total_images': len(dataset),
            'detector_detections': 0,
            'verificator_acceptances': 0,
            'true_positives': 0,
            'false_positives': 0,
            'true_negatives': 0,
            'false_negatives': 0
        }
        
        for img_path, has_fire_gt in tqdm(dataset, desc=f"Threshold {verificator_threshold:.3f}"):
            
            # 1. Detector YOLOv8
            detector_results = self.detector.predict(img_path, conf=detector_threshold, verbose=False)
            has_detections = len(detector_results[0].boxes) > 0
            
            if has_detections:
                metrics['detector_detections'] += 1
                
                # 2. Verificador CNN para cada detection
                image = cv2.imread(img_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image_rgb = cv2.resize(image, (224, 224))
                image_tensor = torch.from_numpy(image_rgb.transpose(2, 0, 1)).float() / 255.0
                image_tensor = image_tensor.unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    verificator_output = self.verificator(image_tensor)
                    verificator_probs = torch.softmax(verificator_output, dim=1)
                    fire_confidence = verificator_probs[0, 0].item()  # Probability of class 0 (true fire)
                
                # 3. Decisión final SAI
                sai_alert = fire_confidence >= verificator_threshold
                
                if sai_alert:
                    metrics['verificator_acceptances'] += 1
            else:
                sai_alert = False
            
            # 4. Actualizar métricas
            true_labels.append(has_fire_gt)
            predicted_labels.append(sai_alert)
            
            if has_fire_gt and sai_alert:
                metrics['true_positives'] += 1
            elif not has_fire_gt and sai_alert:
                metrics['false_positives'] += 1
            elif not has_fire_gt and not sai_alert:
                metrics['true_negatives'] += 1
            elif has_fire_gt and not sai_alert:
                metrics['false_negatives'] += 1
        
        # 5. Calcular métricas finales
        tp = metrics['true_positives']
        fp = metrics['false_positives']
        tn = metrics['true_negatives']
        fn = metrics['false_negatives']
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0.0
        
        metrics.update({
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'accuracy': accuracy,
            'detector_detection_rate': metrics['detector_detections'] / len(dataset),
            'verificator_acceptance_rate': metrics['verificator_acceptances'] / max(1, metrics['detector_detections'])
        })
        
        print(f"   Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")
        
        return metrics
    
    def optimize_threshold(self, dataset: MegaDatasetSubset,
                          threshold_range: tuple = (0.1, 0.5),
                          threshold_step: float = 0.05,
                          min_recall: float = 0.50,
                          min_precision: float = 0.90) -> dict:
        """Encuentra el threshold óptimo del verificador"""
        
        print("🎯 Iniciando optimización de threshold...")
        print(f"   Rango: {threshold_range[0]} - {threshold_range[1]}")
        print(f"   Paso: {threshold_step}")
        print(f"   Recall mínimo: {min_recall}")
        print(f"   Precision mínima: {min_precision}")
        
        # Generar rango de thresholds
        thresholds = np.arange(threshold_range[0], threshold_range[1] + threshold_step, threshold_step)
        
        results = {}
        best_threshold = None
        best_f1 = 0.0
        
        for threshold in thresholds:
            metrics = self.evaluate_threshold(dataset, threshold)
            results[threshold] = metrics
            
            # Verificar si cumple criterios mínimos
            if metrics['recall'] >= min_recall and metrics['precision'] >= min_precision:
                if metrics['f1_score'] > best_f1:
                    best_f1 = metrics['f1_score']
                    best_threshold = threshold
        
        optimization_results = {
            'all_results': results,
            'best_threshold': best_threshold,
            'best_metrics': results[best_threshold] if best_threshold else None,
            'optimization_criteria': {
                'min_recall': min_recall,
                'min_precision': min_precision,
                'achieved_recall': results[best_threshold]['recall'] if best_threshold else 0,
                'achieved_precision': results[best_threshold]['precision'] if best_threshold else 0,
                'achieved_f1': results[best_threshold]['f1_score'] if best_threshold else 0
            }
        }
        
        if best_threshold:
            print(f"🎉 Threshold óptimo encontrado: {best_threshold:.3f}")
            print(f"   Recall: {results[best_threshold]['recall']:.3f}")
            print(f"   Precision: {results[best_threshold]['precision']:.3f}")
            print(f"   F1: {results[best_threshold]['f1_score']:.3f}")
        else:
            print("❌ No se encontró threshold que cumpla criterios mínimos")
            # Buscar el mejor balance disponible
            best_balance = max(results.items(), key=lambda x: x[1]['f1_score'])
            print(f"💡 Mejor balance disponible: threshold {best_balance[0]:.3f}")
            print(f"   Recall: {best_balance[1]['recall']:.3f}")
            print(f"   Precision: {best_balance[1]['precision']:.3f}")
            print(f"   F1: {best_balance[1]['f1_score']:.3f}")
        
        return optimization_results
    
    def generate_report(self, optimization_results: dict, output_dir: str = "threshold_optimization_results"):
        """Genera reporte completo de optimización"""
        
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Guardar resultados JSON
        results_file = output_dir / "optimization_results.json"
        with open(results_file, 'w') as f:
            # Convertir numpy types a Python types para JSON serialización
            serializable_results = {}
            for threshold, metrics in optimization_results['all_results'].items():
                serializable_results[float(threshold)] = {
                    k: float(v) if isinstance(v, (np.float32, np.float64)) else v 
                    for k, v in metrics.items()
                }
            
            json_results = {
                'optimization_summary': optimization_results['optimization_criteria'],
                'best_threshold': float(optimization_results['best_threshold']) if optimization_results['best_threshold'] else None,
                'detailed_results': serializable_results,
                'timestamp': datetime.now().isoformat()
            }
            
            json.dump(json_results, f, indent=2)
        
        # Generar gráficos
        self._generate_plots(optimization_results, output_dir)
        
        # Generar reporte markdown
        self._generate_markdown_report(optimization_results, output_dir)
        
        print(f"📊 Reporte generado en: {output_dir}")
    
    def _generate_plots(self, results: dict, output_dir: Path):
        """Genera gráficos de optimización"""
        
        thresholds = list(results['all_results'].keys())
        precisions = [results['all_results'][t]['precision'] for t in thresholds]
        recalls = [results['all_results'][t]['recall'] for t in thresholds]
        f1_scores = [results['all_results'][t]['f1_score'] for t in thresholds]
        
        # Gráfico de métricas vs threshold
        plt.figure(figsize=(12, 8))
        plt.plot(thresholds, precisions, 'b-', label='Precision', linewidth=2)
        plt.plot(thresholds, recalls, 'r-', label='Recall', linewidth=2)
        plt.plot(thresholds, f1_scores, 'g-', label='F1 Score', linewidth=2)
        
        # Marcar threshold óptimo
        if results['best_threshold']:
            plt.axvline(x=results['best_threshold'], color='orange', linestyle='--', 
                       label=f'Optimal Threshold: {results["best_threshold"]:.3f}', linewidth=2)
        
        # Líneas de criterios mínimos
        plt.axhline(y=results['optimization_criteria']['min_recall'], color='red', 
                   linestyle=':', alpha=0.7, label=f'Min Recall: {results["optimization_criteria"]["min_recall"]}')
        plt.axhline(y=results['optimization_criteria']['min_precision'], color='blue', 
                   linestyle=':', alpha=0.7, label=f'Min Precision: {results["optimization_criteria"]["min_precision"]}')
        
        plt.xlabel('Verificator Threshold')
        plt.ylabel('Metric Value')
        plt.title('SAI Threshold Optimization Results')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xlim(min(thresholds), max(thresholds))
        plt.ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(output_dir / "threshold_optimization_curves.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_markdown_report(self, results: dict, output_dir: Path):
        """Genera reporte en markdown"""
        
        report_file = output_dir / "optimization_report.md"
        
        with open(report_file, 'w') as f:
            f.write("# SAI Threshold Optimization Report\\n\\n")
            f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\\n\\n")
            
            # Resumen ejecutivo
            f.write("## 🎯 Executive Summary\\n\\n")
            if results['best_threshold']:
                best = results['best_metrics']
                f.write(f"**Optimal Threshold Found**: {results['best_threshold']:.3f}\\n\\n")
                f.write("**Performance Metrics**:\\n")
                f.write(f"- Precision: {best['precision']:.3f} ({best['precision']*100:.1f}%)\\n")
                f.write(f"- Recall: {best['recall']:.3f} ({best['recall']*100:.1f}%)\\n")  
                f.write(f"- F1 Score: {best['f1_score']:.3f} ({best['f1_score']*100:.1f}%)\\n")
                f.write(f"- Accuracy: {best['accuracy']:.3f} ({best['accuracy']*100:.1f}%)\\n\\n")
            else:
                f.write("**No optimal threshold found** that meets minimum criteria\\n\\n")
            
            # Tabla de resultados
            f.write("## 📊 Detailed Results\\n\\n")
            f.write("| Threshold | Precision | Recall | F1 Score | TP | FP | TN | FN |\\n")
            f.write("|-----------|-----------|--------|----------|----|----|----|----|\\n")
            
            for threshold in sorted(results['all_results'].keys()):
                metrics = results['all_results'][threshold]
                f.write(f"| {threshold:.3f} | {metrics['precision']:.3f} | {metrics['recall']:.3f} | "
                       f"{metrics['f1_score']:.3f} | {metrics['true_positives']} | {metrics['false_positives']} | "
                       f"{metrics['true_negatives']} | {metrics['false_negatives']} |\\n")
            
            f.write("\\n")
            f.write("![Optimization Curves](threshold_optimization_curves.png)\\n\\n")
        
        print(f"📄 Reporte markdown: {report_file}")

def main():
    parser = argparse.ArgumentParser(description="SAI Threshold Optimization")
    parser.add_argument("--dataset", required=True, help="Path to MEGA dataset")
    parser.add_argument("--subset_size", type=int, default=2000, help="Subset size for optimization")
    parser.add_argument("--min_recall", type=float, default=0.50, help="Minimum recall required")
    parser.add_argument("--min_precision", type=float, default=0.90, help="Minimum precision required")
    parser.add_argument("--threshold_start", type=float, default=0.1, help="Start threshold")
    parser.add_argument("--threshold_end", type=float, default=0.5, help="End threshold")
    parser.add_argument("--threshold_step", type=float, default=0.05, help="Threshold step")
    parser.add_argument("--output_dir", default="threshold_optimization_results", help="Output directory")
    
    args = parser.parse_args()
    
    print("🚀 SAI Threshold Optimization Suite")
    print("="*50)
    
    # Crear dataset subset
    dataset = MegaDatasetSubset(args.dataset, args.subset_size)
    
    # Inicializar optimizador
    optimizer = SAIThresholdOptimizer()
    
    # Optimizar threshold
    results = optimizer.optimize_threshold(
        dataset=dataset,
        threshold_range=(args.threshold_start, args.threshold_end),
        threshold_step=args.threshold_step,
        min_recall=args.min_recall,
        min_precision=args.min_precision
    )
    
    # Generar reporte
    optimizer.generate_report(results, args.output_dir)
    
    print("\\n🎉 Optimización completada!")
    print(f"📊 Resultados guardados en: {args.output_dir}")

if __name__ == "__main__":
    main()