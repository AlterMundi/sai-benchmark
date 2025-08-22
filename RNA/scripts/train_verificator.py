#!/usr/bin/env python3
"""
SAI Etapa B - Entrenamiento del Verificador CNN
Script completo para entrenar el modelo de verificación que confirma/rechaza detecciones del YOLOv8

Uso:
    python3 RNA/scripts/train_verificator.py --dataset RNA/data/verificator_dataset
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import timm
import numpy as np
import cv2
from pathlib import Path
import yaml
import json
from datetime import datetime
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, roc_auc_score
import seaborn as sns
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

class VerificatorDataset(Dataset):
    """Dataset para entrenamiento del verificador CNN"""
    
    def __init__(self, root_dir, split='train', transform=None, temporal_frames=2):
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.temporal_frames = temporal_frames
        
        # Mapeo de clases a índices
        self.class_to_idx = {
            'true_fire': 0,
            'true_smoke': 0,     # Ambos son positivos (true detection)
            'false_positive': 1   # Negativo (false detection)
        }
        
        # Cargar lista de archivos
        self.samples = self._load_samples()
        
    def _load_samples(self):
        """Cargar lista de samples con sus etiquetas"""
        samples = []
        images_dir = self.root_dir / 'images' / self.split
        
        for class_name in ['true_fire', 'true_smoke', 'false_positive']:
            class_dir = images_dir / class_name
            if not class_dir.exists():
                continue
                
            for img_path in class_dir.glob('*.jpg'):
                label = self.class_to_idx[class_name]
                samples.append((str(img_path), label))
        
        print(f"📊 {self.split} dataset: {len(samples)} samples loaded")
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Cargar imagen
        try:
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a black image as fallback
            image = np.zeros((224, 224, 3), dtype=np.uint8)
        
        # Aplicar transformaciones
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        else:
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        
        return image, torch.tensor(label, dtype=torch.long)


class SmokeyNetLite(nn.Module):
    """
    Verificador CNN basado en SmokeyNet simplificado
    Arquitectura: Backbone + LSTM temporal + Clasificador binario
    """
    
    def __init__(self, backbone='efficientnet_b0', num_classes=2, temporal_frames=1, dropout=0.3):
        super(SmokeyNetLite, self).__init__()
        
        self.temporal_frames = temporal_frames
        
        # Backbone pre-entrenado
        if backbone == 'efficientnet_b0':
            self.backbone = timm.create_model('efficientnet_b0', 
                                            pretrained=True, 
                                            num_classes=0,
                                            global_pool='')
            backbone_features = 1280
            self.global_pool = nn.AdaptiveAvgPool2d(1)
        elif backbone == 'resnet34':
            self.backbone = models.resnet34(pretrained=True)
            # Remover la última capa de clasificación
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
            backbone_features = 512
            self.global_pool = nn.AdaptiveAvgPool2d(1)
        else:
            raise ValueError(f"Backbone {backbone} no soportado")
            
        self.backbone_features = backbone_features
        
        # Módulo temporal (LSTM simple) - solo si temporal_frames > 1
        if temporal_frames > 1:
            self.use_temporal = True
            self.lstm = nn.LSTM(backbone_features, 256, 
                               batch_first=True, 
                               bidirectional=True,
                               dropout=dropout)
            classifier_input = 512  # 256 * 2 (bidirectional)
        else:
            self.use_temporal = False
            classifier_input = backbone_features
        
        # Clasificador final
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(classifier_input, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout // 2),
            nn.Linear(64, num_classes)
        )
        
        # Inicialización de pesos
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Inicializar pesos del clasificador"""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # x shape: (batch, channels, height, width) o (batch, frames, channels, height, width)
        
        if self.use_temporal and len(x.shape) == 5:
            # Temporal mode: (batch, frames, channels, height, width)
            batch_size, frames = x.size(0), x.size(1)
            
            # Reshape para procesar todos los frames juntos
            x = x.view(-1, *x.shape[2:])  # (batch*frames, C, H, W)
            
            # Extract features con backbone
            features = self.backbone(x)  # (batch*frames, features, H', W')
            features = self.global_pool(features)  # (batch*frames, features, 1, 1)
            features = features.flatten(1)  # (batch*frames, features)
            
            # Reshape para LSTM
            features = features.view(batch_size, frames, -1)  # (batch, frames, features)
            
            # Módulo temporal
            lstm_out, _ = self.lstm(features)  # (batch, frames, 512)
            
            # Usar último frame para decisión
            final_features = lstm_out[:, -1, :]  # (batch, 512)
            
        else:
            # Single frame mode: (batch, channels, height, width)
            features = self.backbone(x)  # (batch, features, H', W')
            features = self.global_pool(features)  # (batch, features, 1, 1)
            final_features = features.flatten(1)  # (batch, features)
        
        # Clasificación
        output = self.classifier(final_features)
        return output


class VerificatorTrainer:
    """Clase principal para entrenamiento del verificador"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.train_loader = None
        self.val_loader = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        
        # Métricas de entrenamiento
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        self.val_f1_scores = []
        
        print(f"🖥️ Usando device: {self.device}")
        if torch.cuda.is_available():
            print(f"🔥 GPU: {torch.cuda.get_device_name(0)}")
            print(f"💾 VRAM disponible: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    def create_data_transforms(self):
        """Crear transformaciones de datos para entrenamiento y validación"""
        
        # Transformaciones de entrenamiento (con augmentación)
        train_transform = A.Compose([
            A.Resize(224, 224),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.Rotate(limit=15, p=0.3),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
            A.GaussianBlur(blur_limit=(3, 5), p=0.3),
            A.MotionBlur(blur_limit=5, p=0.2),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        
        # Transformaciones de validación (sin augmentación)
        val_transform = A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        
        return train_transform, val_transform
    
    def create_data_loaders(self):
        """Crear data loaders para entrenamiento y validación"""
        
        train_transform, val_transform = self.create_data_transforms()
        
        # Datasets
        train_dataset = VerificatorDataset(
            root_dir=self.config['dataset_path'],
            split='train',
            transform=train_transform,
            temporal_frames=self.config['temporal_frames']
        )
        
        val_dataset = VerificatorDataset(
            root_dir=self.config['dataset_path'],
            split='val', 
            transform=val_transform,
            temporal_frames=self.config['temporal_frames']
        )
        
        # Data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=self.config['num_workers'],
            pin_memory=True if torch.cuda.is_available() else False,
            drop_last=True
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=self.config['num_workers'],
            pin_memory=True if torch.cuda.is_available() else False,
            drop_last=False
        )
        
        print(f"📊 Train samples: {len(train_dataset)}")
        print(f"📊 Val samples: {len(val_dataset)}")
        print(f"📊 Train batches: {len(self.train_loader)}")
        print(f"📊 Val batches: {len(self.val_loader)}")
    
    def create_model(self):
        """Crear modelo del verificador"""
        
        self.model = SmokeyNetLite(
            backbone=self.config['backbone'],
            num_classes=2,  # true_detection vs false_positive
            temporal_frames=self.config['temporal_frames'],
            dropout=self.config['dropout']
        )
        
        self.model.to(self.device)
        
        # Contar parámetros
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"🧠 Modelo: {self.config['backbone']}")
        print(f"📊 Total parámetros: {total_params:,}")
        print(f"🔧 Parámetros entrenables: {trainable_params:,}")
    
    def create_optimizer_scheduler(self):
        """Crear optimizador y scheduler"""
        
        # Optimizador
        if self.config['optimizer'] == 'adamw':
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config['learning_rate'],
                weight_decay=self.config['weight_decay'],
                betas=(0.9, 0.999)
            )
        elif self.config['optimizer'] == 'adam':
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.config['learning_rate'],
                weight_decay=self.config['weight_decay']
            )
        else:
            raise ValueError(f"Optimizer {self.config['optimizer']} no soportado")
        
        # Scheduler
        if self.config['scheduler'] == 'reduce_on_plateau':
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, 
                mode='min',
                patience=self.config['scheduler_patience'],
                factor=0.5,
                verbose=True
            )
        elif self.config['scheduler'] == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['epochs']
            )
        
        # Criterion con class weighting para balancear dataset
        if self.config['use_class_weights']:
            # Calcular pesos basado en frecuencia de clases
            # Asumir que false_positives son ~50% del dataset
            weights = torch.tensor([2.0, 1.0]).to(self.device)  # Dar más peso a true_detection
            self.criterion = nn.CrossEntropyLoss(weight=weights)
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        print(f"⚙️ Optimizador: {self.config['optimizer']}")
        print(f"⚙️ Scheduler: {self.config['scheduler']}")
        print(f"⚙️ Learning rate: {self.config['learning_rate']}")
    
    def train_epoch(self, epoch):
        """Entrenar una época"""
        
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config['epochs']}")
        
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config['grad_clip'] > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['grad_clip'])
            
            self.optimizer.step()
            
            # Estadísticas
            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            # Actualizar progress bar
            if batch_idx % 10 == 0:
                pbar.set_postfix({
                    'Loss': f"{loss.item():.4f}",
                    'Acc': f"{100.*correct/total:.1f}%"
                })
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate_epoch(self):
        """Validar el modelo"""
        
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_targets = []
        all_probs = []
        
        with torch.no_grad():
            for data, target in tqdm(self.val_loader, desc="Validating"):
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                val_loss += self.criterion(output, target).item()
                
                # Predicciones
                probs = F.softmax(output, dim=1)
                _, predicted = output.max(1)
                
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
                
                # Guardar para métricas detalladas
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())  # Probabilidad clase positiva
        
        val_loss /= len(self.val_loader)
        val_acc = 100. * correct / total
        
        # Métricas detalladas
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_targets, all_preds, average='weighted', zero_division=0
        )
        
        # AUC-ROC
        try:
            auc_score = roc_auc_score(all_targets, all_probs)
        except:
            auc_score = 0.0
        
        metrics = {
            'loss': val_loss,
            'accuracy': val_acc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc_score,
            'predictions': all_preds,
            'targets': all_targets,
            'probabilities': all_probs
        }
        
        return metrics
    
    def save_checkpoint(self, epoch, metrics, is_best=False):
        """Guardar checkpoint del modelo"""
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'config': self.config,
            'metrics': metrics,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies,
            'val_f1_scores': self.val_f1_scores
        }
        
        # Crear directorio de salida
        output_dir = Path(self.config['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Guardar checkpoint regular
        checkpoint_path = output_dir / f'verificator_epoch_{epoch+1}.pt'
        torch.save(checkpoint, checkpoint_path)
        
        # Guardar mejor modelo
        if is_best:
            best_path = output_dir / 'verificator_best.pt'
            torch.save(checkpoint, best_path)
            print(f"💾 Mejor modelo guardado: {best_path}")
        
        # Guardar último modelo
        last_path = output_dir / 'verificator_last.pt'
        torch.save(checkpoint, last_path)
    
    def plot_training_curves(self):
        """Crear gráficos de las curvas de entrenamiento"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss
        axes[0, 0].plot(self.train_losses, label='Train Loss', color='blue')
        axes[0, 0].plot(self.val_losses, label='Val Loss', color='red')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy
        axes[0, 1].plot(self.val_accuracies, label='Val Accuracy', color='green')
        axes[0, 1].set_title('Validation Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # F1 Score
        axes[1, 0].plot(self.val_f1_scores, label='Val F1 Score', color='purple')
        axes[1, 0].set_title('Validation F1 Score')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('F1 Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Learning Rate
        if hasattr(self.scheduler, 'get_last_lr'):
            lrs = [group['lr'] for group in self.optimizer.param_groups]
            axes[1, 1].plot(lrs, label='Learning Rate', color='orange')
            axes[1, 1].set_title('Learning Rate Schedule')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        # Guardar gráfico
        output_dir = Path(self.config['output_dir'])
        plot_path = output_dir / 'training_curves.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Curvas de entrenamiento guardadas: {plot_path}")
    
    def create_confusion_matrix(self, metrics):
        """Crear matriz de confusión"""
        
        cm = confusion_matrix(metrics['targets'], metrics['predictions'])
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['True Detection', 'False Positive'],
                   yticklabels=['True Detection', 'False Positive'])
        plt.title('Confusion Matrix - Verificator CNN')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        # Guardar
        output_dir = Path(self.config['output_dir'])
        cm_path = output_dir / 'confusion_matrix.png'
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Matriz de confusión guardada: {cm_path}")
    
    def train(self):
        """Función principal de entrenamiento"""
        
        print("🚀 INICIANDO ENTRENAMIENTO DEL VERIFICADOR SAI")
        print("=" * 60)
        
        # Configuración inicial
        self.create_data_loaders()
        self.create_model()
        self.create_optimizer_scheduler()
        
        # Variables de control
        best_f1 = 0.0
        patience_counter = 0
        start_time = datetime.now()
        
        print(f"⏰ Entrenamiento iniciado: {start_time}")
        print(f"🎯 Objetivo: F1 > 0.80, Reducción FP > 70%")
        
        # Loop de entrenamiento
        for epoch in range(self.config['epochs']):
            
            # Entrenar época
            train_loss, train_acc = self.train_epoch(epoch)
            
            # Validar época
            val_metrics = self.validate_epoch()
            
            # Guardar métricas
            self.train_losses.append(train_loss)
            self.val_losses.append(val_metrics['loss'])
            self.val_accuracies.append(val_metrics['accuracy'])
            self.val_f1_scores.append(val_metrics['f1'])
            
            # Actualizar scheduler
            if self.scheduler:
                if self.config['scheduler'] == 'reduce_on_plateau':
                    self.scheduler.step(val_metrics['loss'])
                else:
                    self.scheduler.step()
            
            # Imprimir métricas
            print(f"\n📊 Época {epoch+1}/{self.config['epochs']}")
            print(f"   Train - Loss: {train_loss:.4f}, Acc: {train_acc:.1f}%")
            print(f"   Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.1f}%")
            print(f"   Val   - Precision: {val_metrics['precision']:.3f}, Recall: {val_metrics['recall']:.3f}")
            print(f"   Val   - F1: {val_metrics['f1']:.3f}, AUC: {val_metrics['auc']:.3f}")
            
            # Early stopping y guardado de mejor modelo
            is_best = val_metrics['f1'] > best_f1
            if is_best:
                best_f1 = val_metrics['f1']
                patience_counter = 0
                print(f"   🎉 ¡Nuevo mejor F1 score: {best_f1:.3f}!")
            else:
                patience_counter += 1
                print(f"   ⏳ Paciencia: {patience_counter}/{self.config['patience']}")
            
            # Guardar checkpoint
            if (epoch + 1) % self.config['save_every'] == 0:
                self.save_checkpoint(epoch, val_metrics, is_best)
            
            # Early stopping
            if patience_counter >= self.config['patience']:
                print(f"\n⏹️ Early stopping en época {epoch+1}")
                print(f"   Mejor F1 score: {best_f1:.3f}")
                break
        
        # Entrenar completado
        end_time = datetime.now()
        training_duration = end_time - start_time
        
        print(f"\n🎉 ENTRENAMIENTO COMPLETADO")
        print("=" * 50)
        print(f"⏰ Duración total: {training_duration}")
        print(f"🏆 Mejor F1 score: {best_f1:.3f}")
        
        # Crear visualizaciones finales
        self.plot_training_curves()
        
        # Evaluar modelo final con mejor checkpoint
        final_metrics = self.validate_epoch()
        self.create_confusion_matrix(final_metrics)
        
        # Guardar checkpoint final
        self.save_checkpoint(epoch, final_metrics, is_best=False)
        
        # Reporte final
        self.create_final_report(training_duration, best_f1, final_metrics)
        
        return best_f1, final_metrics
    
    def create_final_report(self, duration, best_f1, final_metrics):
        """Crear reporte final del entrenamiento"""
        
        report = {
            'training_report': {
                'timestamp': datetime.now().isoformat(),
                'duration': str(duration),
                'config': self.config,
                'best_f1_score': float(best_f1),
                'final_metrics': {
                    'accuracy': float(final_metrics['accuracy']),
                    'precision': float(final_metrics['precision']),
                    'recall': float(final_metrics['recall']),
                    'f1': float(final_metrics['f1']),
                    'auc': float(final_metrics['auc'])
                },
                'training_curves': {
                    'train_losses': [float(x) for x in self.train_losses],
                    'val_losses': [float(x) for x in self.val_losses],
                    'val_accuracies': [float(x) for x in self.val_accuracies],
                    'val_f1_scores': [float(x) for x in self.val_f1_scores]
                }
            }
        }
        
        # Guardar reporte
        output_dir = Path(self.config['output_dir'])
        report_path = output_dir / 'training_report.json'
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"📄 Reporte final guardado: {report_path}")
        
        return report_path


def main():
    parser = argparse.ArgumentParser(description='Entrenar verificador CNN SAI')
    parser.add_argument('--dataset', 
                       default='RNA/data/verificator_dataset',
                       help='Ruta al dataset del verificador')
    parser.add_argument('--output-dir',
                       default='RNA/training/runs/verificator',
                       help='Directorio de salida')
    parser.add_argument('--backbone',
                       default='efficientnet_b0',
                       choices=['efficientnet_b0', 'resnet34'],
                       help='Backbone del modelo')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Tamaño del batch')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Número de épocas')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--gpu-optimized', action='store_true',
                       help='Usar configuración optimizada para GPU potente')
    
    args = parser.parse_args()
    
    # Configuración base
    config = {
        'dataset_path': args.dataset,
        'output_dir': args.output_dir,
        'backbone': args.backbone,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'learning_rate': args.lr,
        'weight_decay': 1e-5,
        'dropout': 0.3,
        'temporal_frames': 1,  # Single frame inicialmente
        'optimizer': 'adamw',
        'scheduler': 'reduce_on_plateau',
        'scheduler_patience': 7,
        'patience': 15,  # Early stopping
        'grad_clip': 1.0,
        'num_workers': 4,
        'save_every': 5,
        'use_class_weights': True
    }
    
    # Configuración optimizada para GPU potente (A100)
    if args.gpu_optimized and torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        if 'A100' in gpu_name:
            print("🚀 Detectado A100 - Aplicando configuración optimizada")
            config.update({
                'batch_size': 64,
                'num_workers': 16,
                'learning_rate': 2e-4,  # LR más alto para batches más grandes
                'scheduler_patience': 5,
                'patience': 12
            })
    
    print("⚙️ CONFIGURACIÓN DEL ENTRENAMIENTO")
    print("=" * 40)
    for key, value in config.items():
        print(f"   {key}: {value}")
    print()
    
    # Crear trainer y entrenar
    trainer = VerificatorTrainer(config)
    best_f1, final_metrics = trainer.train()
    
    # Resumen final
    print(f"\n🎯 RESULTADOS FINALES")
    print("=" * 30)
    print(f"🏆 Mejor F1 Score: {best_f1:.3f}")
    print(f"🎯 Precisión Final: {final_metrics['precision']:.3f}")
    print(f"🎯 Recall Final: {final_metrics['recall']:.3f}")
    print(f"🎯 AUC Final: {final_metrics['auc']:.3f}")
    
    # Evaluar si se cumplen los objetivos
    target_f1 = 0.80
    if best_f1 >= target_f1:
        print(f"✅ ¡Objetivo F1 > {target_f1} ALCANZADO!")
    else:
        print(f"⚠️ Objetivo F1 > {target_f1} no alcanzado (actual: {best_f1:.3f})")
    
    print(f"\n🔥 Modelo verificador listo para integración con detector YOLOv8")
    print(f"📁 Modelos guardados en: {config['output_dir']}")


if __name__ == "__main__":
    main()