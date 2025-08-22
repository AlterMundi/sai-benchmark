#!/usr/bin/env python3
"""
SAI Suite de Validación Continua
Sistema de validación robusta para prevenir corrupción de entrenamiento

Características:
- Validación matemática de métricas
- Detección temprana de anomalías
- Checkpoints automáticos de seguridad
- Tests de integridad de datos
- Recuperación automática de errores
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import yaml
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging
import warnings
import math

class ValidationSuite:
    """Suite completa de validación para entrenamiento SAI"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.setup_logging()
        
        # Historial de validaciones
        self.validation_history = []
        self.anomaly_count = 0
        self.last_valid_metrics = None
        
        # Rangos esperados para métricas
        self.metric_ranges = {
            'mAP50': (0.0, 1.0),
            'mAP50_95': (0.0, 1.0),
            'precision': (0.0, 1.0),
            'recall': (0.0, 1.0),
            'train_loss': (0.0, 100.0),
            'val_loss': (0.0, 100.0),
            'learning_rate': (1e-8, 1.0),
            'accuracy': (0.0, 100.0),
            'f1_score': (0.0, 1.0),
            'auc': (0.0, 1.0)
        }
        
        # Thresholds de anomalías
        self.anomaly_thresholds = {
            'sudden_drop_pct': 0.5,  # 50% drop súbito
            'sudden_spike_multiplier': 3.0,  # 3x spike
            'nan_tolerance': 0,  # Cero tolerancia a NaN
            'inf_tolerance': 0,  # Cero tolerancia a Inf
            'stagnation_epochs': 10,  # 10 épocas sin cambio
            'validation_correlation_min': 0.1  # Correlación mínima train/val
        }
        
        self.logger.info("🛡️ Suite de validación inicializada")
    
    def setup_logging(self):
        """Configurar sistema de logging"""
        
        log_dir = Path(self.config.get('log_dir', 'logs'))
        log_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger('ValidationSuite')
        self.logger.setLevel(logging.INFO)
        
        # Handler para archivo específico de validación
        log_file = log_dir / f"validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Formato detallado
        formatter = logging.Formatter(
            '%(asctime)s | VALIDATION | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
    
    def validate_numeric_metrics(self, metrics: Dict) -> Tuple[bool, List[str]]:
        """Validar métricas numéricas básicas"""
        
        issues = []
        
        for key, value in metrics.items():
            if key in ['epoch', 'timestamp']:
                continue
                
            # Verificar NaN/Inf
            if isinstance(value, (int, float)):
                if math.isnan(value):
                    issues.append(f"NaN detectado en {key}")
                elif math.isinf(value):
                    issues.append(f"Infinito detectado en {key}")
                
                # Verificar rangos válidos
                if key in self.metric_ranges:
                    min_val, max_val = self.metric_ranges[key]
                    if not (min_val <= value <= max_val):
                        issues.append(f"{key}={value} fuera de rango [{min_val}, {max_val}]")
        
        is_valid = len(issues) == 0
        return is_valid, issues
    
    def validate_metric_consistency(self, metrics: Dict) -> Tuple[bool, List[str]]:
        """Validar consistencia entre métricas relacionadas"""
        
        issues = []
        
        # Verificar que precision y recall sean consistentes con mAP
        if all(k in metrics for k in ['precision', 'recall', 'mAP50']):
            p, r, map50 = metrics['precision'], metrics['recall'], metrics['mAP50']
            
            if p > 0 and r > 0 and map50 == 0:
                issues.append("mAP50=0 pero precision y recall > 0 (inconsistente)")
            
            if p == 0 and r == 0 and map50 > 0:
                issues.append("mAP50>0 pero precision y recall = 0 (inconsistente)")
        
        # Verificar que val_loss no sea mucho mayor que train_loss (overfitting extremo)
        if 'train_loss' in metrics and 'val_loss' in metrics:
            train_loss = metrics['train_loss']
            val_loss = metrics['val_loss']
            
            if train_loss > 0 and val_loss > 0:
                ratio = val_loss / train_loss
                if ratio > 10:  # Val loss 10x mayor que train loss
                    issues.append(f"Overfitting extremo: val_loss/train_loss = {ratio:.2f}")
        
        # Verificar coherencia en accuracy vs other metrics
        if all(k in metrics for k in ['accuracy', 'precision', 'recall']):
            acc = metrics['accuracy'] / 100 if metrics['accuracy'] > 1 else metrics['accuracy']
            p, r = metrics['precision'], metrics['recall']
            
            # Accuracy no debería ser mucho mayor que precision/recall promedio
            avg_pr = (p + r) / 2
            if acc > avg_pr + 0.3:  # 30% diferencia
                issues.append(f"Accuracy ({acc:.3f}) inconsistente con P/R promedio ({avg_pr:.3f})")
        
        is_valid = len(issues) == 0
        return is_valid, issues
    
    def validate_temporal_consistency(self, current_metrics: Dict) -> Tuple[bool, List[str]]:
        """Validar consistencia temporal con métricas previas"""
        
        issues = []
        
        if not self.last_valid_metrics:
            return True, []
        
        last = self.last_valid_metrics
        current = current_metrics
        
        # Verificar drops súbitos
        for key in ['mAP50', 'precision', 'recall', 'accuracy']:
            if key in last and key in current:
                last_val = last[key]
                curr_val = current[key]
                
                if last_val > 0:
                    drop_pct = (last_val - curr_val) / last_val
                    if drop_pct > self.anomaly_thresholds['sudden_drop_pct']:
                        issues.append(f"{key} drop súbito: {drop_pct*100:.1f}%")
        
        # Verificar spikes súbitos en loss
        for key in ['train_loss', 'val_loss']:
            if key in last and key in current:
                last_val = last[key]
                curr_val = current[key]
                
                if last_val > 0 and curr_val > 0:
                    spike_ratio = curr_val / last_val
                    if spike_ratio > self.anomaly_thresholds['sudden_spike_multiplier']:
                        issues.append(f"{key} spike súbito: {spike_ratio:.1f}x")
        
        # Verificar estancamiento
        epoch_diff = current.get('epoch', 0) - last.get('epoch', 0)
        if epoch_diff >= self.anomaly_thresholds['stagnation_epochs']:
            
            # Verificar si métricas principales no han cambiado
            stagnant_metrics = []
            for key in ['mAP50', 'val_loss']:
                if key in last and key in current:
                    if abs(current[key] - last[key]) < 1e-6:
                        stagnant_metrics.append(key)
            
            if stagnant_metrics:
                issues.append(f"Estancamiento detectado en: {', '.join(stagnant_metrics)}")
        
        is_valid = len(issues) == 0
        return is_valid, issues
    
    def validate_gradient_health(self, model_path: Optional[str] = None) -> Tuple[bool, List[str]]:
        """Validar salud de gradientes (si el modelo está disponible)"""
        
        issues = []
        
        if not model_path or not Path(model_path).exists():
            return True, []  # Skip si no hay modelo
        
        try:
            # Cargar modelo
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
            
            if 'model' in checkpoint:
                model = checkpoint['model']
                
                # Verificar parámetros
                total_params = 0
                zero_params = 0
                inf_params = 0
                nan_params = 0
                
                for name, param in model.named_parameters():
                    if param is not None:
                        total_params += param.numel()
                        
                        # Verificar valores problemáticos
                        if torch.isnan(param).any():
                            nan_params += torch.isnan(param).sum().item()
                        
                        if torch.isinf(param).any():
                            inf_params += torch.isinf(param).sum().item()
                        
                        if (param == 0).all():
                            zero_params += param.numel()
                
                # Evaluar problemas
                if nan_params > 0:
                    issues.append(f"Parámetros NaN: {nan_params}/{total_params}")
                
                if inf_params > 0:
                    issues.append(f"Parámetros infinitos: {inf_params}/{total_params}")
                
                if zero_params > total_params * 0.9:  # 90% parámetros en cero
                    issues.append(f"Demasiados parámetros en cero: {zero_params}/{total_params}")
            
        except Exception as e:
            issues.append(f"Error validando gradientes: {e}")
        
        is_valid = len(issues) == 0
        return is_valid, issues
    
    def validate_data_integrity(self, data_path: str) -> Tuple[bool, List[str]]:
        """Validar integridad de datos de entrenamiento"""
        
        issues = []
        
        try:
            data_path = Path(data_path)
            
            # Verificar que el dataset existe
            if not data_path.exists():
                issues.append(f"Dataset no encontrado: {data_path}")
                return False, issues
            
            # Verificar estructura básica
            if data_path.is_file() and data_path.suffix == '.csv':
                # Es un CSV de métricas
                df = pd.read_csv(data_path)
                
                if len(df) == 0:
                    issues.append("CSV de métricas vacío")
                
                # Verificar columnas esperadas
                expected_cols = ['epoch', 'train/box_loss', 'val/box_loss', 'metrics/mAP50(B)']
                missing_cols = [col for col in expected_cols if col not in df.columns]
                
                if missing_cols:
                    issues.append(f"Columnas faltantes en CSV: {missing_cols}")
                
                # Verificar últimas épocas para anomalías
                if len(df) > 5:
                    recent_df = df.tail(5)
                    
                    # Verificar si hay demasiados NaN recientes
                    nan_count = recent_df.isnull().sum().sum()
                    total_values = recent_df.size
                    
                    if nan_count > total_values * 0.5:  # 50% NaN en últimas 5 épocas
                        issues.append(f"Demasiados NaN en datos recientes: {nan_count}/{total_values}")
            
            elif data_path.is_dir():
                # Es un directorio de dataset
                
                # Verificar subdirectorios esperados
                expected_dirs = ['images', 'labels']
                missing_dirs = [d for d in expected_dirs if not (data_path / d).exists()]
                
                if missing_dirs:
                    issues.append(f"Directorios faltantes: {missing_dirs}")
                
                # Verificar que hay imágenes
                image_count = 0
                for ext in ['*.jpg', '*.png', '*.jpeg']:
                    image_count += len(list(data_path.rglob(ext)))
                
                if image_count == 0:
                    issues.append("No se encontraron imágenes en el dataset")
                elif image_count < 100:
                    issues.append(f"Pocas imágenes en dataset: {image_count}")
        
        except Exception as e:
            issues.append(f"Error validando datos: {e}")
        
        is_valid = len(issues) == 0
        return is_valid, issues
    
    def create_emergency_backup(self, reason: str, model_path: Optional[str] = None):
        """Crear backup de emergencia"""
        
        try:
            backup_dir = Path(self.config.get('backup_dir', 'backups'))
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            emergency_dir = backup_dir / f"emergency_{timestamp}"
            emergency_dir.mkdir(exist_ok=True)
            
            # Guardar razón del backup
            reason_file = emergency_dir / 'backup_reason.txt'
            with open(reason_file, 'w') as f:
                f.write(f"Emergency Backup\n")
                f.write(f"Timestamp: {datetime.now().isoformat()}\n")
                f.write(f"Reason: {reason}\n")
                f.write(f"Anomaly count: {self.anomaly_count}\n")
            
            # Copiar modelo si existe
            if model_path and Path(model_path).exists():
                import shutil
                backup_model = emergency_dir / f"model_emergency_{timestamp}.pt"
                shutil.copy2(model_path, backup_model)
                self.logger.info(f"💾 Modelo copiado a: {backup_model}")
            
            # Guardar estado de validación
            validation_state = {
                'timestamp': datetime.now().isoformat(),
                'reason': reason,
                'anomaly_count': self.anomaly_count,
                'last_valid_metrics': self.last_valid_metrics,
                'validation_history': self.validation_history[-10:],  # Últimas 10
                'metric_ranges': self.metric_ranges,
                'anomaly_thresholds': self.anomaly_thresholds
            }
            
            state_file = emergency_dir / 'validation_state.json'
            with open(state_file, 'w') as f:
                json.dump(validation_state, f, indent=2)
            
            self.logger.warning(f"🚨 Backup de emergencia creado: {emergency_dir}")
            return str(emergency_dir)
            
        except Exception as e:
            self.logger.error(f"Error creando backup de emergencia: {e}")
            return None
    
    def run_full_validation(self, metrics: Dict, model_path: Optional[str] = None, 
                           data_path: Optional[str] = None) -> Dict:
        """Ejecutar suite completa de validación"""
        
        validation_result = {
            'timestamp': datetime.now().isoformat(),
            'epoch': metrics.get('epoch', -1),
            'overall_valid': True,
            'validation_checks': {},
            'issues': [],
            'warnings': [],
            'emergency_action_taken': False
        }
        
        # 1. Validación numérica básica
        numeric_valid, numeric_issues = self.validate_numeric_metrics(metrics)
        validation_result['validation_checks']['numeric'] = {
            'valid': numeric_valid,
            'issues': numeric_issues
        }
        
        # 2. Validación de consistencia
        consistency_valid, consistency_issues = self.validate_metric_consistency(metrics)
        validation_result['validation_checks']['consistency'] = {
            'valid': consistency_valid,
            'issues': consistency_issues
        }
        
        # 3. Validación temporal
        temporal_valid, temporal_issues = self.validate_temporal_consistency(metrics)
        validation_result['validation_checks']['temporal'] = {
            'valid': temporal_valid,
            'issues': temporal_issues
        }
        
        # 4. Validación de gradientes (opcional)
        if model_path:
            gradient_valid, gradient_issues = self.validate_gradient_health(model_path)
            validation_result['validation_checks']['gradients'] = {
                'valid': gradient_valid,
                'issues': gradient_issues
            }
        
        # 5. Validación de datos (opcional)
        if data_path:
            data_valid, data_issues = self.validate_data_integrity(data_path)
            validation_result['validation_checks']['data'] = {
                'valid': data_valid,
                'issues': data_issues
            }
        
        # Agregar todas las issues
        all_issues = numeric_issues + consistency_issues + temporal_issues
        if model_path:
            all_issues += gradient_issues
        if data_path:
            all_issues += data_issues
        
        validation_result['issues'] = all_issues
        validation_result['overall_valid'] = len(all_issues) == 0
        
        # Contar anomalías
        if not validation_result['overall_valid']:
            self.anomaly_count += 1
        else:
            self.anomaly_count = 0  # Reset si todo está bien
            self.last_valid_metrics = metrics.copy()
        
        # Acción de emergencia si hay demasiadas anomalías
        if self.anomaly_count >= 3:  # 3 validaciones consecutivas fallidas
            emergency_reason = f"Anomalías detectadas por {self.anomaly_count} validaciones consecutivas"
            backup_path = self.create_emergency_backup(emergency_reason, model_path)
            validation_result['emergency_action_taken'] = True
            validation_result['emergency_backup_path'] = backup_path
        
        # Log resultado
        if validation_result['overall_valid']:
            self.logger.info(f"✅ Validación exitosa - Época {metrics.get('epoch', '?')}")
        else:
            self.logger.error(f"❌ Validación fallida - Época {metrics.get('epoch', '?')}")
            for issue in all_issues:
                self.logger.error(f"   - {issue}")
        
        # Guardar en historial
        self.validation_history.append(validation_result)
        
        # Mantener solo últimas 100 validaciones
        if len(self.validation_history) > 100:
            self.validation_history = self.validation_history[-100:]
        
        return validation_result
    
    def get_validation_summary(self) -> Dict:
        """Obtener resumen de validaciones"""
        
        if not self.validation_history:
            return {'status': 'no_validations'}
        
        recent_validations = self.validation_history[-10:]
        
        summary = {
            'total_validations': len(self.validation_history),
            'recent_validations': len(recent_validations),
            'current_anomaly_count': self.anomaly_count,
            'last_validation': self.validation_history[-1]['timestamp'],
            'overall_health': 'healthy' if self.anomaly_count == 0 else 'anomalous',
            'recent_success_rate': sum(1 for v in recent_validations if v['overall_valid']) / len(recent_validations),
            'most_common_issues': self._get_common_issues()
        }
        
        return summary
    
    def _get_common_issues(self) -> List[str]:
        """Obtener issues más comunes en historial reciente"""
        
        issue_counts = {}
        recent_validations = self.validation_history[-20:]  # Últimas 20 validaciones
        
        for validation in recent_validations:
            for issue in validation['issues']:
                # Normalizar issue para contar similares
                normalized = issue.split(':')[0]  # Tomar parte antes de ':'
                issue_counts[normalized] = issue_counts.get(normalized, 0) + 1
        
        # Ordenar por frecuencia
        sorted_issues = sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)
        
        return [issue for issue, count in sorted_issues[:5]]  # Top 5


def main():
    """Función principal para testing de la suite"""
    
    config = {
        'log_dir': 'RNA/validation_logs',
        'backup_dir': 'RNA/emergency_backups'
    }
    
    validator = ValidationSuite(config)
    
    # Test con métricas normales
    normal_metrics = {
        'epoch': 10,
        'mAP50': 0.85,
        'precision': 0.82,
        'recall': 0.88,
        'train_loss': 0.45,
        'val_loss': 0.52,
        'learning_rate': 0.001
    }
    
    result = validator.run_full_validation(normal_metrics)
    print("Validación normal:", result['overall_valid'])
    
    # Test con métricas anómalas
    anomalous_metrics = {
        'epoch': 11,
        'mAP50': float('nan'),
        'precision': 1.5,  # Fuera de rango
        'recall': 0.88,
        'train_loss': 0.45,
        'val_loss': 50.0,  # Spike
        'learning_rate': 0.001
    }
    
    result = validator.run_full_validation(anomalous_metrics)
    print("Validación anómala:", result['overall_valid'])
    print("Issues encontradas:", result['issues'])
    
    # Mostrar resumen
    summary = validator.get_validation_summary()
    print("Resumen:", summary)


if __name__ == "__main__":
    main()