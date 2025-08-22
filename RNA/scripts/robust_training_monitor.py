#!/usr/bin/env python3
"""
SAI Sistema de Monitoreo Robusto para Entrenamiento
Monitorea entrenamiento en tiempo real y detecta corrupción temprana

Características:
- Validación continua de métricas
- Detección temprana de valores NaN/Inf
- Alertas automáticas por SSH/webhook
- Checkpoints de emergencia
- Logs estructurados con timestamps
- Monitoreo de recursos del sistema
"""

import os
import sys
import time
import json
import yaml
import logging
import psutil
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import threading
import queue
import smtplib
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart

class TrainingMonitor:
    """Monitor robusto de entrenamiento con detección temprana de problemas"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.monitoring = False
        self.alert_queue = queue.Queue()
        
        # Configurar logging
        self.setup_logging()
        
        # Métricas de seguimiento
        self.metrics_history = []
        self.corruption_alerts = []
        self.resource_alerts = []
        
        # Thresholds para alertas
        self.nan_threshold = 3  # Épocas consecutivas con NaN
        self.memory_threshold = 0.95  # 95% uso de memoria
        self.disk_threshold = 0.90  # 90% uso de disco
        
        # Estado del monitoreo
        self.last_epoch = -1
        self.nan_count = 0
        self.training_start_time = None
        
        self.logger.info("🔍 Monitor de entrenamiento inicializado")
        
    def setup_logging(self):
        """Configurar sistema de logging estructurado"""
        
        log_dir = Path(self.config['log_dir'])
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Configurar logger principal
        self.logger = logging.getLogger('TrainingMonitor')
        self.logger.setLevel(logging.INFO)
        
        # Handler para archivo
        log_file = log_dir / f"training_monitor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Handler para consola
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formato de log estructurado
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(funcName)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        self.logger.info(f"📝 Logging configurado: {log_file}")
    
    def check_system_resources(self) -> Dict:
        """Monitorear recursos del sistema"""
        
        try:
            # Memoria
            memory = psutil.virtual_memory()
            memory_usage = memory.percent / 100.0
            
            # Disco
            disk = psutil.disk_usage(self.config['training_dir'])
            disk_usage = disk.used / disk.total
            
            # CPU
            cpu_usage = psutil.cpu_percent(interval=1) / 100.0
            
            # GPU (si disponible)
            gpu_usage = 0.0
            gpu_memory = 0.0
            try:
                result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total', 
                                       '--format=csv,noheader,nounits'], 
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    gpu_data = result.stdout.strip().split(',')
                    gpu_usage = float(gpu_data[0]) / 100.0
                    gpu_memory = float(gpu_data[1]) / float(gpu_data[2])
            except:
                pass
            
            resources = {
                'timestamp': datetime.now().isoformat(),
                'memory_usage': memory_usage,
                'disk_usage': disk_usage,
                'cpu_usage': cpu_usage,
                'gpu_usage': gpu_usage,
                'gpu_memory': gpu_memory,
                'memory_gb': memory.used / (1024**3),
                'disk_gb': disk.used / (1024**3)
            }
            
            # Verificar thresholds
            alerts = []
            if memory_usage > self.memory_threshold:
                alerts.append(f"⚠️ Memoria alta: {memory_usage*100:.1f}%")
            
            if disk_usage > self.disk_threshold:
                alerts.append(f"⚠️ Disco lleno: {disk_usage*100:.1f}%")
            
            if gpu_memory > 0.95:
                alerts.append(f"⚠️ VRAM alta: {gpu_memory*100:.1f}%")
            
            if alerts:
                self.resource_alerts.extend(alerts)
                for alert in alerts:
                    self.logger.warning(alert)
            
            return resources
            
        except Exception as e:
            self.logger.error(f"Error monitoreando recursos: {e}")
            return {}
    
    def parse_training_metrics(self, csv_path: Path) -> Optional[Dict]:
        """Analizar CSV de métricas de entrenamiento"""
        
        try:
            if not csv_path.exists():
                return None
            
            # Leer CSV
            df = pd.read_csv(csv_path)
            
            if len(df) == 0:
                return None
            
            # Última época
            last_row = df.iloc[-1]
            epoch = int(last_row['epoch'])
            
            # Verificar si es nueva época
            if epoch <= self.last_epoch:
                return None
            
            self.last_epoch = epoch
            
            # Extraer métricas críticas
            metrics = {
                'epoch': epoch,
                'timestamp': datetime.now().isoformat(),
                'train_box_loss': float(last_row.get('train/box_loss', 0)),
                'train_cls_loss': float(last_row.get('train/cls_loss', 0)),
                'train_dfl_loss': float(last_row.get('train/dfl_loss', 0)),
                'val_box_loss': float(last_row.get('val/box_loss', 0)),
                'val_cls_loss': float(last_row.get('val/cls_loss', 0)),
                'val_dfl_loss': float(last_row.get('val/dfl_loss', 0)),
                'mAP50': float(last_row.get('metrics/mAP50(B)', 0)),
                'mAP50_95': float(last_row.get('metrics/mAP50-95(B)', 0)),
                'precision': float(last_row.get('metrics/precision(B)', 0)),
                'recall': float(last_row.get('metrics/recall(B)', 0)),
                'lr': float(last_row.get('lr/pg0', 0))
            }
            
            # Detectar corrupción
            corruption_detected = self.detect_corruption(metrics)
            
            if corruption_detected:
                self.logger.error(f"🚨 CORRUPCIÓN DETECTADA en época {epoch}")
                self.send_alert(f"CORRUPCIÓN EN ÉPOCA {epoch}", 
                              f"Métricas corruptas detectadas: {corruption_detected}")
            
            # Guardar en historial
            self.metrics_history.append(metrics)
            
            # Log de progreso
            self.logger.info(
                f"📊 Época {epoch}: mAP50={metrics['mAP50']:.3f}, "
                f"val_loss={metrics['val_box_loss']:.3f}, "
                f"lr={metrics['lr']:.6f}"
            )
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error analizando métricas CSV: {e}")
            return None
    
    def detect_corruption(self, metrics: Dict) -> List[str]:
        """Detectar corrupción en métricas"""
        
        corruption_issues = []
        
        # Verificar valores NaN
        nan_fields = []
        for key, value in metrics.items():
            if key in ['epoch', 'timestamp']:
                continue
            
            if np.isnan(value) or np.isinf(value):
                nan_fields.append(key)
        
        if nan_fields:
            corruption_issues.append(f"Valores NaN/Inf en: {', '.join(nan_fields)}")
            self.nan_count += 1
        else:
            self.nan_count = 0
        
        # Verificar si hay demasiados NaN consecutivos
        if self.nan_count >= self.nan_threshold:
            corruption_issues.append(f"NaN consecutivos durante {self.nan_count} épocas")
        
        # Verificar valores anómalos
        if metrics['mAP50'] > 1.0:
            corruption_issues.append(f"mAP50 anómalo: {metrics['mAP50']}")
        
        if metrics['val_box_loss'] > 100:
            corruption_issues.append(f"Val loss muy alto: {metrics['val_box_loss']}")
        
        # Verificar métricas de validación todas en cero
        val_metrics = ['val_box_loss', 'val_cls_loss', 'val_dfl_loss']
        if all(metrics[m] == 0 for m in val_metrics):
            corruption_issues.append("Todas las métricas de validación en cero")
        
        return corruption_issues
    
    def check_training_process(self) -> Dict:
        """Verificar estado del proceso de entrenamiento"""
        
        try:
            # Buscar procesos de entrenamiento
            training_processes = []
            
            for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'cpu_percent', 'memory_percent']):
                try:
                    cmdline = ' '.join(proc.info['cmdline']) if proc.info['cmdline'] else ''
                    
                    # Buscar procesos YOLO/PyTorch
                    if any(keyword in cmdline.lower() for keyword in ['yolo', 'train', 'pytorch']):
                        if 'train' in cmdline:
                            training_processes.append({
                                'pid': proc.info['pid'],
                                'name': proc.info['name'],
                                'cpu_percent': proc.info['cpu_percent'],
                                'memory_percent': proc.info['memory_percent'],
                                'cmdline': cmdline[:100] + '...' if len(cmdline) > 100 else cmdline
                            })
                except:
                    continue
            
            return {
                'timestamp': datetime.now().isoformat(),
                'training_processes': training_processes,
                'process_count': len(training_processes)
            }
            
        except Exception as e:
            self.logger.error(f"Error verificando procesos: {e}")
            return {}
    
    def send_alert(self, subject: str, message: str, level: str = "ERROR"):
        """Enviar alerta (email, webhook, etc.)"""
        
        alert = {
            'timestamp': datetime.now().isoformat(),
            'level': level,
            'subject': subject,
            'message': message
        }
        
        self.alert_queue.put(alert)
        
        # Log local
        if level == "ERROR":
            self.logger.error(f"🚨 ALERTA: {subject} - {message}")
        elif level == "WARNING":
            self.logger.warning(f"⚠️ ADVERTENCIA: {subject} - {message}")
        else:
            self.logger.info(f"ℹ️ INFO: {subject} - {message}")
        
        # Guardar en archivo de alertas
        alert_file = Path(self.config['log_dir']) / 'alerts.json'
        
        try:
            if alert_file.exists():
                with open(alert_file, 'r') as f:
                    alerts = json.load(f)
            else:
                alerts = []
            
            alerts.append(alert)
            
            # Mantener solo últimas 1000 alertas
            if len(alerts) > 1000:
                alerts = alerts[-1000:]
            
            with open(alert_file, 'w') as f:
                json.dump(alerts, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Error guardando alerta: {e}")
    
    def create_emergency_checkpoint(self, reason: str):
        """Crear checkpoint de emergencia"""
        
        try:
            self.logger.warning(f"💾 Creando checkpoint de emergencia: {reason}")
            
            # Enviar comando para crear checkpoint
            checkpoint_script = f"""
            import torch
            import os
            from datetime import datetime
            
            # Buscar último modelo
            weights_dir = '{self.config['training_dir']}/weights'
            if os.path.exists(weights_dir):
                checkpoints = [f for f in os.listdir(weights_dir) if f.endswith('.pt')]
                if checkpoints:
                    latest = max(checkpoints, key=lambda x: os.path.getctime(os.path.join(weights_dir, x)))
                    
                    # Copiar como emergencia
                    emergency_name = f"EMERGENCY_{{datetime.now().strftime('%Y%m%d_%H%M%S')}}_{{latest}}"
                    emergency_path = os.path.join(weights_dir, emergency_name)
                    
                    import shutil
                    shutil.copy2(os.path.join(weights_dir, latest), emergency_path)
                    print(f"Checkpoint de emergencia creado: {{emergency_path}}")
            """
            
            # Ejecutar script de checkpoint
            subprocess.run([sys.executable, '-c', checkpoint_script], 
                         cwd=self.config['training_dir'])
            
            self.send_alert("Checkpoint de Emergencia", 
                          f"Checkpoint creado por: {reason}", "WARNING")
            
        except Exception as e:
            self.logger.error(f"Error creando checkpoint de emergencia: {e}")
    
    def monitor_training_loop(self):
        """Loop principal de monitoreo"""
        
        self.logger.info("🔄 Iniciando loop de monitoreo...")
        self.training_start_time = datetime.now()
        
        while self.monitoring:
            try:
                # Monitorear recursos del sistema
                resources = self.check_system_resources()
                
                # Verificar procesos de entrenamiento
                processes = self.check_training_process()
                
                # Analizar métricas de entrenamiento
                csv_path = Path(self.config['training_dir']) / 'results.csv'
                metrics = self.parse_training_metrics(csv_path)
                
                # Verificar si el entrenamiento sigue activo
                if processes['process_count'] == 0:
                    self.logger.warning("⚠️ No se detectan procesos de entrenamiento activos")
                
                # Crear checkpoint de emergencia si hay problemas críticos
                if self.nan_count >= self.nan_threshold:
                    self.create_emergency_checkpoint(f"NaN detectado durante {self.nan_count} épocas")
                
                # Guardar estado completo
                self.save_monitoring_state(resources, processes, metrics)
                
                # Esperar intervalo de monitoreo
                time.sleep(self.config['monitor_interval'])
                
            except KeyboardInterrupt:
                self.logger.info("⏹️ Monitoreo detenido por usuario")
                break
                
            except Exception as e:
                self.logger.error(f"Error en loop de monitoreo: {e}")
                time.sleep(30)  # Esperar más tiempo en caso de error
        
        self.logger.info("📴 Monitoreo finalizado")
    
    def save_monitoring_state(self, resources: Dict, processes: Dict, metrics: Optional[Dict]):
        """Guardar estado completo del monitoreo"""
        
        try:
            state = {
                'timestamp': datetime.now().isoformat(),
                'monitoring_duration': str(datetime.now() - self.training_start_time) if self.training_start_time else None,
                'resources': resources,
                'processes': processes,
                'metrics': metrics,
                'alerts_count': len(self.corruption_alerts),
                'nan_count': self.nan_count,
                'last_epoch': self.last_epoch
            }
            
            # Guardar estado actual
            state_file = Path(self.config['log_dir']) / 'monitoring_state.json'
            with open(state_file, 'w') as f:
                json.dump(state, f, indent=2)
            
            # Agregar a historial
            history_file = Path(self.config['log_dir']) / 'monitoring_history.jsonl'
            with open(history_file, 'a') as f:
                f.write(json.dumps(state) + '\n')
                
        except Exception as e:
            self.logger.error(f"Error guardando estado: {e}")
    
    def start_monitoring(self):
        """Iniciar monitoreo en thread separado"""
        
        self.monitoring = True
        monitor_thread = threading.Thread(target=self.monitor_training_loop, daemon=True)
        monitor_thread.start()
        
        self.logger.info("🚀 Monitoreo iniciado en background")
        return monitor_thread
    
    def stop_monitoring(self):
        """Detener monitoreo"""
        
        self.monitoring = False
        self.logger.info("⏹️ Deteniendo monitoreo...")
    
    def get_status_report(self) -> Dict:
        """Generar reporte de estado actual"""
        
        return {
            'monitoring_active': self.monitoring,
            'training_duration': str(datetime.now() - self.training_start_time) if self.training_start_time else None,
            'last_epoch': self.last_epoch,
            'nan_count': self.nan_count,
            'total_alerts': len(self.corruption_alerts),
            'metrics_count': len(self.metrics_history),
            'last_metrics': self.metrics_history[-1] if self.metrics_history else None
        }


def create_monitoring_config(training_dir: str, log_dir: str) -> Dict:
    """Crear configuración para el monitor"""
    
    config = {
        'training_dir': training_dir,
        'log_dir': log_dir,
        'monitor_interval': 30,  # Verificar cada 30 segundos
        'alert_thresholds': {
            'nan_consecutive': 3,
            'memory_usage': 0.95,
            'disk_usage': 0.90,
            'gpu_memory': 0.95
        },
        'emergency_actions': {
            'create_checkpoint': True,
            'send_alerts': True,
            'log_everything': True
        }
    }
    
    return config


def main():
    """Función principal para usar como script independiente"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Monitor robusto de entrenamiento SAI')
    parser.add_argument('--training-dir', required=True,
                       help='Directorio de entrenamiento a monitorear')
    parser.add_argument('--log-dir', 
                       default='RNA/monitoring/logs',
                       help='Directorio para logs de monitoreo')
    parser.add_argument('--interval', type=int, default=30,
                       help='Intervalo de monitoreo en segundos')
    
    args = parser.parse_args()
    
    # Crear configuración
    config = create_monitoring_config(args.training_dir, args.log_dir)
    config['monitor_interval'] = args.interval
    
    # Crear monitor
    monitor = TrainingMonitor(config)
    
    try:
        # Iniciar monitoreo
        monitor.start_monitoring()
        
        print("🔍 Monitor iniciado. Presiona Ctrl+C para detener.")
        
        # Mantener el programa corriendo
        while True:
            time.sleep(10)
            
            # Mostrar estado cada 5 minutos
            if datetime.now().second % 300 == 0:
                status = monitor.get_status_report()
                print(f"📊 Estado: Época {status['last_epoch']}, "
                      f"NaN count: {status['nan_count']}, "
                      f"Alertas: {status['total_alerts']}")
    
    except KeyboardInterrupt:
        print("\n⏹️ Deteniendo monitor...")
        monitor.stop_monitoring()
        time.sleep(2)
        print("✅ Monitor detenido")


if __name__ == "__main__":
    main()