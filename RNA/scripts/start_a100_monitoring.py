#!/usr/bin/env python3
"""
SAI Monitoring - Específico para A100-SXM4-40GB
Monitoreo optimizado para hardware del servidor alquilado
"""

import sys
import os
sys.path.append('/data/sai-benchmark')

from RNA.scripts.robust_training_monitor import TrainingMonitor
import json
from datetime import datetime

# Configuración REAL A100
HARDWARE_CONFIG = {
    "gpu_name": "NVIDIA A100-SXM4-40GB",
    "gpu_memory_gb": 40.96,
    "ram_total_gb": 251,
    "ram_available_gb": 243,
    "cpu_cores": 128,
    "disk_total_gb": 300,
    "disk_available_gb": 285,
    "cuda_driver": "550.107.02"
}

MONITORING_CONFIG = {
    "enabled": true,
    "training_dir": "/data/sai-benchmark/RNA/training/runs",
    "log_dir": "/data/sai-benchmark/RNA/training/monitoring",
    "monitor_interval": 10,
    "alert_thresholds": {
        "nan_consecutive": 2,
        "memory_usage": 0.9,
        "ram_usage": 0.85,
        "disk_usage": 0.8,
        "gpu_memory": 0.85,
        "gpu_utilization_min": 0.7,
        "validation_loss_spike": 5.0,
        "no_improvement_epochs": 15
    },
    "monitor_gpu_stats": true,
    "monitor_nvlink": true,
    "monitor_tensor_cores": true,
    "track_memory_fragmentation": true,
    "emergency_actions": {
        "create_checkpoint": true,
        "send_alerts": true,
        "pause_training": false,
        "notify_user": true,
        "save_system_state": true
    },
    "validation_checks": {
        "check_nan_values": true,
        "check_inf_values": true,
        "check_metric_ranges": true,
        "check_gradient_norms": true,
        "check_learning_rate": true,
        "check_gpu_memory_leaks": true,
        "validate_tensor_shapes": true
    }
}

def main():
    """Iniciar monitoreo A100"""
    
    print("🔍 SAI A100 MONITORING")
    print("=" * 30)
    print(f"🖥️ Hardware: {HARDWARE_CONFIG['gpu_name']}")
    print(f"💾 VRAM: {HARDWARE_CONFIG['gpu_memory_gb']}GB")
    print(f"🔢 RAM: {HARDWARE_CONFIG['ram_available_gb']}GB")
    print(f"⚙️ CPU: {HARDWARE_CONFIG['cpu_cores']} cores")
    print(f"💿 Disco: {HARDWARE_CONFIG['disk_available_gb']}GB")
    print()
    
    # Crear monitor con configuración A100
    monitor = TrainingMonitor(MONITORING_CONFIG)
    
    try:
        # Iniciar monitoreo
        monitor.start_monitoring()
        
        print("🔍 Monitoreo A100 iniciado")
        print("📊 Presiona Ctrl+C para detener")
        
        # Mantener corriendo
        import time
        while True:
            time.sleep(60)
            status = monitor.get_status_report()
            print(f"📈 Estado: Época {status.get('last_epoch', '?')}, "
                  f"Anomalías: {status.get('nan_count', 0)}")
    
    except KeyboardInterrupt:
        print("\n⏹️ Deteniendo monitoreo A100...")
        monitor.stop_monitoring()
        print("✅ Monitoreo detenido")

if __name__ == "__main__":
    main()
