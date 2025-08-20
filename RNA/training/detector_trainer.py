#!/usr/bin/env python3
"""
YOLOv8-s Detector Training Pipeline for SAI

Automated training pipeline for YOLOv8-s detector with:
- Automatic dataset preparation from multiple sources
- 1440×808 resolution optimization 
- Early stopping and checkpoint management
- Performance monitoring and logging
- RTX 3090 optimization with mixed precision
"""

import os
import sys
import yaml
import torch
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import time
import json

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

try:
    from ultralytics import YOLO
    from ultralytics.utils import LOGGER
except ImportError:
    print("ERROR: Ultralytics not installed. Install with: pip install ultralytics")
    sys.exit(1)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('detector_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class DatasetPreparer:
    """Prepare combined dataset for YOLO training"""
    
    def __init__(self, data_root: str, output_dir: str):
        self.data_root = Path(data_root)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def prepare_combined_dataset(self) -> str:
        """Prepare combined dataset from all sources"""
        logger.info("Preparing combined dataset for YOLOv8 training...")
        
        # Create YOLO dataset structure
        train_images = self.output_dir / "images" / "train"
        train_labels = self.output_dir / "labels" / "train"
        val_images = self.output_dir / "images" / "val"
        val_labels = self.output_dir / "labels" / "val"
        
        for path in [train_images, train_labels, val_images, val_labels]:
            path.mkdir(parents=True, exist_ok=True)
        
        # Create dataset YAML
        dataset_yaml = self.output_dir / "dataset.yaml"
        dataset_config = {
            'path': str(self.output_dir.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'nc': 2,  # Number of classes
            'names': ['smoke', 'fire']
        }
        
        with open(dataset_yaml, 'w') as f:
            yaml.dump(dataset_config, f)
        
        logger.info(f"Dataset configuration saved to {dataset_yaml}")
        return str(dataset_yaml)
    
    def convert_datasets(self):
        """Convert all available datasets to YOLO format"""
        # This would contain the conversion logic for:
        # - FASDD (COCO -> YOLO)
        # - PyroNear (HF -> YOLO)  
        # - D-Fire (already YOLO)
        # - FIgLib (HF -> YOLO)
        # - NEMO (COCO -> YOLO)
        
        logger.info("Dataset conversion logic would be implemented here")
        logger.info("For now, assuming datasets are already in YOLO format")


class YOLOTrainer:
    """YOLOv8-s Training Pipeline"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.start_time = None
        self.training_stats = {}
        
        # Setup training environment
        self._setup_environment()
        
    def _setup_environment(self):
        """Setup training environment and GPU"""
        # Check GPU availability
        if not torch.cuda.is_available():
            logger.warning("CUDA not available, training will be slow on CPU")
            self.device = 'cpu'
        else:
            self.device = 'cuda'
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"Using GPU: {gpu_name} ({gpu_memory:.1f}GB)")
            
            # Optimize for RTX 3090
            if "3090" in gpu_name:
                logger.info("RTX 3090 detected - optimizing training settings")
                torch.backends.cudnn.benchmark = True
                
        # Create output directories
        self.output_dir = Path(self.config['training']['output_dir'])
        self.weights_dir = self.output_dir / "weights"
        self.logs_dir = self.output_dir / "logs"
        
        for path in [self.output_dir, self.weights_dir, self.logs_dir]:
            path.mkdir(parents=True, exist_ok=True)
            
    def initialize_model(self) -> YOLO:
        """Initialize YOLOv8-s model"""
        logger.info("Initializing YOLOv8-s model...")
        
        # Load pre-trained YOLOv8s
        model = YOLO('yolov8s.pt')
        
        # Log model info
        logger.info(f"Model: {model.model}")
        logger.info(f"Model parameters: {sum(p.numel() for p in model.model.parameters()):,}")
        
        self.model = model
        return model
        
    def train(self, dataset_yaml: str):
        """Train YOLOv8-s detector"""
        if self.model is None:
            self.initialize_model()
            
        logger.info("=" * 60)
        logger.info("STARTING SAI YOLOv8-s DETECTOR TRAINING")
        logger.info("=" * 60)
        
        training_config = self.config['training']['detector']
        
        # Training parameters optimized for 1440×808 and RTX 3090
        train_params = {
            'data': dataset_yaml,
            'epochs': training_config['epochs'],
            'imgsz': training_config['image_size'],  # [1440, 808]
            'batch': training_config['batch_size'],
            'device': self.device,
            'project': str(self.output_dir),
            'name': 'sai_detector_training',
            'save': True,
            'save_period': training_config.get('save_period', 10),
            'cache': training_config.get('cache', True),
            'patience': training_config.get('patience', 50),
            'workers': training_config.get('workers', 8),
            'amp': training_config.get('mixed_precision', True),  # Mixed precision for RTX 3090
            'resume': training_config.get('resume', True),
            'optimizer': training_config.get('optimizer', 'AdamW'),
            'lr0': training_config.get('learning_rate', 0.001),
            'weight_decay': training_config.get('weight_decay', 0.0005),
            'warmup_epochs': training_config.get('warmup_epochs', 3),
            'box': training_config.get('box_loss_gain', 7.5),
            'cls': training_config.get('cls_loss_gain', 0.5),
            'dfl': training_config.get('dfl_loss_gain', 1.5),
            'val': True,
            'plots': True,
            'verbose': True
        }
        
        logger.info("Training Parameters:")
        for key, value in train_params.items():
            logger.info(f"  {key}: {value}")
        
        # Start training
        self.start_time = time.time()
        
        try:
            logger.info("🚀 Starting training process...")
            results = self.model.train(**train_params)
            
            # Training completed successfully
            training_time = time.time() - self.start_time
            logger.info("=" * 60)
            logger.info("✅ TRAINING COMPLETED SUCCESSFULLY!")
            logger.info(f"⏱️  Total training time: {training_time/3600:.2f} hours")
            logger.info("=" * 60)
            
            # Save training statistics
            self._save_training_stats(results, training_time)
            
            # Copy best weights to RNA/weights directory
            self._copy_best_weights()
            
            return results
            
        except KeyboardInterrupt:
            logger.info("🛑 Training interrupted by user")
            return None
        except Exception as e:
            logger.error(f"❌ Training failed: {e}")
            raise
            
    def _save_training_stats(self, results, training_time: float):
        """Save training statistics"""
        stats = {
            'training_completed': datetime.now().isoformat(),
            'training_time_hours': training_time / 3600,
            'total_epochs': getattr(results, 'epoch', 'unknown'),
            'best_fitness': getattr(results, 'best_fitness', 'unknown'),
            'device': self.device,
            'config': self.config
        }
        
        stats_file = self.logs_dir / 'training_stats.json'
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
            
        logger.info(f"Training statistics saved to {stats_file}")
        
    def _copy_best_weights(self):
        """Copy best weights to RNA/weights directory"""
        # Find best weights file
        runs_dir = self.output_dir / "sai_detector_training"
        weights_dir = runs_dir / "weights"
        
        if weights_dir.exists():
            best_pt = weights_dir / "best.pt"
            last_pt = weights_dir / "last.pt"
            
            # Copy to RNA/weights
            rna_weights = Path(__file__).parent.parent / "weights"
            rna_weights.mkdir(exist_ok=True)
            
            if best_pt.exists():
                import shutil
                dest = rna_weights / "detector_best.pt"
                shutil.copy2(best_pt, dest)
                logger.info(f"✅ Best weights copied to {dest}")
                
            if last_pt.exists():
                import shutil
                dest = rna_weights / "detector_last.pt"
                shutil.copy2(last_pt, dest)
                logger.info(f"✅ Last weights copied to {dest}")


def load_config(config_path: str) -> Dict[str, Any]:
    """Load training configuration"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def estimate_training_time(config: Dict[str, Any], dataset_size: int) -> float:
    """Estimate training time based on configuration"""
    # Base estimate for RTX 3090 at 1440×808
    epochs = config['training']['detector']['epochs']
    batch_size = config['training']['detector']['batch_size']
    
    # Estimates based on performance_estimates.md
    time_per_epoch_hours = 0.15  # ~9 minutes per epoch for 173K images
    estimated_hours = epochs * time_per_epoch_hours
    
    logger.info(f"📊 Training time estimate:")
    logger.info(f"   Dataset size: {dataset_size:,} images")
    logger.info(f"   Epochs: {epochs}")
    logger.info(f"   Batch size: {batch_size}")
    logger.info(f"   Estimated time: {estimated_hours:.1f} hours")
    
    return estimated_hours


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8-s detector for SAI")
    parser.add_argument('--config', 
                       default='RNA/configs/sai_cascade_config.yaml',
                       help='Path to training configuration file')
    parser.add_argument('--data-root',
                       default='RNA/data/raw',
                       help='Root directory containing datasets')
    parser.add_argument('--prepare-data',
                       action='store_true',
                       help='Prepare dataset before training')
    parser.add_argument('--auto-resume',
                       action='store_true',
                       help='Automatically resume training if interrupted')
    
    args = parser.parse_args()
    
    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Configuration file not found: {config_path}")
        sys.exit(1)
        
    config = load_config(config_path)
    
    # Estimate dataset size (from our 173K images)
    dataset_size = 173251
    estimated_time = estimate_training_time(config, dataset_size)
    
    # Confirm training start
    logger.info("🔥 SAI YOLOv8-s Detector Training Pipeline")
    logger.info(f"Configuration: {config_path}")
    logger.info(f"Estimated training time: {estimated_time:.1f} hours")
    
    # Prepare dataset if requested
    if args.prepare_data:
        logger.info("Preparing dataset...")
        preparer = DatasetPreparer(args.data_root, "RNA/data/processed/yolo_dataset")
        dataset_yaml = preparer.prepare_combined_dataset()
        preparer.convert_datasets()
    else:
        # Use existing dataset configuration
        dataset_yaml = "RNA/data/processed/yolo_dataset/dataset.yaml"
        if not Path(dataset_yaml).exists():
            logger.error(f"Dataset configuration not found: {dataset_yaml}")
            logger.error("Run with --prepare-data to create dataset")
            sys.exit(1)
    
    # Initialize trainer
    trainer = YOLOTrainer(config)
    
    # Start training
    try:
        results = trainer.train(dataset_yaml)
        
        if results:
            logger.info("🎉 Training completed successfully!")
            logger.info("Next step: Train SmokeyNet-Lite verifier")
            logger.info("Command: python RNA/training/verifier_trainer.py")
        else:
            logger.info("Training was interrupted")
            
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()