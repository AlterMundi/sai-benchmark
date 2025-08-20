#!/usr/bin/env python3
"""
SAI RNA Environment Setup Script

Sets up the development environment for SAI neural network training and inference.
"""

import subprocess
import sys
import os
from pathlib import Path
import yaml


def check_python_version():
    """Check Python version compatibility"""
    if sys.version_info < (3, 8):
        print("Error: Python 3.8 or higher is required")
        sys.exit(1)
    print(f"✓ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")


def check_cuda_availability():
    """Check CUDA availability"""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.cuda.get_device_name()}")
            print(f"  - CUDA version: {torch.version.cuda}")
            print(f"  - GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            return True
        else:
            print("⚠ CUDA not available - using CPU")
            return False
    except ImportError:
        print("⚠ PyTorch not installed - cannot check CUDA")
        return False


def install_dependencies():
    """Install required dependencies"""
    print("Installing dependencies...")
    
    # Core dependencies
    dependencies = [
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "ultralytics>=8.0.0",
        "opencv-python>=4.8.0",
        "numpy>=1.21.0",
        "pillow>=8.3.0",
        "albumentations>=1.3.0",
        "tensorboard>=2.10.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "tqdm>=4.64.0",
        "pyyaml>=6.0",
        "scikit-learn>=1.1.0",
        "scipy>=1.9.0"
    ]
    
    for dep in dependencies:
        print(f"Installing {dep}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", dep])


def create_directories():
    """Create necessary directories"""
    directories = [
        "RNA/weights",
        "RNA/logs",
        "RNA/checkpoints",
        "RNA/data/raw",
        "RNA/data/processed",
        "RNA/results",
        "RNA/experiments"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {directory}")


def download_pretrained_weights():
    """Download pretrained weights if needed"""
    print("Checking for pretrained weights...")
    
    weights_dir = Path("RNA/weights")
    
    # YOLOv8s weights
    yolo_weights = weights_dir / "yolov8s.pt"
    if not yolo_weights.exists():
        print("Downloading YOLOv8s weights...")
        try:
            from ultralytics import YOLO
            model = YOLO('yolov8s.pt')  # This will download if not present
            # Move to our weights directory
            import shutil
            shutil.move('yolov8s.pt', str(yolo_weights))
            print(f"✓ Downloaded YOLOv8s weights to {yolo_weights}")
        except Exception as e:
            print(f"⚠ Could not download YOLOv8s weights: {e}")
    else:
        print(f"✓ YOLOv8s weights found at {yolo_weights}")


def create_sample_config():
    """Create sample configuration files"""
    config_dir = Path("RNA/configs")
    config_dir.mkdir(exist_ok=True)
    
    # Sample training config
    sample_config = {
        'experiment_name': 'sai_cascade_test',
        'model': {
            'detector': {
                'architecture': 'yolov8s',
                'pretrained': True,
                'conf_threshold': 0.3
            },
            'verifier': {
                'architecture': 'smokeynet_lite',
                'sequence_length': 3,
                'hidden_dim': 256
            }
        },
        'training': {
            'epochs': 10,
            'batch_size': 4,
            'learning_rate': 0.001,
            'device': 'auto'
        },
        'data': {
            'train_split': 0.8,
            'val_split': 0.2,
            'image_size': [640, 640]
        }
    }
    
    config_file = config_dir / "sample_training.yaml"
    with open(config_file, 'w') as f:
        yaml.dump(sample_config, f, default_flow_style=False)
    
    print(f"✓ Created sample config: {config_file}")


def setup_logging():
    """Setup logging configuration"""
    logs_dir = Path("RNA/logs")
    logs_dir.mkdir(exist_ok=True)
    
    # Create logging config
    logging_config = """
version: 1
disable_existing_loggers: false

formatters:
  standard:
    format: '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
  detailed:
    format: '%(asctime)s - %(name)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s'

handlers:
  console:
    class: logging.StreamHandler
    level: INFO
    formatter: standard
    stream: ext://sys.stdout
  
  file:
    class: logging.FileHandler
    level: DEBUG
    formatter: detailed
    filename: RNA/logs/sai_rna.log
    mode: a

loggers:
  RNA:
    level: DEBUG
    handlers: [console, file]
    propagate: false

root:
  level: WARNING
  handlers: [console]
"""
    
    with open("RNA/configs/logging.yaml", 'w') as f:
        f.write(logging_config)
    
    print("✓ Created logging configuration")


def verify_installation():
    """Verify the installation"""
    print("\nVerifying installation...")
    
    try:
        # Test PyTorch
        import torch
        print(f"✓ PyTorch {torch.__version__}")
        
        # Test torchvision
        import torchvision
        print(f"✓ Torchvision {torchvision.__version__}")
        
        # Test ultralytics
        from ultralytics import YOLO
        print("✓ Ultralytics YOLO")
        
        # Test OpenCV
        import cv2
        print(f"✓ OpenCV {cv2.__version__}")
        
        # Test our modules
        sys.path.append(str(Path.cwd()))
        from RNA.models.detector.yolov8s_detector import YOLOv8sDetector
        from RNA.models.verifier.smokeynet_lite import SmokeyNetLite
        print("✓ SAI RNA modules")
        
        # Test simple model creation
        detector = YOLOv8sDetector(pretrained=False)
        verifier = SmokeyNetLite()
        print("✓ Model instantiation")
        
        print("\n🎉 Installation verification successful!")
        return True
        
    except Exception as e:
        print(f"\n❌ Installation verification failed: {e}")
        return False


def print_next_steps():
    """Print next steps for the user"""
    print("\n" + "="*60)
    print("SAI RNA ENVIRONMENT SETUP COMPLETE")
    print("="*60)
    print("\nNext steps:")
    print("1. Prepare your datasets in RNA/data/raw/")
    print("2. Update configurations in RNA/configs/")
    print("3. Start training:")
    print("   python RNA/training/detector_trainer.py --config RNA/configs/sample_training.yaml")
    print("\nFor testing:")
    print("   python RNA/scripts/test_installation.py")
    print("\nDocumentation:")
    print("   See RNA/README.md for detailed usage instructions")


def main():
    """Main setup function"""
    print("SAI RNA Environment Setup")
    print("=" * 40)
    
    # Check requirements
    check_python_version()
    
    # Install dependencies
    install_dependencies()
    
    # Check CUDA after torch installation
    has_cuda = check_cuda_availability()
    
    # Setup directories and files
    create_directories()
    download_pretrained_weights()
    create_sample_config()
    setup_logging()
    
    # Verify installation
    if verify_installation():
        print_next_steps()
    else:
        print("\n❌ Setup completed with errors. Please check the logs.")
        sys.exit(1)


if __name__ == "__main__":
    main()