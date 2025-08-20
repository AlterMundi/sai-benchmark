"""
Dataset Loaders for SAI Training

Implements data loaders for various smoke/fire detection datasets:
- FIgLib: Temporal sequences with smoke evolution
- FASDD: Large-scale smoke detection with bounding boxes
- D-Fire: Fire detection dataset
- Nemo: Smoke detection in various conditions
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np
import cv2
from pathlib import Path
import json
import xml.etree.ElementTree as ET
from PIL import Image
import logging

logger = logging.getLogger(__name__)


class BaseFireDataset(Dataset):
    """Base class for fire/smoke detection datasets"""
    
    def __init__(
        self,
        root_dir: str,
        transform=None,
        target_transform=None,
        image_size: Tuple[int, int] = (640, 640)
    ):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.target_transform = target_transform
        self.image_size = image_size
        self.samples = []
        self._load_dataset()
    
    def _load_dataset(self):
        """Override in subclasses to load specific dataset format"""
        raise NotImplementedError
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """Override in subclasses for specific data format"""
        raise NotImplementedError
    
    def _load_image(self, image_path: Union[str, Path]) -> np.ndarray:
        """Load and preprocess image"""
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Resize if needed
            if self.image_size:
                image = cv2.resize(image, self.image_size)
            
            return image
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            # Return dummy image
            return np.zeros((*self.image_size, 3), dtype=np.uint8)


class FIgLibDataset(BaseFireDataset):
    """
    FIgLib Dataset Loader for Temporal Sequences
    
    Dataset structure:
    - Contains temporal sequences of wildfire smoke
    - Used for training temporal verification models
    - Provides sequences with ground truth labels
    """
    
    def __init__(
        self,
        root_dir: str,
        sequence_length: int = 3,
        transform=None,
        target_transform=None,
        image_size: Tuple[int, int] = (224, 224),
        min_frames: int = 2
    ):
        self.sequence_length = sequence_length
        self.min_frames = min_frames
        super().__init__(root_dir, transform, target_transform, image_size)
    
    def _load_dataset(self):
        """Load FIgLib dataset structure"""
        # Look for sequence directories
        sequence_dirs = [d for d in self.root_dir.iterdir() if d.is_dir()]
        
        for seq_dir in sequence_dirs:
            # Find images in sequence
            image_files = []
            for ext in ['*.jpg', '*.jpeg', '*.png']:
                image_files.extend(list(seq_dir.glob(ext)))
            
            if len(image_files) < self.min_frames:
                continue
            
            # Sort by filename (assuming temporal order)
            image_files.sort()
            
            # Look for ground truth file
            gt_file = seq_dir / 'ground_truth.json'
            if not gt_file.exists():
                # Try alternative naming
                gt_file = seq_dir / f'{seq_dir.name}.json'
            
            ground_truth = self._load_ground_truth(gt_file)
            
            # Create temporal sequences
            for i in range(0, len(image_files) - self.sequence_length + 1):
                sequence_images = image_files[i:i + self.sequence_length]
                
                self.samples.append({
                    'sequence_id': seq_dir.name,
                    'images': sequence_images,
                    'ground_truth': ground_truth,
                    'start_frame': i
                })
    
    def _load_ground_truth(self, gt_file: Path) -> Dict[str, Any]:
        """Load ground truth annotations"""
        if gt_file.exists():
            try:
                with open(gt_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        
        # Default ground truth (infer from directory name or filename)
        return {
            'has_smoke': 'smoke' in str(gt_file).lower() or 'fire' in str(gt_file).lower(),
            'confidence': 1.0
        }
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load sequence of images
        sequence = []
        for img_path in sample['images']:
            image = self._load_image(img_path)
            if self.transform:
                image = self.transform(image)
            sequence.append(image)
        
        # Stack sequence
        sequence = torch.stack(sequence) if isinstance(sequence[0], torch.Tensor) else np.array(sequence)
        
        # Prepare target
        target = {
            'has_smoke': int(sample['ground_truth']['has_smoke']),
            'sequence_id': sample['sequence_id'],
            'start_frame': sample['start_frame']
        }
        
        if self.target_transform:
            target = self.target_transform(target)
        
        return sequence, target


class FASSDDataset(BaseFireDataset):
    """
    FASDD Dataset Loader for Object Detection
    
    Dataset structure:
    - Large-scale smoke detection dataset
    - Contains bounding box annotations
    - Used for training detection models
    """
    
    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        transform=None,
        target_transform=None,
        image_size: Tuple[int, int] = (640, 640)
    ):
        self.split = split
        super().__init__(root_dir, transform, target_transform, image_size)
    
    def _load_dataset(self):
        """Load FASDD dataset annotations"""
        # Look for annotation files
        ann_dir = self.root_dir / 'annotations' / self.split
        img_dir = self.root_dir / 'images' / self.split
        
        if not ann_dir.exists() or not img_dir.exists():
            logger.warning(f"FASDD dataset directories not found in {self.root_dir}")
            return
        
        # Load annotations
        for ann_file in ann_dir.glob('*.json'):
            try:
                with open(ann_file, 'r') as f:
                    annotation = json.load(f)
                
                # Find corresponding image
                img_file = img_dir / f"{ann_file.stem}.jpg"
                if not img_file.exists():
                    img_file = img_dir / f"{ann_file.stem}.png"
                
                if img_file.exists():
                    self.samples.append({
                        'image_path': img_file,
                        'annotation': annotation,
                        'image_id': ann_file.stem
                    })
            except Exception as e:
                logger.error(f"Error loading annotation {ann_file}: {e}")
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load image
        image = self._load_image(sample['image_path'])
        
        # Load annotations
        annotation = sample['annotation']
        
        # Extract bounding boxes and labels
        boxes = []
        labels = []
        
        for obj in annotation.get('objects', []):
            bbox = obj.get('bbox', [])
            if len(bbox) >= 4:
                # Convert to normalized coordinates if needed
                x1, y1, x2, y2 = bbox[:4]
                if max(bbox) > 1.0:  # Absolute coordinates
                    h, w = image.shape[:2]
                    x1, x2 = x1 / w, x2 / w
                    y1, y2 = y1 / h, y2 / h
                
                boxes.append([x1, y1, x2, y2])
                
                # Map class names to indices
                class_name = obj.get('class', 'smoke').lower()
                if 'fire' in class_name:
                    labels.append(1)  # Fire class
                else:
                    labels.append(0)  # Smoke class
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Prepare target
        target = {
            'boxes': torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros(0, 4),
            'labels': torch.tensor(labels, dtype=torch.long) if labels else torch.zeros(0, dtype=torch.long),
            'image_id': sample['image_id']
        }
        
        if self.target_transform:
            target = self.target_transform(target)
        
        return image, target


class SmokeDetectionDataset(BaseFireDataset):
    """
    Generic Smoke Detection Dataset
    
    Flexible loader for various smoke detection datasets including:
    - D-Fire
    - Nemo  
    - PyroNear-2024
    """
    
    def __init__(
        self,
        root_dir: str,
        annotation_format: str = 'yolo',  # 'yolo', 'coco', 'pascal'
        transform=None,
        target_transform=None,
        image_size: Tuple[int, int] = (640, 640),
        class_mapping: Optional[Dict[str, int]] = None
    ):
        self.annotation_format = annotation_format
        self.class_mapping = class_mapping or {'smoke': 0, 'fire': 1}
        super().__init__(root_dir, transform, target_transform, image_size)
    
    def _load_dataset(self):
        """Load dataset based on annotation format"""
        if self.annotation_format == 'yolo':
            self._load_yolo_format()
        elif self.annotation_format == 'coco':
            self._load_coco_format()
        elif self.annotation_format == 'pascal':
            self._load_pascal_format()
        else:
            raise ValueError(f"Unsupported annotation format: {self.annotation_format}")
    
    def _load_yolo_format(self):
        """Load YOLO format annotations"""
        images_dir = self.root_dir / 'images'
        labels_dir = self.root_dir / 'labels'
        
        if not images_dir.exists() or not labels_dir.exists():
            logger.warning(f"YOLO format directories not found in {self.root_dir}")
            return
        
        for img_file in images_dir.glob('*.jpg'):
            label_file = labels_dir / f'{img_file.stem}.txt'
            
            if label_file.exists():
                self.samples.append({
                    'image_path': img_file,
                    'label_path': label_file,
                    'image_id': img_file.stem
                })
    
    def _load_coco_format(self):
        """Load COCO format annotations"""
        ann_file = self.root_dir / 'annotations.json'
        
        if not ann_file.exists():
            logger.warning(f"COCO annotation file not found: {ann_file}")
            return
        
        with open(ann_file, 'r') as f:
            coco_data = json.load(f)
        
        # Process COCO annotations
        images = {img['id']: img for img in coco_data['images']}
        
        for img_id, img_info in images.items():
            img_file = self.root_dir / 'images' / img_info['file_name']
            
            if img_file.exists():
                # Find annotations for this image
                img_annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] == img_id]
                
                self.samples.append({
                    'image_path': img_file,
                    'annotations': img_annotations,
                    'image_info': img_info,
                    'image_id': img_id
                })
    
    def _load_pascal_format(self):
        """Load Pascal VOC format annotations"""
        images_dir = self.root_dir / 'images'
        annotations_dir = self.root_dir / 'annotations'
        
        for img_file in images_dir.glob('*.jpg'):
            xml_file = annotations_dir / f'{img_file.stem}.xml'
            
            if xml_file.exists():
                self.samples.append({
                    'image_path': img_file,
                    'xml_path': xml_file,
                    'image_id': img_file.stem
                })
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load image
        image = self._load_image(sample['image_path'])
        
        # Load annotations based on format
        if self.annotation_format == 'yolo':
            target = self._parse_yolo_annotation(sample)
        elif self.annotation_format == 'coco':
            target = self._parse_coco_annotation(sample)
        elif self.annotation_format == 'pascal':
            target = self._parse_pascal_annotation(sample)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        if self.target_transform:
            target = self.target_transform(target)
        
        return image, target
    
    def _parse_yolo_annotation(self, sample) -> Dict[str, torch.Tensor]:
        """Parse YOLO format annotation"""
        boxes = []
        labels = []
        
        try:
            with open(sample['label_path'], 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        x_center, y_center, width, height = map(float, parts[1:5])
                        
                        # Convert from center format to corner format
                        x1 = x_center - width / 2
                        y1 = y_center - height / 2
                        x2 = x_center + width / 2
                        y2 = y_center + height / 2
                        
                        boxes.append([x1, y1, x2, y2])
                        labels.append(class_id)
        except Exception as e:
            logger.error(f"Error parsing YOLO annotation {sample['label_path']}: {e}")
        
        return {
            'boxes': torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros(0, 4),
            'labels': torch.tensor(labels, dtype=torch.long) if labels else torch.zeros(0, dtype=torch.long),
            'image_id': sample['image_id']
        }
    
    def _parse_coco_annotation(self, sample) -> Dict[str, torch.Tensor]:
        """Parse COCO format annotation"""
        boxes = []
        labels = []
        
        for ann in sample['annotations']:
            bbox = ann['bbox']  # [x, y, width, height]
            x1, y1, w, h = bbox
            x2, y2 = x1 + w, y1 + h
            
            # Normalize coordinates
            img_w, img_h = sample['image_info']['width'], sample['image_info']['height']
            x1, x2 = x1 / img_w, x2 / img_w
            y1, y2 = y1 / img_h, y2 / img_h
            
            boxes.append([x1, y1, x2, y2])
            labels.append(ann['category_id'])
        
        return {
            'boxes': torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros(0, 4),
            'labels': torch.tensor(labels, dtype=torch.long) if labels else torch.zeros(0, dtype=torch.long),
            'image_id': sample['image_id']
        }
    
    def _parse_pascal_annotation(self, sample) -> Dict[str, torch.Tensor]:
        """Parse Pascal VOC format annotation"""
        boxes = []
        labels = []
        
        try:
            tree = ET.parse(sample['xml_path'])
            root = tree.getroot()
            
            # Get image dimensions
            size = root.find('size')
            img_width = int(size.find('width').text)
            img_height = int(size.find('height').text)
            
            # Parse objects
            for obj in root.findall('object'):
                class_name = obj.find('name').text.lower()
                class_id = self.class_mapping.get(class_name, 0)
                
                bbox = obj.find('bndbox')
                x1 = float(bbox.find('xmin').text) / img_width
                y1 = float(bbox.find('ymin').text) / img_height
                x2 = float(bbox.find('xmax').text) / img_width
                y2 = float(bbox.find('ymax').text) / img_height
                
                boxes.append([x1, y1, x2, y2])
                labels.append(class_id)
        except Exception as e:
            logger.error(f"Error parsing Pascal annotation {sample['xml_path']}: {e}")
        
        return {
            'boxes': torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros(0, 4),
            'labels': torch.tensor(labels, dtype=torch.long) if labels else torch.zeros(0, dtype=torch.long),
            'image_id': sample['image_id']
        }


class TemporalSequenceDataset(Dataset):
    """
    Dataset for temporal sequence processing
    
    Combines multiple datasets and creates temporal sequences for training
    the temporal verification model.
    """
    
    def __init__(
        self,
        datasets: List[BaseFireDataset],
        sequence_length: int = 3,
        temporal_stride: int = 1,
        transform=None
    ):
        self.datasets = datasets
        self.sequence_length = sequence_length
        self.temporal_stride = temporal_stride
        self.transform = transform
        
        self.sequences = self._create_sequences()
    
    def _create_sequences(self) -> List[Dict[str, Any]]:
        """Create temporal sequences from individual frames"""
        sequences = []
        
        for dataset in self.datasets:
            if isinstance(dataset, FIgLibDataset):
                # FIgLib already has sequences
                for i in range(len(dataset)):
                    sequence, target = dataset[i]
                    sequences.append({
                        'sequence': sequence,
                        'target': target,
                        'dataset_type': 'figlib'
                    })
            else:
                # Create synthetic sequences from detection datasets
                # Group images by scene/similarity
                # This is a simplified version - in practice, you'd want more sophisticated grouping
                dataset_samples = [dataset[i] for i in range(min(len(dataset), 1000))]  # Limit for demo
                
                for i in range(0, len(dataset_samples) - self.sequence_length + 1, self.temporal_stride):
                    sequence_images = []
                    has_smoke = False
                    
                    for j in range(self.sequence_length):
                        image, target = dataset_samples[i + j]
                        sequence_images.append(image)
                        
                        # Check if any frame has smoke/fire
                        if target['labels'].numel() > 0:
                            has_smoke = True
                    
                    sequences.append({
                        'sequence': torch.stack(sequence_images) if isinstance(sequence_images[0], torch.Tensor) else np.array(sequence_images),
                        'target': {'has_smoke': int(has_smoke)},
                        'dataset_type': 'detection'
                    })
        
        return sequences
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sample = self.sequences[idx]
        
        sequence = sample['sequence']
        target = sample['target']
        
        if self.transform:
            # Apply transform to each frame in sequence
            if isinstance(sequence, torch.Tensor):
                transformed_sequence = []
                for frame in sequence:
                    transformed_sequence.append(self.transform(frame))
                sequence = torch.stack(transformed_sequence)
            else:
                sequence = np.array([self.transform(frame) for frame in sequence])
        
        return sequence, target


def create_dataloaders(
    datasets_config: Dict[str, Any],
    batch_size: int = 16,
    num_workers: int = 4,
    train_transform=None,
    val_transform=None
) -> Dict[str, DataLoader]:
    """
    Create dataloaders for training and validation
    
    Args:
        datasets_config: Configuration for datasets
        batch_size: Batch size for training
        num_workers: Number of worker processes
        train_transform: Training data transforms
        val_transform: Validation data transforms
        
    Returns:
        Dictionary with train/val dataloaders
    """
    dataloaders = {}
    
    # Create datasets based on configuration
    for split, config in datasets_config.items():
        datasets = []
        
        for dataset_type, dataset_config in config.items():
            if dataset_type == 'figlib':
                dataset = FIgLibDataset(
                    root_dir=dataset_config['path'],
                    sequence_length=dataset_config.get('sequence_length', 3),
                    transform=train_transform if split == 'train' else val_transform
                )
            elif dataset_type == 'fasdd':
                dataset = FASSDDataset(
                    root_dir=dataset_config['path'],
                    split=split,
                    transform=train_transform if split == 'train' else val_transform
                )
            elif dataset_type in ['dfire', 'nemo', 'pyronear']:
                dataset = SmokeDetectionDataset(
                    root_dir=dataset_config['path'],
                    annotation_format=dataset_config.get('format', 'yolo'),
                    transform=train_transform if split == 'train' else val_transform
                )
            
            if dataset:
                datasets.append(dataset)
        
        # Create combined dataset if multiple datasets
        if len(datasets) == 1:
            combined_dataset = datasets[0]
        else:
            # Combine datasets (simple concatenation)
            from torch.utils.data import ConcatDataset
            combined_dataset = ConcatDataset(datasets)
        
        # Create dataloader
        dataloaders[split] = DataLoader(
            combined_dataset,
            batch_size=batch_size,
            shuffle=(split == 'train'),
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=(split == 'train')
        )
    
    return dataloaders


if __name__ == "__main__":
    # Test dataset loaders
    print("Testing dataset loaders...")
    
    # Test FIgLib dataset
    figlib_dataset = FIgLibDataset(
        root_dir="./test_sequences",
        sequence_length=3
    )
    print(f"FIgLib dataset size: {len(figlib_dataset)}")
    
    # Test FASDD dataset
    fasdd_dataset = FASSDDataset(
        root_dir="./fasdd_data",
        split="train"
    )
    print(f"FASDD dataset size: {len(fassd_dataset)}")
    
    print("Dataset loaders test completed.")