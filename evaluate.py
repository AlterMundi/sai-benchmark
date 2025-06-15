#!/usr/bin/env python3
"""
Qwen 2.5-VL Early-Fire Detection Benchmark
Main evaluation pipeline for wildfire image sequences.
"""

import argparse
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import sys

from tqdm import tqdm

# Import model wrappers
try:
    from models.ollama_qwen import infer as ollama_infer, check_connection as ollama_check
    from models.hf_qwen import infer as hf_infer, check_gpu_available as hf_check_gpu
except ImportError as e:
    print(f"Error importing models: {e}")
    print("Make sure to install dependencies: pip install -r requirements.txt")
    sys.exit(1)


@dataclass 
class SequenceResult:
    """Results for a single image sequence."""
    sequence_name: str
    total_images: int
    predictions: List[Dict]
    ground_truth_labels: List[int]  # 1=smoke, 0=no smoke
    processing_time: float
    early_fire_score: Optional[float] = None
    

@dataclass
class BenchmarkConfig:
    """Configuration for benchmark execution."""
    dataset_path: Path
    engine: str  # 'ollama', 'hf', or 'both'
    output_path: Path
    max_workers: int = 4
    temperature: float = 0.1
    fps_sampling: int = 1  # Process every N frames
    max_images_per_seq: int = 50
    iou_threshold: float = 0.4


class WildfireEvaluator:
    """Main evaluator for wildfire detection using Qwen 2.5-VL."""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.results = []
        self.setup_logging()
        
    def setup_logging(self):
        """Configure logging."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def discover_sequences(self) -> List[Path]:
        """Discover all image sequences in the dataset."""
        sequences = []
        dataset_path = Path(self.config.dataset_path)
        
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset path not found: {dataset_path}")
            
        # Find all directories containing images
        for seq_dir in dataset_path.iterdir():
            if seq_dir.is_dir():
                # Check if directory contains images
                image_files = list(seq_dir.glob("*.jpg")) + \
                             list(seq_dir.glob("*.jpeg")) + \
                             list(seq_dir.glob("*.png"))
                
                if image_files:
                    sequences.append(seq_dir)
                    
        self.logger.info(f"Found {len(sequences)} sequences in {dataset_path}")
        return sorted(sequences)
        
    def parse_ground_truth(self, sequence_dir: Path) -> List[int]:
        """
        Parse ground truth labels for a sequence.
        Assumes sec##-1 = has fire, sec##-0 = no fire pattern.
        Also checks for .txt files with ground truth.
        """
        labels = []
        
        # Get all image files
        image_files = sorted(
            list(sequence_dir.glob("*.jpg")) + 
            list(sequence_dir.glob("*.jpeg")) + 
            list(sequence_dir.glob("*.png"))
        )
        
        # Method 1: Check directory name pattern (sec##-1 vs sec##-0)
        seq_name = sequence_dir.name
        has_fire_in_name = seq_name.endswith('-1')
        
        # Method 2: Check for corresponding .txt files
        for img_file in image_files:
            txt_file = img_file.with_suffix('.txt')
            
            if txt_file.exists():
                # If .txt file exists and is non-empty, assume it contains fire info
                try:
                    content = txt_file.read_text().strip()
                    # Non-empty txt = has smoke/fire, empty = no smoke/fire
                    has_fire = len(content) > 0
                    labels.append(1 if has_fire else 0)
                except:
                    # Fallback to directory name pattern
                    labels.append(1 if has_fire_in_name else 0)
            else:
                # Fallback to directory name pattern
                labels.append(1 if has_fire_in_name else 0)
                
        return labels
        
    def sample_sequence(self, image_files: List[Path]) -> List[Path]:
        """Sample images from sequence based on FPS sampling and max images."""
        if self.config.fps_sampling > 1:
            # Sample every N frames
            sampled = image_files[::self.config.fps_sampling]
        else:
            sampled = image_files
            
        # Limit to max images
        if len(sampled) > self.config.max_images_per_seq:
            sampled = sampled[:self.config.max_images_per_seq]
            
        return sampled
        
    def run_inference(self, image_path: Path, engine: str) -> Dict:
        """Run inference on a single image using specified engine."""
        try:
            if engine == 'ollama':
                return ollama_infer(image_path, temperature=self.config.temperature)
            elif engine == 'hf':
                return hf_infer(image_path, temperature=self.config.temperature)
            else:
                raise ValueError(f"Unknown engine: {engine}")
        except Exception as e:
            self.logger.error(f"Inference failed for {image_path} with {engine}: {e}")
            return {
                "has_smoke": False,
                "bbox": [0, 0, 0, 0],
                "error": str(e)
            }
            
    def process_sequence(self, sequence_dir: Path, engine: str) -> SequenceResult:
        """Process a single image sequence."""
        start_time = time.time()
        
        # Get image files
        image_files = sorted(
            list(sequence_dir.glob("*.jpg")) + 
            list(sequence_dir.glob("*.jpeg")) + 
            list(sequence_dir.glob("*.png"))
        )
        
        if not image_files:
            self.logger.warning(f"No images found in {sequence_dir}")
            return SequenceResult(
                sequence_name=sequence_dir.name,
                total_images=0,
                predictions=[],
                ground_truth_labels=[],
                processing_time=0
            )
            
        # Sample images
        sampled_images = self.sample_sequence(image_files)
        
        # Get ground truth
        ground_truth = self.parse_ground_truth(sequence_dir)
        
        # Align ground truth with sampled images
        if self.config.fps_sampling > 1:
            gt_sampled = ground_truth[::self.config.fps_sampling]
        else:
            gt_sampled = ground_truth
            
        if len(gt_sampled) > len(sampled_images):
            gt_sampled = gt_sampled[:len(sampled_images)]
        elif len(gt_sampled) < len(sampled_images):
            # Pad with last label
            last_label = gt_sampled[-1] if gt_sampled else 0
            gt_sampled.extend([last_label] * (len(sampled_images) - len(gt_sampled)))
            
        # Run inference on sampled images
        predictions = []
        
        with tqdm(sampled_images, desc=f"Processing {sequence_dir.name}", leave=False) as pbar:
            for img_path in pbar:
                pred = self.run_inference(img_path, engine)
                pred['image_path'] = str(img_path)
                predictions.append(pred)
                
        processing_time = time.time() - start_time
        
        # Calculate Early-Fire Score
        early_fire_score = self.calculate_early_fire_score(predictions, gt_sampled)
        
        return SequenceResult(
            sequence_name=sequence_dir.name,
            total_images=len(sampled_images),
            predictions=predictions,
            ground_truth_labels=gt_sampled,
            processing_time=processing_time,
            early_fire_score=early_fire_score
        )
        
    def calculate_early_fire_score(self, predictions: List[Dict], ground_truth: List[int]) -> float:
        """
        Calculate Early-Fire Detection Score based on:
        - Detection accuracy
        - Early detection bonus
        - False positive penalty
        """
        if not predictions or not ground_truth:
            return 0.0
            
        # Basic accuracy
        correct = 0
        total = min(len(predictions), len(ground_truth))
        
        for i in range(total):
            pred_smoke = predictions[i].get('has_smoke', False)
            true_smoke = ground_truth[i] == 1
            
            if pred_smoke == true_smoke:
                correct += 1
                
        accuracy = correct / total if total > 0 else 0
        
        # Early detection bonus: reward early fire detection
        early_bonus = 0
        fire_detected_at = None
        fire_starts_at = None
        
        # Find when fire actually starts
        for i, label in enumerate(ground_truth):
            if label == 1:
                fire_starts_at = i
                break
                
        # Find when model first detects fire
        for i, pred in enumerate(predictions):
            if pred.get('has_smoke', False):
                fire_detected_at = i
                break
                
        # Calculate early detection bonus
        if fire_starts_at is not None and fire_detected_at is not None:
            if fire_detected_at <= fire_starts_at:
                # Bonus for detecting at or before actual fire start
                early_bonus = 0.2 * (1 - fire_detected_at / len(predictions))
            else:
                # Penalty for late detection
                delay = fire_detected_at - fire_starts_at
                early_bonus = -0.1 * (delay / len(predictions))
                
        # Final score (0-1 scale)
        score = accuracy + early_bonus
        return max(0, min(1, score))
        
    def check_backends(self) -> Dict[str, bool]:
        """Check availability of inference backends."""
        backends = {}
        
        # Check Ollama
        backends['ollama'] = ollama_check()
        if not backends['ollama']:
            self.logger.warning("Ollama backend not available")
            
        # Check HuggingFace
        backends['hf'] = hf_check_gpu()
        if not backends['hf']:
            self.logger.warning("HuggingFace GPU backend not available")
            
        return backends
        
    def run_evaluation(self) -> Dict:
        """Run the complete evaluation pipeline."""
        self.logger.info("Starting Qwen 2.5-VL Early-Fire Detection Benchmark")
        
        # Check backends
        backends = self.check_backends()
        
        engines_to_test = []
        if self.config.engine == 'both':
            if backends['ollama']:
                engines_to_test.append('ollama')
            if backends['hf']:
                engines_to_test.append('hf')
        else:
            if self.config.engine in backends and backends[self.config.engine]:
                engines_to_test.append(self.config.engine)
            else:
                raise ValueError(f"Backend {self.config.engine} not available")
                
        if not engines_to_test:
            raise ValueError("No backends available")
            
        # Discover sequences
        sequences = self.discover_sequences()
        
        # Process all sequences with all engines
        all_results = {}
        
        for engine in engines_to_test:
            self.logger.info(f"Running evaluation with {engine} backend")
            engine_results = []
            
            with tqdm(sequences, desc=f"Sequences ({engine})") as pbar:
                for seq_dir in pbar:
                    pbar.set_description(f"Processing {seq_dir.name} ({engine})")
                    result = self.process_sequence(seq_dir, engine)
                    engine_results.append(result)
                    
            all_results[engine] = engine_results
            
        # Generate summary
        summary = self.generate_summary(all_results)
        
        # Save results
        self.save_results(all_results, summary)
        
        return {
            'results': all_results,
            'summary': summary,
            'config': self.config.__dict__
        }
        
    def generate_summary(self, all_results: Dict) -> Dict:
        """Generate benchmark summary statistics."""
        summary = {}
        
        for engine, results in all_results.items():
            if not results:
                continue
                
            # Calculate aggregate metrics
            total_images = sum(r.total_images for r in results)
            total_time = sum(r.processing_time for r in results)
            avg_score = sum(r.early_fire_score or 0 for r in results) / len(results)
            
            # Calculate detection stats
            true_positives = 0
            false_positives = 0
            true_negatives = 0
            false_negatives = 0
            
            for result in results:
                for i, pred in enumerate(result.predictions):
                    if i < len(result.ground_truth_labels):
                        pred_smoke = pred.get('has_smoke', False)
                        true_smoke = result.ground_truth_labels[i] == 1
                        
                        if pred_smoke and true_smoke:
                            true_positives += 1
                        elif pred_smoke and not true_smoke:
                            false_positives += 1
                        elif not pred_smoke and not true_smoke:
                            true_negatives += 1
                        elif not pred_smoke and true_smoke:
                            false_negatives += 1
                            
            # Calculate metrics
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            accuracy = (true_positives + true_negatives) / (true_positives + false_positives + true_negatives + false_negatives) if (true_positives + false_positives + true_negatives + false_negatives) > 0 else 0
            
            summary[engine] = {
                'total_sequences': len(results),
                'total_images': total_images,
                'total_time_seconds': total_time,
                'avg_time_per_image': total_time / total_images if total_images > 0 else 0,
                'avg_early_fire_score': avg_score,
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score,
                'accuracy': accuracy,
                'true_positives': true_positives,
                'false_positives': false_positives,
                'true_negatives': true_negatives,
                'false_negatives': false_negatives
            }
            
        return summary
        
    def save_results(self, all_results: Dict, summary: Dict):
        """Save results to JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.config.output_path / f"qwen_evaluation_{timestamp}.json"
        
        # Convert results to serializable format
        serializable_results = {}
        for engine, results in all_results.items():
            serializable_results[engine] = []
            for result in results:
                serializable_results[engine].append({
                    'sequence_name': result.sequence_name,
                    'total_images': result.total_images,
                    'predictions': result.predictions,
                    'ground_truth_labels': result.ground_truth_labels,
                    'processing_time': result.processing_time,
                    'early_fire_score': result.early_fire_score
                })
                
        output_data = {
            'timestamp': timestamp,
            'config': {
                'dataset_path': str(self.config.dataset_path),
                'engine': self.config.engine,
                'max_workers': self.config.max_workers,
                'temperature': self.config.temperature,
                'fps_sampling': self.config.fps_sampling,
                'max_images_per_seq': self.config.max_images_per_seq,
                'iou_threshold': self.config.iou_threshold
            },
            'summary': summary,
            'results': serializable_results
        }
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
            
        self.logger.info(f"Results saved to {output_file}")
        
        # Print summary to console
        print("\n" + "="*60)
        print("BENCHMARK SUMMARY")
        print("="*60)
        
        for engine, stats in summary.items():
            print(f"\n{engine.upper()} Backend:")
            print(f"  Sequences: {stats['total_sequences']}")
            print(f"  Images: {stats['total_images']}")
            print(f"  Time: {stats['total_time_seconds']:.1f}s")
            print(f"  Avg/image: {stats['avg_time_per_image']:.2f}s")
            print(f"  Early-Fire Score: {stats['avg_early_fire_score']:.3f}")
            print(f"  Precision: {stats['precision']:.3f}")
            print(f"  Recall: {stats['recall']:.3f}")
            print(f"  F1-Score: {stats['f1_score']:.3f}")
            print(f"  Accuracy: {stats['accuracy']:.3f}")


def main():
    parser = argparse.ArgumentParser(description="Qwen 2.5-VL Early-Fire Detection Benchmark")
    parser.add_argument("--dataset", default="~/sequences", help="Path to image sequences dataset")
    parser.add_argument("--engine", choices=['ollama', 'hf', 'both'], default='ollama', 
                       help="Inference engine to use")
    parser.add_argument("--output", default="./out", help="Output directory for results")
    parser.add_argument("--workers", type=int, default=4, help="Number of worker threads")
    parser.add_argument("--temperature", type=float, default=0.1, help="Model temperature")
    parser.add_argument("--fps-sampling", type=int, default=1, help="Sample every N frames")
    parser.add_argument("--max-images", type=int, default=50, help="Max images per sequence")
    parser.add_argument("--iou-threshold", type=float, default=0.4, help="IOU threshold for bbox evaluation")
    
    args = parser.parse_args()
    
    # Create config
    config = BenchmarkConfig(
        dataset_path=Path(args.dataset).expanduser(),
        engine=args.engine,
        output_path=Path(args.output),
        max_workers=args.workers,
        temperature=args.temperature,
        fps_sampling=args.fps_sampling,
        max_images_per_seq=args.max_images,
        iou_threshold=args.iou_threshold
    )
    
    # Create output directory
    config.output_path.mkdir(exist_ok=True)
    
    # Run evaluation
    evaluator = WildfireEvaluator(config)
    
    try:
        results = evaluator.run_evaluation()
        print(f"\nEvaluation completed successfully!")
        return 0
    except Exception as e:
        print(f"Evaluation failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())