#!/usr/bin/env python3
import os
import base64
import time
import re
import json
import logging
from io import BytesIO
import argparse
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path

# Import PIL for image processing
from PIL import Image

# Import ollama for local LLM inference
import ollama

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("image_sequence_test.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("image_sequence_test")

# Data models
class ImageInfo:
    def __init__(self, path: str, filename: Optional[str] = None):
        self.path = path
        self.filename = filename or os.path.basename(path)

class SequenceInfo:
    def __init__(self, folder_path: str, images: List[ImageInfo], ground_truth: bool):
        self.folder_path = folder_path
        self.images = images
        self.ground_truth = ground_truth

class SequenceResult:
    def __init__(self, sequence_id: str, prediction: bool, ground_truth: bool, 
                 confidence: Optional[float], justification: str, 
                 processing_time: float, raw_response: str):
        self.sequence_id = sequence_id
        self.prediction = prediction
        self.ground_truth = ground_truth
        self.confidence = confidence
        self.justification = justification
        self.processing_time = processing_time
        self.raw_response = raw_response
        # Calculate is_correct as a property, not a constructor parameter
        self.is_correct = (prediction == ground_truth)

class EvaluationMetrics:
    def __init__(self, results):
        total = len(results)
        
        true_positives = sum(1 for r in results if r.prediction and r.ground_truth)
        true_negatives = sum(1 for r in results if not r.prediction and not r.ground_truth)
        false_positives = sum(1 for r in results if r.prediction and not r.ground_truth)
        false_negatives = sum(1 for r in results if not r.prediction and r.ground_truth)
        
        self.accuracy = (true_positives + true_negatives) / total if total > 0 else 0
        
        self.precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        self.recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        
        self.f1_score = 2 * (self.precision * self.recall) / (self.precision + self.recall) if (self.precision + self.recall) > 0 else 0
        
        self.total_sequences = total
        self.true_positives = true_positives
        self.true_negatives = true_negatives
        self.false_positives = false_positives
        self.false_negatives = false_negatives
        self.total_processing_time = sum(r.processing_time for r in results)

    def to_dict(self):
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "total_sequences": self.total_sequences,
            "true_positives": self.true_positives,
            "true_negatives": self.true_negatives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "total_processing_time": self.total_processing_time
        }

# Example prompt (you should customize this for your specific detection task)
DEFAULT_PROMPT = """
You are a highly specialized visual analyst trained to detect specific visual patterns in image sequences.

You will now be shown a sequence of consecutive images captured by a fixed surveillance camera. These images represent a time lapse of the same scene. Your job is to carefully analyze the entire sequence, looking for visual indications of the target pattern or anomaly.

This task requires precision and attention to detail. Please focus on subtle changes, shifts in features, wisps, movement or texture that could imply the presence of the target pattern.

Please respond only with:

Judgment: [Yes/No] — Is there visual evidence of the target pattern or anomaly?
Confidence: [a number between 0.0 and 1.0]
Justification: [brief explanation of what you saw in the sequence that supports your judgment]
"""

# Function to convert image to base64
def image_to_base64(image_path):
    try:
        with Image.open(image_path) as img:
            buffered = BytesIO()
            img.save(buffered, format="JPEG")
            return base64.b64encode(buffered.getvalue()).decode("utf-8")
    except Exception as e:
        logger.error(f"Error converting image to base64: {str(e)}")
        raise Exception(f"Error converting image to base64: {str(e)}")

# Function to parse LLM response
def parse_response(output):
    judgment_match = re.search(r'Judg?ment:\s*(Yes|No|yes|no)', output, re.IGNORECASE)
    confidence_match = re.search(r'Confidence:\s*([0-9](?:\.\d+)?)', output)
    justification_match = re.search(r'Justification:\s*(.*)', output, re.DOTALL)

    judgment = judgment_match.group(1).lower() if judgment_match else "unknown"
    judgment_bool = True if judgment.lower() == "yes" else False
    confidence = float(confidence_match.group(1)) if confidence_match and confidence_match.group(1) else None
    justification = justification_match.group(1).strip() if justification_match else "Not provided"
    
    return judgment_bool, confidence, justification

# Function to read ground truth from txt file
def read_ground_truth(txt_path):
    try:
        with open(txt_path, 'r') as f:
            content = f.read().strip()
            # Assuming the content is a single number (0 or 1) or non-empty for true
            return content != ""
    except Exception as e:
        logger.error(f"Error reading ground truth file {txt_path}: {str(e)}")
        return None

# Function to get sequence folders in the dataset
def get_sequence_folders(dataset_path, max_sequences=None):
    try:
        if not os.path.isdir(dataset_path):
            raise Exception(f"Dataset path does not exist: {dataset_path}")
            
        folders = [f for f in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, f))]
        folders.sort()  # Sort for consistent ordering
        
        if max_sequences:
            folders = folders[:max_sequences]
        
        return folders
    except Exception as e:
        logger.error(f"Error getting sequence folders: {str(e)}")
        raise Exception(f"Error getting sequence folders: {str(e)}")

# Function to get images and ground truth for a sequence
def get_sequence_info(dataset_path, sequence_folder):
    sequence_path = os.path.join(dataset_path, sequence_folder)
    
    # Get all jpg/jpeg/png files
    valid_extensions = ['.jpg', '.jpeg', '.png']
    image_files = []
    
    for filename in os.listdir(sequence_path):
        if any(filename.lower().endswith(ext) for ext in valid_extensions):
            full_path = os.path.join(sequence_path, filename)
            image_files.append(ImageInfo(
                path=full_path,
                filename=filename
            ))
    
    # Sort images to ensure consistent ordering
    image_files.sort(key=lambda x: x.filename)
    
    if not image_files:
        raise Exception(f"No image files found in sequence folder: {sequence_path}")
    
    # Initialize ground truth as False (0)
    ground_truth = False
    
    # Check each image's corresponding txt file
    for img_info in image_files:
        txt_filename = os.path.splitext(img_info.filename)[0] + ".txt"
        txt_path = os.path.join(sequence_path, txt_filename)
        
        if os.path.exists(txt_path):
            current_gt = read_ground_truth(txt_path)
            if current_gt is not None and current_gt:
                # If any image has a positive ground truth, the whole sequence is marked as positive
                ground_truth = True
                break
    
    logger.info(f"Ground truth for sequence {sequence_folder}: {ground_truth}")

    return SequenceInfo(
        folder_path=sequence_path,
        images=image_files,
        ground_truth=ground_truth
    )

# Function to get model information - more robust handling
def get_model_info(model_name):
    try:
        # Get the list of models
        models_response = ollama.list()
        
        # Check if we have a proper response with models
        if not isinstance(models_response, dict):
            logger.error(f"Unexpected response from ollama.list(): {models_response}")
            return None
            
        # Get the models array safely
        models = models_response.get("models", [])
        if not models:
            logger.warning("No models found in Ollama")
            return None
            
        # Find our model
        for model in models:
            if isinstance(model, dict) and model.get("name") == model_name:
                return model
                
        logger.warning(f"Model '{model_name}' not found in Ollama's model list")
        return None
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        return None

# Function to analyze a single sequence
async def analyze_sequence(sequence_info, model, prompt):
    start_time = time.time()
    
    # Convert all images to base64
    images_base64 = []
    for img_info in sequence_info.images:
        try:
            img_base64 = image_to_base64(img_info.path)
            images_base64.append({
                "filename": img_info.filename,
                "base64": img_base64
            })
        except Exception as e:
            logger.error(f"Error processing image {img_info.filename}: {str(e)}")
            raise Exception(f"Error processing image {img_info.filename}: {str(e)}")
    
    # Prepare message with all images
    message = {
        "role": "user", 
        "content": prompt,
        "images": [img["base64"] for img in images_base64]
    }
    
    # Call Ollama API with all images at once
    try:
        print(f"\n✓ Processing {len(images_base64)} images with model '{model}'")
        
        response = ollama.chat(model=model, messages=[message])
        output = response["message"]["content"]
        
        # Parse the response
        prediction, confidence, justification = parse_response(output)
        
        processing_time = time.time() - start_time
        
        # Get sequence ID from folder path
        sequence_id = os.path.basename(sequence_info.folder_path)
        
        # Create result object WITHOUT passing is_correct as a parameter
        return SequenceResult(
            sequence_id=sequence_id,
            prediction=prediction,
            ground_truth=sequence_info.ground_truth,
            confidence=confidence,
            justification=justification,
            processing_time=processing_time,
            raw_response=output
        )
    except Exception as e:
        logger.error(f"Error analyzing sequence {os.path.basename(sequence_info.folder_path)}: {str(e)}")
        raise Exception(f"Error analyzing sequence: {str(e)}")

# Main function to run the dataset test
async def run_dataset_test(dataset_path, model="gemma3:12b-it-q8_0", prompt=None, max_sequences=None):
    try:
        logger.info(f"Starting dataset test for: {dataset_path}")
        logger.info(f"Using model: {model}")
        
        # Use provided prompt or default
        test_prompt = prompt or DEFAULT_PROMPT
        
        # Get model information - handle errors gracefully
        model_info = get_model_info(model)
        
        print("\n" + "="*80)
        print(f"🔍 IMAGE SEQUENCE ANALYSIS TEST")
        print(f"Dataset: {os.path.basename(dataset_path)}")
        print(f"Model: {model}")
        print("="*80)
        
        # Print model information if available
        if model_info:
            print("\n📊 MODEL INFORMATION:")
            # Safely access model info with get() to avoid KeyErrors
            print(f"  Name: {model_info.get('name', 'Unknown')}")
            print(f"  Model ID: {model_info.get('model', 'Unknown')}")
            print(f"  Size: {model_info.get('size', 'Unknown')}")
            print(f"  Modified: {model_info.get('modified_at', 'Unknown')}")
            
            # Safely handle tags
            tags = model_info.get('tags', [])
            if tags and isinstance(tags, list):
                print(f"  Tags: {', '.join(tags)}")
            
            # Safely handle details
            details = model_info.get('details', {})
            if details and isinstance(details, dict):
                print(f"  Family: {details.get('family', 'Unknown')}")
                print(f"  Parameter Size: {details.get('parameter_size', 'Unknown')}")
                print(f"  Quantization Level: {details.get('quantization_level', 'Unknown')}")
        
        # Print the prompt being used
        print("\n📝 PROMPT:")
        print("-"*80)
        print(test_prompt)
        print("-"*80)
        
        # Get sequence folders
        sequence_folders = get_sequence_folders(dataset_path, max_sequences)
        print(f"\n📁 Found {len(sequence_folders)} sequence folders in dataset")
        
        all_results = []
        
        # Process each sequence
        for i, folder in enumerate(sequence_folders):
            try:
                print(f"\n\n{'='*80}")
                print(f"📂 PROCESSING SEQUENCE {i+1}/{len(sequence_folders)}: {folder}")
                print(f"{'='*80}")
                
                # Get sequence info
                sequence_info = get_sequence_info(dataset_path, folder)
                
                # Print image list
                print(f"\n📸 Images in sequence: {len(sequence_info.images)}")
                for j, img in enumerate(sequence_info.images[:5]):  # Show only first 5 for brevity
                    print(f"  {j+1}. {img.filename}")
                if len(sequence_info.images) > 5:
                    print(f"  ... and {len(sequence_info.images) - 5} more images")
                
                # Analyze sequence
                result = await analyze_sequence(sequence_info, model, test_prompt)
                all_results.append(result)
                
                # Display model response
                print("\n🤖 MODEL RESPONSE:")
                print("-"*80)
                print(result.raw_response)
                print("-"*80)
                
                # Display parsed results
                print("\n📊 ANALYSIS RESULTS:")
                print(f"  Ground Truth: {'✓ POSITIVE' if result.ground_truth else '✗ NEGATIVE'}")
                print(f"  Prediction: {'✓ DETECTED' if result.prediction else '✗ NOT DETECTED'}")
                
                if result.confidence is not None:
                    print(f"  Confidence: {result.confidence:.2f}")
                else:
                    print(f"  Confidence: Not specified")
                    
                print(f"  Assessment: {'CORRECT ✓' if result.is_correct else 'INCORRECT ✗'}")
                print(f"  Processing Time: {result.processing_time:.2f} seconds")
                
                # Display justification
                print("\n🔍 JUSTIFICATION:")
                print(f"  {result.justification}")
                
            except Exception as e:
                logger.error(f"Error processing sequence {folder}: {str(e)}")
                print(f"\n❌ ERROR processing sequence {folder}: {str(e)}")
                # Continue with next sequence
        
        # Calculate metrics
        if all_results:
            metrics = EvaluationMetrics(all_results)
            
            # Display summary metrics
            print("\n\n" + "="*80)
            print("📈 SUMMARY METRICS")
            print("="*80)
            print(f"Total Sequences: {metrics.total_sequences}")
            print(f"Accuracy: {metrics.accuracy:.4f}")
            print(f"Precision: {metrics.precision:.4f}")
            print(f"Recall: {metrics.recall:.4f}")
            print(f"F1 Score: {metrics.f1_score:.4f}")
            print(f"True Positives: {metrics.true_positives}")
            print(f"True Negatives: {metrics.true_negatives}")
            print(f"False Positives: {metrics.false_positives}")
            print(f"False Negatives: {metrics.false_negatives}")
            print(f"Total Processing Time: {metrics.total_processing_time:.2f} seconds")
            
            # Save results to file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = f"dataset_test_{timestamp}.json"
            
            with open(results_file, 'w') as f:
                json.dump({
                    "test_info": {
                        "dataset_path": dataset_path,
                        "model": model,
                        "model_details": model_info,
                        "prompt": test_prompt,
                        "timestamp": timestamp,
                        "total_sequences": len(sequence_folders)
                    },
                    "metrics": metrics.to_dict(),
                    "results": [
                        {
                            "sequence_id": r.sequence_id,
                            "ground_truth": r.ground_truth,
                            "prediction": r.prediction,
                            "confidence": r.confidence,
                            "justification": r.justification,
                            "processing_time": r.processing_time,
                            "is_correct": r.is_correct,
                            "raw_response": r.raw_response
                        }
                        for r in all_results
                    ]
                }, f, indent=2)
            
            print(f"\n💾 Complete results saved to: {results_file}")
            
            return metrics, all_results
        else:
            print("\n❌ No results generated. Check for errors above.")
            return None, []
            
    except Exception as e:
        logger.error(f"Error in dataset test: {str(e)}")
        print(f"\n❌ ERROR: {str(e)}")
        return None, []

# Command-line interface
def main():
    parser = argparse.ArgumentParser(description="Image Sequence Analysis Test Suite")
    parser.add_argument("--dataset", required=True, help="Path to dataset containing sequence folders")
    parser.add_argument("--model", default="gemma3:12b-it-q8_0", help="Ollama model to use for analysis")
    parser.add_argument("--prompt", help="Custom prompt template for the analysis")
    parser.add_argument("--max-sequences", type=int, help="Maximum number of sequences to process")
    
    args = parser.parse_args()
    
    # Run the dataset test
    import asyncio
    asyncio.run(run_dataset_test(
        args.dataset, 
        args.model, 
        args.prompt,
        args.max_sequences
    ))

if __name__ == "__main__":
    main()
