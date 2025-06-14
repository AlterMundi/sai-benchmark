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

# Import PIL for image processing
from PIL import Image

# Import ollama for local LLM inference
import ollama

# Setup logging - minimal configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("image_test.log")]
)
logger = logging.getLogger("image_test")

# Default prompt template
DEFAULT_PROMPT = """
You are looking at a sequence of images corresponding to an ongoing event. Please analyze them carefully as a skilled observer would looking for features corresponding to the target pattern. Give a single judgment in the given format taking all images into account as part of a single event.
Judgment: [Yes/No]
Confidence: [number between 0.0 and 1.0]
Justification: [brief text citing features identified]
"""

# Function to convert image to base64 - optimized to minimize overhead
def image_to_base64(image_path):
    try:
        with Image.open(image_path) as img:
            buffered = BytesIO()
            img.save(buffered, format="JPEG", quality=85, optimize=True)
            return base64.b64encode(buffered.getvalue()).decode("utf-8")
    except Exception as e:
        logger.error(f"Error converting image: {str(e)}")
        raise

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
            return content != ""
    except Exception as e:
        return None

# Get all sequence folders in dataset
def get_sequence_folders(dataset_path, max_sequences=None):
    folders = [f for f in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, f))]
    folders.sort()
    return folders[:max_sequences] if max_sequences else folders

# Get images for a sequence
def get_sequence_info(dataset_path, sequence_folder):
    sequence_path = os.path.join(dataset_path, sequence_folder)
    print(sequence_path)
    # Get image files
    valid_extensions = ['.jpg', '.jpeg', '.png']
    image_files = []
    
    for filename in os.listdir(sequence_path):
        if any(filename.lower().endswith(ext) for ext in valid_extensions):
            image_files.append({
                "path": os.path.join(sequence_path, filename),
                "filename": filename
            })
    
    # Sort images by filename
    image_files.sort(key=lambda x: x["filename"])
    
    # Check ground truth
    ground_truth = False
    for img_info in image_files:
        txt_filename = os.path.splitext(img_info["filename"])[0] + ".txt"
        txt_path = os.path.join(sequence_path, txt_filename)
        
        if os.path.exists(txt_path):
            current_gt = read_ground_truth(txt_path)
            if current_gt:
                ground_truth = True
                break
    
    return {
        "folder_path": sequence_path,
        "images": image_files,
        "ground_truth": ground_truth
    }

# Analyze a single sequence
async def analyze_sequence(sequence_info, model, prompt):
    start_time = time.time()
    
    # Convert images to base64
    images_base64 = []
    for img_info in sequence_info["images"]:
        try:
            img_base64 = image_to_base64(img_info["path"])
            images_base64.append(img_base64)
        except Exception as e:
            logger.error(f"Error processing image {img_info['filename']}: {str(e)}")
            raise
    
    # Prepare message with all images
    message = {
        "role": "user", 
        "content": prompt,
        "images": images_base64
    }
    
    # Call Ollama API
    response = ollama.chat(model=model, messages=[message])
    output = response["message"]["content"]
    
    # Parse response
    prediction, confidence, justification = parse_response(output)
    
    processing_time = time.time() - start_time
    sequence_id = os.path.basename(sequence_info["folder_path"])
    print(processing_time)
    # Create result dictionary
    return {
        "sequence_id": sequence_id,
        "prediction": prediction,
        "ground_truth": sequence_info["ground_truth"],
        "confidence": confidence,
        "justification": justification,
        "processing_time": processing_time,
        "raw_response": output,
        "is_correct": (prediction == sequence_info["ground_truth"])
    }

# Calculate metrics
def calculate_metrics(results):
    total = len(results)
    if total == 0:
        return {}
        
    true_positives = sum(1 for r in results if r["prediction"] and r["ground_truth"])
    true_negatives = sum(1 for r in results if not r["prediction"] and not r["ground_truth"])
    false_positives = sum(1 for r in results if r["prediction"] and not r["ground_truth"])
    false_negatives = sum(1 for r in results if not r["prediction"] and r["ground_truth"])
    
    accuracy = (true_positives + true_negatives) / total if total > 0 else 0
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "total_sequences": total,
        "true_positives": true_positives,
        "true_negatives": true_negatives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "total_processing_time": sum(r["processing_time"] for r in results)
    }

# Main function to run dataset test - optimized for performance
async def run_dataset_test(dataset_path, model="gemma3:12b-it-q8_0", prompt=None, max_sequences=None):
    print(f"Starting analysis on {dataset_path} with model {model}")
    
    # Use provided prompt or default
    test_prompt = prompt or DEFAULT_PROMPT
    
    # Get sequence folders
    sequence_folders = get_sequence_folders(dataset_path, max_sequences)
    print(f"Found {len(sequence_folders)} sequence folders")
    
    results = []
    
    # Process each sequence
    for i, folder in enumerate(sequence_folders):
        try:
            print(f"[{i+1}/{len(sequence_folders)}] Processing: {folder}")
            
            # Get sequence info
            sequence_info = get_sequence_info(dataset_path, folder)
            
            print(sequence_info)
            # Analyze sequence
            result = await analyze_sequence(sequence_info, model, test_prompt)
            results.append(result)
            
            # Print minimal result info
            status = "✓" if result["is_correct"] else "✗"
            print(f"  {status} Pred: {result['prediction']} | GT: {result['ground_truth']} | Time: {result['processing_time']:.2f}s")
            
        except Exception as e:
            print(f"  Error processing {folder}: {str(e)}")
    
    # Calculate metrics
    if results:
        metrics = calculate_metrics(results)
        
        # Display minimal summary
        print("\nRESULTS SUMMARY:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1 Score: {metrics['f1_score']:.4f}")
        
        # Save results to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"results_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump({
                "test_info": {
                    "dataset_path": dataset_path,
                    "model": model,
                    "prompt": test_prompt,
                    "timestamp": timestamp
                },
                "metrics": metrics,
                "results": results
            }, f)
        
        print(f"Results saved to: {results_file}")
        
    else:
        print("No results generated")

# Command-line interface
def main():
    parser = argparse.ArgumentParser(description="Fast Image Sequence Analysis")
    parser.add_argument("--dataset", required=True, help="Path to dataset containing sequence folders")
    parser.add_argument("--model", default="gemma3:12b-it-q8_0", help="Ollama model to use")
    parser.add_argument("--prompt", help="Custom prompt template")
    parser.add_argument("--max-sequences", type=int, help="Maximum number of sequences to process")
    
    args = parser.parse_args()
    
    # Run dataset test
    import asyncio
    asyncio.run(run_dataset_test(
        args.dataset, 
        args.model, 
        args.prompt,
        args.max_sequences
    ))

if __name__ == "__main__":
    main()
