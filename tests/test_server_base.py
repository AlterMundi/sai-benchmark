#!/usr/bin/env python3
import os
import base64
import time
import re
import json
import argparse
from io import BytesIO
from typing import List, Dict, Any, Optional
from pathlib import Path

# Import PIL for image processing
from PIL import Image

# Import ollama for local LLM inference
import ollama

# Define default prompt for smoke detection
DEFAULT_PROMPT = """
# ROLE AND OBJECTIVE

You are a highly sensitive expert AI system specialized in the early and subtle detection of forest fires through intelligent visual analysis of image sequences.  
Your primary objective is to identify incipient or developing signs of smoke in images captured sequentially by fixed-position cameras in natural environments, ideally before a fire becomes obvious or while confirming its initial development.  
Accuracy and early detection are critical.

# STRICT OUTPUT FORMAT (JSON)

```json
{
  "smoke_detected": "Yes" | "No",
  "justification": "string",
  "confidence": float,
  "images_discarded": boolean,
  "number_of_images": integer
}
"""

# Function to convert image to base64
def image_to_base64(image_path):
    try:
        with Image.open(image_path) as img:
            buffered = BytesIO()
            img.save(buffered, format="JPEG")
            return base64.b64encode(buffered.getvalue()).decode("utf-8")
    except Exception as e:
        print(f"Error converting image to base64: {str(e)}")
        raise Exception(f"Error converting image to base64: {str(e)}")

# Function to parse Ollama response
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
            # Assuming the content is a single number (0 or 1) or not empty for positive
            return content != ""
    except Exception as e:
        print(f"Error reading ground truth file {txt_path}: {str(e)}")
        return None

# Function to get all sequence folders in dataset
def get_sequence_folders(dataset_path, max_sequences=None):
    try:
        folders = [f for f in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, f))]
        folders.sort()  # Sort for consistent ordering
        
        if max_sequences:
            folders = folders[:max_sequences]
        
        return folders
    except Exception as e:
        print(f"Error getting sequence folders: {str(e)}")
        raise Exception(f"Error getting sequence folders: {str(e)}")

# Function to get images and ground truth for a sequence
def get_sequence_info(dataset_path, sequence_folder):
    sequence_path = os.path.join(dataset_path, sequence_folder)
    
    # Get all jpg files
    valid_extensions = ['.jpg', '.jpeg', '.png']
    image_files = []
    
    for filename in os.listdir(sequence_path):
        if any(filename.lower().endswith(ext) for ext in valid_extensions):
            full_path = os.path.join(sequence_path, filename)
            image_files.append({
                "path": full_path,
                "filename": filename
            })
    
    # Sort images to ensure consistent ordering
    image_files.sort(key=lambda x: x["filename"])
    
    # Check all images' corresponding txt files for ground truth
    if image_files:
        # Initialize ground truth as False (0)
        ground_truth = False
        
        # Check each image's corresponding txt file
        for img_info in image_files:
            txt_filename = os.path.splitext(img_info["filename"])[0] + ".txt"
            txt_path = os.path.join(sequence_path, txt_filename)
            
            if os.path.exists(txt_path):
                current_gt = read_ground_truth(txt_path)
                if current_gt is not None and current_gt:
                    # If any image has smoke (GT=1), the whole sequence is marked as having smoke
                    ground_truth = True
                    break
        
        print(f"Ground truth for sequence {sequence_folder}: {ground_truth}")

        return {
            "folder_path": sequence_path,
            "images": image_files,
            "ground_truth": ground_truth
        }
    
    raise Exception(f"No image files found in sequence folder: {sequence_path}")

# Function to analyze a single sequence
def analyze_sequence(sequence_info, model, prompt):
    start_time = time.time()
    
    # Convert all images to base64
    images_base64 = []
    for img_info in sequence_info["images"]:
        try:
            img_base64 = image_to_base64(img_info["path"])
            images_base64.append({
                "filename": img_info["filename"],
                "base64": img_base64
            })
        except Exception as e:
            print(f"Error processing image {img_info['filename']}: {str(e)}")
            raise Exception(f"Error processing image {img_info['filename']}: {str(e)}")
    
    # Create a composite prompt listing all image filenames
    image_list = "\n".join([f"{i+1}. {img['filename']}" for i, img in enumerate(images_base64)])
    composite_prompt = f"{prompt}\n\nPlease analyze the following {len(images_base64)} images as a sequence:\n{image_list}"
    
    # Prepare message with all images
    message = {
        "role": "user", 
        "content": prompt,
        "images": [img["base64"] for img in images_base64]
    }
    
            # Call Ollama API with all images at once
    try:
        response = ollama.chat(model=model, messages=[message])
        output = response["message"]["content"]
        
        print("\n--- MODEL RESPONSE ---")
        print(output)
        print("---------------------\n")
        
        # Parse the response
        prediction, confidence, justification = parse_response(output)
        
        processing_time = time.time() - start_time
        
        # Get sequence ID from folder path
        sequence_id = os.path.basename(sequence_info["folder_path"])
        
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
    except Exception as e:
        print(f"Error analyzing sequence {os.path.basename(sequence_info['folder_path'])}: {str(e)}")
        raise Exception(f"Error analyzing sequence: {str(e)}")

# Function to calculate evaluation metrics
def calculate_metrics(results):
    total = len(results)
    
    true_positives = sum(1 for r in results if r["prediction"] and r["ground_truth"])
    true_negatives = sum(1 for r in results if not r["prediction"] and not r["ground_truth"])
    false_positives = sum(1 for r in results if r["prediction"] and not r["ground_truth"])
    false_negatives = sum(1 for r in results if not r["prediction"] and r["ground_truth"])
    
    accuracy = (true_positives + true_negatives) / total if total > 0 else 0
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    total_processing_time = sum(r["processing_time"] for r in results)
    
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
        "total_processing_time": total_processing_time
    }

def main():
    parser = argparse.ArgumentParser(description="Simple Smoke Detection CLI")
    parser.add_argument("--dataset", required=True, help="Path to dataset containing sequence folders")
    parser.add_argument("--model", default="gemma3:12b-it-q8_0", help="Ollama model to use for analysis")
    parser.add_argument("--sequence", help="Specific sequence folder to analyze (if not specified, will process all)")
    parser.add_argument("--max-sequences", type=int, default=1, help="Maximum number of sequences to process")
    parser.add_argument("--output", help="Output file for results (JSON format)")
    parser.add_argument("--prompt", help="Custom prompt file (if not specified, will use default)")
    
    args = parser.parse_args()
    
    # Validate dataset path
    if not os.path.isdir(args.dataset):
        print(f"Error: Dataset path does not exist: {args.dataset}")
        return
    
    # Get custom prompt if specified
    prompt = DEFAULT_PROMPT
    if args.prompt and os.path.exists(args.prompt):
        try:
            with open(args.prompt, 'r') as f:
                prompt = f.read()
                print(f"Using custom prompt from {args.prompt}")
        except Exception as e:
            print(f"Error reading prompt file, using default prompt: {str(e)}")
    
    # Get sequence folders
    if args.sequence:
        sequence_folders = [args.sequence]
    else:
        sequence_folders = get_sequence_folders(args.dataset, args.max_sequences)
    
    total_sequences = len(sequence_folders)
    print(f"Found {total_sequences} sequence folder(s) to analyze")
    
    results = []
    
    # Process each sequence
    for i, folder in enumerate(sequence_folders):
        try:
            print(f"\nProcessing sequence {i+1}/{total_sequences}: {folder}")
            
            # Get sequence info
            sequence_info = get_sequence_info(args.dataset, folder)
            
            # Analyze sequence
            result = analyze_sequence(sequence_info, args.model, prompt)
            
            results.append(result)
            
            print(f"Completed sequence {folder}:")
            print(f"  Judgment: {'Smoke' if result['prediction'] else 'No Smoke'}")
            print(f"  Ground Truth: {'Smoke' if result['ground_truth'] else 'No Smoke'}")
            print(f"  Correct: {'Yes' if result['is_correct'] else 'No'}")
            print(f"  Confidence: {result['confidence']}")
            print(f"  Processing Time: {result['processing_time']:.2f} seconds")
            
        except Exception as e:
            print(f"Error processing sequence {folder}: {str(e)}")
            # Continue with next sequence
    
    # Calculate metrics
    if results:
        metrics = calculate_metrics(results)
        
        print("\n--- METRICS SUMMARY ---")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1 Score: {metrics['f1_score']:.4f}")
        print(f"Total Sequences: {metrics['total_sequences']}")
        print(f"True Positives: {metrics['true_positives']}")
        print(f"True Negatives: {metrics['true_negatives']}")
        print(f"False Positives: {metrics['false_positives']}")
        print(f"False Negatives: {metrics['false_negatives']}")
        print(f"Total Processing Time: {metrics['total_processing_time']:.2f} seconds")
        
        # Save results to file if specified
        if args.output:
            with open(args.output, 'w') as f:
                json.dump({
                    "metrics": metrics,
                    "results": results
                }, f, indent=2)
            print(f"\nResults saved to {args.output}")

if __name__ == "__main__":
    main()