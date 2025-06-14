#!/usr/bin/env python3
import os
import time
import re
import json
import random
import argparse
from typing import List, Dict, Any, Optional
from datetime import datetime

# Setup basic timing functions
def now():
    return time.time()

# Default prompt template
DEFAULT_PROMPT = """
You are looking at a sequence of images corresponding to an ongoing event. Please analyze them carefully as a skilled observer would looking for features corresponding to the target pattern. Give a single judgment in the given format taking all images into account as part of a single event.
Judgment: [Yes/No]
Confidence: [number between 0.0 and 1.0]
Justification: [brief text citing features identified]
"""

# Function to parse response
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

# Simulated analysis of a single sequence - replaced with prints
async def analyze_sequence(sequence_info, model, prompt):
    start_time = now()
    
    # Print paths instead of processing
    print("\n--- SIMULATED MODEL INFERENCE ---")
    print(f"MODEL: {model}")
    print(f"SEQUENCE: {os.path.basename(sequence_info['folder_path'])}")
    print(f"PROMPT LENGTH: {len(prompt)} characters")
    print(f"IMAGES ({len(sequence_info['images'])}):")
    
    # Print the first 3 image paths (avoid cluttering console)
    for i, img in enumerate(sequence_info["images"][:3]):
        print(f"  {i+1}. {img['path']}")
    
    if len(sequence_info["images"]) > 3:
        print(f"  ... and {len(sequence_info['images'])-3} more images")
    
    # Simulate processing time
    processing_delay = random.uniform(0.1, 0.3)  # Very fast simulation
    time.sleep(processing_delay)
    
    # Generate a simulated response
    ground_truth = sequence_info["ground_truth"]
    
    # Add slight randomness to make results more realistic
    if random.random() < 0.8:  # 80% accuracy
        prediction = ground_truth  # Correct prediction
    else:
        prediction = not ground_truth  # Incorrect prediction
    
    confidence = random.uniform(0.75, 0.98)
    
    if prediction:
        justification = "Pattern detected in the image sequence showing characteristic features."
        raw_response = f"Judgment: Yes\nConfidence: {confidence:.2f}\nJustification: {justification}"
    else:
        justification = "No significant pattern detected in the analyzed sequence."
        raw_response = f"Judgment: No\nConfidence: {confidence:.2f}\nJustification: {justification}"
    
    processing_time = now() - start_time
    sequence_id = os.path.basename(sequence_info["folder_path"])
    
    # Print simulation summary
    print(f"SIMULATION COMPLETE: {processing_time:.2f}s")
    print(f"RESPONSE: {raw_response[:50]}...")
    print("--- END SIMULATION ---")
    
    # Create result dictionary
    return {
        "sequence_id": sequence_id,
        "prediction": prediction,
        "ground_truth": ground_truth,
        "confidence": confidence,
        "justification": justification,
        "processing_time": processing_time,
        "raw_response": raw_response,
        "is_correct": (prediction == ground_truth)
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

# Main function to run dataset test - just prints instead of actual inference
async def run_dataset_test(dataset_path, model="gemma3:12b-it-q8_0", prompt=None, max_sequences=None):
    start_time = now()
    
    print(f"\n{'='*80}")
    print(f"TEST MODE: NO ACTUAL INFERENCE")
    print(f"{'='*80}")
    print(f"Dataset: {dataset_path}")
    print(f"Model (simulated): {model}")
    
    # Use provided prompt or default
    test_prompt = prompt or DEFAULT_PROMPT
    
    # Get sequence folders
    sequence_folders = get_sequence_folders(dataset_path, max_sequences)
    print(f"Found {len(sequence_folders)} sequence folders")
    
    results = []
    
    # Process each sequence
    for i, folder in enumerate(sequence_folders):
        try:
            print(f"\n[{i+1}/{len(sequence_folders)}] Processing: {folder}")
            
            # Get sequence info
            sequence_info = get_sequence_info(dataset_path, folder)
            
            # Simulate sequence analysis
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
        
        # Display summary
        print("\nRESULTS SUMMARY:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1 Score: {metrics['f1_score']:.4f}")
        
        # Save results to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"test_results_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump({
                "test_info": {
                    "dataset_path": dataset_path,
                    "model": model,
                    "prompt": test_prompt,
                    "timestamp": timestamp,
                    "is_simulation": True
                },
                "metrics": metrics,
                "results": results
            }, f)
        
        print(f"Results saved to: {results_file}")
        print(f"Total execution time: {now() - start_time:.2f}s")
        
    else:
        print("No results generated")

# Command-line interface
def main():
    parser = argparse.ArgumentParser(description="Test Image Sequence Analysis (No Inference)")
    parser.add_argument("--dataset", required=True, help="Path to dataset containing sequence folders")
    parser.add_argument("--model", default="gemma3:12b-it-q8_0", help="Model name (for simulation only)")
    parser.add_argument("--prompt", help="Custom prompt template (not used in simulation)")
    parser.add_argument("--max-sequences", type=int, help="Maximum number of sequences to process")
    
    args = parser.parse_args()
    
    # Run simulated dataset test
    import asyncio
    asyncio.run(run_dataset_test(
        args.dataset, 
        args.model, 
        args.prompt,
        args.max_sequences
    ))

if __name__ == "__main__":
    main()
