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

# Setup main logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("image_test.log")]
)
logger = logging.getLogger("image_test")

# Setup performance logging - separate file for timing data
perf_logger = logging.getLogger("performance")
perf_logger.setLevel(logging.DEBUG)
perf_handler = logging.FileHandler("performance_timing.log")
perf_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
perf_logger.addHandler(perf_handler)
perf_logger.propagate = False  # Don't send to root logger

# Default prompt template
DEFAULT_PROMPT = """
You are looking at a sequence of images corresponding to an ongoing event. Please analyze them carefully as a skilled observer would looking for features corresponding to the target pattern. Give a single judgment in the given format taking all images into account as part of a single event.
Judgment: [Yes/No]
Confidence: [number between 0.0 and 1.0]
Justification: [brief text citing features identified]
"""

# Function to log performance data
def log_time(message, start_time=None):
    if start_time:
        elapsed = time.time() - start_time
        perf_logger.debug(f"{message} - {elapsed:.6f}s")
    else:
        perf_logger.debug(f"[START] {message}")
    return time.time()

# Function to convert image to base64 - optimized to minimize overhead
def image_to_base64(image_path):
    start_time = log_time(f"image_to_base64: {os.path.basename(image_path)}")
    try:
        try:
            time_open_start = time.time()
            with Image.open(image_path) as img:
                log_time("Opening image", time_open_start)
                
                time_buffer_start = time.time()
                buffered = BytesIO()
                log_time("Creating buffer", time_buffer_start)
                
                time_save_start = time.time()
                img.save(buffered, format="JPEG", quality=85, optimize=True)
                log_time("Saving to buffer", time_save_start)
                
                time_encode_start = time.time()
                result = base64.b64encode(buffered.getvalue()).decode("utf-8")
                log_time("Base64 encoding", time_encode_start)
                
                log_time(f"Total image_to_base64 for {os.path.basename(image_path)}", start_time)
                return result
        except Exception as e:
            logger.error(f"Error converting image: {str(e)}")
            raise
    finally:
        # Log completion regardless of success/failure
        log_time(f"Exiting image_to_base64 for {os.path.basename(image_path)}", start_time)

# Function to parse LLM response
def parse_response(output):
    start_time = log_time("parse_response")
    judgment_match = re.search(r'Judg?ment:\s*(Yes|No|yes|no)', output, re.IGNORECASE)
    confidence_match = re.search(r'Confidence:\s*([0-9](?:\.\d+)?)', output)
    justification_match = re.search(r'Justification:\s*(.*)', output, re.DOTALL)

    judgment = judgment_match.group(1).lower() if judgment_match else "unknown"
    judgment_bool = True if judgment.lower() == "yes" else False
    confidence = float(confidence_match.group(1)) if confidence_match and confidence_match.group(1) else None
    justification = justification_match.group(1).strip() if justification_match else "Not provided"
    
    log_time("Exiting parse_response", start_time)
    return judgment_bool, confidence, justification

# Function to read ground truth from txt file
def read_ground_truth(txt_path):
    start_time = log_time(f"read_ground_truth: {os.path.basename(txt_path)}")
    try:
        with open(txt_path, 'r') as f:
            content = f.read().strip()
            result = content != ""
            log_time(f"Exiting read_ground_truth: {os.path.basename(txt_path)}", start_time)
            return result
    except Exception as e:
        log_time(f"ERROR in read_ground_truth: {os.path.basename(txt_path)}", start_time)
        return None

# Get all sequence folders in dataset
def get_sequence_folders(dataset_path, max_sequences=None):
    start_time = log_time("get_sequence_folders")
    
    # List directory
    time_listdir_start = time.time()
    dir_contents = os.listdir(dataset_path)
    log_time("os.listdir", time_listdir_start)
    
    # Filter directories
    time_filter_start = time.time()
    folders = [f for f in dir_contents if os.path.isdir(os.path.join(dataset_path, f))]
    log_time("filtering directories", time_filter_start)
    
    # Sort folders
    time_sort_start = time.time()
    folders.sort()
    log_time("sorting folders", time_sort_start)
    
    # Apply max_sequences limit
    if max_sequences:
        folders = folders[:max_sequences]
        
    log_time(f"Exiting get_sequence_folders - Found {len(folders)} folders", start_time)
    return folders

# Get images for a sequence
def get_sequence_info(dataset_path, sequence_folder):
    start_time = log_time(f"get_sequence_info: {sequence_folder}")
    sequence_path = os.path.join(dataset_path, sequence_folder)
    
    # Get file list
    time_listdir_start = time.time()
    dir_contents = os.listdir(sequence_path)
    log_time("os.listdir for sequence folder", time_listdir_start)
    
    # Get image files
    time_filter_start = time.time()
    valid_extensions = ['.jpg', '.jpeg', '.png']
    image_files = []
    
    for filename in dir_contents:
        if any(filename.lower().endswith(ext) for ext in valid_extensions):
            image_files.append({
                "path": os.path.join(sequence_path, filename),
                "filename": filename
            })
    log_time("filtering image files", time_filter_start)
    
    # Sort images
    time_sort_start = time.time()
    image_files.sort(key=lambda x: x["filename"])
    log_time("sorting images", time_sort_start)
    
    # Check ground truth
    time_gt_start = time.time()
    ground_truth = False
    for img_info in image_files:
        txt_filename = os.path.splitext(img_info["filename"])[0] + ".txt"
        txt_path = os.path.join(sequence_path, txt_filename)
        
        if os.path.exists(txt_path):
            current_gt = read_ground_truth(txt_path)
            if current_gt:
                ground_truth = True
                break
    log_time("checking ground truth", time_gt_start)
    
    log_time(f"Exiting get_sequence_info: {sequence_folder} - Found {len(image_files)} images", start_time)
    return {
        "folder_path": sequence_path,
        "images": image_files,
        "ground_truth": ground_truth
    }

# Analyze a single sequence
async def analyze_sequence(sequence_info, model, prompt):
    sequence_id = os.path.basename(sequence_info["folder_path"])
    start_time = log_time(f"analyze_sequence: {sequence_id}")
    
    # Convert images to base64
    time_b64_start = time.time()
    images_base64 = []
    for i, img_info in enumerate(sequence_info["images"]):
        try:
            img_b64_start = time.time()
            img_base64 = image_to_base64(img_info["path"])
            log_time(f"base64 convert image {i+1}/{len(sequence_info['images'])}", img_b64_start)
            images_base64.append(img_base64)
        except Exception as e:
            logger.error(f"Error processing image {img_info['filename']}: {str(e)}")
            raise
    log_time(f"total base64 conversion for {len(sequence_info['images'])} images", time_b64_start)
    
    # Prepare message
    time_msg_start = time.time()
    message = {
        "role": "user", 
        "content": prompt,
        "images": images_base64
    }
    log_time("message preparation", time_msg_start)
    
    # Call Ollama API
    time_api_start = time.time()
    log_time(f"[CRUCIAL] Starting Ollama API call for {sequence_id}")
    perf_logger.debug(f"OLLAMA CALL START: {sequence_id} with {len(images_base64)} images at {datetime.now().strftime('%H:%M:%S.%f')}")
    response = ollama.chat(model=model, messages=[message])
    perf_logger.debug(f"OLLAMA CALL END: {sequence_id} at {datetime.now().strftime('%H:%M:%S.%f')}")
    log_time(f"[CRUCIAL] Ollama API call for {sequence_id}", time_api_start)
    
    output = response["message"]["content"]
    
    # Parse response
    time_parse_start = time.time()
    prediction, confidence, justification = parse_response(output)
    log_time("response parsing", time_parse_start)
    
    processing_time = time.time() - start_time
    
    # Create result dictionary
    result = {
        "sequence_id": sequence_id,
        "prediction": prediction,
        "ground_truth": sequence_info["ground_truth"],
        "confidence": confidence,
        "justification": justification,
        "processing_time": processing_time,
        "raw_response": output,
        "is_correct": (prediction == sequence_info["ground_truth"])
    }
    
    log_time(f"Exiting analyze_sequence: {sequence_id} - Total time: {processing_time:.2f}s", start_time)
    return result

# Calculate metrics
def calculate_metrics(results):
    start_time = log_time("calculate_metrics")
    
    total = len(results)
    if total == 0:
        log_time("Exiting calculate_metrics - No results", start_time)
        return {}
        
    true_positives = sum(1 for r in results if r["prediction"] and r["ground_truth"])
    true_negatives = sum(1 for r in results if not r["prediction"] and not r["ground_truth"])
    false_positives = sum(1 for r in results if r["prediction"] and not r["ground_truth"])
    false_negatives = sum(1 for r in results if not r["prediction"] and r["ground_truth"])
    
    accuracy = (true_positives + true_negatives) / total if total > 0 else 0
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    metrics = {
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
    
    log_time("Exiting calculate_metrics", start_time)
    return metrics

# Main function to run dataset test - optimized for performance
async def run_dataset_test(dataset_path, model="gemma3:12b-it-q8_0", prompt=None, max_sequences=None):
    global_start_time = log_time(f"[MAIN] Starting run_dataset_test with model {model}")
    perf_logger.debug(f"=== FULL RUN STARTED AT {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
    perf_logger.debug(f"Dataset: {dataset_path}")
    perf_logger.debug(f"Model: {model}")
    perf_logger.debug(f"Max sequences: {max_sequences}")
    
    print(f"Starting analysis on {dataset_path} with model {model}")
    perf_logger.debug("Performance log created at: performance_timing.log")
    
    # Use provided prompt or default
    test_prompt = prompt or DEFAULT_PROMPT
    
    # Get sequence folders
    time_folders_start = time.time()
    sequence_folders = get_sequence_folders(dataset_path, max_sequences)
    log_time("Getting sequence folders", time_folders_start)
    print(f"Found {len(sequence_folders)} sequence folders")
    
    results = []
    
    # Process each sequence
    for i, folder in enumerate(sequence_folders):
        sequence_start_time = log_time(f"[SEQUENCE {i+1}/{len(sequence_folders)}] Processing: {folder}")
        try:
            print(f"[{i+1}/{len(sequence_folders)}] Processing: {folder}")
            
            # Get sequence info
            time_info_start = time.time()
            sequence_info = get_sequence_info(dataset_path, folder)
            log_time(f"Getting sequence info for {folder}", time_info_start)
            
            # Analyze sequence
            time_analyze_start = time.time()
            result = await analyze_sequence(sequence_info, model, test_prompt)
            log_time(f"Analyzing sequence {folder}", time_analyze_start)
            
            results.append(result)
            
            # Print minimal result info
            status = "✓" if result["is_correct"] else "✗"
            print(f"  {status} Pred: {result['prediction']} | GT: {result['ground_truth']} | Time: {result['processing_time']:.2f}s")
            
        except Exception as e:
            print(f"  Error processing {folder}: {str(e)}")
            perf_logger.debug(f"ERROR in sequence {folder}: {str(e)}")
        finally:
            log_time(f"[SEQUENCE {i+1}/{len(sequence_folders)}] Completed: {folder}", sequence_start_time)
    
    # Calculate metrics
    if results:
        time_metrics_start = time.time()
        metrics = calculate_metrics(results)
        log_time("Calculating metrics", time_metrics_start)
        
        # Display minimal summary
        print("\nRESULTS SUMMARY:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1 Score: {metrics['f1_score']:.4f}")
        
        # Save results to file
        time_save_start = time.time()
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
        log_time("Saving results to file", time_save_start)
        
        print(f"Results saved to: {results_file}")
        
    else:
        print("No results generated")
        
    # Log performance summary
    total_time = time.time() - global_start_time
    perf_logger.debug(f"=== FULL RUN COMPLETED IN {total_time:.2f} SECONDS ===")
    perf_logger.debug(f"Average time per sequence: {total_time/len(sequence_folders):.2f}s") if sequence_folders else None
    
    if results:
        processing_times = [r["processing_time"] for r in results]
        perf_logger.debug(f"Min processing time: {min(processing_times):.2f}s")
        perf_logger.debug(f"Max processing time: {max(processing_times):.2f}s")
        perf_logger.debug(f"Avg processing time: {sum(processing_times)/len(processing_times):.2f}s")
        
    log_time("[MAIN] Exiting run_dataset_test", global_start_time)
    return metrics, results

# Command-line interface
def main():
    start_time = log_time("[MAIN] Starting main")
    parser = argparse.ArgumentParser(description="Fast Image Sequence Analysis with Timing")
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
    
    log_time("[MAIN] Exiting main", start_time)
    print("\nDetailed performance logs have been saved to 'performance_timing.log'")
    print("Use this file to identify bottlenecks in the script execution.")

if __name__ == "__main__":
    main()
