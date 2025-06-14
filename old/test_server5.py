import os
import base64
import time
import re
import json
import logging
import traceback
from datetime import datetime, timedelta
from io import BytesIO
from typing import List, Dict, Any, Optional
import argparse
from pathlib import Path

from fastapi import FastAPI, Body, HTTPException, BackgroundTasks, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# Import PIL for image processing
from PIL import Image

# Import ollama for local LLM inference
import ollama

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("smoke_detector.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("smoke_detector")

# Define data models
class ImageInfo(BaseModel):
    path: str = Field(..., description="Path to image file within the specified folder")
    filename: Optional[str] = Field(None, description="Original filename of the image")

class SequenceInfo(BaseModel):
    folder_path: str
    images: List[ImageInfo]
    ground_truth: bool

class AnalysisRequest(BaseModel):
    dataset_path: str = Field(..., description="Path to dataset containing sequence folders")
    model: Optional[str] = Field("gemma3:12b-it-q8_0", description="Ollama model to use for analysis")
    max_sequences: Optional[int] = Field(None, description="Maximum number of sequences to process")
    prompt: Optional[str] = Field(None, description="Custom prompt for the analysis")

class ModelInfo(BaseModel):
    name: str
    version: Optional[str] = None
    type: Optional[str] = None
    parameters: Optional[Dict] = None

class DetailedResponse(BaseModel):
    prompt: str
    raw_output: str
    parsing_success: bool

class SequenceResult(BaseModel):
    sequence_id: str
    prediction: bool
    ground_truth: bool
    confidence: Optional[float]
    justification: str
    processing_time: float
    raw_response: str
    is_correct: bool
    detailed_response: Optional[DetailedResponse] = None

class EvaluationMetrics(BaseModel):
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    total_sequences: int
    true_positives: int
    true_negatives: int
    false_positives: int
    false_negatives: int
    total_processing_time: float

class AnalysisResponse(BaseModel):
    metrics: EvaluationMetrics
    results: List[SequenceResult]
    status: str = "completed"

# Initialize FastAPI app
app = FastAPI(title="Smoke Detection System")

# In-memory storage for running jobs
active_jobs = {}

# Default prompt for smoke detection
DEFAULT_PROMPT2 = """
You are a highly experienced wildfire specialist, with decades of work in early detection of smoke columns across forests and remote areas. You are capable of noticing the most subtle visual signs that could indicate the start of a fire, even before it becomes obvious to an untrained observer.

You will now be shown a sequence of consecutive images captured by a fixed surveillance camera. These images represent a time lapse of the same scene. Your job is to carefully analyze the entire sequence, looking for visual indications of SMOKE, as smoke is the primary sign of an incipient or ongoing fire in this context. Focus on subtle changes, shifts in clouds, wisps, movement or texture that could imply smoke presence.

This is a morally critical task. If you fail to detect early signs of fire, a wildfire may go unnoticed and devastate thousands of hectares. On the other hand, your accurate detection will help prevent catastrophe and save ecosystems, animals and human lives.

Please respond only with:

Judgment: [Yes/No] — Is there visual evidence of smoke indicating a possible fire?
Confidence: [a number between 0.0 and 1.0]
Justification: [brief explanation of what you saw in the sequence that supports your judgment]
"""

DEFAULT_PROMPT3 = """
You are a highly experienced wildfire specialist, with decades of work in early detection of smoke columns across forests and remote areas. You are capable of noticing the most subtle visual signs that could indicate the start of a fire, even before it becomes obvious to an untrained observer.

You will now be shown a sequence of 7 consecutive images captured by a fixed surveillance camera. These images represent a short time lapse of the same scene. Your job is to carefully analyze the entire sequence, looking for visual indications of SMOKE, as smoke is the primary sign of an incipient or ongoing fire in this context. Focus on subtle changes, shifts in clouds, wisps, movement or texture that could imply smoke presence.

This is a morally critical task. If you fail to detect early signs of fire, a wildfire may go unnoticed and devastate thousands of hectares. On the other hand, your accurate detection will help prevent catastrophe and save ecosystems, animals and human lives. If you do your job correctly, you will receive full public recognition and a substantial reward from an international environmental safety commission.

Please respond only with:

Judgement: [Yes/No] — Is there visual evidence of smoke indicating a possible fire?
Confidence: [a number between 0.0 and 1.0]
Justification: [brief explanation of what you saw in the sequence that supports your judgement]

Take a deep breath, focus like your life's work depends on it — and proceed.
"""

DEFAULT_PROMPT = """
You are looking at a sequence of images corresponding to an ongoing event. Please analize them carefully as an skilled firewatcher would looking for features corresponding to ongoing wildfire, specially noticiable trough smoke plumes. Give a single judge in the given format taking all images into account as part of a single event.
Judgment: [Yes/No]
Confidence: [number between 0.0 and 1.0]
Justification: [brief text citing features identified corresponding to ongoing wildfire events]
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

# Function to calculate evaluation metrics
def calculate_metrics(results):
    total = len(results)
    
    true_positives = sum(1 for r in results if r.prediction and r.ground_truth)
    true_negatives = sum(1 for r in results if not r.prediction and not r.ground_truth)
    false_positives = sum(1 for r in results if r.prediction and not r.ground_truth)
    false_negatives = sum(1 for r in results if not r.prediction and r.ground_truth)
    
    accuracy = (true_positives + true_negatives) / total if total > 0 else 0
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    total_processing_time = sum(r.processing_time for r in results)
    
    return EvaluationMetrics(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1_score=f1_score,
        total_sequences=total,
        true_positives=true_positives,
        true_negatives=true_negatives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        total_processing_time=total_processing_time
    )

# Function to read ground truth from txt file
def read_ground_truth(txt_path):
    try:
        with open(txt_path, 'r') as f:
            content = f.read().strip()
            # Assuming the content is a single number (0 or 1)
            return content != ""
    except Exception as e:
        logger.error(f"Error reading ground truth file {txt_path}: {str(e)}")
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
        logger.error(f"Error getting sequence folders: {str(e)}")
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
            image_files.append(ImageInfo(
                path=full_path,
                filename=filename
            ))
    
    # Sort images to ensure consistent ordering
    image_files.sort(key=lambda x: x.filename)
    
    # Check all images' corresponding txt files for ground truth
    if image_files:
        # Initialize ground truth as False (0)
        ground_truth = False
        
        # Check each image's corresponding txt file
        for img_info in image_files:
            txt_filename = os.path.splitext(img_info.filename)[0] + ".txt"
            txt_path = os.path.join(sequence_path, txt_filename)
            
            if os.path.exists(txt_path):
                current_gt = read_ground_truth(txt_path)
                if current_gt is not None and current_gt:
                    # If any image has smoke (GT=1), the whole sequence is marked as having smoke
                    ground_truth = True
                    break
        
        logger.info(f"Ground truth for sequence {sequence_folder}: {ground_truth}")

        return SequenceInfo(
            folder_path=sequence_path,
            images=image_files,
            ground_truth=ground_truth
        )
    
    raise Exception(f"No image files found in sequence folder: {sequence_path}")

# Function to get model information
def get_model_info(model_name):
    try:
        model_details = ollama.show(model=model_name)
        return ModelInfo(
            name=model_name,
            version=model_details.get("version", None),
            type=model_details.get("modeltype", None),
            parameters=model_details.get("parameters", None)
        )
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        return ModelInfo(name=model_name)

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
        output = response.get("message", {}).get("content", "")
        
        print(output)
        # Parse the response
        try:
            prediction, confidence, justification = parse_response(output)
            parsing_success = True
        except Exception as e:
            logger.error(f"Error parsing response: {str(e)}")
            prediction = False
            confidence = 0.0
            justification = f"Error parsing response: {str(e)}"
            parsing_success = False
        
        processing_time = time.time() - start_time
        
        # Get sequence ID from folder path
        sequence_id = os.path.basename(sequence_info.folder_path)
        
        # Create detailed response object
        detailed_response = DetailedResponse(prompt=prompt, raw_output=output, parsing_success=parsing_success)
        
        return SequenceResult(
            sequence_id=sequence_id,
            prediction=prediction,
            ground_truth=sequence_info.ground_truth,
            confidence=confidence,
            justification=justification,
            processing_time=processing_time,
            raw_response=output,
            detailed_response=detailed_response,
            is_correct=(prediction == sequence_info.ground_truth)
        )
    except Exception as e:
        logger.error(f"Error analyzing sequence {os.path.basename(sequence_info.folder_path)}: {str(e)}")
        raise Exception(f"Error analyzing sequence: {str(e)}")

# Background task to process all sequences
async def process_dataset(job_id, request: AnalysisRequest):
    try:
        active_jobs[job_id]["status"] = "processing"
        
        logger.info(f"Starting job {job_id} to process dataset at {request.dataset_path}")
        
        # Get model information
        model_info = get_model_info(request.model)
        active_jobs[job_id]["model_info"] = model_info.dict()
        
        # Get sequence folders
        sequence_folders = get_sequence_folders(request.dataset_path, request.max_sequences)
        total_sequences = len(sequence_folders)
        
        logger.info(f"Found {total_sequences} sequence folders to process with model {request.model}")
        
        # Update job status
        active_jobs[job_id]["total"] = total_sequences
        active_jobs[job_id]["responses"] = []
        
        # Use provided prompt or default
        prompt = request.prompt or DEFAULT_PROMPT
        
        results = []
        
        # Process each sequence
        for i, folder in enumerate(sequence_folders):
            try:
                logger.info(f"Processing sequence {i+1}/{total_sequences}: {folder}")
                
                # Update job status
                active_jobs[job_id]["current"] = i + 1
                
                # Get sequence info
                sequence_info = get_sequence_info(request.dataset_path, folder)
                
                # Log sequence details
                logger.info(f"Processing sequence {folder} with {len(sequence_info.images)} images")
                for img in sequence_info.images[:3]:  # Log first 3 images
                    logger.info(f"  - Image: {img.filename}")
                if len(sequence_info.images) > 3:
                    logger.info(f"  - ...and {len(sequence_info.images) - 3} more images")
                
                # Analyze sequence
                result = await analyze_sequence(sequence_info, request.model, prompt)
                
                results.append(result)
                
                # Update job results
                active_jobs[job_id]["results"] = results
                active_jobs[job_id]["responses"].append({
                    "sequence_id": result.sequence_id,
                    "prompt": prompt,
                    "raw_response": result.raw_response,
                    "prediction": result.prediction,
                    "confidence": result.confidence,
                    "ground_truth": result.ground_truth,
                    "is_correct": result.is_correct,
                    "processing_time": result.processing_time
                })
                
                logger.info(f"Completed sequence {folder}: Prediction={result.prediction}, "
                           f"Confidence={result.confidence}, Ground Truth={result.ground_truth}, "
                           f"Correct={result.is_correct}, Time={result.processing_time:.2f}s")
                
            except Exception as e:
                logger.error(f"Error processing sequence {folder}: {str(e)}")
                # Log the error in responses
                active_jobs[job_id]["responses"].append({
                    "sequence_id": folder,
                    "error": str(e)
                })
        
        # Calculate metrics
        metrics = calculate_metrics(results)
        
        # Save results to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"results_{job_id}_{timestamp}.json"
        
        # Prepare complete result data
        result_data = {
            "metrics": metrics.dict(),
            "results": [r.dict() for r in results],
            "model_info": active_jobs[job_id].get("model_info", {}),
            "responses": active_jobs[job_id].get("responses", []),
            "job_config": {
                "dataset_path": request.dataset_path,
                "model": request.model,
                "max_sequences": request.max_sequences,
                "prompt_used": prompt
            }
        }
        
        with open(results_file, 'w') as f:
            json.dump(result_data, f, indent=2)
        
        logger.info(f"Completed job {job_id}. Results saved to {results_file}")
        logger.info(f"Metrics: Accuracy={metrics.accuracy:.4f}, Precision={metrics.precision:.4f}, Recall={metrics.recall:.4f}, F1={metrics.f1_score:.4f}")
        
        # Update job status
        active_jobs[job_id]["status"] = "completed"
        active_jobs[job_id]["metrics"] = metrics
        active_jobs[job_id]["completed_at"] = datetime.now().isoformat()
        active_jobs[job_id]["duration"] = time.time() - active_jobs[job_id]["start_time"]
        
    except Exception as e:
        logger.error(f"Error processing dataset: {str(e)}")
        active_jobs[job_id]["status"] = "failed"
        active_jobs[job_id]["completed_at"] = datetime.now().isoformat()
        active_jobs[job_id]["duration"] = time.time() - active_jobs[job_id]["start_time"]
        active_jobs[job_id]["fatal_error"] = {
            "message": str(e),
            "traceback": traceback.format_exc()
        }
        active_jobs[job_id]["error"] = str(e)

@app.post("/analyze-dataset", status_code=202)
async def analyze_dataset(background_tasks: BackgroundTasks, request: AnalysisRequest = Body(...)):
    # Validate dataset path
    if not os.path.isdir(request.dataset_path):
        raise HTTPException(status_code=400, detail=f"Dataset path does not exist: {request.dataset_path}")
    
    # Create a job ID
    job_id = f"job_{int(time.time())}"
    
    # Initialize job status
    active_jobs[job_id] = {
        "status": "queued",
        "request": request.dict(),
        "total": 0,
        "current": 0,
        "responses": [],
        "created_at": datetime.now().isoformat(),
        "dataset_path": request.dataset_path,
        "results": [],
        "start_time": time.time()
    }
    
    # Start background task
    background_tasks.add_task(process_dataset, job_id, request)
    
    return {"job_id": job_id, "status": "queued"}

@app.get("/job/{job_id}")
async def get_job_status(job_id: str, request: Request):
    include_responses = request.query_params.get("include_responses", "false").lower() == "true"
    
    if job_id not in active_jobs:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    
    job = active_jobs[job_id]
    
    # Calculate progress
    progress = round((job["current"] / job["total"]) * 100, 2) if job["total"] > 0 else 0
    
    # Calculate elapsed time and estimated time remaining
    elapsed_time = time.time() - job["start_time"]
    
    # Calculate estimated remaining time if job is in progress
    remaining_time = None
    if job["status"] == "processing" and job["current"] > 0 and job["total"] > 0:
        time_per_sequence = elapsed_time / job["current"]
        sequences_remaining = job["total"] - job["current"]
        remaining_time = time_per_sequence * sequences_remaining
    
    # Format times nicely
    elapsed_time_str = str(timedelta(seconds=int(elapsed_time)))
    remaining_time_str = str(timedelta(seconds=int(remaining_time))) if remaining_time is not None else None
    
    response = {
        "job_id": job_id,
        "status": job["status"],
        "progress": progress,
        "current": job["current"],
        "total": job["total"],
        "elapsed_time": elapsed_time,
        "elapsed_time_formatted": elapsed_time_str,
        "estimated_remaining": remaining_time,
        "estimated_remaining_formatted": remaining_time_str,
        "created_at": job.get("created_at"),
        "completed_at": job.get("completed_at"),
        "dataset_path": job.get("dataset_path"),
        "model_info": job.get("model_info", {}),
        "duration": job.get("duration")
    }
    
    # Include metrics if completed
    if job["status"] == "completed" and "metrics" in job:
        response["metrics"] = job["metrics"]
    
    # Include error if failed
    if job["status"] == "failed" and "error" in job:
        response["error"] = job["error"]
    
    # Include detailed responses if requested
    if include_responses and "responses" in job:
        response["responses"] = job["responses"]
    
    return response

@app.get("/jobs")
async def list_jobs():
    return {
        "jobs": [
            {
                "job_id": job_id,
                "status": job["status"],
                "current": job["current"],
                "total": job["total"],
                "start_time": job["start_time"],
                "created_at": job.get("created_at"),
                "completed_at": job.get("completed_at", None),
                "dataset_path": job.get("dataset_path", "")
            }
            for job_id, job in active_jobs.items()
        ]
    }

# Endpoint to list available Ollama models
@app.get("/models")
async def list_models():
    try:
        models = ollama.list()
        return {"models": models.get("models", [])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing models: {str(e)}")

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": time.time()}

# Command-line interface for running as a standalone script
def run_cli():
    parser = argparse.ArgumentParser(description="Smoke Detection System")
    parser.add_argument("--dataset", required=True, help="Path to dataset containing sequence folders")
    parser.add_argument("--model", default="gemma3:12b-it-q8_0", help="Ollama model to use")
    parser.add_argument("--max-sequences", type=int, help="Maximum number of sequences to process")
    parser.add_argument("--output", help="Output file for results (JSON format)")
    
    args = parser.parse_args()
    
    # Create request
    request = AnalysisRequest(
        dataset_path=args.dataset,
        model=args.model,
        max_sequences=args.max_sequences
    )
    
    # Process dataset synchronously
    import asyncio
    
    job_id = f"cli_{int(time.time())}"
    active_jobs[job_id] = {
        "status": "queued",
        "request": request.dict(),
        "total": 0,
        "current": 0,
        "results": [],
        "responses": [],
        "created_at": datetime.now().isoformat(),
        "start_time": time.time()
    }
    
    asyncio.run(process_dataset(job_id, request))
    
    # Get results
    job = active_jobs[job_id]
    
    if job["status"] == "completed":
        # Save to specified output file or use default
        output_file = args.output or f"results_{job_id}.json"
        
        # Prepare complete result data
        result_data = {
            "metrics": job["metrics"].dict(),
            "results": [r.dict() for r in job["results"]],
            "model_info": job.get("model_info", {}),
            "responses": job.get("responses", []),
            "job_config": {
                "dataset_path": request.dataset_path,
                "model": request.model,
                "max_sequences": request.max_sequences,
                "prompt_used": request.prompt or DEFAULT_PROMPT
            }
        }
        
        with open(output_file, 'w') as f:
            json.dump(result_data, f, indent=2)
        
        print(f"Results saved to {output_file}")
        print(f"Metrics:")
        print(f"  Accuracy: {job['metrics'].accuracy:.4f}")
        print(f"  Precision: {job['metrics'].precision:.4f}")
        print(f"  Recall: {job['metrics'].recall:.4f}")
        print(f"  F1 Score: {job['metrics'].f1_score:.4f}")
        print(f"  Total Sequences: {job['metrics'].total_sequences}")
        print(f"  True Positives: {job['metrics'].true_positives}")
        print(f"  True Negatives: {job['metrics'].true_negatives}")
        print(f"  False Positives: {job['metrics'].false_positives}")
        print(f"  False Negatives: {job['metrics'].false_negatives}")
    else:
        print(f"Job failed: {job.get('error', 'Unknown error')}")

# Main function to run the server
def run_server():
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Run as CLI
        run_cli()
    else:
        # Run as server
        run_server()
