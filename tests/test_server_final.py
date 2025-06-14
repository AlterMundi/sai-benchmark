import os
import base64
import time
import re
import json
import logging
from datetime import datetime
from io import BytesIO
from typing import List, Dict, Any, Optional
import argparse
from pathlib import Path

from fastapi import FastAPI, Body, HTTPException, BackgroundTasks
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

class SequenceResult(BaseModel):
    sequence_id: str
    prediction: bool
    ground_truth: bool
    confidence: Optional[float]
    justification: str
    processing_time: float
    raw_response: str
    is_correct: bool

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
DEFAULT_PROMPT = """
# ROLE AND OBJECTIVE

You are a highly sensitive expert AI system specialized in the early and subtle detection of forest fires through intelligent visual analysis of image sequences.  
Your primary objective is to identify incipient or developing signs of smoke in images captured sequentially by fixed-position cameras in natural environments, ideally before a fire becomes obvious or while confirming its initial development.  
Accuracy and early detection are critical.

# INPUT

You will receive a sequence of images (`[image_1, image_2, ..., image_N]`) captured from the same fixed-position camera at short, regular time intervals.  
The images are temporally correlated and must be evaluated as a single unified set.  
Note that some images may be corrupted, out of focus, poorly framed, or contain irrelevant close-up obstructions.

# DETAILED ANALYSIS INSTRUCTIONS

Internally perform the following step-by-step analysis thoroughly BEFORE generating the final JSON response:

## 1. Individual Image Quality Evaluation

- For each image in the sequence: Determine whether it is usable for analyzing the natural landscape.
- Consider an image UNUSABLE if it meets ANY of the following conditions:
  - Entirely black, white, or visibly corrupted.
  - Substantially out of focus to the point where the sky, vegetation, or horizon cannot be distinguished.
  - Mainly contains irrelevant objects blocking the view of the landscape (walls, nearby poles, leaves covering >80% of the frame).
  - Does not show a recognizable natural landscape (e.g., aimed at interior or artificial elements).
- Action: Mentally discard all unusable images. Do not use them for smoke inference. Record whether at least one image was discarded.

## 2. Visual Context Reconstruction (with valid images)

- If valid images remain: Reconstruct the observed landscape and evaluate general stability and normal subtle changes between frames (light, atmosphere, non-suspicious cloud movement). Establish a dynamic baseline.

## 3. Zonal Segmentation and Dynamic Evaluation (with valid images)

- Mentally segment and analyze the sequence of valid images as follows:
  - Zone A (Upper/Mid Sky): Look for new, localized, dynamic formations that subtly change across frames (faint plumes, halos, emerging haze possibly originating from below). Differentiate from clouds.
  - Zone B (Horizon and Slopes): Look for the progressive appearance of faint grayish/whitish patches with potentially undulating or ascending patterns, or newly localized mist. Differentiate from natural fog.
  - Zone C (Vegetation and Ground): Analyze the anomalous and localized emergence of low-contrast, diffuse or irregular structures. Differentiate from dust or vapor.

## 4. Detection of Smoke Signatures (Incipient or Developed)

- Actively search for signals consistent with smoke:
  - Plume Formation: Faint or defined vertical/diagonal structures visible across at least 2–3 consecutive frames, with upward or drifting motion.
  - Texture/Opacity Changes: Localized areas showing progressive and anomalous changes in air texture or background visibility.
  - Density/Spread Increase: Structures that grow in opacity or size over the sequence.
  - Dynamic Behavior: Typical upward or undulating dispersion.

## 5. Rigorous Filtering of False Positives

- Review all detected signs. Explicitly discard phenomena that may appear similar but are not smoke:
  - Clouds (static, low fragments without clear lower origin, contrails).
  - Fog/mist (homogeneous, persistent without focal origin, valley fog).
  - Dust/dirt (no upward dynamics or progressive evolution).
  - Optical artifacts (flares, lens reflections).
  - Others (insects, non-anomalous ground vapor, etc.).

# INFERENCE CRITERIA AND JSON OUTPUT GENERATION

Based on the COMPLETE analysis of the full image sequence:

### 1. If ALL images were unusable:
Return the JSON with:
- "smoke_detected": "No"
- "justification": "Unusable sequence, no valid images for analysis"
- "confidence": 0.50 (or lower)
- "images_discarded": true

### 2. If valid images exist AND you detected credible visual patterns of smoke (incipient or developed) after filtering:

Return the JSON with:
- "smoke_detected": "Yes"
- A brief "justification" describing the main detected signal (e.g., "Faint upward plume in Zone B")
- A "confidence" value (0.00–1.00, with two decimal places)
- "images_discarded" (true/false depending on whether any were discarded)

### 3. If valid images exist AND you did NOT detect reasonable signs of smoke after filtering:

Return the JSON with:
- "smoke_detected": "No"
- A brief "justification" indicating the absence of anomalies (e.g., "No persistent anomalous signals detected")
- A "confidence" value (0.00–1.00, with two decimal places)
- "images_discarded" (true/false depending on whether any were discarded)

## Additional Rules

- The decision must be global over the valid sequence.
- The JSON must be exactly as specified.
- The justification must be concise and concrete.
- The confidence value must use two decimal places.

# STRICT OUTPUT FORMAT (JSON)

```json
{
  "smoke_detected": "Yes" | "No",
  "justification": "string",
  "confidence": float,
  "images_discarded": boolean
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
        output = response["message"]["content"]
        
        print(output)
        # Parse the response
        prediction, confidence, justification = parse_response(output)
        
        processing_time = time.time() - start_time
        
        # Get sequence ID from folder path
        sequence_id = os.path.basename(sequence_info.folder_path)
        
        return SequenceResult(
            sequence_id=sequence_id,
            prediction=prediction,
            ground_truth=sequence_info.ground_truth,
            confidence=confidence,
            justification=justification,
            processing_time=processing_time,
            raw_response=output,
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
        
        # Get sequence folders
        sequence_folders = get_sequence_folders(request.dataset_path, request.max_sequences)
        total_sequences = len(sequence_folders)
        
        logger.info(f"Found {total_sequences} sequence folders")
        
        # Update job status
        active_jobs[job_id]["total"] = total_sequences
        
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
                
                # Analyze sequence
                result = await analyze_sequence(sequence_info, request.model, prompt)
                
                results.append(result)
                
                # Update job results
                active_jobs[job_id]["results"] = results
                
                logger.info(f"Completed sequence {folder}: Prediction={result.prediction}, Ground Truth={result.ground_truth}, Correct={result.is_correct}")
                
            except Exception as e:
                logger.error(f"Error processing sequence {folder}: {str(e)}")
                # Continue with next sequence
        
        # Calculate metrics
        metrics = calculate_metrics(results)
        
        # Save results to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"results_{job_id}_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump({
                "metrics": metrics.dict(),
                "results": [r.dict() for r in results]
            }, f, indent=2)
        
        logger.info(f"Completed job {job_id}. Results saved to {results_file}")
        logger.info(f"Metrics: Accuracy={metrics.accuracy:.4f}, Precision={metrics.precision:.4f}, Recall={metrics.recall:.4f}, F1={metrics.f1_score:.4f}")
        
        # Update job status
        active_jobs[job_id]["status"] = "completed"
        active_jobs[job_id]["metrics"] = metrics
        
    except Exception as e:
        logger.error(f"Error processing dataset: {str(e)}")
        active_jobs[job_id]["status"] = "failed"
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
        "results": [],
        "start_time": time.time()
    }
    
    # Start background task
    background_tasks.add_task(process_dataset, job_id, request)
    
    return {"job_id": job_id, "status": "queued"}

@app.get("/job/{job_id}")
async def get_job_status(job_id: str):
    if job_id not in active_jobs:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    
    job = active_jobs[job_id]
    
    # Calculate progress
    progress = (job["current"] / job["total"]) * 100 if job["total"] > 0 else 0
    
    # Calculate elapsed time
    elapsed_time = time.time() - job["start_time"]
    
    response = {
        "job_id": job_id,
        "status": job["status"],
        "progress": progress,
        "current": job["current"],
        "total": job["total"],
        "elapsed_time": elapsed_time
    }
    
    # Include metrics if completed
    if job["status"] == "completed" and "metrics" in job:
        response["metrics"] = job["metrics"]
    
    # Include error if failed
    if job["status"] == "failed" and "error" in job:
        response["error"] = job["error"]
    
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
                "start_time": job["start_time"]
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
        "start_time": time.time()
    }
    
    asyncio.run(process_dataset(job_id, request))
    
    # Get results
    job = active_jobs[job_id]
    
    if job["status"] == "completed":
        # Save to specified output file or use default
        output_file = args.output or f"results_{job_id}.json"
        
        with open(output_file, 'w') as f:
            json.dump({
                "metrics": job["metrics"].dict(),
                "results": [r.dict() for r in job["results"]]
            }, f, indent=2)
        
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
    uvicorn.run("test_server2:app", host="0.0.0.0", port=8000)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Run as CLI
        run_cli()
    else:
        # Run as server
        run_server()
