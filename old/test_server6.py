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
import statistics

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
    context: Optional[Dict[str, Any]] = Field(None, description="Contextual information (reserved for future use)")

class AnalysisRequest(BaseModel):
    dataset_path: str = Field(..., description="Path to dataset containing sequence folders")
    model: Optional[str] = Field("gemma3:12b-it-q8_0", description="Ollama model to use for analysis")
    max_sequences: Optional[int] = Field(None, description="Maximum number of sequences to process")
    prompt: Optional[str] = Field(None, description="Custom prompt for the analysis")
    include_raw_responses: Optional[bool] = Field(True, description="Include raw LLM responses in results")
    experiment_name: Optional[str] = Field(None, description="Name for this experiment run")

class LLMResponse(BaseModel):
    model: str
    raw_text: str
    tokens_used: Optional[int] = None  # Made explicitly optional with default=None
    response_time: float
    parsed_data: Dict[str, Any]

class SequenceResult(BaseModel):
    sequence_id: str
    prediction: bool
    ground_truth: bool
    confidence: Optional[float]
    justification: str
    processing_time: float
    is_correct: bool
    num_images: int
    llm_analysis: Optional[LLMResponse] = None
    error: Optional[str] = None

class DetailedMetrics(BaseModel):
    confidence_stats: Dict[str, float] = Field(
        default_factory=dict,
        description="Statistics about confidence scores (mean, median, std_dev, etc.)"
    )
    average_processing_time: float
    error_rate: float
    prompt_tokens: Optional[int] = None  # Made explicitly optional with default=None
    completion_tokens: Optional[int] = None  # Made explicitly optional with default=None
    category_performance: Dict[str, Dict[str, float]] = Field(
        default_factory=dict,
        description="Performance metrics broken down by categories"
    )

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
    detailed: DetailedMetrics

class AnalysisResponse(BaseModel):
    metrics: EvaluationMetrics
    results: List[SequenceResult]
    status: str = "completed"
    experiment_info: Dict[str, Any] = Field(default_factory=dict)
    prompt_used: str
    model_used: str
    timestamp: str
    context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Reserved for future contextual information"
    )

# Initialize FastAPI app
app = FastAPI(title="Smoke Detection System")

# In-memory storage for running jobs
active_jobs = {}

# Default prompt for smoke detection
DEFAULT_PROMPT = """
You are looking at a sequence of images corresponding to an ongoing event. Please analyze them carefully as a skilled firewatcher would looking for features corresponding to ongoing wildfire, specially noticeable through smoke plumes. Give a single judge in the given format taking all images into account as part of a single event.
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
    
    return {
        "judgment": judgment_bool,
        "confidence": confidence,
        "justification": justification,
        "parsed_successfully": all([judgment_match, confidence_match, justification_match])
    }

# Function to calculate evaluation metrics
def calculate_metrics(results):
    total = len(results)
    
    if total == 0:
        return None
    
    true_positives = sum(1 for r in results if r.prediction and r.ground_truth)
    true_negatives = sum(1 for r in results if not r.prediction and not r.ground_truth)
    false_positives = sum(1 for r in results if r.prediction and not r.ground_truth)
    false_negatives = sum(1 for r in results if not r.prediction and r.ground_truth)
    
    accuracy = (true_positives + true_negatives) / total if total > 0 else 0
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    total_processing_time = sum(r.processing_time for r in results)
    
    # Calculate detailed metrics
    confidences = [r.confidence for r in results if r.confidence is not None]
    processing_times = [r.processing_time for r in results]
    error_count = sum(1 for r in results if r.error is not None)
    
    confidence_stats = {}
    if confidences:
        confidence_stats = {
            "mean": statistics.mean(confidences),
            "median": statistics.median(confidences),
            "min": min(confidences),
            "max": max(confidences)
        }
        if len(confidences) > 1:
            confidence_stats["std_dev"] = statistics.stdev(confidences)
    
    # Calculate performance by prediction category
    category_performance = {
        "smoke_present": {
            "count": sum(1 for r in results if r.ground_truth),
            "accuracy": true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0,
        },
        "smoke_absent": {
            "count": sum(1 for r in results if not r.ground_truth),
            "accuracy": true_negatives / (true_negatives + false_positives) if (true_negatives + false_positives) > 0 else 0,
        }
    }
    
    detailed = DetailedMetrics(
        confidence_stats=confidence_stats,
        average_processing_time=sum(processing_times) / len(processing_times) if processing_times else 0,
        error_rate=error_count / total if total > 0 else 0,
        category_performance=category_performance
    )
    
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
        total_processing_time=total_processing_time,
        detailed=detailed
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
async def analyze_sequence(sequence_info, model, prompt, include_raw_responses=True):
    start_time = time.time()
    
    try:
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
        response_start = time.time()
        response = ollama.chat(model=model, messages=[message])
        response_time = time.time() - response_start
        output = response["message"]["content"]
        
        # Parse the response
        parsed_data = parse_response(output)
        
        processing_time = time.time() - start_time
        
        # Get sequence ID from folder path
        sequence_id = os.path.basename(sequence_info.folder_path)
        
        # Create LLM response object if requested
        llm_analysis = None
        if include_raw_responses:
            llm_analysis = LLMResponse(
                model=model,
                raw_text=output,
                response_time=response_time,
                parsed_data=parsed_data
            )
        
        return SequenceResult(
            sequence_id=sequence_id,
            prediction=parsed_data["judgment"],
            ground_truth=sequence_info.ground_truth,
            confidence=parsed_data["confidence"],
            justification=parsed_data["justification"],
            processing_time=processing_time,
            is_correct=(parsed_data["judgment"] == sequence_info.ground_truth),
            num_images=len(sequence_info.images),
            llm_analysis=llm_analysis,
            error=None
        )
    except Exception as e:
        error_msg = f"Error analyzing sequence: {str(e)}"
        logger.error(f"Error analyzing sequence {os.path.basename(sequence_info.folder_path)}: {str(e)}")
        
        # Get sequence ID from folder path
        sequence_id = os.path.basename(sequence_info.folder_path)
        
        # Return a result with error information
        return SequenceResult(
            sequence_id=sequence_id,
            prediction=False,  # Default value
            ground_truth=sequence_info.ground_truth,
            confidence=None,
            justification="",
            processing_time=time.time() - start_time,
            is_correct=False,  # Default value
            num_images=len(sequence_info.images),
            llm_analysis=None,
            error=error_msg
        )

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
                result = await analyze_sequence(
                    sequence_info, 
                    request.model, 
                    prompt,
                    request.include_raw_responses
                )
                
                results.append(result)
                
                # Update job results
                active_jobs[job_id]["results"] = results
                
                logger.info(f"Completed sequence {folder}: Prediction={result.prediction}, Ground Truth={result.ground_truth}, Correct={result.is_correct}")
                
            except Exception as e:
                logger.error(f"Error processing sequence {folder}: {str(e)}")
                # Continue with next sequence
        
        # Calculate metrics
        metrics = calculate_metrics(results)
        
        # Create experiment info
        experiment_info = {
            "dataset_path": request.dataset_path,
            "experiment_name": request.experiment_name or f"Experiment_{job_id}",
            "sequences_processed": len(results),
            "successful_sequences": sum(1 for r in results if r.error is None),
            "failed_sequences": sum(1 for r in results if r.error is not None)
        }
        
        # Create response
        analysis_response = AnalysisResponse(
            metrics=metrics,
            results=results,
            prompt_used=prompt,
            model_used=request.model,
            timestamp=datetime.now().isoformat(),
            experiment_info=experiment_info
        )
        
        # Save results to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"results_{job_id}_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(analysis_response.model_dump(), f, indent=2)  # Updated from .dict() to .model_dump()
        
        logger.info(f"Completed job {job_id}. Results saved to {results_file}")
        logger.info(f"Metrics: Accuracy={metrics.accuracy:.4f}, Precision={metrics.precision:.4f}, Recall={metrics.recall:.4f}, F1={metrics.f1_score:.4f}")
        
        # Update job status
        active_jobs[job_id]["status"] = "completed"
        active_jobs[job_id]["metrics"] = metrics
        active_jobs[job_id]["response"] = analysis_response
        
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
        "request": request.model_dump(),  # Updated from .dict() to .model_dump()
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

@app.get("/job/{job_id}/results")
async def get_job_results(job_id: str, format: str = "json"):
    if job_id not in active_jobs:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    
    job = active_jobs[job_id]
    
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"Job not completed. Current status: {job['status']}")
    
    if "response" not in job:
        raise HTTPException(status_code=500, detail="Job completed but results not available")
    
    if format.lower() == "markdown":
        # Create markdown summary
        markdown = f"""
# Smoke Detection Analysis Results

## Experiment Information
- **Experiment Name**: {job["response"].experiment_info.get("experiment_name", "Unnamed")}
- **Dataset**: {job["response"].experiment_info.get("dataset_path", "Unknown")}
- **Model**: {job["response"].model_used}
- **Date**: {job["response"].timestamp}

## Results Summary
- **Accuracy**: {job["response"].metrics.accuracy:.4f}
- **Precision**: {job["response"].metrics.precision:.4f}
- **Recall**: {job["response"].metrics.recall:.4f}
- **F1 Score**: {job["response"].metrics.f1_score:.4f}
- **Total Sequences**: {job["response"].metrics.total_sequences}
- **Processing Time**: {job["response"].metrics.total_processing_time:.2f} seconds

### Detailed Metrics
- **Confidence (Mean)**: {job["response"].metrics.detailed.confidence_stats.get("mean", 0):.4f}
- **Avg Processing Time**: {job["response"].metrics.detailed.average_processing_time:.2f} seconds
- **Error Rate**: {job["response"].metrics.detailed.error_rate:.4f}

## Confusion Matrix
|              | Predicted Yes | Predicted No |
|--------------|---------------|--------------|
| **Actual Yes** | {job["response"].metrics.true_positives} | {job["response"].metrics.false_negatives} |
| **Actual No**  | {job["response"].metrics.false_positives} | {job["response"].metrics.true_negatives} |

## Prompt Used
```
{job["response"].prompt_used}
```

## Sample Results
Below are sample results from the sequences analyzed:
"""
        
        # Add a few sample results
        for i, result in enumerate(job["response"].results[:5]):
            markdown += f"""
### Sequence {result.sequence_id}
- **Prediction**: {"Smoke Detected" if result.prediction else "No Smoke"}
- **Ground Truth**: {"Smoke Present" if result.ground_truth else "No Smoke"}
- **Correct**: {"✓" if result.is_correct else "✗"}
- **Confidence**: {result.confidence if result.confidence is not None else "N/A"}
- **Images Analyzed**: {result.num_images}

**Justification**: {result.justification}

"""
        
        return JSONResponse(content={"markdown": markdown})
    else:
        # Return JSON results
        return job["response"]

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
                "elapsed_time": time.time() - job["start_time"],
                "model": job["request"].get("model", "N/A"),
                "experiment_name": job["request"].get("experiment_name", f"Experiment_{job_id}")
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
    parser.add_argument("--prompt", help="Custom prompt for analysis")
    parser.add_argument("--experiment-name", help="Name for this experiment run")
    parser.add_argument("--no-raw-responses", action="store_true", help="Exclude raw LLM responses from results")
    
    args = parser.parse_args()
    
    # Create request
    request = AnalysisRequest(
        dataset_path=args.dataset,
        model=args.model,
        max_sequences=args.max_sequences,
        prompt=args.prompt,
        include_raw_responses=not args.no_raw_responses,
        experiment_name=args.experiment_name
    )
    
    # Process dataset synchronously
    import asyncio
    
    job_id = f"cli_{int(time.time())}"
    active_jobs[job_id] = {
        "status": "queued",
        "request": request.model_dump(),  # Updated from .dict() to .model_dump()
        "total": 0,
        "current": 0,
        "results": [],
        "start_time": time.time()
    }
    
    asyncio.run(process_dataset(job_id, request))
    
    # Get results
    job = active_jobs[job_id]
    
    if job["status"] == "completed" and "response" in job:
        # Save to specified output file or use default
        output_file = args.output or f"results_{job_id}.json"
        
        with open(output_file, 'w') as f:
            json.dump(job["response"].model_dump(), f, indent=2)  # Updated from .dict() to .model_dump()
        
        print(f"Results saved to {output_file}")
        print(f"Metrics:")
        print(f"  Accuracy: {job['response'].metrics.accuracy:.4f}")
        print(f"  Precision: {job['response'].metrics.precision:.4f}")
        print(f"  Recall: {job['response'].metrics.recall:.4f}")
        print(f"  F1 Score: {job['response'].metrics.f1_score:.4f}")
        print(f"  Total Sequences: {job['response'].metrics.total_sequences}")
        print(f"  True Positives: {job['response'].metrics.true_positives}")
        print(f"  True Negatives: {job['response'].metrics.true_negatives}")
        print(f"  False Positives: {job['response'].metrics.false_positives}")
        print(f"  False Negatives: {job['response'].metrics.false_negatives}")
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
