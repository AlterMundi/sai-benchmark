import os
import base64
import time
import re
import random
from io import BytesIO
from typing import List, Dict, Any, Optional
import json

from fastapi import FastAPI, Body, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Import PIL for image processing
from PIL import Image

# Import ollama for local LLM inference
import ollama

# Define data models
class ImageInfo(BaseModel):
    path: str = Field(..., description="Path to image file within the specified folder")
    filename: Optional[str] = Field(None, description="Original filename of the image")

class AnalysisRequest(BaseModel):
    folder_path: str = Field("/root/sequences/", description="Path to folder containing images")
    images: Optional[List[ImageInfo]] = Field(None, description="List of specific images to analyze")
    model: Optional[str] = Field("gemma3:12b-it-q8_0", description="Ollama model to use for analysis")
    max_images: Optional[int] = Field(None, description="Maximum number of images to process")
    prompt: Optional[str] = Field(None, description="Custom prompt for the analysis")

class AnalysisResult(BaseModel):
    filename: str
    judgment: str
    confidence: Optional[float]
    justification: str
    processing_time: float
    raw_response: str

class AnalysisSummary(BaseModel):
    total_processed: int
    processing_time: float
    average_time_per_image: float
    yes_count: int
    no_count: int
    unknown_count: int
    
class AnalysisResponse(BaseModel):
    summary: AnalysisSummary
    results: List[AnalysisResult]

# Initialize FastAPI app
app = FastAPI(title="Image Smoke Analysis API")

# Function to convert image to base64
def image_to_base64(image_path):
    try:
        with Image.open(image_path) as img:
            buffered = BytesIO()
            img.save(buffered, format="JPEG")
            return base64.b64encode(buffered.getvalue()).decode("utf-8")
    except Exception as e:
        raise Exception(f"Error converting image to base64: {str(e)}")

# Function to parse Ollama response
def parse_response(output):
    juicio_match = re.search(r'Judgment:\s*(Yes|No)', output, re.IGNORECASE)
    confianza_match = re.search(r'Confidence:\s*([0-1](?:\.\d+)?)', output)
    justificacion_match = re.search(r'Justification:\s*(.*)', output, re.DOTALL)

    juicio = juicio_match.group(1).capitalize() if juicio_match else "Unknown"
    confianza = float(confianza_match.group(1)) if confianza_match and confianza_match.group(1) else None
    justificacion = justificacion_match.group(1).strip() if justificacion_match else "Not provided"
    
    return juicio, confianza, justificacion

# Default prompt for smoke detection
DEFAULT_PROMPT = """
Do you detect smoke related to wildfires in this image? Consider that smoke can vary in density, color (gray, white, black), and shape (columns, diffuse layers), and may be distant or nearby. Distinguish smoke from fog, mist, or clouds based on its origin (fire) and visual characteristics. Confidence represents your certainty in the judgment: 1.0 means absolute certainty (no doubt), 0.5 means complete uncertainty (equal chance of Yes/No), and values in between reflect your level of certainty based on the evidence. Respond in this format:
Judgment: [Yes/No]
Confidence: [number between 0.0 and 1.0]
Justification: [brief text]
"""

DEFAULT_PROMPT2 = """
You are looking at a sequence of images corresponding to an ongoing event. Please analize them carefully as an skilled firewatcher would looking for features corresponding to ongoing wildfire, specially noticiable trough smoke plumes. Give a single judge in the given format taking all images into account as part of a single event.
Judgment: [Yes/No]
Confidence: [number between 0.0 and 1.0]
Justification: [brief text citing features identified corresponding to ongoing wildfire events]
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


@app.post("/analyze-images", response_model=AnalysisResponse)
async def analyze_images(request: AnalysisRequest = Body(...)):
    start_time = time.time()
    
    # Validate folder path
    if not os.path.isdir(request.folder_path):
        raise HTTPException(status_code=400, detail=f"Folder path does not exist: {request.folder_path}")
    
    # Get list of images to process
    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff']
    image_files = []
    
    for filename in os.listdir(request.folder_path):
        if any(filename.lower().endswith(ext) for ext in valid_extensions):
            full_path = os.path.join(request.folder_path, filename)
            image_files.append({
                "path": full_path,
                "filename": filename
            })
    
    # Limit number of images if specified
    if request.max_images and len(image_files) > request.max_images:
        image_files = image_files[:request.max_images]
    
    if not image_files:
        raise HTTPException(status_code=404, detail=f"No image files found in folder: {request.folder_path}")
    
    # Convert all images to base64
    images_base64 = []
    for img_info in image_files:
        try:
            img_base64 = image_to_base64(img_info["path"])
            images_base64.append({
                "filename": img_info["filename"],
                "base64": img_base64
            })
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error processing image {img_info['filename']}: {str(e)}")
    
    # Use provided prompt or default
    prompt = request.prompt or DEFAULT_PROMPT2
    
    # Create a composite prompt listing all image filenames
    image_list = "\n".join([f"{i+1}. {img['filename']}" for i, img in enumerate(images_base64)])
    composite_prompt = f"{prompt}\n\nPlease analyze the following images:\n{image_list}\n\nProvide your analysis for each image, clearly indicating which image you're referring to."
    
    # Prepare message with all images
    message = {
        "role": "user", 
        "content": composite_prompt,
        "images": [img["base64"] for img in images_base64]
    }
    
    # Call Ollama API with all images at once
    try:
        response = ollama.chat(model=request.model, messages=[message])
        output = response["message"]["content"]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calling Ollama API: {str(e)}")
    
    # Process the response and generate results
    # Since we're getting a single response for all images, we need to parse it differently
    results = []
    
    # Return the raw response and let the client handle parsing
    # We'll include the original file list for reference
    total_time = time.time() - start_time
    
    summary = AnalysisSummary(
        total_processed=len(image_files),
        processing_time=total_time,
        average_time_per_image=total_time / len(image_files),
        yes_count=0,  # Can't determine without parsing individual results
        no_count=0,   # Can't determine without parsing individual results
        unknown_count=0
    )
    
    # Create a single result entry for the combined analysis
    combined_result = AnalysisResult(
        filename="combined_analysis",
        judgment="See raw response",
        confidence=None,
        justification="All images analyzed together. See raw response for details.",
        processing_time=total_time,
        raw_response=output
    )
    
    results.append(combined_result)
    
    return AnalysisResponse(summary=summary, results=results)



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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
