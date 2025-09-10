#!/usr/bin/env python3
"""
AI Service HTTP Server
Provides HTTP endpoints for the TruthSense AI analysis service
"""

import os
import json
import asyncio
import logging
import tempfile
import io
from datetime import datetime
from typing import Optional
from pathlib import Path
from contextlib import asynccontextmanager

import joblib
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ValidationError
from pydub import AudioSegment

from analyse import get_feedback
from schema import PostureFeatures, FrontendResponse
from dotenv import load_dotenv

# Load environment variables
# First try .env.local, then .env
load_dotenv('.env.local')
load_dotenv('.env', override=False)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global variables
fluency_model = None
processing_jobs = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the fluency model on startup
    global fluency_model
    try:
        model_path = Path(__file__).parent / "fluency_model.pkl"
        fluency_model = joblib.load(model_path)
        logger.info("Fluency model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load fluency model: {e}")
        raise
    
    yield
    
    # Cleanup on shutdown
    logger.info("Shutting down AI service")

# Initialize FastAPI app with lifespan
app = FastAPI(
    title="TruthSense AI Service",
    description="Speech and posture analysis service",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AnalysisRequest(BaseModel):
    recording_id: str
    domain: str
    posture_features: dict

class AnalysisResponse(BaseModel):
    success: bool
    recording_id: str
    analysis: Optional[dict] = None
    error: Optional[str] = None

class JobStatus(BaseModel):
    status: str  # "processing", "completed", "failed"
    progress: float
    message: str
    result: Optional[dict] = None
    error: Optional[str] = None


@app.get("/")
async def root_msg():
    return {"message": "Welcome to the AI service of TruthSense"}


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": fluency_model is not None
    }


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_recording(
    background_tasks: BackgroundTasks,
    recording_id: str = Form(...),
    domain: str = Form(...),
    posture_features: str = Form(...),
    audio_file: UploadFile = File(...)
):
    """
    Analyze an audio recording with posture features
    Returns immediately with job ID, actual processing happens in background
    """
    try:
        # Validate input - be more lenient with content types
        logger.info(f"")
        if audio_file.content_type and not (
            audio_file.content_type.startswith('audio/') or 
            audio_file.content_type == 'video/webm' or
            audio_file.content_type == 'application/octet-stream' or
            audio_file.filename.endswith(('.wav', '.mp3', '.m4a', '.webm'))
        ):
            raise HTTPException(status_code=400, detail=f"Invalid audio file format: {audio_file.content_type}\nAUDIO FILE TYPE: {audio_file.content_type}; DOMAIN: {domain}; FEATURES: {posture_features}")
        
        # Parse posture features
        try:
            posture_data = json.loads(posture_features)
            # Validate posture features against schema
            PostureFeatures(**posture_data)
        except (json.JSONDecodeError, ValidationError) as e:
            raise HTTPException(status_code=400, detail=f"Invalid posture features: {str(e)}")
        
        # Initialize job status
        processing_jobs[recording_id] = JobStatus(
            status="processing",
            progress=0.0,
            message="Starting analysis..."
        )
        
        # Read file content before passing to background task
        audio_content = await audio_file.read()
        
        # Start background processing
        background_tasks.add_task(
            process_analysis,
            recording_id,
            domain,
            posture_data,
            audio_content,
            audio_file.filename
        )
        
        return AnalysisResponse(
            success=True,
            recording_id=recording_id,
            analysis=None
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting analysis for {recording_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/status/{recording_id}")
async def get_job_status(recording_id: str):
    """Get the status of a processing job"""
    if recording_id not in processing_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = processing_jobs[recording_id]
    
    response = {
        "recording_id": recording_id,
        "status": job.status,
        "progress": job.progress,
        "message": job.message
    }
    
    if job.status == "completed" and job.result:
        response["analysis"] = job.result
    elif job.status == "failed" and job.error:
        response["error"] = job.error
    
    return response


async def process_analysis(
    recording_id: str,
    domain: str,
    posture_data: dict,
    audio_content: bytes,
    filename: str
):
    """Process the audio analysis in the background"""
    temp_audio_path = None
    try:
        # Update job status
        processing_jobs[recording_id].message = "Converting audio to WAV..."
        processing_jobs[recording_id].progress = 5.0

        # Convert webm to wav in memory
        try:
            webm_file = io.BytesIO(audio_content)
            sound = AudioSegment.from_file(webm_file, format='webm')
            
            wav_file = io.BytesIO()
            sound.export(wav_file, format='wav')
            wav_content = wav_file.getvalue()
            logger.info(f"Successfully converted {filename} to WAV format.")
        except Exception as conversion_error:
            logger.error(f"Audio conversion failed for {filename}: {conversion_error}")
            # Fallback to original content if conversion fails
            wav_content = audio_content


        # Update job status
        processing_jobs[recording_id].message = "Saving audio file..."
        processing_jobs[recording_id].progress = 10.0
        
        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
            temp_file.write(wav_content)
            temp_file.flush()
            temp_audio_path = temp_file.name
        
        # Update job status
        processing_jobs[recording_id].message = "Starting AI analysis..."
        processing_jobs[recording_id].progress = 20.0
        
        # Progress callback for transcription
        def progress_callback(progress, message):
            processing_jobs[recording_id].progress = progress
            processing_jobs[recording_id].message = message
        
        # Perform analysis
        logger.info(f"Starting analysis for recording {recording_id}")
        
        # Call main analysis function with progress tracking
        analysis_result = await get_feedback(
            temp_audio_path,
            fluency_model,
            posture_data,
            FrontendResponse
        )
        
        # Parse the JSON result (info fields are already included)
        result_dict = json.loads(analysis_result)
        
        # Update job status - completed
        processing_jobs[recording_id].status = "completed"
        processing_jobs[recording_id].progress = 100.0
        processing_jobs[recording_id].message = "Analysis completed successfully"
        processing_jobs[recording_id].result = result_dict
        
        logger.info(f"Analysis completed for recording {recording_id}")
        
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        logger.error(f"Analysis failed for recording {recording_id}: {str(e)}\n{error_traceback}")
        
        # Update job status - failed
        processing_jobs[recording_id].status = "failed"
        processing_jobs[recording_id].message = "Analysis failed"
        processing_jobs[recording_id].error = str(e)
        
    finally:
        # Clean up temporary file
        if temp_audio_path and os.path.exists(temp_audio_path):
            try:
                os.unlink(temp_audio_path)
            except Exception as e:
                logger.warning(f"Failed to clean up temp file {temp_audio_path}: {e}")


@app.delete("/jobs/{recording_id}")
async def cleanup_job(recording_id: str):
    """Clean up a completed or failed job"""
    if recording_id in processing_jobs:
        del processing_jobs[recording_id]
        return {"message": "Job cleaned up successfully"}
    else:
        raise HTTPException(status_code=404, detail="Job not found")


@app.get("/jobs")
async def list_jobs():
    """List all current jobs (for debugging)"""
    return {
        "jobs": {
            job_id: {
                "status": job.status,
                "progress": job.progress,
                "message": job.message
            }
            for job_id, job in processing_jobs.items()
        }
    }


if __name__ == "__main__":
    import uvicorn
    
    # Get configuration from environment
    host = os.getenv("AI_SERVER_HOST", "127.0.0.1")
    port = int(os.getenv("AI_SERVER_PORT", "8003"))
    
    logger.info(f"Starting AI service server on {host}:{port}")
    
    uvicorn.run(
        "ai_server:app",
        host=host,
        port=port,
        reload=False,
        log_level="info"
    )