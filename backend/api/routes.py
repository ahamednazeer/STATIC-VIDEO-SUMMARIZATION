"""
API Routes
REST endpoints for video upload, job status, and summary retrieval.
"""

import os
import shutil
import asyncio
import zipfile
import io
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from config import settings, get_job_storage_path, ensure_job_directories
from database.models import (
    create_video, create_job, get_job, get_job_with_video,
    get_keyframes, get_all_jobs, get_pipeline_logs,
    update_job_status, JobStatus
)
from modules.video_ingestion import ingest_video
from modules.summary_generator import get_summary_info


router = APIRouter(prefix="/api", tags=["video-summarization"])


# ===========================================
# RESPONSE MODELS
# ===========================================

class UploadResponse(BaseModel):
    success: bool
    job_id: int
    video_id: int
    message: str


class JobStatusResponse(BaseModel):
    job_id: int
    status: str
    progress: float
    current_stage: str
    stage_detail: Optional[str]
    error_message: Optional[str]
    frames_extracted: int
    frames_filtered: int
    clusters_found: int
    video_filename: Optional[str]
    video_duration: Optional[float]


class KeyframeInfo(BaseModel):
    index: int
    frame_id: int
    timestamp: float
    timestamp_formatted: str
    cluster_id: int
    context: Optional[str] = None
    domain: Optional[str] = None
    importance_score: Optional[float] = None
    importance_confidence: Optional[str] = None
    importance_factors: Optional[dict] = None
    deep_analysis: Optional[dict] = None


class SummaryResponse(BaseModel):
    job_id: int
    video_filename: str
    video_duration: float
    total_frames_original: int
    frames_after_filtering: int
    num_keyframes: int
    keyframes: list
    grid_available: bool
    # Importance metadata
    video_pacing: Optional[str] = None
    summary_confidence: Optional[str] = None
    stability_ratio: Optional[float] = None
    # Advanced metrics
    compression_ratio: Optional[float] = None
    redundancy_removed: Optional[float] = None
    temporal_coverage_score: Optional[float] = None
    summary_reason: Optional[str] = None
    # Scene detection
    scenes: Optional[list] = None
    scene_count: Optional[int] = None


# ===========================================
# ENDPOINTS
# ===========================================

@router.get("/health")
async def health_check():
    """API health check endpoint."""
    return {
        "status": "healthy",
        "service": "video-summarization",
        "max_upload_mb": settings.MAX_VIDEO_SIZE_MB
    }


@router.post("/videos/upload", response_model=UploadResponse)
async def upload_video(video: UploadFile = File(...)):
    """
    Upload a video for summarization.
    
    Creates a processing job that will be picked up by the background worker.
    """
    # Validate file extension
    file_ext = Path(video.filename).suffix.lower()
    if file_ext not in settings.SUPPORTED_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format. Allowed: {settings.SUPPORTED_FORMATS}"
        )
    
    # Create temporary file for initial analysis
    temp_path = settings.STORAGE_DIR / "temp" / video.filename
    temp_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # Save uploaded file temporarily
        with open(temp_path, "wb") as f:
            content = await video.read()
            if len(content) > settings.MAX_VIDEO_SIZE_MB * 1024 * 1024:
                raise HTTPException(
                    status_code=413,
                    detail=f"File too large. Max: {settings.MAX_VIDEO_SIZE_MB}MB"
                )
            f.write(content)
        
        # Extract video metadata
        metadata = ingest_video(temp_path)
        
        # Check duration limit
        if metadata.duration > settings.MAX_VIDEO_DURATION:
            raise HTTPException(
                status_code=400,
                detail=f"Video too long. Max: {settings.MAX_VIDEO_DURATION // 60} minutes"
            )
        
        # Create video record
        video_id = await create_video(
            filename=metadata.filename,
            original_path=str(temp_path),  # Will be moved later
            fps=metadata.fps,
            total_frames=metadata.total_frames,
            duration=metadata.duration,
            width=metadata.width,
            height=metadata.height,
            file_size=len(content)
        )
        
        # Create job
        job_id = await create_job(video_id)
        
        # Setup job directories and move video
        job_paths = ensure_job_directories(job_id)
        final_path = job_paths["base"] / f"original{file_ext}"
        shutil.move(str(temp_path), str(final_path))
        
        # Update video path in database
        from database.models import aiosqlite
        async with aiosqlite.connect(settings.DATABASE_PATH) as db:
            await db.execute(
                "UPDATE videos SET original_path = ? WHERE id = ?",
                (str(final_path), video_id)
            )
            await db.commit()
        
        return UploadResponse(
            success=True,
            job_id=job_id,
            video_id=video_id,
            message="Video uploaded. Processing will begin shortly."
        )
        
    except HTTPException:
        raise
    except Exception as e:
        # Cleanup on error
        if temp_path.exists():
            temp_path.unlink()
        raise HTTPException(status_code=500, detail=str(e))


def safe_int(value, default: int = 0) -> int:
    """Safely convert a value to int, handling bytes/None/corrupted data."""
    if value is None:
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, bytes):
        try:
            # Try to decode as little-endian int
            return int.from_bytes(value[:4], 'little') if len(value) >= 4 else default
        except:
            return default
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: int):
    """Get the current status of a processing job."""
    job = await get_job_with_video(job_id)
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return JobStatusResponse(
        job_id=job['id'],
        status=job['status'],
        progress=float(job.get('progress', 0) or 0),
        current_stage=job['current_stage'] or 'pending',
        stage_detail=job.get('stage_detail'),
        error_message=job.get('error_message'),
        frames_extracted=safe_int(job.get('frames_extracted'), 0),
        frames_filtered=safe_int(job.get('frames_filtered'), 0),
        clusters_found=safe_int(job.get('clusters_found'), 0),
        video_filename=job.get('filename'),
        video_duration=float(job.get('duration') or 0) if job.get('duration') else None
    )


@router.get("/jobs/{job_id}/summary", response_model=SummaryResponse)
async def get_summary(job_id: int):
    """Get the generated summary for a completed job."""
    job = await get_job_with_video(job_id)
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job['status'] != JobStatus.COMPLETED.value:
        raise HTTPException(
            status_code=400,
            detail=f"Job not completed. Current status: {job['status']}"
        )
    
    # Load summary metadata
    summary_info = get_summary_info(job_id)
    
    if not summary_info:
        raise HTTPException(status_code=404, detail="Summary not found")
    
    return SummaryResponse(
        job_id=job_id,
        video_filename=summary_info['video_filename'],
        video_duration=summary_info['video_duration'],
        total_frames_original=summary_info['total_frames_original'],
        frames_after_filtering=summary_info['frames_after_filtering'],
        num_keyframes=summary_info['num_clusters'],
        keyframes=summary_info['keyframes'],
        grid_available=summary_info.get('grid_path') is not None,
        # Importance metadata
        video_pacing=summary_info.get('video_pacing'),
        summary_confidence=summary_info.get('summary_confidence'),
        stability_ratio=summary_info.get('stability_ratio'),
        # Advanced metrics
        compression_ratio=summary_info.get('compression_ratio'),
        redundancy_removed=summary_info.get('redundancy_removed'),
        temporal_coverage_score=summary_info.get('temporal_coverage_score'),
        summary_reason=summary_info.get('summary_reason'),
        # Scene detection
        scenes=summary_info.get('scenes'),
        scene_count=summary_info.get('scene_count')
    )


@router.get("/jobs/{job_id}/keyframes/{keyframe_index}")
async def get_keyframe_image(job_id: int, keyframe_index: int):
    """Get a specific keyframe image."""
    job_path = get_job_storage_path(job_id)
    
    # Find the keyframe file
    summary_path = job_path / "summary"
    keyframe_pattern = f"keyframe_{keyframe_index:03d}.*"
    
    for ext in ['jpg', 'png', 'webp']:
        image_path = summary_path / f"keyframe_{keyframe_index:03d}.{ext}"
        if image_path.exists():
            return FileResponse(
                image_path,
                media_type=f"image/{ext if ext != 'jpg' else 'jpeg'}"
            )
    
    raise HTTPException(status_code=404, detail="Keyframe not found")


@router.get("/jobs/{job_id}/storyboard")
async def get_storyboard(job_id: int):
    """Get the grid storyboard image."""
    job_path = get_job_storage_path(job_id)
    summary_path = job_path / "summary"
    
    for ext in ['jpg', 'png', 'webp']:
        storyboard_path = summary_path / f"storyboard.{ext}"
        if storyboard_path.exists():
            return FileResponse(
                storyboard_path,
                media_type=f"image/{ext if ext != 'jpg' else 'jpeg'}"
            )
    
    raise HTTPException(status_code=404, detail="Storyboard not found")


@router.get("/jobs/{job_id}/video")
async def get_original_video(job_id: int):
    """Stream the original video file for playback."""
    job_path = get_job_storage_path(job_id)
    
    # Find the original video file
    for ext in ['mp4', 'webm', 'avi', 'mov', 'mkv']:
        video_path = job_path / f"original.{ext}"
        if video_path.exists():
            content_type = {
                'mp4': 'video/mp4',
                'webm': 'video/webm',
                'avi': 'video/x-msvideo',
                'mov': 'video/quicktime',
                'mkv': 'video/x-matroska'
            }.get(ext, 'video/mp4')
            
            return FileResponse(
                video_path,
                media_type=content_type,
                filename=f"video_{job_id}.{ext}"
            )
    
    raise HTTPException(status_code=404, detail="Original video not found")


@router.get("/jobs/{job_id}/download-all")
async def download_all_keyframes(job_id: int):
    """Download all keyframes as a ZIP archive."""
    job = await get_job_with_video(job_id)
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job['status'] != JobStatus.COMPLETED.value:
        raise HTTPException(
            status_code=400,
            detail=f"Job not completed. Current status: {job['status']}"
        )
    
    job_path = get_job_storage_path(job_id)
    summary_path = job_path / "summary"
    
    if not summary_path.exists():
        raise HTTPException(status_code=404, detail="Summary folder not found")
    
    # Create ZIP in memory
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        # Add all keyframe images
        for img_file in sorted(summary_path.glob("keyframe_*")):
            zip_file.write(img_file, img_file.name)
        
        # Add storyboard if exists
        for ext in ['jpg', 'png', 'webp']:
            storyboard = summary_path / f"storyboard.{ext}"
            if storyboard.exists():
                zip_file.write(storyboard, storyboard.name)
                break
    
    zip_buffer.seek(0)
    
    # Get video filename for the ZIP name
    video_name = job.get('filename', f'job_{job_id}')
    base_name = Path(video_name).stem
    
    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="{base_name}_keyframes.zip"'
        }
    )


@router.get("/jobs")
async def list_jobs():
    """List all processing jobs."""
    jobs = await get_all_jobs()
    return {"jobs": jobs, "total": len(jobs)}


@router.get("/jobs/{job_id}/logs")
async def get_job_logs(job_id: int):
    """Get processing logs for debugging."""
    logs = await get_pipeline_logs(job_id)
    return {"job_id": job_id, "logs": logs}


@router.get("/config")
async def get_config():
    """Get current pipeline configuration (for UI display)."""
    return {
        "max_video_duration_seconds": settings.MAX_VIDEO_DURATION,
        "max_video_size_mb": settings.MAX_VIDEO_SIZE_MB,
        "supported_formats": settings.SUPPORTED_FORMATS,
        "redundancy_threshold": settings.REDUNDANCY_THRESHOLD,
        "color_space": settings.COLOR_SPACE,
        "min_clusters": settings.MIN_CLUSTERS,
        "max_clusters": settings.MAX_CLUSTERS,
        "output_format": settings.OUTPUT_IMAGE_FORMAT
    }
