"""
Summary Generator Module
Creates the final static summary output with multiple formats.
"""

import cv2
import json
import numpy as np
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

from config import settings, get_job_storage_path
from modules.clustering import RepresentativeFrame
from modules.deep_features import DeepFeatureExtractor


@dataclass
class SummaryMetadata:
    """Metadata about the generated summary."""
    job_id: int
    video_filename: str
    video_duration: float
    total_frames_original: int
    frames_after_filtering: int
    num_clusters: int
    keyframes: List[dict]
    generated_at: str
    output_format: str
    grid_path: Optional[str]
    # NEW: Importance scoring metadata
    video_pacing: Optional[str] = None
    summary_confidence: Optional[str] = None
    stability_ratio: Optional[float] = None
    importance_weights: Optional[dict] = None
    # NEW: Presentation metrics
    compression_ratio: Optional[float] = None
    redundancy_removed: Optional[float] = None
    temporal_coverage_score: Optional[float] = None
    summary_reason: Optional[str] = None


def generate_summary(
    job_id: int,
    representatives: List[RepresentativeFrame],
    video_filename: str,
    video_duration: float,
    total_frames: int,
    frames_filtered: int,
    importance_metadata: dict = None
) -> SummaryMetadata:
    """
    Generate complete summary output.
    
    Creates:
    - Individual keyframe images
    - Grid storyboard layout
    - JSON metadata file with importance scores and advanced metrics
    """
    job_path = get_job_storage_path(job_id)
    summary_path = job_path / "summary"
    summary_path.mkdir(parents=True, exist_ok=True)
    
    keyframe_data = []
    frame_images = []
    
    # Initialize Visual Context Extractor (CLIP)
    vce = DeepFeatureExtractor(model_name="clip")
    
    # 1. Save individual keyframes and collect metadata
    for idx, rep in enumerate(representatives):
        filename = f"keyframe_{idx+1:03d}.{settings.OUTPUT_IMAGE_FORMAT}"
        output_path = summary_path / filename
        
        save_keyframe_image(rep.feature_vector.frame_data, output_path)
        frame_images.append(rep.feature_vector.frame_data)
        
        # Build base kf metadata
        kf_data = {
            "index": idx + 1,
            "frame_id": rep.feature_vector.frame_index,
            "timestamp": rep.feature_vector.timestamp,
            "timestamp_formatted": format_timestamp(rep.feature_vector.timestamp),
            "cluster_id": rep.cluster_id,
            "distance_to_centroid": round(float(rep.distance_to_centroid), 6),
            "path": str(output_path.relative_to(job_path))
        }
        
        # Add importance scores if available
        if importance_metadata and 'keyframe_scores' in importance_metadata:
            for ks in importance_metadata['keyframe_scores']:
                if ks['frame_index'] == rep.feature_vector.frame_index:
                    kf_data['importance_score'] = float(ks['combined_score'])
                    kf_data['importance_confidence'] = ks['confidence']
                    factors = {k: float(v) if v is not None else 0.0 for k, v in ks['factors'].items()}
                    kf_data['importance_factors'] = factors
                    
                    if not vce.using_fallback:
                        # 1. Generative Deep Analysis (BLIP)
                        deep_data = vce.analyze_frame_deep(rep.feature_vector.frame_data)
                        kf_data['deep_analysis'] = deep_data
                        
                        # 2. Industry-Standard Context from Generative Caption
                        # Use the high-accuracy BLIP caption directly
                        kf_data['context'] = deep_data.get('caption', f"Keyframe at {kf_data['timestamp_formatted']}")
                        kf_data['domain'] = deep_data.get('domain', 'Visual Analysis')
                    else:
                        kf_data['context'] = f"Keyframe {idx+1}"
                    break
        
        # Fallback context if not already set
        if 'context' not in kf_data:
            kf_data['context'] = f"Keyframe {idx+1}"
        
        keyframe_data.append(kf_data)
    
    # 2. Create grid layout
    grid_path = None
    if settings.GENERATE_GRID_LAYOUT and frame_images:
        timestamps = [kf['timestamp'] for kf in keyframe_data]
        grid = create_grid_layout(frame_images, timestamps=timestamps)
        grid_filename = f"storyboard.{settings.OUTPUT_IMAGE_FORMAT}"
        grid_output = summary_path / grid_filename
        save_keyframe_image(grid, grid_output)
        grid_path = str(grid_output.relative_to(job_path))
    
    # 3. Calculate Advanced Metrics
    num_keyframes = len(representatives)
    
    # Compression Ratio: video_duration / (num_keyframes * 2.0s display time)
    display_time = num_keyframes * 2.0
    compression_ratio = video_duration / display_time if display_time > 0 else 1.0
    
    # Redundancy Removed: 1.0 - (num_keyframes / frames_after_filtering)
    redundancy_removed = 1.0 - (num_keyframes / frames_filtered) if frames_filtered > 0 else 0.0
    
    # Temporal Coverage Score: Average of 'temporal_coverage' factor
    temporal_coverage_avg = 0.0
    coverage_factors = []
    for kf in keyframe_data:
        if 'importance_factors' in kf and 'temporal_coverage' in kf['importance_factors']:
            coverage_factors.append(kf['importance_factors']['temporal_coverage'])
    if coverage_factors:
        temporal_coverage_avg = sum(coverage_factors) / len(coverage_factors)
    
    # 4. Computed Pacing & Reason
    pacing_ratio = num_keyframes / video_duration if video_duration > 0 else 0
    if pacing_ratio > 0.4:
        computed_pacing = "Fast"
        pacing_desc = "High motion"
    elif pacing_ratio < 0.1:
        computed_pacing = "Slow"
        pacing_desc = "Stable"
    else:
        computed_pacing = "Moderate"
        pacing_desc = "Balanced"
        
    summary_reason = f"{pacing_desc} content summarized into {num_keyframes} key moments (~{float(round(pacing_ratio, 2))} keyframes/sec)."
    
    # 5. Extract scene data from importance_metadata
    scenes = importance_metadata.get('scenes') if importance_metadata else None
    scene_count = importance_metadata.get('scene_count') if importance_metadata else None
    
    # 6. Create final metadata object
    metadata = SummaryMetadata(
        job_id=job_id,
        video_filename=video_filename,
        video_duration=round(float(video_duration), 2),
        total_frames_original=total_frames,
        frames_after_filtering=frames_filtered,
        num_clusters=num_keyframes,
        keyframes=keyframe_data,
        generated_at=datetime.utcnow().isoformat(),
        output_format=settings.OUTPUT_IMAGE_FORMAT,
        grid_path=grid_path,
        # Importance scoring metadata (prefer computed pacing)
        video_pacing=computed_pacing,
        summary_confidence=importance_metadata.get('summary_confidence') if importance_metadata else "MEDIUM",
        stability_ratio=float(importance_metadata.get('stability_ratio', 1.0)) if importance_metadata else 1.0,
        importance_weights=importance_metadata.get('weights_used') if importance_metadata else None,
        # Metrics
        compression_ratio=round(float(compression_ratio), 2),
        redundancy_removed=round(float(redundancy_removed), 4),
        temporal_coverage_score=round(float(temporal_coverage_avg), 2),
        summary_reason=summary_reason
    )
    
    # 7. Save metadata JSON
    metadata_path = summary_path / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(asdict(metadata), f, indent=2)
    
    # 8. Save metrics to database for fast listing
    try:
        import asyncio
        from database.models import aiosqlite
        
        async def update_db_metrics():
            async with aiosqlite.connect(settings.DATABASE_PATH) as db:
                await db.execute(
                    """UPDATE jobs SET 
                       video_pacing = ?, summary_confidence = ?, 
                       stability_ratio = ?, compression_ratio = ?,
                       redundancy_removed = ?, temporal_coverage_score = ?,
                       summary_reason = ?
                       WHERE id = ?""",
                    (
                        metadata.video_pacing, metadata.summary_confidence,
                        metadata.stability_ratio, metadata.compression_ratio,
                        metadata.redundancy_removed, metadata.temporal_coverage_score,
                        metadata.summary_reason, job_id
                    )
                )
                await db.commit()
                
        # Since this might be called from a sync context or a worker
        # we check if there's a running loop
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(update_db_metrics())
            else:
                loop.run_until_complete(update_db_metrics())
        except RuntimeError:
            asyncio.run(update_db_metrics())
            
    except Exception as e:
        print(f"[SummaryGenerator] Failed to save DB metrics: {e}")
    
    return metadata


def save_keyframe_image(
    frame_data: np.ndarray,
    output_path: Path,
    quality: int = None
) -> str:
    """
    Save a single keyframe image.
    
    Args:
        frame_data: BGR image array
        output_path: Path to save image
        quality: JPEG quality (1-100)
        
    Returns:
        String path to saved image
    """
    quality = quality or settings.OUTPUT_IMAGE_QUALITY
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    ext = output_path.suffix.lower()
    
    if ext in ['.jpg', '.jpeg']:
        cv2.imwrite(str(output_path), frame_data, 
                   [cv2.IMWRITE_JPEG_QUALITY, quality])
    elif ext == '.png':
        cv2.imwrite(str(output_path), frame_data,
                   [cv2.IMWRITE_PNG_COMPRESSION, 3])
    elif ext == '.webp':
        cv2.imwrite(str(output_path), frame_data,
                   [cv2.IMWRITE_WEBP_QUALITY, quality])
    else:
        cv2.imwrite(str(output_path), frame_data)
    
    return str(output_path)


def create_grid_layout(
    frames: List[np.ndarray],
    timestamps: List[float] = None,
    columns: int = None,
    padding: int = 10,
    bg_color: tuple = (30, 30, 30)
) -> np.ndarray:
    """
    Create a grid storyboard layout from keyframes with labels.
    
    Args:
        frames: List of BGR image arrays
        timestamps: List of timestamps in seconds for each frame
        columns: Number of columns in grid
        padding: Padding between images in pixels
        bg_color: Background color (BGR)
        
    Returns:
        Grid image as numpy array
    """
    if not frames:
        return np.zeros((100, 100, 3), dtype=np.uint8)
    
    columns = columns or settings.GRID_COLUMNS
    rows = (len(frames) + columns - 1) // columns
    
    # Find consistent size (use first frame dimensions)
    target_h, target_w = frames[0].shape[:2]
    
    # Resize to thumbnail for grid
    thumb_w = min(target_w, 400)
    thumb_h = int(thumb_w * target_h / target_w)
    
    # Calculate grid dimensions
    grid_w = columns * thumb_w + (columns + 1) * padding
    grid_h = rows * thumb_h + (rows + 1) * padding
    
    # Create background
    grid = np.full((grid_h, grid_w, 3), bg_color, dtype=np.uint8)
    
    # Place frames
    for idx, frame in enumerate(frames):
        row = idx // columns
        col = idx % columns
        
        # Resize frame
        thumb = cv2.resize(frame, (thumb_w, thumb_h))
        
        # Calculate position
        y = padding + row * (thumb_h + padding)
        x = padding + col * (thumb_w + padding)
        
        # Place in grid
        grid[y:y+thumb_h, x:x+thumb_w] = thumb
        
        # Add semi-transparent overlay bar at bottom for visibility
        overlay_y = y + thumb_h - 30
        cv2.rectangle(grid, (x, overlay_y), (x + thumb_w, y + thumb_h), (0, 0, 0), -1)
        
        # Add frame number at top-left
        cv2.putText(grid, f"#{idx+1}", (x + 5, y + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add timestamp at bottom if available
        if timestamps and idx < len(timestamps):
            ts_str = format_timestamp(timestamps[idx])
            cv2.putText(grid, ts_str, (x + 5, y + thumb_h - 8),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    return grid


def format_timestamp(seconds: float) -> str:
    """Convert seconds to MM:SS format."""
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins:02d}:{secs:02d}"


def generate_kf_context(factors: dict, kf_index: int) -> str:
    """Generate a human-readable context string for the keyframe."""
    top_factor = max(factors.items(), key=lambda x: x[1]) if factors else (None, 0)
    name, val = top_factor
    
    if name == 'representativeness' and val > 0.7:
        return "Main representative moment."
    elif name == 'visual_novelty' and val > 0.7:
        return "Introduces significant visual novelty."
    elif name == 'motion_context' and val > 0.7:
        return "Peak action highlight."
    elif name == 'visual_richness' and val > 0.7:
        return "Visually detailed highlight."
    elif name == 'temporal_coverage' and val > 0.7:
        return "Critical temporal anchor."
    elif name == 'dominance' and val > 0.7:
        return "Dominant visual state."
    else:
        return "Significant highlight."


def get_summary_info(job_id: int) -> Optional[dict]:
    """
    Load summary metadata for a completed job.
    
    Args:
        job_id: Processing job ID
        
    Returns:
        Summary metadata dict or None if not found
    """
    job_path = get_job_storage_path(job_id)
    metadata_path = job_path / "summary" / "metadata.json"
    
    if not metadata_path.exists():
        return None
    
    with open(metadata_path) as f:
        return json.load(f)
