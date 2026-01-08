"""
Pipeline Orchestrator
Coordinates the complete video summarization workflow.
"""

import shutil
from pathlib import Path
from typing import Optional

from config import settings, ensure_job_directories
from database.models import (
    JobStatus, update_job_status, log_pipeline_event,
    save_keyframes, get_video
)
from modules.video_ingestion import ingest_video
from modules.frame_extractor import extract_frames
from modules.redundancy_filter import filter_redundant_frames
from modules.feature_extractor import extract_and_normalize
from modules.clustering import cluster_and_select_with_importance
from modules.summary_generator import generate_summary
from modules.scene_detection import detect_scenes_from_features, scenes_to_dict


async def process_video(job_id: int, video_id: int, video_path: str):
    """
    Execute the complete video summarization pipeline.
    
    Pipeline stages:
    1. Video ingestion and metadata extraction
    2. Frame decomposition
    3. Redundancy filtering
    4. Feature extraction and normalization
    5. Optimal K discovery (Elbow method)
    6. K-means clustering
    7. Representative frame selection
    8. Summary generation
    
    Args:
        job_id: Processing job ID
        video_id: Video record ID
        video_path: Path to uploaded video file
    """
    try:
        # Setup job directories
        job_paths = ensure_job_directories(job_id)
        
        # Get video metadata
        video_record = await get_video(video_id)
        
        # =====================================================
        # STAGE 1: EXTRACTING FRAMES
        # =====================================================
        await update_job_status(
            job_id, JobStatus.EXTRACTING_FRAMES,
            stage_detail="Decomposing video into frames..."
        )
        await log_pipeline_event(job_id, "extracting_frames", "Starting frame extraction")
        
        # Extract all frames using generator
        all_frames = list(extract_frames(
            video_path, 
            sample_rate=settings.FRAME_SAMPLE_RATE
        ))
        
        frames_extracted = len(all_frames)
        await update_job_status(
            job_id, JobStatus.EXTRACTING_FRAMES,
            stage_detail=f"Extracted {frames_extracted} frames",
            frames_extracted=frames_extracted
        )
        await log_pipeline_event(
            job_id, "extracting_frames", 
            f"Extracted {frames_extracted} frames from video"
        )
        
        # =====================================================
        # STAGE 2: FILTERING REDUNDANCY
        # =====================================================
        await update_job_status(
            job_id, JobStatus.FILTERING_REDUNDANCY,
            stage_detail="Removing similar and blurry frames..."
        )
        await log_pipeline_event(job_id, "filtering", "Starting redundancy and blur filtering")
        
        # Filter redundant, blurry, and dark frames (skip intros)
        candidates = list(filter_redundant_frames(
            iter(all_frames),
            threshold=settings.REDUNDANCY_THRESHOLD,
            method=settings.REDUNDANCY_METHOD,
            blur_threshold=settings.BLUR_THRESHOLD,
            brightness_threshold=settings.BRIGHTNESS_THRESHOLD,
            skip_intro_seconds=settings.SKIP_INTRO_SECONDS
        ))
        
        frames_filtered = len(candidates)
        await update_job_status(
            job_id, JobStatus.FILTERING_REDUNDANCY,
            stage_detail=f"Retained {frames_filtered} unique frames (from {frames_extracted})",
            frames_filtered=frames_filtered
        )
        await log_pipeline_event(
            job_id, "filtering",
            f"Filtered down to {frames_filtered} candidates (removed {frames_extracted - frames_filtered} similar frames)"
        )
        
        # Free memory from all_frames
        del all_frames
        
        # =====================================================
        # STAGE 3: EXTRACTING FEATURES
        # =====================================================
        await update_job_status(
            job_id, JobStatus.EXTRACTING_FEATURES,
            stage_detail="Extracting color, texture, and shape features..."
        )
        await log_pipeline_event(job_id, "features", "Starting feature extraction (Color+Texture+Shape)")
        
        feature_vectors, normalized_features = extract_and_normalize(
            candidates,
            use_combined=True  # Use all 3 feature types per research paper
        )
        
        feature_dim = normalized_features.shape[1]
        await log_pipeline_event(
            job_id, "features",
            f"Extracted {feature_dim}-dimensional feature vectors (HSV + Gabor + Fourier)"
        )
        
        # =====================================================
        # STAGE 4: NORMALIZING
        # =====================================================
        await update_job_status(
            job_id, JobStatus.NORMALIZING,
            stage_detail="Normalizing feature vectors for clustering..."
        )
        await log_pipeline_event(job_id, "normalizing", "Features normalized to unit length")
        
        # =====================================================
        # STAGE 5: FINDING OPTIMAL K
        # =====================================================
        await update_job_status(
            job_id, JobStatus.FINDING_OPTIMAL_K,
            stage_detail="Running elbow method to find optimal cluster count..."
        )
        await log_pipeline_event(job_id, "elbow", "Starting elbow method analysis")
        
        # =====================================================
        # STAGE 6: CLUSTERING + IMPORTANCE SCORING
        # =====================================================
        await update_job_status(
            job_id, JobStatus.CLUSTERING,
            stage_detail="Performing K-means clustering with importance scoring..."
        )
        
        cluster_result, representatives, importance_metadata = cluster_and_select_with_importance(
            feature_vectors,
            normalized_features,
            video_duration=video_record['duration']
        )
        
        await update_job_status(
            job_id, JobStatus.CLUSTERING,
            stage_detail=f"Found {cluster_result.optimal_k} clusters, selected {len(representatives)} keyframes",
            clusters_found=cluster_result.optimal_k
        )
        await log_pipeline_event(
            job_id, "clustering",
            f"K-means completed: K={cluster_result.optimal_k}, silhouette={cluster_result.silhouette:.3f}, "
            f"pacing={importance_metadata['video_pacing']}, confidence={importance_metadata['summary_confidence']}"
        )
        
        # =====================================================
        # SCENE DETECTION (Using extracted features)
        # =====================================================
        timestamps = [fv.timestamp for fv in feature_vectors]
        scenes, cut_indices = detect_scenes_from_features(
            normalized_features.tolist(),
            timestamps,
            threshold=0.3
        )
        
        # Add scene info to metadata
        importance_metadata['scenes'] = scenes_to_dict(scenes)
        importance_metadata['scene_count'] = len(scenes)
        
        await log_pipeline_event(
            job_id, "scene_detection",
            f"Detected {len(scenes)} scenes with {len(cut_indices)} cuts"
        )
        
        # =====================================================
        # STAGE 7: SELECTING KEYFRAMES
        # =====================================================
        await update_job_status(
            job_id, JobStatus.SELECTING_KEYFRAMES,
            stage_detail=f"Selecting {len(representatives)} representative frames..."
        )
        await log_pipeline_event(
            job_id, "selection",
            f"Selected {len(representatives)} frames closest to cluster centroids"
        )
        
        # =====================================================
        # STAGE 8: GENERATING SUMMARY
        # =====================================================
        await update_job_status(
            job_id, JobStatus.GENERATING_SUMMARY,
            stage_detail="Saving keyframes and creating storyboard..."
        )
        await log_pipeline_event(job_id, "summary", "Generating summary output")
        
        # Generate summary outputs
        summary_metadata = generate_summary(
            job_id=job_id,
            representatives=representatives,
            video_filename=video_record['filename'],
            video_duration=video_record['duration'],
            total_frames=video_record['total_frames'],
            frames_filtered=frames_filtered,
            importance_metadata=importance_metadata  # NEW: Include importance scores
        )
        
        # Save keyframe records to database
        keyframe_records = [
            {
                'frame_index': rep.feature_vector.frame_index,
                'timestamp': rep.feature_vector.timestamp,
                'cluster_id': rep.cluster_id,
                'distance_to_centroid': rep.distance_to_centroid,
                'output_path': summary_metadata.keyframes[idx]['path'],
                'is_representative': True
            }
            for idx, rep in enumerate(representatives)
        ]
        await save_keyframes(job_id, keyframe_records)
        
        await log_pipeline_event(
            job_id, "summary",
            f"Generated {len(representatives)} keyframe images and storyboard"
        )
        
        # =====================================================
        # COMPLETED
        # =====================================================
        await update_job_status(
            job_id, JobStatus.COMPLETED,
            stage_detail=f"Summary generated with {len(representatives)} keyframes"
        )
        await log_pipeline_event(job_id, "completed", "Pipeline completed successfully")
        
    except Exception as e:
        await update_job_status(
            job_id, JobStatus.FAILED,
            error=str(e)
        )
        await log_pipeline_event(job_id, "error", f"Pipeline failed: {str(e)}", level="error")
        raise
