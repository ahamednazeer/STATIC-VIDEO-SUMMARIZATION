"""
Application Configuration
Centralized settings for video summarization pipeline.
All parameters are configurable and documented for academic justification.

Research References:
- VSUMM (de Avila et al., 2011): Color histogram clustering
- VSUC Paper: HSV 32H×4S×2V bins, Gabor texture, Fourier shape
- Elbow method + Silhouette score for optimal K
"""

from pydantic_settings import BaseSettings
from pathlib import Path
from typing import Literal


class Settings(BaseSettings):
    """
    Application configuration settings.
    All parameters can be overridden via environment variables.
    """
    
    # ===========================================
    # PATH CONFIGURATION
    # ===========================================
    BASE_DIR: Path = Path(__file__).parent
    STORAGE_DIR: Path = BASE_DIR / "storage"
    DATABASE_PATH: Path = BASE_DIR / "database" / "video_summary.db"
    
    # ===========================================
    # VIDEO PROCESSING LIMITS
    # ===========================================
    MAX_VIDEO_DURATION: int = 10 * 60  # 10 minutes in seconds
    MAX_VIDEO_SIZE_MB: int = 500  # Maximum upload size
    SUPPORTED_FORMATS: tuple = (".mp4", ".avi", ".mov", ".mkv", ".webm")
    
    # ===========================================
    # FRAME SAMPLING PARAMETERS
    # Research: 1 fps is standard for summarization (30fps video → 1 frame/sec)
    # ===========================================
    FRAME_SAMPLE_RATE: int = 1  # Extract every Nth frame (1 = all frames)
    
    # ===========================================
    # REDUNDANCY FILTERING PARAMETERS
    # Research: Histogram intersection threshold 0.6-0.8 is commonly used
    # Lower = more frames retained, Higher = more aggressive filtering
    # ===========================================
    REDUNDANCY_THRESHOLD: float = 0.75  # Lower = keeps more distinct frames (more lenient)
    REDUNDANCY_METHOD: Literal["histogram", "pixel", "combined"] = "histogram"
    BLUR_THRESHOLD: float = 30.0  # Laplacian variance minimum (lower = more lenient)
    BRIGHTNESS_THRESHOLD: float = 25.0  # Skip frames darker than this (0-255)
    SKIP_INTRO_SECONDS: float = 3.0  # Skip first N seconds (logos, ratings)
    
    # ===========================================
    # FEATURE EXTRACTION PARAMETERS (per VSUC paper)
    # HSV: 32 bins H, 4 bins S, 2 bins V = 38 total
    # Gabor: 5 scales × 8 orientations = 80 features
    # Fourier: 32 coefficients × 2 (real+imag) = 64 features
    # ===========================================
    COLOR_SPACE: Literal["hsv", "rgb"] = "hsv"  # HSV more robust to illumination
    HSV_H_BINS: int = 32  # Hue bins (paper specification)
    HSV_S_BINS: int = 4   # Saturation bins
    HSV_V_BINS: int = 2   # Value bins
    GABOR_SCALES: int = 5         # Frequency scales
    GABOR_ORIENTATIONS: int = 8   # Direction orientations
    FOURIER_COEFFICIENTS: int = 32  # Shape descriptor size
    USE_COMBINED_FEATURES: bool = True  # Use all 3 feature types
    
    # ===========================================
    # CLUSTERING PARAMETERS
    # Research: Silhouette score > 0.5 indicates good clustering
    # Min clusters 5-8 for meaningful summaries of trailers/short videos
    # ===========================================
    MIN_CLUSTERS: int = 4   # Minimum keyframes in summary (lowered for short videos)
    MAX_CLUSTERS: int = 30  # Maximum keyframes in summary (increased)
    KMEANS_MAX_ITERATIONS: int = 300  # K-means convergence limit
    KMEANS_N_INIT: int = 15  # Number of K-means initializations (increased for stability)
    ELBOW_SENSITIVITY: float = 0.15  # Higher = prefer more clusters (less aggressive elbow)
    MIN_KEYFRAME_GAP_SECONDS: float = 1.0  # Minimum time between keyframes (reduced)
    
    # ===========================================
    # OUTPUT CONFIGURATION
    # ===========================================
    OUTPUT_IMAGE_FORMAT: Literal["jpg", "png", "webp"] = "jpg"
    OUTPUT_IMAGE_QUALITY: int = 95  # JPEG quality (1-100)
    GENERATE_GRID_LAYOUT: bool = True  # Create storyboard grid
    GRID_COLUMNS: int = 4  # Columns in grid layout (4 better than 3 for wider displays)
    
    # ===========================================
    # IMPORTANCE SCORING WEIGHTS (must sum to 1.0)
    # Research: Each factor contributes to overall frame importance
    # ===========================================
    WEIGHT_REPRESENTATIVENESS: float = 0.25  # How well frame represents cluster
    WEIGHT_DOMINANCE: float = 0.15           # Scene duration importance
    WEIGHT_VISUAL_RICHNESS: float = 0.20     # Visual information content
    WEIGHT_TEMPORAL_COVERAGE: float = 0.15   # Timeline distribution
    WEIGHT_VISUAL_NOVELTY: float = 0.15      # Difference from selected
    WEIGHT_MOTION_CONTEXT: float = 0.10      # Change detection
    
    # Pre-filter thresholds (Stage 1 hard rejection) - STRICTER
    PREFILTER_RICHNESS_THRESHOLD: float = 0.25  # Min richness to pass (raised from 0.15)
    PREFILTER_BRIGHTNESS_MIN: float = 25.0      # Min brightness
    PREFILTER_BRIGHTNESS_MAX: float = 240.0     # Max brightness
    PREFILTER_MOTION_MIN: float = 0.1           # NEW: Reject near-static frames
    
    # Cluster selection constraints - STRICTER
    MIN_FRAMES_PER_CLUSTER: int = 1  # At least 1 per cluster
    MAX_FRAMES_PER_CLUSTER: int = 2  # At most 2 per cluster (reduced from 3)
    
    # Confidence thresholds
    HIGH_CONFIDENCE_THRESHOLD: float = 0.7
    MEDIUM_CONFIDENCE_THRESHOLD: float = 0.4
    
    # ===========================================
    # API CONFIGURATION
    # ===========================================
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    CORS_ORIGINS: list = ["http://localhost:3000", "http://127.0.0.1:3000"]
    
    # ===========================================
    # WORKER CONFIGURATION
    # ===========================================
    WORKER_POLL_INTERVAL: float = 1.0  # Seconds between job checks
    MAX_CONCURRENT_JOBS: int = 1  # Single worker for academic project
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()


def get_job_storage_path(job_id: int) -> Path:
    """
    Get the storage directory for a specific job.
    
    Structure:
    storage/jobs/{job_id}/
        ├── original.mp4
        ├── frames/
        ├── filtered_frames/
        ├── features.npy
        ├── clusters.json
        └── summary/
    """
    job_path = settings.STORAGE_DIR / "jobs" / str(job_id)
    return job_path


def ensure_job_directories(job_id: int) -> dict:
    """
    Create all required directories for a job.
    Returns dict of paths.
    """
    base = get_job_storage_path(job_id)
    
    paths = {
        "base": base,
        "frames": base / "frames",
        "filtered": base / "filtered_frames", 
        "summary": base / "summary"
    }
    
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    
    return paths


# Ensure base storage directory exists
settings.STORAGE_DIR.mkdir(parents=True, exist_ok=True)
settings.DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
