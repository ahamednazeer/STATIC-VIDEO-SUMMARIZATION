"""
Video Ingestion Module
Opens video file and extracts metadata (FPS, frame count, resolution, duration).
"""

import cv2
from pathlib import Path
from dataclasses import dataclass
from typing import Union


@dataclass
class VideoMetadata:
    """Container for video metadata."""
    filename: str
    filepath: str
    fps: float
    total_frames: int
    duration: float
    width: int
    height: int
    codec: str


def ingest_video(video_path: Union[str, Path]) -> VideoMetadata:
    """
    Open video file and extract metadata.
    
    Args:
        video_path: Path to the video file
        
    Returns:
        VideoMetadata object with all extracted information
        
    Raises:
        FileNotFoundError: If video file doesn't exist
        ValueError: If video cannot be opened or is invalid
    """
    path = Path(video_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    cap = cv2.VideoCapture(str(path))
    
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")
    
    try:
        # Extract metadata
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Get codec as fourcc
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
        
        # Calculate duration
        duration = total_frames / fps if fps > 0 else 0
        
        # Validate extracted data
        if fps <= 0 or total_frames <= 0:
            raise ValueError(f"Invalid video: FPS={fps}, Frames={total_frames}")
        
        return VideoMetadata(
            filename=path.name,
            filepath=str(path.absolute()),
            fps=fps,
            total_frames=total_frames,
            duration=duration,
            width=width,
            height=height,
            codec=codec
        )
        
    finally:
        cap.release()


def create_video_capture(video_path: Union[str, Path]) -> cv2.VideoCapture:
    """
    Create a video capture object for frame-by-frame reading.
    
    Args:
        video_path: Path to the video file
        
    Returns:
        cv2.VideoCapture object ready for reading
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    return cap
