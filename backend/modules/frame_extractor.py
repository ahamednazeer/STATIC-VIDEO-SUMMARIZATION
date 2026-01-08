"""
Frame Extraction Module
Decomposes video into discrete frames with ID and timestamp.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Generator, List, Union
from dataclasses import dataclass


@dataclass
class Frame:
    """Container for a single video frame."""
    index: int
    timestamp: float  # in seconds
    data: np.ndarray  # BGR image matrix


def extract_frames(
    video_path: Union[str, Path],
    sample_rate: int = 1
) -> Generator[Frame, None, None]:
    """
    Extract frames from video as a generator.
    
    Args:
        video_path: Path to the video file
        sample_rate: Extract every Nth frame (1 = all frames)
        
    Yields:
        Frame objects containing index, timestamp, and pixel data
    """
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_index = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_index % sample_rate == 0:
                timestamp = frame_index / fps if fps > 0 else 0
                yield Frame(
                    index=frame_index,
                    timestamp=timestamp,
                    data=frame
                )
            
            frame_index += 1
            
    finally:
        cap.release()


def extract_frames_batch(
    video_path: Union[str, Path],
    batch_size: int = 100,
    sample_rate: int = 1
) -> Generator[List[Frame], None, None]:
    """
    Extract frames in batches for memory efficiency.
    
    Args:
        video_path: Path to the video file
        batch_size: Number of frames per batch
        sample_rate: Extract every Nth frame
        
    Yields:
        Lists of Frame objects
    """
    batch = []
    
    for frame in extract_frames(video_path, sample_rate):
        batch.append(frame)
        
        if len(batch) >= batch_size:
            yield batch
            batch = []
    
    if batch:  # Yield remaining frames
        yield batch


def extract_specific_frames(
    video_path: Union[str, Path],
    frame_indices: List[int]
) -> List[Frame]:
    """
    Extract specific frames by their indices.
    
    Args:
        video_path: Path to the video file
        frame_indices: List of frame indices to extract
        
    Returns:
        List of Frame objects for requested indices
    """
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    sorted_indices = sorted(set(frame_indices))
    
    try:
        for target_index in sorted_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_index)
            ret, frame = cap.read()
            
            if ret:
                timestamp = target_index / fps if fps > 0 else 0
                frames.append(Frame(
                    index=target_index,
                    timestamp=timestamp,
                    data=frame
                ))
                
    finally:
        cap.release()
    
    return frames


def get_frame_at_timestamp(
    video_path: Union[str, Path],
    timestamp: float
) -> Frame:
    """
    Extract a single frame at a specific timestamp.
    
    Args:
        video_path: Path to the video file
        timestamp: Time in seconds
        
    Returns:
        Frame object at the specified timestamp
    """
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_index = int(timestamp * fps)
    
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        
        if not ret:
            raise ValueError(f"Cannot read frame at timestamp {timestamp}")
        
        return Frame(
            index=frame_index,
            timestamp=timestamp,
            data=frame
        )
        
    finally:
        cap.release()
