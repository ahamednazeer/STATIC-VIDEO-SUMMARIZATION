"""
Redundancy Filter Module
Removes visually similar consecutive frames and filters out blurry frames.
Uses histogram comparison and Laplacian variance for blur detection.
"""

import cv2
import numpy as np
from typing import List, Generator, Tuple
from dataclasses import dataclass

from modules.frame_extractor import Frame


@dataclass
class KeyframeCandidate:
    """A frame that passed redundancy and blur filtering."""
    frame: Frame
    similarity_score: float  # How different from previous frame
    sharpness_score: float = 0.0  # Higher = sharper (Laplacian variance)


def compute_blur_score(frame_data: np.ndarray) -> float:
    """
    Compute sharpness/blur score using Laplacian variance.
    
    Higher variance = sharper image, lower variance = blurry.
    This is a standard technique for blur detection.
    
    Args:
        frame_data: BGR image array
        
    Returns:
        Laplacian variance (higher = sharper)
    """
    gray = cv2.cvtColor(frame_data, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    variance = laplacian.var()
    return float(variance)


def compute_brightness(frame_data: np.ndarray) -> float:
    """
    Compute average brightness of a frame.
    
    Used to detect black/dark frames (intro screens, fades).
    
    Args:
        frame_data: BGR image array
        
    Returns:
        Average brightness (0-255, higher = brighter)
    """
    gray = cv2.cvtColor(frame_data, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray))


def compute_histogram(frame_data: np.ndarray, bins: int = 256) -> np.ndarray:
    """
    Compute color histogram for a frame.
    
    Args:
        frame_data: BGR image array
        bins: Number of histogram bins per channel
        
    Returns:
        Flattened histogram array
    """
    histograms = []
    
    for channel in range(3):
        hist = cv2.calcHist([frame_data], [channel], None, [bins], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        histograms.append(hist)
    
    return np.concatenate(histograms)


def compute_histogram_difference(hist1: np.ndarray, hist2: np.ndarray) -> float:
    """
    Compute similarity between two histograms using Bhattacharyya coefficient.
    
    Research Note: Bhattacharyya distance is recommended in VSUC paper for
    comparing histogram-based features. It's more robust than correlation.
    
    Args:
        hist1: First histogram (normalized)
        hist2: Second histogram (normalized)
        
    Returns:
        Similarity score between 0 and 1 (1 = identical, 0 = completely different)
    """
    # Bhattacharyya distance: measures overlap between distributions
    # Lower distance = more similar
    bhattacharyya_dist = cv2.compareHist(
        hist1.astype(np.float32),
        hist2.astype(np.float32),
        cv2.HISTCMP_BHATTACHARYYA
    )
    
    # Convert distance to similarity (0-1 scale)
    # Bhattacharyya distance ranges from 0 (identical) to 1 (completely different)
    similarity = 1.0 - bhattacharyya_dist
    return max(0.0, min(1.0, similarity))


def compute_pixel_difference(frame1: np.ndarray, frame2: np.ndarray) -> float:
    """
    Compute normalized pixel difference between frames.
    
    Args:
        frame1: First frame (BGR)
        frame2: Second frame (BGR)
        
    Returns:
        Similarity score between 0 and 1
    """
    # Convert to grayscale for faster computation
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    
    # Compute absolute difference
    diff = cv2.absdiff(gray1, gray2)
    
    # Normalize to 0-1 range (0 = identical, 1 = completely different)
    diff_score = np.mean(diff) / 255.0
    
    # Convert to similarity (1 = identical, 0 = completely different)
    return 1.0 - diff_score


def compute_structural_similarity(frame1: np.ndarray, frame2: np.ndarray) -> float:
    """
    Compute structural similarity using frame differencing.
    Faster alternative to full SSIM.
    
    Args:
        frame1: First frame (BGR)
        frame2: Second frame (BGR)
        
    Returns:
        Similarity score between 0 and 1
    """
    # Resize for faster computation
    size = (160, 90)  # 16:9 thumbnail
    small1 = cv2.resize(frame1, size)
    small2 = cv2.resize(frame2, size)
    
    # Compute mean squared error
    mse = np.mean((small1.astype(float) - small2.astype(float)) ** 2)
    
    # Convert to similarity (higher MSE = lower similarity)
    max_mse = 255.0 ** 2
    similarity = 1.0 - (mse / max_mse)
    
    return similarity


def filter_redundant_frames(
    frames: Generator[Frame, None, None],
    threshold: float = 0.95,
    method: str = "combined",
    blur_threshold: float = 50.0,
    brightness_threshold: float = 25.0,
    skip_intro_seconds: float = 0.0
) -> Generator[KeyframeCandidate, None, None]:
    """
    Filter out visually similar consecutive frames, blurry frames, and dark frames.
    
    Args:
        frames: Generator of Frame objects
        threshold: Similarity threshold (frames above this are discarded)
        method: 'histogram', 'pixel', or 'combined'
        blur_threshold: Minimum sharpness score (Laplacian variance)
        brightness_threshold: Minimum brightness (0-255, skip dark frames)
        skip_intro_seconds: Skip frames in first N seconds (logos, ratings)
        
    Yields:
        KeyframeCandidate objects that passed the filter
    """
    previous_frame = None
    previous_hist = None
    first_valid_frame_found = False
    
    for frame in frames:
        # Skip intro frames (logos, rating screens)
        if frame.timestamp < skip_intro_seconds:
            continue
        
        # Compute frame quality metrics
        sharpness = compute_blur_score(frame.data)
        brightness = compute_brightness(frame.data)
        
        # Skip dark/black frames
        if brightness < brightness_threshold:
            continue
            
        # Skip blurry frames
        if sharpness < blur_threshold:
            continue
            
        if previous_frame is None:
            # Keep first valid frame after intro
            previous_frame = frame
            previous_hist = compute_histogram(frame.data)
            first_valid_frame_found = True
            yield KeyframeCandidate(
                frame=frame, 
                similarity_score=0.0,
                sharpness_score=sharpness
            )
            continue
        
        # Compute similarity based on method
        if method == "histogram":
            current_hist = compute_histogram(frame.data)
            similarity = compute_histogram_difference(previous_hist, current_hist)
        elif method == "pixel":
            similarity = compute_pixel_difference(previous_frame.data, frame.data)
        else:  # combined
            current_hist = compute_histogram(frame.data)
            hist_sim = compute_histogram_difference(previous_hist, current_hist)
            pixel_sim = compute_structural_similarity(previous_frame.data, frame.data)
            similarity = (hist_sim + pixel_sim) / 2
        
        # Keep frame if sufficiently different
        if similarity < threshold:
            previous_frame = frame
            if method != "pixel":
                previous_hist = compute_histogram(frame.data)
            yield KeyframeCandidate(
                frame=frame,
                similarity_score=1.0 - similarity,  # Convert to difference score
                sharpness_score=sharpness
            )


def filter_frames_adaptive(
    frames: List[Frame],
    min_frames: int = 10,
    max_frames: int = 100,
    initial_threshold: float = 0.95
) -> List[KeyframeCandidate]:
    """
    Adaptively filter frames to get a target range of keyframe candidates.
    
    Args:
        frames: List of Frame objects
        min_frames: Minimum desired keyframe candidates
        max_frames: Maximum desired keyframe candidates
        initial_threshold: Starting similarity threshold
        
    Returns:
        List of KeyframeCandidate objects
    """
    threshold = initial_threshold
    candidates = []
    
    # Binary search for optimal threshold
    for _ in range(10):  # Max iterations
        candidates = list(filter_redundant_frames(
            iter(frames),
            threshold=threshold,
            method="combined"
        ))
        
        count = len(candidates)
        
        if min_frames <= count <= max_frames:
            break
        elif count < min_frames:
            # Too strict, lower threshold to keep more frames
            threshold += 0.02
        else:
            # Too lenient, raise threshold to filter more
            threshold -= 0.02
        
        # Clamp threshold
        threshold = max(0.5, min(0.99, threshold))
    
    return candidates
