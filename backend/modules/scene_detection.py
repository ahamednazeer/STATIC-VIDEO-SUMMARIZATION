"""
Scene Detection Module
Detects scene boundaries/cuts in video using histogram comparison.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class Scene:
    """Represents a detected scene."""
    scene_id: int
    start_time: float
    end_time: float
    start_frame: int
    end_frame: int
    duration: float
    keyframe_count: int = 0


def compute_histogram(frame: np.ndarray, bins: int = 64) -> np.ndarray:
    """
    Compute normalized HSV histogram for a frame.
    
    Args:
        frame: BGR image array
        bins: Number of bins per channel
        
    Returns:
        Flattened normalized histogram
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Compute histogram for H and S channels (V is less informative for scene detection)
    hist_h = cv2.calcHist([hsv], [0], None, [bins], [0, 180])
    hist_s = cv2.calcHist([hsv], [1], None, [bins], [0, 256])
    
    # Normalize
    cv2.normalize(hist_h, hist_h)
    cv2.normalize(hist_s, hist_s)
    
    return np.concatenate([hist_h.flatten(), hist_s.flatten()])


def compute_histogram_similarity(hist1: np.ndarray, hist2: np.ndarray) -> float:
    """
    Compute similarity between two histograms using correlation.
    
    Returns:
        Similarity score between -1 and 1 (1 = identical)
    """
    return cv2.compareHist(
        hist1.astype(np.float32), 
        hist2.astype(np.float32), 
        cv2.HISTCMP_CORREL
    )


def detect_scene_boundaries(
    frames: List[np.ndarray],
    timestamps: List[float],
    threshold: float = 0.5,
    min_scene_duration: float = 1.0
) -> Tuple[List[Scene], List[int]]:
    """
    Detect scene boundaries using histogram comparison.
    
    Args:
        frames: List of BGR frame arrays
        timestamps: List of timestamps for each frame
        threshold: Similarity threshold below which a cut is detected
        min_scene_duration: Minimum scene duration in seconds
        
    Returns:
        Tuple of (list of Scene objects, list of cut frame indices)
    """
    if len(frames) < 2:
        return [Scene(
            scene_id=0,
            start_time=timestamps[0] if timestamps else 0,
            end_time=timestamps[-1] if timestamps else 0,
            start_frame=0,
            end_frame=len(frames) - 1,
            duration=timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0
        )], []
    
    # Compute histograms
    histograms = [compute_histogram(f) for f in frames]
    
    # Compute frame-to-frame similarities
    similarities = []
    for i in range(len(histograms) - 1):
        sim = compute_histogram_similarity(histograms[i], histograms[i + 1])
        similarities.append(sim)
    
    # Detect cuts (low similarity = potential scene change)
    cut_indices = []
    for i, sim in enumerate(similarities):
        if sim < threshold:
            # Check minimum duration constraint
            last_cut = cut_indices[-1] if cut_indices else 0
            time_since_last = timestamps[i + 1] - timestamps[last_cut]
            
            if time_since_last >= min_scene_duration:
                cut_indices.append(i + 1)
    
    # Build scene list
    scenes = []
    scene_starts = [0] + cut_indices
    scene_ends = cut_indices + [len(frames) - 1]
    
    for i, (start_idx, end_idx) in enumerate(zip(scene_starts, scene_ends)):
        scenes.append(Scene(
            scene_id=i,
            start_time=float(timestamps[start_idx]),
            end_time=float(timestamps[end_idx]),
            start_frame=start_idx,
            end_frame=end_idx,
            duration=float(timestamps[end_idx] - timestamps[start_idx])
        ))
    
    return scenes, cut_indices


def detect_scenes_from_features(
    feature_vectors: List[np.ndarray],
    timestamps: List[float],
    threshold: float = 0.3
) -> Tuple[List[Scene], List[int]]:
    """
    Detect scene boundaries using pre-computed feature vectors.
    Uses cosine distance for comparison.
    
    Args:
        feature_vectors: List of normalized feature vectors
        timestamps: List of timestamps
        threshold: Distance threshold for scene cut
        
    Returns:
        Tuple of (scenes, cut_indices)
    """
    if len(feature_vectors) < 2:
        return [Scene(
            scene_id=0,
            start_time=timestamps[0] if timestamps else 0,
            end_time=timestamps[-1] if timestamps else 0,
            start_frame=0,
            end_frame=len(feature_vectors) - 1,
            duration=timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0
        )], []
    
    # Compute cosine distances between consecutive frames
    distances = []
    for i in range(len(feature_vectors) - 1):
        f1 = feature_vectors[i]
        f2 = feature_vectors[i + 1]
        
        # Cosine distance
        dot = np.dot(f1, f2)
        norm = np.linalg.norm(f1) * np.linalg.norm(f2)
        cos_sim = dot / norm if norm > 0 else 0
        distance = 1 - cos_sim
        
        distances.append(distance)
    
    # Adaptive threshold: mean + 2 * std
    if len(distances) > 5:
        mean_dist = np.mean(distances)
        std_dist = np.std(distances)
        adaptive_threshold = mean_dist + 2 * std_dist
        threshold = max(threshold, adaptive_threshold)
    
    # Detect cuts
    cut_indices = []
    for i, dist in enumerate(distances):
        if dist > threshold:
            cut_indices.append(i + 1)
    
    # Build scenes
    scenes = []
    scene_starts = [0] + cut_indices
    scene_ends = cut_indices + [len(feature_vectors) - 1]
    
    for i, (start_idx, end_idx) in enumerate(zip(scene_starts, scene_ends)):
        scenes.append(Scene(
            scene_id=i,
            start_time=float(timestamps[start_idx]),
            end_time=float(timestamps[end_idx]),
            start_frame=start_idx,
            end_frame=end_idx,
            duration=float(timestamps[end_idx] - timestamps[start_idx])
        ))
    
    return scenes, cut_indices


def scenes_to_dict(scenes: List[Scene]) -> List[dict]:
    """Convert scenes to JSON-serializable format."""
    return [
        {
            "scene_id": s.scene_id,
            "start_time": round(s.start_time, 2),
            "end_time": round(s.end_time, 2),
            "start_frame": s.start_frame,
            "end_frame": s.end_frame,
            "duration": round(s.duration, 2),
            "keyframe_count": s.keyframe_count
        }
        for s in scenes
    ]
