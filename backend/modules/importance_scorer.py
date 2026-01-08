"""
Importance Scoring Module
Multi-factor importance scoring for keyframe selection.

Implements 6 importance factors with 7 architectural improvements:
1. Per-factor normalization to [0,1]
2. Two-stage selection (pre-filter + rank)
3. Cluster-aware constraints
4. Global temporal coverage (timeline segments)
5. Adaptive weights by video pacing
6. Importance confidence levels
7. Stability check

Research Foundation:
- "A frame is important if it is visually informative, representative 
   of many frames, and contributes to overall video coverage."
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum

from config import settings
from modules.feature_extractor import FeatureVector


class Confidence(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class VideoPacing(str, Enum):
    FAST = "fast"
    NORMAL = "normal"
    SLOW = "slow"


@dataclass
class ImportanceScore:
    """Container for a frame's importance evaluation."""
    frame_index: int
    timestamp: float
    cluster_id: int
    
    # Raw factor values (before normalization)
    raw_factors: Dict[str, float] = field(default_factory=dict)
    
    # Normalized factors [0,1]
    normalized_factors: Dict[str, float] = field(default_factory=dict)
    
    # Combined weighted score [0,1]
    combined_score: float = 0.0
    
    # Confidence level
    confidence: Confidence = Confidence.MEDIUM
    
    # Pre-filter status
    passed_prefilter: bool = True
    rejection_reason: Optional[str] = None
    
    # NEW: Entropy-based tracking (FIX 5: Reason tag)
    is_low_entropy: bool = False
    color_entropy: float = 0.0
    reason_tags: List[str] = field(default_factory=list)


# =============================================================================
# FACTOR 1: CLUSTER REPRESENTATIVENESS
# =============================================================================

def compute_cluster_representativeness(
    distance_to_centroid: float,
    max_distance: float,
    min_distance: float = 0.0
) -> float:
    """
    Compute how well this frame represents its cluster.
    
    Refined logic: Min-Max inverted normalization.
    - Frame with min_distance -> 1.0 (Best)
    - Frame with max_distance -> 0.0 (Worst)
    
    Args:
        distance_to_centroid: Frame's distance to its cluster centroid
        max_distance: Maximum distance in the cluster
        min_distance: Minimum distance in the cluster
        
    Returns:
        Representativeness score [0,1]
    """
    range_val = max_distance - min_distance
    if range_val <= 1e-6:
        # If all points are at same distance (e.g. single point), return 1.0
        return 1.0
    
    normalized = (distance_to_centroid - min_distance) / range_val
    return max(0.0, min(1.0, 1.0 - normalized))


# =============================================================================
# FACTOR 2: CLUSTER DOMINANCE (Scene Duration)
# =============================================================================

def compute_cluster_dominance(
    cluster_size: int,
    total_frames: int
) -> float:
    """
    Compute scene dominance based on cluster size.
    
    Larger clusters = scenes that appear longer = more important.
    
    Args:
        cluster_size: Number of frames in this cluster
        total_frames: Total number of candidate frames
        
    Returns:
        Dominance score [0,1] where 1 = largest cluster
    """
    if total_frames <= 0:
        return 0.0
    
    return cluster_size / total_frames


# =============================================================================
# FACTOR 3: VISUAL INFORMATION CONTENT (Richness) - ENTROPY-AWARE
# =============================================================================

def compute_color_entropy(frame_data: np.ndarray) -> float:
    """
    Compute color entropy of a frame.
    
    High entropy = diverse colors = rich content
    Low entropy = uniform colors = flat/trivial content
    
    Args:
        frame_data: BGR image array
        
    Returns:
        Color entropy value (higher = more diverse)
    """
    # Convert to HSV for better color analysis
    hsv = cv2.cvtColor(frame_data, cv2.COLOR_BGR2HSV)
    
    # Compute histograms for H and S channels
    h_hist = cv2.calcHist([hsv], [0], None, [30], [0, 180])  # 30 hue bins
    s_hist = cv2.calcHist([hsv], [1], None, [16], [0, 256])  # 16 saturation bins
    
    # Normalize to probabilities
    h_prob = h_hist.flatten() / (h_hist.sum() + 1e-10)
    s_prob = s_hist.flatten() / (s_hist.sum() + 1e-10)
    
    # Compute entropy: -sum(p * log(p))
    h_entropy = -np.sum(h_prob * np.log2(h_prob + 1e-10))
    s_entropy = -np.sum(s_prob * np.log2(s_prob + 1e-10))
    
    # Combined entropy (max possible ~4.9 for H, ~4 for S)
    combined_entropy = (h_entropy + s_entropy) / 2.0
    
    return float(combined_entropy)


def detect_uniform_region(frame_data: np.ndarray, threshold_ratio: float = 0.6) -> bool:
    """
    Detect if frame has a large uniform (single-color) region.
    
    Returns True if frame is mostly uniform (flat surface).
    """
    gray = cv2.cvtColor(frame_data, cv2.COLOR_BGR2GRAY)
    
    # Compute local standard deviation
    local_std = cv2.GaussianBlur(gray, (21, 21), 0)
    diff = cv2.absdiff(gray, local_std)
    
    # Count pixels with very low local variation
    uniform_pixels = np.sum(diff < 10)
    total_pixels = gray.size
    
    uniform_ratio = uniform_pixels / total_pixels
    
    return uniform_ratio > threshold_ratio


def compute_visual_richness(frame_data: np.ndarray) -> Tuple[float, float, bool]:
    """
    Compute visual information content of a frame (ENTROPY-AWARE).
    
    Visual richness is HIGH only if frame has:
    - Diverse colors (high entropy)
    - Structural edges
    - Non-uniform textures
    
    Brightness alone does NOT increase richness.
    
    Args:
        frame_data: BGR image array
        
    Returns:
        Tuple of (richness_score, entropy_score, is_low_entropy)
    """
    # 1. Color entropy (THE KEY FIX)
    entropy = compute_color_entropy(frame_data)
    
    # 2. Edge density (structural content)
    gray = cv2.cvtColor(frame_data, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size
    
    # 3. Texture richness (Laplacian variance)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    texture_richness = np.var(laplacian)
    
    # 4. Check for uniform regions
    is_uniform = detect_uniform_region(frame_data)
    
    # Determine if low entropy (threshold: ~2.5 on combined scale)
    is_low_entropy = entropy < 2.5 or is_uniform
    
    # APPLY LOW-ENTROPY PENALTY (FIX 2)
    entropy_multiplier = 1.0
    if is_low_entropy:
        entropy_multiplier = 0.3  # Heavily penalize
    elif entropy < 3.0:
        entropy_multiplier = 0.6  # Moderate penalty
    
    # Combine factors with entropy weighting
    # Note: Edge density and texture are MORE important than before
    base_richness = (
        (min(edge_density, 0.2) / 0.2) * 0.4 +           # Edges: 40%
        (min(texture_richness, 300) / 300.0) * 0.3 +     # Texture: 30%
        (min(entropy, 4.0) / 4.0) * 0.3                  # Entropy: 30%
    )
    
    # Apply entropy penalty
    final_richness = base_richness * entropy_multiplier
    
    return float(final_richness), float(entropy), is_low_entropy


def compute_brightness(frame_data: np.ndarray) -> float:
    """Compute average brightness of frame."""
    gray = cv2.cvtColor(frame_data, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray))


# =============================================================================
# FACTOR 4: TEMPORAL COVERAGE (Timeline Segments)
# =============================================================================

def compute_temporal_segments(
    video_duration: float,
    num_segments: int = 3
) -> List[Tuple[float, float]]:
    """
    Divide video timeline into segments for coverage analysis.
    
    Args:
        video_duration: Video length in seconds
        num_segments: Number of segments (default: 3 = Beginning/Middle/End)
        
    Returns:
        List of (start_time, end_time) tuples
    """
    segment_length = video_duration / num_segments
    segments = []
    
    for i in range(num_segments):
        start = i * segment_length
        end = (i + 1) * segment_length
        segments.append((start, end))
    
    return segments


def compute_temporal_coverage(
    timestamp: float,
    segment_coverage: Dict[int, int],
    segments: List[Tuple[float, float]],
    target_per_segment: int
) -> float:
    """
    Compute temporal coverage bonus for this frame.
    
    Frames in under-covered segments get higher scores.
    
    Args:
        timestamp: Frame's timestamp
        segment_coverage: Current coverage count per segment
        segments: Timeline segments
        target_per_segment: Target number of frames per segment
        
    Returns:
        Temporal coverage score [0,1] where 1 = high need for this segment
    """
    # Find which segment this frame belongs to
    segment_idx = 0
    for i, (start, end) in enumerate(segments):
        if start <= timestamp < end:
            segment_idx = i
            break
        if timestamp >= end and i == len(segments) - 1:
            segment_idx = i
    
    # Compute under-coverage bonus
    current_coverage = segment_coverage.get(segment_idx, 0)
    if target_per_segment <= 0:
        return 0.5
    
    # More bonus if segment is under-covered
    coverage_ratio = current_coverage / target_per_segment
    bonus = max(0.0, 1.0 - coverage_ratio)
    
    return bonus


# =============================================================================
# FACTOR 5: VISUAL NOVELTY
# =============================================================================

def compute_visual_novelty(
    frame_features: np.ndarray,
    selected_features: List[np.ndarray]
) -> float:
    """
    Compute how different this frame is from already selected keyframes.
    
    Higher novelty = less redundant with existing selections.
    
    Args:
        frame_features: Feature vector of current frame
        selected_features: List of feature vectors of already selected frames
        
    Returns:
        Novelty score [0,1] where 1 = very different from all selected
    """
    if not selected_features:
        return 1.0  # First frame is always novel
    
    # Compute minimum distance to any selected frame
    min_distance = float('inf')
    for sel_feat in selected_features:
        distance = np.linalg.norm(frame_features - sel_feat)
        min_distance = min(min_distance, distance)
    
    # Normalize (assuming features are unit-normalized, max distance ≈ 2)
    max_possible_distance = 2.0
    novelty = min(min_distance / max_possible_distance, 1.0)
    
    return novelty


# =============================================================================
# FACTOR 6: MOTION CONTEXT
# =============================================================================

def compute_motion_context(
    prev_frame: Optional[np.ndarray],
    curr_frame: np.ndarray,
    next_frame: Optional[np.ndarray]
) -> float:
    """
    Compute motion/change context around this frame.
    
    Frames near significant visual change are more important.
    
    Args:
        prev_frame: Previous frame data (or None)
        curr_frame: Current frame data
        next_frame: Next frame data (or None)
        
    Returns:
        Motion context score (raw, will be normalized)
    """
    motion_score = 0.0
    
    # Resize for efficiency
    size = (160, 90)
    curr_small = cv2.resize(curr_frame, size)
    curr_gray = cv2.cvtColor(curr_small, cv2.COLOR_BGR2GRAY)
    
    if prev_frame is not None:
        prev_small = cv2.resize(prev_frame, size)
        prev_gray = cv2.cvtColor(prev_small, cv2.COLOR_BGR2GRAY)
        diff_prev = np.mean(cv2.absdiff(curr_gray, prev_gray))
        motion_score += diff_prev
    
    if next_frame is not None:
        next_small = cv2.resize(next_frame, size)
        next_gray = cv2.cvtColor(next_small, cv2.COLOR_BGR2GRAY)
        diff_next = np.mean(cv2.absdiff(curr_gray, next_gray))
        motion_score += diff_next
    
    # Average if we have both
    if prev_frame is not None and next_frame is not None:
        motion_score /= 2.0
    
    return motion_score


# =============================================================================
# IMPROVEMENT 1: NORMALIZE FACTORS
# =============================================================================

def normalize_factor_scores(
    scores: List[ImportanceScore],
    factor_name: str
) -> None:
    """
    Normalize a specific factor across all frames to [0,1] range.
    
    Uses min-max normalization.
    Modifies scores in place.
    """
    values = [float(s.raw_factors.get(factor_name, 0)) for s in scores]
    
    if not values:
        return
    
    min_val = min(values)
    max_val = max(values)
    range_val = max_val - min_val
    
    if range_val <= 0:
        # All values are the same
        for score in scores:
            score.normalized_factors[factor_name] = 0.5
    else:
        for score in scores:
            raw = float(score.raw_factors.get(factor_name, 0))
            normalized = (raw - min_val) / range_val
            score.normalized_factors[factor_name] = float(normalized)


# =============================================================================
# IMPROVEMENT 2: TWO-STAGE SELECTION (PRE-FILTER)
# =============================================================================

def prefilter_frames(
    scores: List[ImportanceScore],
    richness_threshold: float = 0.25,
    brightness_min: float = 25.0,
    brightness_max: float = 240.0,
    motion_min: float = 0.1,
    entropy_min: float = 2.5  # NEW: Minimum entropy threshold
) -> List[ImportanceScore]:
    """
    Stage 1: Hard rejection of low-quality frames.
    
    Rejects (FIX 3 - Flat Surface Rejection):
    - Low visual richness (blank, flat frames)
    - Too dark or too bright frames
    - Near-static frames (low motion context)
    - LOW ENTROPY frames (uniform surfaces like white plates/lids)
    
    Args:
        scores: List of ImportanceScore objects
        richness_threshold: Minimum normalized richness to pass
        brightness_min: Minimum brightness
        brightness_max: Maximum brightness
        motion_min: Minimum normalized motion to pass
        entropy_min: Minimum color entropy to pass (FIX 3)
        
    Returns:
        Filtered list of frames that passed pre-filter
    """
    passed = []
    
    for score in scores:
        richness = score.normalized_factors.get('visual_richness', 0.5)
        brightness = score.raw_factors.get('brightness', 128)
        motion = score.normalized_factors.get('motion_context', 0.5)
        edge_density = score.raw_factors.get('edge_density', 0.1)
        
        # FIX 3: FLAT SURFACE REJECTION - Check low entropy first (most important)
        if score.is_low_entropy:
            # Additional check: if also low edge density = definitely flat
            if edge_density < 0.05:
                score.passed_prefilter = False
                score.rejection_reason = "Flat surface (low entropy + low edges)"
                score.reason_tags.append("flat_surface_rejected")
                continue
            # If low entropy but some edges, apply softer rejection
            elif score.color_entropy < 2.0:
                score.passed_prefilter = False
                score.rejection_reason = "Uniform surface (very low entropy)"
                score.reason_tags.append("low_entropy_rejected")
                continue
        
        if richness < richness_threshold:
            score.passed_prefilter = False
            score.rejection_reason = "Low visual richness"
            score.reason_tags.append("low_richness_rejected")
            continue
        
        if brightness < brightness_min:
            score.passed_prefilter = False
            score.rejection_reason = "Too dark"
            score.reason_tags.append("dark_rejected")
            continue
        
        if brightness > brightness_max:
            score.passed_prefilter = False
            score.rejection_reason = "Too bright/washed out"
            score.reason_tags.append("bright_rejected")
            continue
        
        # Reject near-static frames (only if motion is normalized)
        if motion < motion_min and score.normalized_factors.get('dominance', 0) < 0.3:
            score.passed_prefilter = False
            score.rejection_reason = "Static and unimportant"
            score.reason_tags.append("static_rejected")
            continue
        
        passed.append(score)
    
    return passed


# =============================================================================
# IMPROVEMENT 5: ADAPTIVE WEIGHTS BY VIDEO PACING
# =============================================================================

def detect_video_pacing(motion_scores: List[float]) -> VideoPacing:
    """
    Detect if video is fast-paced, slow, or normal.
    
    Args:
        motion_scores: List of motion context scores for all frames
        
    Returns:
        VideoPacing enum
    """
    if not motion_scores:
        return VideoPacing.NORMAL
    
    avg_motion = np.mean(motion_scores)
    std_motion = np.std(motion_scores)
    
    # Thresholds (can be tuned)
    if avg_motion > 20 or std_motion > 15:
        return VideoPacing.FAST
    elif avg_motion < 5 and std_motion < 5:
        return VideoPacing.SLOW
    else:
        return VideoPacing.NORMAL


def get_adaptive_weights(pacing: VideoPacing) -> Dict[str, float]:
    """
    Get adjusted weights based on video pacing.
    
    Fast video → higher motion weight
    Slow video → higher visual richness weight
    """
    base_weights = {
        'representativeness': 0.25,
        'dominance': 0.15,
        'visual_richness': 0.20,
        'temporal_coverage': 0.15,
        'visual_novelty': 0.15,
        'motion_context': 0.10
    }
    
    if pacing == VideoPacing.FAST:
        # Boost motion, reduce dominance
        base_weights['motion_context'] = 0.15
        base_weights['dominance'] = 0.10
    elif pacing == VideoPacing.SLOW:
        # Boost visual richness, reduce motion
        base_weights['visual_richness'] = 0.25
        base_weights['motion_context'] = 0.05
    
    # Ensure weights sum to 1.0
    total = sum(base_weights.values())
    return {k: v / total for k, v in base_weights.items()}


# =============================================================================
# IMPROVEMENT 6: CONFIDENCE LEVELS
# =============================================================================

def compute_confidence(
    score: float,
    all_scores: List[float],
    high_threshold: float = 0.7,  # Unused but kept for validation
    medium_threshold: float = 0.4 # Unused but kept for validation
) -> Confidence:
    """
    Compute confidence level based on RELATIVE RANKING (Percentiles).
    
    Top 20% -> HIGH
    Middle 60% -> MEDIUM
    Bottom 20% -> LOW
    
    Args:
        score: This frame's combined score
        all_scores: All combined scores in the video
        
    Returns:
        Confidence level
    """
    if not all_scores:
        return Confidence.MEDIUM
        
    # Sort scores to find percentiles
    sorted_scores = sorted(all_scores)
    n = len(sorted_scores)
    
    # Handle small number of frames
    if n < 5:
        if score > 0.7: return Confidence.HIGH
        if score < 0.4: return Confidence.LOW
        return Confidence.MEDIUM
    
    # Calculate ranking
    low_idx = int(n * 0.2)  # Bottom 20%
    high_idx = int(n * 0.8) # Top 20% starts here
    
    low_threshold = sorted_scores[low_idx]
    high_threshold = sorted_scores[high_idx]
    
    if score >= high_threshold:
        return Confidence.HIGH
    elif score <= low_threshold:
        return Confidence.LOW
    else:
        return Confidence.MEDIUM


# =============================================================================
# IMPROVEMENT 7: STABILITY CHECK
# =============================================================================

def check_summary_stability(
    scores: List[ImportanceScore],
    weights: Dict[str, float],
    top_n: int,
    perturbation: float = 0.05,
    num_trials: int = 5
) -> Tuple[bool, float]:
    """
    Check if summary is stable under weight perturbation.
    
    Recomputes importance with slightly different weights.
    If top frames remain the same, summary is stable.
    
    Args:
        scores: List of ImportanceScore objects
        weights: Current weights
        top_n: Number of top frames to compare
        perturbation: Weight perturbation amount (±)
        num_trials: Number of perturbation trials
        
    Returns:
        Tuple of (is_stable, stability_ratio)
    """
    # Get original top frames
    sorted_scores = sorted(scores, key=lambda s: s.combined_score, reverse=True)
    original_top = set(s.frame_index for s in sorted_scores[:top_n])
    
    stability_matches = 0
    
    for _ in range(num_trials):
        # Perturb weights
        perturbed = {}
        for k, v in weights.items():
            delta = np.random.uniform(-perturbation, perturbation)
            perturbed[k] = max(0.01, v + delta)
        
        # Normalize perturbed weights
        total = sum(perturbed.values())
        perturbed = {k: v / total for k, v in perturbed.items()}
        
        # Recompute combined scores
        for score in scores:
            new_combined = sum(
                perturbed.get(factor, 0) * score.normalized_factors.get(factor, 0)
                for factor in perturbed.keys()
            )
            score._perturbed_score = new_combined
        
        # Get new top frames
        resorted = sorted(scores, key=lambda s: getattr(s, '_perturbed_score', 0), reverse=True)
        new_top = set(s.frame_index for s in resorted[:top_n])
        
        # Check overlap
        overlap = len(original_top & new_top) / top_n
        stability_matches += overlap
    
    avg_stability = stability_matches / num_trials
    is_stable = avg_stability >= 0.8  # 80% overlap = stable
    
    return is_stable, avg_stability


# =============================================================================
# IMPROVEMENT 3: CLUSTER-AWARE SELECTION
# =============================================================================

def select_with_cluster_constraints(
    scores: List[ImportanceScore],
    num_clusters: int,
    min_per_cluster: int = 1,
    max_per_cluster: int = 3,
    target_keyframes: int = None,
    min_score: float = 0.40,
    min_time_gap: float = 1.0
) -> List[ImportanceScore]:
    """
    Select keyframes ensuring cluster coverage and importance.
    
    Includes:
    - Temporal Deduplication (min_time_gap)
    - Low Score Filtering (min_score)
    - Cluster Constraints
    
    Args:
        scores: List of ImportanceScore objects
        num_clusters: Total number of clusters
        min_per_cluster: Minimum frames per cluster (soft constraint)
        max_per_cluster: Maximum frames per cluster (hard constraint)
        target_keyframes: Target total keyframes
        min_score: Minimum importance score to be selected
        min_time_gap: Minimum time gap (seconds) between selected frames
        
    Returns:
        Selected keyframes respecting constraints
    """
    # Group by cluster
    cluster_frames: Dict[int, List[ImportanceScore]] = {}
    for score in scores:
        cid = score.cluster_id
        if cid not in cluster_frames:
            cluster_frames[cid] = []
        cluster_frames[cid].append(score)
    
    # Sort each cluster by importance
    for cid in cluster_frames:
        cluster_frames[cid].sort(key=lambda s: s.combined_score, reverse=True)
    
    selected = []
    cluster_counts = {cid: 0 for cid in range(num_clusters)}
    
    def is_temporally_distinct(frame, current_selected, gap):
        for s in current_selected:
            if abs(frame.timestamp - s.timestamp) < gap:
                return False
        return True
    
    # Stage 1 - Select best from each cluster
    for cid in range(num_clusters):
        if cid in cluster_frames and cluster_frames[cid]:
            best = cluster_frames[cid][0]
            
            # Skip cluster if best frame is low-entropy
            if best.is_low_entropy and best.color_entropy < 2.5:
                continue
            
            # Skip cluster if best frame has very low score
            if best.combined_score < min_score:
                continue
            
            # Use reduced gap for primary cluster representatives
            if is_temporally_distinct(best, selected, min_time_gap * 0.5):
                selected.append(best)
                cluster_counts[cid] = 1
    
    # Stage 2: Fill remaining slots by global importance ranking
    if target_keyframes:
        remaining_slots = target_keyframes - len(selected)
    else:
        remaining_slots = num_clusters * 2 - len(selected)
    
    # Get all remaining candidates
    remaining = []
    for cid, frames in cluster_frames.items():
        for frame in frames: # frames are sorted
            if frame in selected: continue 
            
            # Pre-check score before adding to candidate pool
            if frame.combined_score < min_score:
                continue
                
            if cluster_counts[cid] < max_per_cluster:
                if frame not in remaining: 
                    remaining.append(frame)
    
    # Sort by importance and select
    remaining.sort(key=lambda s: s.combined_score, reverse=True)
    
    for frame in remaining:
        if remaining_slots <= 0:
            break
            
        if not is_temporally_distinct(frame, selected, min_time_gap):
            continue
            
        if cluster_counts[frame.cluster_id] < max_per_cluster:
            selected.append(frame)
            cluster_counts[frame.cluster_id] += 1
            remaining_slots -= 1
    
    # Sort by timestamp for temporal ordering
    selected.sort(key=lambda s: s.timestamp)
    
    return selected


# =============================================================================
# MAIN IMPORTANCE COMPUTATION
# =============================================================================

def compute_importance_scores(
    feature_vectors: List[FeatureVector],
    normalized_features: np.ndarray,
    cluster_labels: np.ndarray,
    centroids: np.ndarray,
    video_duration: float,
    candidate_frames: List = None  # For accessing raw frame data
) -> Tuple[List[ImportanceScore], VideoPacing, Dict[str, float]]:
    """
    Compute importance scores for all candidate frames.
    
    This is the master function that:
    1. Computes all 6 raw factors
    2. Normalizes factors to [0,1]
    3. Detects video pacing and adjusts weights
    4. Computes combined scores
    5. Assigns confidence levels
    
    Args:
        feature_vectors: List of FeatureVector objects
        normalized_features: Normalized feature matrix
        cluster_labels: Cluster assignment for each frame
        centroids: Cluster centroids
        video_duration: Video length in seconds
        candidate_frames: Optional list of KeyframeCandidate for frame data
        
    Returns:
        Tuple of (importance_scores, video_pacing, final_weights)
    """
    n_samples = len(feature_vectors)
    n_clusters = len(centroids)
    
    # Compute cluster sizes
    cluster_sizes = {}
    for label in cluster_labels:
        cluster_sizes[label] = cluster_sizes.get(label, 0) + 1
    
    # Compute max/min distance per cluster (for normalization)
    cluster_max_distances = {}
    cluster_min_distances = {}  # NEW: For inverted normalization
    
    for i, label in enumerate(cluster_labels):
        dist = np.linalg.norm(normalized_features[i] - centroids[label])
        if label not in cluster_max_distances:
            cluster_max_distances[label] = dist
            cluster_min_distances[label] = dist
        else:
            cluster_max_distances[label] = max(cluster_max_distances[label], dist)
            cluster_min_distances[label] = min(cluster_min_distances[label], dist)
    
    # Temporal segments
    segments = compute_temporal_segments(video_duration, num_segments=3)
    target_per_segment = max(1, n_clusters // 3)
    
    # Storage for all scores
    scores: List[ImportanceScore] = []
    motion_scores_raw = []
    
    # First pass: Compute all raw factors
    for i, fv in enumerate(feature_vectors):
        cluster_id = int(cluster_labels[i])
        
        # Factor 1: Cluster representativeness (FIX 1: Use min_dist)
        dist = np.linalg.norm(normalized_features[i] - centroids[cluster_id])
        max_dist = cluster_max_distances.get(cluster_id, 1.0)
        min_dist = cluster_min_distances.get(cluster_id, 0.0)
        representativeness = compute_cluster_representativeness(dist, max_dist, min_dist)
        
        # Factor 2: Cluster dominance
        dominance = compute_cluster_dominance(cluster_sizes[cluster_id], n_samples)
        
        # Factor 3: Visual richness (ENTROPY-AWARE - returns tuple)
        richness, color_entropy, is_low_entropy = compute_visual_richness(fv.frame_data)
        brightness = compute_brightness(fv.frame_data)
        
        # Also compute edge density for flat surface detection
        gray = cv2.cvtColor(fv.frame_data, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = float(np.sum(edges > 0) / edges.size)
        
        # Factor 6: Motion context (need adjacent frames)
        prev_frame = feature_vectors[i-1].frame_data if i > 0 else None
        next_frame = feature_vectors[i+1].frame_data if i < n_samples - 1 else None
        motion = compute_motion_context(prev_frame, fv.frame_data, next_frame)
        motion_scores_raw.append(motion)
        
        score = ImportanceScore(
            frame_index=fv.frame_index,
            timestamp=float(fv.timestamp),
            cluster_id=cluster_id,
            raw_factors={
                'representativeness': float(representativeness),
                'dominance': float(dominance),
                'visual_richness': float(richness),
                'brightness': float(brightness),
                'motion_context': float(motion),
                'edge_density': edge_density
            },
            is_low_entropy=is_low_entropy,
            color_entropy=float(color_entropy),
            reason_tags=[]
        )
        
        if is_low_entropy:
            score.reason_tags.append("low_entropy_detected")
        
        scores.append(score)
    
    # Detect video pacing
    pacing = detect_video_pacing(motion_scores_raw)
    
    # Get adaptive weights
    weights = get_adaptive_weights(pacing)
    
    # Normalize only factors that need global normalization
    # visual_richness is naturally [0,1], do NOT normalize globally (preserves penalties)
    for factor in ['motion_context']:
        normalize_factor_scores(scores, factor)
    
    # For representativeness: just copy raw to normalized (already [0,1])
    for score in scores:
        score.normalized_factors['representativeness'] = score.raw_factors.get('representativeness', 0.5)
        # Copy visual_richness since we skipped normalization
        score.normalized_factors['visual_richness'] = score.raw_factors.get('visual_richness', 0.5)
    
    # For dominance: normalize relative to max cluster size BUT CAP IT (FIX 3)
    max_dominance = max(s.raw_factors.get('dominance', 0) for s in scores) if scores else 1.0
    for score in scores:
        raw_dom = score.raw_factors.get('dominance', 0)
        normalized_dom = float(raw_dom / max_dominance) if max_dominance > 0 else 0.5
        # Soft cap at 0.8
        score.normalized_factors['dominance'] = min(normalized_dom, 0.8)
    
    # ===========================================
    # FIX 2 & 3: COMPUTE REAL NOVELTY AND TEMPORAL COVERAGE
    # ===========================================
    
    # Temporal coverage: Divide video into 3 segments
    segment_duration = video_duration / 3.0
    segment_counts = [0, 0, 0]  # Beginning, Middle, End
    
    # First pass: count frames per segment
    for score in scores:
        seg_idx = min(2, int(score.timestamp / segment_duration))
        segment_counts[seg_idx] += 1
    
    # Compute ideal count per segment
    total_count = sum(segment_counts)
    ideal_per_segment = total_count / 3.0 if total_count > 0 else 1
    
    # Assign temporal coverage based on under-coverage (FIX 2: Softer decay + Cap)
    for score in scores:
        seg_idx = min(2, int(score.timestamp / segment_duration))
        coverage_ratio = segment_counts[seg_idx] / ideal_per_segment if ideal_per_segment > 0 else 1.0
        
        # Softer decay function: 1 / max(1, ratio)
        if coverage_ratio < 1.0:
            # Under-covered: Boost linearly from 1.0 to 1.5
            temporal_score = 1.0 + (1.0 - coverage_ratio) * 0.5
        else:
            # Over-covered: Soft decay
            temporal_score = 1.0 / coverage_ratio
            
        # Scale by 0.6 to enforce "Bonus not Gate" (Max contribution cap)
        # Result range: [0, 0.9] (since max score is 1.5 * 0.6 = 0.9)
        # Normal coverage (ratio=1) -> 0.6
        score.normalized_factors['temporal_coverage'] = float(min(1.0, temporal_score * 0.6))
    
    # Visual novelty: Compare each frame to others in same timestamp range
    for i, score in enumerate(scores):
        similar_count = 0
        for j, other in enumerate(scores):
            if i == j:
                continue
            if abs(score.timestamp - other.timestamp) < 1.0:
                similar_count += 1
        
        max_similar = 10
        novelty = max(0.0, 1.0 - (similar_count / max_similar))
        score.normalized_factors['visual_novelty'] = float(novelty)
    
    # Compute initial combined scores
    import random
    for score in scores:
        combined = sum(
            float(weights.get(factor, 0)) * float(score.normalized_factors.get(factor, 0))
            for factor in weights.keys()
        )
        
        # FIX 4: Score Spreading (Tie-Breaker)
        # Add tiny random noise + motion bonus to break exact ties
        tie_breaker = random.uniform(0, 0.002) + (score.normalized_factors.get('motion_context', 0) * 0.001)
        score.combined_score = float(combined + tie_breaker)
    
    # FIX 4: HARD LOW-ENTROPY PENALTY (NON-NEGOTIABLE)
    # Low-information frames are severely penalized
    for score in scores:
        if score.is_low_entropy:
            edge_density = score.raw_factors.get('edge_density', 0.1)
            
            # VERY low entropy + low edges = cap at 0.25 (very low)
            if score.color_entropy < 2.0 and edge_density < 0.05:
                if score.combined_score > 0.25:
                    score.combined_score = 0.25
                    score.reason_tags.append("hard_entropy_penalty_very_low")
            # Low entropy but some content = cap at 0.35
            elif score.color_entropy < 2.5:
                if score.combined_score > 0.35:
                    score.combined_score = 0.35
                    score.reason_tags.append("hard_entropy_penalty_low")
            # Moderate low entropy - NO LONGER CAP AT 0.45
            # Trust the natural penalty on visual_richness (0.3x multiplier)
            else:
                 score.reason_tags.append("entropy_penalty_moderate_applied")
    
    # Assign confidence levels
    all_combined = [s.combined_score for s in scores]
    for score in scores:
        score.confidence = compute_confidence(score.combined_score, all_combined)
        
        # FIX 4: Override - low entropy frames are ALWAYS LOW confidence
        if score.is_low_entropy:
            if score.color_entropy < 2.5:
                score.confidence = Confidence.LOW
                score.reason_tags.append("confidence_set_to_low_entropy")
            else:
                score.confidence = Confidence.MEDIUM
                score.reason_tags.append("confidence_downgraded_low_entropy")
    
    return scores, pacing, weights


def select_keyframes_with_importance(
    feature_vectors: List[FeatureVector],
    normalized_features: np.ndarray,
    cluster_labels: np.ndarray,
    centroids: np.ndarray,
    video_duration: float,
    target_keyframes: int = None
) -> Tuple[List[ImportanceScore], Dict]:
    """
    Full keyframe selection pipeline with importance scoring.
    
    Returns selected keyframes and metadata about the selection process.
    """
    n_clusters = len(centroids)
    
    # Compute initial importance scores
    scores, pacing, weights = compute_importance_scores(
        feature_vectors,
        normalized_features,
        cluster_labels,
        centroids,
        video_duration
    )
    
    # Pre-filter (Stage 1)
    passed = prefilter_frames(scores)
    
    if not passed:
        # Fallback: use all if pre-filter is too aggressive
        passed = scores
    
    # Stage 2: Selection
    # The scores already have valid factors from compute_importance_scores
    # including global novelty and temporal coverage.
    # We do NOT need to recompute them iteratively here.

    
    # Select with cluster constraints
    selected = select_with_cluster_constraints(
        passed,
        n_clusters,
        min_per_cluster=1,
        max_per_cluster=3,
        target_keyframes=target_keyframes
    )
    
    # Check stability
    is_stable, stability_ratio = check_summary_stability(
        passed, weights, len(selected)
    )
    
    # Prepare metadata
    metadata = {
        'video_pacing': pacing.value,
        'weights_used': weights,
        'summary_stable': is_stable,
        'stability_ratio': round(stability_ratio, 3),
        'total_candidates': len(scores),
        'passed_prefilter': len(passed),
        'selected_keyframes': len(selected),
        'summary_confidence': 'HIGH' if is_stable and stability_ratio > 0.9 else 
                              'MEDIUM' if stability_ratio > 0.7 else 'LOW'
    }
    
    return selected, metadata
