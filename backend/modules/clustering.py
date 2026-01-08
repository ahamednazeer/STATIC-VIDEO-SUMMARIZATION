"""
Clustering Module
Implements K-means clustering with elbow method for optimal K selection.
Now enhanced with multi-factor importance scoring for keyframe selection.
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from config import settings
from modules.feature_extractor import FeatureVector


@dataclass
class ClusterResult:
    """Result of clustering operation."""
    labels: np.ndarray
    centroids: np.ndarray
    optimal_k: int
    wcss_values: List[float]  # For elbow plot
    silhouette: float


@dataclass
class RepresentativeFrame:
    """A frame selected as cluster representative."""
    feature_vector: FeatureVector
    cluster_id: int
    distance_to_centroid: float


def compute_wcss(features: np.ndarray, k: int) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Compute Within-Cluster Sum of Squares for given K.
    
    Args:
        features: Normalized feature matrix
        k: Number of clusters
        
    Returns:
        Tuple of (WCSS value, cluster labels, centroids)
    """
    kmeans = KMeans(
        n_clusters=k,
        max_iter=settings.KMEANS_MAX_ITERATIONS,
        n_init=settings.KMEANS_N_INIT,
        random_state=42
    )
    labels = kmeans.fit_predict(features)
    
    return kmeans.inertia_, labels, kmeans.cluster_centers_


def find_elbow_point(wcss_values: List[float]) -> int:
    """
    Find the elbow point in WCSS curve using the kneedle algorithm.
    
    The elbow is where the rate of decrease sharply changes.
    
    Args:
        wcss_values: List of WCSS values for K=min_k to K=max_k
        
    Returns:
        Index of elbow point (0-indexed relative to min_k)
    """
    if len(wcss_values) < 3:
        return 0
    
    # Normalize values
    x = np.arange(len(wcss_values))
    y = np.array(wcss_values)
    
    # Normalize to 0-1 range
    x_norm = (x - x.min()) / (x.max() - x.min())
    y_norm = (y - y.min()) / (y.max() - y.min() + 1e-10)
    
    # Compute distance from each point to the line connecting first and last points
    # Line from (0, y_norm[0]) to (1, y_norm[-1])
    # Distance = |ax + by + c| / sqrt(a^2 + b^2)
    # where line is: (y[-1] - y[0])x - y + y[0] = 0
    
    a = y_norm[-1] - y_norm[0]
    b = -1
    c = y_norm[0]
    
    distances = np.abs(a * x_norm + b * y_norm + c) / np.sqrt(a**2 + b**2)
    
    # Find point with maximum distance (the elbow)
    elbow_idx = np.argmax(distances)
    
    return elbow_idx


def find_optimal_k(
    features: np.ndarray,
    min_k: int = None,
    max_k: int = None,
    video_duration: float = None
) -> Tuple[int, List[float]]:
    """
    Find optimal number of clusters using elbow method combined with silhouette score.
    
    Improved to prefer more keyframes for better video coverage.
    Uses video duration to scale minimum clusters appropriately.
    
    Args:
        features: Normalized feature matrix
        min_k: Minimum clusters to try
        max_k: Maximum clusters to try
        video_duration: Video length in seconds (for adaptive scaling)
        
    Returns:
        Tuple of (optimal_k, wcss_values)
    """
    min_k = min_k or settings.MIN_CLUSTERS
    max_k = max_k or settings.MAX_CLUSTERS
    
    # Adaptive min_k based on video duration (aim for ~1 keyframe per 4-5 seconds)
    if video_duration is not None and video_duration > 0:
        duration_based_min = max(4, int(video_duration / 4))
        min_k = max(min_k, duration_based_min)
    
    # Ensure max_k doesn't exceed number of samples
    n_samples = features.shape[0]
    max_k = min(max_k, n_samples - 1) if n_samples > 2 else 2
    min_k = min(min_k, max_k)
    
    wcss_values = []
    silhouette_scores = []
    
    for k in range(min_k, max_k + 1):
        wcss, labels, _ = compute_wcss(features, k)
        wcss_values.append(wcss)
        
        # Compute silhouette score for this K
        if k >= 2 and n_samples > k:
            try:
                sil = silhouette_score(features, labels)
            except ValueError:
                sil = 0.0
        else:
            sil = 0.0
        silhouette_scores.append(sil)
    
    # Find elbow point
    elbow_idx = find_elbow_point(wcss_values)
    
    # Find best silhouette score
    best_sil_idx = np.argmax(silhouette_scores) if silhouette_scores else 0
    
    # Combine elbow and silhouette: prefer higher K between them
    # This ensures we don't under-cluster
    combined_idx = max(elbow_idx, best_sil_idx)
    
    # Additional check: if we have many samples, prefer at least sqrt(n) clusters
    sqrt_n = int(np.sqrt(n_samples))
    if sqrt_n > min_k and sqrt_n <= max_k - min_k:
        combined_idx = max(combined_idx, sqrt_n - min_k)
    
    optimal_k = min_k + combined_idx
    optimal_k = min(optimal_k, max_k)  # Ensure we don't exceed max
    
    return optimal_k, wcss_values


def cluster_features(
    features: np.ndarray,
    k: Optional[int] = None,
    video_duration: float = None
) -> ClusterResult:
    """
    Cluster feature vectors using K-means.
    
    Args:
        features: Normalized feature matrix
        k: Number of clusters (if None, uses elbow method)
        video_duration: Video length in seconds (for adaptive K selection)
        
    Returns:
        ClusterResult with labels, centroids, and metrics
    """
    n_samples = features.shape[0]
    
    # Edge case: too few samples for meaningful clustering
    if n_samples < 3:
        # Return all frames as separate clusters
        labels = np.arange(n_samples)
        centroids = features.copy()
        return ClusterResult(
            labels=labels,
            centroids=centroids,
            optimal_k=n_samples,
            wcss_values=[0.0],
            silhouette=0.0
        )
    
    if k is None:
        optimal_k, wcss_values = find_optimal_k(features, video_duration=video_duration)
    else:
        optimal_k = k
        _, wcss_values = find_optimal_k(features, video_duration=video_duration)  # Still compute for visualization
    
    # Perform final clustering with optimal K
    _, labels, centroids = compute_wcss(features, optimal_k)
    
    # Compute silhouette score only if we have enough samples
    # silhouette_score requires: 2 <= n_labels <= n_samples - 1
    n_unique_labels = len(np.unique(labels))
    if n_unique_labels >= 2 and n_samples > n_unique_labels:
        try:
            silhouette = silhouette_score(features, labels)
        except ValueError:
            silhouette = 0.0
    else:
        silhouette = 0.0
    
    return ClusterResult(
        labels=labels,
        centroids=centroids,
        optimal_k=optimal_k,
        wcss_values=wcss_values,
        silhouette=silhouette
    )


def select_representative_frames(
    feature_vectors: List[FeatureVector],
    normalized_features: np.ndarray,
    cluster_result: ClusterResult
) -> List[RepresentativeFrame]:
    """
    Select one representative frame per cluster (closest to centroid).
    
    Args:
        feature_vectors: Original feature vectors with frame data
        normalized_features: Normalized feature matrix
        cluster_result: Clustering result
        
    Returns:
        List of RepresentativeFrame objects, one per cluster
    """
    representatives = []
    
    for cluster_id in range(cluster_result.optimal_k):
        # Get indices of frames in this cluster
        cluster_indices = np.where(cluster_result.labels == cluster_id)[0]
        
        if len(cluster_indices) == 0:
            continue
        
        # Get centroid for this cluster
        centroid = cluster_result.centroids[cluster_id]
        
        # Compute distance of each frame to centroid
        cluster_features = normalized_features[cluster_indices]
        distances = np.linalg.norm(cluster_features - centroid, axis=1)
        
        # Select frame with minimum distance
        min_idx = np.argmin(distances)
        best_frame_idx = cluster_indices[min_idx]
        
        representatives.append(RepresentativeFrame(
            feature_vector=feature_vectors[best_frame_idx],
            cluster_id=cluster_id,
            distance_to_centroid=float(distances[min_idx])
        ))
    
    # Sort by timestamp for temporal ordering
    representatives.sort(key=lambda r: r.feature_vector.timestamp)
    
    return representatives


def apply_temporal_spacing(
    representatives: List[RepresentativeFrame],
    min_gap_seconds: float = 2.0
) -> List[RepresentativeFrame]:
    """
    Remove keyframes that are too close in time.
    
    Ensures minimum temporal spacing between consecutive keyframes.
    When two keyframes are too close, keeps the one with lower distance to centroid.
    
    Args:
        representatives: List sorted by timestamp
        min_gap_seconds: Minimum time gap between keyframes
        
    Returns:
        Filtered list with temporal spacing enforced
    """
    if not representatives or min_gap_seconds <= 0:
        return representatives
    
    filtered = [representatives[0]]  # Always keep first
    
    for rep in representatives[1:]:
        last_ts = filtered[-1].feature_vector.timestamp
        curr_ts = rep.feature_vector.timestamp
        
        if curr_ts - last_ts >= min_gap_seconds:
            # Sufficient gap, keep this frame
            filtered.append(rep)
        else:
            # Too close - keep the one with smaller distance to centroid
            if rep.distance_to_centroid < filtered[-1].distance_to_centroid:
                filtered[-1] = rep  # Replace with better representative
    
    return filtered


def cluster_and_select(
    feature_vectors: List[FeatureVector],
    normalized_features: np.ndarray,
    k: Optional[int] = None,
    min_gap_seconds: float = 0.0,
    video_duration: float = None
) -> Tuple[ClusterResult, List[RepresentativeFrame]]:
    """
    Perform clustering and select representative frames in one step.
    (Legacy function - kept for backward compatibility)
    
    Args:
        feature_vectors: List of FeatureVector objects
        normalized_features: Normalized feature matrix
        k: Optional fixed number of clusters
        min_gap_seconds: Minimum time between keyframes (0 = no spacing)
        video_duration: Video length in seconds (for adaptive clustering)
        
    Returns:
        Tuple of (ClusterResult, list of RepresentativeFrame)
    """
    cluster_result = cluster_features(normalized_features, k, video_duration)
    representatives = select_representative_frames(
        feature_vectors, normalized_features, cluster_result
    )
    
    # Apply temporal spacing if configured
    if min_gap_seconds > 0:
        representatives = apply_temporal_spacing(representatives, min_gap_seconds)
    
    return cluster_result, representatives


def cluster_and_select_with_importance(
    feature_vectors: List[FeatureVector],
    normalized_features: np.ndarray,
    video_duration: float,
    k: Optional[int] = None
) -> Tuple[ClusterResult, List[RepresentativeFrame], dict]:
    """
    Perform clustering and select keyframes using multi-factor importance scoring.
    
    This is the enhanced selection method that uses 6 importance factors:
    1. Cluster Representativeness
    2. Cluster Dominance (scene duration)
    3. Visual Richness
    4. Temporal Coverage
    5. Visual Novelty
    6. Motion Context
    
    Args:
        feature_vectors: List of FeatureVector objects
        normalized_features: Normalized feature matrix
        video_duration: Video length in seconds
        k: Optional fixed number of clusters
        
    Returns:
        Tuple of (ClusterResult, list of RepresentativeFrame, importance_metadata)
    """
    # Import here to avoid circular imports
    from modules.importance_scorer import (
        compute_importance_scores,
        select_with_cluster_constraints,
        prefilter_frames,
        check_summary_stability
    )
    
    # Perform clustering
    cluster_result = cluster_features(normalized_features, k, video_duration)
    
    # Compute importance scores for all frames
    importance_scores, pacing, weights = compute_importance_scores(
        feature_vectors,
        normalized_features,
        cluster_result.labels,
        cluster_result.centroids,
        video_duration
    )
    
    # Pre-filter low-quality frames
    passed_prefilter = prefilter_frames(
        importance_scores,
        richness_threshold=settings.PREFILTER_RICHNESS_THRESHOLD,
        brightness_min=settings.PREFILTER_BRIGHTNESS_MIN,
        brightness_max=settings.PREFILTER_BRIGHTNESS_MAX
    )
    
    # If too aggressive, fallback
    if len(passed_prefilter) < cluster_result.optimal_k:
        passed_prefilter = importance_scores
    
    # Select with cluster constraints
    selected_scores = select_with_cluster_constraints(
        passed_prefilter,
        cluster_result.optimal_k,
        min_per_cluster=settings.MIN_FRAMES_PER_CLUSTER,
        max_per_cluster=settings.MAX_FRAMES_PER_CLUSTER
    )
    
    # Check stability
    is_stable, stability_ratio = check_summary_stability(
        passed_prefilter, weights, len(selected_scores)
    )
    
    # Convert ImportanceScore objects to RepresentativeFrame objects
    # Create a mapping from frame_index to feature_vector
    fv_map = {fv.frame_index: fv for fv in feature_vectors}
    
    representatives = []
    for score in selected_scores:
        fv = fv_map.get(score.frame_index)
        if fv:
            # Compute distance to centroid
            fv_idx = feature_vectors.index(fv)
            centroid = cluster_result.centroids[score.cluster_id]
            distance = float(np.linalg.norm(normalized_features[fv_idx] - centroid))
            
            representatives.append(RepresentativeFrame(
                feature_vector=fv,
                cluster_id=score.cluster_id,
                distance_to_centroid=distance
            ))
    
    # Sort by timestamp
    representatives.sort(key=lambda r: r.feature_vector.timestamp)
    
    # Prepare importance metadata
    importance_metadata = {
        'video_pacing': pacing.value,
        'weights_used': {k: round(v, 3) for k, v in weights.items()},
        'summary_stable': is_stable,
        'stability_ratio': round(stability_ratio, 3),
        'total_candidates': len(importance_scores),
        'passed_prefilter': len(passed_prefilter),
        'selected_keyframes': len(representatives),
        'summary_confidence': 'HIGH' if is_stable and stability_ratio > 0.9 else 
                              'MEDIUM' if stability_ratio > 0.7 else 'LOW',
        'keyframe_scores': [
            {
                'frame_index': int(score.frame_index),
                'timestamp': float(score.timestamp),
                'combined_score': float(round(score.combined_score, 4)),
                'confidence': score.confidence.value,
                'factors': {k: float(round(v, 3)) for k, v in score.normalized_factors.items()},
                # NEW: Entropy debugging info (FIX 5)
                'is_low_entropy': score.is_low_entropy,
                'color_entropy': float(round(score.color_entropy, 2)),
                'reason_tags': score.reason_tags
            }
            for score in selected_scores
        ]
    }
    
    return cluster_result, representatives, importance_metadata

