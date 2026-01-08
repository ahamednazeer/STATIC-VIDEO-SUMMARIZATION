"""
Feature Extraction Module (Enhanced per VSUC Paper)
Extracts three types of features:
1. Color: HSV histogram (32H × 4S × 2V = 256 bins)
2. Texture: Gabor filter bank (5 scales × 8 orientations)
3. Shape: Fourier descriptors from edge contours
"""

import cv2
import numpy as np
from typing import List, Tuple
from dataclasses import dataclass

from modules.frame_extractor import Frame
from modules.redundancy_filter import KeyframeCandidate
from config import settings


@dataclass
class FeatureVector:
    """Container for a frame's feature representation."""
    frame_index: int
    timestamp: float
    features: np.ndarray
    frame_data: np.ndarray  # Original frame data for saving


# ============== COLOR FEATURES (HSV Histogram per paper) ==============

def extract_color_features(frame_data: np.ndarray) -> np.ndarray:
    """
    Extract HSV color histogram as per paper specification.
    Uses 32 bins for H, 4 bins for S, 2 bins for V = 256 total bins.
    """
    hsv = cv2.cvtColor(frame_data, cv2.COLOR_BGR2HSV)
    
    # Paper specifies: 32 bins H, 4 bins S, 2 bins V
    h_bins, s_bins, v_bins = 32, 4, 2
    
    # H channel: 0-180, S/V: 0-256
    h_hist = cv2.calcHist([hsv], [0], None, [h_bins], [0, 180])
    s_hist = cv2.calcHist([hsv], [1], None, [s_bins], [0, 256])
    v_hist = cv2.calcHist([hsv], [2], None, [v_bins], [0, 256])
    
    # Normalize each histogram
    h_hist = cv2.normalize(h_hist, h_hist).flatten()
    s_hist = cv2.normalize(s_hist, s_hist).flatten()
    v_hist = cv2.normalize(v_hist, v_hist).flatten()
    
    return np.concatenate([h_hist, s_hist, v_hist])  # 256 dims


# ============== TEXTURE FEATURES (Gabor Filters) ==============

def build_gabor_filters(
    scales: int = 5,
    orientations: int = 8,
    ksize: int = 31
) -> List[np.ndarray]:
    """
    Build a bank of Gabor filters for texture extraction.
    
    Args:
        scales: Number of frequency scales (paper uses 5)
        orientations: Number of orientations (paper uses 8)
        ksize: Kernel size
        
    Returns:
        List of Gabor filter kernels
    """
    filters = []
    for scale in range(scales):
        frequency = 0.05 + (scale * 0.1)  # Varying frequencies
        sigma = 3 + scale * 2
        lambd = 10.0 / (scale + 1)
        
        for orientation in range(orientations):
            theta = orientation * np.pi / orientations
            kernel = cv2.getGaborKernel(
                (ksize, ksize), sigma, theta, lambd, 0.5, 0, ktype=cv2.CV_32F
            )
            kernel /= kernel.sum()  # Normalize
            filters.append(kernel)
    
    return filters


# Pre-build Gabor filters for efficiency
GABOR_FILTERS = build_gabor_filters(scales=5, orientations=8)


def extract_texture_features(frame_data: np.ndarray) -> np.ndarray:
    """
    Extract texture features using Gabor filter bank.
    For each filter, compute mean and std of response = 2 features.
    5 scales × 8 orientations × 2 stats = 80 features.
    """
    gray = cv2.cvtColor(frame_data, cv2.COLOR_BGR2GRAY)
    
    # Resize for efficiency
    gray = cv2.resize(gray, (128, 128))
    
    features = []
    for kernel in GABOR_FILTERS:
        response = cv2.filter2D(gray, cv2.CV_32F, kernel)
        features.append(np.mean(response))
        features.append(np.std(response))
    
    return np.array(features, dtype=np.float32)  # 80 dims


# ============== SHAPE FEATURES (Fourier Descriptors) ==============

def extract_shape_features(frame_data: np.ndarray, num_coeffs: int = 32) -> np.ndarray:
    """
    Extract shape features using Fourier descriptors of edge contours.
    
    Args:
        frame_data: BGR image
        num_coeffs: Number of Fourier coefficients to keep
        
    Returns:
        Fourier descriptor feature vector
    """
    gray = cv2.cvtColor(frame_data, cv2.COLOR_BGR2GRAY)
    
    # Resize for efficiency
    gray = cv2.resize(gray, (128, 128))
    
    # Edge detection
    edges = cv2.Canny(gray, 50, 150)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return np.zeros(num_coeffs * 2, dtype=np.float32)
    
    # Use the largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    
    if len(largest_contour) < 4:
        return np.zeros(num_coeffs * 2, dtype=np.float32)
    
    # Flatten contour to complex numbers
    contour_points = largest_contour.reshape(-1, 2)
    complex_contour = contour_points[:, 0] + 1j * contour_points[:, 1]
    
    # Apply FFT
    fft_result = np.fft.fft(complex_contour)
    
    # Keep first N coefficients (normalized by DC component for scale invariance)
    if np.abs(fft_result[0]) > 0:
        fft_normalized = fft_result / np.abs(fft_result[0])
    else:
        fft_normalized = fft_result
    
    # Take magnitude of first num_coeffs coefficients
    coeffs = fft_normalized[:num_coeffs]
    
    # Pad if necessary
    if len(coeffs) < num_coeffs:
        coeffs = np.pad(coeffs, (0, num_coeffs - len(coeffs)))
    
    # Return real and imaginary parts
    return np.concatenate([np.real(coeffs), np.imag(coeffs)]).astype(np.float32)  # 64 dims


# ============== COMBINED FEATURE EXTRACTION ==============

def extract_combined_features(frame_data: np.ndarray) -> np.ndarray:
    """
    Extract all three feature types and concatenate them.
    
    Total dimensions: 256 (color) + 80 (texture) + 64 (shape) = 400
    """
    color = extract_color_features(frame_data)    # 256 dims
    texture = extract_texture_features(frame_data)  # 80 dims
    shape = extract_shape_features(frame_data)      # 64 dims
    
    return np.concatenate([color, texture, shape])


def extract_features(
    candidates: List[KeyframeCandidate],
    use_combined: bool = True
) -> List[FeatureVector]:
    """
    Extract feature vectors from keyframe candidates.
    
    Args:
        candidates: List of KeyframeCandidate objects
        use_combined: If True, use all 3 feature types; else HSV only
        
    Returns:
        List of FeatureVector objects
    """
    feature_vectors = []
    
    for candidate in candidates:
        frame = candidate.frame
        
        if use_combined:
            features = extract_combined_features(frame.data)
        else:
            features = extract_color_features(frame.data)
        
        feature_vectors.append(FeatureVector(
            frame_index=frame.index,
            timestamp=frame.timestamp,
            features=features,
            frame_data=frame.data
        ))
    
    return feature_vectors


def normalize_features(feature_vectors: List[FeatureVector]) -> np.ndarray:
    """
    Normalize feature vectors to unit length for K-means clustering.
    
    Args:
        feature_vectors: List of FeatureVector objects
        
    Returns:
        Normalized feature matrix (num_samples, num_features)
    """
    # Stack features into matrix
    feature_matrix = np.vstack([fv.features for fv in feature_vectors])
    
    # L2 normalize each feature vector
    norms = np.linalg.norm(feature_matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1  # Prevent division by zero
    
    normalized = feature_matrix / norms
    
    return normalized


def extract_and_normalize(
    candidates: List[KeyframeCandidate],
    use_combined: bool = True
) -> Tuple[List[FeatureVector], np.ndarray]:
    """
    Extract and normalize features in one step.
    
    Args:
        candidates: List of KeyframeCandidate objects
        use_combined: If True, use all 3 feature types
        
    Returns:
        Tuple of (feature_vectors, normalized_feature_matrix)
    """
    feature_vectors = extract_features(candidates, use_combined)
    normalized_features = normalize_features(feature_vectors)
    
    return feature_vectors, normalized_features
