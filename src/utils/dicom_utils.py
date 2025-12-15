"""
DICOM Utilities
===============

Helper functions for working with DICOM files.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path

import numpy as np


def get_dicom_info(dcm) -> Dict[str, Any]:
    """
    Extract relevant information from a DICOM dataset.
    
    Args:
        dcm: pydicom Dataset object
        
    Returns:
        Dict with DICOM metadata
    """
    info = {
        # Patient Info
        'PatientID': getattr(dcm, 'PatientID', 'Unknown'),
        'PatientAge': getattr(dcm, 'PatientAge', 'Unknown'),
        'PatientSex': getattr(dcm, 'PatientSex', 'Unknown'),
        
        # Study Info
        'StudyInstanceUID': getattr(dcm, 'StudyInstanceUID', 'Unknown'),
        'StudyDate': getattr(dcm, 'StudyDate', 'Unknown'),
        'StudyDescription': getattr(dcm, 'StudyDescription', 'Unknown'),
        
        # Series Info
        'SeriesInstanceUID': getattr(dcm, 'SeriesInstanceUID', 'Unknown'),
        'SeriesDescription': getattr(dcm, 'SeriesDescription', 'Unknown'),
        'Modality': getattr(dcm, 'Modality', 'Unknown'),
        
        # Image Info
        'Rows': getattr(dcm, 'Rows', None),
        'Columns': getattr(dcm, 'Columns', None),
        'PixelSpacing': list(getattr(dcm, 'PixelSpacing', [1.0, 1.0])),
        'SliceThickness': getattr(dcm, 'SliceThickness', None),
        'ImagePositionPatient': list(getattr(dcm, 'ImagePositionPatient', [0, 0, 0])),
        'ImageOrientationPatient': list(getattr(dcm, 'ImageOrientationPatient', [1, 0, 0, 0, 1, 0])),
        
        # CT Specific
        'KVP': getattr(dcm, 'KVP', None),
        'ConvolutionKernel': getattr(dcm, 'ConvolutionKernel', 'Unknown'),
        'RescaleIntercept': getattr(dcm, 'RescaleIntercept', 0),
        'RescaleSlope': getattr(dcm, 'RescaleSlope', 1),
        'WindowCenter': getattr(dcm, 'WindowCenter', None),
        'WindowWidth': getattr(dcm, 'WindowWidth', None),
    }
    
    return info


def windowing(
    volume: np.ndarray,
    window_center: float,
    window_width: float,
    rescale_intercept: float = 0,
    rescale_slope: float = 1,
) -> np.ndarray:
    """
    Apply window/level transformation to CT image.
    
    Common CT windows:
    - Brain: W=80, L=40
    - Stroke: W=8, L=32
    - Bone: W=2800, L=600
    - CTA (vessels): W=600, L=170
    - Soft tissue: W=400, L=50
    
    Args:
        volume: Image array (raw or HU)
        window_center: Window center (level)
        window_width: Window width
        rescale_intercept: DICOM rescale intercept
        rescale_slope: DICOM rescale slope
        
    Returns:
        Windowed image normalized to [0, 1]
    """
    # Convert to HU if needed
    img = volume * rescale_slope + rescale_intercept
    
    # Calculate window bounds
    min_val = window_center - window_width / 2
    max_val = window_center + window_width / 2
    
    # Apply window
    img = np.clip(img, min_val, max_val)
    
    # Normalize to [0, 1]
    img = (img - min_val) / (max_val - min_val)
    
    return img.astype(np.float32)


def normalize_volume(
    volume: np.ndarray,
    method: str = 'minmax',
    clip_percentile: Optional[Tuple[float, float]] = None,
) -> np.ndarray:
    """
    Normalize volume using various methods.
    
    Args:
        volume: 3D numpy array
        method: 'minmax', 'zscore', or 'percentile'
        clip_percentile: Optional percentile range for clipping (e.g., (1, 99))
        
    Returns:
        Normalized volume
    """
    volume = volume.astype(np.float32)
    
    # Clip outliers if specified
    if clip_percentile:
        low, high = np.percentile(volume, clip_percentile)
        volume = np.clip(volume, low, high)
    
    if method == 'minmax':
        v_min, v_max = volume.min(), volume.max()
        if v_max - v_min > 0:
            volume = (volume - v_min) / (v_max - v_min)
    
    elif method == 'zscore':
        mean, std = volume.mean(), volume.std()
        if std > 0:
            volume = (volume - mean) / std
    
    elif method == 'percentile':
        p1, p99 = np.percentile(volume, [1, 99])
        volume = np.clip(volume, p1, p99)
        volume = (volume - p1) / (p99 - p1 + 1e-8)
    
    return volume


def resample_to_isotropic(
    volume: np.ndarray,
    current_spacing: Tuple[float, float, float],
    target_spacing: float = 1.0,
    order: int = 1,
) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    """
    Resample volume to isotropic spacing.
    
    Args:
        volume: 3D numpy array (Z, Y, X)
        current_spacing: Current voxel spacing (z, y, x) in mm
        target_spacing: Target isotropic spacing
        order: Interpolation order
        
    Returns:
        Resampled volume and new spacing
    """
    from scipy import ndimage
    
    # Calculate zoom factors
    zoom_factors = [
        current_spacing[i] / target_spacing
        for i in range(3)
    ]
    
    # Resample
    resampled = ndimage.zoom(volume, zoom_factors, order=order)
    
    new_spacing = (target_spacing, target_spacing, target_spacing)
    
    return resampled, new_spacing


def get_spacing_from_dicom(dcm) -> Tuple[float, float, float]:
    """
    Extract voxel spacing from DICOM dataset.
    
    Args:
        dcm: pydicom Dataset
        
    Returns:
        Spacing as (z, y, x) tuple in mm
    """
    pixel_spacing = list(getattr(dcm, 'PixelSpacing', [1.0, 1.0]))
    slice_thickness = float(getattr(dcm, 'SliceThickness', 1.0))
    
    # DICOM PixelSpacing is (row, col) = (y, x)
    return (slice_thickness, pixel_spacing[0], pixel_spacing[1])


def create_brain_mask(
    volume: np.ndarray,
    threshold: float = -100,
) -> np.ndarray:
    """
    Create a simple brain mask from CT image.
    
    Args:
        volume: CT volume in HU
        threshold: HU threshold for brain tissue
        
    Returns:
        Binary mask
    """
    from scipy import ndimage
    
    # Initial threshold
    mask = volume > threshold
    
    # Fill holes
    mask = ndimage.binary_fill_holes(mask)
    
    # Keep largest connected component
    labeled, num_features = ndimage.label(mask)
    if num_features > 0:
        sizes = ndimage.sum(mask, labeled, range(1, num_features + 1))
        largest_label = np.argmax(sizes) + 1
        mask = labeled == largest_label
    
    # Smooth
    mask = ndimage.binary_erosion(mask, iterations=2)
    mask = ndimage.binary_dilation(mask, iterations=2)
    
    return mask.astype(np.uint8)


def crop_to_brain(
    volume: np.ndarray,
    mask: Optional[np.ndarray] = None,
    margin: int = 10,
) -> Tuple[np.ndarray, Tuple[slice, slice, slice]]:
    """
    Crop volume to brain bounding box.
    
    Args:
        volume: 3D numpy array
        mask: Optional brain mask (will create if not provided)
        margin: Margin in voxels around brain
        
    Returns:
        Cropped volume and bounding box slices
    """
    if mask is None:
        mask = create_brain_mask(volume)
    
    # Find bounding box
    coords = np.where(mask > 0)
    
    z_min, z_max = coords[0].min(), coords[0].max()
    y_min, y_max = coords[1].min(), coords[1].max()
    x_min, x_max = coords[2].min(), coords[2].max()
    
    # Add margin
    z_min = max(0, z_min - margin)
    y_min = max(0, y_min - margin)
    x_min = max(0, x_min - margin)
    z_max = min(volume.shape[0], z_max + margin)
    y_max = min(volume.shape[1], y_max + margin)
    x_max = min(volume.shape[2], x_max + margin)
    
    bbox = (slice(z_min, z_max), slice(y_min, y_max), slice(x_min, x_max))
    
    return volume[bbox], bbox

