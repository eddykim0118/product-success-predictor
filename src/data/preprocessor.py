"""
CT/MRA Preprocessor
===================

This module handles preprocessing of 3D medical volumes.

WHY IS PREPROCESSING CRITICAL FOR MEDICAL AI?
----------------------------------------------

Unlike natural images (photos), medical images have these challenges:

1. PHYSICAL MEANING
   - Pixel values in CT are Hounsfield Units (HU), a physical measurement
   - We can't just normalize to [0,1] without losing diagnostic information
   - Must apply clinically meaningful "windowing"

2. VARIABLE RESOLUTION
   - Different scanners produce different voxel sizes
   - Patient A: 0.5 x 0.5 x 1.0 mm
   - Patient B: 0.7 x 0.7 x 2.5 mm
   - We must resample to consistent resolution for the CNN

3. 3D NATURE
   - Natural images: 2D, typically 3 channels (RGB)
   - Medical volumes: 3D, typically 1 channel (intensity)
   - Operations like resampling are much more expensive

PREPROCESSING PIPELINE
----------------------
Raw DICOM → [HU Window] → [Resample] → [Normalize] → [Crop/Pad] → Ready for CNN

Each step has a specific purpose:
- HU Windowing: Focus on relevant tissue types
- Resampling: Consistent physical scale across patients
- Normalization: Neural network-friendly value range
- Crop/Pad: Consistent array dimensions for batching

INTERVIEW TIP: Be ready to explain WHY each step exists and the trade-offs involved.

Author: Eddy Kim
"""

from typing import Dict, Optional, Tuple

import numpy as np
from scipy import ndimage


class CTPreprocessor:
    """
    Preprocessing pipeline for CT/MRA volumes.
    
    Why a Class Instead of Functions?
    ---------------------------------
    1. STATE: Store configuration (target_spacing, window, etc.)
    2. CONSISTENCY: Same preprocessing for train/val/test
    3. SERIALIZATION: Can save/load preprocessor with model
    4. COMPOSABILITY: Chain with augmentations using transforms
    
    Design Principle: Immutable Configuration
    -----------------------------------------
    Once initialized, the preprocessor behavior is fixed.
    This prevents subtle bugs from changing settings mid-training.
    
    Usage Example
    -------------
    >>> preprocessor = CTPreprocessor(
    ...     target_spacing=(1.0, 1.0, 1.0),  # 1mm isotropic
    ...     hu_window=(-100, 400),           # CTA window
    ... )
    >>> result = preprocessor(volume, spacing=(2.5, 0.5, 0.5))
    >>> processed_volume = result['volume']
    """
    
    def __init__(
        self,
        target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        hu_window: Tuple[float, float] = (-100, 400),
        clip_values: bool = True,
        normalize: bool = True,
    ):
        """
        Initialize the preprocessor.
        
        Parameters
        ----------
        target_spacing : tuple of float
            Target voxel spacing in mm as (z, y, x).
            
            ISOTROPIC VOXELS (same spacing in all dimensions):
            - Makes the 3D convolution kernels meaningful
            - A 3x3x3 kernel covers the same physical space in all directions
            - Standard choice: 1.0 mm for high detail, 2.0 mm for speed
            
            NON-ISOTROPIC (different z spacing):
            - Some pipelines use (2.0, 1.0, 1.0) - coarser in z-axis
            - Trade-off: Less detail vs less memory
            
        hu_window : tuple of float
            (min_hu, max_hu) for windowing.
            
            WINDOWING EXPLAINED:
            CT values range roughly -1000 (air) to +3000 (dense bone).
            We can't see everything at once - need to pick a range.
            
            Common windows:
            - Brain: (40, 80) - center 40, width 80 → range [0, 80]
            - Bone: (400, 1800) - see skeletal structures
            - Lung: (-600, 1600) - see lung parenchyma
            - CTA/vessels: (-100, 400) - see contrast-enhanced vessels
            
            For aneurysm detection (blood vessels with contrast):
            - Use CTA window: (-100, 400)
            - This makes vessels bright, other tissue gray
            
        clip_values : bool
            Whether to clip HU values to window range.
            Almost always True - values outside window are not informative.
            
        normalize : bool
            Whether to normalize to [0, 1] range after windowing.
            True for neural networks (they expect normalized inputs).
            
        Interview Tip: Explain why different windows show different anatomy!
        """
        self.target_spacing = target_spacing
        self.hu_window = hu_window
        self.clip_values = clip_values
        self.normalize = normalize
        
    def __call__(
        self,
        volume: np.ndarray,
        spacing: Tuple[float, float, float],
        target_shape: Optional[Tuple[int, int, int]] = None,
    ) -> Dict:
        """
        Apply full preprocessing pipeline.
        
        This is the main entry point. Can also be used as a transform.
        
        Parameters
        ----------
        volume : np.ndarray
            3D array of shape (D, H, W) in Hounsfield Units
        spacing : tuple of float
            Current voxel spacing (z, y, x) in mm
        target_shape : tuple of int, optional
            If provided, crop/pad to this shape after resampling
            
        Returns
        -------
        dict with keys:
            - 'volume': Preprocessed 3D array
            - 'spacing': New voxel spacing (should match target_spacing)
            - 'original_shape': Shape before preprocessing
            - 'original_spacing': Spacing before preprocessing
            
        Pipeline Order
        --------------
        1. Resample FIRST (changes physical scale)
        2. Window SECOND (changes value range)
        3. Normalize THIRD (scales to [0,1])
        4. Crop/Pad LAST (changes array size)
        
        Why this order?
        - Resampling needs original HU values for proper interpolation
        - Windowing clips values (destructive) - do after resampling
        - Normalization just scales - do after windowing
        - Crop/pad is purely array manipulation - do last
        """
        # Store original info for reference (useful for debugging/analysis)
        original_shape = volume.shape
        original_spacing = spacing
        
        # Step 1: Resample to target spacing
        volume, new_spacing = self.resample_volume(volume, spacing)
        
        # Step 2: Apply HU windowing
        volume = self.apply_hu_window(volume)
        
        # Step 3: Crop or pad if target shape specified
        if target_shape is not None:
            volume = self.pad_or_crop(volume, target_shape)
        
        return {
            'volume': volume,
            'spacing': new_spacing,
            'original_shape': original_shape,
            'original_spacing': original_spacing,
            'final_shape': volume.shape,
        }
    
    def resample_volume(
        self,
        volume: np.ndarray,
        current_spacing: Tuple[float, float, float],
        order: int = 1,
    ) -> Tuple[np.ndarray, Tuple[float, float, float]]:
        """
        Resample volume to target spacing.
        
        WHY RESAMPLING IS NECESSARY
        ---------------------------
        Different scanners/protocols produce different resolutions.
        
        Example:
        - Scanner A: 0.5 x 0.5 x 1.0 mm → 500 x 500 x 300 voxels
        - Scanner B: 0.7 x 0.7 x 2.5 mm → 360 x 360 x 120 voxels
        
        Without resampling:
        - A 3x3x3 kernel covers different physical volumes!
        - Features learned on Scanner A won't transfer to Scanner B
        - Batch training impossible (different sizes)
        
        With resampling to 1x1x1 mm:
        - Both become consistent physical scale
        - 3x3x3 kernel = 3x3x3 mm everywhere
        - Model generalizes across scanners
        
        Parameters
        ----------
        volume : np.ndarray
            Input volume (D, H, W)
        current_spacing : tuple
            Current voxel spacing (z, y, x) in mm
        order : int
            Interpolation order:
            - 0: Nearest neighbor (fast, preserves exact values)
            - 1: Linear/trilinear (default, smooth)
            - 3: Cubic (smoother but slower)
            
            For intensity images: order=1 is standard
            For segmentation masks: order=0 (preserve labels)
            
        Returns
        -------
        resampled : np.ndarray
            Resampled volume
        new_spacing : tuple
            Should match target_spacing
            
        MATH: Zoom Factors
        ------------------
        zoom_factor = current_spacing / target_spacing
        
        If current=2mm, target=1mm → zoom=2 (upsample, more voxels)
        If current=0.5mm, target=1mm → zoom=0.5 (downsample, fewer voxels)
        
        New shape = old_shape * zoom_factor
        """
        # Calculate zoom factors for each dimension
        zoom_factors = tuple(
            current_spacing[i] / self.target_spacing[i]
            for i in range(3)
        )
        
        # scipy.ndimage.zoom handles the interpolation
        # Uses efficient C implementation under the hood
        resampled = ndimage.zoom(volume, zoom_factors, order=order)
        
        return resampled, self.target_spacing
    
    def apply_hu_window(self, volume: np.ndarray) -> np.ndarray:
        """
        Apply Hounsfield Unit windowing and optional normalization.
        
        WINDOWING IN DETAIL
        -------------------
        CT values span a huge range: -1000 (air) to +3000 (metal)
        Monitors only display ~256 gray levels (8-bit)
        Our eyes can distinguish maybe 100 gray levels
        
        Solution: Pick a WINDOW (range) to display:
        - Values below window → black (0)
        - Values above window → white (1)
        - Values within window → linear mapping
        
        For CTA (CT Angiography) with contrast:
        - Window: (-100, 400) HU
        - Blood vessels with contrast: 100-300 HU → appear bright
        - Soft tissue: 30-80 HU → appears mid-gray
        - Air/fat: negative HU → appears dark/black
        
        Parameters
        ----------
        volume : np.ndarray
            Volume in Hounsfield Units
            
        Returns
        -------
        np.ndarray
            Windowed (and optionally normalized) volume
            
        Visualization Tip
        -----------------
        To verify windowing works:
        >>> import matplotlib.pyplot as plt
        >>> plt.subplot(121); plt.imshow(volume[100], cmap='gray')  # Raw
        >>> plt.subplot(122); plt.imshow(windowed[100], cmap='gray')  # Windowed
        >>> plt.show()
        
        Good windowing: vessels are clearly visible
        Bad windowing: everything looks uniform gray
        """
        min_hu, max_hu = self.hu_window
        
        if self.clip_values:
            # Clip values outside window
            # This is non-reversible but reduces noise from extreme values
            volume = np.clip(volume, min_hu, max_hu)
        
        if self.normalize:
            # Linear mapping: [min_hu, max_hu] → [0, 1]
            # Formula: normalized = (value - min) / (max - min)
            volume = (volume - min_hu) / (max_hu - min_hu)
            
            # Ensure [0, 1] range (handles edge cases)
            volume = np.clip(volume, 0, 1)
        
        return volume.astype(np.float32)
    
    def pad_or_crop(
        self,
        volume: np.ndarray,
        target_shape: Tuple[int, int, int],
    ) -> np.ndarray:
        """
        Pad or crop volume to exact target shape.
        
        WHY FIXED SHAPES?
        -----------------
        Neural networks (especially fully connected layers) need fixed input size.
        Even for fully convolutional networks, batching requires same shapes.
        
        Options for handling variable sizes:
        1. Pad/crop to fixed size (this method)
        2. Resize (changes physical scale - not ideal)
        3. Process patches instead of full volumes
        4. Use dynamic batching (complex)
        
        We use CENTER crop/pad:
        - Preserves the central anatomy (usually most important)
        - Symmetric padding/cropping
        
        Parameters
        ----------
        volume : np.ndarray
            Input volume
        target_shape : tuple of int
            Desired shape (D, H, W)
            
        Returns
        -------
        np.ndarray
            Volume with exact target shape
            
        Alternative: Random Crop
        ------------------------
        For training, random crops can be used as augmentation.
        See augmentation.py for RandomCrop3D.
        """
        current_shape = volume.shape
        result = np.zeros(target_shape, dtype=volume.dtype)
        
        # Calculate start indices for each dimension
        # Positive = crop (take center), Negative = pad (add zeros)
        slices_src = []
        slices_dst = []
        
        for i in range(3):
            diff = current_shape[i] - target_shape[i]
            
            if diff > 0:
                # Current larger → crop (center crop)
                start_src = diff // 2
                end_src = start_src + target_shape[i]
                slices_src.append(slice(start_src, end_src))
                slices_dst.append(slice(None))  # Full destination
            else:
                # Current smaller → pad (center pad)
                start_dst = (-diff) // 2
                end_dst = start_dst + current_shape[i]
                slices_src.append(slice(None))  # Full source
                slices_dst.append(slice(start_dst, end_dst))
        
        # Copy data
        result[tuple(slices_dst)] = volume[tuple(slices_src)]
        
        return result


def create_preprocessor_from_config(config: dict) -> CTPreprocessor:
    """
    Factory function to create preprocessor from config dict.
    
    Useful for loading configuration from YAML files.
    
    Parameters
    ----------
    config : dict
        Configuration with keys:
        - target_spacing: [z, y, x] in mm
        - hu_window: [min, max]
        - clip_values: bool
        - normalize: bool
        
    Returns
    -------
    CTPreprocessor
        Configured preprocessor instance
        
    Example Config (YAML)
    ---------------------
    preprocessing:
      target_spacing: [1.0, 1.0, 1.0]
      hu_window: [-100, 400]
      clip_values: true
      normalize: true
    """
    return CTPreprocessor(
        target_spacing=tuple(config.get('target_spacing', [1.0, 1.0, 1.0])),
        hu_window=tuple(config.get('hu_window', [-100, 400])),
        clip_values=config.get('clip_values', True),
        normalize=config.get('normalize', True),
    )


# ============================================================================
# COMMON WINDOW PRESETS
# ============================================================================
# These are clinically-validated window settings for different purposes.
# Useful reference for understanding CT visualization.

WINDOW_PRESETS = {
    # Brain windows
    'brain': {'center': 40, 'width': 80},      # Standard brain tissue
    'stroke': {'center': 32, 'width': 8},       # Subtle hemorrhage
    'subdural': {'center': 75, 'width': 215},   # Subdural hematoma
    
    # Vascular (our main interest for aneurysms)
    'cta': {'center': 170, 'width': 600},       # CT Angiography
    'cta_narrow': {'center': 150, 'width': 300}, # Better vessel detail
    
    # Other common windows
    'bone': {'center': 600, 'width': 2800},     # Skull, spine
    'soft_tissue': {'center': 50, 'width': 400}, # General soft tissue
    'lung': {'center': -600, 'width': 1600},    # Lung parenchyma
}


def window_preset_to_range(preset_name: str) -> Tuple[float, float]:
    """
    Convert window preset to (min, max) range.
    
    Window Level (Center) and Window Width are the clinical standard,
    but our code uses (min, max) for clarity.
    
    Conversion:
        min_hu = center - width/2
        max_hu = center + width/2
        
    Parameters
    ----------
    preset_name : str
        Name of preset (e.g., 'cta', 'brain', 'bone')
        
    Returns
    -------
    tuple of float
        (min_hu, max_hu) range
        
    Example
    -------
    >>> window_preset_to_range('cta')
    (-130.0, 470.0)  # center=170, width=600
    """
    if preset_name not in WINDOW_PRESETS:
        available = list(WINDOW_PRESETS.keys())
        raise ValueError(f"Unknown preset '{preset_name}'. Available: {available}")
    
    preset = WINDOW_PRESETS[preset_name]
    center = preset['center']
    width = preset['width']
    
    min_hu = center - width / 2
    max_hu = center + width / 2
    
    return (min_hu, max_hu)
