"""
Data Loading and Preprocessing Pipeline
========================================

This subpackage handles everything related to getting medical imaging data
ready for neural networks.

Module Overview
---------------
- loader.py: DICOM file reading and 3D volume construction
- preprocessor.py: CT-specific preprocessing (resampling, windowing, normalization)
- augmentation.py: 3D data augmentation for training

Pipeline Flow
-------------
1. DICOM files → DICOMLoader → 3D NumPy array in Hounsfield Units
2. Raw HU volume → CTPreprocessor → Normalized, resampled volume
3. Clean volume → Augmentation transforms → Augmented training sample
4. Augmented sample → ToTensor → PyTorch tensor ready for model

Why This Separation?
--------------------
Each step has a distinct responsibility:
- Loading: Handle file format complexity (DICOM is messy!)
- Preprocessing: Make data consistent across different scanners/protocols
- Augmentation: Increase training data diversity to prevent overfitting

This separation makes the code:
- Testable: Each component can be tested independently
- Maintainable: Changes to augmentation don't affect loading
- Reusable: Same loader works for different preprocessing strategies

Quick Start
-----------
>>> from src.data import DICOMLoader, CTPreprocessor, get_train_transforms
>>>
>>> # Load a DICOM series
>>> loader = DICOMLoader("data/raw")
>>> volume, metadata = loader.load_series("path/to/series/")
>>>
>>> # Preprocess
>>> preprocessor = CTPreprocessor()
>>> processed = preprocessor(volume, metadata['spacing'])
>>>
>>> # For training with augmentation
>>> transforms = get_train_transforms(crop_size=(64, 128, 128))
>>> sample = transforms({'volume': processed['volume']})
"""

# Import main classes for convenient access
from .loader import DICOMLoader, AneurysmDataset, get_sample_dicom_info
from .preprocessor import (
    CTPreprocessor,
    create_preprocessor_from_config,
    WINDOW_PRESETS,
    window_preset_to_range,
)
from .augmentation import (
    # Core transform classes
    Compose,
    RandomRotation3D,
    RandomFlip3D,
    RandomIntensityShift,
    GaussianNoise,
    GaussianBlur3D,
    RandomCrop3D,
    CenterCrop3D,
    Resize3D,
    ElasticDeformation3D,
    Normalize,
    ToTensor,
    # Convenience functions
    get_train_transforms,
    get_val_transforms,
)

__all__ = [
    # Data loading
    'DICOMLoader',
    'AneurysmDataset',
    'get_sample_dicom_info',
    
    # Preprocessing
    'CTPreprocessor',
    'create_preprocessor_from_config',
    'WINDOW_PRESETS',
    'window_preset_to_range',
    
    # Augmentation transforms
    'Compose',
    'RandomRotation3D',
    'RandomFlip3D',
    'RandomIntensityShift',
    'GaussianNoise',
    'GaussianBlur3D',
    'RandomCrop3D',
    'CenterCrop3D',
    'Resize3D',
    'ElasticDeformation3D',
    'Normalize',
    'ToTensor',
    
    # Transform factories
    'get_train_transforms',
    'get_val_transforms',
]
