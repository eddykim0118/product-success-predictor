"""
RSNA Intracranial Aneurysm Detection
====================================

A deep learning project for detecting intracranial aneurysms from CT Angiography images.

Modules:
    - data: DICOM loading and preprocessing
    - models: 3D CNN and U-Net architectures
    - training: Training utilities and metrics
    - utils: Visualization and DICOM utilities
"""

__version__ = "0.1.0"
__author__ = "Eddy Kim"

from . import data
from . import models
from . import training
from . import utils

