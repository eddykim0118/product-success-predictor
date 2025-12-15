"""
DICOM Data Loader
=================

This module handles loading medical imaging data from DICOM files.

WHAT IS DICOM?
--------------
DICOM (Digital Imaging and Communications in Medicine) is THE standard format
for medical images. Unlike JPEG or PNG which just store pixels, DICOM files
contain:
    1. Pixel data (the actual image)
    2. Metadata (patient info, scan parameters, positioning, etc.)

This metadata is CRITICAL for medical imaging because:
    - Different scanners produce different resolutions
    - Pixel values have physical meaning (Hounsfield Units in CT)
    - 3D reconstruction requires knowing slice spacing/orientation

WHY CAN'T WE JUST USE PNG/JPEG?
-------------------------------
1. Loss of information: DICOM stores 12-16 bit values, JPEG is 8-bit
2. No metadata: We'd lose pixel spacing, slice thickness, etc.
3. No standard orientation: MRI/CT have specific coordinate systems

DICOM STRUCTURE FOR CT SCANS
----------------------------
A typical CT scan is organized as:
    Patient/
    └── Study/          (one scan session)
        └── Series/     (one acquisition, e.g., "Head CTA")
            ├── slice_001.dcm
            ├── slice_002.dcm
            └── ... (100-500 slices typically)

Each .dcm file = one 2D slice. We stack these to create a 3D volume.

KEY DICOM TAGS WE NEED
----------------------
- PixelSpacing: Physical size of each pixel in mm (row_spacing, col_spacing)
- SliceThickness: Distance between slice centers in mm
- ImagePositionPatient: 3D coordinates of the first pixel
- RescaleSlope/Intercept: Convert raw values to Hounsfield Units

INTERVIEW TIP: Be ready to explain why medical images need more than pixels!

Author: Eddy Kim
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# Note: pydicom is the standard library for DICOM in Python
# It handles the complex binary format and tag parsing
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut


class DICOMLoader:
    """
    Loads DICOM CT/MRA series and converts to 3D NumPy arrays.
    
    Design Philosophy
    -----------------
    This class separates CONCERNS:
    - Loading (this class): Handle DICOM complexity, produce clean NumPy arrays
    - Preprocessing (separate): Normalize, resample, window (CTPreprocessor)
    - Augmentation (separate): Random transforms for training
    
    Why this separation?
    1. Single Responsibility: Each class does ONE thing well
    2. Testability: Can test loader without preprocessing logic
    3. Flexibility: Different preprocessing for different tasks
    
    Usage Example
    -------------
    >>> loader = DICOMLoader("data/raw/")
    >>> volume, metadata = loader.load_series("patient001/study/series/")
    >>> print(volume.shape)  # (num_slices, height, width)
    >>> print(metadata['spacing'])  # Physical voxel size in mm
    """
    
    def __init__(self, data_dir: str = "data/raw"):
        """
        Initialize the DICOM loader.
        
        Parameters
        ----------
        data_dir : str
            Root directory containing DICOM files.
            
            Typical structure:
            data_dir/
            ├── patient_001/
            │   └── study_xxx/
            │       └── series_yyy/
            │           ├── 001.dcm
            │           └── ...
            └── patient_002/
                └── ...
        
        Design Decision: We store root path, not load everything at init.
        Why? Medical datasets can be HUGE (100GB+). Lazy loading is essential.
        """
        self.data_dir = Path(data_dir)
        
        # Cache for metadata - speeds up repeated queries
        # Trade-off: Memory vs Speed. For large datasets, might skip caching.
        self._metadata_cache: Optional[pd.DataFrame] = None
    
    def load_dicom_file(self, filepath: Union[str, Path]) -> pydicom.Dataset:
        """
        Load a single DICOM file.
        
        Parameters
        ----------
        filepath : str or Path
            Path to the .dcm file
            
        Returns
        -------
        pydicom.Dataset
            DICOM dataset object containing all tags and pixel data
            
        Why pydicom.dcmread()?
        - Handles DICOM's complex binary format
        - Parses hundreds of standardized tags
        - Manages different transfer syntaxes (compression, byte order)
        
        Interview Note: DICOM can be compressed (JPEG2000, RLE, etc.)
        pydicom handles decompression automatically.
        """
        return pydicom.dcmread(filepath)
    
    def dicom_to_array(
        self, 
        dcm: pydicom.Dataset, 
        apply_lut: bool = True
    ) -> np.ndarray:
        """
        Convert DICOM pixel data to NumPy array with proper values.
        
        This is where we handle the PHYSICS of medical imaging!
        
        Parameters
        ----------
        dcm : pydicom.Dataset
            Loaded DICOM file
        apply_lut : bool
            Whether to apply Value of Interest Look-Up Table.
            Usually True - converts to display-ready values.
            
        Returns
        -------
        np.ndarray
            2D array of Hounsfield Units (for CT) or signal intensity (MRI)
        
        HOUNSFIELD UNITS (HU) - CRITICAL CONCEPT
        ----------------------------------------
        CT scanners measure X-ray attenuation, stored as HU:
        - Water = 0 HU (by definition)
        - Air = -1000 HU
        - Bone = +400 to +1000 HU
        - Soft tissue = +40 to +80 HU
        - Blood vessels (with contrast) = +100 to +400 HU
        
        Raw DICOM values → HU conversion:
            HU = pixel_value * RescaleSlope + RescaleIntercept
        
        Why this matters for aneurysm detection:
        - Aneurysms are in blood vessels (visible with contrast agent)
        - We need to "window" to the right HU range to see them
        - Different HU ranges highlight different tissues
        
        INTERVIEW TIP: Explain HU and why CT values have physical meaning
        unlike arbitrary pixel values in natural images.
        """
        # Step 1: Get raw pixel array
        # DICOM stores pixels in a specific format (often 12 or 16 bit)
        if apply_lut:
            # apply_voi_lut handles display transformations
            # stored in DICOM headers (e.g., Window Center/Width)
            pixels = apply_voi_lut(dcm.pixel_array, dcm)
        else:
            pixels = dcm.pixel_array.astype(np.float32)
        
        # Step 2: Convert to Hounsfield Units
        # These values are stored in DICOM headers
        # Default values handle edge cases (some DICOM files lack these tags)
        slope = float(getattr(dcm, 'RescaleSlope', 1))
        intercept = float(getattr(dcm, 'RescaleIntercept', 0))
        
        # Apply the linear transformation: HU = slope * raw + intercept
        # This is the physics calibration of the CT scanner
        pixels = pixels * slope + intercept
        
        return pixels.astype(np.float32)
    
    def load_series(
        self, 
        series_dir: Union[str, Path],
        sort_by_position: bool = True
    ) -> Tuple[np.ndarray, Dict]:
        """
        Load all DICOM slices in a series directory into a 3D volume.
        
        This is the MAIN METHOD for getting usable data.
        
        Parameters
        ----------
        series_dir : str or Path
            Directory containing DICOM files for ONE series
        sort_by_position : bool
            Sort slices by physical position (almost always True)
            
        Returns
        -------
        volume : np.ndarray
            3D array of shape (num_slices, height, width) in HU
        metadata : dict
            Essential information for preprocessing:
            - patient_id: Unique patient identifier
            - spacing: Voxel dimensions in mm (z, y, x)
            - shape: Volume dimensions
            - etc.
        
        WHY SORTING MATTERS
        -------------------
        DICOM filenames don't guarantee order!
        Example: 001.dcm might be the 50th slice physically.
        
        We sort by ImagePositionPatient[2] (z-coordinate) to ensure
        the volume is anatomically correct (feet-to-head or vice versa).
        
        Getting this wrong = catastrophic errors in 3D analysis!
        
        MEMORY CONSIDERATIONS
        ---------------------
        A typical CT volume:
        - 512 x 512 x 300 slices
        - Float32 = 4 bytes per voxel
        - Total = 512 * 512 * 300 * 4 = ~300 MB per volume
        
        For batch training, we'll need strategies like:
        - Loading patches instead of full volumes
        - Data generators that load on-demand
        - Mixed precision to reduce memory
        """
        series_dir = Path(series_dir)
        
        # Find all DICOM files (handle both .dcm and .DCM extensions)
        dcm_files = list(series_dir.glob("*.dcm")) + list(series_dir.glob("*.DCM"))
        
        if not dcm_files:
            raise ValueError(
                f"No DICOM files found in {series_dir}\n"
                "Expected files with .dcm or .DCM extension."
            )
        
        # Load all slices into memory
        # Note: For very large datasets, consider lazy loading
        slices = []
        for filepath in dcm_files:
            try:
                dcm = self.load_dicom_file(filepath)
                slices.append(dcm)
            except Exception as e:
                # Log but continue - some files might be corrupted
                print(f"Warning: Could not load {filepath}: {e}")
        
        if not slices:
            raise ValueError(f"No valid DICOM files could be loaded from {series_dir}")
        
        # CRITICAL: Sort by physical position
        if sort_by_position:
            # ImagePositionPatient = (x, y, z) coordinates of first voxel
            # We sort by z to stack slices correctly
            try:
                slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
            except AttributeError:
                # Fallback: sort by InstanceNumber if position unavailable
                print("Warning: No ImagePositionPatient, using InstanceNumber")
                slices.sort(key=lambda x: int(getattr(x, 'InstanceNumber', 0)))
        
        # Stack 2D slices into 3D volume
        # Each slice becomes one "layer" in the z-axis
        volume = np.stack(
            [self.dicom_to_array(s) for s in slices], 
            axis=0  # Stack along new first axis → (Z, H, W)
        )
        
        # Extract metadata from first slice (representative of series)
        first_slice = slices[0]
        
        # Calculate slice spacing from positions if available
        if len(slices) > 1 and hasattr(slices[0], 'ImagePositionPatient'):
            # Distance between consecutive slices
            z_positions = [float(s.ImagePositionPatient[2]) for s in slices]
            slice_spacing = abs(z_positions[1] - z_positions[0])
        else:
            # Fallback to SliceThickness tag
            slice_spacing = float(getattr(first_slice, 'SliceThickness', 1.0))
        
        # PixelSpacing is (row_spacing, col_spacing) = (y, x)
        pixel_spacing = [float(x) for x in getattr(first_slice, 'PixelSpacing', [1.0, 1.0])]
        
        metadata = {
            # Patient/Study identifiers
            'patient_id': str(getattr(first_slice, 'PatientID', 'Unknown')),
            'study_uid': str(getattr(first_slice, 'StudyInstanceUID', 'Unknown')),
            'series_uid': str(getattr(first_slice, 'SeriesInstanceUID', 'Unknown')),
            
            # Image properties
            'modality': str(getattr(first_slice, 'Modality', 'Unknown')),
            
            # CRITICAL: Physical dimensions in mm
            # spacing = (z_spacing, y_spacing, x_spacing)
            # This is needed for:
            # 1. Resampling to isotropic voxels
            # 2. Converting pixel distances to real-world mm
            # 3. Proper visualization aspect ratios
            'spacing': (slice_spacing, pixel_spacing[0], pixel_spacing[1]),
            
            # Volume dimensions
            'shape': volume.shape,
            'num_slices': len(slices),
            
            # Image dimensions
            'rows': int(first_slice.Rows),
            'columns': int(first_slice.Columns),
        }
        
        return volume, metadata
    
    def scan_directory(self, verbose: bool = True) -> pd.DataFrame:
        """
        Scan the data directory and create an index of available series.
        
        This is useful for:
        - Getting an overview of the dataset
        - Creating train/val/test splits
        - Filtering by modality, patient, etc.
        
        Parameters
        ----------
        verbose : bool
            Print progress information
            
        Returns
        -------
        pd.DataFrame
            Index with columns:
            - patient_id, study_id, series_id
            - path (to series directory)
            - num_slices, modality, etc.
        
        Performance Note
        ----------------
        This scans ALL directories and reads ONE DICOM file per series.
        For 1000 patients, expect ~30 seconds. Results are cached.
        """
        if self._metadata_cache is not None:
            return self._metadata_cache
        
        records = []
        
        # Walk directory tree looking for DICOM files
        # We assume standard structure: patient/study/series/files.dcm
        for root, dirs, files in os.walk(self.data_dir):
            dcm_files = [f for f in files if f.lower().endswith('.dcm')]
            
            if dcm_files:
                # Found a directory with DICOM files
                series_path = Path(root)
                
                # Read first file to get metadata
                try:
                    sample_dcm = self.load_dicom_file(series_path / dcm_files[0])
                    
                    records.append({
                        'patient_id': str(getattr(sample_dcm, 'PatientID', 'Unknown')),
                        'study_id': str(getattr(sample_dcm, 'StudyInstanceUID', 'Unknown')),
                        'series_id': str(getattr(sample_dcm, 'SeriesInstanceUID', 'Unknown')),
                        'path': str(series_path),
                        'num_slices': len(dcm_files),
                        'modality': str(getattr(sample_dcm, 'Modality', 'Unknown')),
                        'rows': getattr(sample_dcm, 'Rows', None),
                        'columns': getattr(sample_dcm, 'Columns', None),
                    })
                except Exception as e:
                    if verbose:
                        print(f"Warning: Could not read {series_path}: {e}")
        
        self._metadata_cache = pd.DataFrame(records)
        
        if verbose:
            print(f"\nDataset Summary:")
            print(f"  Total series: {len(records)}")
            print(f"  Unique patients: {self._metadata_cache['patient_id'].nunique()}")
            if 'modality' in self._metadata_cache.columns:
                print(f"  Modalities: {self._metadata_cache['modality'].unique().tolist()}")
        
        return self._metadata_cache


class AneurysmDataset:
    """
    PyTorch-compatible dataset for aneurysm detection.
    
    This class bridges our DICOM loading with PyTorch's DataLoader.
    
    Design Pattern: Adapter
    -----------------------
    - DICOMLoader: Knows DICOM format
    - AneurysmDataset: Knows PyTorch interface
    - This separates concerns and allows testing each independently
    
    Why not inherit from torch.utils.data.Dataset?
    We keep this framework-agnostic for now. Easy to add later:
    
    >>> class TorchAneurysmDataset(torch.utils.data.Dataset, AneurysmDataset):
    ...     pass
    
    Usage
    -----
    >>> dataset = AneurysmDataset("data/raw", labels_file="labels.csv")
    >>> sample = dataset[0]
    >>> print(sample['volume'].shape)
    >>> print(sample['label'])  # Multi-label: [aneurysm_present, ICA_L, ...]
    """
    
    def __init__(
        self,
        data_dir: str,
        labels_file: Optional[str] = None,
        transform=None,
    ):
        """
        Initialize the dataset.
        
        Parameters
        ----------
        data_dir : str
            Directory containing DICOM data
        labels_file : str, optional
            Path to CSV with labels. Expected columns:
            - study_id or patient_id (for matching)
            - aneurysm_present (0 or 1)
            - ICA_L, ICA_R, MCA_L, ... (0 or 1 each)
        transform : callable, optional
            Transform to apply to each sample.
            Should accept dict with 'volume' and return transformed dict.
            
        Multi-Label Classification Note
        --------------------------------
        RSNA competition has 14 targets:
        1. aneurysm_present (weighted 13x in evaluation!)
        2-14. Location-specific labels (13 locations)
        
        This is MULTI-LABEL (not multi-class):
        - Multiple labels can be 1 simultaneously
        - A patient might have aneurysms in multiple locations
        - We use sigmoid (not softmax) for prediction
        """
        self.loader = DICOMLoader(data_dir)
        self.transform = transform
        
        # Scan available data
        self.metadata_df = self.loader.scan_directory(verbose=True)
        
        # Load labels if provided
        self.labels_df = None
        if labels_file and os.path.exists(labels_file):
            self.labels_df = pd.read_csv(labels_file)
            print(f"Loaded labels: {len(self.labels_df)} rows")
    
    def __len__(self) -> int:
        """Number of samples in dataset."""
        return len(self.metadata_df)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Get a single sample.
        
        Parameters
        ----------
        idx : int
            Sample index
            
        Returns
        -------
        dict with keys:
            - 'volume': 3D numpy array (or tensor if transform applied)
            - 'metadata': Dict with spacing, patient_id, etc.
            - 'label': numpy array of shape (14,) if labels available
            - 'study_id': For matching predictions to submission
        
        Why return a dict?
        ------------------
        Flexibility! Different training stages need different data:
        - Training: volume + labels
        - Inference: volume + study_id (for submission)
        - Debugging: volume + metadata + visualization info
        
        A dict accommodates all these without changing the interface.
        """
        row = self.metadata_df.iloc[idx]
        
        # Load the 3D volume
        volume, metadata = self.loader.load_series(row['path'])
        
        # Get label if available
        label = None
        if self.labels_df is not None:
            # Match by study_id or patient_id
            match_col = 'study_id' if 'study_id' in self.labels_df.columns else 'patient_id'
            match_val = row.get('study_id', row.get('patient_id'))
            
            label_row = self.labels_df[self.labels_df[match_col] == match_val]
            
            if len(label_row) > 0:
                # Extract the 14 target columns
                target_cols = [
                    'aneurysm_present',
                    'ICA_L', 'ICA_R', 'MCA_L', 'MCA_R',
                    'ACA_L', 'ACA_R', 'PCA_L', 'PCA_R',
                    'PCOM_L', 'PCOM_R', 'BA', 'VA_L', 'VA_R'
                ]
                # Get available columns (some datasets might have subset)
                available_cols = [c for c in target_cols if c in label_row.columns]
                label = label_row[available_cols].values[0].astype(np.float32)
        
        # Build sample dict
        sample = {
            'volume': volume,
            'metadata': metadata,
            'label': label,
            'study_id': row.get('study_id', row.get('patient_id')),
        }
        
        # Apply transforms (preprocessing + augmentation)
        if self.transform is not None:
            sample = self.transform(sample)
        
        return sample


def get_sample_dicom_info(filepath: str) -> Dict:
    """
    Quick utility to inspect a single DICOM file.
    
    Useful for debugging and understanding new datasets.
    
    Parameters
    ----------
    filepath : str
        Path to a .dcm file
        
    Returns
    -------
    dict
        Key DICOM tags and their values
    
    Usage
    -----
    >>> info = get_sample_dicom_info("data/raw/patient001/001.dcm")
    >>> print(info['Modality'])
    >>> print(info['PixelSpacing'])
    """
    dcm = pydicom.dcmread(filepath)
    
    # Tags most relevant for medical image analysis
    important_tags = [
        'PatientID', 'PatientAge', 'PatientSex',
        'Modality', 'Manufacturer', 'ManufacturerModelName',
        'Rows', 'Columns', 'PixelSpacing', 'SliceThickness',
        'ImagePositionPatient', 'ImageOrientationPatient',
        'RescaleSlope', 'RescaleIntercept',
        'WindowCenter', 'WindowWidth',
        'BitsAllocated', 'BitsStored',
    ]
    
    info = {}
    for tag in important_tags:
        value = getattr(dcm, tag, None)
        if value is not None:
            # Convert to Python types for easier handling
            if hasattr(value, 'tolist'):
                value = value.tolist()
            info[tag] = value
    
    return info
