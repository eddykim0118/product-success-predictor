"""
Tests for data loading and preprocessing.
"""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.preprocessor import CTPreprocessor
from src.data.augmentation import (
    RandomRotation3D,
    RandomFlip3D,
    RandomIntensityShift,
    GaussianNoise,
    Compose,
    get_train_transforms,
    get_val_transforms,
)


class TestCTPreprocessor:
    """Tests for CTPreprocessor."""
    
    def test_init(self):
        """Test preprocessor initialization."""
        preprocessor = CTPreprocessor(
            target_spacing=(1.0, 1.0, 1.0),
            hu_window=(-100, 400),
        )
        assert preprocessor.target_spacing == (1.0, 1.0, 1.0)
        assert preprocessor.hu_window == (-100, 400)
    
    def test_apply_hu_window(self):
        """Test HU windowing."""
        preprocessor = CTPreprocessor(hu_window=(0, 100))
        
        volume = np.array([[-50, 0, 50], [100, 150, 200]], dtype=np.float32)
        result = preprocessor.apply_hu_window(volume)
        
        # Should be normalized to [0, 1]
        assert result.min() >= 0
        assert result.max() <= 1
    
    def test_resample_volume(self):
        """Test volume resampling."""
        preprocessor = CTPreprocessor(target_spacing=(2.0, 2.0, 2.0))
        
        volume = np.random.rand(10, 20, 20).astype(np.float32)
        current_spacing = (1.0, 1.0, 1.0)
        
        resampled, new_spacing = preprocessor.resample_volume(volume, current_spacing)
        
        # Should be roughly half the size
        assert resampled.shape[0] == pytest.approx(5, abs=1)
        assert new_spacing == (2.0, 2.0, 2.0)
    
    def test_pad_or_crop(self):
        """Test padding/cropping."""
        preprocessor = CTPreprocessor()
        
        # Test cropping
        volume = np.random.rand(100, 200, 200).astype(np.float32)
        result = preprocessor.pad_or_crop(volume, (50, 100, 100))
        assert result.shape == (50, 100, 100)
        
        # Test padding
        volume = np.random.rand(20, 50, 50).astype(np.float32)
        result = preprocessor.pad_or_crop(volume, (50, 100, 100))
        assert result.shape == (50, 100, 100)


class TestAugmentations:
    """Tests for augmentation transforms."""
    
    def setup_method(self):
        """Create test data."""
        self.volume = np.random.rand(32, 64, 64).astype(np.float32)
        self.data = {'volume': self.volume, 'mask': None}
    
    def test_random_rotation(self):
        """Test random rotation."""
        transform = RandomRotation3D(angle_range=(-10, 10), prob=1.0)
        result = transform(self.data.copy())
        
        assert result['volume'].shape == self.volume.shape
    
    def test_random_flip(self):
        """Test random flip."""
        transform = RandomFlip3D(axes=(1, 2), prob=1.0)
        result = transform(self.data.copy())
        
        assert result['volume'].shape == self.volume.shape
    
    def test_intensity_shift(self):
        """Test intensity shift."""
        transform = RandomIntensityShift(prob=1.0)
        result = transform(self.data.copy())
        
        assert result['volume'].shape == self.volume.shape
        # Values should be clipped to [0, 1]
        assert result['volume'].min() >= 0
        assert result['volume'].max() <= 1
    
    def test_gaussian_noise(self):
        """Test Gaussian noise."""
        transform = GaussianNoise(std_range=(0.1, 0.1), prob=1.0)
        result = transform(self.data.copy())
        
        assert result['volume'].shape == self.volume.shape
        # Should have added some noise
        assert not np.allclose(result['volume'], self.volume)
    
    def test_compose(self):
        """Test transform composition."""
        transforms = Compose([
            RandomFlip3D(prob=1.0),
            GaussianNoise(prob=1.0),
        ])
        result = transforms(self.data.copy())
        
        assert result['volume'].shape == self.volume.shape
    
    def test_get_train_transforms(self):
        """Test training transform factory."""
        transforms = get_train_transforms(crop_size=(32, 64, 64))
        result = transforms(self.data.copy())
        
        # Should be a tensor
        assert hasattr(result['volume'], 'shape')
    
    def test_get_val_transforms(self):
        """Test validation transform factory."""
        transforms = get_val_transforms(crop_size=(32, 64, 64))
        result = transforms(self.data.copy())
        
        assert hasattr(result['volume'], 'shape')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

