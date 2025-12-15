"""
Tests for model architectures.
"""

import pytest
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.resnet3d import (
    ResNet3D,
    BasicBlock,
    Bottleneck,
    resnet10_3d,
    resnet18_3d,
    resnet34_3d,
    resnet50_3d,
)
from src.models.unet3d import UNet3D, AttentionUNet3D
from src.models.ensemble import EnsembleModel, TestTimeAugmentation


class TestResNet3D:
    """Tests for 3D ResNet."""
    
    def test_basic_block(self):
        """Test BasicBlock."""
        block = BasicBlock(64, 64)
        x = torch.randn(2, 64, 8, 16, 16)
        out = block(x)
        assert out.shape == x.shape
    
    def test_bottleneck_block(self):
        """Test Bottleneck block."""
        block = Bottleneck(64, 16)
        x = torch.randn(2, 64, 8, 16, 16)
        out = block(x)
        assert out.shape == (2, 64, 8, 16, 16)
    
    def test_resnet10_3d(self):
        """Test ResNet-10 3D."""
        model = resnet10_3d(in_channels=1, num_classes=14)
        x = torch.randn(2, 1, 32, 64, 64)
        out = model(x)
        assert out.shape == (2, 14)
    
    def test_resnet18_3d(self):
        """Test ResNet-18 3D."""
        model = resnet18_3d(in_channels=1, num_classes=14)
        x = torch.randn(2, 1, 32, 64, 64)
        out = model(x)
        assert out.shape == (2, 14)
    
    def test_resnet50_3d(self):
        """Test ResNet-50 3D."""
        model = resnet50_3d(in_channels=1, num_classes=14)
        x = torch.randn(1, 1, 32, 64, 64)
        out = model(x)
        assert out.shape == (1, 14)
    
    def test_forward_features(self):
        """Test feature extraction."""
        model = resnet18_3d(num_classes=14)
        x = torch.randn(2, 1, 32, 64, 64)
        features = model.forward_features(x)
        
        # Should be (B, C, D', H', W')
        assert len(features.shape) == 5
        assert features.shape[0] == 2


class TestUNet3D:
    """Tests for 3D U-Net."""
    
    def test_unet3d_basic(self):
        """Test basic U-Net forward pass."""
        model = UNet3D(in_channels=1, out_channels=1, base_filters=16)
        x = torch.randn(1, 1, 32, 64, 64)
        out = model(x)
        
        assert out.shape == x.shape
    
    def test_unet3d_multiclass(self):
        """Test U-Net with multiple output channels."""
        model = UNet3D(in_channels=1, out_channels=3, base_filters=16)
        x = torch.randn(1, 1, 32, 64, 64)
        out = model(x)
        
        assert out.shape == (1, 3, 32, 64, 64)
    
    def test_unet3d_deep_supervision(self):
        """Test U-Net with deep supervision."""
        model = UNet3D(
            in_channels=1,
            out_channels=1,
            base_filters=16,
            deep_supervision=True,
        )
        model.train()
        x = torch.randn(1, 1, 32, 64, 64)
        out = model(x)
        
        # Should return tuple (main_output, [ds_outputs])
        assert isinstance(out, tuple)
        assert len(out) == 2
    
    def test_attention_unet3d(self):
        """Test Attention U-Net."""
        model = AttentionUNet3D(in_channels=1, out_channels=1, base_filters=16)
        x = torch.randn(1, 1, 32, 64, 64)
        out = model(x)
        
        assert out.shape == x.shape
    
    def test_unet3d_predict(self):
        """Test U-Net prediction method."""
        model = UNet3D(in_channels=1, out_channels=1, base_filters=16)
        x = torch.randn(1, 1, 32, 64, 64)
        
        pred = model.predict(x, threshold=0.5)
        
        # Should be binary
        assert torch.all((pred == 0) | (pred == 1))


class TestEnsemble:
    """Tests for ensemble models."""
    
    def test_ensemble_model(self):
        """Test basic ensemble."""
        models = [
            resnet10_3d(num_classes=14),
            resnet10_3d(num_classes=14),
        ]
        ensemble = EnsembleModel(models, aggregation='mean')
        
        x = torch.randn(2, 1, 32, 64, 64)
        out = ensemble(x)
        
        assert out.shape == (2, 14)
    
    def test_ensemble_weighted(self):
        """Test weighted ensemble."""
        models = [
            resnet10_3d(num_classes=14),
            resnet10_3d(num_classes=14),
        ]
        weights = [0.7, 0.3]
        ensemble = EnsembleModel(models, weights=weights, aggregation='weighted')
        
        x = torch.randn(2, 1, 32, 64, 64)
        out = ensemble(x)
        
        assert out.shape == (2, 14)
    
    def test_tta(self):
        """Test test-time augmentation."""
        model = resnet10_3d(num_classes=14)
        tta = TestTimeAugmentation(
            model,
            transforms=['original', 'flip_x'],
            aggregation='mean',
        )
        
        x = torch.randn(2, 1, 32, 64, 64)
        out = tta(x)
        
        assert out.shape == (2, 14)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

