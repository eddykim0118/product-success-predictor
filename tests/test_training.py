"""
Tests for training utilities.
"""

import pytest
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.losses import (
    FocalLoss,
    AsymmetricLoss,
    DiceLoss,
    DiceBCELoss,
    WeightedMultiLabelLoss,
    get_loss_function,
)
from src.training.metrics import (
    dice_score,
    iou_score,
    sensitivity,
    specificity,
    compute_detection_metrics,
)


class TestLosses:
    """Tests for loss functions."""
    
    def setup_method(self):
        """Create test tensors."""
        self.pred = torch.randn(4, 14)  # logits
        self.target = torch.randint(0, 2, (4, 14)).float()
    
    def test_focal_loss(self):
        """Test Focal Loss."""
        loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
        loss = loss_fn(self.pred, self.target)
        
        assert loss.dim() == 0  # Scalar
        assert loss.item() >= 0
    
    def test_asymmetric_loss(self):
        """Test Asymmetric Loss."""
        loss_fn = AsymmetricLoss(gamma_neg=4.0, gamma_pos=1.0)
        loss = loss_fn(self.pred, self.target)
        
        assert loss.dim() == 0
        assert loss.item() >= 0
    
    def test_dice_loss(self):
        """Test Dice Loss."""
        loss_fn = DiceLoss()
        
        pred = torch.randn(2, 1, 16, 32, 32)
        target = torch.randint(0, 2, (2, 1, 16, 32, 32)).float()
        
        loss = loss_fn(pred, target)
        
        assert loss.dim() == 0
        assert 0 <= loss.item() <= 1
    
    def test_dice_bce_loss(self):
        """Test combined Dice + BCE Loss."""
        loss_fn = DiceBCELoss(dice_weight=0.5)
        
        pred = torch.randn(2, 1, 16, 32, 32)
        target = torch.randint(0, 2, (2, 1, 16, 32, 32)).float()
        
        loss = loss_fn(pred, target)
        
        assert loss.dim() == 0
        assert loss.item() >= 0
    
    def test_weighted_multilabel_loss(self):
        """Test weighted multi-label loss."""
        loss_fn = WeightedMultiLabelLoss(
            num_classes=14,
            aneurysm_present_idx=0,
            aneurysm_weight=13.0,
        )
        
        loss = loss_fn(self.pred, self.target)
        
        assert loss.dim() == 0
        assert loss.item() >= 0
    
    def test_get_loss_function(self):
        """Test loss function factory."""
        # Test different loss types
        for loss_type in ['bce', 'focal', 'asymmetric']:
            loss_fn = get_loss_function(loss_type)
            loss = loss_fn(self.pred, self.target)
            assert loss.dim() == 0


class TestMetrics:
    """Tests for evaluation metrics."""
    
    def setup_method(self):
        """Create test tensors."""
        self.pred = torch.randn(4, 1, 16, 32, 32)
        self.target = torch.randint(0, 2, (4, 1, 16, 32, 32)).float()
    
    def test_dice_score(self):
        """Test Dice score."""
        score = dice_score(self.pred, self.target)
        
        assert score.dim() == 0
        assert 0 <= score.item() <= 1
    
    def test_iou_score(self):
        """Test IoU score."""
        score = iou_score(self.pred, self.target)
        
        assert score.dim() == 0
        assert 0 <= score.item() <= 1
    
    def test_sensitivity(self):
        """Test sensitivity."""
        score = sensitivity(self.pred, self.target)
        
        assert score.dim() == 0
        assert 0 <= score.item() <= 1
    
    def test_specificity(self):
        """Test specificity."""
        score = specificity(self.pred, self.target)
        
        assert score.dim() == 0
        assert 0 <= score.item() <= 1
    
    def test_compute_detection_metrics(self):
        """Test detection metrics computation."""
        pred_probs = np.random.rand(100, 14)
        targets = np.random.randint(0, 2, (100, 14))
        
        metrics = compute_detection_metrics(pred_probs, targets)
        
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1' in metrics
        assert 'auc' in metrics
        
        for value in metrics.values():
            assert 0 <= value <= 1


class TestTrainer:
    """Tests for trainer functionality."""
    
    def test_trainer_init(self):
        """Test trainer initialization."""
        from src.training.trainer import Trainer
        from src.models.resnet3d import resnet10_3d
        
        model = resnet10_3d(num_classes=14)
        optimizer = torch.optim.Adam(model.parameters())
        criterion = nn.BCEWithLogitsLoss()
        
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            device='cpu',
        )
        
        assert trainer.model is not None
        assert trainer.optimizer is not None
        assert trainer.criterion is not None
    
    def test_save_load_checkpoint(self, tmp_path):
        """Test checkpoint save/load."""
        from src.training.trainer import Trainer
        from src.models.resnet3d import resnet10_3d
        
        model = resnet10_3d(num_classes=14)
        optimizer = torch.optim.Adam(model.parameters())
        criterion = nn.BCEWithLogitsLoss()
        
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            device='cpu',
            checkpoint_dir=str(tmp_path),
        )
        
        # Save
        trainer.save_checkpoint('test_checkpoint.pth')
        assert (tmp_path / 'test_checkpoint.pth').exists()
        
        # Load
        trainer.load_checkpoint('test_checkpoint.pth')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

