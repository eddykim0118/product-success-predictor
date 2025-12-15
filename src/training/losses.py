"""
Custom Loss Functions
=====================

Loss functions for aneurysm detection task.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedBCEWithLogitsLoss(nn.Module):
    """
    Binary Cross Entropy with class weights.
    """
    
    def __init__(
        self,
        pos_weight: Optional[torch.Tensor] = None,
        reduction: str = 'mean',
    ):
        """
        Args:
            pos_weight: Weight for positive class per target
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.pos_weight = pos_weight
        self.reduction = reduction
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pred: Predictions (B, C) - logits
            target: Targets (B, C) - binary
            
        Returns:
            Loss value
        """
        return F.binary_cross_entropy_with_logits(
            pred, target,
            pos_weight=self.pos_weight,
            reduction=self.reduction
        )


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = 'mean',
    ):
        """
        Args:
            alpha: Weighting factor
            gamma: Focusing parameter
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pred: Predictions (B, C) - logits
            target: Targets (B, C) - binary
        """
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        
        pred_prob = torch.sigmoid(pred)
        p_t = pred_prob * target + (1 - pred_prob) * (1 - target)
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        focal_loss = focal_weight * bce
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss for multi-label classification.
    
    From: "Asymmetric Loss For Multi-Label Classification"
    """
    
    def __init__(
        self,
        gamma_neg: float = 4.0,
        gamma_pos: float = 1.0,
        clip: float = 0.05,
        reduction: str = 'mean',
    ):
        """
        Args:
            gamma_neg: Focusing parameter for negative samples
            gamma_pos: Focusing parameter for positive samples
            clip: Probability margin for hard thresholding
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.reduction = reduction
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pred: Predictions (B, C) - logits
            target: Targets (B, C) - binary
        """
        # Probabilities
        pred_prob = torch.sigmoid(pred)
        pred_prob_pos = pred_prob
        pred_prob_neg = 1 - pred_prob
        
        # Asymmetric clipping
        if self.clip > 0:
            pred_prob_neg = (pred_prob_neg + self.clip).clamp(max=1)
        
        # Losses
        loss_pos = target * torch.log(pred_prob_pos.clamp(min=1e-8))
        loss_neg = (1 - target) * torch.log(pred_prob_neg.clamp(min=1e-8))
        
        # Focal weights
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            pt_pos = pred_prob_pos * target
            pt_neg = pred_prob_neg * (1 - target)
            
            focal_weight_pos = (1 - pt_pos) ** self.gamma_pos
            focal_weight_neg = (1 - pt_neg) ** self.gamma_neg
            
            loss_pos = loss_pos * focal_weight_pos
            loss_neg = loss_neg * focal_weight_neg
        
        loss = -(loss_pos + loss_neg)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class DiceLoss(nn.Module):
    """
    Dice Loss for segmentation.
    """
    
    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pred: Predictions - logits
            target: Target binary mask
        """
        pred = torch.sigmoid(pred)
        
        pred_flat = pred.view(pred.size(0), -1)
        target_flat = target.view(target.size(0), -1)
        
        intersection = (pred_flat * target_flat).sum(dim=1)
        union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        
        return 1 - dice.mean()


class DiceBCELoss(nn.Module):
    """
    Combined Dice and BCE loss.
    """
    
    def __init__(
        self,
        dice_weight: float = 0.5,
        smooth: float = 1e-6,
    ):
        super().__init__()
        self.dice_weight = dice_weight
        self.dice_loss = DiceLoss(smooth)
        self.bce_loss = nn.BCEWithLogitsLoss()
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        dice = self.dice_loss(pred, target)
        bce = self.bce_loss(pred, target)
        
        return self.dice_weight * dice + (1 - self.dice_weight) * bce


class WeightedMultiLabelLoss(nn.Module):
    """
    Weighted multi-label loss for RSNA competition.
    
    The competition uses weighted AUC where Aneurysm Present has weight 13
    and other labels have weight 1.
    """
    
    def __init__(
        self,
        label_weights: Optional[torch.Tensor] = None,
        num_classes: int = 14,
        aneurysm_present_idx: int = 0,
        aneurysm_weight: float = 13.0,
    ):
        """
        Args:
            label_weights: Custom weights per label
            num_classes: Number of target classes
            aneurysm_present_idx: Index of Aneurysm Present label
            aneurysm_weight: Weight for Aneurysm Present (default: 13)
        """
        super().__init__()
        
        if label_weights is None:
            # Competition weighting
            label_weights = torch.ones(num_classes)
            label_weights[aneurysm_present_idx] = aneurysm_weight
            # Normalize so mean weight = 1
            label_weights = label_weights / label_weights.mean()
        
        self.register_buffer('label_weights', label_weights)
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pred: Predictions (B, C) - logits
            target: Targets (B, C) - binary
        """
        loss = self.bce(pred, target)  # (B, C)
        weighted_loss = loss * self.label_weights
        return weighted_loss.mean()


class LabelSmoothingLoss(nn.Module):
    """
    Label smoothing for multi-label classification.
    """
    
    def __init__(
        self,
        smoothing: float = 0.1,
    ):
        """
        Args:
            smoothing: Smoothing factor (0 = no smoothing)
        """
        super().__init__()
        self.smoothing = smoothing
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pred: Predictions (B, C) - logits
            target: Targets (B, C) - binary
        """
        # Smooth targets
        with torch.no_grad():
            smoothed = target * (1 - self.smoothing) + self.smoothing / 2
        
        return F.binary_cross_entropy_with_logits(pred, smoothed)


def get_loss_function(
    loss_type: str = 'bce',
    **kwargs,
) -> nn.Module:
    """
    Get loss function by name.
    
    Args:
        loss_type: Loss function name
        **kwargs: Additional arguments for loss function
        
    Returns:
        Loss function module
    """
    loss_map = {
        'bce': nn.BCEWithLogitsLoss,
        'weighted_bce': WeightedBCEWithLogitsLoss,
        'focal': FocalLoss,
        'asymmetric': AsymmetricLoss,
        'dice': DiceLoss,
        'dice_bce': DiceBCELoss,
        'weighted_multilabel': WeightedMultiLabelLoss,
        'label_smoothing': LabelSmoothingLoss,
    }
    
    if loss_type not in loss_map:
        raise ValueError(f"Unknown loss type: {loss_type}")
    
    return loss_map[loss_type](**kwargs)

