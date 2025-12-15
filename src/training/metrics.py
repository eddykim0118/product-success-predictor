"""
Evaluation Metrics
==================

Metrics for evaluating aneurysm detection and segmentation models.
"""

from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F


def dice_score(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    smooth: float = 1e-6,
) -> torch.Tensor:
    """
    Compute Dice coefficient (F1 score for segmentation).
    
    Args:
        pred: Predicted probabilities (B, C, D, H, W) or (B, C, H, W)
        target: Ground truth binary mask
        threshold: Threshold for binary prediction
        smooth: Smoothing factor to avoid division by zero
        
    Returns:
        Dice score
    """
    pred = (torch.sigmoid(pred) > threshold).float()
    
    # Flatten spatial dimensions
    pred_flat = pred.view(pred.size(0), -1)
    target_flat = target.view(target.size(0), -1)
    
    intersection = (pred_flat * target_flat).sum(dim=1)
    union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)
    
    dice = (2.0 * intersection + smooth) / (union + smooth)
    
    return dice.mean()


def iou_score(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    smooth: float = 1e-6,
) -> torch.Tensor:
    """
    Compute Intersection over Union (Jaccard index).
    
    Args:
        pred: Predicted probabilities
        target: Ground truth binary mask
        threshold: Threshold for binary prediction
        smooth: Smoothing factor
        
    Returns:
        IoU score
    """
    pred = (torch.sigmoid(pred) > threshold).float()
    
    pred_flat = pred.view(pred.size(0), -1)
    target_flat = target.view(target.size(0), -1)
    
    intersection = (pred_flat * target_flat).sum(dim=1)
    union = pred_flat.sum(dim=1) + target_flat.sum(dim=1) - intersection
    
    iou = (intersection + smooth) / (union + smooth)
    
    return iou.mean()


def sensitivity(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    smooth: float = 1e-6,
) -> torch.Tensor:
    """
    Compute sensitivity (recall, true positive rate).
    
    Args:
        pred: Predicted probabilities
        target: Ground truth
        threshold: Threshold for binary prediction
        smooth: Smoothing factor
        
    Returns:
        Sensitivity score
    """
    pred = (torch.sigmoid(pred) > threshold).float()
    
    pred_flat = pred.view(pred.size(0), -1)
    target_flat = target.view(target.size(0), -1)
    
    true_positives = (pred_flat * target_flat).sum(dim=1)
    actual_positives = target_flat.sum(dim=1)
    
    sens = (true_positives + smooth) / (actual_positives + smooth)
    
    return sens.mean()


def specificity(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    smooth: float = 1e-6,
) -> torch.Tensor:
    """
    Compute specificity (true negative rate).
    
    Args:
        pred: Predicted probabilities
        target: Ground truth
        threshold: Threshold for binary prediction
        smooth: Smoothing factor
        
    Returns:
        Specificity score
    """
    pred = (torch.sigmoid(pred) > threshold).float()
    
    pred_flat = pred.view(pred.size(0), -1)
    target_flat = target.view(target.size(0), -1)
    
    true_negatives = ((1 - pred_flat) * (1 - target_flat)).sum(dim=1)
    actual_negatives = (1 - target_flat).sum(dim=1)
    
    spec = (true_negatives + smooth) / (actual_negatives + smooth)
    
    return spec.mean()


def compute_detection_metrics(
    pred_probs: Union[torch.Tensor, np.ndarray],
    targets: Union[torch.Tensor, np.ndarray],
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute classification metrics for detection task.
    
    Args:
        pred_probs: Predicted probabilities (N,) or (N, 2)
        targets: Ground truth labels (N,)
        threshold: Classification threshold
        
    Returns:
        Dict with accuracy, precision, recall, f1, auc
    """
    if isinstance(pred_probs, torch.Tensor):
        pred_probs = pred_probs.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()
    
    # Handle multi-class output
    if pred_probs.ndim == 2:
        pred_probs = pred_probs[:, 1]  # Take positive class
    
    # Binary predictions
    preds = (pred_probs > threshold).astype(int)
    
    # Basic metrics
    tp = ((preds == 1) & (targets == 1)).sum()
    tn = ((preds == 0) & (targets == 0)).sum()
    fp = ((preds == 1) & (targets == 0)).sum()
    fn = ((preds == 0) & (targets == 1)).sum()
    
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-6)
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    
    # AUC
    try:
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(targets, pred_probs)
    except:
        auc = 0.0
    
    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'auc': float(auc),
    }


class DiceLoss(torch.nn.Module):
    """
    Dice loss for segmentation.
    """
    
    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = torch.sigmoid(pred)
        
        pred_flat = pred.view(pred.size(0), -1)
        target_flat = target.view(target.size(0), -1)
        
        intersection = (pred_flat * target_flat).sum(dim=1)
        union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        
        return 1 - dice.mean()


class DiceBCELoss(torch.nn.Module):
    """
    Combined Dice and Binary Cross Entropy loss.
    """
    
    def __init__(self, dice_weight: float = 0.5, smooth: float = 1e-6):
        super().__init__()
        self.dice_weight = dice_weight
        self.dice_loss = DiceLoss(smooth)
        self.bce_loss = torch.nn.BCEWithLogitsLoss()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        dice = self.dice_loss(pred, target)
        bce = self.bce_loss(pred, target)
        
        return self.dice_weight * dice + (1 - self.dice_weight) * bce


class FocalLoss(torch.nn.Module):
    """
    Focal loss for handling class imbalance.
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        
        pred_prob = torch.sigmoid(pred)
        p_t = pred_prob * target + (1 - pred_prob) * (1 - target)
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        focal_loss = focal_weight * bce
        
        return focal_loss.mean()

