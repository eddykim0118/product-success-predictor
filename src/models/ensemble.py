"""
Ensemble Models
===============

Model ensembling techniques for improved predictions.
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class EnsembleModel(nn.Module):
    """
    Ensemble of multiple models with various aggregation strategies.
    """
    
    def __init__(
        self,
        models: List[nn.Module],
        weights: Optional[List[float]] = None,
        aggregation: str = 'mean',
    ):
        """
        Args:
            models: List of models to ensemble
            weights: Optional weights for weighted averaging
            aggregation: 'mean', 'weighted', 'voting', or 'stacking'
        """
        super().__init__()
        
        self.models = nn.ModuleList(models)
        self.num_models = len(models)
        
        if weights is None:
            weights = [1.0 / self.num_models] * self.num_models
        self.register_buffer('weights', torch.tensor(weights))
        
        self.aggregation = aggregation
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through all models and aggregate.
        
        Args:
            x: Input tensor
            
        Returns:
            Aggregated predictions
        """
        outputs = []
        
        for model in self.models:
            model.eval()
            with torch.no_grad():
                out = model(x)
                outputs.append(out)
        
        outputs = torch.stack(outputs, dim=0)  # (num_models, B, C)
        
        if self.aggregation == 'mean':
            return outputs.mean(dim=0)
        elif self.aggregation == 'weighted':
            weights = self.weights.view(-1, 1, 1)
            return (outputs * weights).sum(dim=0)
        elif self.aggregation == 'voting':
            # Hard voting for classification
            votes = outputs.argmax(dim=-1)  # (num_models, B)
            # Return mode
            result = torch.zeros_like(outputs[0])
            for i in range(outputs.shape[1]):  # batch
                for j in range(outputs.shape[2]):  # class
                    result[i, j] = (votes[:, i] == j).float().sum()
            return result
        else:
            return outputs.mean(dim=0)


class StackingEnsemble(nn.Module):
    """
    Stacking ensemble with a meta-learner.
    """
    
    def __init__(
        self,
        base_models: List[nn.Module],
        num_classes: int = 14,
        meta_hidden: int = 64,
    ):
        """
        Args:
            base_models: List of base models
            num_classes: Number of output classes
            meta_hidden: Hidden dimension of meta-learner
        """
        super().__init__()
        
        self.base_models = nn.ModuleList(base_models)
        self.num_models = len(base_models)
        
        # Freeze base models
        for model in self.base_models:
            for param in model.parameters():
                param.requires_grad = False
        
        # Meta-learner
        input_dim = num_classes * self.num_models
        self.meta_learner = nn.Sequential(
            nn.Linear(input_dim, meta_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(meta_hidden, num_classes),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor
            
        Returns:
            Meta-learner output
        """
        base_outputs = []
        
        for model in self.base_models:
            model.eval()
            with torch.no_grad():
                out = model(x)
                out = torch.sigmoid(out)  # Probabilities
                base_outputs.append(out)
        
        # Concatenate base outputs
        stacked = torch.cat(base_outputs, dim=1)  # (B, num_models * num_classes)
        
        # Meta-learner
        return self.meta_learner(stacked)


class TestTimeAugmentation:
    """
    Test-time augmentation for improved predictions.
    """
    
    def __init__(
        self,
        model: nn.Module,
        transforms: List[str] = ['original', 'flip_x', 'flip_y'],
        aggregation: str = 'mean',
    ):
        """
        Args:
            model: Model to apply TTA to
            transforms: List of transforms to apply
            aggregation: How to aggregate predictions
        """
        self.model = model
        self.transforms = transforms
        self.aggregation = aggregation
    
    def _apply_transform(self, x: torch.Tensor, transform: str) -> torch.Tensor:
        """Apply a transform to input."""
        if transform == 'original':
            return x
        elif transform == 'flip_x':
            return torch.flip(x, dims=[-1])
        elif transform == 'flip_y':
            return torch.flip(x, dims=[-2])
        elif transform == 'flip_z':
            return torch.flip(x, dims=[-3])
        elif transform == 'flip_xy':
            return torch.flip(x, dims=[-1, -2])
        else:
            return x
    
    @torch.no_grad()
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply TTA and aggregate predictions.
        
        Args:
            x: Input tensor
            
        Returns:
            Aggregated predictions
        """
        self.model.eval()
        predictions = []
        
        for transform in self.transforms:
            x_aug = self._apply_transform(x, transform)
            pred = self.model(x_aug)
            predictions.append(pred)
        
        predictions = torch.stack(predictions, dim=0)
        
        if self.aggregation == 'mean':
            return predictions.mean(dim=0)
        elif self.aggregation == 'max':
            return predictions.max(dim=0)[0]
        else:
            return predictions.mean(dim=0)


def load_ensemble_from_checkpoints(
    model_class: type,
    checkpoint_paths: List[str],
    model_kwargs: Dict = None,
    device: str = 'cuda',
) -> EnsembleModel:
    """
    Load ensemble from multiple checkpoints.
    
    Args:
        model_class: Model class to instantiate
        checkpoint_paths: Paths to checkpoints
        model_kwargs: Kwargs for model initialization
        device: Device to load models to
        
    Returns:
        EnsembleModel
    """
    if model_kwargs is None:
        model_kwargs = {}
    
    models = []
    
    for path in checkpoint_paths:
        model = model_class(**model_kwargs)
        checkpoint = torch.load(path, map_location=device)
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.to(device)
        model.eval()
        models.append(model)
    
    return EnsembleModel(models)


class MultiScaleEnsemble(nn.Module):
    """
    Ensemble predictions from multiple input scales.
    """
    
    def __init__(
        self,
        model: nn.Module,
        scales: List[float] = [0.8, 1.0, 1.2],
    ):
        """
        Args:
            model: Model to use
            scales: Scale factors to apply
        """
        super().__init__()
        self.model = model
        self.scales = scales
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (B, C, D, H, W)
            
        Returns:
            Aggregated predictions
        """
        predictions = []
        original_size = x.shape[2:]
        
        for scale in self.scales:
            if scale == 1.0:
                x_scaled = x
            else:
                new_size = tuple(int(s * scale) for s in original_size)
                x_scaled = F.interpolate(
                    x, size=new_size, mode='trilinear', align_corners=False
                )
            
            pred = self.model(x_scaled)
            predictions.append(pred)
        
        return torch.stack(predictions, dim=0).mean(dim=0)

