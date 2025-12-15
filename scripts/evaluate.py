#!/usr/bin/env python
"""
Evaluation Script
=================

Evaluate trained aneurysm detection models.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import Conv3DClassifier, UNet3D
from src.training.metrics import (
    dice_score,
    iou_score,
    sensitivity,
    specificity,
    compute_detection_metrics,
)


def load_model(checkpoint_path: str, model_type: str = 'conv3d', device: str = 'cpu') -> nn.Module:
    """
    Load model from checkpoint.
    
    Args:
        checkpoint_path: Path to model checkpoint
        model_type: Type of model ('conv3d' or 'unet3d')
        device: Device to load model on
        
    Returns:
        Loaded model
    """
    if model_type == 'conv3d':
        model = Conv3DClassifier()
    elif model_type == 'unet3d':
        model = UNet3D()
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    return model


@torch.no_grad()
def evaluate_classification(
    model: nn.Module,
    dataloader: DataLoader,
    device: str = 'cpu',
) -> Dict[str, float]:
    """
    Evaluate classification model.
    
    Args:
        model: Trained model
        dataloader: Test data loader
        device: Device
        
    Returns:
        Dict with evaluation metrics
    """
    model.eval()
    
    all_preds = []
    all_targets = []
    
    for batch in tqdm(dataloader, desc="Evaluating"):
        inputs = batch['volume'].to(device)
        targets = batch['label']
        
        outputs = model(inputs)
        probs = torch.softmax(outputs, dim=1)
        
        all_preds.append(probs.cpu().numpy())
        all_targets.append(targets.numpy())
    
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    metrics = compute_detection_metrics(all_preds, all_targets)
    
    return metrics


@torch.no_grad()
def evaluate_segmentation(
    model: nn.Module,
    dataloader: DataLoader,
    device: str = 'cpu',
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Evaluate segmentation model.
    
    Args:
        model: Trained model
        dataloader: Test data loader
        device: Device
        threshold: Segmentation threshold
        
    Returns:
        Dict with evaluation metrics
    """
    model.eval()
    
    dice_scores = []
    iou_scores = []
    sens_scores = []
    spec_scores = []
    
    for batch in tqdm(dataloader, desc="Evaluating"):
        inputs = batch['volume'].to(device)
        targets = batch['mask'].to(device)
        
        outputs = model(inputs)
        
        # Compute metrics
        dice = dice_score(outputs, targets, threshold)
        iou = iou_score(outputs, targets, threshold)
        sens = sensitivity(outputs, targets, threshold)
        spec = specificity(outputs, targets, threshold)
        
        dice_scores.append(dice.item())
        iou_scores.append(iou.item())
        sens_scores.append(sens.item())
        spec_scores.append(spec.item())
    
    metrics = {
        'dice': np.mean(dice_scores),
        'dice_std': np.std(dice_scores),
        'iou': np.mean(iou_scores),
        'iou_std': np.std(iou_scores),
        'sensitivity': np.mean(sens_scores),
        'specificity': np.mean(spec_scores),
    }
    
    return metrics


def print_metrics(metrics: Dict[str, float], title: str = "Evaluation Results"):
    """Print metrics in a formatted way."""
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)
    
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Evaluate aneurysm detection model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="conv3d",
        choices=["conv3d", "unet3d"],
        help="Type of model"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/processed",
        help="Path to test data"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Path to save results (JSON)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to evaluate on"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for evaluation"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold for segmentation/classification"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("RSNA Intracranial Aneurysm Detection - Evaluation")
    print("=" * 60)
    print(f"\nCheckpoint: {args.checkpoint}")
    print(f"Model type: {args.model_type}")
    print(f"Device: {args.device}")
    
    # Check checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"\nError: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)
    
    # Load model
    print("\nLoading model...")
    model = load_model(args.checkpoint, args.model_type, args.device)
    print(f"Model loaded successfully!")
    
    # Placeholder for data loading
    print("\n" + "=" * 60)
    print("Data loading placeholder")
    print("=" * 60)
    print("\nTo evaluate with real data:")
    print("1. Prepare test dataset")
    print("2. Update data loading code in this script")
    print("\nExample evaluation:")
    print("""
    # Load test dataset
    # test_dataset = AneurysmDataset(args.data_dir, split='test')
    # test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    
    # Evaluate
    # if args.model_type == 'conv3d':
    #     metrics = evaluate_classification(model, test_loader, args.device)
    # else:
    #     metrics = evaluate_segmentation(model, test_loader, args.device, args.threshold)
    
    # print_metrics(metrics)
    """)
    
    # Save results
    if args.output_file:
        import json
        results = {
            'checkpoint': args.checkpoint,
            'model_type': args.model_type,
            'threshold': args.threshold,
            'metrics': {},  # Add actual metrics here
        }
        
        with open(args.output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output_file}")
    
    print("\n" + "=" * 60)
    print("Evaluation script ready!")
    print("=" * 60)


if __name__ == "__main__":
    main()

