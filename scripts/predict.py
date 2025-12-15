#!/usr/bin/env python
"""
Prediction Script
=================

Run inference on new data.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import resnet18_3d, UNet3D
from src.models.ensemble import EnsembleModel, TestTimeAugmentation


def load_model(
    checkpoint_path: str,
    model_type: str = 'resnet18',
    num_classes: int = 14,
    device: str = 'cuda',
) -> torch.nn.Module:
    """
    Load model from checkpoint.
    """
    from src.models.resnet3d import resnet18_3d, resnet34_3d, resnet50_3d
    
    model_map = {
        'resnet18': resnet18_3d,
        'resnet34': resnet34_3d,
        'resnet50': resnet50_3d,
    }
    
    if model_type not in model_map:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model = model_map[model_type](num_classes=num_classes)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    return model


@torch.no_grad()
def predict_batch(
    model: torch.nn.Module,
    batch: Dict,
    device: str = 'cuda',
    use_tta: bool = False,
) -> np.ndarray:
    """
    Run prediction on a batch.
    
    Args:
        model: Model to use
        batch: Batch dict with 'volume'
        device: Device
        use_tta: Use test-time augmentation
        
    Returns:
        Predictions array
    """
    volume = batch['volume'].to(device)
    
    if use_tta:
        tta = TestTimeAugmentation(
            model,
            transforms=['original', 'flip_x', 'flip_y'],
        )
        logits = tta(volume)
    else:
        logits = model(volume)
    
    probs = torch.sigmoid(logits)
    return probs.cpu().numpy()


def predict_dataset(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: str = 'cuda',
    use_tta: bool = False,
) -> pd.DataFrame:
    """
    Run prediction on entire dataset.
    
    Args:
        model: Model
        dataloader: Data loader
        device: Device
        use_tta: Use TTA
        
    Returns:
        DataFrame with predictions
    """
    model.eval()
    
    all_predictions = []
    all_ids = []
    
    for batch in tqdm(dataloader, desc="Predicting"):
        preds = predict_batch(model, batch, device, use_tta)
        all_predictions.append(preds)
        
        if 'study_id' in batch:
            all_ids.extend(batch['study_id'])
    
    predictions = np.concatenate(all_predictions, axis=0)
    
    # Create submission dataframe
    # RSNA competition format
    columns = [
        'aneurysm_present',
        'ICA_L', 'ICA_R',
        'MCA_L', 'MCA_R',
        'ACA_L', 'ACA_R',
        'PCA_L', 'PCA_R',
        'PCOM_L', 'PCOM_R',
        'BA',
        'VA_L', 'VA_R',
    ]
    
    df = pd.DataFrame(predictions, columns=columns[:predictions.shape[1]])
    
    if all_ids:
        df.insert(0, 'study_id', all_ids)
    
    return df


def create_submission(
    predictions_df: pd.DataFrame,
    sample_submission_path: str,
    output_path: str,
):
    """
    Create submission file in competition format.
    
    Args:
        predictions_df: DataFrame with predictions
        sample_submission_path: Path to sample submission
        output_path: Output path for submission
    """
    sample = pd.read_csv(sample_submission_path)
    
    # Merge predictions with sample submission format
    # This depends on the exact competition format
    submission = sample.copy()
    
    # Update with predictions
    for col in predictions_df.columns:
        if col in submission.columns:
            submission[col] = predictions_df[col].values
    
    submission.to_csv(output_path, index=False)
    print(f"Submission saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Run prediction")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="resnet18",
        help="Model architecture",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Path to test data",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="predictions.csv",
        help="Output file path",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--tta",
        action="store_true",
        help="Use test-time augmentation",
    )
    parser.add_argument(
        "--sample-submission",
        type=str,
        default=None,
        help="Path to sample submission for formatting",
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("RSNA Intracranial Aneurysm Detection - Prediction")
    print("=" * 60)
    print(f"\nCheckpoint: {args.checkpoint}")
    print(f"Model: {args.model_type}")
    print(f"Device: {args.device}")
    print(f"TTA: {args.tta}")
    
    # Load model
    print("\nLoading model...")
    model = load_model(
        args.checkpoint,
        args.model_type,
        device=args.device,
    )
    print("Model loaded!")
    
    # Placeholder for data loading
    print("\n" + "=" * 60)
    print("Data loading placeholder")
    print("=" * 60)
    print("\nTo run predictions:")
    print("1. Implement test dataset loader")
    print("2. Create DataLoader")
    print("3. Run predict_dataset()")
    print("\nExample:")
    print("""
    from src.data import AneurysmDataset
    from src.data.augmentation import get_val_transforms
    
    test_dataset = AneurysmDataset(
        args.data_dir,
        transform=get_val_transforms()
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False
    )
    
    predictions = predict_dataset(
        model, test_loader,
        device=args.device,
        use_tta=args.tta
    )
    
    predictions.to_csv(args.output, index=False)
    """)
    
    print(f"\nOutput would be saved to: {args.output}")


if __name__ == "__main__":
    main()

