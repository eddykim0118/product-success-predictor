#!/usr/bin/env python
"""
Training Script
===============

Train aneurysm detection models.
"""

import argparse
import os
import sys
from pathlib import Path

import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import resnet18_3d, resnet34_3d, resnet50_3d, UNet3D
from src.training import Trainer, SegmentationTrainer
from src.training.losses import get_loss_function


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_model(config: dict) -> nn.Module:
    """Create model based on config."""
    model_config = config.get('model', {})
    model_type = model_config.get('type', 'resnet18')
    num_classes = model_config.get('num_classes', 14)
    in_channels = model_config.get('in_channels', 1)
    dropout = model_config.get('dropout', 0.5)
    
    model_map = {
        'resnet18': resnet18_3d,
        'resnet34': resnet34_3d,
        'resnet50': resnet50_3d,
    }
    
    if model_type in model_map:
        return model_map[model_type](
            in_channels=in_channels,
            num_classes=num_classes,
            dropout=dropout,
        )
    elif model_type == 'unet3d':
        return UNet3D(
            in_channels=in_channels,
            out_channels=model_config.get('out_channels', 1),
            base_filters=model_config.get('base_filters', 32),
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def get_optimizer(model: nn.Module, config: dict) -> torch.optim.Optimizer:
    """Create optimizer based on config."""
    opt_config = config.get('optimizer', {})
    opt_type = opt_config.get('type', 'adam')
    lr = opt_config.get('lr', 1e-3)
    weight_decay = opt_config.get('weight_decay', 1e-4)
    
    if opt_type == 'adam':
        return torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
    elif opt_type == 'adamw':
        return torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
    elif opt_type == 'sgd':
        momentum = opt_config.get('momentum', 0.9)
        return torch.optim.SGD(
            model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer: {opt_type}")


def get_scheduler(optimizer, config: dict):
    """Create learning rate scheduler based on config."""
    sched_config = config.get('scheduler', {})
    sched_type = sched_config.get('type', 'cosine')
    
    if sched_type == 'cosine':
        epochs = config.get('training', {}).get('epochs', 100)
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif sched_type == 'plateau':
        patience = sched_config.get('patience', 5)
        factor = sched_config.get('factor', 0.5)
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=patience, factor=factor
        )
    elif sched_type == 'step':
        step_size = sched_config.get('step_size', 30)
        gamma = sched_config.get('gamma', 0.1)
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size, gamma)
    else:
        return None


def get_criterion(config: dict) -> nn.Module:
    """Create loss function based on config."""
    loss_config = config.get('training', {}).get('loss', {})
    
    if isinstance(loss_config, str):
        loss_type = loss_config
        loss_kwargs = {}
    else:
        loss_type = loss_config.get('type', 'bce')
        loss_kwargs = {k: v for k, v in loss_config.items() if k != 'type'}
    
    return get_loss_function(loss_type, **loss_kwargs)


def main():
    parser = argparse.ArgumentParser(description="Train aneurysm detection model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/baseline.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/processed",
        help="Path to processed data"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Output directory for checkpoints and logs"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to train on"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )
    
    args = parser.parse_args()
    
    # Load config
    if os.path.exists(args.config):
        config = load_config(args.config)
    else:
        print(f"Config file not found: {args.config}")
        print("Using default configuration...")
        config = {
            'model': {'type': 'conv3d', 'base_filters': 32},
            'optimizer': {'type': 'adam', 'lr': 1e-3},
            'scheduler': {'type': 'cosine'},
            'training': {
                'epochs': 100,
                'batch_size': 4,
                'loss': 'cross_entropy',
            }
        }
    
    print("=" * 60)
    print("RSNA Intracranial Aneurysm Detection - Training")
    print("=" * 60)
    print(f"\nDevice: {args.device}")
    print(f"Config: {args.config}")
    print(f"Data: {args.data_dir}")
    print(f"Output: {args.output_dir}")
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create model
    print("\nCreating model...")
    model = get_model(config)
    print(f"Model: {config['model']['type']}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create optimizer and scheduler
    optimizer = get_optimizer(model, config)
    scheduler = get_scheduler(optimizer, config)
    criterion = get_criterion(config)
    
    # Create trainer
    trainer_cls = (
        SegmentationTrainer 
        if config['model']['type'] == 'unet3d' 
        else Trainer
    )
    
    trainer = trainer_cls(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=args.device,
        scheduler=scheduler,
        checkpoint_dir=str(output_path / 'checkpoints'),
    )
    
    # Resume from checkpoint
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    print("\n" + "=" * 60)
    print("Data loading placeholder")
    print("=" * 60)
    print("\nTo start training with real data:")
    print("1. Download data from Kaggle")
    print("2. Preprocess the DICOM data")
    print("3. Update the data loading code below")
    print("\nExample:")
    print("""
    from src.data import AneurysmDataset
    from src.data.augmentation import get_train_transforms, get_val_transforms
    
    train_dataset = AneurysmDataset(
        data_dir=args.data_dir,
        labels_file='data/train.csv',
        transform=get_train_transforms()
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=4
    )
    
    history = trainer.fit(
        train_loader,
        val_loader,
        epochs=config['training']['epochs'],
    )
    """)
    
    print("\n" + "=" * 60)
    print("Training script ready!")
    print("=" * 60)


if __name__ == "__main__":
    main()

