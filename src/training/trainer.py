"""
Model Trainer
=============

Training loop and utilities for model training.
"""

import os
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


class Trainer:
    """
    General purpose trainer for PyTorch models.
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: str = 'cuda',
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        mixed_precision: bool = True,
        gradient_clip: Optional[float] = 1.0,
        checkpoint_dir: str = 'checkpoints',
    ):
        """
        Args:
            model: PyTorch model
            optimizer: Optimizer
            criterion: Loss function
            device: Device to train on
            scheduler: Learning rate scheduler
            mixed_precision: Use automatic mixed precision
            gradient_clip: Max gradient norm for clipping
            checkpoint_dir: Directory to save checkpoints
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.scheduler = scheduler
        self.gradient_clip = gradient_clip
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Mixed precision
        self.mixed_precision = mixed_precision and device == 'cuda'
        self.scaler = torch.cuda.amp.GradScaler() if self.mixed_precision else None
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.history: Dict[str, List[float]] = {
            'train_loss': [],
            'val_loss': [],
            'train_metric': [],
            'val_metric': [],
        }
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        metric_fn: Optional[Callable] = None,
    ) -> Tuple[float, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            metric_fn: Optional metric function
            
        Returns:
            Tuple of (average loss, average metric)
        """
        self.model.train()
        total_loss = 0.0
        total_metric = 0.0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {self.current_epoch} [Train]')
        
        for batch in pbar:
            # Get data
            inputs = batch['volume'].to(self.device)
            targets = batch['label'].to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if self.mixed_precision:
                with torch.cuda.amp.autocast():
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                
                # Backward pass
                self.scaler.scale(loss).backward()
                
                if self.gradient_clip:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.gradient_clip
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                
                if self.gradient_clip:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.gradient_clip
                    )
                
                self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            if metric_fn:
                with torch.no_grad():
                    metric = metric_fn(outputs, targets)
                    total_metric += metric
            
            num_batches += 1
            pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / num_batches
        avg_metric = total_metric / num_batches if metric_fn else 0.0
        
        return avg_loss, avg_metric
    
    @torch.no_grad()
    def validate(
        self,
        val_loader: DataLoader,
        metric_fn: Optional[Callable] = None,
    ) -> Tuple[float, float]:
        """
        Validate the model.
        
        Args:
            val_loader: Validation data loader
            metric_fn: Optional metric function
            
        Returns:
            Tuple of (average loss, average metric)
        """
        self.model.eval()
        total_loss = 0.0
        total_metric = 0.0
        num_batches = 0
        
        pbar = tqdm(val_loader, desc=f'Epoch {self.current_epoch} [Val]')
        
        for batch in pbar:
            inputs = batch['volume'].to(self.device)
            targets = batch['label'].to(self.device)
            
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            total_loss += loss.item()
            if metric_fn:
                metric = metric_fn(outputs, targets)
                total_metric += metric
            
            num_batches += 1
            pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / num_batches
        avg_metric = total_metric / num_batches if metric_fn else 0.0
        
        return avg_loss, avg_metric
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 100,
        metric_fn: Optional[Callable] = None,
        early_stopping: int = 10,
        save_best: bool = True,
    ) -> Dict[str, List[float]]:
        """
        Full training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs
            metric_fn: Metric function for evaluation
            early_stopping: Patience for early stopping
            save_best: Save best model checkpoint
            
        Returns:
            Training history
        """
        patience_counter = 0
        
        for epoch in range(epochs):
            self.current_epoch = epoch
            start_time = time.time()
            
            # Train
            train_loss, train_metric = self.train_epoch(train_loader, metric_fn)
            self.history['train_loss'].append(train_loss)
            self.history['train_metric'].append(train_metric)
            
            # Validate
            if val_loader:
                val_loss, val_metric = self.validate(val_loader, metric_fn)
                self.history['val_loss'].append(val_loss)
                self.history['val_metric'].append(val_metric)
                
                # Learning rate scheduling
                if self.scheduler:
                    if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(val_loss)
                    else:
                        self.scheduler.step()
                
                # Save best model
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    patience_counter = 0
                    if save_best:
                        self.save_checkpoint('best_model.pth')
                else:
                    patience_counter += 1
                
                # Early stopping
                if patience_counter >= early_stopping:
                    print(f'\nEarly stopping at epoch {epoch}')
                    break
            
            # Logging
            epoch_time = time.time() - start_time
            lr = self.optimizer.param_groups[0]['lr']
            
            log_msg = f'Epoch {epoch}: train_loss={train_loss:.4f}'
            if val_loader:
                log_msg += f', val_loss={val_loss:.4f}'
            if metric_fn:
                log_msg += f', train_metric={train_metric:.4f}'
                if val_loader:
                    log_msg += f', val_metric={val_metric:.4f}'
            log_msg += f', lr={lr:.6f}, time={epoch_time:.1f}s'
            
            print(log_msg)
        
        return self.history
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'history': self.history,
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, self.checkpoint_dir / filename)
        print(f'Checkpoint saved: {self.checkpoint_dir / filename}')
    
    def load_checkpoint(self, filename: str):
        """Load model checkpoint."""
        checkpoint = torch.load(self.checkpoint_dir / filename, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.history = checkpoint['history']
        
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        print(f'Checkpoint loaded: {filename}')


class SegmentationTrainer(Trainer):
    """
    Specialized trainer for segmentation tasks.
    """
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        metric_fn: Optional[Callable] = None,
    ) -> Tuple[float, float]:
        """Override for segmentation-specific training."""
        self.model.train()
        total_loss = 0.0
        total_metric = 0.0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {self.current_epoch} [Train]')
        
        for batch in pbar:
            inputs = batch['volume'].to(self.device)
            masks = batch['mask'].to(self.device)
            
            self.optimizer.zero_grad()
            
            if self.mixed_precision:
                with torch.cuda.amp.autocast():
                    outputs = self.model(inputs)
                    
                    # Handle deep supervision
                    if isinstance(outputs, tuple):
                        main_out, ds_outputs = outputs
                        loss = self.criterion(main_out, masks)
                        for ds_out in ds_outputs:
                            loss += 0.5 * self.criterion(ds_out, masks)
                    else:
                        loss = self.criterion(outputs, masks)
                
                self.scaler.scale(loss).backward()
                
                if self.gradient_clip:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.gradient_clip
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(inputs)
                
                if isinstance(outputs, tuple):
                    main_out, ds_outputs = outputs
                    loss = self.criterion(main_out, masks)
                    for ds_out in ds_outputs:
                        loss += 0.5 * self.criterion(ds_out, masks)
                    outputs = main_out
                else:
                    loss = self.criterion(outputs, masks)
                
                loss.backward()
                
                if self.gradient_clip:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.gradient_clip
                    )
                
                self.optimizer.step()
            
            total_loss += loss.item()
            if metric_fn:
                with torch.no_grad():
                    if isinstance(outputs, tuple):
                        outputs = outputs[0]
                    metric = metric_fn(outputs, masks)
                    total_metric += metric
            
            num_batches += 1
            pbar.set_postfix({'loss': loss.item()})
        
        return total_loss / num_batches, total_metric / num_batches if metric_fn else 0.0

