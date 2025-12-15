"""
Visualization Utilities
=======================

Functions for visualizing medical images and training progress.
"""

from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np


def plot_slice(
    volume: np.ndarray,
    slice_idx: Optional[int] = None,
    axis: int = 0,
    title: str = "",
    cmap: str = "gray",
    figsize: Tuple[int, int] = (8, 8),
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    overlay: Optional[np.ndarray] = None,
    overlay_alpha: float = 0.3,
    overlay_cmap: str = "Reds",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot a single slice from a 3D volume.
    
    Args:
        volume: 3D numpy array
        slice_idx: Index of slice to display (default: middle)
        axis: Axis to slice along (0=axial, 1=coronal, 2=sagittal)
        title: Plot title
        cmap: Colormap for main image
        figsize: Figure size
        vmin: Minimum value for colormap
        vmax: Maximum value for colormap
        overlay: Optional overlay mask
        overlay_alpha: Transparency of overlay
        overlay_cmap: Colormap for overlay
        save_path: Path to save figure
        
    Returns:
        matplotlib Figure
    """
    # Get slice
    if slice_idx is None:
        slice_idx = volume.shape[axis] // 2
    
    if axis == 0:
        img = volume[slice_idx, :, :]
        overlay_slice = overlay[slice_idx, :, :] if overlay is not None else None
    elif axis == 1:
        img = volume[:, slice_idx, :]
        overlay_slice = overlay[:, slice_idx, :] if overlay is not None else None
    else:
        img = volume[:, :, slice_idx]
        overlay_slice = overlay[:, :, slice_idx] if overlay is not None else None
    
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    # Plot main image
    ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
    
    # Plot overlay if provided
    if overlay_slice is not None:
        masked_overlay = np.ma.masked_where(overlay_slice == 0, overlay_slice)
        ax.imshow(masked_overlay, cmap=overlay_cmap, alpha=overlay_alpha)
    
    ax.set_title(title)
    ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_volume_grid(
    volume: np.ndarray,
    num_slices: int = 16,
    axis: int = 0,
    title: str = "",
    cmap: str = "gray",
    figsize: Tuple[int, int] = (16, 16),
    overlay: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot a grid of slices from a 3D volume.
    
    Args:
        volume: 3D numpy array
        num_slices: Number of slices to display
        axis: Axis to slice along
        title: Main title
        cmap: Colormap
        figsize: Figure size
        overlay: Optional overlay mask
        save_path: Path to save figure
        
    Returns:
        matplotlib Figure
    """
    # Calculate grid dimensions
    cols = int(np.ceil(np.sqrt(num_slices)))
    rows = int(np.ceil(num_slices / cols))
    
    # Get evenly spaced slice indices
    total_slices = volume.shape[axis]
    indices = np.linspace(0, total_slices - 1, num_slices, dtype=int)
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten()
    
    for i, (ax, idx) in enumerate(zip(axes, indices)):
        if axis == 0:
            img = volume[idx, :, :]
            overlay_slice = overlay[idx, :, :] if overlay is not None else None
        elif axis == 1:
            img = volume[:, idx, :]
            overlay_slice = overlay[:, idx, :] if overlay is not None else None
        else:
            img = volume[:, :, idx]
            overlay_slice = overlay[:, :, idx] if overlay is not None else None
        
        ax.imshow(img, cmap=cmap)
        
        if overlay_slice is not None:
            masked_overlay = np.ma.masked_where(overlay_slice == 0, overlay_slice)
            ax.imshow(masked_overlay, cmap='Reds', alpha=0.3)
        
        ax.set_title(f'Slice {idx}')
        ax.axis('off')
    
    # Hide empty subplots
    for ax in axes[num_slices:]:
        ax.axis('off')
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_3d_volume(
    volume: np.ndarray,
    threshold: float = 0.5,
    title: str = "3D Volume",
    figsize: Tuple[int, int] = (10, 10),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot 3D volume using matplotlib (simplified 3D view).
    
    For interactive 3D visualization, consider using plotly.
    
    Args:
        volume: 3D numpy array
        threshold: Threshold for surface rendering
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure
        
    Returns:
        matplotlib Figure
    """
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Get coordinates of voxels above threshold
    coords = np.where(volume > threshold)
    
    # Subsample if too many points
    max_points = 10000
    if len(coords[0]) > max_points:
        idx = np.random.choice(len(coords[0]), max_points, replace=False)
        coords = (coords[0][idx], coords[1][idx], coords[2][idx])
    
    # Plot points
    ax.scatter(
        coords[2], coords[1], coords[0],
        c=volume[coords],
        cmap='viridis',
        alpha=0.1,
        s=1
    )
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_training_history(
    history: Dict[str, List[float]],
    title: str = "Training History",
    figsize: Tuple[int, int] = (14, 5),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot training history curves.
    
    Args:
        history: Dict with 'train_loss', 'val_loss', etc.
        title: Main title
        figsize: Figure size
        save_path: Path to save figure
        
    Returns:
        matplotlib Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Plot loss
    ax = axes[0]
    if 'train_loss' in history:
        ax.plot(history['train_loss'], label='Train Loss', linewidth=2)
    if 'val_loss' in history:
        ax.plot(history['val_loss'], label='Val Loss', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Loss Curves')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot metrics
    ax = axes[1]
    if 'train_metric' in history:
        ax.plot(history['train_metric'], label='Train Metric', linewidth=2)
    if 'val_metric' in history:
        ax.plot(history['val_metric'], label='Val Metric', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Metric')
    ax.set_title('Metric Curves')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_predictions(
    volume: np.ndarray,
    prediction: np.ndarray,
    ground_truth: Optional[np.ndarray] = None,
    slice_idx: Optional[int] = None,
    figsize: Tuple[int, int] = (15, 5),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot input, prediction, and optionally ground truth side by side.
    
    Args:
        volume: Input volume
        prediction: Model prediction
        ground_truth: Optional ground truth
        slice_idx: Slice index to display
        figsize: Figure size
        save_path: Path to save figure
        
    Returns:
        matplotlib Figure
    """
    if slice_idx is None:
        slice_idx = volume.shape[0] // 2
    
    num_cols = 3 if ground_truth is not None else 2
    fig, axes = plt.subplots(1, num_cols, figsize=figsize)
    
    # Input
    axes[0].imshow(volume[slice_idx], cmap='gray')
    axes[0].set_title('Input')
    axes[0].axis('off')
    
    # Prediction
    axes[1].imshow(volume[slice_idx], cmap='gray')
    masked_pred = np.ma.masked_where(prediction[slice_idx] < 0.5, prediction[slice_idx])
    axes[1].imshow(masked_pred, cmap='Reds', alpha=0.5)
    axes[1].set_title('Prediction')
    axes[1].axis('off')
    
    # Ground truth
    if ground_truth is not None:
        axes[2].imshow(volume[slice_idx], cmap='gray')
        masked_gt = np.ma.masked_where(ground_truth[slice_idx] < 0.5, ground_truth[slice_idx])
        axes[2].imshow(masked_gt, cmap='Greens', alpha=0.5)
        axes[2].set_title('Ground Truth')
        axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig

