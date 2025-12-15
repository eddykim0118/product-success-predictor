#!/usr/bin/env python
"""
Visualization Script
====================

Visualize training results and predictions.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def plot_training_curves(
    history: Dict[str, List[float]],
    output_path: Optional[str] = None,
    figsize: tuple = (14, 5),
):
    """
    Plot training loss and metric curves.
    
    Args:
        history: Training history dict
        output_path: Optional path to save figure
        figsize: Figure size
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Loss curves
    ax = axes[0]
    if 'train_loss' in history:
        ax.plot(history['train_loss'], label='Train Loss', linewidth=2)
    if 'val_loss' in history:
        ax.plot(history['val_loss'], label='Val Loss', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Metric curves
    ax = axes[1]
    if 'train_auc' in history:
        ax.plot(history['train_auc'], label='Train AUC', linewidth=2)
    if 'val_auc' in history:
        ax.plot(history['val_auc'], label='Val AUC', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('AUC')
    ax.set_title('Training AUC')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved to: {output_path}")
    
    plt.show()


def plot_per_class_metrics(
    metrics: Dict[str, float],
    class_names: List[str],
    output_path: Optional[str] = None,
    figsize: tuple = (12, 6),
):
    """
    Plot per-class AUC scores.
    
    Args:
        metrics: Dict with AUC per class
        class_names: List of class names
        output_path: Optional save path
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    aucs = [metrics.get(f'auc_{name}', 0) for name in class_names]
    colors = ['#e74c3c' if name == 'aneurysm_present' else '#3498db' for name in class_names]
    
    bars = ax.barh(class_names, aucs, color=colors)
    
    # Add value labels
    for bar, auc in zip(bars, aucs):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f'{auc:.3f}', va='center')
    
    ax.set_xlabel('AUC Score')
    ax.set_title('Per-Class AUC Scores')
    ax.set_xlim([0, 1.1])
    ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved to: {output_path}")
    
    plt.show()


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    output_path: Optional[str] = None,
    figsize: tuple = (10, 8),
):
    """
    Plot confusion matrix for multi-label classification.
    
    Args:
        y_true: True labels (N, C)
        y_pred: Predicted labels (N, C) - binary
        class_names: Class names
        output_path: Optional save path
        figsize: Figure size
    """
    from sklearn.metrics import multilabel_confusion_matrix
    
    # Get confusion matrices
    cms = multilabel_confusion_matrix(y_true, y_pred)
    
    # Plot
    n_classes = len(class_names)
    n_cols = 4
    n_rows = (n_classes + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()
    
    for i, (cm, name) in enumerate(zip(cms, class_names)):
        ax = axes[i]
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Neg', 'Pos'],
            yticklabels=['Neg', 'Pos'],
            ax=ax
        )
        ax.set_title(name)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
    
    # Hide empty subplots
    for i in range(n_classes, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved to: {output_path}")
    
    plt.show()


def plot_roc_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: List[str],
    output_path: Optional[str] = None,
    figsize: tuple = (10, 8),
):
    """
    Plot ROC curves for each class.
    
    Args:
        y_true: True labels (N, C)
        y_prob: Predicted probabilities (N, C)
        class_names: Class names
        output_path: Optional save path
        figsize: Figure size
    """
    from sklearn.metrics import roc_curve, auc
    
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = plt.cm.tab20(np.linspace(0, 1, len(class_names)))
    
    for i, (name, color) in enumerate(zip(class_names, colors)):
        fpr, tpr, _ = roc_curve(y_true[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        
        linewidth = 3 if name == 'aneurysm_present' else 1.5
        ax.plot(fpr, tpr, color=color, lw=linewidth,
                label=f'{name} (AUC = {roc_auc:.3f})')
    
    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves')
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved to: {output_path}")
    
    plt.show()


def plot_prediction_distribution(
    y_prob: np.ndarray,
    y_true: np.ndarray,
    class_idx: int = 0,
    class_name: str = 'aneurysm_present',
    output_path: Optional[str] = None,
    figsize: tuple = (10, 5),
):
    """
    Plot distribution of predicted probabilities.
    
    Args:
        y_prob: Predicted probabilities
        y_true: True labels
        class_idx: Index of class to plot
        class_name: Name of class
        output_path: Optional save path
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Separate positive and negative samples
    pos_probs = y_prob[y_true[:, class_idx] == 1, class_idx]
    neg_probs = y_prob[y_true[:, class_idx] == 0, class_idx]
    
    ax.hist(neg_probs, bins=50, alpha=0.7, label='Negative', color='blue', density=True)
    ax.hist(pos_probs, bins=50, alpha=0.7, label='Positive', color='red', density=True)
    
    ax.axvline(x=0.5, color='black', linestyle='--', label='Threshold')
    
    ax.set_xlabel('Predicted Probability')
    ax.set_ylabel('Density')
    ax.set_title(f'Prediction Distribution - {class_name}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved to: {output_path}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Visualize results")
    parser.add_argument(
        "--history",
        type=str,
        help="Path to training history JSON",
    )
    parser.add_argument(
        "--predictions",
        type=str,
        help="Path to predictions CSV",
    )
    parser.add_argument(
        "--ground-truth",
        type=str,
        help="Path to ground truth CSV",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/figures",
        help="Output directory for figures",
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("RSNA Intracranial Aneurysm Detection - Visualization")
    print("=" * 60)
    
    # Class names for RSNA competition
    class_names = [
        'aneurysm_present',
        'ICA_L', 'ICA_R',
        'MCA_L', 'MCA_R',
        'ACA_L', 'ACA_R',
        'PCA_L', 'PCA_R',
        'PCOM_L', 'PCOM_R',
        'BA',
        'VA_L', 'VA_R',
    ]
    
    # Plot training curves
    if args.history and os.path.exists(args.history):
        print("\nPlotting training curves...")
        with open(args.history) as f:
            history = json.load(f)
        plot_training_curves(
            history,
            output_path=str(output_dir / 'training_curves.png')
        )
    
    # Plot prediction analysis
    if args.predictions and args.ground_truth:
        if os.path.exists(args.predictions) and os.path.exists(args.ground_truth):
            print("\nPlotting prediction analysis...")
            
            pred_df = pd.read_csv(args.predictions)
            true_df = pd.read_csv(args.ground_truth)
            
            # Get probability columns
            prob_cols = [c for c in class_names if c in pred_df.columns]
            
            y_prob = pred_df[prob_cols].values
            y_true = true_df[prob_cols].values
            
            # ROC curves
            plot_roc_curves(
                y_true, y_prob, prob_cols,
                output_path=str(output_dir / 'roc_curves.png')
            )
            
            # Prediction distribution
            plot_prediction_distribution(
                y_prob, y_true, class_idx=0,
                class_name='aneurysm_present',
                output_path=str(output_dir / 'pred_distribution.png')
            )
    
    print("\n" + "=" * 60)
    print("Visualization complete!")
    print(f"Figures saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()

