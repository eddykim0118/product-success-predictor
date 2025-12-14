import numpy as np


class Loss:
    """Base class for loss functions."""
    
    def forward(self, y_true, y_pred):
        """
        Compute loss value.
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted values
            
        Returns:
            Loss value (scalar)
        """
        raise NotImplementedError
    
    def backward(self, y_true, y_pred):
        """
        Compute gradient of loss w.r.t predictions.
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted values
            
        Returns:
            Gradient array (same shape as y_pred)
        """
        raise NotImplementedError


class MSE(Loss):
    """
    Mean Squared Error Loss
    
    L = (1/n) × Σ(y_true - y_pred)²
    
    Used for: Regression problems
    """
    
    def forward(self, y_true, y_pred):
        """
        Args:
            y_true: True values, shape (batch_size, output_dim)
            y_pred: Predicted values, shape (batch_size, output_dim)
            
        Returns:
            Mean squared error (scalar)
        """
        return np.mean((y_true - y_pred) ** 2)
    
    def backward(self, y_true, y_pred):
        """
        Gradient: ∂L/∂y_pred = (2/n) × (y_pred - y_true)
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Gradient of same shape as y_pred
        """
        n = y_true.shape[0]  # batch size
        return (2 / n) * (y_pred - y_true)


class BinaryCrossEntropy(Loss):
    """
    Binary Cross-Entropy Loss
    
    L = -(1/n) × Σ[y×log(ŷ) + (1-y)×log(1-ŷ)]
    
    Used for: Binary classification (0 or 1)
    Pair with: Sigmoid activation
    """
    
    def __init__(self, epsilon=1e-15):
        """
        Args:
            epsilon: Small constant to prevent log(0)
        """
        self.epsilon = epsilon
    
    def forward(self, y_true, y_pred):
        """
        Args:
            y_true: True labels (0 or 1), shape (batch_size, 1)
            y_pred: Predicted probabilities, shape (batch_size, 1)
            
        Returns:
            Binary cross-entropy loss (scalar)
        """
        # Clip predictions to prevent log(0)
        y_pred = np.clip(y_pred, self.epsilon, 1 - self.epsilon)
        
        # BCE formula
        loss = -np.mean(
            y_true * np.log(y_pred) + 
            (1 - y_true) * np.log(1 - y_pred)
        )
        return loss
    
    def backward(self, y_true, y_pred):
        """
        Gradient: ∂L/∂y_pred = (1/n) × [(ŷ - y) / (ŷ(1-ŷ))]
        
        Simplified when paired with sigmoid:
        ∂L/∂y_pred = (1/n) × (ŷ - y)
        
        Args:
            y_true: True labels
            y_pred: Predicted probabilities
            
        Returns:
            Gradient
        """
        y_pred = np.clip(y_pred, self.epsilon, 1 - self.epsilon)
        n = y_true.shape[0]
        
        # Gradient
        grad = (y_pred - y_true) / (y_pred * (1 - y_pred))
        return grad / n


class CategoricalCrossEntropy(Loss):
    """
    Categorical Cross-Entropy Loss
    
    L = -(1/n) × Σᵢ Σⱼ y_ij × log(ŷ_ij)
    
    Used for: Multi-class classification
    Pair with: Softmax activation
    """
    
    def __init__(self, epsilon=1e-15):
        """
        Args:
            epsilon: Small constant to prevent log(0)
        """
        self.epsilon = epsilon
    
    def forward(self, y_true, y_pred):
        """
        Args:
            y_true: One-hot encoded labels, shape (batch_size, num_classes)
            y_pred: Predicted probabilities, shape (batch_size, num_classes)
            
        Returns:
            Categorical cross-entropy loss (scalar)
        """
        # Clip predictions
        y_pred = np.clip(y_pred, self.epsilon, 1 - self.epsilon)
        
        # CCE formula
        loss = -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]
        return loss
    
    def backward(self, y_true, y_pred):
        """
        Gradient: ∂L/∂y_pred = -(1/n) × (y / ŷ)
        
        Simplified when paired with softmax:
        ∂L/∂y_pred = (1/n) × (ŷ - y)
        
        Args:
            y_true: One-hot encoded labels
            y_pred: Predicted probabilities
            
        Returns:
            Gradient
        """
        y_pred = np.clip(y_pred, self.epsilon, 1 - self.epsilon)
        n = y_true.shape[0]
        
        # Simplified gradient (works with softmax)
        return (y_pred - y_true) / n


class SoftmaxCategoricalCrossEntropy(Loss):
    """
    Combined Softmax + Categorical Cross-Entropy
    
    More numerically stable than computing separately.
    
    Used for: Multi-class classification
    """
    
    def __init__(self, epsilon=1e-15):
        self.epsilon = epsilon
    
    def forward(self, y_true, logits):
        """
        Args:
            y_true: One-hot encoded labels
            logits: Raw network outputs (before softmax)
            
        Returns:
            Loss value
        """
        # Apply softmax
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        y_pred = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        
        # Clip and compute CCE
        y_pred = np.clip(y_pred, self.epsilon, 1 - self.epsilon)
        loss = -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]
        return loss
    
    def backward(self, y_true, logits):
        """
        Combined gradient is simply: (softmax(logits) - y_true) / n
        
        This is the magic of softmax + CCE combination!
        """
        # Apply softmax
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        y_pred = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        
        # Simple gradient
        n = y_true.shape[0]
        return (y_pred - y_true) / n


# Convenience function
def get_loss(name):
    """
    Get loss function by name.
    
    Args:
        name: String name of loss
        
    Returns:
        Loss instance
    """
    losses = {
        'mse': MSE(),
        'binary_crossentropy': BinaryCrossEntropy(),
        'categorical_crossentropy': CategoricalCrossEntropy(),
        'softmax_crossentropy': SoftmaxCategoricalCrossEntropy(),
    }
    
    name = name.lower()
    if name not in losses:
        raise ValueError(f"Unknown loss: {name}. Choose from {list(losses.keys())}")
    
    return losses[name]