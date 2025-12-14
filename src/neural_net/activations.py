import numpy as np


class Activation:
    """Base class for activation functions."""
    
    def forward(self, x):
        """Apply activation function."""
        raise NotImplementedError
    
    def backward(self, x):
        """Compute derivative of activation function."""
        raise NotImplementedError


class Sigmoid(Activation):
    """
    Sigmoid activation: σ(x) = 1 / (1 + e^(-x))
    
    Range: (0, 1)
    Used for: Binary classification output layers
    """
    
    def forward(self, x):
        """
        Args:
            x: Input array of any shape
            
        Returns:
            Output array of same shape, values in (0, 1)
        """
        # Clip to prevent overflow
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))
    
    def backward(self, x):
        """
        Derivative: σ'(x) = σ(x) * (1 - σ(x))
        
        Args:
            x: Input array (same as forward pass)
            
        Returns:
            Derivative at each point
        """
        s = self.forward(x)
        return s * (1 - s)


class ReLU(Activation):
    """
    Rectified Linear Unit: f(x) = max(0, x)
    
    Range: [0, ∞)
    Used for: Hidden layers (most common)
    """
    
    def forward(self, x):
        """
        Args:
            x: Input array
            
        Returns:
            Output array, negative values set to 0
        """
        return np.maximum(0, x)
    
    def backward(self, x):
        """
        Derivative: f'(x) = 1 if x > 0, else 0
        
        Args:
            x: Input array
            
        Returns:
            Gradient: 1 where x > 0, 0 elsewhere
        """
        return (x > 0).astype(float)


class Tanh(Activation):
    """
    Hyperbolic tangent: tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
    
    Range: (-1, 1)
    Used for: Hidden layers (centered around 0)
    """
    
    def forward(self, x):
        """
        Args:
            x: Input array
            
        Returns:
            Output array, values in (-1, 1)
        """
        return np.tanh(x)
    
    def backward(self, x):
        """
        Derivative: tanh'(x) = 1 - tanh²(x)
        
        Args:
            x: Input array
            
        Returns:
            Gradient at each point
        """
        t = self.forward(x)
        return 1 - t**2


class LeakyReLU(Activation):
    """
    Leaky ReLU: f(x) = x if x > 0, else alpha * x
    
    Range: (-∞, ∞)
    Used for: Hidden layers (prevents dying ReLU problem)
    """
    
    def __init__(self, alpha=0.01):
        """
        Args:
            alpha: Slope for negative values (default: 0.01)
        """
        self.alpha = alpha
    
    def forward(self, x):
        """
        Args:
            x: Input array
            
        Returns:
            Output array
        """
        return np.where(x > 0, x, self.alpha * x)
    
    def backward(self, x):
        """
        Derivative: f'(x) = 1 if x > 0, else alpha
        
        Args:
            x: Input array
            
        Returns:
            Gradient
        """
        return np.where(x > 0, 1, self.alpha)


class Softmax(Activation):
    """
    Softmax: σ(x)_i = e^(x_i) / Σ(e^(x_j))
    
    Range: (0, 1), sum to 1
    Used for: Multi-class classification output
    """
    
    def forward(self, x):
        """
        Args:
            x: Input array of shape (batch_size, num_classes)
            
        Returns:
            Probability distribution over classes
        """
        # Subtract max for numerical stability
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def backward(self, x):
        """
        Note: Softmax derivative is usually computed together with
        cross-entropy loss for efficiency. This is the Jacobian.
        
        Args:
            x: Input array
            
        Returns:
            Jacobian matrix (not typically used directly)
        """
        s = self.forward(x)
        # For batch processing, this returns the diagonal of Jacobian
        return s * (1 - s)


# Convenience function
def get_activation(name):
    """
    Get activation function by name.
    
    Args:
        name: String name of activation ('sigmoid', 'relu', 'tanh', etc.)
        
    Returns:
        Activation instance
        
    Example:
        >>> activation = get_activation('relu')
        >>> output = activation.forward(x)
    """
    activations = {
        'sigmoid': Sigmoid(),
        'relu': ReLU(),
        'tanh': Tanh(),
        'leaky_relu': LeakyReLU(),
        'softmax': Softmax(),
    }
    
    name = name.lower()
    if name not in activations:
        raise ValueError(f"Unknown activation: {name}. Choose from {list(activations.keys())}")
    
    return activations[name]