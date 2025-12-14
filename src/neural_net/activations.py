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
    
    def forward(self, x):
        # Clip to prevent overflow
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))
    
    def backward(self, x):
        s = self.forward(x)
        return s * (1 - s)


class ReLU(Activation):
    
    def forward(self, x):
        return np.maximum(0, x)
    
    def backward(self, x):
        return (x > 0).astype(float)


class Tanh(Activation):
    
    def forward(self, x):
        return np.tanh(x)
    
    def backward(self, x):
        t = self.forward(x)
        return 1 - t**2


class LeakyReLU(Activation):
    
    def __init__(self, alpha=0.01):
        self.alpha = alpha
    
    def forward(self, x):
        return np.where(x > 0, x, self.alpha * x)
    
    def backward(self, x):
        return np.where(x > 0, 1, self.alpha)


class Softmax(Activation):
    
    def forward(self, x):
        # Subtract max for numerical stability
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def backward(self, x):
        s = self.forward(x)
        # For batch processing, this returns the diagonal of Jacobian
        return s * (1 - s)


# Convenience function
def get_activation(name):
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