import numpy as np
from .activations import get_activation


class Layer:
    """Base class for all layers."""
    
    def forward(self, x, training=True):
        """Forward pass."""
        raise NotImplementedError
    
    def backward(self, grad_output):
        """Backward pass."""
        raise NotImplementedError
    
    def get_params(self):
        """Return trainable parameters."""
        return []
    
    def get_grads(self):
        """Return gradients."""
        return []


class Dense(Layer):
    """
    Fully Connected (Dense) Layer
    
    Computes: output = activation(W @ input + b)
    
    Parameters:
        W: Weight matrix of shape (input_dim, output_dim)
        b: Bias vector of shape (output_dim,)
    """
    
    def __init__(self, input_dim, output_dim, activation='relu'):
        """
        Args:
            input_dim: Number of input features
            output_dim: Number of output features
            activation: Activation function name or instance
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Initialize weights with He initialization (good for ReLU)
        # W ~ N(0, sqrt(2/input_dim))
        self.W = np.random.randn(input_dim, output_dim) * np.sqrt(2.0 / input_dim)
        
        # Initialize biases to zero
        self.b = np.zeros(output_dim)
        
        # Activation function
        if isinstance(activation, str):
            self.activation = get_activation(activation)
        else:
            self.activation = activation
        
        # Cache for backward pass
        self.cache = {}
        
        # Gradients
        self.grad_W = None
        self.grad_b = None
    
    def forward(self, x, training=True):
        """
        Forward pass.
        
        Args:
            x: Input array of shape (batch_size, input_dim)
            training: Whether in training mode
            
        Returns:
            Output array of shape (batch_size, output_dim)
        """
        # Cache input for backward pass
        self.cache['x'] = x
        
        # Linear transformation: z = xW + b
        z = x @ self.W + self.b
        self.cache['z'] = z
        
        # Apply activation
        a = self.activation.forward(z)
        self.cache['a'] = a
        
        return a
    
    def backward(self, grad_output):
        """
        Backward pass.
        
        Args:
            grad_output: Gradient from next layer, shape (batch_size, output_dim)
            
        Returns:
            grad_input: Gradient to pass to previous layer, shape (batch_size, input_dim)
        """
        # Retrieve cached values
        x = self.cache['x']
        z = self.cache['z']
        
        # Gradient through activation: ∂L/∂z = ∂L/∂a × ∂a/∂z
        grad_z = grad_output * self.activation.backward(z)
        
        # Gradient w.r.t weights: ∂L/∂W = x^T @ ∂L/∂z
        self.grad_W = x.T @ grad_z
        
        # Gradient w.r.t bias: ∂L/∂b = sum(∂L/∂z, axis=0)
        self.grad_b = np.sum(grad_z, axis=0)
        
        # Gradient w.r.t input: ∂L/∂x = ∂L/∂z @ W^T
        grad_input = grad_z @ self.W.T
        
        return grad_input
    
    def get_params(self):
        """Return list of trainable parameters."""
        return [self.W, self.b]
    
    def get_grads(self):
        """Return list of gradients."""
        return [self.grad_W, self.grad_b]
    
    def __repr__(self):
        return f"Dense({self.input_dim} → {self.output_dim}, activation={self.activation.__class__.__name__})"


class Dropout(Layer):
    """
    Dropout Layer for regularization.
    
    During training: Randomly sets a fraction of inputs to 0
    During inference: Scales all inputs by keep_prob
    """
    
    def __init__(self, drop_prob=0.5):
        """
        Args:
            drop_prob: Probability of dropping a unit (0 to 1)
        """
        self.drop_prob = drop_prob
        self.keep_prob = 1 - drop_prob
        self.mask = None
    
    def forward(self, x, training=True):
        """
        Args:
            x: Input array
            training: If True, apply dropout; if False, scale by keep_prob
            
        Returns:
            Output array
        """
        if training:
            # Generate dropout mask
            self.mask = np.random.binomial(1, self.keep_prob, size=x.shape)
            # Apply mask and scale
            return x * self.mask / self.keep_prob
        else:
            # No dropout during inference
            return x
    
    def backward(self, grad_output):
        """
        Args:
            grad_output: Gradient from next layer
            
        Returns:
            Gradient scaled by dropout mask
        """
        return grad_output * self.mask / self.keep_prob
    
    def __repr__(self):
        return f"Dropout(drop_prob={self.drop_prob})"


class BatchNormalization(Layer):
    """
    Batch Normalization Layer
    
    Normalizes inputs across the batch dimension.
    """
    
    def __init__(self, num_features, epsilon=1e-5, momentum=0.9):
        """
        Args:
            num_features: Number of features to normalize
            epsilon: Small constant for numerical stability
            momentum: Momentum for running mean/variance
        """
        self.num_features = num_features
        self.epsilon = epsilon
        self.momentum = momentum
        
        # Learnable parameters
        self.gamma = np.ones(num_features)  # Scale
        self.beta = np.zeros(num_features)  # Shift
        
        # Running statistics (for inference)
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)
        
        # Cache
        self.cache = {}
        
        # Gradients
        self.grad_gamma = None
        self.grad_beta = None
    
    def forward(self, x, training=True):
        """
        Args:
            x: Input array of shape (batch_size, num_features)
            training: Whether in training mode
            
        Returns:
            Normalized output
        """
        if training:
            # Compute batch statistics
            batch_mean = np.mean(x, axis=0)
            batch_var = np.var(x, axis=0)
            
            # Normalize
            x_centered = x - batch_mean
            std = np.sqrt(batch_var + self.epsilon)
            x_norm = x_centered / std
            
            # Scale and shift
            out = self.gamma * x_norm + self.beta
            
            # Update running statistics
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * batch_mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * batch_var
            
            # Cache for backward
            self.cache = {
                'x_norm': x_norm,
                'x_centered': x_centered,
                'std': std,
                'batch_size': x.shape[0]
            }
        else:
            # Use running statistics
            x_norm = (x - self.running_mean) / np.sqrt(self.running_var + self.epsilon)
            out = self.gamma * x_norm + self.beta
        
        return out
    
    def backward(self, grad_output):
        """
        Backward pass (simplified version).
        """
        x_norm = self.cache['x_norm']
        x_centered = self.cache['x_centered']
        std = self.cache['std']
        N = self.cache['batch_size']
        
        # Gradients w.r.t gamma and beta
        self.grad_gamma = np.sum(grad_output * x_norm, axis=0)
        self.grad_beta = np.sum(grad_output, axis=0)
        
        # Gradient w.r.t input (simplified)
        grad_x_norm = grad_output * self.gamma
        grad_var = np.sum(grad_x_norm * x_centered * -0.5 * std**(-3), axis=0)
        grad_mean = np.sum(grad_x_norm * -1/std, axis=0) + grad_var * np.sum(-2 * x_centered, axis=0) / N
        
        grad_input = (grad_x_norm / std) + (grad_var * 2 * x_centered / N) + (grad_mean / N)
        
        return grad_input
    
    def get_params(self):
        return [self.gamma, self.beta]
    
    def get_grads(self):
        return [self.grad_gamma, self.grad_beta]
    
    def __repr__(self):
        return f"BatchNormalization({self.num_features})"