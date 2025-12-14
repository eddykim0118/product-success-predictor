import numpy as np


class Optimizer:
    """Base class for optimizers."""
    
    def __init__(self, learning_rate=0.01):
        """
        Args:
            learning_rate: Step size for parameter updates
        """
        self.learning_rate = learning_rate
    
    def update(self, params, grads):
        """
        Update parameters using gradients.
        
        Args:
            params: List of parameter arrays
            grads: List of gradient arrays (same order as params)
        """
        raise NotImplementedError


class SGD(Optimizer):
    """
    Stochastic Gradient Descent
    
    Update rule:
        θ = θ - α × ∇L(θ)
    
    where:
        θ: parameters (weights)
        α: learning rate
        ∇L(θ): gradients
    """
    
    def __init__(self, learning_rate=0.01):
        """
        Args:
            learning_rate: Learning rate (step size)
        """
        super().__init__(learning_rate)
    
    def update(self, params, grads):
        """
        Update parameters.
        
        Args:
            params: List of parameter arrays [W1, b1, W2, b2, ...]
            grads: List of gradient arrays [dW1, db1, dW2, db2, ...]
        """
        for param, grad in zip(params, grads):
            param -= self.learning_rate * grad
    
    def __repr__(self):
        return f"SGD(lr={self.learning_rate})"


class SGDMomentum(Optimizer):
    """
    SGD with Momentum
    
    Update rule:
        v = β × v + α × ∇L(θ)
        θ = θ - v
    
    Momentum helps accelerate SGD in relevant directions and dampens oscillations.
    
    Think of it as a ball rolling down a hill:
    - Gains speed (momentum) going downhill
    - Doesn't stop immediately when gradient changes
    """
    
    def __init__(self, learning_rate=0.01, momentum=0.9):
        """
        Args:
            learning_rate: Learning rate
            momentum: Momentum coefficient (usually 0.9 or 0.99)
        """
        super().__init__(learning_rate)
        self.momentum = momentum
        self.velocities = None
    
    def update(self, params, grads):
        """
        Update parameters with momentum.
        """
        # Initialize velocities on first update
        if self.velocities is None:
            self.velocities = [np.zeros_like(p) for p in params]
        
        # Update each parameter
        for param, grad, velocity in zip(params, grads, self.velocities):
            # Update velocity: v = β×v + α×grad
            velocity *= self.momentum
            velocity += self.learning_rate * grad
            
            # Update parameter: θ = θ - v
            param -= velocity
    
    def __repr__(self):
        return f"SGDMomentum(lr={self.learning_rate}, momentum={self.momentum})"


class RMSprop(Optimizer):
    """
    RMSprop (Root Mean Square Propagation)
    
    Update rule:
        s = β × s + (1-β) × (∇L)²
        θ = θ - α × ∇L / (√s + ε)
    
    Adapts learning rate for each parameter:
    - Large gradients → smaller effective learning rate
    - Small gradients → larger effective learning rate
    """
    
    def __init__(self, learning_rate=0.001, decay_rate=0.9, epsilon=1e-8):
        """
        Args:
            learning_rate: Learning rate
            decay_rate: Decay rate for moving average (usually 0.9)
            epsilon: Small constant to prevent division by zero
        """
        super().__init__(learning_rate)
        self.decay_rate = decay_rate
        self.epsilon = epsilon
        self.cache = None
    
    def update(self, params, grads):
        """
        Update parameters with RMSprop.
        """
        # Initialize cache on first update
        if self.cache is None:
            self.cache = [np.zeros_like(p) for p in params]
        
        # Update each parameter
        for param, grad, c in zip(params, grads, self.cache):
            # Update cache: s = β×s + (1-β)×grad²
            c *= self.decay_rate
            c += (1 - self.decay_rate) * grad**2
            
            # Update parameter: θ = θ - α×grad / (√s + ε)
            param -= self.learning_rate * grad / (np.sqrt(c) + self.epsilon)
    
    def __repr__(self):
        return f"RMSprop(lr={self.learning_rate}, decay={self.decay_rate})"


class Adam(Optimizer):
    """
    Adam (Adaptive Moment Estimation)
    
    Combines momentum + RMSprop:
    
    Update rule:
        m = β₁ × m + (1-β₁) × ∇L        (first moment - mean)
        v = β₂ × v + (1-β₂) × (∇L)²     (second moment - variance)
        
        m̂ = m / (1 - β₁ᵗ)                (bias correction)
        v̂ = v / (1 - β₂ᵗ)                (bias correction)
        
        θ = θ - α × m̂ / (√v̂ + ε)
    
    Most popular optimizer in practice!
    """
    
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """
        Args:
            learning_rate: Learning rate (often called α or eta)
            beta1: Exponential decay rate for first moment (usually 0.9)
            beta2: Exponential decay rate for second moment (usually 0.999)
            epsilon: Small constant for numerical stability
        """
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        
        # Moments
        self.m = None  # First moment (mean)
        self.v = None  # Second moment (variance)
        
        # Time step
        self.t = 0
    
    def update(self, params, grads):
        """
        Update parameters with Adam.
        """
        # Initialize moments on first update
        if self.m is None:
            self.m = [np.zeros_like(p) for p in params]
            self.v = [np.zeros_like(p) for p in params]
        
        # Increment time step
        self.t += 1
        
        # Update each parameter
        for param, grad, m, v in zip(params, grads, self.m, self.v):
            # Update biased first moment: m = β₁×m + (1-β₁)×grad
            m *= self.beta1
            m += (1 - self.beta1) * grad
            
            # Update biased second moment: v = β₂×v + (1-β₂)×grad²
            v *= self.beta2
            v += (1 - self.beta2) * grad**2
            
            # Bias correction
            m_hat = m / (1 - self.beta1**self.t)
            v_hat = v / (1 - self.beta2**self.t)
            
            # Update parameter: θ = θ - α×m̂/(√v̂ + ε)
            param -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
    
    def __repr__(self):
        return f"Adam(lr={self.learning_rate}, β₁={self.beta1}, β₂={self.beta2})"


class AdamW(Optimizer):
    """
    AdamW (Adam with decoupled Weight Decay)
    
    Adam + L2 regularization done right.
    
    Update rule:
        Same as Adam, but with weight decay applied separately:
        θ = θ - α × (m̂/(√v̂ + ε) + λ×θ)
    
    Better than Adam + L2 regularization.
    """
    
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, 
                 epsilon=1e-8, weight_decay=0.01):
        """
        Args:
            learning_rate: Learning rate
            beta1: First moment decay
            beta2: Second moment decay
            epsilon: Numerical stability
            weight_decay: Weight decay coefficient (L2 penalty)
        """
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        
        self.m = None
        self.v = None
        self.t = 0
    
    def update(self, params, grads):
        """
        Update parameters with AdamW.
        """
        if self.m is None:
            self.m = [np.zeros_like(p) for p in params]
            self.v = [np.zeros_like(p) for p in params]
        
        self.t += 1
        
        for param, grad, m, v in zip(params, grads, self.m, self.v):
            # Update moments (same as Adam)
            m *= self.beta1
            m += (1 - self.beta1) * grad
            
            v *= self.beta2
            v += (1 - self.beta2) * grad**2
            
            # Bias correction
            m_hat = m / (1 - self.beta1**self.t)
            v_hat = v / (1 - self.beta2**self.t)
            
            # Update with decoupled weight decay
            param -= self.learning_rate * (
                m_hat / (np.sqrt(v_hat) + self.epsilon) + 
                self.weight_decay * param
            )
    
    def __repr__(self):
        return f"AdamW(lr={self.learning_rate}, wd={self.weight_decay})"


# Convenience function
def get_optimizer(name, **kwargs):
    """
    Get optimizer by name.
    
    Args:
        name: Optimizer name ('sgd', 'momentum', 'adam', etc.)
        **kwargs: Optimizer-specific parameters
        
    Returns:
        Optimizer instance
        
    Example:
        >>> opt = get_optimizer('adam', learning_rate=0.001)
    """
    optimizers = {
        'sgd': SGD,
        'momentum': SGDMomentum,
        'rmsprop': RMSprop,
        'adam': Adam,
        'adamw': AdamW,
    }
    
    name = name.lower()
    if name not in optimizers:
        raise ValueError(f"Unknown optimizer: {name}. Choose from {list(optimizers.keys())}")
    
    return optimizers[name](**kwargs)