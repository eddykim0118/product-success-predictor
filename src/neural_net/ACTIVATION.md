# activations.py

## Why Needed?
Implements various activation functions and their derivatives from scratch.
All functions work with NumPy arrays for vectorized operations.

### **Sigmoid**
- **Activation:** σ(x) = 1 / (1 + e^(-x))
- **Range:** (0, 1)
- **Used For:** Binary classification output layers
- **S-Shaped curve**
- **Probability** --> Binary Classificaiton

**Forward:** 
- Args:
    x: Input array of any shape
- Returns:
    Output array of same shape, values in (0, 1)

**Backward:**
- Derivative: σ'(x) = σ(x) * (1 - σ(x))
- Args:
    x: Input array (same as forward pass)          
- Returns:
    Derivative at each point


### **ReLU** (Rectified Linear Unit)
- f(x) = max(0, x)
- Range: [0, ∞)
- Used for: Hidden layers (most common)

**Forward:**
- Args:
    x: Input array
- Returns:
    Output array, negative values set to 0

**Backward:**
- Derivative: f'(x) = 1 if x > 0, else 0
- Args:
    x: Input array
- Returns:
    Gradient: 1 where x > 0, 0 elsewhere


### **Tanh**
- **Hyperbolic tangent:** tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
- **Range:** (-1, 1)
- **Used for:** Hidden layers (centered around 0)

**Forward:**
- Args:
    x: Input array
- Returns:
    Output array, values in (-1, 1)

**Backward:**
- **Derivative:** tanh'(x) = 1 - tanh²(x)
- Args:
    x: Input array
- Returns:
    Gradient at each point


### **LeakyReLU**
- **Leaky ReLU:** f(x) = x if x > 0, else alpha * x
- **Range:** (-∞, ∞)
- **Used for:** Hidden layers (prevents dying ReLU problem)

**__init__:**
- Args:
    alpha: Slope for negative values (default: 0.01)

**Forward:**
- Args:
    x: Input array
- Returns:
    Output array

**Backward:**
- **Derivative:** f'(x) = 1 if x > 0, else alpha
- Args:
    x: Input array
- Returns:
    Gradient


### **Softmax**
- **Softmax:** σ(x)_i = e^(x_i) / Σ(e^(x_j))
- **Range:** (0, 1), sum to 1
- **Used for:** Multi-class classification output

**Forward:**
- Args:
    x: Input array of shape (batch_size, num_classes)
- Returns:
    Probability distribution over classes

**Backward:**
- Note: Softmax derivative is usually computed together with
        cross-entropy loss for efficiency. This is the Jacobian.
        
- Args:
    x: Input array
- Returns:
    Jacobian matrix (not typically used directly)


## get_activation(name)
Get activation function by name.
    
Args:
    name: String name of activation ('sigmoid', 'relu', 'tanh', etc.)
    
Returns:
    Activation instance
    
Example:
    >>> activation = get_activation('relu')
    >>> output = activation.forward(x)