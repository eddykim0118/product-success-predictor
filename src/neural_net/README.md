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