# activations.py

## Why Needed?
Implements various activation functions and their derivatives from scratch.
All functions work with NumPy arrays for vectorized operations.

### **Sigmoid**
- Activation: σ(x) = 1 / (1 + e^(-x))
- Range: (0, 1)
- Used For: Binary classification output layers

**Forward:** 
- Args:
    x: Input array of any shape
            
- Returns:
    Output array of same shape, values in (0, 1)