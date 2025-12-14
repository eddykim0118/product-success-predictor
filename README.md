# Product Success Predictor

A neural network implementation **from scratch** (using only NumPy) to predict product success on e-commerce platforms.

## 🎯 Project Goal

Build a deep learning model from first principles to:
- Understand neural network fundamentals deeply
- Implement forward/backward propagation manually
- Create custom optimizers and loss functions
- Apply to real-world product data

## 🚀 Features

- **Custom Neural Network** - No TensorFlow/PyTorch, built from ground up
- **Multiple Activation Functions** - Sigmoid, ReLU, Tanh, Softmax
- **Various Optimizers** - SGD, Momentum, Adam
- **Regularization** - L2, Dropout
- **Real Dataset** - Amazon product data (2M+ records)

## 📊 Dataset

Amazon Beauty Products dataset containing:
- Product metadata
- Customer reviews and ratings
- Sales rankings
- Price information

## 🛠️ Tech Stack

- **Python 3.14**
- **NumPy** - Matrix operations only
- **Pandas** - Data preprocessing
- **Matplotlib/Seaborn** - Visualization
- **scikit-learn** - Comparison baseline only

## 📁 Project Structure

```
product-success-predictor/
├── data/
│   ├── raw/              # Original datasets
│   └── processed/        # Cleaned data
├── src/
│   ├── neural_net/       # Core NN implementation
│   │   ├── layers.py     # Dense, Activation layers
│   │   ├── activations.py
│   │   ├── losses.py
│   │   ├── optimizers.py
│   │   └── network.py
│   ├── data/             # Data processing
│   └── utils/            # Helper functions
├── notebooks/            # EDA and experiments
├── tests/                # Unit tests
├── examples/             # Usage examples
└── docs/                 # Documentation

```

## 🏃 Getting Started

```bash
# Install dependencies
pip install -r requirements.txt

# Run simple example
python examples/mnist_example.py

# Run product prediction
python examples/product_prediction.py
```

## 📈 Results

Coming soon...

## 📚 Learning Journey

This project implements concepts from:
- Neural Networks and Deep Learning (Michael Nielsen)
- Deep Learning (Goodfellow, Bengio, Courville)
- CS231n Stanford Course

## 📝 License

MIT License

## 👤 Author

Eddy Kim
- Statistics Major with Data Science Emphasis @ BYU
- Machine Learning Engineer @ HEAL USA

---

*Built to understand ML from first principles* 🧠