"""
Neural Network
==============

Main Network class that combines layers, loss, and optimizer.
"""

import numpy as np
from tqdm import tqdm
from .layers import Layer
from .losses import get_loss
from .optimizers import get_optimizer


class Network:
    """
    Neural Network class.
    
    Combines layers, loss function, and optimizer into a complete model.
    
    Example:
        >>> model = Network()
        >>> model.add(Dense(784, 128, activation='relu'))
        >>> model.add(Dense(128, 10, activation='softmax'))
        >>> model.compile(optimizer='adam', loss='categorical_crossentropy')
        >>> model.fit(X_train, y_train, epochs=10, batch_size=32)
        >>> predictions = model.predict(X_test)
    """
    
    def __init__(self):
        """Initialize empty network."""
        self.layers = []
        self.loss_function = None
        self.optimizer = None
        self.history = {
            'loss': [],
            'val_loss': []
        }
    
    def add(self, layer):
        """
        Add a layer to the network.
        
        Args:
            layer: Layer instance (Dense, Dropout, etc.)
        """
        if not isinstance(layer, Layer):
            raise TypeError(f"Expected Layer instance, got {type(layer)}")
        
        self.layers.append(layer)
    
    def compile(self, optimizer='adam', loss='mse', learning_rate=0.001, **optimizer_kwargs):
        """
        Configure the network for training.
        
        Args:
            optimizer: Optimizer name or instance
            loss: Loss function name or instance
            learning_rate: Learning rate for optimizer
            **optimizer_kwargs: Additional optimizer parameters
        """
        # Set up loss function
        if isinstance(loss, str):
            self.loss_function = get_loss(loss)
        else:
            self.loss_function = loss
        
        # Set up optimizer
        if isinstance(optimizer, str):
            self.optimizer = get_optimizer(
                optimizer, 
                learning_rate=learning_rate,
                **optimizer_kwargs
            )
        else:
            self.optimizer = optimizer
        
        print(f"Model compiled with {self.optimizer} and {self.loss_function.__class__.__name__}")
    
    def forward(self, x, training=True):
        """
        Forward pass through all layers.
        
        Args:
            x: Input data
            training: Whether in training mode (affects Dropout, BatchNorm)
            
        Returns:
            Output of final layer
        """
        output = x
        for layer in self.layers:
            output = layer.forward(output, training=training)
        return output
    
    def backward(self, loss_grad):
        """
        Backward pass through all layers.
        
        Args:
            loss_grad: Gradient of loss w.r.t network output
        """
        grad = loss_grad
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
    
    def get_params_and_grads(self):
        """
        Get all parameters and gradients from all layers.
        
        Returns:
            params: List of all parameter arrays
            grads: List of all gradient arrays
        """
        params = []
        grads = []
        
        for layer in self.layers:
            params.extend(layer.get_params())
            grads.extend(layer.get_grads())
        
        return params, grads
    
    def train_on_batch(self, x_batch, y_batch):
        """
        Train on a single batch.
        
        Args:
            x_batch: Input batch
            y_batch: Target batch
            
        Returns:
            Loss value for this batch
        """
        # Forward pass
        output = self.forward(x_batch, training=True)
        
        # Compute loss
        loss = self.loss_function.forward(y_batch, output)
        
        # Backward pass
        loss_grad = self.loss_function.backward(y_batch, output)
        self.backward(loss_grad)
        
        # Update parameters
        params, grads = self.get_params_and_grads()
        self.optimizer.update(params, grads)
        
        return loss
    
    def fit(self, X, y, epochs=10, batch_size=32, validation_data=None, verbose=1):
        """
        Train the network.
        
        Args:
            X: Training data, shape (num_samples, input_dim)
            y: Training labels, shape (num_samples, output_dim)
            epochs: Number of epochs to train
            batch_size: Batch size
            validation_data: Optional (X_val, y_val) tuple
            verbose: 0 = silent, 1 = progress bar, 2 = one line per epoch
            
        Returns:
            Training history
        """
        if self.optimizer is None or self.loss_function is None:
            raise ValueError("Model must be compiled before training. Call model.compile() first.")
        
        num_samples = X.shape[0]
        num_batches = int(np.ceil(num_samples / batch_size))
        
        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(num_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            # Training
            epoch_losses = []
            
            if verbose == 1:
                pbar = tqdm(range(num_batches), desc=f'Epoch {epoch+1}/{epochs}')
            else:
                pbar = range(num_batches)
            
            for batch_idx in pbar:
                # Get batch
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, num_samples)
                
                x_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]
                
                # Train on batch
                loss = self.train_on_batch(x_batch, y_batch)
                epoch_losses.append(loss)
                
                if verbose == 1:
                    pbar.set_postfix({'loss': f'{np.mean(epoch_losses):.4f}'})
            
            # Record training loss
            avg_loss = np.mean(epoch_losses)
            self.history['loss'].append(avg_loss)
            
            # Validation
            val_loss_str = ""
            if validation_data is not None:
                X_val, y_val = validation_data
                val_output = self.predict(X_val)
                val_loss = self.loss_function.forward(y_val, val_output)
                self.history['val_loss'].append(val_loss)
                val_loss_str = f", val_loss: {val_loss:.4f}"
            
            if verbose == 2:
                print(f"Epoch {epoch+1}/{epochs} - loss: {avg_loss:.4f}{val_loss_str}")
        
        return self.history
    
    def predict(self, X):
        """
        Make predictions.
        
        Args:
            X: Input data
            
        Returns:
            Predictions
        """
        return self.forward(X, training=False)
    
    def evaluate(self, X, y):
        """
        Evaluate the model.
        
        Args:
            X: Input data
            y: True labels
            
        Returns:
            Loss value
        """
        predictions = self.predict(X)
        loss = self.loss_function.forward(y, predictions)
        return loss
    
    def save_weights(self, filepath):
        """
        Save model weights to file.
        
        Args:
            filepath: Path to save weights
        """
        weights = {}
        for i, layer in enumerate(self.layers):
            params = layer.get_params()
            if params:
                weights[f'layer_{i}'] = params
        
        np.save(filepath, weights)
        print(f"Weights saved to {filepath}")
    
    def load_weights(self, filepath):
        """
        Load model weights from file.
        
        Args:
            filepath: Path to load weights from
        """
        weights = np.load(filepath, allow_pickle=True).item()
        
        for i, layer in enumerate(self.layers):
            if f'layer_{i}' in weights:
                params = weights[f'layer_{i}']
                layer_params = layer.get_params()
                
                for j, param in enumerate(layer_params):
                    param[:] = params[j]
        
        print(f"Weights loaded from {filepath}")
    
    def summary(self):
        """
        Print model summary.
        """
        print("=" * 70)
        print(f"{'Layer (type)':<30} {'Output Shape':<20} {'Param #':<15}")
        print("=" * 70)
        
        total_params = 0
        
        for i, layer in enumerate(self.layers):
            params = layer.get_params()
            num_params = sum(p.size for p in params)
            total_params += num_params
            
            # Get output shape (approximation)
            layer_name = f"{layer.__class__.__name__}_{i}"
            output_shape = "?"
            
            if hasattr(layer, 'output_dim'):
                output_shape = f"(None, {layer.output_dim})"
            
            print(f"{layer_name:<30} {output_shape:<20} {num_params:<15,}")
        
        print("=" * 70)
        print(f"Total params: {total_params:,}")
        print(f"Trainable params: {total_params:,}")
        print(f"Non-trainable params: 0")
        print("=" * 70)
    
    def __repr__(self):
        return f"Network({len(self.layers)} layers)"