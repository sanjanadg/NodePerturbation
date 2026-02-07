import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from danp import DANPMLP
from bp import BPMLP
from wp import WPMLP
import time
import numpy as np
import matplotlib.pyplot as plt
import os
import json

class SyntheticDataset(Dataset):
    """
    Creates a fixed synthetic dataset by generating data from a teacher network.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, n_samples=10000, seed=42, teacher_layer_sizes=None):
        """
        Args:
            input_dim: dimension of input
            hidden_dim: hidden layer size for teacher network (used if teacher_layer_sizes is None)
            output_dim: dimension of output
            n_samples: number of samples to generate
            seed: random seed for reproducibility
            teacher_layer_sizes: optional list of layer sizes for teacher network (e.g., [10, 128, 128, 1])
                                If None, uses simple 2-layer network [input_dim, hidden_dim, output_dim]
        """
        torch.manual_seed(seed)
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_samples = n_samples
        
        # Determine teacher network architecture
        if teacher_layer_sizes is not None:
            self.teacher_layer_sizes = teacher_layer_sizes
        else:
            # Default: simple 2-layer network
            self.teacher_layer_sizes = [input_dim, hidden_dim, output_dim]
        
        # Create teacher network weights (fixed)
        self.W = []
        for i in range(len(self.teacher_layer_sizes) - 1):
            fan_in = self.teacher_layer_sizes[i]
            fan_out = self.teacher_layer_sizes[i + 1]
            W = torch.randn(fan_out, fan_in) / fan_in**0.5
            self.W.append(W)
        
        # Generate fixed dataset
        self.inputs = torch.randn(n_samples, input_dim)
        self.targets = self._generate_targets(self.inputs)
        
    def _generate_targets(self, x):
        """Generate targets using teacher network"""
        # x shape: (n_samples, input_dim) or (input_dim,)
        is_batch = x.dim() > 1
        
        # Forward pass through teacher network
        h = x
        for i, W in enumerate(self.W):
            if is_batch:
                h = h @ W.T
            else:
                h = W @ h
            
            # Apply ReLU to all layers except output
            if i < len(self.W) - 1:
                h = torch.relu(h)
        
        return h
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

# TRAINING EXAMPLES WITH DATASETS

def train_bp_model(layer_sizes, dataset, n_epochs=50, lr=1e-3, batch_size=1, 
                   train_split=0.8, seed=42, device='cpu'):
    """Train BP model and return final train and test losses, with gradient norms as delta loss proxy."""
    torch.manual_seed(seed + 1000)
    
    # Split dataset
    train_size = int(train_split * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size],
        generator=torch.Generator().manual_seed(seed)
    )

    # debugging 
    print(f"BP batch size: {batch_size}")
    print(f"Samples per epoch BP: {len(train_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Create model
    model = BPMLP(layer_sizes).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    
    # Training loop
    iteration = 0
    train_losses_per_epoch = []
    test_losses_per_epoch = []
    delta_losses_per_epoch = []  # Gradient norms as proxy
    
    for epoch in range(n_epochs):
        epoch_start_time = time.time()
        model.train()
        
        epoch_grad_norms = []
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = F.mse_loss(pred, y)
            loss.backward()
            
            # Compute gradient norm as proxy for delta loss
            total_grad_norm = 0.0
            for param in model.parameters():
                if param.grad is not None:
                    total_grad_norm += param.grad.data.norm(2).item() ** 2
            epoch_grad_norms.append(total_grad_norm ** 0.5)
            
            optimizer.step()
            iteration += 1
        
        # Evaluate on training set (after all updates)
        model.eval()
        train_losses = []
        with torch.no_grad():
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                loss = F.mse_loss(pred, y)
                train_losses.append(loss.item())
        
        # Evaluate on test set
        test_losses = []
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                loss = F.mse_loss(pred, y)
                test_losses.append(loss.item())
        
        epoch_time = time.time() - epoch_start_time
        avg_train = np.mean(train_losses)
        avg_test = np.mean(test_losses)
        avg_delta = np.mean(epoch_grad_norms) if epoch_grad_norms else 0.0
        
        # Store per-epoch losses
        train_losses_per_epoch.append(avg_train)
        test_losses_per_epoch.append(avg_test)
        delta_losses_per_epoch.append(avg_delta)
        
        # Print epoch progress
        print(f"  Epoch {epoch+1:3d}/{n_epochs} | Train Loss: {avg_train:.6f} | Test Loss: {avg_test:.6f} | Delta: {avg_delta:.6f} | Time: {epoch_time:.3f}s")
    
    final_train_loss = train_losses_per_epoch[-1]
    final_test_loss = test_losses_per_epoch[-1]
    final_delta_loss = delta_losses_per_epoch[-1]
    
    # Final summary
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    print(f"Final Train Loss: {final_train_loss:.6f}")
    print(f"Final Test Loss:  {final_test_loss:.6f}")
    print(f"Final Delta Loss: {final_delta_loss:.6f}")
    print("="*80 + "\n")
    
    return {
        'train_losses': train_losses_per_epoch,
        'test_losses': test_losses_per_epoch,
        'delta_losses': delta_losses_per_epoch,
        'final_train': final_train_loss,
        'final_test': final_test_loss,
        'final_delta': final_delta_loss
    }

def train_danp_model(layer_sizes, dataset, n_epochs=50, eta=1e-3, alpha=1e-4, 
                     sigma=0.001, batch_size=1, train_split=0.8, seed=42, device='cpu'):
    """Train DANP model and return final train and test losses."""
    torch.manual_seed(seed + 1000)  # Ensure model initialization is different from dataset generation
    
    # Split dataset
    train_size = int(train_split * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size],
        generator=torch.Generator().manual_seed(seed)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Create model
    model = DANPMLP(layer_sizes, sigma=sigma, eta=eta, alpha=alpha)
    # Move model parameters to device
    for i in range(len(model.W)):
        model.W[i] = model.W[i].to(device)
    for i in range(len(model.R)):
        model.R[i] = model.R[i].to(device)
    
    # Training loop
    train_losses_per_epoch = []
    test_losses_per_epoch = []
    
    for epoch in range(n_epochs):
        epoch_start_time = time.time()
        
        # Training updates
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            # Process each sample in the batch individually
            for i in range(x.shape[0]):
                x_sample = x[i]
                y_sample = y[i]
                model.step(x_sample, y_sample)
        
        # Evaluate on training set (after all updates)
        train_losses = []
        with torch.no_grad():
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                # Process each sample in the batch individually
                for i in range(x.shape[0]):
                    x_sample = x[i]
                    y_sample = y[i]
                    x_clean, _ = model.forward_clean(x_sample)
                    loss = F.mse_loss(x_clean[-1], y_sample).item()
                    train_losses.append(loss)
        
        # Evaluate on test set
        test_losses = []
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                # Process each sample in the batch individually
                for i in range(x.shape[0]):
                    x_sample = x[i]
                    y_sample = y[i]
                    x_clean, _ = model.forward_clean(x_sample)
                    loss = F.mse_loss(x_clean[-1], y_sample).item()
                    test_losses.append(loss)
        
        epoch_time = time.time() - epoch_start_time
        avg_train = np.mean(train_losses)
        avg_test = np.mean(test_losses)
        
        # Store per-epoch losses
        train_losses_per_epoch.append(avg_train)
        test_losses_per_epoch.append(avg_test)
        
        # Print epoch progress
        print(f"  Epoch {epoch+1:3d}/{n_epochs} | Train Loss: {avg_train:.6f} | Test Loss: {avg_test:.6f} | Time: {epoch_time:.3f}s")

    final_train_loss = train_losses_per_epoch[-1]
    final_test_loss = test_losses_per_epoch[-1]
    
    # Final summary
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    print(f"Final Train Loss: {final_train_loss:.6f}")
    print(f"Final Test Loss:  {final_test_loss:.6f}")
    print("="*80 + "\n")
    
    return {
        'train_losses': train_losses_per_epoch,
        'test_losses': test_losses_per_epoch,
        'final_train': final_train_loss,
        'final_test': final_test_loss
    }


def train_danp_batch(model, x_batch, y_batch):
    """
    Train DANP model on a batch by accumulating gradient estimates and applying averaged updates.
    
    Args:
        model: DANPMLP model instance
        x_batch: batch of inputs, shape (batch_size, input_dim)
        y_batch: batch of targets, shape (batch_size, output_dim)
    
    Returns:
        average_loss: average clean loss over the batch
        average_delta_L: average loss difference (noisy - clean) over the batch
    """
    batch_size = x_batch.shape[0]
    
    # Initialize accumulated updates
    accumulated_weight_updates = [torch.zeros_like(W) for W in model.W]
    accumulated_decorrelation_updates = [torch.zeros_like(R) for R in model.R[1:]]  # Skip R[0]
    
    total_loss = 0.0
    total_delta_L = 0.0
    
    # Compute gradient estimate for each sample in the batch
    for i in range(batch_size):
        x_sample = x_batch[i]
        y_sample = y_batch[i]
        
        # Clean and noisy forward passes to get delta_L
        x_clean, a_clean = model.forward_clean(x_sample)
        x_noisy, a_noisy = model.forward_noisy(x_sample)
        L_clean = F.mse_loss(x_clean[-1], y_sample).item()
        L_noisy = F.mse_loss(x_noisy[-1], y_sample).item()
        delta_L = abs(L_noisy - L_clean)
        total_delta_L += delta_L
        
        # Get gradient estimate (but don't update yet)
        weight_grads, decorrelation_grads, loss_clean = model.compute_grad_estimate(x_sample, y_sample)
        
        # Accumulate weight updates
        for j, grad in enumerate(weight_grads):
            accumulated_weight_updates[j] += grad
        
        # Accumulate decorrelation updates
        for j, grad in enumerate(decorrelation_grads):
            accumulated_decorrelation_updates[j] += grad
        
        total_loss += loss_clean
    
    # Average and apply updates once
    for j in range(len(model.W)):
        model.W[j] -= accumulated_weight_updates[j] / batch_size
    
    # Apply averaged decorrelation updates
    for j in range(len(accumulated_decorrelation_updates)):
        model.R[j + 1] -= accumulated_decorrelation_updates[j] / batch_size  # +1 because R[0] is input layer
    
    return total_loss / batch_size, total_delta_L / batch_size


def train_danp_model_batch(layer_sizes, dataset, n_epochs=50, eta=1e-3, alpha=1e-4, 
                           sigma=0.001, batch_size=1, train_split=0.8, seed=42, device='cpu'):
    """
    Train DANP model with proper batch processing using train_danp_batch.
    Returns final train and test losses, and per-epoch losses with delta losses.
    """
    torch.manual_seed(seed + 1000)
    
    # Split dataset
    train_size = int(train_split * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size],
        generator=torch.Generator().manual_seed(seed)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Create model with scaled learning rate for batch training
    # Scale eta by batch_size to maintain same effective learning rate per sample
    eta_scaled = eta * batch_size
    model = DANPMLP(layer_sizes, sigma=sigma, eta=eta_scaled, alpha=alpha)
    # Move model parameters to device
    for i in range(len(model.W)):
        model.W[i] = model.W[i].to(device)
    for i in range(len(model.R)):
        model.R[i] = model.R[i].to(device)
    
    # Training loop
    train_losses_per_epoch = []
    test_losses_per_epoch = []
    delta_losses_per_epoch = []
    
    for epoch in range(n_epochs):
        epoch_start_time = time.time()
        epoch_delta_losses = []
        
        # Training updates using batch processing
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            _, delta_L = train_danp_batch(model, x, y)
            epoch_delta_losses.append(delta_L)
        
        # Evaluate on training set (after all updates)
        train_losses = []
        with torch.no_grad():
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                # Process each sample in the batch individually for evaluation
                for i in range(x.shape[0]):
                    x_sample = x[i]
                    y_sample = y[i]
                    x_clean, _ = model.forward_clean(x_sample)
                    loss = F.mse_loss(x_clean[-1], y_sample).item()
                    train_losses.append(loss)
        
        # Evaluate on test set
        test_losses = []
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                # Process each sample in the batch individually for evaluation
                for i in range(x.shape[0]):
                    x_sample = x[i]
                    y_sample = y[i]
                    x_clean, _ = model.forward_clean(x_sample)
                    loss = F.mse_loss(x_clean[-1], y_sample).item()
                    test_losses.append(loss)
        
        epoch_time = time.time() - epoch_start_time
        avg_train = np.mean(train_losses)
        avg_test = np.mean(test_losses)
        avg_delta = np.mean(epoch_delta_losses)
        
        # Store per-epoch losses
        train_losses_per_epoch.append(avg_train)
        test_losses_per_epoch.append(avg_test)
        delta_losses_per_epoch.append(avg_delta)
        
        # Print epoch progress
        print(f"  Epoch {epoch+1:3d}/{n_epochs} | Train Loss: {avg_train:.6f} | Test Loss: {avg_test:.6f} | Delta Loss: {avg_delta:.6f} | Time: {epoch_time:.3f}s")

    final_train_loss = train_losses_per_epoch[-1]
    final_test_loss = test_losses_per_epoch[-1]
    final_delta_loss = delta_losses_per_epoch[-1]
    
    # Final summary
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    print(f"Final Train Loss: {final_train_loss:.6f}")
    print(f"Final Test Loss:  {final_test_loss:.6f}")
    print(f"Final Delta Loss: {final_delta_loss:.6f}")
    print("="*80 + "\n")
    
    return {
        'train_losses': train_losses_per_epoch,
        'test_losses': test_losses_per_epoch,
        'delta_losses': delta_losses_per_epoch,
        'final_train': final_train_loss,
        'final_test': final_test_loss,
        'final_delta': final_delta_loss
    }


def train_wp_model(layer_sizes, dataset, n_epochs=50, sigma=0.001, alpha=1e-3, 
                   batch_size=1, population_size=1, train_split=0.8, seed=42, device='cpu'):
    """
    Train WP (Weight Perturbation) model and return final train and test losses.
    
    Args:
        layer_sizes: list of layer dimensions [input_dim, hidden1, ..., output_dim]
        dataset: SyntheticDataset instance
        n_epochs: number of training epochs
        sigma: noise standard deviation for weight perturbations
        alpha: learning rate for weight updates
        batch_size: batch size for processing samples (currently processes sequentially)
        population_size: number of perturbations per sample (1 = single perturbation, >1 = population-based)
        train_split: fraction of data for training
        seed: random seed
        device: device to use
    
    Returns:
        dict with 'train_losses', 'test_losses', 'final_train', 'final_test'
    """
    torch.manual_seed(seed + 1000)  # Ensure model initialization is different from dataset generation
    
    # Split dataset
    train_size = int(train_split * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size],
        generator=torch.Generator().manual_seed(seed)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Create model
    model = WPMLP(layer_sizes, sigma=sigma, alpha=alpha)
    # Move model parameters to device
    for i in range(len(model.W)):
        model.W[i] = model.W[i].to(device)
    
    # Training loop
    train_losses_per_epoch = []
    test_losses_per_epoch = []
    
    for epoch in range(n_epochs):
        epoch_start_time = time.time()
        
        # Training updates
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            # Process each sample in the batch individually
            for i in range(x.shape[0]):
                x_sample = x[i]
                y_sample = y[i]
                
                if population_size > 1:
                    # Use population-based update
                    model.step_population(x_sample, y_sample, population_size=population_size)
                else:
                    # Use single perturbation update
                    model.step(x_sample, y_sample)
        
        # Evaluate on training set (after all updates)
        train_losses = []
        with torch.no_grad():
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                # Process each sample in the batch individually
                for i in range(x.shape[0]):
                    x_sample = x[i]
                    y_sample = y[i]
                    x_clean = model.forward(x_sample)
                    loss = F.mse_loss(x_clean[-1], y_sample).item()
                    train_losses.append(loss)
        
        # Evaluate on test set
        test_losses = []
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                # Process each sample in the batch individually
                for i in range(x.shape[0]):
                    x_sample = x[i]
                    y_sample = y[i]
                    x_clean = model.forward(x_sample)
                    loss = F.mse_loss(x_clean[-1], y_sample).item()
                    test_losses.append(loss)
        
        epoch_time = time.time() - epoch_start_time
        avg_train = np.mean(train_losses)
        avg_test = np.mean(test_losses)
        
        # Store per-epoch losses
        train_losses_per_epoch.append(avg_train)
        test_losses_per_epoch.append(avg_test)
        
        # Print epoch progress
        print(f"  Epoch {epoch+1:3d}/{n_epochs} | Train Loss: {avg_train:.6f} | Test Loss: {avg_test:.6f} | Time: {epoch_time:.3f}s")

    final_train_loss = train_losses_per_epoch[-1]
    final_test_loss = test_losses_per_epoch[-1]
    
    # Final summary
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    print(f"Final Train Loss: {final_train_loss:.6f}")
    print(f"Final Test Loss:  {final_test_loss:.6f}")
    print("="*80 + "\n")
    
    return {
        'train_losses': train_losses_per_epoch,
        'test_losses': test_losses_per_epoch,
        'final_train': final_train_loss,
        'final_test': final_test_loss
    }


def train_wp_batch(model, x_batch, y_batch):
    """
    Train WP model on a batch by accumulating gradient estimates and applying averaged updates.
    
    Args:
        model: WPMLP model instance
        x_batch: batch of inputs, shape (batch_size, input_dim)
        y_batch: batch of targets, shape (batch_size, output_dim)
    
    Returns:
        average_loss: average clean loss over the batch
        average_delta_L: average loss difference (noisy - clean) over the batch
    """
    batch_size = x_batch.shape[0]
    
    # Initialize accumulated updates
    accumulated_weight_updates = [torch.zeros_like(W) for W in model.W]
    
    total_loss = 0.0
    total_delta_L = 0.0
    
    # Compute gradient estimate for each sample in the batch
    for i in range(batch_size):
        x_sample = x_batch[i]
        y_sample = y_batch[i]
        
        # Forward pass with clean weights
        x_clean = model.forward(x_sample)
        loss_clean = F.mse_loss(x_clean[-1], y_sample).item()
        
        # Generate noise for each weight matrix
        noise_list = []
        for l in range(model.L):
            noise = torch.randn_like(model.W[l])
            noise_list.append(noise)
        
        # Forward pass with perturbed weights
        x_noisy = model.forward_with_perturbed_weights(x_sample, noise_list)
        loss_noisy = F.mse_loss(x_noisy[-1], y_sample).item()
        
        # Compute loss difference
        delta_L = abs(loss_noisy - loss_clean)
        total_delta_L += delta_L
        
        # Compute weight gradients: gradient = -alpha * delta_L * noise
        for l in range(model.L):
            grad = -model.alpha * delta_L * noise_list[l]
            grad = torch.clamp(grad, -1.0, 1.0)
            accumulated_weight_updates[l] += grad
        
        total_loss += loss_clean
    
    # Average and apply updates once
    for j in range(len(model.W)):
        model.W[j] += accumulated_weight_updates[j] / batch_size
    
    return total_loss / batch_size, total_delta_L / batch_size


def train_wp_model_batch(layer_sizes, dataset, n_epochs=50, sigma=0.001, alpha=1e-3,
                        batch_size=1, train_split=0.8, seed=42, device='cpu'):
    """
    Train WP model with proper batch processing using train_wp_batch.
    Returns final train and test losses, and per-epoch losses with delta losses.
    """
    torch.manual_seed(seed + 1000)
    
    # Split dataset
    train_size = int(train_split * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size],
        generator=torch.Generator().manual_seed(seed)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Create model with scaled learning rate for batch training
    # Scale alpha by batch_size to maintain same effective learning rate per sample
    alpha_scaled = alpha * batch_size
    model = WPMLP(layer_sizes, sigma=sigma, alpha=alpha_scaled)
    # Move model parameters to device
    for i in range(len(model.W)):
        model.W[i] = model.W[i].to(device)
    
    # Training loop
    train_losses_per_epoch = []
    test_losses_per_epoch = []
    delta_losses_per_epoch = []
    
    for epoch in range(n_epochs):
        epoch_start_time = time.time()
        epoch_delta_losses = []
        
        # Training updates using batch processing
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            _, delta_L = train_wp_batch(model, x, y)
            epoch_delta_losses.append(delta_L)
        
        # Evaluate on training set (after all updates)
        train_losses = []
        with torch.no_grad():
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                for i in range(x.shape[0]):
                    x_sample = x[i]
                    y_sample = y[i]
                    x_clean = model.forward(x_sample)
                    loss = F.mse_loss(x_clean[-1], y_sample).item()
                    train_losses.append(loss)
        
        # Evaluate on test set
        test_losses = []
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                for i in range(x.shape[0]):
                    x_sample = x[i]
                    y_sample = y[i]
                    x_clean = model.forward(x_sample)
                    loss = F.mse_loss(x_clean[-1], y_sample).item()
                    test_losses.append(loss)
        
        epoch_time = time.time() - epoch_start_time
        avg_train = np.mean(train_losses)
        avg_test = np.mean(test_losses)
        avg_delta = np.mean(epoch_delta_losses)
        
        # Store per-epoch losses
        train_losses_per_epoch.append(avg_train)
        test_losses_per_epoch.append(avg_test)
        delta_losses_per_epoch.append(avg_delta)
        
        # Print epoch progress
        print(f"  Epoch {epoch+1:3d}/{n_epochs} | Train Loss: {avg_train:.6f} | Test Loss: {avg_test:.6f} | Delta Loss: {avg_delta:.6f} | Time: {epoch_time:.3f}s")
    
    final_train_loss = train_losses_per_epoch[-1]
    final_test_loss = test_losses_per_epoch[-1]
    final_delta_loss = delta_losses_per_epoch[-1]

    # Final summary
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    print(f"Final Train Loss: {final_train_loss:.6f}")
    print(f"Final Test Loss:  {final_test_loss:.6f}")
    print(f"Final Delta Loss: {final_delta_loss:.6f}")
    print("="*80 + "\n")
    
    return {
        'train_losses': train_losses_per_epoch,
        'test_losses': test_losses_per_epoch,
        'delta_losses': delta_losses_per_epoch,
        'final_train': final_train_loss,
        'final_test': final_test_loss,
        'final_delta': final_delta_loss
    }


def run_batch_size_experiment(layer_sizes=[10, 32, 1], n_samples=5000, n_epochs=50,
                              base_lr=1e-3, base_eta=1e-3, base_alpha=1e-3,
                              alpha_danp=1e-5, sigma=0.001, batch_sizes=None,
                              train_split=0.8, seed=42, device='cpu',
                              output_dir='./batch_size_experiment_results'):
    """
    Run batch size experiment comparing BP, DANP, and WP with linearly scaled learning rates.
    
    Args:
        layer_sizes: model architecture
        n_samples: dataset size
        n_epochs: number of training epochs
        base_lr: base learning rate for BP (will be scaled by batch_size)
        base_eta: base learning rate for DANP (will be scaled by batch_size)
        base_alpha: base learning rate for WP (will be scaled by batch_size)
        alpha_danp: decorrelation learning rate for DANP
        sigma: noise std for DANP/WP
        batch_sizes: list of batch sizes to test (default: [2, 4, 8, 16, 32])
        train_split: fraction of data for training
        seed: random seed
        device: device to use
        output_dir: directory to save results
    
    Returns:
        results dictionary with all experiment data
    """
    if batch_sizes is None:
        batch_sizes = [2, 4, 8, 16, 32]
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create dataset
    print("="*80)
    print("BATCH SIZE EXPERIMENT")
    print("="*80)
    print(f"Architecture: {layer_sizes}")
    print(f"Batch sizes to test: {batch_sizes}")
    print(f"Base learning rates - BP: {base_lr}, DANP: {base_eta}, WP: {base_alpha}")
    print("="*80)
    
    dataset = SyntheticDataset(
        input_dim=layer_sizes[0],
        hidden_dim=layer_sizes[1] if len(layer_sizes) > 2 else 32,
        output_dim=layer_sizes[-1],
        n_samples=n_samples,
        seed=seed,
        teacher_layer_sizes=layer_sizes
    )
    
    results = {
        'config': {
            'layer_sizes': layer_sizes,
            'n_samples': n_samples,
            'n_epochs': n_epochs,
            'base_lr': base_lr,
            'base_eta': base_eta,
            'base_alpha': base_alpha,
            'alpha_danp': alpha_danp,
            'sigma': sigma,
            'batch_sizes': batch_sizes,
            'seed': seed
        },
        'experiments': {}
    }
    
    # Run experiments for each batch size
    for batch_size in batch_sizes:
        print(f"\n{'='*80}")
        print(f"BATCH SIZE: {batch_size}")
        print(f"{'='*80}")
        
        # Scale learning rates linearly with batch size
        lr = base_lr * batch_size
        eta_scaled = base_eta * batch_size
        alpha_wp_scaled = base_alpha * batch_size
        
        print(f"Scaled learning rates - BP lr: {lr:.6f}, DANP eta: {eta_scaled:.6f}, WP alpha: {alpha_wp_scaled:.6f}")
        
        # Train BP
        print(f"\n--- Training BP (batch_size={batch_size}) ---")
        results_bp = train_bp_model(
            layer_sizes=layer_sizes,
            dataset=dataset,
            n_epochs=n_epochs,
            lr=lr,
            batch_size=batch_size,
            train_split=train_split,
            seed=seed,
            device=device
        )
        
        # Train DANP (eta will be scaled inside train_danp_model_batch, so pass base_eta)
        print(f"\n--- Training DANP (batch_size={batch_size}) ---")
        results_danp = train_danp_model_batch(
            layer_sizes=layer_sizes,
            dataset=dataset,
            n_epochs=n_epochs,
            eta=base_eta,  # Will be scaled by batch_size inside the function
            alpha=alpha_danp,
            sigma=sigma,
            batch_size=batch_size,
            train_split=train_split,
            seed=seed,
            device=device
        )
        
        # Train WP (alpha will be scaled inside train_wp_model_batch, so pass base_alpha)
        print(f"\n--- Training WP (batch_size={batch_size}) ---")
        results_wp = train_wp_model_batch(
            layer_sizes=layer_sizes,
            dataset=dataset,
            n_epochs=n_epochs,
            sigma=sigma,
            alpha=base_alpha,  # Will be scaled by batch_size inside the function
            batch_size=batch_size,
            train_split=train_split,
            seed=seed,
            device=device
        )
        
        results['experiments'][f'batch_{batch_size}'] = {
            'batch_size': batch_size,
            'bp': results_bp,
            'danp': results_danp,
            'wp': results_wp
        }
    
    # Create visualizations
    print(f"\n{'='*80}")
    print("Creating visualizations...")
    print(f"{'='*80}")
    plot_batch_size_results(results, output_dir)
    
    # Save results
    results_file = os.path.join(output_dir, 'batch_size_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {results_file}")
    
    return results


def plot_batch_size_results(results, output_dir):
    """
    Create three visualizations:
    1. Delta loss vs batch size for each algorithm
    2. Train/test loss vs batch size
    3. Train/test loss/delta loss vs iterations/epochs (for each batch size)
    """
    batch_sizes = results['config']['batch_sizes']
    n_epochs = results['config']['n_epochs']
    
    # Extract data
    bp_final_deltas = []
    danp_final_deltas = []
    wp_final_deltas = []
    
    bp_final_train = []
    bp_final_test = []
    danp_final_train = []
    danp_final_test = []
    wp_final_train = []
    wp_final_test = []
    
    for bs in batch_sizes:
        exp = results['experiments'][f'batch_{bs}']
        bp_final_deltas.append(exp['bp']['final_delta'])
        danp_final_deltas.append(exp['danp']['final_delta'])
        wp_final_deltas.append(exp['wp']['final_delta'])
        
        bp_final_train.append(exp['bp']['final_train'])
        bp_final_test.append(exp['bp']['final_test'])
        danp_final_train.append(exp['danp']['final_train'])
        danp_final_test.append(exp['danp']['final_test'])
        wp_final_train.append(exp['wp']['final_train'])
        wp_final_test.append(exp['wp']['final_test'])
    
    # Plot 1: Delta loss vs batch size
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    ax = axes[0]
    ax.plot(batch_sizes, bp_final_deltas, 'o-', label='BP (grad norm)', linewidth=2, markersize=8)
    ax.plot(batch_sizes, danp_final_deltas, 's-', label='DANP', linewidth=2, markersize=8)
    ax.plot(batch_sizes, wp_final_deltas, '^-', label='WP', linewidth=2, markersize=8)
    ax.set_xlabel('Batch Size', fontsize=12)
    ax.set_ylabel('Delta Loss', fontsize=12)
    ax.set_title('Delta Loss vs Batch Size', fontsize=14)
    ax.legend(fontsize=11)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(batch_sizes)
    
    # Plot 2: Train/Test loss vs batch size
    ax = axes[1]
    ax.plot(batch_sizes, bp_final_train, 'o--', label='BP Train', linewidth=2, markersize=8, alpha=0.7)
    ax.plot(batch_sizes, bp_final_test, 'o-', label='BP Test', linewidth=2, markersize=8)
    ax.plot(batch_sizes, danp_final_train, 's--', label='DANP Train', linewidth=2, markersize=8, alpha=0.7)
    ax.plot(batch_sizes, danp_final_test, 's-', label='DANP Test', linewidth=2, markersize=8)
    ax.plot(batch_sizes, wp_final_train, '^--', label='WP Train', linewidth=2, markersize=8, alpha=0.7)
    ax.plot(batch_sizes, wp_final_test, '^-', label='WP Test', linewidth=2, markersize=8)
    ax.set_xlabel('Batch Size', fontsize=12)
    ax.set_ylabel('Final Loss', fontsize=12)
    ax.set_title('Final Loss vs Batch Size', fontsize=14)
    ax.legend(fontsize=10, ncol=2)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(batch_sizes)
    
    # Plot 3: Loss curves vs epochs for each batch size (subplot for each batch size)
    ax = axes[2]
    # Show one example batch size (largest) for loss curves
    example_bs = batch_sizes[-1]
    exp = results['experiments'][f'batch_{example_bs}']
    epochs = range(1, n_epochs + 1)
    
    ax.plot(epochs, exp['bp']['train_losses'], '--', label='BP Train', linewidth=2, alpha=0.7, color='blue')
    ax.plot(epochs, exp['bp']['test_losses'], '-', label='BP Test', linewidth=2, color='blue')
    ax.plot(epochs, exp['danp']['train_losses'], '--', label='DANP Train', linewidth=2, alpha=0.7, color='red')
    ax.plot(epochs, exp['danp']['test_losses'], '-', label='DANP Test', linewidth=2, color='red')
    ax.plot(epochs, exp['wp']['train_losses'], '--', label='WP Train', linewidth=2, alpha=0.7, color='green')
    ax.plot(epochs, exp['wp']['test_losses'], '-', label='WP Test', linewidth=2, color='green')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title(f'Loss Curves (Batch Size={example_bs})', fontsize=14)
    ax.legend(fontsize=9, ncol=2)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_file = os.path.join(output_dir, 'batch_size_comparison.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Plot 1-3 saved to {plot_file}")
    plt.close()
    
    # Plot 4: Delta loss vs epochs for each batch size
    fig, axes = plt.subplots(1, len(batch_sizes), figsize=(5*len(batch_sizes), 5))
    if len(batch_sizes) == 1:
        axes = [axes]
    
    for idx, bs in enumerate(batch_sizes):
        ax = axes[idx]
        exp = results['experiments'][f'batch_{bs}']
        epochs = range(1, n_epochs + 1)
        
        ax.plot(epochs, exp['bp']['delta_losses'], '-', label='BP', linewidth=2, color='blue')
        ax.plot(epochs, exp['danp']['delta_losses'], '-', label='DANP', linewidth=2, color='red')
        ax.plot(epochs, exp['wp']['delta_losses'], '-', label='WP', linewidth=2, color='green')
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel('Delta Loss', fontsize=11)
        ax.set_title(f'Delta Loss vs Epoch\n(Batch Size={bs})', fontsize=12)
        ax.legend(fontsize=10)
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_file = os.path.join(output_dir, 'delta_loss_vs_epochs.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Delta loss curves saved to {plot_file}")
    plt.close()
    
    # Plot 5: Combined loss and delta loss vs epochs (for all batch sizes, one method at a time)
    for method in ['bp', 'danp', 'wp']:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Left: Train/Test loss
        ax = axes[0]
        for bs in batch_sizes:
            exp = results['experiments'][f'batch_{bs}']
            epochs = range(1, n_epochs + 1)
            ax.plot(epochs, exp[method]['train_losses'], '--', 
                   label=f'Train (BS={bs})', linewidth=2, alpha=0.7)
            ax.plot(epochs, exp[method]['test_losses'], '-', 
                   label=f'Test (BS={bs})', linewidth=2)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title(f'{method.upper()}: Train/Test Loss vs Epoch', fontsize=14)
        ax.legend(fontsize=9, ncol=2)
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        
        # Right: Delta loss
        ax = axes[1]
        for bs in batch_sizes:
            exp = results['experiments'][f'batch_{bs}']
            epochs = range(1, n_epochs + 1)
            ax.plot(epochs, exp[method]['delta_losses'], '-', 
                   label=f'BS={bs}', linewidth=2)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Delta Loss', fontsize=12)
        ax.set_title(f'{method.upper()}: Delta Loss vs Epoch', fontsize=14)
        ax.legend(fontsize=10)
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_file = os.path.join(output_dir, f'{method}_loss_vs_epochs.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"{method.upper()} loss curves saved to {plot_file}")
        plt.close()


def plot_training_results(results_bp, results_danp, results_wp, output_dir='./toy_model_results', 
                         layer_sizes=None, config=None):
    """
    Plot training results comparing BP, DANP, and WP methods.
    
    Args:
        results_bp: results dict from train_bp_model
        results_danp: results dict from train_danp_model
        results_wp: results dict from train_wp_model
        output_dir: directory to save plots
        layer_sizes: optional layer sizes for title
        config: optional config dict for parameter display
    """
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Convergence curves
    ax = axes[0]
    
    # BP results
    if results_bp:
        epochs_bp = range(len(results_bp['train_losses']))
        ax.plot(epochs_bp, results_bp['train_losses'], '--', label='BP Train', 
                linewidth=2, alpha=0.7, color='blue')
        ax.plot(epochs_bp, results_bp['test_losses'], '-', label='BP Test', 
                linewidth=2, color='blue')
    
    # DANP results
    if results_danp:
        epochs_danp = range(len(results_danp['train_losses']))
        ax.plot(epochs_danp, results_danp['train_losses'], '--', label='DANP Train', 
                linewidth=2, alpha=0.7, color='red')
        ax.plot(epochs_danp, results_danp['test_losses'], '-', label='DANP Test', 
                linewidth=2, color='red')
    
    # WP results
    if results_wp:
        epochs_wp = range(len(results_wp['train_losses']))
        ax.plot(epochs_wp, results_wp['train_losses'], '--', label='WP Train', 
                linewidth=2, alpha=0.7, color='green')
        ax.plot(epochs_wp, results_wp['test_losses'], '-', label='WP Test', 
                linewidth=2, color='green')
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    title = 'Convergence Curves: BP vs DANP vs WP'
    if layer_sizes:
        title += f'\nArchitecture: {layer_sizes}'
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=11)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Final losses comparison
    ax = axes[1]
    methods = []
    train_losses = []
    test_losses = []
    
    if results_bp:
        methods.append('BP')
        train_losses.append(results_bp['final_train'])
        test_losses.append(results_bp['final_test'])
    
    if results_danp:
        methods.append('DANP')
        train_losses.append(results_danp['final_train'])
        test_losses.append(results_danp['final_test'])
    
    if results_wp:
        methods.append('WP')
        train_losses.append(results_wp['final_train'])
        test_losses.append(results_wp['final_test'])
    
    if methods:
        x = np.arange(len(methods))
        width = 0.35
        ax.bar(x - width/2, train_losses, width, label='Train Loss', alpha=0.7)
        ax.bar(x + width/2, test_losses, width, label='Test Loss', alpha=0.7)
        ax.set_xlabel('Method', fontsize=12)
        ax.set_ylabel('Final Loss', fontsize=12)
        ax.set_title('Final Loss Comparison', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(methods)
        ax.legend(fontsize=11)
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3, axis='y')
    
    # Add parameter text box if config provided
    if config:
        param_text = "Parameters:\n"
        if 'lr' in config:
            param_text += f"BP lr: {config['lr']}\n"
        if 'eta' in config:
            param_text += f"DANP eta: {config['eta']}, alpha: {config.get('alpha', 'N/A')}, sigma: {config.get('sigma', 'N/A')}\n"
        if 'alpha' in config and 'eta' not in config:
            param_text += f"WP alpha: {config['alpha']}, sigma: {config.get('sigma', 'N/A')}\n"
        if 'batch_size' in config:
            param_text += f"Batch size: {config['batch_size']}\n"
        if 'population_size' in config:
            param_text += f"WP population size: {config['population_size']}\n"
        
        fig.text(0.02, 0.02, param_text, fontsize=9, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5), 
                family='monospace', verticalalignment='bottom')
    
    plt.tight_layout()
    plot_file = os.path.join(output_dir, 'training_comparison.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved to {plot_file}")
    plt.close()
    
    # Print summary
    print("\n" + "="*80)
    print("FINAL RESULTS SUMMARY")
    print("="*80)
    if results_bp:
        print(f"BP   - Final Train Loss: {results_bp['final_train']:.6f}, Final Test Loss: {results_bp['final_test']:.6f}")
    if results_danp:
        print(f"DANP - Final Train Loss: {results_danp['final_train']:.6f}, Final Test Loss: {results_danp['final_test']:.6f}")
    if results_wp:
        print(f"WP   - Final Train Loss: {results_wp['final_train']:.6f}, Final Test Loss: {results_wp['final_test']:.6f}")
    print("="*80)


if __name__ == "__main__":
    # Training examples
    # dataset = SyntheticDataset(10, 32, 1, 5000, 0)

    # print("-"*80)
    # print("Training BP model...")
    # print("-"*80)
    
    # # wp variable
    # population_size = 10

    # results_bp = train_bp_model(
    #     layer_sizes=[10, 32, 1],
    #     dataset=dataset,
    #     n_epochs=10,
    #     lr=1e-3,
    #     batch_size=1,
    #     train_split=0.8,
    #     seed=42
    # )

    # print("-"*80)
    # print("Training DANP model...")
    # print("-"*80)

    # results_danp = train_danp_model(
    #     layer_sizes=[10, 32, 1],
    #     dataset=dataset,
    #     n_epochs=10,
    #     eta=1e-3,
    #     alpha=1e-5,
    #     sigma=0.001,
    #     batch_size=1,
    #     train_split=0.8,
    #     seed=42
    # )

    # print("-"*80)
    # print("Training WP model...")
    # print("-"*80)

    # results_wp = train_wp_model(
    #     layer_sizes=[10, 32, 1],
    #     dataset=dataset,
    #     n_epochs=10,
    #     sigma=0.001,
    #     alpha=1e-3,
    #     batch_size=1,
    #     population_size=population_size,
    #     train_split=0.8,
    #     seed=42
    # )

    # # Plot results
    # print("\n" + "="*80)
    # print("Creating visualization...")
    # print("="*80)
    
    # config = {
    #     'lr': 1e-3,
    #     'eta': 1e-3,
    #     'alpha': 1e-5,
    #     'sigma': 0.001,
    #     'batch_size': 1,
    #     'population_size': population_size
    # }
    
    # plot_training_results(
    #     results_bp=results_bp,
    #     results_danp=results_danp,
    #     results_wp=results_wp,
    #     output_dir='./toy_model_results',
    #     layer_sizes=[10, 32, 1],
    #     config=config
    # )

    # # ============================================================================
    # # BATCH SIZE EXPERIMENT
    # # ============================================================================
    # print("\n" + "="*80)
    # print("STARTING BATCH SIZE EXPERIMENT")
    # print("="*80)
    
    # run_batch_size_experiment(
    #     layer_sizes=[10, 32, 1],
    #     n_samples=5000,
    #     n_epochs=50,
    #     base_lr=1e-3,
    #     base_eta=1e-3,
    #     base_alpha=1e-3,
    #     alpha_danp=1e-5,
    #     sigma=0.001,
    #     batch_sizes=[2, 4, 8, 16, 32],
    #     train_split=0.8,
    #     seed=42,
    #     device='cpu',
    #     output_dir='./batch_size_experiment_results'
    # )

    # # ============================================================================
    # # LARGER ARCHITECTURE EXPERIMENT: 10 → 128 → 128 → 128 → 128 → 1
    # # ============================================================================
    print("\n" + "="*80)
    print("LARGER ARCHITECTURE EXPERIMENT")
    print("Architecture: 10 → 128 → 128 → 128 → 128 → 1")
    print("="*80)
    
    # Create dataset with matching teacher architecture
    layer_sizes_large = [10, 128, 128, 128, 128, 1]
    dataset_large = SyntheticDataset(
        input_dim=10,
        hidden_dim=128,  # Not used since teacher_layer_sizes is provided
        output_dim=1,
        n_samples=5000,
        seed=0,
        teacher_layer_sizes=layer_sizes_large
    )
    
    print("-"*80)
    print("Training BP model (large architecture)...")
    print("-"*80)
    
    results_bp_large = train_bp_model(
        layer_sizes=layer_sizes_large,
        dataset=dataset_large,
        n_epochs=50,
        lr=1e-3,
        batch_size=1,
        train_split=0.8,
        seed=42
    )
    
    print("-"*80)
    print("Training DANP model (large architecture)...")
    print("-"*80)
    
    results_danp_large = train_danp_model(
        layer_sizes=layer_sizes_large,
        dataset=dataset_large,
        n_epochs=50,
        eta=1e-3,
        alpha=1e-5,
        sigma=0.001,
        batch_size=1,
        train_split=0.8,
        seed=42
    )
    
    print("-"*80)
    print("Training WP model (large architecture)...")
    print("-"*80)
    
    results_wp_large = train_wp_model(
        layer_sizes=layer_sizes_large,
        dataset=dataset_large,
        n_epochs=50,
        sigma=0.001,
        alpha=1e-5,
        batch_size=1,
        population_size=10,
        train_split=0.8,
        seed=42
    )
    
    # Plot results for large architecture
    print("\n" + "="*80)
    print("Creating visualization for large architecture...")
    print("="*80)
    
    config_large = {
        'lr': 1e-3,
        'eta': 1e-3,
        'alpha': 1e-5,
        'sigma': 0.001,
        'batch_size': 1,  # BP uses batch_size=1, DANP/WP use batch_size=1
        'population_size': 10
    }
    
    plot_training_results(
        results_bp=results_bp_large,
        results_danp=results_danp_large,
        results_wp=results_wp_large,
        output_dir='./toy_model_results_large',
        layer_sizes=layer_sizes_large,
        config=config_large
    )