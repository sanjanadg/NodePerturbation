import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from danp import DANPMLP
from bp import BPMLP
import time
import numpy as np

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
    """Train BP model and return final train and test losses."""
    torch.manual_seed(seed)
    
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
    final_train_loss = None
    final_test_loss = None
    
    for epoch in range(n_epochs):
        epoch_start_time = time.time()
        model.train()
        
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = F.mse_loss(pred, y)
            loss.backward()
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
        
        # Store final epoch losses
        final_train_loss = avg_train
        final_test_loss = avg_test
        
        # Print epoch progress
        print(f"  Epoch {epoch+1:3d}/{n_epochs} | Train Loss: {avg_train:.6f} | Test Loss: {avg_test:.6f} | Time: {epoch_time:.3f}s")
    
    # Final summary
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    print(f"Final Train Loss: {final_train_loss:.6f}")
    print(f"Final Test Loss:  {final_test_loss:.6f}")
    print("="*80 + "\n")
    
    return {
        'train_losses': train_losses,
        'test_losses': test_losses,
        'final_train': final_train_loss,
        'final_test': final_test_loss
    }

def train_danp_model(layer_sizes, dataset, n_epochs=50, eta=1e-3, alpha=1e-4, 
                     sigma=0.001, batch_size=1, train_split=0.8, seed=42, device='cpu'):
    """Train DANP model and return final train and test losses."""
    torch.manual_seed(seed)
    
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
    final_train_loss = None
    final_test_loss = None
    
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
        
        # Store final epoch losses
        final_train_loss = avg_train
        final_test_loss = avg_test
        
        # Print epoch progress
        print(f"  Epoch {epoch+1:3d}/{n_epochs} | Train Loss: {avg_train:.6f} | Test Loss: {avg_test:.6f} | Time: {epoch_time:.3f}s")

    # Final summary
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    print(f"Final Train Loss: {final_train_loss:.6f}")
    print(f"Final Test Loss:  {final_test_loss:.6f}")
    print("="*80 + "\n")

    return {
        'train_losses': train_losses,
        'test_losses': test_losses,
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
    """
    batch_size = x_batch.shape[0]
    
    # Initialize accumulated updates
    accumulated_weight_updates = [torch.zeros_like(W) for W in model.W]
    accumulated_decorrelation_updates = [torch.zeros_like(R) for R in model.R[1:]]  # Skip R[0]
    
    total_loss = 0.0
    
    # Compute gradient estimate for each sample in the batch
    for i in range(batch_size):
        x_sample = x_batch[i]
        y_sample = y_batch[i]
        
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
    
    return total_loss / batch_size


def train_danp_model_batch(layer_sizes, dataset, n_epochs=50, eta=1e-3, alpha=1e-4, 
                           sigma=0.001, batch_size=1, train_split=0.8, seed=42, device='cpu'):
    """
    Train DANP model with proper batch processing using train_danp_batch.
    Returns final train and test losses.
    """
    torch.manual_seed(seed)
    
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
    final_train_loss = None
    final_test_loss = None
    
    for epoch in range(n_epochs):
        epoch_start_time = time.time()
        
        # Training updates using batch processing
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            train_danp_batch(model, x, y)
        
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
        
        # Store final epoch losses
        final_train_loss = avg_train
        final_test_loss = avg_test
        
        # Print epoch progress
        print(f"  Epoch {epoch+1:3d}/{n_epochs} | Train Loss: {avg_train:.6f} | Test Loss: {avg_test:.6f} | Time: {epoch_time:.3f}s")
    
    # Final summary
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    print(f"Final Train Loss: {final_train_loss:.6f}")
    print(f"Final Test Loss:  {final_test_loss:.6f}")
    print("="*80 + "\n")
    
    return {
        'train_losses': train_losses,
        'test_losses': test_losses,
        'final_train': final_train_loss,
        'final_test': final_test_loss
    }


if __name__ == "__main__":
    # Training examples
    dataset = SyntheticDataset(10, 32, 1, 5000, 0)

    print("-"*80)
    print("Training BP model...")
    print("-"*80)

    results_bp = train_bp_model(
        layer_sizes=[10, 32, 1],
        dataset=dataset,
        n_epochs=10,
        lr=1e-3,
        batch_size=1,
        train_split=0.8,
        seed=0
    )

    print("-"*80)
    print("Training DANP model...")
    print("-"*80)

    results_danp = train_danp_model(
        layer_sizes=[10, 32, 1],
        dataset=dataset,
        n_epochs=10,
        eta=1e-3,
        alpha=1e-5,
        sigma=0.001,
        batch_size=1,
        train_split=0.8,
        seed=0
    )