import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from danp import DANPMLP
from bp import BPMLP

class SyntheticDataset(Dataset):
    """
    Creates a fixed synthetic dataset by generating data from a teacher network.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, n_samples=10000, seed=42):
        """
        Args:
            input_dim: dimension of input
            hidden_dim: hidden layer size for teacher network
            output_dim: dimension of output
            n_samples: number of samples to generate
            seed: random seed for reproducibility
        """
        torch.manual_seed(seed)
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_samples = n_samples
        
        # Create teacher network weights (fixed)
        self.W1 = torch.randn(hidden_dim, input_dim) / input_dim**0.5
        self.W2 = torch.randn(output_dim, hidden_dim) / hidden_dim**0.5
        
        # Generate fixed dataset
        self.inputs = torch.randn(n_samples, input_dim)
        self.targets = self._generate_targets(self.inputs)
        
    def _generate_targets(self, x):
        """Generate targets using teacher network"""
        # x shape: (n_samples, input_dim) or (input_dim,)
        if x.dim() == 1:
            h = torch.relu(self.W1 @ x)
            y = self.W2 @ h
        else:
            h = torch.relu(x @ self.W1.T)
            y = h @ self.W2.T
        return y
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

# TRAINING EXAMPLES WITH DATASETS

def train_bp_with_dataset(input_dim=10, hidden_dim=32, output_dim=1, n_samples=5000, n_epochs=10, eta=1e-3, batch_size=32, train_split=0.8, seed=42):
    """
    Train BP model using PyTorch dataset.
    
    Args:
        input_dim: Input dimension
        hidden_dim: Hidden layer size
        output_dim: Output dimension
        n_samples: Total number of samples in dataset
        n_epochs: Number of training epochs
        eta: Learning rate
        batch_size: Batch size
        train_split: Fraction of data for training
        seed: Random seed
    
    Returns:
        dict with 'train_losses' and 'test_losses' per epoch
    """
    
    print("\n" + "="*80)
    print("TRAINING BP WITH SYNTHETIC DATASET")
    print("="*80)
    print(f"Dataset size: {n_samples}")
    print(f"Epochs: {n_epochs}")
    print(f"Network: {input_dim} → {hidden_dim} → {output_dim}")
    print(f"η={eta}, batch_size={batch_size}")
    print("="*80 + "\n")
    
    # Create dataset
    dataset = SyntheticDataset(input_dim, hidden_dim, output_dim, n_samples=n_samples, seed=seed)
    
    # Split into train/test
    train_size = int(train_split * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size],
        generator=torch.Generator().manual_seed(seed)
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}\n")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = BPMLP([input_dim, hidden_dim, output_dim])
    optimizer = torch.optim.SGD(model.parameters(), lr=eta)
    
    train_losses_per_epoch = []
    test_losses_per_epoch = []
    
    # Training loop
    for epoch in range(n_epochs):
        model.train()
        train_losses = []
        
        for x, y in train_loader:
            optimizer.zero_grad()
            pred = model(x)
            loss = F.mse_loss(pred, y)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        
        avg_train_loss = sum(train_losses) / len(train_losses)
        train_losses_per_epoch.append(avg_train_loss)
        
        # Evaluation
        model.eval()
        test_losses = []
        with torch.no_grad():
            for x, y in test_loader:
                pred = model(x)
                loss = F.mse_loss(pred, y)
                test_losses.append(loss.item())
        
        avg_test_loss = sum(test_losses) / len(test_losses)
        test_losses_per_epoch.append(avg_test_loss)
        
        print(f"Epoch {epoch+1:2d}/{n_epochs} | "
              f"Train Loss: {avg_train_loss:.6f} | "
              f"Test Loss: {avg_test_loss:.6f}")
    
    # Final summary
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    print(f"Final Train Loss: {train_losses_per_epoch[-1]:.6f}")
    print(f"Final Test Loss:  {test_losses_per_epoch[-1]:.6f}")
    print("="*80 + "\n")
    
    return {
        'train_losses': train_losses_per_epoch,
        'test_losses': test_losses_per_epoch,
        'final_train': train_losses_per_epoch[-1],
        'final_test': test_losses_per_epoch[-1]
    }

def train_danp_with_dataset(input_dim=10, hidden_dim=32, output_dim=1, n_samples=5000, n_epochs=10, eta=1e-3, alpha=1e-5, sigma=0.001, batch_size=1, train_split=0.8, seed=42):
    """
    Train DANP using a fixed synthetic dataset.
    
    Args:
        input_dim: Input dimension
        hidden_dim: Hidden layer size
        output_dim: Output dimension
        n_samples: Total number of samples in dataset
        n_epochs: Number of training epochs
        eta: Weight learning rate
        alpha: Decorrelation learning rate (0 for ANP)
        sigma: Noise standard deviation
        batch_size: Batch size (use 1 for DANP)
        train_split: Fraction of data for training
        seed: Random seed
    
    Returns:
        dict with 'train_losses' and 'test_losses' per epoch
    """
    
    print("\n" + "="*80)
    print("TRAINING DANP WITH SYNTHETIC DATASET")
    print("="*80)
    print(f"Dataset size: {n_samples}")
    print(f"Epochs: {n_epochs}")
    print(f"Network: {input_dim} → {hidden_dim} → {output_dim}")
    print(f"η={eta}, α={alpha}, σ={sigma}")
    print("="*80 + "\n")
    
    # Create dataset
    dataset = SyntheticDataset(input_dim, hidden_dim, output_dim, n_samples=n_samples, seed=seed)
    
    # Split into train/test
    train_size = int(train_split * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size],
        generator=torch.Generator().manual_seed(seed)
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}\n")
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Create model
    model = DANPMLP(
        [input_dim, hidden_dim, output_dim],
        sigma=sigma,
        eta=eta,
        alpha=alpha
    )
    
    train_losses_per_epoch = []
    test_losses_per_epoch = []
    
    # Training loop
    for epoch in range(n_epochs):
        # Training
        train_losses = []
        for x, y in train_loader:
            x = x.squeeze(0)  # Remove batch dimension
            y = y.squeeze(0)
            
            loss, _ = model.step(x, y)
            train_losses.append(loss)
        
        avg_train_loss = sum(train_losses) / len(train_losses)
        train_losses_per_epoch.append(avg_train_loss)
        
        # Evaluation on test set
        test_losses = []
        for x, y in test_loader:
            x = x.squeeze(0)
            y = y.squeeze(0)
            
            # Forward pass only (no updates)
            x_clean, _ = model.forward_clean(x)
            loss = F.mse_loss(x_clean[-1], y).item()
            test_losses.append(loss)
        
        avg_test_loss = sum(test_losses) / len(test_losses)
        test_losses_per_epoch.append(avg_test_loss)
        
        print(f"Epoch {epoch+1:2d}/{n_epochs} | "
              f"Train Loss: {avg_train_loss:.6f} | "
              f"Test Loss: {avg_test_loss:.6f}")
    
    # Final summary
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    print(f"Final Train Loss: {train_losses_per_epoch[-1]:.6f}")
    print(f"Final Test Loss:  {test_losses_per_epoch[-1]:.6f}")
    

    print("="*80 + "\n")
    
    return {
        'train_losses': train_losses_per_epoch,
        'test_losses': test_losses_per_epoch,
        'final_train': train_losses_per_epoch[-1],
        'final_test': test_losses_per_epoch[-1]
    }


if __name__ == "__main__":
    # Training examples
    results_banp = train_bp_with_dataset(
        input_dim = 10,
        hidden_dim =32,
        output_dim=1,
        n_samples =5000,
        n_epochs = 10,
        eta=1e-3,
        batch_size=10, 
        train_split=0.8, 
        seed=42
    )
    
    results_danp = train_danp_with_dataset(
        input_dim=10,
        hidden_dim=32,
        output_dim=1,
        n_samples=5000,
        n_epochs=10,
        eta=1e-3,
        alpha=1e-5,
        sigma=0.001
    )
    # print(results_banp)
    # print(results_danp)