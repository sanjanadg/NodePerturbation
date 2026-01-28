#!/usr/bin/env python3
"""
Transformer Block Experiment: Compare DANP vs BP on Transformer Blocks.

This script tests how DANP and BP perform when using transformer block architectures.
Uses the same transformer block structure for teacher, BP student, and DANP student.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import time
import sys
import torch.optim.adam
from torch.utils.data import Dataset, DataLoader

# Import transformer blocks
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'transformer_block_model'))
from transformer_blocks import TransformerBlock, BPTransformerBlock, DANPTransformerBlock

class TransformerSyntheticDataset(Dataset):
    """
    Creates a synthetic dataset using a regular transformer block as the teacher network.
    Uses the same architecture as the student networks.
    """
    def __init__(self, d_model, n_heads, d_ff, seq_len, n_samples=10000, seed=42):
        """
        Args:
            d_model: model dimension
            n_heads: number of attention heads
            d_ff: feed-forward dimension
            seq_len: sequence length
            n_samples: number of samples to generate
            seed: random seed
        """
        torch.manual_seed(seed)
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.seq_len = seq_len
        self.n_samples = n_samples
        
        # Create teacher transformer block (regular transformer, not BP-specific)
        self.teacher = TransformerBlock(d_model, n_heads, d_ff, dropout=0.0)
        self.teacher.eval()
        
        # Generate fixed dataset
        self.inputs = torch.randn(n_samples, seq_len, d_model)
        # print(f"{self.inputs}")
        with torch.no_grad():
            self.targets = self.teacher(self.inputs)
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


def train_bp(d_model, n_heads, d_ff, seq_len, dataset, n_epochs=50, lr=1e-3, 
             batch_size=1, train_split=0.8, seed=42, device='cpu'):
    """
    Train BP transformer block and return final train and test losses.
    
    Args:
        d_model: model dimension
        n_heads: number of attention heads
        d_ff: feed-forward dimension
        seq_len: sequence length
        dataset: TransformerSyntheticDataset instance
        n_epochs: number of training epochs
        lr: learning rate
        batch_size: batch size
        train_split: fraction of data for training
        seed: random seed
        device: device to use
    
    Returns:
        dict with 'train_losses', 'test_losses', 'final_train', 'final_test'
    """
    torch.manual_seed(seed+1000)
    
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
    model = BPTransformerBlock(d_model, n_heads, d_ff, dropout=0.0).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Store per-epoch losses for visualization
    train_losses_per_epoch = []
    test_losses_per_epoch = []
    
    for epoch in range(n_epochs):
        epoch_start_time = time.time()
        model.train()
        
        # Training updates
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = F.mse_loss(pred, y)
            loss.backward()
            optimizer.step()
        
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
        
        # Store per-epoch averages
        train_losses_per_epoch.append(avg_train)
        test_losses_per_epoch.append(avg_test)
        
        print(f"  Epoch {epoch+1:3d}/{n_epochs} | Train Loss: {avg_train:.6f} | Test Loss: {avg_test:.6f} | Time: {epoch_time:.3f}s")
    
    final_train_loss = train_losses_per_epoch[-1]
    final_test_loss = test_losses_per_epoch[-1]
    
    print("\n" + "="*80)
    print("FINAL RESULTS (BP)")
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


def train_danp(d_model, n_heads, d_ff, seq_len, dataset, n_epochs=50, eta=1e-3,
               alpha=1e-4, sigma=0.001, batch_size=1, train_split=0.8, seed=42, device='cpu'):
    """
    Train DANP transformer block (sequential processing) and return final train and test losses.
    
    Args:
        d_model: model dimension
        n_heads: number of attention heads
        d_ff: feed-forward dimension
        seq_len: sequence length
        dataset: TransformerSyntheticDataset instance
        n_epochs: number of training epochs
        eta: weight learning rate
        alpha: decorrelation learning rate
        sigma: noise standard deviation
        batch_size: batch size (processed sequentially)
        train_split: fraction of data for training
        seed: random seed
        device: device to use
    
    Returns:
        dict with 'train_losses', 'test_losses', 'final_train', 'final_test'
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
    model = DANPTransformerBlock(d_model, n_heads, d_ff, sigma=sigma, eta=eta, alpha=alpha)
    # Move model parameters to device
    for h in range(len(model.W_q)):
        model.W_q[h] = model.W_q[h].to(device)
        model.W_k[h] = model.W_k[h].to(device)
        model.W_v[h] = model.W_v[h].to(device)
    model.W_o = model.W_o.to(device)
    model.W_ffn1 = model.W_ffn1.to(device)
    model.W_ffn2 = model.W_ffn2.to(device)
    model.norm1_gamma = model.norm1_gamma.to(device)
    model.norm1_beta = model.norm1_beta.to(device)
    model.norm2_gamma = model.norm2_gamma.to(device)
    model.norm2_beta = model.norm2_beta.to(device)
    for i in range(len(model.R)):
        model.R[i] = model.R[i].to(device)
    
    # Store per-epoch losses for visualization
    train_losses_per_epoch = []
    test_losses_per_epoch = []
    
    for epoch in range(n_epochs):
        epoch_start_time = time.time()
        
        # Training updates (sequential processing)
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            # Process each sample in the batch individually
            for i in range(x.shape[0]):
                x_sample = x[i:i+1]  # Keep batch dimension
                y_sample = y[i:i+1]
                model.step(x_sample, y_sample)
        
        # Evaluate on training set (after all updates)
        train_losses = []
        with torch.no_grad():
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                for i in range(x.shape[0]):
                    x_sample = x[i:i+1]
                    y_sample = y[i:i+1]
                    x_clean, _ = model.forward_clean(x_sample)
                    loss = F.mse_loss(x_clean[-1], y_sample).item()
                    train_losses.append(loss)
        
        # Evaluate on test set
        test_losses = []
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                for i in range(x.shape[0]):
                    x_sample = x[i:i+1]
                    y_sample = y[i:i+1]
                    x_clean, _ = model.forward_clean(x_sample)
                    loss = F.mse_loss(x_clean[-1], y_sample).item()
                    test_losses.append(loss)
        
        epoch_time = time.time() - epoch_start_time
        avg_train = np.mean(train_losses)
        avg_test = np.mean(test_losses)
        
        # Store per-epoch averages
        train_losses_per_epoch.append(avg_train)
        test_losses_per_epoch.append(avg_test)
        
        print(f"  Epoch {epoch+1:3d}/{n_epochs} | Train Loss: {avg_train:.6f} | Test Loss: {avg_test:.6f} | Time: {epoch_time:.3f}s")
    
    final_train_loss = train_losses_per_epoch[-1]
    final_test_loss = test_losses_per_epoch[-1]
    
    print("\n" + "="*80)
    print("FINAL RESULTS (DANP)")
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


def train_danp_batch(d_model, n_heads, d_ff, seq_len, dataset, n_epochs=50, eta=1e-3,
                    alpha=1e-4, sigma=0.001, batch_size=1, train_split=0.8, seed=42, device='cpu'):
    """
    Train DANP transformer block with batch processing (accumulating gradients across batch).
    
    Args:
        d_model: model dimension
        n_heads: number of attention heads
        d_ff: feed-forward dimension
        seq_len: sequence length
        dataset: TransformerSyntheticDataset instance
        n_epochs: number of training epochs
        eta: weight learning rate (will be scaled by batch_size)
        alpha: decorrelation learning rate
        sigma: noise standard deviation
        batch_size: batch size
        train_split: fraction of data for training
        seed: random seed
        device: device to use
    
    Returns:
        dict with 'train_losses', 'test_losses', 'final_train', 'final_test'
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
    
    # Create model with scaled learning rate for batch training
    # Scale eta by batch_size to maintain same effective learning rate per sample
    # This matches the pattern from toy_model/sythentic_data.py train_danp_model_batch
    # When we accumulate gradients and divide by batch_size, we need to scale eta by batch_size
    # to keep the effective learning rate per sample constant
    model = DANPTransformerBlock(d_model, n_heads, d_ff, sigma=sigma, eta=eta, alpha=alpha)
    # Move model parameters to device
    for h in range(len(model.W_q)):
        model.W_q[h] = model.W_q[h].to(device)
        model.W_k[h] = model.W_k[h].to(device)
        model.W_v[h] = model.W_v[h].to(device)
    model.W_o = model.W_o.to(device)
    model.W_ffn1 = model.W_ffn1.to(device)
    model.W_ffn2 = model.W_ffn2.to(device)
    model.norm1_gamma = model.norm1_gamma.to(device)
    model.norm1_beta = model.norm1_beta.to(device)
    model.norm2_gamma = model.norm2_gamma.to(device)
    model.norm2_beta = model.norm2_beta.to(device)
    for i in range(len(model.R)):
        model.R[i] = model.R[i].to(device)
    
    # Store per-epoch losses for visualization
    train_losses_per_epoch = []
    test_losses_per_epoch = []
    
    for epoch in range(n_epochs):
        epoch_start_time = time.time()
        
        # Training updates using batch processing
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            train_danp_transformer_batch(model, x, y)
        
        # Evaluate on training set (after all updates)
        train_losses = []
        with torch.no_grad():
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                for i in range(x.shape[0]):
                    x_sample = x[i:i+1]
                    y_sample = y[i:i+1]
                    x_clean, _ = model.forward_clean(x_sample)
                    loss = F.mse_loss(x_clean[-1], y_sample).item()
                    train_losses.append(loss)
        
        # Evaluate on test set
        test_losses = []
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                for i in range(x.shape[0]):
                    x_sample = x[i:i+1]
                    y_sample = y[i:i+1]
                    x_clean, _ = model.forward_clean(x_sample)
                    loss = F.mse_loss(x_clean[-1], y_sample).item()
                    test_losses.append(loss)
        
        epoch_time = time.time() - epoch_start_time
        avg_train = np.mean(train_losses)
        avg_test = np.mean(test_losses)
        
        # Store per-epoch averages
        train_losses_per_epoch.append(avg_train)
        test_losses_per_epoch.append(avg_test)
        
        print(f"  Epoch {epoch+1:3d}/{n_epochs} | Train Loss: {avg_train:.6f} | Test Loss: {avg_test:.6f} | Time: {epoch_time:.3f}s")
    
    final_train_loss = train_losses_per_epoch[-1]
    final_test_loss = test_losses_per_epoch[-1]
    
    print("\n" + "="*80)
    print("FINAL RESULTS (DANP Batch)")
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


def train_danp_transformer_batch(model, x_batch, y_batch):
    """
    Train DANP transformer block on a batch by accumulating gradient estimates.
    
    Args:
        model: DANPTransformerBlock instance
        x_batch: batch of inputs, shape (batch_size, seq_len, d_model)
        y_batch: batch of targets, shape (batch_size, seq_len, d_model)
    
    Returns:
        average_loss: average clean loss over the batch
    """
    batch_size = x_batch.shape[0]
    
    # Initialize accumulated updates
    accumulated_updates = {
        'W_q': [torch.zeros_like(W) for W in model.W_q],
        'W_k': [torch.zeros_like(W) for W in model.W_k],
        'W_v': [torch.zeros_like(W) for W in model.W_v],
        'W_o': torch.zeros_like(model.W_o),
        'W_ffn1': torch.zeros_like(model.W_ffn1),
        'W_ffn2': torch.zeros_like(model.W_ffn2),
    }
    accumulated_decorrelation_updates = [torch.zeros_like(R) for R in model.R]
    
    total_loss = 0.0
    
    # Process each sample in the batch
    for i in range(batch_size):
        x_sample = x_batch[i:i+1]  # Keep batch dimension
        y_sample = y_batch[i:i+1]
        
        weight_grads, decorrelation_grads, loss_clean = model.compute_grad_estimate(x_sample, y_sample)
        
        # Accumulate weight updates
        for h in range(len(model.W_q)):
            accumulated_updates['W_q'][h] += weight_grads['W_q'][h]
            accumulated_updates['W_k'][h] += weight_grads['W_k'][h]
            accumulated_updates['W_v'][h] += weight_grads['W_v'][h]
        accumulated_updates['W_o'] += weight_grads['W_o']
        accumulated_updates['W_ffn1'] += weight_grads['W_ffn1']
        accumulated_updates['W_ffn2'] += weight_grads['W_ffn2']
        
        # Accumulate decorrelation updates
        for j, grad in enumerate(decorrelation_grads):
            accumulated_decorrelation_updates[j] += grad
        
        total_loss += loss_clean
    
    # Average and apply updates
    for h in range(len(model.W_q)):
        model.W_q[h] -= accumulated_updates['W_q'][h] / batch_size
        model.W_k[h] -= accumulated_updates['W_k'][h] / batch_size
        model.W_v[h] -= accumulated_updates['W_v'][h] / batch_size
    model.W_o -= accumulated_updates['W_o'] / batch_size
    model.W_ffn1 -= accumulated_updates['W_ffn1'] / batch_size
    model.W_ffn2 -= accumulated_updates['W_ffn2'] / batch_size
    
    for j in range(len(model.R)):
        model.R[j] -= accumulated_decorrelation_updates[j] / batch_size
    
    return total_loss / batch_size


def run_transformer_experiment(config, output_dir='./transformer_block_results'):
    """Run transformer block experiment comparing BP, DANP, and DANP batch."""
    os.makedirs(output_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Generate synthetic dataset
    print(f"\nGenerating synthetic dataset...")
    dataset = TransformerSyntheticDataset(
        d_model=config['d_model'],
        n_heads=config['n_heads'],
        d_ff=config['d_ff'],
        seq_len=config['seq_len'],
        n_samples=config['n_samples'],
        seed=config['seed']
    )
    print(f"Dataset size: {len(dataset)}")
    print(f"Transformer config: d_model={config['d_model']}, n_heads={config['n_heads']}, "
          f"d_ff={config['d_ff']}, seq_len={config['seq_len']}")
    
    results = {
        'config': config,
        'experiments': {}
    }
    
    print(f"\n{'='*80}")
    print("TRANSFORMER BLOCK EXPERIMENT: BP vs DANP vs DANP Batch")
    print(f"{'='*80}\n")
    
    # Train BP
    if config.get('train_bp', True):
        print("Training BP transformer block...")
        bp_results = train_bp(
            d_model=config['d_model'],
            n_heads=config['n_heads'],
            d_ff=config['d_ff'],
            seq_len=config['seq_len'],
            dataset=dataset,
            n_epochs=config['n_epochs'],
            lr=0.001,
            batch_size=config.get('batch_size', 1),
            train_split=0.8,
            seed=config['seed'],
            device=device
        )
        results['experiments']['bp'] = bp_results
    
    # Train DANP (sequential)
    if config.get('train_danp', True):
        print("\nTraining DANP transformer block (sequential)...")
        danp_results = train_danp(
            d_model=config['d_model'],
            n_heads=config['n_heads'],
            d_ff=config['d_ff'],
            seq_len=config['seq_len'],
            dataset=dataset,
            n_epochs=config['n_epochs'],
            eta=1e-3,  
            alpha=1e-5,  
            sigma=0.001,  
            batch_size=1,
            train_split=0.8,
            seed=config['seed'],
            device=device
        )
        results['experiments']['danp'] = danp_results

    # # Train DANP Batch
    if config.get('train_danp_batch', True):
        print("\nTraining DANP transformer block (batch processing)...")
        danp_batch_results = train_danp_batch(
            d_model=config['d_model'],
            n_heads=config['n_heads'],
            d_ff=config['d_ff'],
            seq_len=config['seq_len'],
            dataset=dataset,
            n_epochs=config['n_epochs'],
            eta=1e-3,  
            alpha=1e-5,  
            sigma=0.001, 
            batch_size=32,
            train_split=0.8,
            seed=config['seed'],
            device=device
        )
        results['experiments']['danp_batch'] = danp_batch_results
    
    # Create visualization
    create_visualization(results, output_dir)
    
    # Save results
    results_file = os.path.join(output_dir, 'results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_file}")
    
    return results


def create_visualization(results, output_dir):
    """Create visualization plots."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Convergence curves
    ax = axes[0]
    for method, exp in results['experiments'].items():
        epochs = range(len(exp['train_losses']))
        ax.plot(epochs, exp['train_losses'], '--', label=f'{method.upper()} Train', linewidth=2, alpha=0.7)
        ax.plot(epochs, exp['test_losses'], '-', label=f'{method.upper()} Test', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Convergence Curves: Transformer Blocks', fontsize=14)
    ax.legend(fontsize=11)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Final losses comparison
    ax = axes[1]
    methods = list(results['experiments'].keys())
    train_losses = [results['experiments'][m]['final_train'] for m in methods]
    test_losses = [results['experiments'][m]['final_test'] for m in methods]
    
    x = np.arange(len(methods))
    width = 0.35
    ax.bar(x - width/2, train_losses, width, label='Train Loss', alpha=0.7)
    ax.bar(x + width/2, test_losses, width, label='Test Loss', alpha=0.7)
    ax.set_xlabel('Method', fontsize=12)
    ax.set_ylabel('Final Loss', fontsize=12)
    ax.set_title('Final Loss Comparison', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in methods])
    ax.legend(fontsize=11)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plot_file = os.path.join(output_dir, 'transformer_results.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to {plot_file}")
    plt.close()


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Transformer block experiment: BP vs DANP')
    parser.add_argument('--d_model', type=int, default=128, help='Model dimension')
    parser.add_argument('--n_heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--d_ff', type=int, default=512, help='Feed-forward dimension')
    parser.add_argument('--seq_len', type=int, default=10, help='Sequence length')
    parser.add_argument('--n_samples', type=int, default=5000, help='Dataset size')
    parser.add_argument('--n_epochs', type=int, default=50, help='Training epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--sigma', type=float, default=0.001, help='DANP noise std')
    parser.add_argument('--alpha', type=float, default=1e-4, help='DANP decorrelation rate')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output_dir', type=str, default='./transformer_block_results', 
                       help='Output directory')
    parser.add_argument('--train_bp', action='store_true', default=True, help='Train BP model')
    parser.add_argument('--train_danp', action='store_true', default=True, help='Train DANP model')
    parser.add_argument('--train_danp_batch', action='store_true', default=True, help='Train DANP batch model')
    
    args = parser.parse_args()
    
    config = {
        'd_model': args.d_model,
        'n_heads': args.n_heads,
        'd_ff': args.d_ff,
        'seq_len': args.seq_len,
        'n_samples': args.n_samples,
        'n_epochs': args.n_epochs,
        'lr': args.lr,
        'sigma': args.sigma,
        'alpha': args.alpha,
        'batch_size': args.batch_size,
        'seed': args.seed,
        'train_bp': args.train_bp,
        'train_danp': args.train_danp,
        'train_danp_batch': args.train_danp_batch
    }
    
    print("="*80)
    print("TRANSFORMER BLOCK EXPERIMENT: BP vs DANP vs DANP Batch")
    print("="*80)

    # these are often customized for the experiment, so we don't print them here
    # print(f"Configuration:")
    # for key, value in config.items():
    #     print(f"  {key}: {value}")
    # print("="*80)
    
    results = run_transformer_experiment(config, args.output_dir)
    
    print(f"\nResults saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
