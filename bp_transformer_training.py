#!/usr/bin/env python3
"""
BP Transformer Training Experiment: Compare different optimizers for BP transformer blocks.

This script tests how different optimizers (SGD, Adam, SGD with weight decay, Adam with weight decay)
affect the training of transformer blocks using backpropagation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import time
import sys
from torch.utils.data import Dataset, DataLoader

# Import transformer blocks and dataset
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'transformer_block_model'))
from transformer_block_model.transformer_blocks import BPTransformerBlock
from transformer_block_model.transformer_block_experiment import TransformerSyntheticDataset


def train_bp_transformer_optimizer(d_model, n_heads, d_ff, seq_len, dataset, optimizer_name='sgd',
                                   n_epochs=50, lr=1e-3, weight_decay=0.0, batch_size=1,
                                   train_split=0.8, seed=42, device='cpu'):
    """
    Train BP transformer block with specified optimizer and return final train and test losses.
    
    Args:
        d_model: model dimension
        n_heads: number of attention heads
        d_ff: feed-forward dimension
        seq_len: sequence length
        dataset: TransformerSyntheticDataset instance
        optimizer_name: 'sgd', 'adam', 'sgd_wd', 'adam_wd'
        n_epochs: number of training epochs
        lr: learning rate
        weight_decay: weight decay coefficient (used if optimizer_name ends with '_wd')
        batch_size: batch size
        train_split: fraction of data for training
        seed: random seed
        device: device to use
    
    Returns:
        dict with 'train_losses', 'test_losses', 'final_train', 'final_test', 'optimizer_name'
    """
    torch.manual_seed(seed+1000) # need to ensure initial weight initalizations are not the same as the dataset
    
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
    
    # Create optimizer based on optimizer_name
    if optimizer_name == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr)
    elif optimizer_name == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == 'sgd_wd':
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'adam_wd':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    # Store losses per epoch (not per batch)
    train_losses_per_epoch = []
    test_losses_per_epoch = []
    final_train_loss = None
    final_test_loss = None
    
    # Print description based on optimizer type
    if optimizer_name in ['sgd_wd', 'adam_wd']:
        print(f"\nTraining with {optimizer_name.upper()} (lr={lr}, weight_decay={weight_decay})...")
    else:
        print(f"\nTraining with {optimizer_name.upper()} (lr={lr})...")
    
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
        
        final_train_loss = avg_train
        final_test_loss = avg_test
        
        print(f"  Epoch {epoch+1:3d}/{n_epochs} | Train Loss: {avg_train:.6f} | Test Loss: {avg_test:.6f} | Time: {epoch_time:.3f}s")
    
    print(f"\n{optimizer_name.upper()} Final Results:")
    print(f"  Final Train Loss: {final_train_loss:.6f}")
    print(f"  Final Test Loss:  {final_test_loss:.6f}")
    print("="*80 + "\n")
    
    return {
        'optimizer_name': optimizer_name,
        'train_losses': train_losses_per_epoch,
        'test_losses': test_losses_per_epoch,
        'final_train': final_train_loss,
        'final_test': final_test_loss
    }


def run_optimizer_experiment(config, output_dir='./bp_transformer_training_results'):
    """Run optimizer comparison experiment."""
    os.makedirs(output_dir, exist_ok=True)

    # use gpu if available
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
    print("BP TRANSFORMER TRAINING: OPTIMIZER COMPARISON")
    print(f"{'='*80}\n")
    
    # Test different optimizers
    optimizers_to_test = config.get('optimizers', ['sgd', 'adam', 'sgd_wd', 'adam_wd'])
    
    for opt_name in optimizers_to_test:
        opt_results = train_bp_transformer_optimizer(
            d_model=config['d_model'],
            n_heads=config['n_heads'],
            d_ff=config['d_ff'],
            seq_len=config['seq_len'],
            dataset=dataset,
            optimizer_name=opt_name,
            n_epochs=config['n_epochs'],
            lr=config['lr'],
            weight_decay=config.get('weight_decay', 1e-4),
            batch_size=config.get('batch_size', 1),
            train_split=0.8,
            seed=config['seed'],
            device=device
        )
        results['experiments'][opt_name] = opt_results
    
    # Create visualization
    create_visualization(results, output_dir)
    
    # Print summary table
    print_summary_table(results)
    
    # Save results
    results_file = os.path.join(output_dir, 'results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_file}")
    
    return results


def create_visualization(results, output_dir):
    """Create visualization plots."""
    experiments = results['experiments']
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Convergence curves
    ax = axes[0]
    for opt_name, exp in experiments.items():
        epochs = range(len(exp['train_losses']))
        label = opt_name.upper().replace('_', ' ')
        ax.plot(epochs, exp['train_losses'], '--', label=f'{label} Train', linewidth=2, alpha=0.7)
        ax.plot(epochs, exp['test_losses'], '-', label=f'{label} Test', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Convergence Curves: Different Optimizers', fontsize=14)
    ax.legend(fontsize=10)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Final losses comparison
    ax = axes[1]
    optimizers = list(experiments.keys())
    train_losses = [experiments[opt]['final_train'] for opt in optimizers]
    test_losses = [experiments[opt]['final_test'] for opt in optimizers]
    
    x = np.arange(len(optimizers))
    width = 0.35
    ax.bar(x - width/2, train_losses, width, label='Train Loss', alpha=0.7)
    ax.bar(x + width/2, test_losses, width, label='Test Loss', alpha=0.7)
    ax.set_xlabel('Optimizer', fontsize=12)
    ax.set_ylabel('Final Loss', fontsize=12)
    ax.set_title('Final Loss Comparison', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([opt.upper().replace('_', ' ') for opt in optimizers], rotation=45, ha='right')
    ax.legend(fontsize=11)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plot_file = os.path.join(output_dir, 'optimizer_comparison.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to {plot_file}")
    plt.close()


def print_summary_table(results):
    """Print a summary table of results."""
    print("\n" + "="*80)
    print("OPTIMIZER COMPARISON SUMMARY")
    print("="*80)
    print(f"{'Optimizer':<15} {'Train Loss':<15} {'Test Loss':<15}")
    print("-"*80)
    
    for opt_name, exp in sorted(results['experiments'].items()):
        train_loss = exp['final_train']
        test_loss = exp['final_test']
        opt_display = opt_name.upper().replace('_', ' ')
        print(f"{opt_display:<15} {train_loss:<15.6f} {test_loss:<15.6f}")
    
    print("="*80)


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='BP Transformer Training: Optimizer Comparison')
    parser.add_argument('--d_model', type=int, default=128, help='Model dimension')
    parser.add_argument('--n_heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--d_ff', type=int, default=512, help='Feed-forward dimension')
    parser.add_argument('--seq_len', type=int, default=10, help='Sequence length')
    parser.add_argument('--n_samples', type=int, default=5000, help='Dataset size')
    parser.add_argument('--n_epochs', type=int, default=50, help='Training epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay coefficient')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--optimizers', type=str, nargs='+', 
                       default=['sgd', 'adam', 'sgd_wd', 'adam_wd'],
                       choices=['sgd', 'adam', 'sgd_wd', 'adam_wd'],
                       help='Optimizers to test')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output_dir', type=str, default='./bp_transformer_training_results', 
                       help='Output directory')
    
    args = parser.parse_args()
    
    config = {
        'd_model': args.d_model,
        'n_heads': args.n_heads,
        'd_ff': args.d_ff,
        'seq_len': args.seq_len,
        'n_samples': args.n_samples,
        'n_epochs': args.n_epochs,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'batch_size': args.batch_size,
        'optimizers': args.optimizers,
        'seed': args.seed
    }
    
    print("="*80)
    print("BP TRANSFORMER TRAINING: OPTIMIZER COMPARISON")
    print("="*80)
    print(f"Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("="*80)
    
    results = run_optimizer_experiment(config, args.output_dir)
    
    print(f"\nResults saved to: {args.output_dir}")


if __name__ == '__main__':
    main()

