#!/usr/bin/env python3
"""
Batch Size Experiment: Compare performance of DANP vs BP across different batch sizes.

This script tests how batch size affects the convergence and final performance
of both Backpropagation (BP) and Decorrelated Activity-based Node Perturbation (DANP).


# With linear scaling (recommended)
python3 batch_size_experiment.py --lr_scaling linear

# With square root scaling
python3 batch_size_experiment.py --lr_scaling sqrt

# Without scaling (your current behavior)
python3 batch_size_experiment.py --lr_scaling none

python3 batch_size_experiment.py --layer_sizes 10 128 128 128 128 1 --output_dir ./bs_large_lin_2 --lr_scaling linear

python3 batch_size_experiment.py --layer_sizes 10 128 128 128 128 1 --output_dir ./bs_large_lin_2 --lr_scaling linear

"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from datetime import datetime

# Import the training functions and dataset
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'toy_model'))
from toy_model.sythentic_data import SyntheticDataset, train_bp_model, train_danp_model_batch


def run_batch_size_experiment(config, output_dir='./batch_size_results'):
    """
    Run batch size experiment for different batch sizes.
    """
    os.makedirs(output_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Generate synthetic dataset
    print(f"\nGenerating synthetic dataset...")
    # Create teacher network that matches student network depth (but can be simpler)
    layer_sizes = config['layer_sizes']
    if len(layer_sizes) > 3:
        # For deeper student networks, create a deeper teacher network
        # Use same input/output, but use max width and create a teacher with depth = student_depth - 1
        # or at least 2-3 layers
        max_width = max(layer_sizes[1:-1])
        student_depth = len(layer_sizes) - 1
        # Teacher depth: dependent on the student network depth
        teacher_depth = max(2,student_depth)
        teacher_layer_sizes = [config['input_dim']]
        for _ in range(teacher_depth - 1):
            teacher_layer_sizes.append(max_width)
        teacher_layer_sizes.append(config['output_dim'])
    else:
        # Simple network: use default 2-layer teacher
        teacher_layer_sizes = None
        max_width = config.get('teacher_hidden_dim', layer_sizes[1] if len(layer_sizes) > 1 else 32)
    
    dataset = SyntheticDataset(
        input_dim=config['input_dim'],
        hidden_dim=max_width,  # Used only if teacher_layer_sizes is None
        output_dim=config['output_dim'],
        n_samples=config['n_samples'],
        seed=config['seed'],
        teacher_layer_sizes=teacher_layer_sizes
    )
    print(f"Dataset size: {len(dataset)}")
    if teacher_layer_sizes:
        print(f"Teacher network: {' → '.join(map(str, teacher_layer_sizes))}")
    else:
        print(f"Teacher network: {config['input_dim']} → {max_width} → {config['output_dim']}")
    
    results = {
        'config': config,
        'experiments': []
    }
    
    print(f"\n{'='*80}")
    print("BATCH SIZE EXPERIMENT: DANP vs BP")
    print(f"{'='*80}\n")
    
    # Base learning rate and batch size for scaling
    base_lr = config['lr']
    base_batch_size = config.get('base_batch_size', 1)  # Reference batch size
    lr_scaling = config.get('lr_scaling', 'linear')  # 'linear', 'sqrt', or 'none'
    
    # Test different batch sizes
    for batch_size in config['batch_sizes']:
        print(f"\n{'='*80}")
        print(f"Testing Batch Size: {batch_size}")
        print(f"{'='*80}\n")
        
        # Scale learning rate based on batch size
        if lr_scaling == 'linear':
            # Linear scaling: lr = base_lr * (batch_size / base_batch_size)
            scaled_lr = base_lr * (batch_size / base_batch_size)
        elif lr_scaling == 'sqrt':
            # Square root scaling: lr = base_lr * sqrt(batch_size / base_batch_size)
            scaled_lr = base_lr * np.sqrt(batch_size / base_batch_size)
        else:  # 'none'
            scaled_lr = base_lr
        
        print(f"Using learning rate: {scaled_lr:.6f} (scaling: {lr_scaling})")
        
        # Train BP model
        print(f"\nTraining BP model with batch_size={batch_size}, lr={scaled_lr:.6f}...")
        bp_results = train_bp_model(
            layer_sizes=config['layer_sizes'],
            dataset=dataset,
            n_epochs=config['n_epochs'],
            lr=scaled_lr,  # ← Use scaled learning rate
            batch_size=batch_size,
            train_split=0.8,
            seed=config['seed'],
            device=device
        )
        
        # Train DANP model with batch processing
        # NOTE: train_danp_model_batch scales eta by batch_size internally,
        # so we pass the base learning rate (not the scaled one) to avoid double scaling
        print(f"\nTraining DANP model with batch_size={batch_size}, base_lr={base_lr:.6f}...")
        danp_results = train_danp_model_batch(
            layer_sizes=config['layer_sizes'],
            dataset=dataset,
            n_epochs=config['n_epochs'],
            eta=base_lr,  # ← Pass base learning rate (will be scaled inside function)
            alpha=config.get('alpha', 1e-4),
            sigma=config.get('sigma', 0.001),
            batch_size=batch_size,
            train_split=0.8,
            seed=config['seed'],
            device=device
        )
        
        # Store results
        experiment_result = {
            'batch_size': batch_size,
            'learning_rate': scaled_lr,  # Store the scaled LR
            'bp': {
                'final_train_loss': bp_results['final_train'],
                'final_test_loss': bp_results['final_test'],
                'train_losses': bp_results['train_losses'],
                'test_losses': bp_results['test_losses']
            },
            'danp': {
                'final_train_loss': danp_results['final_train'],
                'final_test_loss': danp_results['final_test'],
                'train_losses': danp_results['train_losses'],
                'test_losses': danp_results['test_losses']
            }
        }
        
        results['experiments'].append(experiment_result)
        
        print(f"\nBatch Size {batch_size} Results (lr={scaled_lr:.6f}):")
        print(f"  BP   - Final Train Loss: {bp_results['final_train']:.6f}, Final Test Loss: {bp_results['final_test']:.6f}")
        print(f"  DANP - Final Train Loss: {danp_results['final_train']:.6f}, Final Test Loss: {danp_results['final_test']:.6f}")
    
    # Create visualizations
    create_visualizations(results, output_dir)
    
    # Print summary table
    print_summary_table(results)
    
    return results


def create_visualizations(results, output_dir):
    """Create visualization plots for the batch size experiment."""
    
    experiments = results['experiments']
    batch_sizes = [exp['batch_size'] for exp in experiments]
    
    # 1. Final loss vs batch size
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Final train loss
    ax = axes[0]
    bp_train_losses = [exp['bp']['final_train_loss'] for exp in experiments]
    danp_train_losses = [exp['danp']['final_train_loss'] for exp in experiments]
    
    ax.plot(batch_sizes, bp_train_losses, 'o-', label='BP', linewidth=2, markersize=8)
    ax.plot(batch_sizes, danp_train_losses, 's-', label='DANP', linewidth=2, markersize=8)
    ax.set_xlabel('Batch Size', fontsize=12)
    ax.set_ylabel('Final Train Loss', fontsize=12)
    ax.set_title('Final Train Loss vs Batch Size', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    
    # Plot 2: Final test loss
    ax = axes[1]
    bp_test_losses = [exp['bp']['final_test_loss'] for exp in experiments]
    danp_test_losses = [exp['danp']['final_test_loss'] for exp in experiments]
    
    ax.plot(batch_sizes, bp_test_losses, 'o-', label='BP', linewidth=2, markersize=8)
    ax.plot(batch_sizes, danp_test_losses, 's-', label='DANP', linewidth=2, markersize=8)
    ax.set_xlabel('Batch Size', fontsize=12)
    ax.set_ylabel('Final Test Loss', fontsize=12)
    ax.set_title('Final Test Loss vs Batch Size', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    
    plt.tight_layout()
    plot_file = os.path.join(output_dir, 'final_loss_vs_batch_size.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"\nFinal loss plot saved to {plot_file}")
    plt.close()
    
    # 2. Convergence curves for different batch sizes
    n_batch_sizes = len(batch_sizes)
    n_cols = min(3, n_batch_sizes)
    n_rows = (n_batch_sizes + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes
    else:
        axes = axes.flatten()
    
    fig.suptitle('Convergence Curves: Train and Test Loss vs Epoch', fontsize=16)
    
    for idx, exp in enumerate(experiments):
        ax = axes[idx]
        batch_size = exp['batch_size']
        epochs = range(len(exp['bp']['train_losses']))
        
        ax.plot(epochs, exp['bp']['train_losses'], '--', label=f'BP Train', linewidth=2, alpha=0.7)
        ax.plot(epochs, exp['bp']['test_losses'], '-', label=f'BP Test', linewidth=2)
        ax.plot(epochs, exp['danp']['train_losses'], '--', label=f'DANP Train', linewidth=2, alpha=0.7)
        ax.plot(epochs, exp['danp']['test_losses'], '-', label=f'DANP Test', linewidth=2)
        
        ax.set_xlabel('Epoch', fontsize=10)
        ax.set_ylabel('Loss', fontsize=10)
        ax.set_title(f'Batch Size = {batch_size}', fontsize=12)
        ax.legend(fontsize=9)
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(len(experiments), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plot_file = os.path.join(output_dir, 'convergence_curves.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Convergence curves plot saved to {plot_file}")
    plt.close()


def print_summary_table(results):
    """Print a summary table of results."""
    print("\n" + "="*90)
    print("EXPERIMENT SUMMARY")
    print("="*90)
    print(f"{'Batch Size':<12} {'LR':<12} {'BP Train':<12} {'BP Test':<12} {'DANP Train':<14} {'DANP Test':<14}")
    print("-"*90)
    
    for exp in results['experiments']:
        batch_size = exp['batch_size']
        lr = exp['learning_rate']
        bp_train = exp['bp']['final_train_loss']
        bp_test = exp['bp']['final_test_loss']
        danp_train = exp['danp']['final_train_loss']
        danp_test = exp['danp']['final_test_loss']
        
        print(f"{batch_size:<12} {lr:<12.6f} {bp_train:<12.6f} {bp_test:<12.6f} {danp_train:<14.6f} {danp_test:<14.6f}")
    
    print("="*90)


def main():
    """Main function to run the batch size experiment."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Batch size experiment: DANP vs BP performance')
    parser.add_argument('--input_dim', type=int, default=10, help='Input dimension')
    parser.add_argument('--hidden_dim', type=int, default=32, help='Hidden layer dimension (for simple network)')
    parser.add_argument('--output_dim', type=int, default=1, help='Output dimension')
    parser.add_argument('--layer_sizes', type=int, nargs='+', default=None,
                       help='Custom layer sizes (e.g., --layer_sizes 10 128 128 128 128 1). If not provided, uses [input_dim, hidden_dim, output_dim]')
    parser.add_argument('--n_samples', type=int, default=5000, help='Dataset size')
    parser.add_argument('--n_epochs', type=int, default=50, help='Training epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Base learning rate (for batch_size=1)')
    parser.add_argument('--sigma', type=float, default=0.001, help='DANP noise std')
    parser.add_argument('--alpha', type=float, default=1e-4, help='DANP decorrelation rate')
    parser.add_argument('--batch_sizes', type=int, nargs='+', default=[32], 
                       help='Batch sizes to test')
    parser.add_argument('--lr_scaling', type=str, default='linear', 
                       choices=['linear', 'sqrt', 'none'],
                       help='Learning rate scaling strategy')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output_dir', type=str, default=None, 
                       help='Output directory (default: ./batch_size_results_YYYYMMDD)')
    
    args = parser.parse_args()
    
    # Create date-stamped output directory if not provided
    if args.output_dir is None:
        date_str = datetime.now().strftime('%Y%m%d')
        args.output_dir = f'./batch_size_results_{date_str}'
    
    # Determine layer sizes
    if args.layer_sizes is not None:
        layer_sizes = args.layer_sizes
        if len(layer_sizes) < 2:
            raise ValueError("layer_sizes must have at least 2 elements (input and output)")
        input_dim = layer_sizes[0]
        output_dim = layer_sizes[-1]
        # Use first hidden layer size for teacher network (or default)
        teacher_hidden_dim = layer_sizes[1] if len(layer_sizes) > 2 else args.hidden_dim
    else:
        # Default simple network
        layer_sizes = [args.input_dim, args.hidden_dim, args.output_dim]
        input_dim = args.input_dim
        output_dim = args.output_dim
        teacher_hidden_dim = args.hidden_dim
    
    config = {
        'input_dim': input_dim,
        'output_dim': output_dim,
        'teacher_hidden_dim': teacher_hidden_dim,
        'layer_sizes': layer_sizes,
        'n_samples': args.n_samples,
        'n_epochs': args.n_epochs,
        'lr': args.lr,
        'sigma': args.sigma,
        'alpha': args.alpha,
        'batch_sizes': args.batch_sizes,
        'base_batch_size': 1,  # ← Add this
        'lr_scaling': args.lr_scaling,  # ← Add this
        'seed': args.seed
    }
    
    print("="*80)
    print("BATCH SIZE EXPERIMENT: DANP vs BP")
    print("="*80)
    print(f"Configuration:")
    print(f"  Layer sizes: {config['layer_sizes']}")
    print(f"  Network depth: {len(config['layer_sizes']) - 1}")
    print(f"  Network width: {max(config['layer_sizes'][1:-1]) if len(config['layer_sizes']) > 2 else config['layer_sizes'][1]}")
    for key, value in config.items():
        if key != 'layer_sizes':  # Already printed above
            print(f"  {key}: {value}")
    print("="*80)
    
    results = run_batch_size_experiment(config, args.output_dir)
    
    print(f"\nResults saved to: {args.output_dir}")


if __name__ == '__main__':
    main()

