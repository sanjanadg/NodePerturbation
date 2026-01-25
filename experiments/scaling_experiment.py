#!/usr/bin/env python3
"""
Scaling Experiment: Compare convergence rates of DANP vs BP as depth and width increase.

This script systematically tests different network architectures (varying depth and width)
and compares how Node Perturbation (DANP) and Backpropagation (BP) converge.

The main hypothesis: As depth and width increase, DANP will converge at a slower rate
compared to BP, demonstrating the scalability limitations of node perturbation.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import time
from pathlib import Path
from tqdm import tqdm
import sys

# Import the MLP classes from toy_model
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'toy_model'))
from toy_model.danp import DANPMLP
from toy_model.bp import BPMLP
from toy_model.sythentic_data import SyntheticDataset
from torch.utils.data import DataLoader

class ConvergenceTracker:
    """Track convergence metrics during training."""
    def __init__(self):
        self.train_losses = []
        self.test_losses = []
        self.iterations = []
        
    def record(self, iteration, train_loss, test_loss):
        self.iterations.append(iteration)
        self.train_losses.append(train_loss)
        self.test_losses.append(test_loss)
    
    def get_convergence_rate(self, target_loss=0.01):
        """Calculate iterations to reach target loss."""
        if not self.test_losses:
            return len(self.iterations)
        for i, loss in enumerate(self.test_losses):
            if loss <= target_loss:
                return i + 1
        return len(self.test_losses)  # Didn't converge
    
    def get_final_loss(self):
        return self.test_losses[-1] if self.test_losses else float('inf')


def train_bp_model(layer_sizes, dataset, n_epochs=50, lr=1e-3, batch_size=1, 
                   train_split=0.8, seed=42, device='cpu'):
    """Train BP model and return convergence tracker."""
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
    
    tracker = ConvergenceTracker()
    
    # Training loop
    iteration = 0
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
        tracker.record(epoch, avg_train, avg_test)
        
        # Print epoch progress
        print(f"  Epoch {epoch+1:3d}/{n_epochs} | Train Loss: {avg_train:.6f} | Test Loss: {avg_test:.6f} | Time: {epoch_time:.3f}s")
    
    return tracker


def train_danp_model(layer_sizes, dataset, n_epochs=50, eta=1e-3, alpha=1e-4, 
                     sigma=0.001, batch_size=1, train_split=0.8, seed=42, device='cpu'):
    """Train DANP model and return convergence tracker."""
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
    
    tracker = ConvergenceTracker()
    
    # Training loop
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
        tracker.record(epoch, avg_train, avg_test)
        
        # Print epoch progress
        print(f"  Epoch {epoch+1:3d}/{n_epochs} | Train Loss: {avg_train:.6f} | Test Loss: {avg_test:.6f} | Time: {epoch_time:.3f}s")
    
    return tracker


def run_scaling_experiment(config, output_dir='./scaling_results'):
    """
    Run scaling experiment for a given configuration.
    
    Args:
        config: dict with keys:
            - input_dim, output_dim
            - depths: list of depths to test
            - widths: list of widths to test
            - n_samples, n_epochs
            - lr, sigma, alpha
            - seed
        output_dir: directory to save results
    """
    os.makedirs(output_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    results = {
        'config': config,
        'experiments': []
    }
    
    # Generate synthetic dataset
    print(f"\nGenerating synthetic dataset...")
    dataset = SyntheticDataset(
        input_dim=config['input_dim'],
        hidden_dim=config.get('teacher_hidden_dim', 48),
        output_dim=config['output_dim'],
        n_samples=config['n_samples'],
        seed=config['seed']
    )
    print(f"Dataset size: {len(dataset)}")
    
    # Test different architectures
    all_experiments = []
    total_experiments = len(config['depths']) * len(config['widths'])
    experiment_num = 0
    
    for depth in config['depths']:
        for width in config['widths']:
            experiment_num += 1
            # Build layer sizes: [input, hidden, ..., hidden, output]
            layer_sizes = [config['input_dim']]
            for _ in range(depth - 1):
                layer_sizes.append(width)
            layer_sizes.append(config['output_dim'])
            
            total_params = sum(layer_sizes[i] * layer_sizes[i+1] for i in range(len(layer_sizes)-1))
            
            print(f"\n{'='*80}")
            print(f"Experiment {experiment_num}/{total_experiments}")
            print(f"Testing: Depth={depth}, Width={width}, Params={total_params}")
            print(f"Architecture: {' → '.join(map(str, layer_sizes))}")
            print(f"{'='*80}")
            
            # Train BP
            print("\nTraining BP model...")
            bp_tracker = train_bp_model(
                layer_sizes=layer_sizes,
                dataset=dataset,
                n_epochs=config['n_epochs'],
                lr=config['lr'],
                batch_size=config.get('batch_size', 1),
                train_split=0.8,
                seed=config['seed'],
                device=device
            )
            
            # Train DANP
            print("\nTraining DANP model...")
            danp_tracker = train_danp_model(
                layer_sizes=layer_sizes,
                dataset=dataset,
                n_epochs=config['n_epochs'],
                eta=config['lr'],
                alpha=config.get('alpha', 1e-4),
                sigma=config.get('sigma', 0.001),
                batch_size=config.get('batch_size', 1),
                train_split=0.8,
                seed=config['seed'],
                device=device
            )
            
            # Calculate convergence metrics
            target_loss = config.get('target_loss', 0.01)
            bp_convergence = bp_tracker.get_convergence_rate(target_loss)
            danp_convergence = danp_tracker.get_convergence_rate(target_loss)
            
            experiment_result = {
                'depth': depth,
                'width': width,
                'layer_sizes': layer_sizes,
                'total_params': total_params,
                'bp': {
                    'final_train_loss': bp_tracker.get_final_loss(),
                    'final_test_loss': bp_tracker.test_losses[-1],
                    'convergence_iter': bp_convergence,
                    'train_losses': bp_tracker.train_losses,
                    'test_losses': bp_tracker.test_losses
                },
                'danp': {
                    'final_train_loss': danp_tracker.get_final_loss(),
                    'final_test_loss': danp_tracker.test_losses[-1],
                    'convergence_iter': danp_convergence,
                    'train_losses': danp_tracker.train_losses,
                    'test_losses': danp_tracker.test_losses
                },
                'convergence_ratio': danp_convergence / bp_convergence if bp_convergence > 0 else float('inf')
            }
            
            all_experiments.append(experiment_result)
            
            print(f"\nResults:")
            print(f"  BP - Final Loss: {experiment_result['bp']['final_test_loss']:.6f}, "
                  f"Convergence: {bp_convergence} epochs")
            print(f"  DANP - Final Loss: {experiment_result['danp']['final_test_loss']:.6f}, "
                  f"Convergence: {danp_convergence} epochs")
            print(f"  Convergence Ratio (DANP/BP): {experiment_result['convergence_ratio']:.2f}x")
    
    results['experiments'] = all_experiments
    
    # Save results
    results_file = os.path.join(output_dir, 'results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_file}")
    
    # Generate visualizations
    create_visualizations(results, output_dir)
    
    return results


def create_visualizations(results, output_dir):
    """Create visualization plots for the scaling experiment."""
    
    experiments = results['experiments']
    
    # 1. Convergence curves for different depths (fixed width)
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Convergence Comparison: DANP vs BP', fontsize=16)
    
    # Group by depth
    depths = sorted(set(e['depth'] for e in experiments))
    widths = sorted(set(e['width'] for e in experiments))
    
    # Plot 1: Convergence by depth (using first width)
    ax = axes[0, 0]
    if widths:
        fixed_width = widths[0]
        for depth in depths:
            exp = next((e for e in experiments if e['depth'] == depth and e['width'] == fixed_width), None)
            if exp:
                epochs = range(len(exp['bp']['test_losses']))
                ax.plot(epochs, exp['bp']['test_losses'], '--', label=f'BP (depth={depth})', linewidth=2)
                ax.plot(epochs, exp['danp']['test_losses'], '-', label=f'DANP (depth={depth})', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Test Loss')
        ax.set_title(f'Convergence by Depth (width={fixed_width})')
        ax.legend()
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
    
    # Plot 2: Convergence by width (using first depth)
    ax = axes[0, 1]
    if depths:
        fixed_depth = depths[0]
        for width in widths:
            exp = next((e for e in experiments if e['depth'] == fixed_depth and e['width'] == width), None)
            if exp:
                epochs = range(len(exp['bp']['test_losses']))
                ax.plot(epochs, exp['bp']['test_losses'], '--', label=f'BP (width={width})', linewidth=2)
                ax.plot(epochs, exp['danp']['test_losses'], '-', label=f'DANP (width={width})', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Test Loss')
        ax.set_title(f'Convergence by Width (depth={fixed_depth})')
        ax.legend()
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
    
    # Plot 3: Convergence ratio vs depth
    ax = axes[1, 0]
    if widths:
        fixed_width = widths[0]
        depth_vals = []
        ratio_vals = []
        for depth in sorted(depths):
            exp = next((e for e in experiments if e['depth'] == depth and e['width'] == fixed_width), None)
            if exp and exp['convergence_ratio'] != float('inf'):
                depth_vals.append(depth)
                ratio_vals.append(exp['convergence_ratio'])
        if depth_vals:
            ax.plot(depth_vals, ratio_vals, 'o-', linewidth=2, markersize=8)
            ax.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Equal convergence')
            ax.set_xlabel('Network Depth')
            ax.set_ylabel('Convergence Ratio (DANP/BP)')
            ax.set_title(f'Convergence Slowdown vs Depth (width={fixed_width})')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    # Plot 4: Convergence ratio vs width
    ax = axes[1, 1]
    if depths:
        fixed_depth = depths[0]
        width_vals = []
        ratio_vals = []
        for width in sorted(widths):
            exp = next((e for e in experiments if e['depth'] == fixed_depth and e['width'] == width), None)
            if exp and exp['convergence_ratio'] != float('inf'):
                width_vals.append(width)
                ratio_vals.append(exp['convergence_ratio'])
        if width_vals:
            ax.plot(width_vals, ratio_vals, 'o-', linewidth=2, markersize=8)
            ax.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Equal convergence')
            ax.set_xlabel('Network Width')
            ax.set_ylabel('Convergence Ratio (DANP/BP)')
            ax.set_title(f'Convergence Slowdown vs Width (depth={fixed_depth})')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_file = os.path.join(output_dir, 'convergence_comparison.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Convergence plots saved to {plot_file}")
    
    # 2. Heatmap of convergence ratios
    if len(depths) > 1 and len(widths) > 1:
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create matrix
        ratio_matrix = np.zeros((len(depths), len(widths)))
        for i, depth in enumerate(sorted(depths)):
            for j, width in enumerate(sorted(widths)):
                exp = next((e for e in experiments if e['depth'] == depth and e['width'] == width), None)
                if exp and exp['convergence_ratio'] != float('inf'):
                    ratio_matrix[i, j] = exp['convergence_ratio']
                else:
                    ratio_matrix[i, j] = np.nan
        
        im = ax.imshow(ratio_matrix, aspect='auto', cmap='YlOrRd', interpolation='nearest')
        ax.set_xticks(range(len(widths)))
        ax.set_xticklabels([str(w) for w in sorted(widths)])
        ax.set_yticks(range(len(depths)))
        ax.set_yticklabels([str(d) for d in sorted(depths)])
        ax.set_xlabel('Width')
        ax.set_ylabel('Depth')
        ax.set_title('Convergence Ratio Heatmap (DANP/BP)\nHigher = DANP converges slower')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Convergence Ratio (DANP/BP)', rotation=270, labelpad=20)
        
        # Add text annotations
        for i in range(len(depths)):
            for j in range(len(widths)):
                if not np.isnan(ratio_matrix[i, j]):
                    text = ax.text(j, i, f'{ratio_matrix[i, j]:.2f}',
                                 ha="center", va="center", color="black", fontsize=8)
        
        plt.tight_layout()
        heatmap_file = os.path.join(output_dir, 'convergence_heatmap.png')
        plt.savefig(heatmap_file, dpi=300, bbox_inches='tight')
        print(f"Heatmap saved to {heatmap_file}")
    
    plt.close('all')


def main():
    """Main function to run the scaling experiment."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Scaling experiment: DANP vs BP convergence')
    parser.add_argument('--input_dim', type=int, default=10, help='Input dimension')
    parser.add_argument('--output_dim', type=int, default=1, help='Output dimension')
    parser.add_argument('--depths', type=int, nargs='+', default=[2, 3, 4, 5], 
                       help='Network depths to test')
    parser.add_argument('--widths', type=int, nargs='+', default=[16, 32, 64, 128], 
                       help='Network widths to test')
    parser.add_argument('--n_samples', type=int, default=5000, help='Dataset size')
    parser.add_argument('--n_epochs', type=int, default=50, help='Training epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--sigma', type=float, default=0.001, help='DANP noise std')
    parser.add_argument('--alpha', type=float, default=1e-4, help='DANP decorrelation rate')
    parser.add_argument('--target_loss', type=float, default=0.01, 
                       help='Target loss for convergence metric')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output_dir', type=str, default='./scaling_results', 
                       help='Output directory')
    
    args = parser.parse_args()
    
    config = {
        'input_dim': args.input_dim,
        'output_dim': args.output_dim,
        'depths': args.depths,
        'widths': args.widths,
        'n_samples': args.n_samples,
        'n_epochs': args.n_epochs,
        'lr': args.lr,
        'sigma': args.sigma,
        'alpha': args.alpha,
        'target_loss': args.target_loss,
        'seed': args.seed,
        'teacher_hidden_dim': 48,
        'batch_size': 1
    }
    
    print("="*80)
    print("SCALING EXPERIMENT: DANP vs BP Convergence")
    print("="*80)
    print(f"Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("="*80)
    
    results = run_scaling_experiment(config, args.output_dir)
    
    # Print summary
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)
    print(f"{'Depth':<8} {'Width':<8} {'Params':<10} {'BP Conv':<10} {'DANP Conv':<12} {'Ratio':<8}")
    print("-"*80)
    for exp in sorted(results['experiments'], key=lambda x: (x['depth'], x['width'])):
        bp_conv = exp['bp']['convergence_iter']
        danp_conv = exp['danp']['convergence_iter']
        ratio = exp['convergence_ratio']
        print(f"{exp['depth']:<8} {exp['width']:<8} {exp['total_params']:<10} "
              f"{bp_conv:<10} {danp_conv:<12} {ratio:<8.2f}x")
    print("="*80)
    print(f"\nResults saved to: {args.output_dir}")


if __name__ == '__main__':
    main()

