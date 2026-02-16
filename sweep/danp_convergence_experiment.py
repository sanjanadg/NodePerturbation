"""
DANP Convergence Experiment

Trains DANP until convergence and compares terminal loss to achievable (teacher) loss.
Tests different hyperparameter combinations to find optimal settings.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import sys
from datetime import datetime
import time

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
sys.path.insert(0, os.path.join(parent_dir, 'toy_model'))

from toy_model.sythentic_data import SyntheticDataset
from toy_model.danp import DANPMLP


def compute_achievable_loss(dataset, test_indices, device='cpu'):
    """
    Compute the achievable loss using the teacher network.
    This represents the minimum possible loss (teacher's loss on test set).
    
    Args:
        dataset: SyntheticDataset instance (contains teacher network)
        test_indices: indices of test samples
        device: device to use
    
    Returns:
        achievable_loss: average MSE loss of teacher network on test set
    """
    losses = []
    with torch.no_grad():
        for idx in test_indices:
            x, y_target = dataset[idx]
            x = x.to(device)
            y_target = y_target.to(device)
            
            # Teacher network forward pass
            h = x
            for i, W in enumerate(dataset.W):
                h = W.to(device) @ h
                if i < len(dataset.W) - 1:
                    h = torch.relu(h)
            
            # Targets ARE teacher outputs, so this should be ~0
            # But we compute it anyway to verify
            loss = F.mse_loss(h, y_target).item()
            losses.append(loss)
    
    return np.mean(losses)


def train_danp_until_convergence(layer_sizes, dataset, eta=1e-3, alpha=1e-4, sigma=0.001,
                                 batch_size=1, train_split=0.8, seed=42, device='cpu',
                                 max_epochs=500, patience=20, min_delta=1e-6,
                                 convergence_threshold=1e-5):
    """
    Train DANP model until convergence.
    
    Args:
        layer_sizes: list of layer dimensions
        dataset: SyntheticDataset instance
        eta: weight learning rate
        alpha: decorrelation learning rate
        sigma: noise std
        batch_size: batch size
        train_split: fraction of data for training
        seed: random seed
        device: device to use
        max_epochs: maximum number of epochs (safety limit)
        patience: number of epochs to wait for improvement before stopping
        min_delta: minimum change in loss to be considered improvement
        convergence_threshold: relative change threshold for convergence
    
    Returns:
        dict with training history and final results
    """
    torch.manual_seed(seed + 1000)  # Model initialization seed
    
    # Split dataset
    train_size = int(train_split * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size],
        generator=torch.Generator().manual_seed(seed)
    )
    
    # Get test indices for achievable loss computation
    test_indices = test_dataset.indices
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Create model
    model = DANPMLP(layer_sizes, sigma=sigma, eta=eta, alpha=alpha)
    # Move model parameters to device
    for i in range(len(model.W)):
        model.W[i] = model.W[i].to(device)
    for i in range(len(model.R)):
        model.R[i] = model.R[i].to(device)
    
    # Compute achievable loss (teacher network loss)
    achievable_loss = compute_achievable_loss(dataset, test_indices, device=device)
    
    # Training history
    train_losses_per_epoch = []
    test_losses_per_epoch = []
    epoch_times = []
    
    best_test_loss = float('inf')
    epochs_without_improvement = 0
    converged = False
    
    print(f"\n{'='*80}")
    print(f"Training DANP until convergence")
    print(f"{'='*80}")
    print(f"Hyperparameters: eta={eta:.4e}, alpha={alpha:.4e}, sigma={sigma:.4e}")
    print(f"Architecture: {layer_sizes}")
    print(f"Achievable loss (teacher): {achievable_loss:.8e}")
    print(f"Max epochs: {max_epochs}, Patience: {patience}, Min delta: {min_delta:.2e}")
    print(f"{'='*80}\n")
    
    start_time = time.time()
    
    for epoch in range(max_epochs):
        epoch_start_time = time.time()
        
        # Training updates
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            # Process each sample individually
            for i in range(x.shape[0]):
                x_sample = x[i]
                y_sample = y[i]
                model.step(x_sample, y_sample)
        
        # Evaluate on training set
        train_losses = []
        with torch.no_grad():
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
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
                for i in range(x.shape[0]):
                    x_sample = x[i]
                    y_sample = y[i]
                    x_clean, _ = model.forward_clean(x_sample)
                    loss = F.mse_loss(x_clean[-1], y_sample).item()
                    test_losses.append(loss)
        
        avg_train = np.mean(train_losses)
        avg_test = np.mean(test_losses)
        epoch_time = time.time() - epoch_start_time
        
        train_losses_per_epoch.append(avg_train)
        test_losses_per_epoch.append(avg_test)
        epoch_times.append(epoch_time)
        
        # Check for improvement
        improvement = best_test_loss - avg_test
        if improvement > min_delta:
            best_test_loss = avg_test
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        
        # Check convergence
        if epoch > 0:
            # Relative change in test loss
            prev_test_loss = test_losses_per_epoch[-2]
            relative_change = abs(avg_test - prev_test_loss) / (prev_test_loss + 1e-10)
            
            if relative_change < convergence_threshold:
                converged = True
                print(f"  Epoch {epoch+1:4d} | Train: {avg_train:.8f} | Test: {avg_test:.8f} | "
                      f"Time: {epoch_time:.3f}s | CONVERGED (relative change: {relative_change:.2e})")
                break
        
        # Early stopping
        if epochs_without_improvement >= patience:
            print(f"  Epoch {epoch+1:4d} | Train: {avg_train:.8f} | Test: {avg_test:.8f} | "
                  f"Time: {epoch_time:.3f}s | Early stopping (no improvement for {patience} epochs)")
            break
        
        # Print progress every 10 epochs or if significant improvement
        if (epoch + 1) % 10 == 0 or improvement > min_delta * 10:
            print(f"  Epoch {epoch+1:4d} | Train: {avg_train:.8f} | Test: {avg_test:.8f} | "
                  f"Time: {epoch_time:.3f}s | Best: {best_test_loss:.8f} | "
                  f"Gap to achievable: {avg_test - achievable_loss:.8f}")
    
    total_time = time.time() - start_time
    
    final_train_loss = train_losses_per_epoch[-1]
    final_test_loss = test_losses_per_epoch[-1]
    
    # Compute metrics
    gap_to_achievable = final_test_loss - achievable_loss
    relative_gap = gap_to_achievable / (achievable_loss + 1e-10)
    
    print(f"\n{'='*80}")
    print("CONVERGENCE RESULTS")
    print(f"{'='*80}")
    print(f"Converged: {converged}")
    print(f"Total epochs: {len(train_losses_per_epoch)}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Final train loss: {final_train_loss:.8f}")
    print(f"Final test loss:  {final_test_loss:.8f}")
    print(f"Achievable loss:  {achievable_loss:.8f}")
    print(f"Gap to achievable: {gap_to_achievable:.8f}")
    print(f"Relative gap: {relative_gap:.4%}")
    print(f"{'='*80}\n")
    
    return {
        'train_losses': train_losses_per_epoch,
        'test_losses': test_losses_per_epoch,
        'epoch_times': epoch_times,
        'final_train': final_train_loss,
        'final_test': final_test_loss,
        'achievable_loss': achievable_loss,
        'gap_to_achievable': gap_to_achievable,
        'relative_gap': relative_gap,
        'converged': converged,
        'total_epochs': len(train_losses_per_epoch),
        'total_time': total_time,
        'hyperparameters': {
            'eta': eta,
            'alpha': alpha,
            'sigma': sigma
        }
    }


def run_hyperparameter_sweep(layer_sizes, teacher_sizes, n_samples=5000, 
                             train_split=0.8, seed=42, device='cpu',
                             max_epochs=500, patience=20, output_dir=None):
    """
    Run convergence experiments for different hyperparameter combinations.
    
    Args:
        layer_sizes: student network architecture
        teacher_sizes: teacher network architecture
        n_samples: number of samples in dataset
        train_split: fraction for training
        seed: random seed
        device: device to use
        max_epochs: maximum epochs per run
        patience: early stopping patience
        output_dir: output directory
    
    Returns:
        dict with all results
    """
    if output_dir is None:
        output_dir = f"danp_convergence_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Hyperparameter combinations based on user's context
    # eta=1e-3, alpha=[1e-4, 1e-3], sigma=1e-3
    hyperparameter_configs = [
        {'eta': 1e-3, 'alpha': 1e-4, 'sigma': 1e-3},
        {'eta': 1e-3, 'alpha': 1e-3, 'sigma': 1e-3},
        # Additional combinations for comparison
        {'eta': 1e-3, 'alpha': 1e-4, 'sigma': 0.01},
        {'eta': 1e-3, 'alpha': 1e-3, 'sigma': 0.01},
        {'eta': 1e-2, 'alpha': 1e-4, 'sigma': 1e-3},
        {'eta': 1e-2, 'alpha': 1e-3, 'sigma': 1e-3},
    ]
    
    print("="*80)
    print("DANP CONVERGENCE EXPERIMENT")
    print("="*80)
    print(f"Student architecture: {layer_sizes}")
    print(f"Teacher architecture: {teacher_sizes}")
    print(f"Number of hyperparameter configs: {len(hyperparameter_configs)}")
    print("="*80)
    
    # Create dataset
    dataset = SyntheticDataset(
        input_dim=teacher_sizes[0],
        hidden_dim=teacher_sizes[1] if len(teacher_sizes) == 3 else None,
        output_dim=teacher_sizes[-1],
        n_samples=n_samples,
        seed=seed,
        teacher_layer_sizes=teacher_sizes
    )
    
    all_results = {}
    
    for i, config in enumerate(hyperparameter_configs):
        print(f"\n{'='*80}")
        print(f"Config {i+1}/{len(hyperparameter_configs)}: eta={config['eta']:.4e}, "
              f"alpha={config['alpha']:.4e}, sigma={config['sigma']:.4e}")
        print(f"{'='*80}")
        
        results = train_danp_until_convergence(
            layer_sizes=layer_sizes,
            dataset=dataset,
            eta=config['eta'],
            alpha=config['alpha'],
            sigma=config['sigma'],
            batch_size=1,
            train_split=train_split,
            seed=seed,
            device=device,
            max_epochs=max_epochs,
            patience=patience
        )
        
        config_key = f"eta_{config['eta']:.0e}_alpha_{config['alpha']:.0e}_sigma_{config['sigma']:.0e}"
        all_results[config_key] = results
    
    # Save results
    results_file = os.path.join(output_dir, 'results.json')
    
    # Convert to JSON-serializable format
    serializable_results = {}
    for key, value in all_results.items():
        serializable_results[key] = {
            'train_losses': [float(x) for x in value['train_losses']],
            'test_losses': [float(x) for x in value['test_losses']],
            'epoch_times': [float(x) for x in value['epoch_times']],
            'final_train': float(value['final_train']),
            'final_test': float(value['final_test']),
            'achievable_loss': float(value['achievable_loss']),
            'gap_to_achievable': float(value['gap_to_achievable']),
            'relative_gap': float(value['relative_gap']),
            'converged': value['converged'],
            'total_epochs': value['total_epochs'],
            'total_time': float(value['total_time']),
            'hyperparameters': value['hyperparameters']
        }
    
    with open(results_file, 'w') as f:
        json.dump({
            'all_results': serializable_results,
            'layer_sizes': layer_sizes,
            'teacher_sizes': teacher_sizes,
            'n_samples': n_samples,
            'seed': seed
        }, f, indent=2)
    
    print(f"\nResults saved to {results_file}")
    
    return {
        'all_results': all_results,
        'output_dir': output_dir,
        'layer_sizes': layer_sizes,
        'teacher_sizes': teacher_sizes
    }


def create_visualizations(results_dict, output_dir):
    """
    Create visualizations for convergence experiments.
    """
    all_results = results_dict['all_results']
    
    # 1. Loss curves for all hyperparameter configs
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_results)))
    
    for (config_key, results), color in zip(all_results.items(), colors):
        epochs = range(1, len(results['train_losses']) + 1)
        
        # Train loss
        axes[0].plot(epochs, results['train_losses'], 
                    label=f"Train: {config_key}", color=color, alpha=0.7, linestyle='--')
        
        # Test loss
        axes[1].plot(epochs, results['test_losses'], 
                    label=f"Test: {config_key}", color=color, alpha=0.9, linewidth=2)
        
        # Achievable loss line
        if config_key == list(all_results.keys())[0]:  # Only plot once
            axes[0].axhline(y=results['achievable_loss'], color='red', 
                          linestyle=':', linewidth=2, label='Achievable Loss')
            axes[1].axhline(y=results['achievable_loss'], color='red', 
                          linestyle=':', linewidth=2, label='Achievable Loss')
    
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Train Loss')
    axes[0].set_title('Training Loss vs Epochs (All Hyperparameter Configs)')
    axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_yscale('log')
    
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Test Loss')
    axes[1].set_title('Test Loss vs Epochs (All Hyperparameter Configs)')
    axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_yscale('log')
    
    plt.tight_layout()
    loss_curves_file = os.path.join(output_dir, 'loss_curves_all_configs.png')
    plt.savefig(loss_curves_file, dpi=300, bbox_inches='tight')
    print(f"Loss curves saved to {loss_curves_file}")
    plt.close()
    
    # 2. Comparison of final losses and gaps
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    config_names = list(all_results.keys())
    final_test_losses = [all_results[k]['final_test'] for k in config_names]
    final_train_losses = [all_results[k]['final_train'] for k in config_names]
    achievable_losses = [all_results[k]['achievable_loss'] for k in config_names]
    gaps = [all_results[k]['gap_to_achievable'] for k in config_names]
    relative_gaps = [all_results[k]['relative_gap'] * 100 for k in config_names]  # Convert to %
    total_epochs = [all_results[k]['total_epochs'] for k in config_names]
    total_times = [all_results[k]['total_time'] for k in config_names]
    
    # Final test loss comparison
    axes[0, 0].bar(range(len(config_names)), final_test_losses, alpha=0.7, label='Final Test Loss')
    axes[0, 0].axhline(y=achievable_losses[0], color='red', linestyle='--', 
                      linewidth=2, label='Achievable Loss')
    axes[0, 0].set_xlabel('Hyperparameter Config')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Final Test Loss Comparison')
    axes[0, 0].set_xticks(range(len(config_names)))
    axes[0, 0].set_xticklabels([k.replace('_', '\n') for k in config_names], 
                               rotation=45, ha='right', fontsize=8)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_yscale('log')
    
    # Gap to achievable
    axes[0, 1].bar(range(len(config_names)), gaps, alpha=0.7, color='orange')
    axes[0, 1].set_xlabel('Hyperparameter Config')
    axes[0, 1].set_ylabel('Gap to Achievable Loss')
    axes[0, 1].set_title('Gap to Achievable Loss')
    axes[0, 1].set_xticks(range(len(config_names)))
    axes[0, 1].set_xticklabels([k.replace('_', '\n') for k in config_names], 
                               rotation=45, ha='right', fontsize=8)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_yscale('log')
    
    # Relative gap (%)
    axes[1, 0].bar(range(len(config_names)), relative_gaps, alpha=0.7, color='green')
    axes[1, 0].set_xlabel('Hyperparameter Config')
    axes[1, 0].set_ylabel('Relative Gap (%)')
    axes[1, 0].set_title('Relative Gap to Achievable Loss (%)')
    axes[1, 0].set_xticks(range(len(config_names)))
    axes[1, 0].set_xticklabels([k.replace('_', '\n') for k in config_names], 
                               rotation=45, ha='right', fontsize=8)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Convergence time
    axes[1, 1].bar(range(len(config_names)), total_times, alpha=0.7, color='purple')
    axes[1, 1].set_xlabel('Hyperparameter Config')
    axes[1, 1].set_ylabel('Time (seconds)')
    axes[1, 1].set_title('Time to Convergence')
    axes[1, 1].set_xticks(range(len(config_names)))
    axes[1, 1].set_xticklabels([k.replace('_', '\n') for k in config_names], 
                               rotation=45, ha='right', fontsize=8)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    comparison_file = os.path.join(output_dir, 'hyperparameter_comparison.png')
    plt.savefig(comparison_file, dpi=300, bbox_inches='tight')
    print(f"Hyperparameter comparison saved to {comparison_file}")
    plt.close()
    
    # 3. Summary table
    print("\n" + "="*80)
    print("HYPERPARAMETER COMPARISON SUMMARY")
    print("="*80)
    print(f"{'Config':<40} {'Final Test':<12} {'Gap':<12} {'Rel Gap %':<10} {'Epochs':<8} {'Time (s)':<10}")
    print("-"*80)
    
    for config_key in config_names:
        r = all_results[config_key]
        print(f"{config_key:<40} {r['final_test']:<12.6e} {r['gap_to_achievable']:<12.6e} "
              f"{r['relative_gap']*100:<10.2f} {r['total_epochs']:<8} {r['total_time']:<10.2f}")
    
    print("="*80)
    
    # Find best config
    best_config = min(config_names, key=lambda k: all_results[k]['final_test'])
    best_results = all_results[best_config]
    
    print(f"\nBEST CONFIGURATION: {best_config}")
    print(f"  Final test loss: {best_results['final_test']:.8f}")
    print(f"  Gap to achievable: {best_results['gap_to_achievable']:.8f}")
    print(f"  Relative gap: {best_results['relative_gap']*100:.2f}%")
    print(f"  Total epochs: {best_results['total_epochs']}")
    print(f"  Total time: {best_results['total_time']:.2f}s")
    print("="*80 + "\n")


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='DANP convergence experiment')
    parser.add_argument('--layer_sizes', type=int, nargs='+', default=[10, 32, 1],
                       help='Student network layer sizes (default: 10 32 1)')
    parser.add_argument('--teacher_sizes', type=int, nargs='+', default=[10, 32, 1],
                       help='Teacher network layer sizes (default: 10 32 1)')
    parser.add_argument('--n_samples', type=int, default=5000,
                       help='Number of samples in dataset (default: 5000)')
    parser.add_argument('--max_epochs', type=int, default=500,
                       help='Maximum epochs per run (default: 500)')
    parser.add_argument('--patience', type=int, default=20,
                       help='Early stopping patience (default: 20)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device to use (default: cpu)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory (default: auto-generated)')
    
    args = parser.parse_args()
    
    # Run experiment
    results_dict = run_hyperparameter_sweep(
        layer_sizes=args.layer_sizes,
        teacher_sizes=args.teacher_sizes,
        n_samples=args.n_samples,
        seed=args.seed,
        device=args.device,
        max_epochs=args.max_epochs,
        patience=args.patience,
        output_dir=args.output_dir
    )
    
    # Create visualizations
    print("\nCreating visualizations...")
    create_visualizations(results_dict, results_dict['output_dir'])
    
    print(f"\nExperiment complete! Results saved to: {results_dict['output_dir']}")


if __name__ == '__main__':
    main()

