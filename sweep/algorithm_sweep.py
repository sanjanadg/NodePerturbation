"""
Comprehensive sweep experiment comparing BP, DANP (NP), and WP across:
- Model sizes (student network complexity)
- Problem dimensions (teacher network complexity)
- Algorithm hyperparameters (learning rates)

Results are visualized as heatmaps showing which method performs best in each scenario.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from datetime import datetime
import sys

# Add parent directory to path to import toy_model modules
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
sys.path.insert(0, os.path.join(parent_dir, 'toy_model'))

from toy_model.sythentic_data import SyntheticDataset, train_bp_model, train_danp_model, train_wp_model


def run_single_experiment(model_sizes, teacher_sizes, bp_lr, danp_eta, danp_alpha, danp_sigma, 
                          wp_alpha, wp_sigma, n_epochs=30, n_samples=5000, seed=42, device='cpu'):
    """
    Run a single experiment comparing BP, DANP, and WP.
    
    Args:
        model_sizes: List of layer sizes for student network (e.g., [10, 32, 1])
        teacher_sizes: List of layer sizes for teacher network (e.g., [10, 32, 1])
        bp_lr: Learning rate for BP
        danp_eta: Learning rate (eta) for DANP
        danp_alpha: Decorrelation rate (alpha) for DANP
        danp_sigma: Noise std (sigma) for DANP
        wp_alpha: Learning rate (alpha) for WP
        wp_sigma: Noise std (sigma) for WP
        n_epochs: Number of training epochs
        n_samples: Number of samples in synthetic dataset
        seed: Random seed
        device: Device to use
    
    Returns:
        dict with final test losses for each method
    """
    results = {}
    
    # Create synthetic dataset with teacher network
    dataset = SyntheticDataset(
        input_dim=teacher_sizes[0],
        hidden_dim=teacher_sizes[1] if len(teacher_sizes) == 3 else None,
        output_dim=teacher_sizes[-1],
        n_samples=n_samples,
        seed=seed,
        teacher_layer_sizes=teacher_sizes
    )
    
    # Train BP
    try:
        bp_results = train_bp_model(
            layer_sizes=model_sizes,
            dataset=dataset,
            n_epochs=n_epochs,
            lr=bp_lr,
            batch_size=1,
            seed=seed,
            device=device
        )
        results['bp'] = bp_results['final_test']
    except Exception as e:
        print(f"BP failed: {e}")
        results['bp'] = float('inf')
    
    # Train DANP
    try:
        danp_results = train_danp_model(
            layer_sizes=model_sizes,
            dataset=dataset,
            n_epochs=n_epochs,
            eta=danp_eta,
            alpha=danp_alpha,
            sigma=danp_sigma,
            batch_size=1,
            seed=seed,
            device=device
        )
        results['danp'] = danp_results['final_test']
    except Exception as e:
        print(f"DANP failed: {e}")
        results['danp'] = float('inf')
    
    # Train WP
    try:
        wp_results = train_wp_model(
            layer_sizes=model_sizes,
            dataset=dataset,
            n_epochs=n_epochs,
            sigma=wp_sigma,
            alpha=wp_alpha,
            batch_size=1,
            population_size=1,
            seed=seed,
            device=device
        )
        results['wp'] = wp_results['final_test']
    except Exception as e:
        print(f"WP failed: {e}")
        results['wp'] = float('inf')
    
    return results


def determine_best_method(results):
    """Determine which method has the lowest test loss."""
    if all(v == float('inf') for v in results.values()):
        return None
    
    # Filter out infinite values
    valid_results = {k: v for k, v in results.items() if v != float('inf') and not np.isnan(v)}
    
    if not valid_results:
        return None
    
    return min(valid_results, key=valid_results.get)


def run_sweep_experiment(
    model_size_configs,
    teacher_size_configs,
    bp_lrs,
    danp_etas,
    danp_alphas=[1e-4],
    danp_sigmas=[0.001],
    wp_alphas=[1e-3],
    wp_sigmas=[0.001],
    n_epochs=100,
    n_samples=5000,
    seed=42,
    device='cpu',
    output_dir=None
):
    """
    Run comprehensive sweep experiment.
    
    Args:
        model_size_configs: List of model size configs (e.g., [[10, 32, 1], [10, 64, 1]])
        teacher_size_configs: List of teacher size configs (e.g., [[10, 32, 1], [10, 64, 1]])
        bp_lrs: List of learning rates for BP
        danp_etas: List of learning rates (eta) for DANP
        danp_alphas: List of decorrelation rates for DANP
        danp_sigmas: List of noise stds for DANP
        wp_alphas: List of learning rates for WP
        wp_sigmas: List of noise stds for WP
        n_epochs: Number of training epochs
        n_samples: Number of samples in dataset
        seed: Random seed
        device: Device to use
        output_dir: Output directory (default: sweep_results_YYYYMMDD)
    
    Returns:
        dict with all results and metadata
    """
    if output_dir is None:
        output_dir = f"sweep_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    os.makedirs(output_dir, exist_ok=True)
    
    all_results = {}
    best_method_grid = {}  # For heatmap: (model_size_idx, teacher_size_idx) -> method
    
    # Calculate total experiments: each method is optimized independently
    total_experiments = (len(model_size_configs) * len(teacher_size_configs) * 
                        (len(bp_lrs) + 
                         len(danp_etas) * len(danp_alphas) * len(danp_sigmas) +
                         len(wp_alphas) * len(wp_sigmas)))
    
    print(f"Running {total_experiments} experiments...")
    print(f"Model sizes: {len(model_size_configs)}, Teacher sizes: {len(teacher_size_configs)}")
    print(f"BP LRs: {len(bp_lrs)}, DANP etas: {len(danp_etas)}, WP alphas: {len(wp_alphas)}")
    
    exp_count = 0
    
    for model_idx, model_sizes in enumerate(model_size_configs):
        for teacher_idx, teacher_sizes in enumerate(teacher_size_configs):
            # For each model/teacher combination, optimize each method independently
            best_bp_loss = float('inf')
            best_danp_loss = float('inf')
            best_wp_loss = float('inf')
            
            best_bp_config = None
            best_danp_config = None
            best_wp_config = None
            
            # Optimize BP independently
            print(f"\n{'='*80}")
            print(f"Optimizing BP for Model {model_sizes} / Teacher {teacher_sizes}")
            print(f"{'='*80}")
            for bp_lr in bp_lrs:
                exp_count += 1
                config_key = (tuple(model_sizes), tuple(teacher_sizes), 'bp', bp_lr)
                
                print(f"\n[{exp_count}/{total_experiments}] BP: lr={bp_lr:.4f}")
                
                try:
                    dataset = SyntheticDataset(
                        input_dim=teacher_sizes[0],
                        hidden_dim=teacher_sizes[1] if len(teacher_sizes) == 3 else None,
                        output_dim=teacher_sizes[-1],
                        n_samples=n_samples,
                        seed=seed,
                        teacher_layer_sizes=teacher_sizes
                    )
                    bp_results = train_bp_model(
                        layer_sizes=model_sizes,
                        dataset=dataset,
                        n_epochs=n_epochs,
                        lr=bp_lr,
                        batch_size=1,
                        seed=seed,
                        device=device
                    )
                    bp_loss = bp_results['final_test']
                    all_results[config_key] = {'bp': bp_loss}
                    
                    if bp_loss < best_bp_loss and not np.isnan(bp_loss):
                        best_bp_loss = bp_loss
                        best_bp_config = config_key
                    print(f"  BP loss: {bp_loss:.6f}")
                except Exception as e:
                    print(f"  BP failed: {e}")
                    all_results[config_key] = {'bp': float('inf')}
            
            # Optimize DANP independently
            print(f"\n{'='*80}")
            print(f"Optimizing DANP for Model {model_sizes} / Teacher {teacher_sizes}")
            print(f"{'='*80}")
            for danp_eta in danp_etas:
                for danp_alpha in danp_alphas:
                    for danp_sigma in danp_sigmas:
                        exp_count += 1
                        config_key = (tuple(model_sizes), tuple(teacher_sizes), 'danp', 
                                     danp_eta, danp_alpha, danp_sigma)
                        
                        print(f"\n[{exp_count}/{total_experiments}] DANP: "
                              f"eta={danp_eta:.4f}, alpha={danp_alpha:.4f}, sigma={danp_sigma:.4f}")
                        
                        try:
                            dataset = SyntheticDataset(
                                input_dim=teacher_sizes[0],
                                hidden_dim=teacher_sizes[1] if len(teacher_sizes) == 3 else None,
                                output_dim=teacher_sizes[-1],
                                n_samples=n_samples,
                                seed=seed,
                                teacher_layer_sizes=teacher_sizes
                            )
                            danp_results = train_danp_model(
                                layer_sizes=model_sizes,
                                dataset=dataset,
                                n_epochs=n_epochs,
                                eta=danp_eta,
                                alpha=danp_alpha,
                                sigma=danp_sigma,
                                batch_size=1,
                                seed=seed,
                                device=device
                            )
                            danp_loss = danp_results['final_test']
                            all_results[config_key] = {'danp': danp_loss}
                            
                            if danp_loss < best_danp_loss and not np.isnan(danp_loss):
                                best_danp_loss = danp_loss
                                best_danp_config = config_key
                            print(f"  DANP loss: {danp_loss:.6f}")
                        except Exception as e:
                            print(f"  DANP failed: {e}")
                            all_results[config_key] = {'danp': float('inf')}
            
            # Optimize WP independently
            print(f"\n{'='*80}")
            print(f"Optimizing WP for Model {model_sizes} / Teacher {teacher_sizes}")
            print(f"{'='*80}")
            for wp_alpha in wp_alphas:
                for wp_sigma in wp_sigmas:
                    exp_count += 1
                    config_key = (tuple(model_sizes), tuple(teacher_sizes), 'wp', 
                                 wp_alpha, wp_sigma)
                    
                    print(f"\n[{exp_count}/{total_experiments}] WP: "
                          f"alpha={wp_alpha:.4f}, sigma={wp_sigma:.4f}")
                    
                    try:
                        dataset = SyntheticDataset(
                            input_dim=teacher_sizes[0],
                            hidden_dim=teacher_sizes[1] if len(teacher_sizes) == 3 else None,
                            output_dim=teacher_sizes[-1],
                            n_samples=n_samples,
                            seed=seed,
                            teacher_layer_sizes=teacher_sizes
                        )   
                        wp_results = train_wp_model(
                            layer_sizes=model_sizes,
                            dataset=dataset,
                            n_epochs=n_epochs,
                            sigma=wp_sigma,
                            alpha=wp_alpha,
                            batch_size=1,
                            population_size=1,
                            seed=seed,
                            device=device
                        )
                        wp_loss = wp_results['final_test']
                        all_results[config_key] = {'wp': wp_loss}
                        
                        if wp_loss < best_wp_loss and not np.isnan(wp_loss):
                            best_wp_loss = wp_loss
                            best_wp_config = config_key
                        print(f"  WP loss: {wp_loss:.6f}")
                    except Exception as e:
                        print(f"  WP failed: {e}")
                        all_results[config_key] = {'wp': float('inf')}
            
            # Determine best method for this model/teacher combination
            best_losses = {
                'bp': best_bp_loss,
                'danp': best_danp_loss,
                'wp': best_wp_loss
            }
            best_method = determine_best_method(best_losses)
            best_method_grid[(model_idx, teacher_idx)] = best_method
            
            print(f"\nBest for Model {model_sizes} / Teacher {teacher_sizes}: {best_method}")
            print(f"  BP: {best_bp_loss:.6f}, DANP: {best_danp_loss:.6f}, WP: {best_wp_loss:.6f}")
    
    # Save results
    results_file = os.path.join(output_dir, 'results.json')
    
    # Convert to JSON-serializable format
    serializable_results = {}
    for key, value in all_results.items():
        key_str = str(key)
        result_dict = {}
        if 'bp' in value:
            result_dict['bp'] = float(value['bp']) if value['bp'] != float('inf') else None
        if 'danp' in value:
            result_dict['danp'] = float(value['danp']) if value['danp'] != float('inf') else None
        if 'wp' in value:
            result_dict['wp'] = float(value['wp']) if value['wp'] != float('inf') else None
        serializable_results[key_str] = result_dict
    
    with open(results_file, 'w') as f:
        json.dump({
            'all_results': serializable_results,
            'best_method_grid': {str(k): v for k, v in best_method_grid.items()},
            'model_size_configs': [list(ms) for ms in model_size_configs],
            'teacher_size_configs': [list(ts) for ts in teacher_size_configs],
            'hyperparameters': {
                'bp_lrs': bp_lrs,
                'danp_etas': danp_etas,
                'danp_alphas': danp_alphas,
                'danp_sigmas': danp_sigmas,
                'wp_alphas': wp_alphas,
                'wp_sigmas': wp_sigmas
            },
            'n_epochs': n_epochs,
            'n_samples': n_samples,
            'seed': seed
        }, f, indent=2)
    
    print(f"\nResults saved to {results_file}")
    
    return {
        'all_results': all_results,
        'best_method_grid': best_method_grid,
        'model_size_configs': model_size_configs,
        'teacher_size_configs': teacher_size_configs,
        'output_dir': output_dir
    }


def create_heatmap(results_dict, output_dir):
    """
    Create heatmap showing which method performs best in each scenario.
    Colors: BP (red), DANP/NP (green), WP (blue)
    """
    best_method_grid = results_dict['best_method_grid']
    model_size_configs = results_dict['model_size_configs']
    teacher_size_configs = results_dict['teacher_size_configs']
    
    # Create matrix for heatmap
    n_models = len(model_size_configs)
    n_teachers = len(teacher_size_configs)
    
    # Method to color mapping
    method_to_color = {
        'bp': 0,      # Red
        'danp': 1,    # Green
        'wp': 2,      # Blue
        None: 3       # Gray (no valid result)
    }
    
    heatmap_data = np.zeros((n_models, n_teachers))
    method_labels = np.empty((n_models, n_teachers), dtype=object)
    
    for (model_idx, teacher_idx), method in best_method_grid.items():
        heatmap_data[model_idx, teacher_idx] = method_to_color.get(method, 3)
        method_labels[model_idx, teacher_idx] = method.upper() if method else 'N/A'
    
    # Create figure
    fig, ax = plt.subplots(figsize=(max(8, n_teachers * 1.5), max(6, n_models * 1.2)))
    
    # Create custom colormap: Red (BP), Green (DANP), Blue (WP), Gray (None)
    from matplotlib.colors import ListedColormap
    colors = ['#FF4444', '#44FF44', '#4444FF', '#CCCCCC']  # Red, Green, Blue, Gray
    cmap = ListedColormap(colors)
    
    # Create heatmap
    sns.heatmap(heatmap_data, 
                annot=method_labels,
                fmt='',
                cmap=cmap,
                cbar=False,
                xticklabels=[f"Teacher: {ts}" for ts in teacher_size_configs],
                yticklabels=[f"Model: {ms}" for ms in model_size_configs],
                ax=ax,
                vmin=0, vmax=3)
    
    ax.set_title('Best Method by Model Size and Problem Dimension\n'
                 'BP (Red) | DANP/NP (Green) | WP (Blue)',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Problem Dimension (Teacher Network Size)', fontsize=12)
    ax.set_ylabel('Model Size (Student Network Size)', fontsize=12)
    
    plt.tight_layout()
    
    # Save figure
    heatmap_file = os.path.join(output_dir, 'best_method_heatmap.png')
    plt.savefig(heatmap_file, dpi=300, bbox_inches='tight')
    print(f"Heatmap saved to {heatmap_file}")
    
    plt.close()


def create_loss_comparison_heatmaps(results_dict, output_dir):
    """
    Create separate heatmaps for each method showing final test loss.
    """
    all_results = results_dict['all_results']
    model_size_configs = results_dict['model_size_configs']
    teacher_size_configs = results_dict['teacher_size_configs']
    
    n_models = len(model_size_configs)
    n_teachers = len(teacher_size_configs)
    
    # For each method, find best loss for each model/teacher combination
    for method in ['bp', 'danp', 'wp']:
        loss_matrix = np.full((n_models, n_teachers), np.nan)
        
        for model_idx, model_sizes in enumerate(model_size_configs):
            for teacher_idx, teacher_sizes in enumerate(teacher_size_configs):
                # Find best loss for this method across all hyperparameter combinations
                best_loss = float('inf')
                
                for config_key, results in all_results.items():
                    # Check if this result is for the right model/teacher and method
                    if (isinstance(config_key, tuple) and len(config_key) >= 3 and
                        config_key[0] == tuple(model_sizes) and 
                        config_key[1] == tuple(teacher_sizes) and
                        config_key[2] == method):
                        if method in results:
                            loss = results[method]
                            if loss != float('inf') and not np.isnan(loss) and loss < best_loss:
                                best_loss = loss
                
                if best_loss != float('inf'):
                    loss_matrix[model_idx, teacher_idx] = best_loss
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(max(8, n_teachers * 1.5), max(6, n_models * 1.2)))
        
        # Use log scale for better visualization
        log_loss_matrix = np.log10(loss_matrix + 1e-10)
        
        sns.heatmap(log_loss_matrix,
                    annot=True,
                    fmt='.2f',
                    cmap='viridis_r',  # Lower is better, so reverse colormap
                    xticklabels=[f"Teacher: {ts}" for ts in teacher_size_configs],
                    yticklabels=[f"Model: {ms}" for ms in model_size_configs],
                    ax=ax,
                    cbar_kws={'label': 'log10(Test Loss)'})
        
        method_name = method.upper() if method != 'danp' else 'DANP/NP'
        ax.set_title(f'{method_name} Final Test Loss (log10 scale)\n'
                     'Lower is better',
                     fontsize=14, fontweight='bold')
        ax.set_xlabel('Problem Dimension (Teacher Network Size)', fontsize=12)
        ax.set_ylabel('Model Size (Student Network Size)', fontsize=12)
        
        plt.tight_layout()
        
        loss_heatmap_file = os.path.join(output_dir, f'{method}_loss_heatmap.png')
        plt.savefig(loss_heatmap_file, dpi=300, bbox_inches='tight')
        print(f"{method_name} loss heatmap saved to {loss_heatmap_file}")
        
        plt.close()


def main():
    """Main function to run the sweep experiment."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Sweep experiment comparing BP, DANP, and WP')
    parser.add_argument('--n_epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--n_samples', type=int, default=5000, help='Number of samples in dataset')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use (cpu/cuda)')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory')
    
    args = parser.parse_args()
    
    # Define model size configurations (student networks)
    # Format: [input_dim, hidden1, ..., output_dim]
    model_size_configs = [
        [10, 32, 1],
        [10, 64, 1],
        [10, 128, 1],
        [10, 32, 32, 1],
        [10, 64, 64, 1],
    ]
    
    # Define teacher size configurations (problem dimensions)
    teacher_size_configs = [
        [10, 32, 1],
        [10, 64, 1],
        [10, 128, 1],
        [10, 32, 32, 1],
        [10, 64, 64, 1],
    ]
    
    # Define hyperparameter ranges
    # bp_lrs = [1e-4, 1e-3, 1e-2]
    # danp_etas = [1e-5, 1e-4, 1e-3]
    # danp_alphas = [1e-5,1e-4, 1e-3]  # Decorrelation rate
    # danp_sigmas = [0.001, 0.01]  # Noise std
    # wp_alphas = [1e-4, 1e-3, 1e-2]  # Learning rate
    # wp_sigmas = [0.001, 0.01]  # Noise std

    bp_lrs = [1e-4, 1e-3]
    danp_etas = [5e-3, 1e-3]
    danp_alphas = [1e-5,1e-4, 1e-3]  # Decorrelation rate
    danp_sigmas = [1e-3]  # Noise std
    wp_alphas = [1e-4, 1e-3]  # Learning rate
    wp_sigmas = [0.001]  # Noise std
    
    print("=" * 80)
    print("ALGORITHM SWEEP EXPERIMENT")
    print("=" * 80)
    print(f"Model sizes: {len(model_size_configs)} configurations")
    print(f"Teacher sizes: {len(teacher_size_configs)} configurations")
    print(f"BP learning rates: {bp_lrs}")
    print(f"DANP etas: {danp_etas}, alphas: {danp_alphas}, sigmas: {danp_sigmas}")
    print(f"WP alphas: {wp_alphas}, sigmas: {wp_sigmas}")
    print(f"Epochs: {args.n_epochs}, Samples: {args.n_samples}")
    print("=" * 80)
    
    # Run sweep
    results_dict = run_sweep_experiment(
        model_size_configs=model_size_configs,
        teacher_size_configs=teacher_size_configs,
        bp_lrs=bp_lrs,
        danp_etas=danp_etas,
        danp_alphas=danp_alphas,
        danp_sigmas=danp_sigmas,
        wp_alphas=wp_alphas,
        wp_sigmas=wp_sigmas,
        n_epochs=args.n_epochs,
        n_samples=args.n_samples,
        seed=args.seed,
        device=args.device,
        output_dir=args.output_dir
    )
    
    # Create visualizations
    print("\nCreating visualizations...")
    create_heatmap(results_dict, results_dict['output_dir'])
    create_loss_comparison_heatmaps(results_dict, results_dict['output_dir'])
    
    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE")
    print("=" * 80)
    print(f"Results saved to: {results_dict['output_dir']}")


if __name__ == '__main__':
    main()

