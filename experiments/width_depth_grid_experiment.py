#!/usr/bin/env python3
"""
Width-Depth Grid Experiment: Compare BP vs DANP across network architectures.

Creates a 10x10 grid where:
  - X-axis: network width (hidden layer size)
  - Y-axis: network depth (number of layers)

Each cell trains BP and DANP until convergence (or max epochs), then computes
the residual difference: (BP final loss) - (DANP final loss).

Positive values = BP worse (higher loss), Negative values = DANP worse.

Usage:
  python3 width_depth_grid_experiment.py
  python3 width_depth_grid_experiment.py --max_epochs 500 --patience 20
  python3 width_depth_grid_experiment.py --output_dir ./width_depth_results
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless runs
import matplotlib.pyplot as plt
import sys
import os
import json
from datetime import datetime
from torch.utils.data import DataLoader

# Add project root and toy_model to path (sythentic_data uses relative imports)
_root = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, _root)
sys.path.insert(0, os.path.join(_root, 'toy_model'))
from toy_model.sythentic_data import SyntheticDataset, train_danp_batch
from toy_model.danp import DANPMLP
from toy_model.bp import BPMLP


def train_bp_until_convergence(
    layer_sizes,
    dataset,
    lr=1e-3,
    batch_size=32,
    train_split=0.8,
    seed=42,
    device='cpu',
    max_epochs=500,
    patience=25,
    min_delta=1e-7,
    verbose=False,
):
    """
    Train BP model until convergence (or max_epochs).
    Convergence: no improvement in test loss for `patience` consecutive epochs.
    """
    torch.manual_seed(seed + 1000)
    np.random.seed(seed + 1000)

    train_size = int(train_split * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size],
        generator=torch.Generator().manual_seed(seed)
    )

    # Use generator for reproducible shuffle - same seed ensures BP and DANP see identical batch order
    loader_generator = torch.Generator().manual_seed(seed + 2000)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=loader_generator)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = BPMLP(layer_sizes).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    train_losses = []
    test_losses = []
    best_test_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(max_epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = F.mse_loss(pred, y)
            loss.backward()
            optimizer.step()

        model.eval()
        train_loss = 0.0
        with torch.no_grad():
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                train_loss += F.mse_loss(pred, y).item() * x.shape[0]
        train_loss /= len(train_dataset)
        train_losses.append(train_loss)

        test_loss = 0.0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                test_loss += F.mse_loss(pred, y).item() * x.shape[0]
        test_loss /= len(test_dataset)
        test_losses.append(test_loss)

        if test_loss < best_test_loss - min_delta:
            best_test_loss = test_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if verbose:
            print(f"  BP Epoch {epoch+1}/{max_epochs} | Train: {train_loss:.6f} | Test: {test_loss:.6f} | No improve: {epochs_no_improve}/{patience}")

        if epochs_no_improve >= patience:
            if verbose:
                print(f"  BP converged at epoch {epoch+1}")
            break

    return {
        'final_train': train_losses[-1],
        'final_test': test_losses[-1],
        'train_losses': train_losses,
        'test_losses': test_losses,
        'epochs_run': len(train_losses),
        'converged': epochs_no_improve >= patience,
    }


def train_danp_until_convergence(
    layer_sizes,
    dataset,
    eta=1e-3,
    alpha=1e-4,
    sigma=0.001,
    relu_clip=10.0,
    batch_size=32,
    train_split=0.8,
    seed=42,
    device='cpu',
    max_epochs=500,
    patience=25,
    min_delta=1e-7,
    verbose=False,
):
    """
    Train DANP model until convergence (or max_epochs).
    """
    torch.manual_seed(seed + 1000)
    np.random.seed(seed + 1000)

    train_size = int(train_split * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size],
        generator=torch.Generator().manual_seed(seed)
    )

    # Same generator seed as BP so both see identical batch order
    loader_generator = torch.Generator().manual_seed(seed + 2000)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=loader_generator)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    eta_scaled = eta * batch_size
    model = DANPMLP(layer_sizes, sigma=sigma, eta=eta_scaled, alpha=alpha, relu_clip=relu_clip)
    for i in range(len(model.W)):
        model.W[i] = model.W[i].to(device)
    for i in range(len(model.R)):
        model.R[i] = model.R[i].to(device)

    train_losses = []
    test_losses = []
    best_test_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(max_epochs):
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            train_danp_batch(model, x, y)

        train_loss = 0.0
        with torch.no_grad():
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                for i in range(x.shape[0]):
                    x_clean, _ = model.forward_clean(x[i])
                    train_loss += F.mse_loss(x_clean[-1], y[i]).item()
        train_loss /= len(train_dataset)
        train_losses.append(train_loss)

        test_loss = 0.0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                for i in range(x.shape[0]):
                    x_clean, _ = model.forward_clean(x[i])
                    test_loss += F.mse_loss(x_clean[-1], y[i]).item()
        test_loss /= len(test_dataset)
        test_losses.append(test_loss)

        if test_loss < best_test_loss - min_delta:
            best_test_loss = test_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if verbose:
            print(f"  DANP Epoch {epoch+1}/{max_epochs} | Train: {train_loss:.6f} | Test: {test_loss:.6f} | No improve: {epochs_no_improve}/{patience}")

        if epochs_no_improve >= patience:
            if verbose:
                print(f"  DANP converged at epoch {epoch+1}")
            break

    return {
        'final_train': train_losses[-1],
        'final_test': test_losses[-1],
        'train_losses': train_losses,
        'test_losses': test_losses,
        'epochs_run': len(train_losses),
        'converged': epochs_no_improve >= patience,
    }


def run_width_depth_experiment(
    widths=None,
    depths=None,
    input_dim=10,
    output_dim=1,
    n_samples=3000,
    lr=1e-3,
    eta=1e-3,
    alpha=1e-4,
    sigma=0.001,
    relu_clip=10.0,
    batch_size=32,
    max_epochs=500,
    patience=25,
    min_delta=1e-7,
    seed=42,
    output_dir=None,
    verbose_grid=False,
):
    """
    Run the 10x10 width-depth grid experiment.
    
    Returns residual difference (BP - DANP) for each cell.
    Positive = BP better, Negative = DANP better.
    """
    if widths is None:
        # 10 widths: log-spaced from 8 to 256
        widths = [int(w) for w in np.logspace(np.log10(8), np.log10(256), 10)]
    if depths is None:
        # 10 depths: 2 to 11 layers (1 input, d-1 hidden, 1 output)
        depths = list(range(2, 12))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Enforce reproducibility on GPU
    if device == 'cuda':
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    torch.manual_seed(seed)
    np.random.seed(seed)

    if output_dir is None:
        output_dir = f"./results/width_depth_grid_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(output_dir, exist_ok=True)

    # Build teacher once (use max width/depth for teacher capacity)
    max_width = max(widths)
    max_depth = max(depths)
    teacher_layer_sizes = [input_dim]
    for _ in range(max_depth - 1):
        teacher_layer_sizes.append(max_width)
    teacher_layer_sizes.append(output_dim)

    dataset = SyntheticDataset(
        input_dim=input_dim,
        hidden_dim=max_width,
        output_dim=output_dim,
        n_samples=n_samples,
        seed=seed,
        teacher_layer_sizes=teacher_layer_sizes,
    )
    print(f"Dataset: {len(dataset)} samples, teacher: {teacher_layer_sizes[:3]}...{teacher_layer_sizes[-1]}")

    n_widths = len(widths)
    n_depths = len(depths)

    # Matrices for heatmaps: [depth_idx, width_idx]
    residual_train = np.full((n_depths, n_widths), np.nan)
    residual_test = np.full((n_depths, n_widths), np.nan)
    bp_final_test = np.full((n_depths, n_widths), np.nan)
    danp_final_test = np.full((n_depths, n_widths), np.nan)
    bp_epochs = np.zeros((n_depths, n_widths), dtype=int)
    danp_epochs = np.zeros((n_depths, n_widths), dtype=int)

    total_cells = n_depths * n_widths
    cell_idx = 0

    for di, depth in enumerate(depths):
        for wi, width in enumerate(widths):
            cell_idx += 1
            layer_sizes = [input_dim] + [width] * (depth - 1) + [output_dim]

            print(f"\n[{cell_idx}/{total_cells}] Width={width}, Depth={depth} | Architecture: {layer_sizes}")

            try:
                bp_results = train_bp_until_convergence(
                    layer_sizes=layer_sizes,
                    dataset=dataset,
                    lr=lr,
                    batch_size=batch_size,
                    train_split=0.8,
                    seed=seed,
                    device=device,
                    max_epochs=max_epochs,
                    patience=patience,
                    min_delta=min_delta,
                    verbose=verbose_grid,
                )

                danp_results = train_danp_until_convergence(
                    layer_sizes=layer_sizes,
                    dataset=dataset,
                    eta=eta,
                    alpha=alpha,
                    sigma=sigma,
                    relu_clip=relu_clip,
                    batch_size=batch_size,
                    train_split=0.8,
                    seed=seed,
                    device=device,
                    max_epochs=max_epochs,
                    patience=patience,
                    min_delta=min_delta,
                    verbose=verbose_grid,
                )

                # Residual difference: BP - DANP (positive => BP has higher loss, i.e. BP worse)
                # User said "residual difference in performance" - typically we want:
                # BP_final_loss - DANP_final_loss
                # Positive = BP worse (higher loss), Negative = DANP worse
                residual_train[di, wi] = bp_results['final_train'] - danp_results['final_train']
                residual_test[di, wi] = bp_results['final_test'] - danp_results['final_test']
                bp_final_test[di, wi] = bp_results['final_test']
                danp_final_test[di, wi] = danp_results['final_test']
                bp_epochs[di, wi] = bp_results['epochs_run']
                danp_epochs[di, wi] = danp_results['epochs_run']

                print(f"  BP: test={bp_results['final_test']:.6f} ({bp_results['epochs_run']} ep) | "
                      f"DANP: test={danp_results['final_test']:.6f} ({danp_results['epochs_run']} ep) | "
                      f"Residual (BP-DANP): {residual_test[di, wi]:+.6f}")

            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback
                traceback.print_exc()

    results = {
        'config': {
            'widths': widths,
            'depths': depths,
            'input_dim': input_dim,
            'output_dim': output_dim,
            'n_samples': n_samples,
            'lr': lr,
            'eta': eta,
            'max_epochs': max_epochs,
            'patience': patience,
            'batch_size': batch_size,
            'relu_clip': relu_clip,
            'seed': seed,
        },
        'residual_train': residual_train.tolist(),
        'residual_test': residual_test.tolist(),
        'bp_final_test': bp_final_test.tolist(),
        'danp_final_test': danp_final_test.tolist(),
        'bp_epochs': bp_epochs.tolist(),
        'danp_epochs': danp_epochs.tolist(),
    }

    # Save results
    results_path = os.path.join(output_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")

    # Create heatmaps
    create_heatmaps(results, widths, depths, output_dir)

    return results


def create_heatmaps(results, widths, depths, output_dir):
    """Create heatmap visualizations of residual difference."""
    residual_train = np.array(results['residual_train'])
    residual_test = np.array(results['residual_test'])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Shared colormap: blue = DANP better (negative residual), red = BP better (positive)
    vmax = max(np.nanmax(np.abs(residual_train)), np.nanmax(np.abs(residual_test)), 1e-6)
    vmin = -vmax

    for ax, data, title in [
        (axes[0], residual_train, 'Residual (BP - DANP) Final Train Loss'),
        (axes[1], residual_test, 'Residual (BP - DANP) Final Test Loss'),
    ]:
        im = ax.imshow(data, aspect='auto', cmap='RdBu_r', vmin=vmin, vmax=vmax)
        ax.set_xticks(np.arange(len(widths)))
        ax.set_yticks(np.arange(len(depths)))
        ax.set_xticklabels(widths)
        ax.set_yticklabels(depths)
        ax.set_xlabel('Network Width', fontsize=12)
        ax.set_ylabel('Network Depth', fontsize=12)
        ax.set_title(title, fontsize=14)
        plt.colorbar(im, ax=ax, label='BP loss - DANP loss\n(positive = DANP better)')

        # Annotate cells with values
        for i in range(len(depths)):
            for j in range(len(widths)):
                if not np.isnan(data[i, j]):
                    val = data[i, j]
                    text = ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                                   color='black' if abs(val) < vmax * 0.5 else 'white', fontsize=7)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'residual_heatmap.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Heatmap saved to {plot_path}")
    plt.close()

    # Additional: absolute loss comparison heatmaps
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    bp_test = np.array(results['bp_final_test'])
    danp_test = np.array(results['danp_final_test'])

    for ax, data, title in [
        (axes[0], np.log10(bp_test + 1e-10), 'log10(BP Final Test Loss)'),
        (axes[1], np.log10(danp_test + 1e-10), 'log10(DANP Final Test Loss)'),
    ]:
        im = ax.imshow(data, aspect='auto', cmap='viridis')
        ax.set_xticks(np.arange(len(widths)))
        ax.set_yticks(np.arange(len(depths)))
        ax.set_xticklabels(widths)
        ax.set_yticklabels(depths)
        ax.set_xlabel('Network Width', fontsize=12)
        ax.set_ylabel('Network Depth', fontsize=12)
        ax.set_title(title, fontsize=14)
        plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plot_path2 = os.path.join(output_dir, 'loss_heatmaps.png')
    plt.savefig(plot_path2, dpi=150, bbox_inches='tight')
    print(f"Loss heatmaps saved to {plot_path2}")
    plt.close()


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Width-Depth Grid: BP vs DANP residual difference')
    parser.add_argument('--input_dim', type=int, default=10)
    parser.add_argument('--output_dim', type=int, default=1)
    parser.add_argument('--n_samples', type=int, default=3000)
    parser.add_argument('--lr', type=float, default=1e-3, help='BP learning rate')
    parser.add_argument('--eta', type=float, default=1e-3, help='DANP learning rate')
    parser.add_argument('--alpha', type=float, default=1e-4)
    parser.add_argument('--sigma', type=float, default=0.001)
    parser.add_argument('--relu_clip', type=float, default=10.0,
                       help='Max ReLU output value for DANP (prevents explosion; use large value for no clipping)')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_epochs', type=int, default=500, help='Epoch limit if no convergence')
    parser.add_argument('--patience', type=int, default=25, help='Epochs without improvement to declare convergence')
    parser.add_argument('--min_delta', type=float, default=1e-7, help='Min improvement to count as progress')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--verbose', action='store_true', help='Print per-epoch progress for each cell')
    parser.add_argument('--grid_size', type=int, default=10, help='Grid size (NxN)')

    args = parser.parse_args()

    n = args.grid_size
    widths = [int(w) for w in np.logspace(np.log10(8), np.log10(256), n)]
    depths = list(range(2, 2 + n))

    print("=" * 80)
    print("WIDTH-DEPTH GRID EXPERIMENT: BP vs DANP")
    print("=" * 80)
    print(f"Grid: {n}x{n} (width x depth)")
    print(f"Widths: {widths}")
    print(f"Depths: {depths}")
    print(f"Max epochs: {args.max_epochs}, Patience: {args.patience}")
    print("=" * 80)

    run_width_depth_experiment(
        widths=widths,
        depths=depths,
        input_dim=args.input_dim,
        output_dim=args.output_dim,
        n_samples=args.n_samples,
        lr=args.lr,
        eta=args.eta,
        alpha=args.alpha,
        sigma=args.sigma,
        relu_clip=args.relu_clip,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        min_delta=args.min_delta,
        seed=args.seed,
        output_dir=args.output_dir,
        verbose_grid=args.verbose,
    )


if __name__ == '__main__':
    main()
