#!/usr/bin/env python3
"""
Grid Experiments: Multiple BP vs DANP comparison sweeps.

Experiment types:
  teacher_wd        - Teacher width × Teacher depth (fixed student 32×4)
  student_teacher_w - Student width × Teacher width (fixed depth 4)
  batch_student_w   - Batch size × Student width (fixed depth 4)
  batch_student_d   - Batch size × Student depth (fixed width 32)
  student_wd        - Student width × Student depth (original style)

Uses faster defaults: 6×6 grid, 200 max epochs, 15 patience, 2000 samples.

Usage:
  python3 grid_experiments.py --exp_type batch_student_d
  python3 grid_experiments.py --exp_type student_wd
  python3 grid_experiments.py --exp_type student_teacher_w
  python3 grid_experiments.py --exp_type teacher_wd --grid_size 5
  python3 grid_experiments.py --plot_only results/grid_batch_student_w_*/results.json
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import os
import json
from datetime import datetime
from torch.utils.data import DataLoader

_root = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, _root)
sys.path.insert(0, os.path.join(_root, 'toy_model'))
from toy_model.sythentic_data import SyntheticDataset, train_danp_batch
from toy_model.danp import DANPMLP
from toy_model.bp import BPMLP


# ============== Training (same as width_depth, with batch_size in signature) ==============

def train_bp_until_convergence(layer_sizes, dataset, lr=1e-3, batch_size=32, train_split=0.8,
    seed=42, device='cpu', max_epochs=1000, patience=25, min_delta=1e-6, min_delta_rel=1e-3, verbose=False):
    torch.manual_seed(seed + 1000)
    np.random.seed(seed + 1000)
    train_size = int(train_split * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size],
        generator=torch.Generator().manual_seed(seed)
    )
    loader_generator = torch.Generator().manual_seed(seed + 2000)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=loader_generator)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = BPMLP(layer_sizes).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    train_losses, test_losses = [], []
    best_test_loss, epochs_no_improve = float('inf'), 0

    for epoch in range(max_epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = F.mse_loss(model(x), y)
            loss.backward()
            optimizer.step()

        model.eval()
        train_loss = sum(F.mse_loss(model(x.to(device)), y.to(device)).item() * x.shape[0]
                        for x, y in train_loader) / len(train_dataset)
        test_loss = sum(F.mse_loss(model(x.to(device)), y.to(device)).item() * x.shape[0]
                       for x, y in test_loader) / len(test_dataset)
        train_losses.append(train_loss)
        test_losses.append(test_loss)

        # Use relative threshold: require improvement of min_delta OR min_delta_rel * best
        # Handle inf: first epoch always counts as improvement
        if best_test_loss == float('inf'):
            best_test_loss, epochs_no_improve = test_loss, 0
        else:
            delta = max(min_delta, best_test_loss * min_delta_rel)
            if test_loss < best_test_loss - delta:
                best_test_loss, epochs_no_improve = test_loss, 0
            else:
                epochs_no_improve += 1
        if epochs_no_improve >= patience:
            break

    return {'final_train': train_losses[-1], 'final_test': test_losses[-1], 'epochs_run': len(train_losses)}


def train_danp_until_convergence(layer_sizes, dataset, eta=1e-3, alpha=1e-4, sigma=0.001, relu_clip=10.0,
    batch_size=32, train_split=0.8, seed=42, device='cpu', max_epochs=1000, patience=25, min_delta=1e-6, min_delta_rel=1e-3, verbose=False):
    torch.manual_seed(seed + 1000)
    np.random.seed(seed + 1000)
    train_size = int(train_split * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size],
        generator=torch.Generator().manual_seed(seed)
    )
    loader_generator = torch.Generator().manual_seed(seed + 2000)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=loader_generator)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    eta_scaled = eta * batch_size
    model = DANPMLP(layer_sizes, sigma=sigma, eta=eta_scaled, alpha=alpha, relu_clip=relu_clip)
    for i in range(len(model.W)):
        model.W[i] = model.W[i].to(device)
    for i in range(len(model.R)):
        model.R[i] = model.R[i].to(device)

    train_losses, test_losses = [], []
    best_test_loss, epochs_no_improve = float('inf'), 0

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
        test_loss = 0.0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                for i in range(x.shape[0]):
                    x_clean, _ = model.forward_clean(x[i])
                    test_loss += F.mse_loss(x_clean[-1], y[i]).item()
        test_loss /= len(test_dataset)
        train_losses.append(train_loss)
        test_losses.append(test_loss)

        if best_test_loss == float('inf'):
            best_test_loss, epochs_no_improve = test_loss, 0
        else:
            delta = max(min_delta, best_test_loss * min_delta_rel)
            if test_loss < best_test_loss - delta:
                best_test_loss, epochs_no_improve = test_loss, 0
            else:
                epochs_no_improve += 1
        if epochs_no_improve >= patience:
            break

    return {'final_train': train_losses[-1], 'final_test': test_losses[-1], 'epochs_run': len(train_losses)}


# ============== Heatmap creation (generic axis labels) ==============

def create_heatmaps(results, x_labels, y_labels, output_dir, x_label='X', y_label='Y', scale_percentile=50):
    residual_train = np.array(results['residual_train'])
    residual_test = np.array(results['residual_test'])

    def robust_vmax(data):
        finite = data[np.isfinite(data)]
        well_behaved = np.abs(finite[np.abs(finite) < 1.0])
        if len(well_behaved) == 0:
            return 1e-6
        return max(float(np.percentile(well_behaved, scale_percentile)), 1e-7)

    vmax = max(robust_vmax(residual_train), robust_vmax(residual_test))
    vmin = -vmax

    def fmt_val(val):
        if abs(val) >= 1e6:
            return f'{val:.1e}'
        if abs(val) >= 1:
            return f'{val:.1f}'
        if abs(val) >= 1e-3:
            return f'{val:.2e}'
        if abs(val) >= 1e-6:
            return f'{val:.1e}'
        return f'{val:.0e}'

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, data, title in [
        (axes[0], residual_train, 'Residual (BP - DANP) Train'),
        (axes[1], residual_test, 'Residual (BP - DANP) Test'),
    ]:
        data_display = np.clip(data, vmin, vmax)
        im = ax.imshow(data_display, aspect='auto', cmap='RdBu_r', vmin=vmin, vmax=vmax)
        ax.set_xticks(np.arange(len(x_labels)))
        ax.set_yticks(np.arange(len(y_labels)))
        ax.set_xticklabels(x_labels, fontsize=9)
        ax.set_yticklabels(y_labels, fontsize=9)
        ax.set_xlabel(x_label, fontsize=11)
        ax.set_ylabel(y_label, fontsize=11)
        ax.set_title(title, fontsize=12)
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('BP - DANP (positive = DANP better)', fontsize=9)
        for i in range(len(y_labels)):
            for j in range(len(x_labels)):
                if not np.isnan(data[i, j]):
                    val = data[i, j]
                    tc = 'white' if abs(np.clip(val, vmin, vmax)) > vmax * 0.4 else 'black'
                    ax.text(j, i, fmt_val(val), ha='center', va='center', fontsize=8, color=tc)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'residual_heatmap.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Heatmap saved to {output_dir}/residual_heatmap.png")

    # Epochs-to-convergence heatmaps (BP and DANP)
    if 'bp_epochs' in results and 'danp_epochs' in results:
        bp_ep = np.array(results['bp_epochs'])
        danp_ep = np.array(results['danp_epochs'])
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        for ax, data, title in [
            (axes[0], bp_ep, 'BP Epochs to Convergence'),
            (axes[1], danp_ep, 'DANP Epochs to Convergence'),
        ]:
            im = ax.imshow(data, aspect='auto', cmap='viridis')
            ax.set_xticks(np.arange(len(x_labels)))
            ax.set_yticks(np.arange(len(y_labels)))
            ax.set_xticklabels(x_labels, fontsize=9)
            ax.set_yticklabels(y_labels, fontsize=9)
            ax.set_xlabel(x_label, fontsize=11)
            ax.set_ylabel(y_label, fontsize=11)
            ax.set_title(title, fontsize=12)
            plt.colorbar(im, ax=ax, label='Epochs')
            valid = data[data > 0]
            mid = np.median(valid) if len(valid) > 0 else 1
            for i in range(len(y_labels)):
                for j in range(len(x_labels)):
                    val = data[i, j]
                    if val > 0:
                        tc = 'black' if val < mid else 'white'
                        ax.text(j, i, int(val), ha='center', va='center', fontsize=8, color=tc, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'epochs_heatmap.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Epochs heatmap saved to {output_dir}/epochs_heatmap.png")


# ============== Experiment runners ==============

def run_grid_experiment(exp_type, grid_size=6, input_dim=10, output_dim=1, n_samples=2000,
    lr=1e-3, eta=1e-3, alpha=1e-4, sigma=0.001, relu_clip=10.0, max_epochs=1000, patience=25,
    min_delta=1e-6, min_delta_rel=1e-3, seed=42, output_dir=None, verbose=False, **kwargs):
    """
    Run a grid experiment. exp_type: teacher_wd | student_teacher_w | batch_student_w | batch_student_d | student_wd
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    np.random.seed(seed)

    if output_dir is None:
        output_dir = f"./results/grid_{exp_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(output_dir, exist_ok=True)

    n = grid_size

    if exp_type == 'teacher_wd':
        # X: teacher width, Y: teacher depth. Student fixed (e.g. 32 width, 4 depth)
        fixed_student_w = kwargs.get('fixed_student_w', 32)
        fixed_student_d = kwargs.get('fixed_student_d', 4)
        teacher_widths = [int(w) for w in np.logspace(np.log10(8), np.log10(128), n)]
        teacher_depths = list(range(2, 2 + n))
        x_labels, y_labels = teacher_widths, teacher_depths
        x_name, y_name = 'Teacher Width', 'Teacher Depth'
        student_layer_sizes = [input_dim] + [fixed_student_w] * (fixed_student_d - 1) + [output_dim]
        batch_size = kwargs.get('batch_size', 32)

    elif exp_type == 'student_teacher_w':
        # X: student width, Y: teacher width. Fixed depth (e.g. 4)
        fixed_depth = kwargs.get('fixed_depth', 4)
        student_widths = [int(w) for w in np.logspace(np.log10(8), np.log10(128), n)]
        teacher_widths = [int(w) for w in np.logspace(np.log10(8), np.log10(128), n)]
        x_labels, y_labels = student_widths, teacher_widths
        x_name, y_name = 'Student Width', 'Teacher Width'
        batch_size = kwargs.get('batch_size', 32)

    elif exp_type == 'batch_student_w':
        # X: batch size, Y: student width. Fixed teacher and student depth
        fixed_depth = kwargs.get('fixed_depth', 4)
        batch_sizes = [2**k for k in range(1, n + 1)]  # 2, 4, 8, 16, 32, 64
        student_widths = [int(w) for w in np.logspace(np.log10(8), np.log10(128), n)]
        x_labels, y_labels = batch_sizes, student_widths
        x_name, y_name = 'Batch Size', 'Student Width'
        teacher_layer_sizes = [input_dim] + [128] * (fixed_depth - 1) + [output_dim]
        dataset = SyntheticDataset(input_dim=input_dim, hidden_dim=128, output_dim=output_dim,
            n_samples=n_samples, seed=seed, teacher_layer_sizes=teacher_layer_sizes)

    elif exp_type == 'batch_student_d':
        # X: batch size, Y: student depth. Fixed teacher and student width
        fixed_width = kwargs.get('fixed_width', 32)
        batch_sizes = [2**k for k in range(1, n + 1)]
        student_depths = list(range(2, 2 + n))
        x_labels, y_labels = batch_sizes, student_depths
        x_name, y_name = 'Batch Size', 'Student Depth'
        teacher_layer_sizes = [input_dim] + [fixed_width] * (n + 1) + [output_dim]
        dataset = SyntheticDataset(input_dim=input_dim, hidden_dim=fixed_width, output_dim=output_dim,
            n_samples=n_samples, seed=seed, teacher_layer_sizes=teacher_layer_sizes)

    elif exp_type == 'student_wd':
        # Original: X: student width, Y: student depth
        widths = [int(w) for w in np.logspace(np.log10(8), np.log10(128), n)]
        depths = list(range(2, 2 + n))
        x_labels, y_labels = widths, depths
        x_name, y_name = 'Student Width', 'Student Depth'
        teacher_layer_sizes = [input_dim] + [max(widths)] * (max(depths) - 1) + [output_dim]
        dataset = SyntheticDataset(input_dim=input_dim, hidden_dim=max(widths), output_dim=output_dim,
            n_samples=n_samples, seed=seed, teacher_layer_sizes=teacher_layer_sizes)
        batch_size = kwargs.get('batch_size', 32)

    else:
        raise ValueError(f"Unknown exp_type: {exp_type}")

    # Build result matrices [y_idx, x_idx]
    ny, nx = len(y_labels), len(x_labels)
    residual_train = np.full((ny, nx), np.nan)
    residual_test = np.full((ny, nx), np.nan)
    bp_final_test = np.full((ny, nx), np.nan)
    danp_final_test = np.full((ny, nx), np.nan)
    bp_epochs = np.zeros((ny, nx), dtype=int)
    danp_epochs = np.zeros((ny, nx), dtype=int)

    total = ny * nx
    idx = 0

    for yi, y_val in enumerate(y_labels):
        for xi, x_val in enumerate(x_labels):
            idx += 1
            if exp_type == 'teacher_wd':
                teacher_layer_sizes = [input_dim] + [x_val] * (y_val - 1) + [output_dim]
                dataset = SyntheticDataset(input_dim=input_dim, hidden_dim=x_val, output_dim=output_dim,
                    n_samples=n_samples, seed=seed, teacher_layer_sizes=teacher_layer_sizes)
                layer_sizes = student_layer_sizes
                bs = batch_size
            elif exp_type == 'student_teacher_w':
                teacher_layer_sizes = [input_dim] + [y_val] * (fixed_depth - 1) + [output_dim]
                dataset = SyntheticDataset(input_dim=input_dim, hidden_dim=y_val, output_dim=output_dim,
                    n_samples=n_samples, seed=seed, teacher_layer_sizes=teacher_layer_sizes)
                layer_sizes = [input_dim] + [x_val] * (fixed_depth - 1) + [output_dim]
                bs = batch_size
            elif exp_type == 'batch_student_w':
                layer_sizes = [input_dim] + [y_val] * (fixed_depth - 1) + [output_dim]
                bs = x_val
            elif exp_type == 'batch_student_d':
                layer_sizes = [input_dim] + [fixed_width] * (y_val - 1) + [output_dim]
                bs = x_val
            else:  # student_wd
                layer_sizes = [input_dim] + [x_val] * (y_val - 1) + [output_dim]
                bs = batch_size

            if exp_type not in ('batch_student_w', 'batch_student_d', 'student_wd'):
                pass  # dataset already set above
            # else dataset was set once before the loop

            print(f"[{idx}/{total}] {x_name}={x_val}, {y_name}={y_val} | arch={layer_sizes[:3]}...{layer_sizes[-1] if len(layer_sizes)>3 else ''} bs={bs}")

            try:
                # Scale BP LR with batch size for batch experiments (linear scaling)
                bs_lr = lr * (bs / 32) if exp_type in ('batch_student_w', 'batch_student_d') else lr
                bp_res = train_bp_until_convergence(layer_sizes=layer_sizes, dataset=dataset, lr=bs_lr,
                    batch_size=bs, seed=seed, device=device, max_epochs=max_epochs, patience=patience,
                    min_delta=min_delta, min_delta_rel=min_delta_rel)
                danp_res = train_danp_until_convergence(layer_sizes=layer_sizes, dataset=dataset, eta=eta,
                    alpha=alpha, sigma=sigma, relu_clip=relu_clip, batch_size=bs, seed=seed, device=device,
                    max_epochs=max_epochs, patience=patience, min_delta=min_delta, min_delta_rel=min_delta_rel)
                residual_train[yi, xi] = bp_res['final_train'] - danp_res['final_train']
                residual_test[yi, xi] = bp_res['final_test'] - danp_res['final_test']
                bp_final_test[yi, xi] = bp_res['final_test']
                danp_final_test[yi, xi] = danp_res['final_test']
                bp_epochs[yi, xi] = bp_res['epochs_run']
                danp_epochs[yi, xi] = danp_res['epochs_run']
                print(f"  BP={bp_res['final_test']:.2e} ({bp_res['epochs_run']}ep) DANP={danp_res['final_test']:.2e} ({danp_res['epochs_run']}ep) res={residual_test[yi,xi]:+.2e}")
            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback
                traceback.print_exc()

    results = {
        'config': {'exp_type': exp_type, 'x_labels': x_labels, 'y_labels': y_labels,
                   'n_samples': n_samples, 'max_epochs': max_epochs, 'patience': patience,
                   'min_delta': min_delta, 'min_delta_rel': min_delta_rel, 'seed': seed},
        'residual_train': residual_train.tolist(),
        'residual_test': residual_test.tolist(),
        'bp_final_test': bp_final_test.tolist(),
        'danp_final_test': danp_final_test.tolist(),
        'bp_epochs': bp_epochs.tolist(),
        'danp_epochs': danp_epochs.tolist(),
    }
    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)
    create_heatmaps(results, x_labels, y_labels, output_dir, x_name, y_name)
    return results


def main():
    import argparse
    p = argparse.ArgumentParser(description='Grid experiments: BP vs DANP')
    p.add_argument('--exp_type', type=str, default='batch_student_w',
                   choices=['teacher_wd', 'student_teacher_w', 'batch_student_w', 'batch_student_d', 'student_wd'],
                   help='Experiment type')
    p.add_argument('--grid_size', type=int, default=6)
    p.add_argument('--n_samples', type=int, default=2000)
    p.add_argument('--max_epochs', type=int, default=1000)
    p.add_argument('--patience', type=int, default=25)
    p.add_argument('--min_delta', type=float, default=1e-6, help='Absolute floor for convergence threshold')
    p.add_argument('--min_delta_rel', type=float, default=1e-3, help='Relative improvement (e.g. 1e-3 = 0.1%%)')
    p.add_argument('--relu_clip', type=float, default=10.0)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--output_dir', type=str, default=None,
                   help='Exact output directory (overrides --output_base)')
    p.add_argument('--output_base', type=str, default=None,
                   help='Base folder for results; each run goes to {base}/grid_{exp_type}_{timestamp}')
    p.add_argument('--plot_only', type=str, default=None,
                   help='Path to results.json to regenerate heatmap only')
    args = p.parse_args()

    if args.plot_only:
        with open(args.plot_only) as f:
            results = json.load(f)
        output_dir = os.path.dirname(os.path.abspath(args.plot_only))
        x_labels = results['config']['x_labels']
        y_labels = results['config']['y_labels']
        exp_type = results['config']['exp_type']
        labels = {'teacher_wd': ('Teacher Width', 'Teacher Depth'),
                  'student_teacher_w': ('Student Width', 'Teacher Width'),
                  'batch_student_w': ('Batch Size', 'Student Width'),
                  'batch_student_d': ('Batch Size', 'Student Depth'),
                  'student_wd': ('Student Width', 'Student Depth')}
        x_name, y_name = labels.get(exp_type, ('X', 'Y'))
        print(f"Regenerating heatmap from {args.plot_only}")
        create_heatmaps(results, x_labels, y_labels, output_dir, x_name, y_name)
        return

    output_dir = args.output_dir
    if output_dir is None and args.output_base:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = os.path.join(args.output_base, f'grid_{args.exp_type}_{ts}')
    print(f"Grid experiment: {args.exp_type} ({args.grid_size}x{args.grid_size})")
    run_grid_experiment(exp_type=args.exp_type, grid_size=args.grid_size, n_samples=args.n_samples,
        max_epochs=args.max_epochs, patience=args.patience, min_delta=args.min_delta,
        min_delta_rel=args.min_delta_rel, relu_clip=args.relu_clip,
        seed=args.seed, output_dir=output_dir)


if __name__ == '__main__':
    main()
