#!/usr/bin/env python3
"""
Quick script to plot transformer block experiment results from terminal output.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# Extract data from terminal output
bp_train_losses = [
    0.139474, 0.136049, 0.132967, 0.130193, 0.127697, 0.125452, 0.123430, 0.121610, 0.119971, 0.118493,
    0.117161, 0.115958, 0.114870, 0.113887, 0.112996, 0.112187, 0.111452, 0.110784, 0.110174, 0.109616,
    0.109105, 0.108636, 0.108205, 0.107806, 0.107438, 0.107096, 0.106778, 0.106481, 0.106203, 0.105942,
    0.105696, 0.105463, 0.105243, 0.105033, 0.104833, 0.104642, 0.104459, 0.104282, 0.104112, 0.103948,
    0.103788, 0.103633, 0.103483, 0.103335, 0.103192, 0.103051, 0.102913, 0.102777, 0.102644, 0.102512
]

bp_test_losses = [
    0.139258, 0.135852, 0.132788, 0.130031, 0.127551, 0.125320, 0.123313, 0.121507, 0.119881, 0.118415,
    0.117094, 0.115902, 0.114826, 0.113852, 0.112970, 0.112170, 0.111444, 0.110783, 0.110181, 0.109630,
    0.109126, 0.108664, 0.108238, 0.107846, 0.107483, 0.107146, 0.106833, 0.106541, 0.106267, 0.106011,
    0.105769, 0.105540, 0.105324, 0.105118, 0.104921, 0.104734, 0.104553, 0.104380, 0.104213, 0.104051,
    0.103894, 0.103742, 0.103594, 0.103449, 0.103308, 0.103169, 0.103033, 0.102900, 0.102768, 0.102639
]

danp_train_losses = [
    0.674983, 0.610385, 0.581186, 0.576144, 0.540083, 0.523509, 0.494910, 0.475084, 0.452982, 0.455844,
    0.454291, 0.447924, 0.438260, 0.439183, 0.433964, 0.433563, 0.405310, 0.396153, 0.402018, 0.383913,
    0.375198, 0.364585, 0.361565, 0.346677, 0.350524, 0.347685, 0.341918, 0.335757, 0.321839, 0.319319,
    0.311088, 0.313868, 0.300646, 0.290631, 0.289049, 0.287154, 0.280368, 0.284691, 0.275876, 0.266255,
    0.254796, 0.249788, 0.258743, 0.245827, 0.246647, 0.240365, 0.243446, 0.224124, 0.232354, 0.227751
]

danp_test_losses = [
    0.678297, 0.614333, 0.585244, 0.580548, 0.543468, 0.527207, 0.498450, 0.478349, 0.456699, 0.459360,
    0.459069, 0.451620, 0.442212, 0.443138, 0.438297, 0.436719, 0.409924, 0.400647, 0.406007, 0.388346,
    0.378535, 0.367874, 0.364827, 0.350156, 0.354270, 0.350750, 0.345664, 0.339370, 0.325143, 0.322099,
    0.314445, 0.317512, 0.304716, 0.294428, 0.292184, 0.290713, 0.284381, 0.288217, 0.278999, 0.269380,
    0.257624, 0.253241, 0.262106, 0.249439, 0.250318, 0.243923, 0.246919, 0.227813, 0.235684, 0.231233
]

danp_batch_train_losses = [
    0.698646, 0.662661, 0.639454, 0.621479, 0.608292, 0.597738, 0.590227, 0.583780, 0.577922, 0.572546,
    0.567466, 0.562497, 0.559062, 0.555555, 0.551958, 0.549386, 0.545626, 0.542763, 0.539953, 0.537027,
    0.534594, 0.531802, 0.529177, 0.526990, 0.524828, 0.522248, 0.520047, 0.518124, 0.515837, 0.513686,
    0.511428, 0.509707, 0.506948, 0.504279, 0.502430, 0.500460, 0.498917, 0.496992, 0.495071, 0.492806,
    0.491346, 0.489428, 0.488083, 0.485979, 0.484113, 0.482453, 0.481364, 0.479576, 0.477596, 0.475669
]

danp_batch_test_losses = [
    0.700390, 0.664479, 0.641175, 0.623268, 0.609998, 0.599441, 0.592017, 0.585764, 0.579972, 0.574470,
    0.569491, 0.564508, 0.561125, 0.557701, 0.553895, 0.551282, 0.547701, 0.545003, 0.542097, 0.539100,
    0.536639, 0.533874, 0.531295, 0.529216, 0.527114, 0.524536, 0.522283, 0.520349, 0.518015, 0.515853,
    0.513557, 0.511848, 0.509132, 0.506555, 0.504726, 0.502853, 0.501308, 0.499439, 0.497578, 0.495232,
    0.493718, 0.491824, 0.490472, 0.488355, 0.486445, 0.484762, 0.483809, 0.482139, 0.480175, 0.478316
]

# Create results dictionary in the expected format
results = {
    'config': {
        'd_model': 128,
        'n_heads': 4,
        'd_ff': 512,
        'seq_len': 10,
        'n_samples': 5000,
        'n_epochs': 50,
        'lr': 0.001,
        'sigma': 0.001,
        'alpha': 0.0001,
        'batch_size': 32,
        'seed': 42
    },
    'experiments': {
        'bp': {
            'train_losses': bp_train_losses,
            'test_losses': bp_test_losses,
            'final_train': bp_train_losses[-1],
            'final_test': bp_test_losses[-1]
        },
        'danp': {
            'train_losses': danp_train_losses,
            'test_losses': danp_test_losses,
            'final_train': danp_train_losses[-1],
            'final_test': danp_test_losses[-1]
        },
        'danp_batch': {
            'train_losses': danp_batch_train_losses,
            'test_losses': danp_batch_test_losses,
            'final_train': danp_batch_train_losses[-1],
            'final_test': danp_batch_test_losses[-1]
        }
    }
}

# Create visualization
output_dir = './transformer_block_results'
os.makedirs(output_dir, exist_ok=True)

fig = plt.figure(figsize=(16, 7))
gs = fig.add_gridspec(2, 2, height_ratios=[0.15, 1], width_ratios=[1, 1], hspace=0.3, wspace=0.3)

# Add parameter text box at the top
param_ax = fig.add_subplot(gs[0, :])
param_ax.axis('off')
config = results['config']
param_text = (
    f"d_model={config['d_model']}, n_heads={config['n_heads']}, d_ff={config['d_ff']}, "
    f"seq_len={config['seq_len']}, n_samples={config['n_samples']}, n_epochs={config['n_epochs']}, "
    f"batch_size={config['batch_size']}, seed={config['seed']}\n"
    f"BP: lr={config['lr']:.4f} | "
    f"DANP: eta={config['lr']:.4f}, alpha={config['alpha']:.4f}, sigma={config['sigma']:.4f} | "
    f"DANP Batch: eta={config['lr']:.4f}, alpha={config['alpha']:.4f}, sigma={config['sigma']:.4f}"
)
param_ax.text(0.5, 0.5, param_text, ha='center', va='center', fontsize=10, 
              bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5), family='monospace')

# Plot 1: Convergence curves
ax = fig.add_subplot(gs[1, 0])
for method, exp in results['experiments'].items():
    epochs = range(len(exp['train_losses']))
    ax.plot(epochs, exp['train_losses'], '--', label=f'{method.upper()} Train', linewidth=2, alpha=0.7)
    ax.plot(epochs, exp['test_losses'], '-', label=f'{method.upper()} Test', linewidth=2)
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Loss', fontsize=12)
ax.set_title('Convergence Curves: Transformer Blocks', fontsize=14)
ax.legend(fontsize=10)
ax.set_yscale('log')
ax.grid(True, alpha=0.3)

# Plot 2: Final losses comparison
ax = fig.add_subplot(gs[1, 1])
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
ax.legend(fontsize=10)
ax.set_yscale('log')
ax.grid(True, alpha=0.3, axis='y')
plot_file = os.path.join(output_dir, 'transformer_results.png')
plt.savefig(plot_file, dpi=300, bbox_inches='tight')
print(f"Visualization saved to {plot_file}")
plt.close()

print("\nSummary:")
print("="*80)
for method, exp in results['experiments'].items():
    print(f"{method.upper()}:")
    print(f"  Final Train Loss: {exp['final_train']:.6f}")
    print(f"  Final Test Loss:  {exp['final_test']:.6f}")
print("="*80)

