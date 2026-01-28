#!/usr/bin/env python3
"""
Plot results from transformer block experiment terminal output.
"""

import matplotlib.pyplot as plt
import numpy as np
import os

# Extract data from terminal output
bp_train_losses = [
    0.118469, 0.109597, 0.105933, 0.103946, 0.102511, 0.101277, 0.100127, 0.099026, 0.097959, 0.096921,
    0.095911, 0.094927, 0.093967, 0.093031, 0.092118, 0.091227, 0.090357, 0.089507, 0.088677, 0.087866,
    0.087073, 0.086298, 0.085540, 0.084798, 0.084072, 0.083362, 0.082666, 0.081985, 0.081318, 0.080665,
    0.080024, 0.079397, 0.078781, 0.078178, 0.077586, 0.077005, 0.076435, 0.075876, 0.075327, 0.074789,
    0.074260, 0.073740, 0.073230, 0.072728, 0.072236, 0.071751, 0.071276, 0.070808, 0.070348, 0.069895
]

bp_test_losses = [
    0.118391, 0.109612, 0.106003, 0.104049, 0.102637, 0.101421, 0.100285, 0.099196, 0.098139, 0.097111,
    0.096110, 0.095135, 0.094185, 0.093257, 0.092352, 0.091469, 0.090606, 0.089764, 0.088941, 0.088137,
    0.087351, 0.086583, 0.085831, 0.085096, 0.084377, 0.083673, 0.082983, 0.082309, 0.081648, 0.081000,
    0.080364, 0.079742, 0.079132, 0.078534, 0.077946, 0.077371, 0.076806, 0.076252, 0.075708, 0.075174,
    0.074650, 0.074134, 0.073628, 0.073131, 0.072642, 0.072162, 0.071691, 0.071227, 0.070770, 0.070322
]

danp_train_losses = [
    0.655887, 0.643756, 0.585410, 0.562242, 0.522165, 0.515433, 0.490426, 0.462494, 0.458583, 0.441467,
    0.429804, 0.419285, 0.409863, 0.397101, 0.388660, 0.376171, 0.372891, 0.363800, 0.357054, 0.350032,
    0.344921, 0.333309, 0.335185, 0.328146, 0.320984, 0.303969, 0.302502, 0.300750, 0.291513, 0.290528,
    0.287335, 0.285352, 0.280959, 0.274289, 0.272654, 0.273795, 0.265125, 0.253982, 0.268469, 0.250333,
    0.253403, 0.243353, 0.248424, 0.240529, 0.237094, 0.247489, 0.240941, 0.231493, 0.223776, 0.232093
]

danp_test_losses = [
    0.654265, 0.644262, 0.585015, 0.563652, 0.523065, 0.516030, 0.491772, 0.463378, 0.460604, 0.442690,
    0.431307, 0.420565, 0.412238, 0.399241, 0.390511, 0.377738, 0.374682, 0.365230, 0.359411, 0.351466,
    0.346849, 0.335713, 0.337414, 0.331225, 0.323688, 0.306275, 0.304761, 0.303130, 0.293705, 0.293102,
    0.290051, 0.287960, 0.283753, 0.276960, 0.275614, 0.276889, 0.268070, 0.257184, 0.271275, 0.253246,
    0.256514, 0.246310, 0.251561, 0.243980, 0.240420, 0.250246, 0.243823, 0.234379, 0.226697, 0.234652
]

danp_batch_train_losses = [
    0.659095, 0.653679, 0.588090, 0.564217, 0.524871, 0.516676, 0.490801, 0.466837, 0.457546, 0.443909,
    0.438553, 0.429601, 0.412113, 0.400569, 0.389538, 0.377548, 0.368638, 0.366506, 0.359892, 0.353230,
    0.343263, 0.329969, 0.335458, 0.332948, 0.318727, 0.306510, 0.299732, 0.301592, 0.294374, 0.290012,
    0.286322, 0.286661, 0.279803, 0.276301, 0.272033, 0.271980, 0.264544, 0.250680, 0.265096, 0.249774,
    0.252748, 0.241119, 0.249716, 0.238991, 0.235237, 0.241073, 0.238971, 0.229280, 0.221679, 0.231068
]

danp_batch_test_losses = [
    0.657293, 0.654182, 0.587662, 0.565522, 0.525539, 0.517184, 0.492177, 0.467472, 0.459246, 0.444885,
    0.439852, 0.430633, 0.414153, 0.402412, 0.391216, 0.378825, 0.370404, 0.367906, 0.362371, 0.354766,
    0.345175, 0.332254, 0.337345, 0.335912, 0.321236, 0.308568, 0.301975, 0.303751, 0.296510, 0.292425,
    0.288760, 0.288965, 0.282425, 0.278725, 0.274796, 0.274854, 0.267259, 0.253726, 0.267954, 0.252433,
    0.255828, 0.243967, 0.252848, 0.242171, 0.238293, 0.243502, 0.241750, 0.231982, 0.224382, 0.233642
]

# Final losses
bp_final_train = 0.069895
bp_final_test = 0.070322
danp_final_train = 0.232093
danp_final_test = 0.234652
danp_batch_final_train = 0.231068
danp_batch_final_test = 0.233642

# Create visualization
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Plot 1: Convergence curves
ax = axes[0]
epochs = range(1, len(bp_train_losses) + 1)

ax.plot(epochs, bp_train_losses, '--', label='BP Train', linewidth=2, alpha=0.7, color='blue')
ax.plot(epochs, bp_test_losses, '-', label='BP Test', linewidth=2, color='blue')
ax.plot(epochs, danp_train_losses, '--', label='DANP Train', linewidth=2, alpha=0.7, color='red')
ax.plot(epochs, danp_test_losses, '-', label='DANP Test', linewidth=2, color='red')
ax.plot(epochs, danp_batch_train_losses, '--', label='DANP Batch Train', linewidth=2, alpha=0.7, color='green')
ax.plot(epochs, danp_batch_test_losses, '-', label='DANP Batch Test', linewidth=2, color='green')

ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Loss', fontsize=12)
ax.set_title('Convergence Curves: Transformer Blocks', fontsize=14)
ax.legend(fontsize=11)
ax.set_yscale('log')
ax.grid(True, alpha=0.3)

# Plot 2: Final losses comparison
ax = axes[1]
methods = ['BP', 'DANP', 'DANP Batch']
train_losses = [bp_final_train, danp_final_train, danp_batch_final_train]
test_losses = [bp_final_test, danp_final_test, danp_batch_final_test]

x = np.arange(len(methods))
width = 0.35
ax.bar(x - width/2, train_losses, width, label='Train Loss', alpha=0.7)
ax.bar(x + width/2, test_losses, width, label='Test Loss', alpha=0.7)
ax.set_xlabel('Method', fontsize=12)
ax.set_ylabel('Final Loss', fontsize=12)
ax.set_title('Final Loss Comparison', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(methods)
ax.legend(fontsize=11)
ax.set_yscale('log')
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()

# Save plot
output_dir = './transformer_block_results'
os.makedirs(output_dir, exist_ok=True)
plot_file = os.path.join(output_dir, 'transformer_results.png')
plt.savefig(plot_file, dpi=300, bbox_inches='tight')
print(f"Visualization saved to {plot_file}")
plt.close()

print("\nFinal Results Summary:")
print("="*80)
print(f"BP - Final Train Loss: {bp_final_train:.6f}, Final Test Loss: {bp_final_test:.6f}")
print(f"DANP - Final Train Loss: {danp_final_train:.6f}, Final Test Loss: {danp_final_test:.6f}")
print(f"DANP Batch - Final Train Loss: {danp_batch_final_train:.6f}, Final Test Loss: {danp_batch_final_test:.6f}")
print("="*80)

