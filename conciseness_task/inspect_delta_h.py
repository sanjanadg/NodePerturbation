#!/usr/bin/env python3
"""
inspect_delta_h.py

Computes delta_h_t = ||h̃_t - h_t|| at each layer t for both WP and DANP,
then plots delta_h_t vs layer depth for direct comparison.

WP  (GLOBAL): Perturb ALL weight matrices W_0...W_L simultaneously using the
              same seed logic as wp_conciseness.py, then run one forward pass.
              delta_h_t at layer t reflects:
                - the noisy weights at layer t, AND
                - the already-noisy input arriving from layer t-1
              This matches how ES actually works in training.

DANP:         Add eps directly to activations at each layer (cumulative:
              noise at layer t propagates into layer t+1's input).
              delta_h_t at layer t reflects accumulated upstream noise.

For each candidate index, WP and DANP use the same integer seed (drawn once in
``main()`` after ``np.random.seed(SEED)``) so RNG initialization is aligned.

Both methods produce a "snowball effect" but through different mechanisms:
  WP:   snowball via noisy weights transforming already-noisy activations
  DANP: snowball via noisy activations propagating through clean weights

Output:
    delta_h_plot.png      — delta_h_t vs layer depth, WP vs DANP (same axes; scales may differ)
    delta_h_plot_wp.png   — WP only (y-axis scaled for WP)
    delta_h_plot_danp.png — DANP only (y-axis scaled for DANP)
    delta_h.csv           — raw numbers

USAGE:
    python3 -m conciseness_task.inspect_delta_h.py
"""

import torch
import numpy as np
import csv, os, re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging as hf_logging

hf_logging.set_verbosity_error()

# UPDATE to match your project structure
from danp_llm_v3 import DANPHook, get_target_linears

# ---------------------------------------------------------------------------
# Config — match your training scripts exactly
# ---------------------------------------------------------------------------
SIGMA           = 0.001
POPULATION_SIZE = 30
SEED            = 33
MODEL_NAME      = "Qwen/Qwen2.5-0.5B-Instruct"
HF_CACHE_DIR    = "huggingface_cache"
OUTPUT_DIR      = "."

DATASET = [
    ("Solve: 3 + 5 =", "8"),
    ("If all birds can fly and penguins are birds, can penguins fly?", "No"),
]

PROMPT_LABELS = [
    "Prompt 0: '3+5'",
    "Prompt 1: 'Penguins'",
]


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def _block_index(name: str):
    m = re.search(r'\.layers\.(\d+)\.', name) or re.search(r'\.h\.(\d+)\.', name)
    return int(m.group(1)) if m else -1


# ===========================================================================
# WP: GLOBAL weight perturbation — delta_h_t per layer
# ===========================================================================
def inspect_wp_delta_h(model, tok, device, seeds):
    """
    Global WP matching wp_conciseness.py exactly (per candidate seed).

    ``seeds`` must match ``inspect_danp_delta_h`` so each candidate index
    uses the same RNG root for fair WP vs DANP comparison.

    Args:
        seeds: length-POPULATION_SIZE list of ints (e.g. from main()).
        1. Run one clean forward pass. Capture h_t at every target layer.
        2. Perturb ALL weight parameters simultaneously:
               W_l += sigma * eps_l   for all l in 0..L
           using the same per-parameter seeding as the ES training script.
        3. Run one full forward pass. Capture h̃_t at every target layer.
        4. Restore ALL weights (subtract the same sigma * eps).
        5. delta_h_t = ||h̃_t - h_t||

    (Order is clean → noisy → restore so weights are nominal before the next
    candidate; ||h̃ - h|| is the same as if you ran noisy first then clean.)

    Because all weights are perturbed together, delta_h_t at layer t
    compounds naturally: the noisy weights at layer t act on an input
    that is already noisy from layers 0..t-1.
    """
    print("\n" + "="*70)
    print("WP (GLOBAL): computing induced delta_h_t per layer")
    print("="*70)

    # Get all named parameters (same as wp_conciseness.py: model.named_parameters())
    all_params = [(n, p) for n, p in model.named_parameters()]

    # Get target linear layers for capturing activations
    hook_obj     = DANPHook(model, include="all", last_k=0, sigma=SIGMA, base_seed=SEED)
    target_names = [name for name, _ in hook_obj._targets]
    n_layers     = len(target_names)
    print(f"  Weight params to perturb: {len(all_params)}")
    print(f"  Activation layers to capture: {n_layers}")

    # results[layer_name][prompt_idx] = list of delta_h over candidates
    results = {name: {pi: [] for pi in range(len(DATASET))}
               for name in target_names}

    summary_idxs = sorted(set([
        0, n_layers//4, n_layers//2, 3*n_layers//4, n_layers-1
    ]))

    for prompt_idx, (prompt, _) in enumerate(DATASET):
        inp = tok(prompt, return_tensors="pt").to(device)
        print(f"\n  Prompt {prompt_idx}: {prompt!r}")
        print(f"  {'Layer':<55} {'depth':>5} {'mean delta_h':>14}")
        print(f"  {'-'*80}")

        for cand_idx, seed in enumerate(seeds):
            seed = int(seed)

            # Storage for activations
            h_clean    = {}
            h_perturbed = {}

            def make_capture_hook(storage, name):
                def fn(module, input, output):
                    storage[name] = output.detach().float()
                return fn

            # Register forward hooks on all target layers
            handles = []
            for tname, tmod in hook_obj._targets:
                handles.append(
                    tmod.register_forward_hook(make_capture_hook(h_clean, tname))
                )

            # --- Clean forward pass ---
            with torch.no_grad():
                model(**inp)

            for h in handles:
                h.remove()

            # --- Perturb ALL weights simultaneously (global WP) ---
            # Matches wp_conciseness.py: same seed, same noise for every param
            with torch.no_grad():
                for param_name, param in all_params:
                    gen = torch.Generator(device=param.device)
                    gen.manual_seed(seed)
                    noise = torch.randn(param.shape, generator=gen,
                                        device=param.device, dtype=param.dtype)
                    param.data.add_(SIGMA * noise)

            # --- Perturbed forward pass ---
            handles = []
            for tname, tmod in hook_obj._targets:
                handles.append(
                    tmod.register_forward_hook(make_capture_hook(h_perturbed, tname))
                )

            with torch.no_grad():
                model(**inp)

            for h in handles:
                h.remove()

            # --- Restore ALL weights ---
            with torch.no_grad():
                for param_name, param in all_params:
                    gen = torch.Generator(device=param.device)
                    gen.manual_seed(seed)
                    noise = torch.randn(param.shape, generator=gen,
                                        device=param.device, dtype=param.dtype)
                    param.data.sub_(SIGMA * noise)

            # --- Compute delta_h_t at each layer ---
            for layer_name in target_names:
                if layer_name not in h_clean or layer_name not in h_perturbed:
                    continue
                delta_h = (h_perturbed[layer_name] - h_clean[layer_name]).norm().item()
                results[layer_name][prompt_idx].append(delta_h)

        # Print summary layers
        for layer_idx, layer_name in enumerate(target_names):
            if layer_idx in summary_idxs:
                vals  = results[layer_name][prompt_idx]
                mean  = float(np.mean(vals)) if vals else 0.0
                depth = _block_index(layer_name)
                short = (layer_name if len(layer_name) <= 55
                         else "..." + layer_name[-52:])
                print(f"  {short:<55} {depth:>5} {mean:>14.6f}")

    return results, hook_obj._targets


# ===========================================================================
# DANP: cumulative activation perturbation — delta_h_t per layer
# ===========================================================================
def inspect_danp_delta_h(model, tok, device, seeds):
    """
    For each candidate (same ``seeds`` list as WP):
        1. Run clean forward  → capture h_t at every layer
        2. Run noisy forward  → capture h̃_t at every layer
           (noise at layer t propagates into layer t+1's input)
        3. delta_h_t = ||h̃_t - h_t|| at each layer

    ``hook.base_seed`` is set to seeds[cand_idx] so RNG matches WP per candidate.

    Averaged over POPULATION_SIZE candidates.
    """
    print("\n" + "="*70)
    print("DANP: computing delta_h_t per layer (cumulative activation perturbation)")
    print("="*70)

    hook = DANPHook(model, include="all", last_k=0, sigma=SIGMA, base_seed=SEED)
    for name in hook.R:
        hook.R[name] = hook.R[name].to(device)

    target_names = [name for name, _ in hook._targets]
    n_layers     = len(target_names)
    print(f"  Target layers: {n_layers}")

    summary_idxs = sorted(set([
        0, n_layers//4, n_layers//2, 3*n_layers//4, n_layers-1
    ]))

    results = {name: {pi: [] for pi in range(len(DATASET))}
               for name in target_names}

    for prompt_idx, (prompt, _) in enumerate(DATASET):
        inp = tok(prompt, return_tensors="pt").to(device)
        print(f"\n  Prompt {prompt_idx}: {prompt!r}")
        print(f"  {'Layer':<55} {'depth':>5} {'mean delta_h':>14}")
        print(f"  {'-'*80}")

        for cand_idx, seed in enumerate(seeds):
            pop_seed = int(seed)

            # Clean forward
            hook.base_seed = pop_seed
            hook.attach("clean")
            with torch.no_grad():
                model(**inp)
            captured_clean = {k: {kk: v.clone() for kk, v in vv.items()}
                              for k, vv in hook._captured_clean.items()}
            hook.detach()

            # Noisy forward (cumulative)
            hook.base_seed = pop_seed
            hook.attach("noisy")
            with torch.no_grad():
                model(**inp)
            captured_noisy = {k: {kk: v.clone() for kk, v in vv.items()}
                              for k, vv in hook._captured_noisy.items()}
            hook.detach()

            for layer_name in target_names:
                if layer_name not in captured_clean or layer_name not in captured_noisy:
                    continue
                a_clean = captured_clean[layer_name]["a_clean"]
                a_noisy = captured_noisy[layer_name]["a_noisy"]
                delta_h = (a_noisy - a_clean).norm().item()
                results[layer_name][prompt_idx].append(delta_h)

        # Print summary layers
        for layer_idx, layer_name in enumerate(target_names):
            if layer_idx in summary_idxs:
                vals  = results[layer_name][prompt_idx]
                mean  = float(np.mean(vals)) if vals else 0.0
                depth = _block_index(layer_name)
                short = (layer_name if len(layer_name) <= 55
                         else "..." + layer_name[-52:])
                print(f"  {short:<55} {depth:>5} {mean:>14.6f}")

    return results, hook._targets


# ===========================================================================
# Plot and save CSV
# ===========================================================================
def _plot_one_method_subplots(results, targets, suptitle, out_filename, marker, color, series_label):
    """Two prompts side by side; y-axis auto-scaled for this method only."""
    target_names = [name for name, _ in targets]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
    for prompt_idx, ax in enumerate(axes):
        points = []
        for layer_idx, name in enumerate(target_names):
            vals = results[name][prompt_idx]
            if vals:
                points.append((layer_idx, float(np.mean(vals)), name))
        points.sort(key=lambda x: x[0])
        x = [p[0] for p in points]
        y = [p[1] for p in points]
        ax.plot(
            x, y, f"{marker}-", color=color, markersize=2,
            linewidth=1.5, label=series_label,
        )
        ax.set_title(PROMPT_LABELS[prompt_idx], fontsize=11)
        ax.set_xlabel("Layer index (t)", fontsize=10)
        ax.set_ylabel("δh_t  (activation change norm)", fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    fig.suptitle(suptitle, fontsize=13, fontweight="bold")
    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, out_filename)
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved → {plot_path}")


def plot_and_save(wp_results, wp_targets, danp_results, danp_targets):
    target_names_wp   = [name for name, _ in wp_targets]
    target_names_danp = [name for name, _ in danp_targets]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
    rows = []

    for prompt_idx, (prompt_label, ax) in enumerate(zip(PROMPT_LABELS, axes)):

        # WP series: sort by layer index for clean x-axis
        wp_points = []
        for layer_idx, name in enumerate(target_names_wp):
            vals = wp_results[name][prompt_idx]
            if vals:
                wp_points.append((layer_idx, float(np.mean(vals)), name))
        wp_points.sort(key=lambda x: x[0])

        # DANP series
        danp_points = []
        for layer_idx, name in enumerate(target_names_danp):
            vals = danp_results[name][prompt_idx]
            if vals:
                danp_points.append((layer_idx, float(np.mean(vals)), name))
        danp_points.sort(key=lambda x: x[0])

        wp_x,   wp_y   = [p[0] for p in wp_points],   [p[1] for p in wp_points]
        danp_x, danp_y = [p[0] for p in danp_points], [p[1] for p in danp_points]

        ax.plot(wp_x,   wp_y,   "o-", color="steelblue",  markersize=2,
                linewidth=1.5, label="WP (global weight perturbation)")
        ax.plot(danp_x, danp_y, "s-", color="tomato", markersize=2,
                linewidth=1.5, label="DANP (cumulative activation perturbation)")

        ax.set_title(prompt_label, fontsize=11)
        ax.set_xlabel("Layer index (t)", fontsize=10)
        ax.set_ylabel("δh_t  (activation change norm)", fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        for idx, mean, name in wp_points:
            rows.append({"method": "WP",   "prompt_idx": prompt_idx,
                         "layer_idx": idx, "layer_name": name,
                         "depth": _block_index(name), "mean_delta_h": mean})
        for idx, mean, name in danp_points:
            rows.append({"method": "DANP", "prompt_idx": prompt_idx,
                         "layer_idx": idx, "layer_name": name,
                         "depth": _block_index(name), "mean_delta_h": mean})

    fig.suptitle("δh_t vs Layer Depth: Global WP vs DANP", fontsize=13, fontweight="bold")
    plt.tight_layout()

    plot_path = os.path.join(OUTPUT_DIR, "delta_h_plot.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Plot saved → {plot_path}")

    _plot_one_method_subplots(
        wp_results, wp_targets,
        "δh_t vs Layer Depth: WP (global weight perturbation)",
        "delta_h_plot_wp.png",
        "o", "steelblue",
        "WP (global weight perturbation)",
    )
    _plot_one_method_subplots(
        danp_results, danp_targets,
        "δh_t vs Layer Depth: DANP (cumulative activation perturbation)",
        "delta_h_plot_danp.png",
        "s", "tomato",
        "DANP (cumulative activation perturbation)",
    )

    csv_path = os.path.join(OUTPUT_DIR, "delta_h.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"  CSV  saved → {csv_path}")

    # Print bottom-line summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    for prompt_idx in range(len(DATASET)):
        print(f"\n  Prompt {prompt_idx}:")
        for method, points in [("WP", wp_points if prompt_idx==0 else []),
                                 ("DANP", danp_points if prompt_idx==0 else [])]:
            # recompute for this prompt
            if method == "WP":
                pts = [(i, float(np.mean(wp_results[n][prompt_idx])), n)
                       for i, n in enumerate(target_names_wp)
                       if wp_results[n][prompt_idx]]
            else:
                pts = [(i, float(np.mean(danp_results[n][prompt_idx])), n)
                       for i, n in enumerate(target_names_danp)
                       if danp_results[n][prompt_idx]]
            if not pts:
                continue
            first = pts[0][1]
            last  = pts[-1][1]
            print(f"    {method}: first layer={first:.4f}  last layer={last:.4f}  "
                  f"ratio={last/(first+1e-10):.2f}x")


# ===========================================================================
# Main
# ===========================================================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading {MODEL_NAME} ...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, cache_dir=HF_CACHE_DIR,
        torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
    )
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=HF_CACHE_DIR)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    tok.padding_side = "left"
    model = model.to(device).eval()
    print(f"Device: {device}")

    np.random.seed(SEED)
    seeds = np.random.randint(0, 2**30, size=POPULATION_SIZE, dtype=np.int64).tolist()

    wp_results,   wp_targets   = inspect_wp_delta_h(model, tok, device, seeds)
    danp_results, danp_targets = inspect_danp_delta_h(model, tok, device, seeds)
    plot_and_save(wp_results, wp_targets, danp_results, danp_targets)


if __name__ == "__main__":
    main()