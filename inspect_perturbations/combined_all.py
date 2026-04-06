#!/usr/bin/env python3
"""
inspect_epoch1_perturbations.py

Captures perturbations for the entire first epoch for both WP and DANP v2,
across all target layers, both dataset examples, and all 30 population samples.

For WP:   shows weight-space perturbation per candidate (input-blind, uniform norm).
For DANP: shows eps_norm vs delta_a_norm per layer per candidate, revealing
          whether cumulative propagation is working (delta_a_norm > eps_norm
          at deeper layers).

USAGE:
    python inspect_epoch1_perturbations.py

Outputs:
    wp_epoch1.csv       — one row per (prompt, candidate, layer)
    danp_epoch1.csv     — one row per (prompt, candidate, layer)
    summary printed to stdout comparing first/mid/last layer statistics
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import csv, os, re

# ---------------------------------------------------------------------------
# UPDATE THIS IMPORT to match your project structure
# ---------------------------------------------------------------------------
from conciseness_task.danp_llm_v3 import DANPHook, get_target_linears

# ---------------------------------------------------------------------------
# Shared config — match your training scripts exactly
# ---------------------------------------------------------------------------
SIGMA           = 0.001
POPULATION_SIZE = 30
WP_SEED         = 33     # initial_seed in wp_conciseness.py
DANP_SEED       = 33     # base_seed in danp_llm_v2.py
MODEL_NAME      = "Qwen/Qwen2.5-0.5B-Instruct"
HF_CACHE_DIR    = "huggingface_cache"
OUTPUT_DIR      = "."

DATASET = [
    ("Solve: 3 + 5 =", "8"),
    ("If all birds can fly and penguins are birds, can penguins fly?", "No"),
]


# ---------------------------------------------------------------------------
# Helper: extract transformer block depth from layer name
# ---------------------------------------------------------------------------
def _block_index(name: str):
    m = re.search(r'\.layers\.(\d+)\.', name) or re.search(r'\.h\.(\d+)\.', name)
    return int(m.group(1)) if m else -1


# ===========================================================================
# WP / ES perturbation inspection
# ===========================================================================
def inspect_wp(model, device):
    """
    WP perturbs weights, not activations, so the perturbation is:
        sigma * eps   where eps ~ N(0, I), shape = param.shape

    Key properties we expect to confirm:
      - norm is IDENTICAL across all candidates (depends only on param size)
      - norm is IDENTICAL across both prompts (input-blind)
      - std ≈ sigma for every candidate
    """
    print("\n" + "="*70)
    print("WP PERTURBATION INSPECTION — epoch 1, all layers, both prompts")
    print("="*70)

    # Use same layer filter as DANP for fair comparison
    target_layers = [(n, p) for n, p in model.named_parameters()
                     if isinstance(model.get_submodule(
                         n.rsplit(".", 1)[0]
                         if "." in n else n
                     ), torch.nn.Linear)
                     and ("mlp" in n or "attn" in n or "q_proj" in n
                          or "k_proj" in n or "v_proj" in n)]

    np.random.seed(WP_SEED)
    seeds = np.random.randint(0, 2**30, size=POPULATION_SIZE, dtype=np.int64).tolist()

    rows = []

    # Summary print: just candidate 0, first/mid/last layer, both prompts
    n_layers = len(target_layers)
    summary_idxs = sorted(set([0, n_layers // 2, n_layers - 1]))

    for prompt_idx, (prompt, target_text) in enumerate(DATASET):
        if prompt_idx == 0:
            print(f"\n  {'Cand':>5} {'Layer':<50} {'depth':>5} "
                  f"{'norm':>10} {'mean':>10} {'std':>10}")
            print(f"  {'-'*95}")

        for cand_idx, seed in enumerate(seeds):
            for layer_idx, (layer_name, param) in enumerate(target_layers):
                gen = torch.Generator(device=param.device)
                gen.manual_seed(int(seed))
                noise        = torch.randn(param.shape, generator=gen,
                                           device=param.device, dtype=param.dtype)
                perturbation = SIGMA * noise
                depth        = _block_index(layer_name)

                row = {
                    "prompt_idx":  prompt_idx,
                    "prompt":      prompt,
                    "cand_idx":    cand_idx,
                    "seed":        seed,
                    "layer_idx":   layer_idx,
                    "layer_name":  layer_name,
                    "layer_depth": depth,
                    "norm":        round(perturbation.norm().item(), 8),
                    "mean":        round(perturbation.mean().item(), 8),
                    "std":         round(perturbation.std().item(),  8),
                    "min":         round(perturbation.min().item(),  8),
                    "max":         round(perturbation.max().item(),  8),
                }
                rows.append(row)

                # Print candidate 0, summary layers, prompt 0 only
                if cand_idx == 0 and prompt_idx == 0 and layer_idx in summary_idxs:
                    short = (layer_name if len(layer_name) <= 50
                             else "..." + layer_name[-47:])
                    print(f"  {cand_idx:>5} {short:<50} {depth:>5} "
                          f"{row['norm']:>10.6f} {row['mean']:>10.6f} "
                          f"{row['std']:>10.6f}")

    out_path = os.path.join(OUTPUT_DIR, "wp_epoch1.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n  Saved {len(rows)} rows → {out_path}")
    return rows


# ===========================================================================
# DANP perturbation inspection
# ===========================================================================
def inspect_danp(model, tok, device):
    """
    DANP perturbs activations. For each candidate we run:
        1. clean forward  → capture a_clean at every target layer
        2. noisy forward  → capture a_noisy at every target layer
                            (noise at layer l propagates into layer l+1's input)

    We then compute:
        eps_norm     = norm of raw eps added at THIS layer alone
                       (approximated as sigma * sqrt(n_activation_units))
        delta_a_norm = norm of (a_noisy - a_clean)
                       At layer 0: delta_a_norm ≈ eps_norm  (no upstream noise)
                       At deeper layers: delta_a_norm > eps_norm if cumulative
                       propagation is working correctly (v2 fix)

    ratio = delta_a_norm / eps_norm:
        ≈ 1.0 at layer 0  (baseline)
        > 1.0 at deeper layers  (upstream noise has compounded)
        varies between prompts  (input-dependence)
    """
    print("\n" + "="*70)
    print("DANP PERTURBATION INSPECTION — epoch 1, all layers, both prompts")
    print("="*70)

    hook = DANPHook(model, include="all", last_k=0, sigma=SIGMA, base_seed=DANP_SEED)
    for name in hook.R:
        hook.R[name] = hook.R[name].to(device)

    target_names = [name for name, _ in hook._targets]
    n_layers     = len(target_names)
    print(f"  Total target layers: {n_layers}")

    summary_idxs = sorted(set([
        0,
        n_layers // 4,
        n_layers // 2,
        3 * n_layers // 4,
        n_layers - 1,
    ]))

    rows = []

    for prompt_idx, (prompt, target_text) in enumerate(DATASET):
        print(f"\n  Prompt {prompt_idx}: {prompt!r}")
        print(f"  {'Cand':>5} {'Layer':<50} {'depth':>5} "
              f"{'eps_norm':>10} {'delta_norm':>10} {'ratio':>7}")
        print(f"  {'-'*95}")

        inp = tok(prompt, return_tensors="pt").to(device)

        for cand_idx in range(POPULATION_SIZE):
            pop_seed = DANP_SEED + cand_idx * 31337

            # --- Clean forward ---
            hook.base_seed = pop_seed
            hook.attach("clean")
            with torch.no_grad():
                model(**inp)
            captured_clean = {
                k: {kk: v.clone() for kk, v in vv.items()}
                for k, vv in hook._captured_clean.items()
            }
            hook.detach()

            # --- Noisy forward ---
            # In v2, returning a_noisy from each hook causes it to propagate
            # forward into the next layer's input, matching Algorithm 1.
            hook.base_seed = pop_seed
            hook.attach("noisy")
            with torch.no_grad():
                model(**inp)
            captured_noisy = {
                k: {kk: v.clone() for kk, v in vv.items()}
                for k, vv in hook._captured_noisy.items()
            }
            hook.detach()

            for layer_idx, layer_name in enumerate(target_names):
                if layer_name not in captured_clean or layer_name not in captured_noisy:
                    continue

                a_clean      = captured_clean[layer_name]["a_clean"]
                a_noisy      = captured_noisy[layer_name]["a_noisy"]
                delta_a      = a_noisy - a_clean
                delta_a_norm = delta_a.norm().item()

                # eps_norm reference: what sigma*randn of this shape would give
                # At layer 0 delta_a_norm should equal this; deeper layers exceed it
                eps_norm = SIGMA * (a_clean.numel() ** 0.5)
                ratio    = delta_a_norm / (eps_norm + 1e-10)
                depth    = _block_index(layer_name)

                row = {
                    "prompt_idx":   prompt_idx,
                    "prompt":       prompt,
                    "cand_idx":     cand_idx,
                    "pop_seed":     pop_seed,
                    "layer_idx":    layer_idx,
                    "layer_name":   layer_name,
                    "layer_depth":  depth,
                    "n_units":      a_clean.numel(),
                    "eps_norm":     round(eps_norm,      8),
                    "delta_a_norm": round(delta_a_norm,  8),
                    "ratio":        round(ratio,          6),
                }
                rows.append(row)

                # Print: candidate 0, summary layers only
                if cand_idx == 0 and layer_idx in summary_idxs:
                    short = (layer_name if len(layer_name) <= 50
                             else "..." + layer_name[-47:])
                    print(f"  {cand_idx:>5} {short:<50} {depth:>5} "
                          f"{eps_norm:>10.4f} {delta_a_norm:>10.4f} {ratio:>7.3f}")

    out_path = os.path.join(OUTPUT_DIR, "danp_epoch1.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n  Saved {len(rows)} rows → {out_path}")
    return rows


# ===========================================================================
# Summary comparison
# ===========================================================================
def print_summary(wp_rows, danp_rows):
    print("\n" + "="*70)
    print("SUMMARY — WP vs DANP epoch 1")
    print("="*70)

    # ------------------------------------------------------------------
    # WP: confirm norm is uniform across candidates, layers, and prompts
    # ------------------------------------------------------------------
    wp_norms = [r["norm"] for r in wp_rows]
    print(f"\nWP weight-space perturbation norm (all candidates, layers, prompts):")
    print(f"  mean = {np.mean(wp_norms):.6f}")
    print(f"  std  = {np.std(wp_norms):.8f}  ← should be ≈ 0 (uniform)")
    print(f"  min  = {np.min(wp_norms):.6f}")
    print(f"  max  = {np.max(wp_norms):.6f}")

    # Confirm prompt-blindness: same norm for prompt 0 vs prompt 1
    wp_p0 = np.mean([r["norm"] for r in wp_rows if r["prompt_idx"] == 0])
    wp_p1 = np.mean([r["norm"] for r in wp_rows if r["prompt_idx"] == 1])
    print(f"\n  Norm by prompt: prompt0={wp_p0:.6f}  prompt1={wp_p1:.6f}")
    print(f"  → WP is input-blind: norms {'ARE' if abs(wp_p0-wp_p1)<1e-6 else 'are NOT'} identical")

    # ------------------------------------------------------------------
    # DANP: show how delta_a_norm grows with depth, differs by prompt
    # ------------------------------------------------------------------
    layer_idxs = sorted(set(r["layer_idx"] for r in danp_rows))
    first_idx  = layer_idxs[0]
    last_idx   = layer_idxs[-1]
    mid_idx    = layer_idxs[len(layer_idxs) // 2]

    print(f"\nDANP activation-space perturbation (delta_a_norm vs eps_norm):")
    print(f"  {'Layer':>10} {'prompt':>8} {'eps_norm':>10} {'delta_norm':>11} {'ratio':>7}")
    print(f"  {'-'*52}")

    for layer_label, layer_idx in [("first", first_idx), ("mid", mid_idx), ("last", last_idx)]:
        for prompt_idx in [0, 1]:
            subset      = [r for r in danp_rows
                           if r["layer_idx"] == layer_idx
                           and r["prompt_idx"] == prompt_idx]
            if not subset:
                continue
            eps_m   = np.mean([r["eps_norm"]     for r in subset])
            delta_m = np.mean([r["delta_a_norm"] for r in subset])
            ratio_m = np.mean([r["ratio"]        for r in subset])
            print(f"  {layer_label:>10} {prompt_idx:>8} {eps_m:>10.4f} "
                  f"{delta_m:>11.4f} {ratio_m:>7.3f}")

    # Cumulative propagation check
    first_delta = np.mean([r["delta_a_norm"] for r in danp_rows if r["layer_idx"] == first_idx])
    last_delta  = np.mean([r["delta_a_norm"] for r in danp_rows if r["layer_idx"] == last_idx])
    print(f"\n  Cumulative propagation: first layer={first_delta:.6f}  last layer={last_delta:.6f}")
    if last_delta > first_delta * 1.05:
        print("  ✓ delta_a_norm grows with depth — noise is compounding across layers (v2 fix working)")
    else:
        print("  ✗ delta_a_norm does NOT grow — check that hook returns a_noisy (not a_clean)")

    # Input-dependence check at last layer
    last_p0 = np.mean([r["delta_a_norm"] for r in danp_rows
                        if r["layer_idx"] == last_idx and r["prompt_idx"] == 0])
    last_p1 = np.mean([r["delta_a_norm"] for r in danp_rows
                        if r["layer_idx"] == last_idx and r["prompt_idx"] == 1])
    diff_pct = abs(last_p0 - last_p1) / (max(last_p0, last_p1) + 1e-10) * 100
    print(f"\n  Last layer delta_a_norm by prompt:")
    print(f"    prompt0 = {last_p0:.6f}")
    print(f"    prompt1 = {last_p1:.6f}")
    print(f"    difference = {diff_pct:.2f}%")
    if diff_pct > 1.0:
        print("  ✓ Perturbation differs between prompts — input-dependence confirmed")
    else:
        print("  ✗ Perturbation same across prompts — still input-blind at this layer")

    # Final comparison
    print(f"\n{'='*70}")
    print("BOTTOM LINE FOR MENTOR:")
    print(f"  WP   norm std across all candidates/layers/prompts: "
          f"{np.std(wp_norms):.2e}  (≈ 0, fully isotropic)")
    print(f"  DANP ratio (delta_a / eps) at first layer: "
          f"{np.mean([r['ratio'] for r in danp_rows if r['layer_idx']==first_idx]):.3f}  (≈ 1.0)")
    print(f"  DANP ratio (delta_a / eps) at last  layer: "
          f"{np.mean([r['ratio'] for r in danp_rows if r['layer_idx']==last_idx]):.3f}  (> 1.0 = directed)")
    print(f"{'='*70}\n")


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
    print(f"Device: {device}\n")

    wp_rows   = inspect_wp(model, device)
    danp_rows = inspect_danp(model, tok, device)
    print_summary(wp_rows, danp_rows)


if __name__ == "__main__":
    main()