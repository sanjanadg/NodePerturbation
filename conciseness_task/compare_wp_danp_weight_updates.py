#!/usr/bin/env python3
"""
Compare per-iteration weight updates for WP (wp_conciseness.py) vs DANP (danp_llm_v4.py).

Runs two independent training trajectories from the same initial weights, logging
||ΔW||_F each iteration. On the WP model weights, also computes the DANP update
(counterfactual) for apples-to-apples direction comparison on overlapping Linear
weights.

Default hyperparameters (override via CLI):
  WP:   sigma=0.001, alpha=0.0005, pop=30, iterations=100
  DANP: eta=0.001, sigma=0.001, alpha=0.0001, pop=30, epochs=100

USAGE (from repo root):
  python3 conciseness_task/compare_wp_danp_weight_updates.py
  python3 conciseness_task/compare_wp_danp_weight_updates.py --model_name Qwen/Qwen2.5-0.5B-Instruct --no_save_plots
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import sys
from dataclasses import dataclass, asdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging as hf_logging

hf_logging.set_verbosity_error()

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from danp_llm_v4 import (  # noqa: E402
    CANONICAL_SCALE_N_MODES,
    DANPHook,
    WP_DUMMY_EXAMPLES,
    compute_reward,
    danp_grad_single,
    normalize_scale_n_mode,
)

# Match wp_conciseness.py
MAX_NEW_TOKENS = 100
DO_SAMPLE = False
INITIAL_SEED = 33


@dataclass
class IterMetrics:
    iteration: int
    wp_update_frob_all: float
    wp_update_frob_linear: float
    danp_update_frob: float
    danp_at_wp_frob: float
    cosine_wp_vs_danp_at_wp: float
    wp_reward_mean: float
    wp_reward_std: float
    danp_reward_mean: float
    danp_scale_mean: float
    danp_delta_L_mean: float


def parse_args():
    p = argparse.ArgumentParser(description="Compare WP vs DANP weight updates per iteration")
    p.add_argument("--model_name", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--hf_cache_dir", default="huggingface_cache")
    p.add_argument("--precision", default="bf16", choices=["fp16", "bf16", "fp32"])
    p.add_argument("--output_dir", default="./out_compare_wp_danp_updates")
    p.add_argument("--iterations", type=int, default=100)
    p.add_argument("--seed", type=int, default=INITIAL_SEED)
    # WP
    p.add_argument("--wp_sigma", type=float, default=0.001)
    p.add_argument("--wp_alpha", type=float, default=0.0005)
    p.add_argument("--wp_population", type=int, default=30)
    # DANP
    p.add_argument("--danp_eta", type=float, default=0.001)
    p.add_argument("--danp_sigma", type=float, default=0.001)
    p.add_argument("--danp_alpha", type=float, default=0.0001)
    p.add_argument("--danp_population", type=int, default=30)
    p.add_argument("--danp_np_include", default="all", choices=["head", "attn", "mlp", "all"])
    p.add_argument("--danp_last_k", type=int, default=0)
    p.add_argument(
        "--danp_scale_n_mode",
        default="sqrt_n",
        choices=sorted(CANONICAL_SCALE_N_MODES),
        type=normalize_scale_n_mode,
        help=(
            "DANP f(N) in scale = eta·f(N)·delta_L/||delta_a||^2. "
            "Choices: n (N), n_half (N/2), sqrt_n (sqrt(N)), "
            "cube_root_n (N^(1/3)), two_sqrt_n (2*sqrt(N)), "
            "half_sqrt_n (sqrt(N)/2), n_2_3 (N^(2/3))."
        ),
    )
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Device for model and tensors (default: cuda > mps > cpu).",
    )
    p.add_argument("--no_save_plots", action="store_true")
    return p.parse_args()


def resolve_device(request: str) -> torch.device:
    if request != "auto":
        dev = torch.device(request)
        if request == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is not available")
        if request == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("--device mps requested but MPS is not available")
        return dev
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def clone_model_from_state(model: torch.nn.Module, device: torch.device) -> torch.nn.Module:
    """Second trajectory copy without copy.deepcopy (much lower peak RAM)."""
    clone = AutoModelForCausalLM.from_config(model.config)
    clone.load_state_dict(model.state_dict(), assign=True)
    return clone.to(device=device, dtype=next(model.parameters()).dtype).eval()


def _linear_weight_keys(hook: DANPHook) -> list[str]:
    return [name + ".weight" for name, _ in hook._targets]


def frob_norm_dict(updates: dict[str, torch.Tensor], keys: list[str] | None = None) -> float:
    total = 0.0
    for key, tensor in updates.items():
        if keys is not None and key not in keys:
            continue
        total += float(torch.sum(tensor.float() ** 2).item())
    return float(np.sqrt(total))


def cosine_updates(
    a: dict[str, torch.Tensor],
    b: dict[str, torch.Tensor],
    keys: list[str],
) -> float:
    parts_a, parts_b = [], []
    for key in keys:
        if key not in a or key not in b:
            continue
        parts_a.append(a[key].float().reshape(-1))
        parts_b.append(b[key].float().reshape(-1))
    if not parts_a:
        return float("nan")
    va = torch.cat(parts_a)
    vb = torch.cat(parts_b)
    denom = va.norm() * vb.norm()
    if float(denom) < 1e-30:
        return float("nan")
    return float((va @ vb / denom).item())


@torch.no_grad()
def evaluate_wp_reward(
    model,
    tokenizer,
    device,
    prompt: str,
    target: str,
) -> float:
    inp = tokenizer(prompt, return_tensors="pt", padding=True, padding_side="left")
    ids = inp["input_ids"].to(device)
    mask = inp["attention_mask"].to(device)
    out = model.generate(
        ids,
        attention_mask=mask,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=DO_SAMPLE,
    )
    generated = tokenizer.decode(out[0], skip_special_tokens=True)
    return float(compute_reward(generated, target))


@torch.no_grad()
def wp_population_rewards(
    model,
    tokenizer,
    device,
    seeds: list[int],
    sigma: float,
    dataset: list[tuple[str, str]],
) -> list[float]:
    """One scalar reward per seed (mean over dataset), matching wp_conciseness.py."""
    rewards = []
    for seed in seeds:
        for _name, param in model.named_parameters():
            gen = torch.Generator(device=param.device)
            gen.manual_seed(int(seed))
            noise = torch.randn(
                param.shape,
                generator=gen,
                device=param.device,
                dtype=param.dtype,
            )
            param.data.add_(sigma * noise)

        total = 0.0
        for prompt, target in dataset:
            total += evaluate_wp_reward(model, tokenizer, device, prompt, target)
        rewards.append(total / len(dataset))

        for _name, param in model.named_parameters():
            gen = torch.Generator(device=param.device)
            gen.manual_seed(int(seed))
            noise = torch.randn(
                param.shape,
                generator=gen,
                device=param.device,
                dtype=param.dtype,
            )
            param.data.add_(-sigma * noise)

    return rewards


@torch.no_grad()
def compute_wp_updates(
    model,
    seeds: list[int],
    rewards_normalized: np.ndarray,
    alpha: float,
    sigma: float,
) -> dict[str, torch.Tensor]:
    """Return dict param_name -> ΔW (amount added in wp_conciseness: param += ΔW)."""
    pop = len(seeds)
    updates: dict[str, torch.Tensor] = {}
    for name, param in model.named_parameters():
        gen = torch.Generator(device=param.device)
        update = torch.zeros_like(param)
        for seed_idx, seed in enumerate(seeds):
            r_norm = float(rewards_normalized[seed_idx])
            gen.manual_seed(int(seed))
            noise = torch.randn(
                param.shape,
                generator=gen,
                device=param.device,
                dtype=param.dtype,
            )
            noise.mul_(r_norm)
            update.add_(noise)
        update.div_(pop)
        updates[name] = alpha * update
    return updates


@torch.no_grad()
def apply_wp_updates(model, updates: dict[str, torch.Tensor]) -> None:
    for name, param in model.named_parameters():
        param.data.add_(updates[name])


@torch.no_grad()
def compute_danp_batch_updates(
    model,
    tokenizer,
    hook: DANPHook,
    batch: list[tuple[str, str]],
    device,
    *,
    eta: float,
    n_population: int,
    alpha: float,
    base_seed: int,
    scale_n_mode: str,
    max_new_tokens: int,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], float, float, list[float]]:
    """Accumulate DANP weight/R deltas (pre-apply) averaged over batch; same keys as danp_batch_step."""
    accumulated: dict[str, torch.Tensor] = {}
    accumulated_dec: dict[str, torch.Tensor] = {}
    total_dL = 0.0
    total_scale = 0.0
    batch_pop_noisy_rewards: list[float] = []

    for i, example in enumerate(batch):
        example_seed = base_seed + i * 997
        weight_updates, dec_updates, _L, dL, sc_m, _upn, pop_rw = danp_grad_single(
            model,
            tokenizer,
            example,
            device,
            hook,
            eta=eta,
            max_length=512,
            objective="reward",
            max_new_tokens=max_new_tokens,
            reward_do_sample=DO_SAMPLE,
            reward_continuation_only=True,
            n_population=n_population,
            alpha=alpha,
            base_seed=example_seed,
            verbose=False,
            scale_n_mode=scale_n_mode,
        )
        total_dL += dL
        total_scale += sc_m
        batch_pop_noisy_rewards.extend(pop_rw)
        for key, upd in weight_updates.items():
            if key not in accumulated:
                accumulated[key] = torch.zeros_like(upd)
            accumulated[key] += upd
        for name, dR in dec_updates.items():
            if name not in accumulated_dec:
                accumulated_dec[name] = torch.zeros_like(dR)
            accumulated_dec[name] += dR

    B = len(batch)
    out: dict[str, torch.Tensor] = {}
    for key, acc in accumulated.items():
        out[key] = (acc / B).clone()
    dec_out: dict[str, torch.Tensor] = {}
    for name, acc in accumulated_dec.items():
        dec_out[name] = (acc / B).clone()
    if batch_pop_noisy_rewards and B > 0:
        expected = B * n_population
        if len(batch_pop_noisy_rewards) == expected:
            arr = np.asarray(batch_pop_noisy_rewards, dtype=np.float64).reshape(B, n_population)
            batch_pop_noisy_rewards = [float(arr[:, j].mean()) for j in range(n_population)]
    return out, dec_out, total_dL / B, total_scale / B, batch_pop_noisy_rewards


@torch.no_grad()
def apply_danp_updates(
    model,
    hook: DANPHook,
    updates: dict[str, torch.Tensor],
    decorrelation_updates: dict[str, torch.Tensor],
) -> None:
    for key, avg in updates.items():
        mod_path = key[:-7]
        mod = model.get_submodule(mod_path)
        mod.weight.sub_(avg.to(mod.weight.dtype))
    for name, avg in decorrelation_updates.items():
        hook.R[name].sub_(avg.to(device=hook.R[name].device, dtype=hook.R[name].dtype))


def run_experiment(args) -> list[IterMetrics]:
    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    dtype = dtype_map[args.precision]
    device = resolve_device(args.device)
    dataset = list(WP_DUMMY_EXAMPLES)

    print(f"Loading {args.model_name} on {device} ...", flush=True)
    if device.type == "cpu":
        print(
            "  Note: running on CPU — this script keeps two model copies in RAM and "
            "runs many forward passes per iteration. Use --device cuda or --device mps "
            "if available; reduce --iterations / population on low-memory machines.",
            flush=True,
        )

    wp_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        cache_dir=args.hf_cache_dir,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    tok = AutoTokenizer.from_pretrained(args.model_name, cache_dir=args.hf_cache_dir)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    tok.padding_side = "left"

    wp_model = wp_model.to(device).eval()
    print("  Cloning model for DANP trajectory (state_dict copy, not deepcopy) ...", flush=True)
    danp_model = clone_model_from_state(wp_model, device)
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    hook_wp = DANPHook(
        wp_model, args.danp_np_include, args.danp_last_k,
        sigma=args.danp_sigma, base_seed=args.seed,
    )
    hook_danp = DANPHook(
        danp_model, args.danp_np_include, args.danp_last_k,
        sigma=args.danp_sigma, base_seed=args.seed,
    )
    for h in (hook_wp, hook_danp):
        for name in h.R:
            h.R[name] = h.R[name].to(device)

    linear_keys = _linear_weight_keys(hook_wp)
    print(f"Linear target layers: {len(linear_keys)}")
    print(
        f"WP: sigma={args.wp_sigma}, alpha={args.wp_alpha}, pop={args.wp_population}, "
        f"iters={args.iterations}"
    )
    print(
        f"DANP: eta={args.danp_eta}, sigma={args.danp_sigma}, alpha={args.danp_alpha}, "
        f"pop={args.danp_population}, np_include={args.danp_np_include}, "
        f"scale_n_mode={args.danp_scale_n_mode}"
    )

    np.random.seed(args.seed)
    metrics: list[IterMetrics] = []

    for iteration in range(args.iterations):
        seeds = np.random.randint(0, 2**30, size=args.wp_population, dtype=np.int64).tolist()
        danp_base_seed = args.seed + iteration * 100_000

        # --- WP on wp_model trajectory ---
        wp_rewards = wp_population_rewards(
            wp_model, tok, device, seeds, args.wp_sigma, dataset,
        )
        r_arr = np.array(wp_rewards, dtype=np.float32)
        r_norm = (r_arr - r_arr.mean()) / (r_arr.std() + 1e-8)
        wp_updates = compute_wp_updates(
            wp_model, seeds, r_norm, args.wp_alpha, args.wp_sigma,
        )

        # --- DANP on danp_model trajectory ---
        danp_updates, danp_dec, dL_mean, scale_mean, danp_pop_rewards = compute_danp_batch_updates(
            danp_model,
            tok,
            hook_danp,
            dataset,
            device,
            eta=args.danp_eta,
            n_population=args.danp_population,
            alpha=args.danp_alpha,
            base_seed=danp_base_seed,
            scale_n_mode=args.danp_scale_n_mode,
            max_new_tokens=MAX_NEW_TOKENS,
        )

        # --- DANP update computed on wp_model weights (same W_t as WP step) ---
        danp_at_wp, _danp_dec_at_wp, _, _, _ = compute_danp_batch_updates(
            wp_model,
            tok,
            hook_wp,
            dataset,
            device,
            eta=args.danp_eta,
            n_population=args.danp_population,
            alpha=args.danp_alpha,
            base_seed=danp_base_seed,
            scale_n_mode=args.danp_scale_n_mode,
            max_new_tokens=MAX_NEW_TOKENS,
        )

        wp_linear = {k: wp_updates[k] for k in linear_keys if k in wp_updates}
        cos = cosine_updates(wp_linear, danp_at_wp, linear_keys)

        m = IterMetrics(
            iteration=iteration + 1,
            wp_update_frob_all=frob_norm_dict(wp_updates),
            wp_update_frob_linear=frob_norm_dict(wp_updates, linear_keys),
            danp_update_frob=frob_norm_dict(danp_updates),
            danp_at_wp_frob=frob_norm_dict(danp_at_wp),
            cosine_wp_vs_danp_at_wp=cos,
            wp_reward_mean=float(r_arr.mean()),
            wp_reward_std=float(r_arr.std()),
            danp_reward_mean=float(np.mean(danp_pop_rewards)) if danp_pop_rewards else float("nan"),
            danp_scale_mean=float(scale_mean),
            danp_delta_L_mean=float(dL_mean),
        )
        metrics.append(m)

        apply_wp_updates(wp_model, wp_updates)
        apply_danp_updates(danp_model, hook_danp, danp_updates, danp_dec)

        print(
            f"iter {iteration + 1:3d}/{args.iterations} | "
            f"||ΔW||_WP={m.wp_update_frob_all:.4e} "
            f"(linear={m.wp_update_frob_linear:.4e}) | "
            f"||ΔW||_DANP={m.danp_update_frob:.4e} | "
            f"||ΔW||_DANP@WP={m.danp_at_wp_frob:.4e} | "
            f"cos(WP,DANP@WP)={m.cosine_wp_vs_danp_at_wp:.4f}",
            flush=True,
        )

        if device.type == "cuda":
            torch.cuda.empty_cache()

    return metrics


def save_results(metrics: list[IterMetrics], args, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "weight_update_metrics.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(metrics[0]).keys()))
        writer.writeheader()
        for m in metrics:
            writer.writerow(asdict(m))
    print(f"Wrote {csv_path}")

    meta = {
        "model_name": args.model_name,
        "iterations": args.iterations,
        "seed": args.seed,
        "wp": {
            "sigma": args.wp_sigma,
            "alpha": args.wp_alpha,
            "population": args.wp_population,
        },
        "danp": {
            "eta": args.danp_eta,
            "sigma": args.danp_sigma,
            "alpha": args.danp_alpha,
            "population": args.danp_population,
            "np_include": args.danp_np_include,
            "scale_n_mode": args.danp_scale_n_mode,
        },
        "dataset": WP_DUMMY_EXAMPLES,
    }
    json_path = os.path.join(output_dir, "config.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote {json_path}")

    if args.no_save_plots:
        return

    iters = [m.iteration for m in metrics]
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax = axes[0]
    ax.plot(iters, [m.wp_update_frob_all for m in metrics], label="WP (all params)", color="steelblue")
    ax.plot(iters, [m.wp_update_frob_linear for m in metrics], "--", label="WP (linear only)", color="cornflowerblue")
    ax.plot(iters, [m.danp_update_frob for m in metrics], label="DANP (trajectory)", color="tomato")
    ax.plot(iters, [m.danp_at_wp_frob for m in metrics], ":", label="DANP @ WP weights", color="darkorange")
    ax.set_ylabel("||ΔW||_F")
    ax.set_title("Weight update magnitude per iteration")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    ax2 = axes[1]
    ax2.plot(iters, [m.cosine_wp_vs_danp_at_wp for m in metrics], color="purple", marker=".", ms=3)
    ax2.axhline(0.0, color="gray", lw=0.8)
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("cosine(WP ΔW, DANP ΔW)")
    ax2.set_title("Direction alignment on linear weights (DANP computed at WP weights)")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "wp_vs_danp_weight_updates.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {plot_path}")


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    metrics = run_experiment(args)
    save_results(metrics, args, args.output_dir)
    print("Done.")


if __name__ == "__main__":
    main()
