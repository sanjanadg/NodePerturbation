#!/usr/bin/env python3
"""
Run a grid of danp_llm_v4.py jobs (subprocess per combo) and aggregate metrics.

Each run uses --no_save and --metrics_json so you do not write a full HF checkpoint
every trial, and you get baseline vs eval curve in JSON.

Practical knobs (comma-separated lists, no spaces inside numbers):

  Shell: do not put a space after a comma in a list, or the next word becomes a
  separate argv token (e.g. ``--scale_n_modes sqrt_n, n`` breaks: use
  ``--scale_n_modes sqrt_n,n`` or ``--scale_n_modes 'sqrt_n, n'``).

  --sigmas   --etas   --alphas   --n_populations   --np_includes   --last_ks
  --scale_n_modes   f(N) in η·f(N)·δL/‖δa‖²; validated by normalize_scale_n_mode()
    in this file (aliases e.g. cbrt_n, n_pow_2_3 → canonical names).

  Ablate: --scale_n_modes n_half,sqrt_n,n,cube_root_n,two_sqrt_n,half_sqrt_n,n_2_3

  Reward-only generation: --reward_do_samples false,true
    (passed through as danp_llm_v4.py --reward_do_sample when true; default is
    false,true for --objective reward, and false only for --objective ce)

  The sweep only forwards flags it defines. For v4-only options (e.g. generation
  dumps), pass the matching sweep flag such as --print_generation_each_epoch.

Example (reward — same grid knobs as CE; lists below are the defaults if omitted):

  cd /path/to/NodePerturbation
  TQDM_DISABLE=1 python3 sweep_danp_llm_v4.py \
    --epochs 20 --objective reward \
    --sigmas 0.001 \
    --etas 0.001 \
    --alphas 0 \
    --n_populations 1,10,30 \
    --np_includes mlp,all \
    --last_ks 0 \
    --scale_n_modes sqrt_n,cube_root_n,two_sqrt_n,half_sqrt_n,n_2_3 \
    --reward_do_samples false,true \
    --print_generation_each_epoch \
    --results_dir conciseness_task/results_danp_sweep_reward_413_3

CE objective often shows clearer loss movement on the dummy task than reward.

Example (CE):

  TQDM_DISABLE=1 python3 conciseness_task/sweep_danp_llm_v4.py \\
    --epochs 30 --objective ce \\
    --sigmas 0.001,0.01 \\
    --etas 0.001,0.01,0.1 \\
    --scale_n_modes n_half,sqrt_n,n \\
    --results_dir conciseness_task/results_danp_sweep_ce
"""

import argparse
import csv
import itertools
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List

# --- scale_n_mode (keep in sync with danp_llm_v4.py) --------------------------------
CANONICAL_SCALE_N_MODES = frozenset(
    {
        "n",
        "n_half",
        "sqrt_n",
        "cube_root_n",
        "two_sqrt_n",
        "half_sqrt_n",
        "n_2_3",
    }
)


def normalize_scale_n_mode(mode: str) -> str:
    """Return canonical --scale_n_mode name or raise ValueError."""
    m = (mode or "n").strip().lower().replace("-", "_")
    if m in ("n", "full_n"):
        return "n"
    if m in ("n_half", "half_n", "n_div_2"):
        return "n_half"
    if m in ("sqrt_n", "sqrtn", "sqrt"):
        return "sqrt_n"
    if m in ("cube_root_n", "cbrt_n", "n_cbrt", "nthroot_3"):
        return "cube_root_n"
    if m in ("two_sqrt_n", "2_sqrt_n", "double_sqrt_n", "sqrt_n_times_2"):
        return "two_sqrt_n"
    if m in (
        "half_sqrt_n",
        "sqrt_n_half",
        "one_half_sqrt_n",
        "0_5_sqrt_n",
        "sqrt_n_div_2",
    ):
        return "half_sqrt_n"
    if m in ("n_2_3", "n_pow_2_3", "nthroot_2_3", "n_two_thirds", "two_thirds_n"):
        return "n_2_3"
    raise ValueError(
        "Unknown scale_n_mode; use one of "
        + ", ".join(sorted(CANONICAL_SCALE_N_MODES))
        + " (aliases allowed, e.g. 2_sqrt_n → two_sqrt_n)."
    )


def _parse_float_list(s):
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def _parse_int_list(s):
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _parse_str_list(s):
    return [x.strip() for x in s.split(",") if x.strip()]


def _parse_bool_list(s: str) -> List[bool]:
    out: List[bool] = []
    for tok in s.split(","):
        t = tok.strip().lower()
        if not t:
            continue
        if t in ("0", "false", "f", "no", "n"):
            out.append(False)
        elif t in ("1", "true", "t", "yes", "y"):
            out.append(True)
        else:
            sys.exit(f"Invalid bool in --reward_do_samples: {tok!r} (use false,true)")
    if not out:
        sys.exit("--reward_do_samples must list at least one of false,true")
    return out


def main():
    p = argparse.ArgumentParser(description="Hyperparameter sweep for danp_llm_v4.py")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--eval_interval", type=int, default=1)
    p.add_argument("--model_name", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--hf_cache_dir", default="huggingface_cache")
    p.add_argument("--precision", default="bf16", choices=["fp16", "bf16", "fp32"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--objective", default="ce", choices=["ce", "reward"])
    p.add_argument("--max_new_tokens", type=int, default=100)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--sigmas", default="0.001,0.005,0.01", help="Comma-separated floats")
    p.add_argument("--etas", default="0.001,0.003,0.01", help="Comma-separated floats")
    p.add_argument("--alphas", default="0,0.0001", help="Comma-separated floats (0 = off)")
    p.add_argument("--n_populations", default="1,4", help="Comma-separated ints")
    p.add_argument("--np_includes", default="mlp,all", help="Comma-separated: head,attn,mlp,all")
    p.add_argument("--last_ks", default="0", help="Comma-separated ints")
    p.add_argument(
        "--scale_n_modes",
        default="n",
        help=        "Comma-separated --scale_n_mode values for danp_llm_v4.py (normalized "
        "in this script; keep in sync with danp_llm_v4). "
        f"Canonical: {', '.join(sorted(CANONICAL_SCALE_N_MODES))}. "
        "Shell: no space after commas unless quoted. Aliases allowed.",
    )
    p.add_argument(
        "--reward_do_samples",
        default=None,
        help="Comma-separated false/true: reward-path generation do_sample in "
        "danp_llm_v4.py. Default: false,true when --objective reward; false only when ce.",
    )
    p.add_argument("--results_dir", default="", help="Output dir for metrics + summary CSV")
    p.add_argument("--dry_run", action="store_true", help="Print commands only")
    p.add_argument(
        "--print_generation_each_epoch",
        action="store_true",
        help="Forward to danp_llm_v4.py (print sample generations after each epoch).",
    )
    args = p.parse_args()

    if args.reward_do_samples is None:
        args.reward_do_samples = (
            "false,true" if args.objective == "reward" else "false"
        )
    reward_do_samples = _parse_bool_list(args.reward_do_samples)

    repo_root = Path(__file__).resolve().parents[1]
    v4_script = Path(__file__).resolve().parent / "danp_llm_v4.py"

    sigmas = _parse_float_list(args.sigmas)
    etas = _parse_float_list(args.etas)
    alphas = _parse_float_list(args.alphas)
    n_pops = _parse_int_list(args.n_populations)
    includes = _parse_str_list(args.np_includes)
    last_ks = _parse_int_list(args.last_ks)
    scale_modes = []
    seen_sn = set()
    for sm in _parse_str_list(args.scale_n_modes):
        try:
            canon = normalize_scale_n_mode(sm)
        except ValueError as e:
            sys.exit(f"Invalid scale_n_mode {sm!r}: {e}")
        if canon not in seen_sn:
            seen_sn.add(canon)
            scale_modes.append(canon)

    for inc in includes:
        if inc not in {"head", "attn", "mlp", "all"}:
            sys.exit(f"Invalid np_include: {inc}")

    if not args.results_dir:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        results_dir = repo_root / "conciseness_task" / f"results_danp_v4_sweep_{stamp}"
    else:
        results_dir = Path(args.results_dir)
        if not results_dir.is_absolute():
            results_dir = repo_root / results_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    combos = list(
        itertools.product(
            sigmas,
            etas,
            alphas,
            n_pops,
            includes,
            last_ks,
            scale_modes,
            reward_do_samples,
        )
    )
    print(f"Total runs: {len(combos)}  ->  {results_dir}")

    summary_rows = []  # type: List[Dict]
    env = os.environ.copy()
    env.setdefault("TQDM_DISABLE", "1")

    for run_idx, (sigma, eta, alpha, n_pop, np_inc, last_k, sn_mode, r_sample) in enumerate(
        combos
    ):
        rs_tag = "rs1" if r_sample else "rs0"
        tag = (
            f"run_{run_idx:04d}_s{sigma:g}_e{eta:g}_a{alpha:g}_p{n_pop}_{np_inc}_k{last_k}_"
            f"sn_{sn_mode}_{rs_tag}"
        )
        tag = tag.replace(".", "p")  # filesystem-friendly
        out_dir = results_dir / tag
        out_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = out_dir / "metrics.json"

        cmd = [
            sys.executable,
            str(v4_script),
            "--model_name",
            args.model_name,
            "--hf_cache_dir",
            str(repo_root / args.hf_cache_dir),
            "--output_dir",
            str(out_dir),
            "--epochs",
            str(args.epochs),
            "--batch_size",
            str(args.batch_size),
            "--eval_interval",
            str(args.eval_interval),
            "--sigma",
            str(sigma),
            "--eta",
            str(eta),
            "--alpha",
            str(alpha),
            "--n_population",
            str(n_pop),
            "--np_include",
            np_inc,
            "--last_k",
            str(last_k),
            "--scale_n_mode",
            sn_mode,
            "--objective",
            args.objective,
            "--precision",
            args.precision,
            "--seed",
            str(args.seed),
            "--max_length",
            str(args.max_length),
            "--max_new_tokens",
            str(args.max_new_tokens),
            "--no_save",
            "--metrics_json",
            str(metrics_path),
        ]
        if r_sample:
            cmd.append("--reward_do_sample")
        if args.print_generation_each_epoch:
            cmd.append("--print_generation_each_epoch")

        print(f"\n[{run_idx + 1}/{len(combos)}] {tag}")
        if args.dry_run:
            print(" ", " ".join(cmd))
            continue

        t0 = time.time()
        proc = subprocess.run(cmd, cwd=str(repo_root), env=env)
        elapsed = time.time() - t0
        row = {
            "run_idx": run_idx,
            "sigma": sigma,
            "eta": eta,
            "alpha": alpha,
            "n_population": n_pop,
            "np_include": np_inc,
            "last_k": last_k,
            "scale_n_mode": sn_mode,
            "reward_do_sample": int(r_sample),
            "exit_code": proc.returncode,
            "seconds": round(elapsed, 2),
        }
        if proc.returncode != 0:
            row["baseline"] = ""
            row["final_eval"] = ""
            row["delta_eval"] = ""
            summary_rows.append(row)
            print(f"  FAILED exit={proc.returncode} ({elapsed:.1f}s)")
            continue

        with open(metrics_path, encoding="utf-8") as f:
            m = json.load(f)
        ev = m.get("eval_metric_history") or []
        bl = float(m.get("baseline", float("nan")))
        final = float(ev[-1]) if ev else float("nan")
        row["baseline"] = bl
        row["final_eval"] = final
        row["delta_eval"] = final - bl if ev else ""
        summary_rows.append(row)
        imp = row["delta_eval"]
        print(f"  baseline={bl:.6f}  final_eval={final:.6f}  delta={imp}  ({elapsed:.1f}s)")

    csv_path = results_dir / "sweep_summary.csv"
    if summary_rows and not args.dry_run:
        fields = [
            "run_idx",
            "sigma",
            "eta",
            "alpha",
            "n_population",
            "np_include",
            "last_k",
            "scale_n_mode",
            "reward_do_sample",
            "baseline",
            "final_eval",
            "delta_eval",
            "exit_code",
            "seconds",
        ]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            for r in summary_rows:
                w.writerow(r)
        print(f"\nWrote {csv_path}")

        ok = [
            r
            for r in summary_rows
            if r.get("exit_code") == 0 and r.get("delta_eval") != ""
        ]
        if ok and args.objective == "reward":
            best = max(ok, key=lambda r: float(r["delta_eval"]))
            print(
                "Best by delta_eval (reward — higher is better): "
                f"sigma={best['sigma']} eta={best['eta']} alpha={best['alpha']} "
                f"n_pop={best['n_population']} include={best['np_include']} last_k={best['last_k']} "
                f"reward_do_sample={best['reward_do_sample']} "
                f"delta={best['delta_eval']}"
            )
        elif ok and args.objective == "ce":
            best = min(ok, key=lambda r: float(r["delta_eval"]))
            print(
                "Best by delta_eval (CE — more negative delta = larger loss drop): "
                f"sigma={best['sigma']} eta={best['eta']} alpha={best['alpha']} "
                f"n_pop={best['n_population']} include={best['np_include']} last_k={best['last_k']} "
                f"reward_do_sample={best['reward_do_sample']} "
                f"delta={best['delta_eval']}  (final={best['final_eval']})"
            )


if __name__ == "__main__":
    main()
