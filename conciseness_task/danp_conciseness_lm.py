#!/usr/bin/env python3
"""
DANP finetuning on the conciseness task (same data pipeline as danp_llm_conciseness_full.py).

Uses Algorithm 1 (activity perturbation + decorrelation) from the toy model, implemented for
HF causal LMs via DANPFullHook in the repo root module.

Run from repo root:
  python conciseness_task/danp_conciseness_lm.py --model_name Qwen/Qwen2.5-0.5B-Instruct --output_dir ./out_danp_conc

Or from this directory:
  cd conciseness_task && python danp_conciseness_lm.py --help
"""

import os
import sys
import json
import argparse

# Repo root (parent of conciseness_task/)
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

_CT = os.path.dirname(os.path.abspath(__file__))
if _CT not in sys.path:
    sys.path.insert(0, _CT)

import numpy as np
import torch
from tqdm import tqdm

from wp_conciseness_task import compute_reward as wp_compute_reward, expand_wp_dummy_dataset

from danp_llm_conciseness_full import (
    DANPFullHook,
    compute_loss,
    run_danp_batch_step,
    load_conciseness_dataset,
    evaluate_mean_reward,
)
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerFast


def parse_args():
    p = argparse.ArgumentParser(description="DANP finetuning: conciseness task (CE loss, hooked linears)")
    p.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--hf_cache_dir", type=str, default=None)
    p.add_argument("--output_dir", type=str, default="./out_danp_conciseness_lm")
    p.add_argument("--n_train", type=int, default=64)
    p.add_argument("--n_eval", type=int, default=50)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--eta", type=float, default=1e-3)
    p.add_argument("--sigma", type=float, default=1e-3)
    p.add_argument("--alpha", type=float, default=1e-4)
    p.add_argument("--np_include", type=str, default="mlp", choices=["head", "attn", "mlp", "all"])
    p.add_argument("--last_k", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval_interval", type=int, default=1)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--n_population", type=int, default=1)
    p.add_argument("--max_scale", type=float, default=1e2)
    p.add_argument("--max_update", type=float, default=1.0)
    p.add_argument(
        "--objective",
        type=str,
        default="ce",
        choices=["ce", "reward"],
        help="ce: teacher-forcing CE for δL; reward: δL=R_clean-R_noisy from generate()",
    )
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--reward_do_sample", action="store_true")
    p.add_argument(
        "--precision",
        type=str,
        default="bf16",
        choices=["fp16", "bf16", "fp32"],
        help="Model dtype (matches wp_conciseness-style flag)",
    )
    p.add_argument(
        "--dataset",
        type=str,
        default="wp_dummy",
        choices=["wp_dummy", "hf"],
        help="wp_dummy: same 2 prompts as wp_conciseness.py; hf: xsum/cnn/synthetic via load_conciseness_dataset",
    )
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    if "HF_HOME" not in os.environ and "HUGGINGFACE_HUB_CACHE" not in os.environ:
        hf_cache = os.path.join(_ROOT, ".cache", "huggingface")
        os.makedirs(hf_cache, exist_ok=True)
        os.environ["HF_HOME"] = hf_cache

    if args.dataset == "wp_dummy":
        train_data, eval_data = expand_wp_dummy_dataset(args.n_train, args.n_eval)
        reward_fn = wp_compute_reward
        print("Dataset: wp_dummy (same as wp_conciseness.py)")
    else:
        print("Loading HF conciseness dataset (xsum/cnn fallback / synthetic)...")
        train_data, eval_data = load_conciseness_dataset(args.n_train, args.n_eval)
        reward_fn = wp_compute_reward
    print(f"Train: {len(train_data)}, Eval: {len(eval_data)}")

    print(f"Loading model {args.model_name}...")
    dtype = (
        torch.float32
        if args.precision == "fp32"
        else (torch.bfloat16 if args.precision == "bf16" else torch.float16)
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        cache_dir=args.hf_cache_dir,
        torch_dtype=dtype,
    )
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=args.hf_cache_dir)
    except (ValueError, OSError):
        tokenizer = PreTrainedTokenizerFast.from_pretrained(args.model_name, cache_dir=args.hf_cache_dir)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    model.eval()
    if args.objective == "reward":
        baseline_eval = evaluate_mean_reward(
            model, tokenizer, eval_data, device,
            args.max_new_tokens, args.reward_do_sample, reward_fn,
        )
        print(f"[BASELINE] Eval mean reward: {baseline_eval:.4f}")
    else:
        with torch.no_grad():
            eval_losses = []
            for i in range(0, len(eval_data), args.batch_size):
                batch = eval_data[i : i + args.batch_size]
                eval_losses.append(compute_loss(model, tokenizer, batch, device, args.max_length).item())
        baseline_eval = float(np.mean(eval_losses))
        print(f"[BASELINE] Eval CE loss: {baseline_eval:.4f}")

    train_losses = []
    eval_losses_hist = []

    for epoch in range(args.epochs):
        model.train()
        indices = np.random.permutation(len(train_data))
        epoch_losses = []

        hook = DANPFullHook(
            model,
            args.np_include,
            args.last_k,
            sigma=args.sigma,
            alpha=args.alpha,
            base_seed=args.seed + epoch * 10000,
        )
        for name in hook.R:
            hook.R[name] = hook.R[name].to(device)

        pbar = tqdm(
            range(0, len(train_data), args.batch_size),
            desc=f"Epoch {epoch + 1}/{args.epochs}",
        )
        for step, start in enumerate(pbar):
            batch_idx = indices[start : start + args.batch_size]
            batch = [train_data[i] for i in batch_idx]
            hook.base_seed = args.seed + epoch * 10000 + step

            L_clean, delta_L = run_danp_batch_step(
                model,
                tokenizer,
                batch,
                device,
                hook,
                eta=args.eta,
                alpha=args.alpha,
                max_length=args.max_length,
                objective=args.objective,
                compute_reward_fn=reward_fn,
                max_new_tokens=args.max_new_tokens,
                reward_do_sample=args.reward_do_sample,
                n_population=args.n_population,
                verbose=args.verbose and step == 0,
                max_scale=args.max_scale,
                max_update=args.max_update,
            )
            epoch_losses.append(L_clean)
            pbar.set_postfix(loss=f"{L_clean:.4f}", delta_L=f"{delta_L:.4f}")

        train_losses.append(float(np.mean(epoch_losses)))

        if (epoch + 1) % args.eval_interval == 0:
            model.eval()
            if args.objective == "reward":
                eval_loss = evaluate_mean_reward(
                    model, tokenizer, eval_data, device,
                    args.max_new_tokens, args.reward_do_sample, reward_fn,
                )
            else:
                with torch.no_grad():
                    ev = []
                    for i in range(0, len(eval_data), args.batch_size):
                        batch = eval_data[i : i + args.batch_size]
                        ev.append(compute_loss(model, tokenizer, batch, device, args.max_length).item())
                eval_loss = float(np.mean(ev))
            eval_losses_hist.append(eval_loss)
            tag = "reward" if args.objective == "reward" else "CE loss"
            print(f"[Epoch {epoch + 1}] Train: {train_losses[-1]:.4f}, Eval {tag}: {eval_loss:.4f}")

    n_eval_hist = len(eval_losses_hist)
    results = {
        "config": vars(args),
        "train_losses": train_losses,
        "eval_losses": eval_losses_hist,
        "eval_epochs": list(range(1, n_eval_hist + 1)),
        "baseline_eval": baseline_eval,
    }
    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {args.output_dir}/results.json")

    save_dir = os.path.join(args.output_dir, "final_model")
    print(f"Saving model to {save_dir}...")
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print("Done.")


if __name__ == "__main__":
    main()
