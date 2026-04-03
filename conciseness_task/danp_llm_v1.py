#!/usr/bin/env python3
"""
DANP for LLMs — incremental v1.

STAGE 1 GOALS (this file):
  1. R matrix semantics aligned with toy danp.py:
       x_star = R @ x  (decorrelate the INPUT, not the output)
       a = W @ x_star + eps  (noise on pre-activation, before activation fn)
     The current hook applies R to the *output* (y → R @ y), which is inconsistent
     with the toy model and makes the gradient computation wrong.
  2. global norm_sq computed by concatenating delta_a across ALL target layers,
     exactly as in the toy model's `compute_grad_estimate`.
  3. No _get_R_in_producer heuristics — we maintain one R per layer that
     decorrelates that layer's input. We track captured inputs directly.
  4. Reward sign is explicit and documented.
  5. Everything else kept as close to wp_conciseness.py (ES / WP) as possible
     so results are comparable.

WHAT IS NOT IN THIS VERSION:
  - Decorrelation updates (R is initialised to I and frozen). Add in v2
    once the weight update gradient is verified to be flowing.
  - Population-size > 1 averaging (kept in API but defaults to 1 for clarity).
  - Multi-GPU / accelerate (single GPU focus).

  R decorrelation: when --alpha > 0, same rule as toy_model/danp.py (batched cov over tokens).

DATA:
  - Training always cycles the same two WP dummy examples (see WP_DUMMY_EXAMPLES);
    fixed order every epoch (no shuffle). Eval/baseline uses the same two.

USAGE:
  python danp_llm_v1.py --objective ce --epochs 2 --batch_size 2 --verbose
  python danp_llm_v1.py --objective reward --epochs 2 --batch_size 2 --verbose
  # Debug: print target / generated text / reward (reward objective only)
  python danp_llm_v1.py --objective reward --log_first_batch_each_epoch
  python danp_llm_v1.py --objective reward --log_generations_every 5
  python danp_llm_v1.py --objective ce --print_generation_each_epoch
"""

import os, sys, re, time, hashlib, argparse
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerFast
from transformers.utils import logging as hf_logging

hf_logging.set_verbosity_error()
torch.backends.cuda.matmul.allow_tf32 = True

# ---------------------------------------------------------------------------
# Defaults (aligned with wp_conciseness.py where applicable)
# ---------------------------------------------------------------------------
DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_ETA = 1e-3          # weight learning rate (same role as ALPHA in ES)
DEFAULT_SIGMA = 1e-3        # noise std for node perturbation
DEFAULT_ALPHA = 1e-4        # decorrelation rate for R update (same role as toy danp.alpha)
DEFAULT_MAX_NEW_TOKENS = 100
DEFAULT_POPULATION = 1      # increase for lower-variance gradient estimates


# ===========================================================================
# Argument parsing
# ===========================================================================
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", default=DEFAULT_MODEL)
    p.add_argument("--hf_cache_dir", default="huggingface_cache")
    p.add_argument("--output_dir", default="./out_danp_v1")
    p.add_argument("--epochs",  type=int, default=5)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--max_length",    type=int, default=512,
                   help="Sequence length cap for teacher-forcing (CE objective)")
    p.add_argument("--max_new_tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS,
                   help="Generation cap (reward objective)")
    p.add_argument("--eta",   type=float, default=DEFAULT_ETA)
    p.add_argument("--sigma", type=float, default=DEFAULT_SIGMA)
    p.add_argument(
        "--alpha",
        type=float,
        default=DEFAULT_ALPHA,
        help="Decorrelation rate for R: R -= alpha*(cov-diag)@R from clean x_star (0 disables)",
    )
    p.add_argument("--np_include", default="mlp",
                   choices=["head", "attn", "mlp", "all"],
                   help="Which Linear layers to perturb")
    p.add_argument("--last_k", type=int, default=0,
                   help="If >0, only perturb the last k transformer blocks")
    p.add_argument("--n_population", type=int, default=DEFAULT_POPULATION,
                   help="Noisy-forward samples to average gradient over")
    p.add_argument("--max_scale",  type=float, default=1e2,
                   help="Hard clip on |eta*N*delta_L/norm_sq| to prevent explosion")
    p.add_argument("--max_update", type=float, default=1.0,
                   help="Hard clip on per-element weight update")
    p.add_argument("--objective", default="ce", choices=["ce", "reward"])
    p.add_argument("--reward_do_sample", action="store_true")
    p.add_argument("--precision", default="bf16", choices=["fp16", "bf16", "fp32"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval_interval", type=int, default=1)
    p.add_argument("--verbose", action="store_true")
    p.add_argument(
        "--log_generations_every",
        type=int,
        default=0,
        help="reward only: print target / generated text / reward every N global steps (0=off)",
    )
    p.add_argument(
        "--log_first_batch_each_epoch",
        action="store_true",
        help="reward only: also print on the first batch of each epoch",
    )
    p.add_argument(
        "--log_generations_max_chars",
        type=int,
        default=600,
        help="truncate printed generated text to this many characters",
    )
    p.add_argument(
        "--print_generation_each_epoch",
        action="store_true",
        help="After each epoch, print greedy generations for each WP example (hooks detached)",
    )
    p.add_argument(
        "--reward_full_string",
        action="store_true",
        help="Reward objective only: score len(full decode) vs target (legacy). "
        "Default is continuation-only (new tokens) vs target length.",
    )
    return p.parse_args()


# ===========================================================================
# Fixed training data (same two examples as wp_conciseness / wp_conciseness_task)
# ===========================================================================
# _DUMMY_PAIRS = [
#     ("Summarise in one sentence: The cat sat on the mat and looked around.", " A cat sat on a mat."),
#     ("Summarise in one sentence: She went to the store and bought milk.", " She bought milk."),
#     ("Summarise in one sentence: The sun rose over the mountains at dawn.", " The sun rose at dawn."),
#     ("Summarise in one sentence: He read a book by the fireplace all evening.", " He read by the fireplace."),
# ]

WP_DUMMY_EXAMPLES = [
    ("Solve: 3 + 5 =", "8"),
    ("If all birds can fly and penguins are birds, can penguins fly?", "No"),
]


def compute_reward(generated_text: str, target_text: str) -> float:
    """Negative absolute length difference (higher = better).

    `generated_text` is either the full decoded output or the continuation only,
    depending on how `generate_reward` is called.
    """
    return -abs(len(generated_text) - len(target_text))


# ===========================================================================
# Layer selection helpers
# ===========================================================================
def _infer_total_blocks(model) -> int:
    names = [n for n, _ in model.named_modules()
             if re.search(r'\.layers\.(\d+)\.', n) or re.search(r'\.h\.(\d+)\.', n)]
    pat = r'\.layers\.(\d+)\.' if any('layers' in n for n in names) else r'\.h\.(\d+)\.'
    idxs = [int(re.search(pat, n).group(1)) for n in names if re.search(pat, n)]
    return max(idxs) + 1 if idxs else 0


def _block_index(name: str):
    m = re.search(r'\.layers\.(\d+)\.', name) or re.search(r'\.h\.(\d+)\.', name)
    return int(m.group(1)) if m else None


def _is_target_linear(name, mod, include, last_k, total_blocks):
    if not isinstance(mod, torch.nn.Linear):
        return False
    lname = name.lower()
    if last_k > 0 and total_blocks > 0:
        bi = _block_index(name)
        if bi is not None and bi < total_blocks - last_k:
            return False
    if include == "head":
        return lname.endswith("lm_head")
    if include == "attn":
        return any(k in lname for k in
                   ("attn", "attention", "q_proj", "k_proj", "v_proj",
                    "o_proj", "in_proj", "c_attn", "c_proj")) and "mlp" not in lname
    if include == "mlp":
        return any(k in lname for k in
                   ("mlp", "ffn", "gate_proj", "up_proj", "down_proj",
                    "fc1", "fc2", "c_fc")) or (
                   "c_proj" in lname and "mlp" in lname)
    return True  # "all"


def get_target_linears(model, include, last_k):
    total_blocks = _infer_total_blocks(model)
    return [(name, mod) for name, mod in model.named_modules()
            if _is_target_linear(name, mod, include, last_k, total_blocks)]


# ===========================================================================
# DANP Hook — v1: R decorrelates INPUT (matching toy danp.py)
# ===========================================================================
class DANPHook:
    """
    Hooks into target Linear layers.

    Forward pass (with_noise=True):
        x_star = R @ x        # decorrelate input  [matches toy model]
        a      = W @ x_star   # standard linear
        a_noisy= a + eps      # eps ~ N(0, sigma^2 I)
        return a_noisy        # NOTE: no R on output in this version

    Forward pass (with_noise=False):
        x_star = R @ x
        a      = W @ x_star
        return a

    Captured per layer (key = layer name):
        x_star_clean: R @ x_clean   (input to W in clean pass)
        a_clean:      W @ x_star    (pre-activation, clean)
        a_noisy:      a + eps       (pre-activation, noisy)  — None in clean pass

    R matrices:
        Initialised to identity; shape (in_features, in_features).
        Updated after each sample (batch-averaged) as in toy danp.py:
            R -= alpha * (cov - diag) @ R
        with cov from clean x_star (batched across tokens).

    WHY INPUT, NOT OUTPUT:
        The toy model (danp.py) computes:
            x_star = R[l-1] @ x[l-1]
            a_l    = W[l-1] @ x_star
            delta_a = a_noisy - a_clean      # both computed with same x_star
            grad = outer(delta_a, x_star)    # x_star is the LAYER INPUT
        If we apply R to the output instead, grad would use a different x_star
        than the one that produced a_clean / a_noisy, breaking the estimate.
    """

    def __init__(self, model, include, last_k, sigma, base_seed):
        self.sigma = float(sigma)
        self.base_seed = int(base_seed)
        self._targets = get_target_linears(model, include, last_k)
        self._orig_fwd = {}

        # R[name]: shape (in_features, in_features), on CPU initially
        self.R = {
            name: torch.eye(mod.in_features, dtype=torch.float32)
            for name, mod in self._targets
        }

        # Captured activations — populated during hooked forward passes
        self._captured_clean: dict = {}
        self._captured_noisy: dict = {}
        self._mode = "clean"   # "clean" | "noisy"

    # ------------------------------------------------------------------ #
    # Attach / detach                                                       #
    # ------------------------------------------------------------------ #
    def attach(self, mode: str):
        """mode: 'clean' or 'noisy'"""
        assert mode in ("clean", "noisy")
        self._mode = mode
        if mode == "clean":
            self._captured_clean = {}
        else:
            self._captured_noisy = {}

        for name, mod in self._targets:
            self._wrap(name, mod)

    def detach(self):
        for name, mod in self._targets:
            if name in self._orig_fwd:
                mod.forward = self._orig_fwd.pop(name)

    # ------------------------------------------------------------------ #
    # Internal forward wrapper                                              #
    # ------------------------------------------------------------------ #
    def _wrap(self, name: str, mod: torch.nn.Linear):
        orig = mod.forward
        self._orig_fwd[name] = orig

        R   = self.R[name]
        sig = self.sigma
        uid = _stable_uid(name)
        mode_ref = [self._mode]   # mutable cell so the closure sees updates

        def hooked_forward(x):
            # --- reshape to 2-D for matrix ops ---
            orig_shape = x.shape
            x2 = x.reshape(-1, orig_shape[-1]).float()

            # x_star = R @ x  (decorrelate input)
            R_dev = R.to(x2.device)
            x_star = x2 @ R_dev.T          # shape (N, in_features)

            with torch.no_grad():
                # Standard linear on decorrelated input
                # We call orig with x_star cast back to the weight dtype
                a = orig(x_star.to(mod.weight.dtype))   # (N, out_features)
                a32 = a.float()

                if mode_ref[0] == "noisy":
                    g = torch.Generator(device=a32.device)
                    g.manual_seed(_seed_for(uid, self.base_seed, x2.shape[0]))
                    # randn_like(..., generator=) needs PyTorch 2.0+; randn supports older releases
                    eps = torch.randn(
                        a32.shape, device=a32.device, dtype=a32.dtype, generator=g
                    ) * sig
                    a_out = a32 + eps
                    # Store for gradient computation
                    self._captured_noisy[name] = {
                        "x_star": x_star.detach().clone(),
                        "a_noisy": a_out.detach().clone(),
                    }
                else:
                    a_out = a32
                    self._captured_clean[name] = {
                        "x_star": x_star.detach().clone(),
                        "a_clean": a32.detach().clone(),
                    }

            # Reshape back and cast to original dtype
            a_out = a_out.to(x.dtype).reshape(*orig_shape[:-1], a_out.shape[-1])
            return a_out

        mod.forward = hooked_forward


def decorrelation_delta_r(
    x_star: torch.Tensor,
    R: torch.Tensor,
    alpha: float,
    dec_clip: float = 0.1,
):
    """
    Single decorrelation step matching toy_model/danp.py (per-vector outer product),
    extended to many tokens: cov = (X^T X) / N, diag = diag(mean(X^2, dim=0)).

    Returns ΔR such that the caller should do R -= ΔR (same as synthetic_data.train_danp_batch).
    """
    if alpha <= 0:
        return None
    x = x_star.float().reshape(-1, x_star.shape[-1])
    n, d = x.shape
    if n == 0 or d == 0:
        return None
    Rf = R.float()
    if n > 1:
        cov = (x.T @ x) / n
    else:
        cov = x.T @ x
    diag = torch.diag((x ** 2).mean(dim=0))
    dec = alpha * (cov - diag) @ Rf
    dec = torch.clamp(dec, -dec_clip, dec_clip)
    if torch.isnan(dec).any() or torch.isinf(dec).any():
        return None
    return dec


def _stable_uid(name: str) -> int:
    h = hashlib.blake2s(name.encode(), digest_size=4).digest()
    return int.from_bytes(h, "little") & 0x7FFFFFFF


def _seed_for(uid: int, base: int, n_tokens: int) -> int:
    return (base ^ uid ^ (n_tokens * 2654435761)) & 0x7FFFFFFF


# ===========================================================================
# Batch preparation  (teacher-forcing)
# ===========================================================================
def prepare_batch(batch, tokenizer, device, max_length, label_ignore=-100):
    input_ids_list, labels_list = [], []
    for prompt, target in batch:
        p_ids = tokenizer.encode(prompt, add_special_tokens=True)
        t_ids = tokenizer.encode(target, add_special_tokens=False)
        full  = (p_ids + t_ids)[:max_length]
        labs  = ([label_ignore] * len(p_ids) + t_ids)[:max_length]
        input_ids_list.append(full)
        labels_list.append(labs)

    pad = tokenizer.pad_token_id or 0
    L   = max(len(x) for x in input_ids_list)
    B   = len(batch)
    ids  = torch.full((B, L), pad, dtype=torch.long, device=device)
    mask = torch.zeros((B, L), dtype=torch.long, device=device)
    labs = torch.full((B, L), label_ignore, dtype=torch.long, device=device)
    for i in range(B):
        n = len(input_ids_list[i])
        ids [i, :n] = torch.tensor(input_ids_list[i], device=device)
        mask[i, :n] = 1
        labs[i, :n] = torch.tensor(labels_list[i], device=device)
    return ids, mask, labs


# ===========================================================================
# CE loss (teacher-forcing)
# ===========================================================================
def ce_loss(model, ids, mask, labs):
    out = model(input_ids=ids, attention_mask=mask)
    return F.cross_entropy(
        out.logits.view(-1, model.config.vocab_size),
        labs.view(-1),
        ignore_index=-100,
    )


# ===========================================================================
# Reward generation (greedy by default, matching WP)
# ===========================================================================
@torch.no_grad()
def generate_reward(
    model,
    tokenizer,
    prompt,
    target,
    device,
    max_new_tokens,
    do_sample,
    reward_continuation_only: bool = True,
):
    """Generate and compute reward.

    Returns:
        r: scalar reward
        full_text: full decoded sequence (prompt + continuation), for logging
        scored_text: substring used in `compute_reward` (continuation if
            reward_continuation_only else full_text)
    """
    inp = tokenizer(prompt, return_tensors="pt", padding=True, padding_side="left")
    ids = inp["input_ids"].to(device)
    amsk = inp["attention_mask"].to(device)
    out = model.generate(
        input_ids=ids,
        attention_mask=amsk,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        pad_token_id=tokenizer.pad_token_id,
    )
    prompt_len = ids.shape[1]
    full_text = tokenizer.decode(out[0], skip_special_tokens=True)
    if reward_continuation_only:
        cont_ids = out[0][prompt_len:]
        scored_text = tokenizer.decode(cont_ids, skip_special_tokens=True)
    else:
        scored_text = full_text
    r = compute_reward(scored_text, target)
    return r, full_text, scored_text


# ===========================================================================
# Core DANP gradient estimate — single sample
# ===========================================================================
def danp_grad_single(
    model, tokenizer, example, device, hook,
    eta, max_length,
    objective,
    max_new_tokens, reward_do_sample,
    reward_continuation_only,
    n_population,
    max_scale, max_update,
    alpha,
    verbose=False,
):
    """
    Compute weight update dict for a single (prompt, target) example.

    Returns:
        weight_updates: dict  name+".weight" -> update tensor (same dtype as param)
        decorrelation_updates: dict  layer name -> ΔR (same shape as hook.R[name])
        L_clean:        float  clean loss / reward
        delta_L_mean:   float  mean(L_noisy - L_clean) over population
    """
    prompt, target = example

    # ------------------------------------------------------------------ #
    # Clean forward                                                         #
    # ------------------------------------------------------------------ #
    hook.attach("clean")
    with torch.no_grad():
        if objective == "ce":
            ids, mask, labs = prepare_batch([example], tokenizer, device, max_length)
            out = model(input_ids=ids, attention_mask=mask)
            L_clean = F.cross_entropy(
                out.logits.view(-1, model.config.vocab_size),
                labs.view(-1), ignore_index=-100,
            ).item()
        else:
            # reward objective: generate, then reward
            # SIGN NOTE: compute_reward returns higher = better.
            # We want to MINIMISE -reward, so L_clean = -R_clean.
            # Then delta_L = L_noisy - L_clean = -R_noisy - (-R_clean) = R_clean - R_noisy.
            # If noise HURT reward (R_noisy < R_clean), delta_L > 0, update subtracts noise dir.
            R_clean, _, _ = generate_reward(
                model, tokenizer, prompt, target, device,
                max_new_tokens, reward_do_sample, reward_continuation_only,
            )
            L_clean = -float(R_clean)   # convert to loss convention (lower = better)
    captured_clean = {k: {kk: v.clone() for kk, v in vv.items()}
                      for k, vv in hook._captured_clean.items()}
    hook.detach()

    # ------------------------------------------------------------------ #
    # Noisy forward(s) — accumulate gradient estimates                     #
    # ------------------------------------------------------------------ #
    accumulated = {}   # name+".weight" -> sum of updates
    delta_L_sum = 0.0

    for pop_i in range(n_population):
        hook.base_seed = hook.base_seed + pop_i * 31337   # vary seed per sample
        hook.attach("noisy")
        with torch.no_grad():
            if objective == "ce":
                ids, mask, labs = prepare_batch([example], tokenizer, device, max_length)
                out = model(input_ids=ids, attention_mask=mask)
                L_noisy = F.cross_entropy(
                    out.logits.view(-1, model.config.vocab_size),
                    labs.view(-1), ignore_index=-100,
                ).item()
            else:
                R_noisy, _, _ = generate_reward(
                    model, tokenizer, prompt, target, device,
                    max_new_tokens, reward_do_sample, reward_continuation_only,
                )
                L_noisy = -float(R_noisy)
        captured_noisy = {k: {kk: v.clone() for kk, v in vv.items()}
                          for k, vv in hook._captured_noisy.items()}
        hook.detach()

        if not (np.isfinite(L_clean) and np.isfinite(L_noisy)):
            continue

        delta_L = L_noisy - L_clean   # positive => noise made things worse
        delta_L_sum += delta_L

        # ---------------------------------------------------------------- #
        # global norm_sq: concatenate delta_a across ALL target layers     #
        # This exactly matches the toy model's compute_grad_estimate.       #
        # ---------------------------------------------------------------- #
        delta_a_parts = []
        for name, _ in hook._targets:
            if name not in captured_clean or name not in captured_noisy:
                continue
            a_c = captured_clean[name]["a_clean"]
            a_n = captured_noisy[name]["a_noisy"]
            delta_a_parts.append((a_n - a_c).flatten())

        if not delta_a_parts:
            continue

        delta_a_cat = torch.cat(delta_a_parts)
        norm_sq = float((delta_a_cat ** 2).sum().item()) + 1e-8
        # N = total number of activation units across all layers
        N = delta_a_cat.numel()

        # scale = eta * N * delta_L / norm_sq  (toy model Eq. 6)
        delta_L_clamped = float(np.clip(delta_L, -1e4, 1e4))
        scale_raw = eta * N * delta_L_clamped / norm_sq
        scale = float(np.clip(scale_raw, -max_scale, max_scale))

        if not np.isfinite(scale):
            continue

        if verbose and pop_i == 0:
            print(f"\n[DANP grad] L_clean={L_clean:.4f} L_noisy={L_noisy:.4f} "
                  f"delta_L={delta_L:.6f} N={N} norm_sq={norm_sq:.4e} "
                  f"scale_raw={scale_raw:.4e} scale={scale:.4e}")

        # Per-layer weight update  (toy model: W[l] -= update, see `step`)
        for name, mod in hook._targets:
            if name not in captured_clean or name not in captured_noisy:
                continue
            x_star = captured_clean[name]["x_star"]   # input to W in clean pass
            a_c    = captured_clean[name]["a_clean"]
            a_n    = captured_noisy[name]["a_noisy"]
            delta_a = a_n - a_c

            # Reshape to (T, D) — T = sequence tokens * batch
            if x_star.dim() > 2:
                x_star  = x_star.reshape(-1, x_star.shape[-1])
                delta_a = delta_a.reshape(-1, delta_a.shape[-1])

            # grad = outer(delta_a, x_star) — but batched:
            # sum over token dimension => (out_features, in_features)
            grad = delta_a.T @ x_star   # (out_f, in_f)
            update = scale * grad       # toy: W -= eta*N*delta_L*grad/norm_sq
            update = torch.clamp(update.float(), -max_update, max_update)

            if torch.isnan(update).any() or torch.isinf(update).any():
                continue

            key = name + ".weight"
            if key not in accumulated:
                accumulated[key] = torch.zeros_like(update)
            accumulated[key] += update

    # Average over population
    weight_updates = {}
    for key, acc in accumulated.items():
        avg = acc / max(1, n_population)
        # Cast to param dtype
        mod_path = key[:-7]                  # strip ".weight"
        mod = model.get_submodule(mod_path)
        weight_updates[key] = avg.to(mod.weight.dtype)

    delta_L_mean = delta_L_sum / max(1, n_population)

    # Decorrelation: R -= alpha (cov - diag) @ R from clean x_star (toy danp.py)
    decorrelation_updates = {}
    if alpha > 0:
        for name in captured_clean:
            x_star = captured_clean[name]["x_star"]
            if x_star.dim() > 2:
                x_star = x_star.reshape(-1, x_star.shape[-1])
            dec = decorrelation_delta_r(x_star, hook.R[name], alpha)
            if dec is not None:
                decorrelation_updates[name] = dec

    return weight_updates, decorrelation_updates, float(L_clean), float(delta_L_mean)


# ===========================================================================
# Batch step — accumulate then apply (like train_danp_batch in synthetic_data.py)
# ===========================================================================
def danp_batch_step(model, tokenizer, batch, device, hook,
                    eta, max_length, objective,
                    max_new_tokens, reward_do_sample,
                    reward_continuation_only,
                    n_population, max_scale, max_update,
                    alpha,
                    base_seed, verbose=False):
    """
    One batch update step.

    For each example:
      - compute weight update dict (gradient estimate)
    Accumulate across batch, then apply averaged update.

    Returns: (mean_L_clean, mean_delta_L)
    """
    total_L, total_dL = 0.0, 0.0
    accumulated = {}
    accumulated_dec = {}

    for i, example in enumerate(batch):
        hook.base_seed = base_seed + i * 997   # different seed per example

        updates, dec_updates, L_clean, dL = danp_grad_single(
            model, tokenizer, example, device, hook,
            eta=eta, max_length=max_length,
            objective=objective,
            max_new_tokens=max_new_tokens,
            reward_do_sample=reward_do_sample,
            reward_continuation_only=reward_continuation_only,
            n_population=n_population,
            max_scale=max_scale, max_update=max_update,
            alpha=alpha,
            verbose=(verbose and i == 0),
        )
        total_L  += L_clean
        total_dL += dL

        for key, upd in updates.items():
            if key not in accumulated:
                accumulated[key] = torch.zeros_like(upd)
            accumulated[key] += upd

        for name, dR in dec_updates.items():
            if name not in accumulated_dec:
                accumulated_dec[name] = torch.zeros_like(dR)
            accumulated_dec[name] += dR

    # Apply averaged updates (weights first, then R — same order as synthetic_data.train_danp_batch)
    B = len(batch)
    with torch.no_grad():
        for key, acc in accumulated.items():
            avg = acc / B
            if torch.isnan(avg).any() or torch.isinf(avg).any():
                continue
            mod_path = key[:-7]
            mod = model.get_submodule(mod_path)
            # W -= update  (toy model sign convention)
            mod.weight.sub_(avg.to(mod.weight.dtype))

        for name, acc in accumulated_dec.items():
            avg = acc / B
            if torch.isnan(avg).any() or torch.isinf(avg).any():
                continue
            R = hook.R[name]
            R.sub_(avg.to(device=R.device, dtype=R.dtype))

    return total_L / B, total_dL / B


# ===========================================================================
# Evaluation
# ===========================================================================
@torch.no_grad()
def eval_ce(model, tokenizer, data, device, max_length, batch_size):
    losses = []
    for i in range(0, len(data), batch_size):
        b = data[i:i + batch_size]
        ids, mask, labs = prepare_batch(b, tokenizer, device, max_length)
        out = model(input_ids=ids, attention_mask=mask)
        l = F.cross_entropy(
            out.logits.view(-1, model.config.vocab_size),
            labs.view(-1), ignore_index=-100,
        ).item()
        losses.append(l)
    return float(np.mean(losses))


@torch.no_grad()
def eval_reward(
    model, tokenizer, data, device, max_new_tokens, do_sample,
    reward_continuation_only: bool = True,
):
    rewards = []
    for prompt, target in data:
        r, _, _ = generate_reward(
            model, tokenizer, prompt, target, device,
            max_new_tokens, do_sample, reward_continuation_only,
        )
        rewards.append(r)
    return float(np.mean(rewards))


def print_generations_after_epoch(
    hook,
    model,
    tokenizer,
    pairs,
    device,
    max_new_tokens,
    do_sample,
    max_chars,
    epoch_idx,
    objective,
    reward_continuation_only: bool = True,
):
    """Detach hooks and print model.generate output for every (prompt, target) pair."""
    hook.detach()
    model.eval()
    print(f"\n========== Epoch {epoch_idx + 1} — generations (no hooks) ==========", flush=True)
    for i, (prompt, target) in enumerate(pairs):
        r, full_text, scored = generate_reward(
            model, tokenizer, prompt, target, device,
            max_new_tokens, do_sample, reward_continuation_only,
        )
        shown = full_text if len(full_text) <= max_chars else full_text[:max_chars] + "..."
        line = (
            f"  [{i}] prompt: {prompt!r}\n"
            f"      target: {target!r}\n"
            f"      generated ({len(full_text)} chars full"
        )
        if objective == "reward" and reward_continuation_only:
            line += f", {len(scored)} chars continuation"
        line += f"): {shown!r}"
        if objective == "reward":
            line += f"\n      compute_reward: {r:.4f}"
        print(line, flush=True)
    print("============================================================\n", flush=True)


def log_reward_generation_sample(
    model,
    tokenizer,
    batch,
    device,
    max_new_tokens,
    do_sample,
    max_chars,
    epoch_idx,
    step_in_epoch,
    global_step,
    reward_continuation_only: bool = True,
):
    """
    Debug: after a training step, run one greedy/sampled generate (no DANP hooks)
    on the first example in the batch and print target, text, reward, lengths.
    """
    prompt, target = batch[0]
    model.eval()
    r, full_text, scored = generate_reward(
        model, tokenizer, prompt, target, device,
        max_new_tokens, do_sample, reward_continuation_only,
    )
    shown = full_text if len(full_text) <= max_chars else full_text[:max_chars] + "..."
    len_note = f"len(full)={len(full_text)}"
    if reward_continuation_only:
        len_note += f" len(continuation)={len(scored)}"
    print(
        f"\n[gen_log] epoch={epoch_idx + 1} batch_step={step_in_epoch} global_step={global_step}\n"
        f"  target (repr): {target!r}\n"
        f"  reward: {r:.4f}  |  {len_note}  len(target)={len(target)}\n"
        f"  generated (repr, truncated): {shown!r}\n",
        flush=True,
    )


# ===========================================================================
# Main training loop
# ===========================================================================
def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # Always the same two (prompt, target) pairs; no shuffle — same order every epoch/step.
    train_data = list(WP_DUMMY_EXAMPLES)
    eval_data = list(WP_DUMMY_EXAMPLES)
    reward_continuation_only = not args.reward_full_string
    print(f"Train: {len(train_data)} fixed WP examples (eval uses the same two).")

    # Load model
    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    dtype = dtype_map[args.precision]
    print(f"Loading {args.model_name} ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, cache_dir=args.hf_cache_dir,
        torch_dtype=dtype, low_cpu_mem_usage=True,
    )
    try:
        tok = AutoTokenizer.from_pretrained(args.model_name, cache_dir=args.hf_cache_dir)
    except (ValueError, OSError):
        tok = PreTrainedTokenizerFast.from_pretrained(args.model_name, cache_dir=args.hf_cache_dir)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    tok.padding_side = "left"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # Move R matrices to device
    hook = DANPHook(
        model, args.np_include, args.last_k,
        sigma=args.sigma, base_seed=args.seed,
    )
    for name in hook.R:
        hook.R[name] = hook.R[name].to(device)

    # Print how many layers we'll perturb
    n_targets = len(hook._targets)
    total_params = sum(mod.weight.numel() for _, mod in hook._targets)
    print(f"Perturbing {n_targets} Linear layers, {total_params:,} weight parameters")
    if args.alpha > 0:
        print(f"R decorrelation enabled: alpha={args.alpha} (R -= alpha*(cov-diag)@R per toy danp.py)")
    else:
        print("R decorrelation disabled (alpha=0)")

    # Baseline
    if args.objective == "ce":
        baseline = eval_ce(model, tok, eval_data, device, args.max_length, args.batch_size)
        print(f"[BASELINE] Eval CE loss: {baseline:.4f}")
    else:
        baseline = eval_reward(
            model, tok, eval_data, device, args.max_new_tokens, args.reward_do_sample,
            reward_continuation_only=reward_continuation_only,
        )
        rmode = (
            "continuation vs target length"
            if reward_continuation_only
            else "full decode vs target length (legacy)"
        )
        print(f"[BASELINE] Eval mean reward ({rmode}): {baseline:.4f}")

    train_metric_history = []
    eval_metric_history  = []
    global_step = 0
    log_reward = args.objective == "reward" and (
        args.log_generations_every > 0 or args.log_first_batch_each_epoch
    )

    for epoch in range(args.epochs):
        model.eval()   # keep BN/dropout off; DANP doesn't use gradients
        indices = np.arange(len(train_data))
        epoch_L, epoch_dL = [], []

        pbar = tqdm(range(0, len(train_data), args.batch_size),
                    desc=f"Epoch {epoch+1}/{args.epochs}")
        for step, start in enumerate(pbar):
            batch_idx = indices[start:start + args.batch_size]
            batch = [train_data[i] for i in batch_idx]
            base_seed = args.seed + epoch * 100_000 + step * 1000

            L, dL = danp_batch_step(
                model, tok, batch, device, hook,
                eta=args.eta, max_length=args.max_length,
                objective=args.objective,
                max_new_tokens=args.max_new_tokens,
                reward_do_sample=args.reward_do_sample,
                reward_continuation_only=reward_continuation_only,
                n_population=args.n_population,
                max_scale=args.max_scale,
                max_update=args.max_update,
                alpha=args.alpha,
                base_seed=base_seed,
                verbose=(args.verbose and step == 0 and epoch == 0),
            )
            epoch_L.append(L)
            epoch_dL.append(dL)
            pbar.set_postfix({"L": f"{L:.4f}", "dL": f"{dL:.4e}"})

            if log_reward:
                should_log = False
                if args.log_first_batch_each_epoch and step == 0:
                    should_log = True
                if args.log_generations_every > 0 and global_step % args.log_generations_every == 0:
                    should_log = True
                if should_log:
                    log_reward_generation_sample(
                        model,
                        tok,
                        batch,
                        device,
                        args.max_new_tokens,
                        args.reward_do_sample,
                        args.log_generations_max_chars,
                        epoch,
                        step,
                        global_step,
                        reward_continuation_only=reward_continuation_only,
                    )
            global_step += 1

            if device.type == "cuda":
                torch.cuda.empty_cache()

        mean_L  = float(np.mean(epoch_L))
        mean_dL = float(np.mean(epoch_dL))
        train_metric_history.append(mean_L)

        if args.print_generation_each_epoch:
            print_generations_after_epoch(
                hook,
                model,
                tok,
                train_data,
                device,
                args.max_new_tokens,
                args.reward_do_sample,
                args.log_generations_max_chars,
                epoch,
                args.objective,
                reward_continuation_only=reward_continuation_only,
            )

        if (epoch + 1) % args.eval_interval == 0:
            if args.objective == "ce":
                ev = eval_ce(model, tok, eval_data, device, args.max_length, args.batch_size)
            else:
                ev = eval_reward(
                    model, tok, eval_data, device,
                    args.max_new_tokens, args.reward_do_sample,
                    reward_continuation_only=reward_continuation_only,
                )
            eval_metric_history.append(ev)
            tag = "CE" if args.objective == "ce" else "reward"
            print(f"[Epoch {epoch+1}] Train metric: {mean_L:.4f} | "
                  f"mean δL: {mean_dL:.4e} | Eval {tag}: {ev:.4f}")
            if device.type == "cuda":
                print(f"  GPU: {torch.cuda.memory_allocated()/1024**2:.1f}MB alloc, "
                      f"{torch.cuda.max_memory_allocated()/1024**2:.1f}MB peak")

    # Save
    save_dir = os.path.join(args.output_dir, "final_model")
    print(f"Saving to {save_dir} ...")
    model.save_pretrained(save_dir)
    tok.save_pretrained(save_dir)
    print("Done.")


if __name__ == "__main__":
    main()
