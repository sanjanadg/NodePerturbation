#!/usr/bin/env python3
"""
DANP for LLMs — v4.

Same logic as ``danp_llm_v3`` (cumulative noisy forward, clean x_star in the
outer product, reward/CE paths) but **no clamping**, **no numerical guards**
(no eps on ``norm_sq``, no finite/NaN checks, no ``max(1, ·)`` divisors), and
**no explicit validation** (no ``assert``/``raise`` for captures or attach mode).

KEY FIXES vs v1 (inherited from v3):
  1. Cumulative noisy forward pass: noise at layer l propagates into layer l+1's
     input, exactly matching Algorithm 1's noisy forward pass loop.
     In v1 every layer saw the clean input from the previous layer — this meant
     delta_a = eps exactly, making perturbations input-blind.

  2. Weight update uses **clean** x*_{l-1} in the outer product with (ã_l - a_l):
         W_l ← W_l - η N δL * (ã_l - a_l) (x*_{l-1})^T / ||δa||²
     where x*_{l-1} is from the clean forward (same as toy danp / v1).
     The noisy forward still supplies ã_l and drives cumulative perturbations.

  3. base_seed mutation bug fixed: v1 permanently mutated hook.base_seed
     inside the population loop causing seeds to drift across batches/epochs.

  4. Reward sign convention cleaned up and documented clearly.

  5. delta_a = ã_l - a_l is input-dependent: the noisy pass feeds cumulative
     x̃ into deeper layers, so ã_l reflects upstream noise, not only ε_l.

WHAT IS NOT IN THIS VERSION:
  - Multi-GPU / accelerate (single GPU focus).

USAGE:
python danp_llm_v4.py \
  --objective reward \
  --sigma 0.01 \
  --eta 0.001 \
  --alpha 0 \
  --n_population 5 \
  --epochs 100 \
  --batch_size 2 \
  --np_include all \
  --last_k 0 \
  --print_generation_each_epoch \
  --verbose

python danp_llm_v4.py \
  --objective reward \
  --sigma 0.01 \
  --eta 0.0005 \
  --alpha 0 \
  --n_population 5 \
  --epochs 100 \
  --batch_size 2 \
  --np_include all \
  --last_k 0 \
  --print_generation_each_epoch \
  --verbose

match toy
  python danp_llm_v4.py \
  --objective reward \
  --sigma 0.02 \
  --eta 0.01 \
  --alpha 0 \
  --n_population 5 \
  --epochs 100 \
  --batch_size 2 \
  --np_include all \
  --last_k 0 \
  --print_generation_each_epoch \
  --verbose
"""

import os, re, hashlib, argparse
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging as hf_logging

hf_logging.set_verbosity_error()
torch.backends.cuda.matmul.allow_tf32 = True

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_MODEL      = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_ETA        = 1e-3
DEFAULT_SIGMA      = 1e-3
DEFAULT_ALPHA      = 1e-4
DEFAULT_MAX_NEW_TOKENS = 100
DEFAULT_POPULATION = 1


# ===========================================================================
# Argument parsing
# ===========================================================================
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name",    default=DEFAULT_MODEL)
    p.add_argument("--hf_cache_dir",  default="huggingface_cache")
    p.add_argument("--output_dir",    default="./out_danp_v4")
    p.add_argument("--epochs",        type=int,   default=5)
    p.add_argument("--batch_size",    type=int,   default=2)
    p.add_argument("--max_length",    type=int,   default=512)
    p.add_argument("--max_new_tokens",type=int,   default=DEFAULT_MAX_NEW_TOKENS)
    p.add_argument("--eta",           type=float, default=DEFAULT_ETA)
    p.add_argument("--sigma",         type=float, default=DEFAULT_SIGMA)
    p.add_argument("--alpha",         type=float, default=DEFAULT_ALPHA,
                   help="Decorrelation rate for R (0 disables)")
    p.add_argument("--np_include",    default="mlp",
                   choices=["head", "attn", "mlp", "all"])
    p.add_argument("--last_k",        type=int,   default=0)
    p.add_argument("--n_population",  type=int,   default=DEFAULT_POPULATION)
    p.add_argument("--objective",     default="ce", choices=["ce", "reward"])
    p.add_argument("--reward_do_sample", action="store_true")
    p.add_argument("--precision",     default="bf16", choices=["fp16","bf16","fp32"])
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--eval_interval", type=int,   default=1)
    p.add_argument("--verbose",       action="store_true")
    p.add_argument("--log_generations_every",   type=int, default=0)
    p.add_argument("--log_first_batch_each_epoch", action="store_true")
    p.add_argument("--log_generations_max_chars",  type=int, default=600)
    p.add_argument("--print_generation_each_epoch", action="store_true")
    p.add_argument(
        "--reward_full_string", action="store_true",
        help="Log full decode as 'scored' segment; reward always uses full output length vs target.",
    )
    return p.parse_args()


# ===========================================================================
# Dataset
# ===========================================================================
WP_DUMMY_EXAMPLES = [
    ("Solve: 3 + 5 =", "8"),
    ("If all birds can fly and penguins are birds, can penguins fly?", "No"),
]

def compute_reward(generated_text: str, target_text: str) -> float:
    return -abs(len(generated_text) - len(target_text))


# ===========================================================================
# Layer selection helpers  (unchanged from v1)
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
                   ("attn","attention","q_proj","k_proj","v_proj",
                    "o_proj","in_proj","c_attn","c_proj")) and "mlp" not in lname
    if include == "mlp":
        return any(k in lname for k in
                   ("mlp","ffn","gate_proj","up_proj","down_proj",
                    "fc1","fc2","c_fc")) or (
                   "c_proj" in lname and "mlp" in lname)
    return True  # "all"

def get_target_linears(model, include, last_k):
    total_blocks = _infer_total_blocks(model)
    return [(name, mod) for name, mod in model.named_modules()
            if _is_target_linear(name, mod, include, last_k, total_blocks)]

def _stable_uid(name: str) -> int:
    h = hashlib.blake2s(name.encode(), digest_size=4).digest()
    return int.from_bytes(h, "little") & 0x7FFFFFFF

def _seed_for(uid: int, base: int, n_tokens: int) -> int:
    return (base ^ uid ^ (n_tokens * 2654435761)) & 0x7FFFFFFF


# ===========================================================================
# DANP Hook — v4: same cumulative noisy forward as v3
# ===========================================================================
class DANPHook:
    """
    Cumulative noisy forward (matches the noisy pass in Algorithm 1).
    The ES weight update uses clean x* in the outer product; see ``danp_grad_single``.

    CLEAN forward pass (attach mode="clean"):
        x*_{l-1} = R_{l-1} @ x_{l-1}          # decorrelate input
        a_l      = W_l @ x*_{l-1}              # standard linear
        x_l      = f(a_l)                      # activation fn (handled by model)

    NOISY forward pass (attach mode="noisy"):
        x̃*_{l-1} = R_{l-1} @ x̃_{l-1}          # decorrelate NOISY input from prev layer
        ã_l      = W_l @ x̃*_{l-1} + ε_l        # linear on noisy input, PLUS fresh noise
        x̃_l      = f(ã_l)                      # feeds into next layer's x̃_{l-1}

    KEY DIFFERENCE FROM v1:
        In v1, the noisy forward pass used clean x from the previous layer.
        So delta_a = a_noisy - a_clean = eps exactly for every layer.
        In v2, the noisy output x̃_l = f(ã_l) is what actually gets passed to
        layer l+1 — because we return a_noisy from the hook, the model's own
        residual stream and activation functions propagate it forward.
        This means by layer l, x̃*_{l-1} has accumulated perturbations from
        ALL upstream layers, making delta_a truly input-dependent.

    Captured per layer:
        clean:  x_star  = R @ x_clean      (input to W, clean pass)
                a_clean = W @ x_star       — used with delta_a and **clean** x_star in grad
        noisy:  x_star_noisy = R @ x̃       (input to W, noisy pass — x̃ is cumulative)
                a_noisy      = W @ x_star_noisy + eps
    """

    def __init__(self, model, include, last_k, sigma, base_seed):
        self.sigma      = float(sigma)
        self.base_seed  = int(base_seed)
        self._targets   = get_target_linears(model, include, last_k)
        self._orig_fwd  = {}
        self.R = {
            name: torch.eye(mod.in_features, dtype=torch.float32)
            for name, mod in self._targets
        }
        self._captured_clean: dict = {}
        self._captured_noisy: dict = {}
        self._mode = "clean"

    def attach(self, mode: str):
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

    def _wrap(self, name: str, mod: torch.nn.Linear):
        orig           = mod.forward
        self._orig_fwd[name] = orig
        R              = self.R[name]
        sig            = self.sigma
        uid            = _stable_uid(name)
        mode_ref       = [self._mode]
        captured_clean = self._captured_clean
        captured_noisy = self._captured_noisy
        base_seed_ref  = [self.base_seed]   # FIX: snapshot seed at attach time

        def hooked_forward(x):
            orig_shape = x.shape
            x2 = x.reshape(-1, orig_shape[-1]).float()

            # x_star = R @ x  — decorrelate input (clean OR noisy, whatever came in)
            R_dev  = R.to(x2.device)
            x_star = x2 @ R_dev.T          # (N, in_features)

            with torch.no_grad():
                # Standard linear on decorrelated input
                a    = orig(x_star.to(mod.weight.dtype))   # (N, out_features)
                a32  = a.float()

                if mode_ref[0] == "noisy":
                    # FIX: use snapshotted seed, not self.base_seed which may have changed
                    g = torch.Generator(device=a32.device)
                    g.manual_seed(_seed_for(uid, base_seed_ref[0], x2.shape[0]))
                    eps   = torch.randn(a32.shape, device=a32.device,
                                        dtype=a32.dtype, generator=g) * sig
                    a_out = a32 + eps

                    # x̃*_{l-1} for debugging; grad uses clean x*_{l-1} from captured_clean.
                    captured_noisy[name] = {
                        "x_star_noisy": x_star.detach().clone(),   # R @ x̃_{l-1}
                        "a_noisy":      a_out.detach().clone(),     # W @ x̃*_{l-1} + eps_l # note: eps_l is fresh noise at each laye 
                    }
                else:
                    a_out = a32
                    captured_clean[name] = {
                        "x_star": x_star.detach().clone(),   # R @ x_{l-1}
                        "a_clean": a32.detach().clone(),     # W @ x*_{l-1}
                    }

            # Return noisy output — this is what propagates to the next layer.
            # For the noisy pass this means x̃_l = f(a_noisy) flows forward,
            # so layer l+1 receives cumulative noise, matching Algorithm 1.
            a_out = a_out.to(x.dtype).reshape(*orig_shape[:-1], a_out.shape[-1])
            return a_out

        mod.forward = hooked_forward

    def update_base_seed(self, new_seed: int):
        """Call this before attach() for a new population sample."""
        self.base_seed = new_seed
        # Update all closure snapshots
        for name, mod in self._targets:
            if name in self._orig_fwd:
                # Re-wrap with new seed snapshot — easier to just detach and re-attach
                pass  # handled by detach+attach cycle in danp_grad_single


# ===========================================================================
# Decorrelation  (unchanged from v1)
# ===========================================================================
def decorrelation_delta_r(x_star, R, alpha):
    x   = x_star.float().reshape(-1, x_star.shape[-1])
    n, d = x.shape
    Rf  = R.float()
    cov = (x.T @ x) / n if n > 1 else x.T @ x
    diag = torch.diag((x ** 2).mean(dim=0))
    return alpha * (cov - diag) @ Rf


# ===========================================================================
# Batch preparation
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
    ids  = torch.full((B, L), pad,          dtype=torch.long,  device=device)
    mask = torch.zeros((B, L),              dtype=torch.long,  device=device)
    labs = torch.full((B, L), label_ignore, dtype=torch.long,  device=device)
    for i in range(B):
        n = len(input_ids_list[i])
        ids [i, :n] = torch.tensor(input_ids_list[i], device=device)
        mask[i, :n] = 1
        labs[i, :n] = torch.tensor(labels_list[i],    device=device)
    return ids, mask, labs


# ===========================================================================
# CE loss
# ===========================================================================
def ce_loss(model, ids, mask, labs):
    out = model(input_ids=ids, attention_mask=mask)
    return F.cross_entropy(
        out.logits.view(-1, model.config.vocab_size),
        labs.view(-1), ignore_index=-100,
    )


# ===========================================================================
# Reward generation
# ===========================================================================
@torch.no_grad()
def generate_reward(model, tokenizer, prompt, target, device,
                    max_new_tokens, do_sample, reward_continuation_only=False):
    inp  = tokenizer(prompt, return_tensors="pt", padding=True, padding_side="left")
    ids  = inp["input_ids"].to(device)
    amsk = inp["attention_mask"].to(device)
    out  = model.generate(input_ids=ids, attention_mask=amsk,
                          max_new_tokens=max_new_tokens, do_sample=do_sample,
                          pad_token_id=tokenizer.pad_token_id)
    prompt_len  = ids.shape[1]
    full_text   = tokenizer.decode(out[0], skip_special_tokens=True)
    if reward_continuation_only:
        cont_ids    = out[0][prompt_len:]
        scored_text = tokenizer.decode(cont_ids, skip_special_tokens=True)
    else:
        scored_text = full_text
    # Reward compares target length to full model output, not continuation-only length.
    r = compute_reward(full_text, target)
    return r, full_text, scored_text


# ===========================================================================
# Core DANP gradient estimate — single example
# ===========================================================================
def danp_grad_single(
    model, tokenizer, example, device, hook,
    eta, max_length, objective,
    max_new_tokens, reward_do_sample, reward_continuation_only,
    n_population, alpha,
    base_seed, verbose=False,
):
    """
    Compute weight update for one (prompt, target) example.

    Weight update (outer product uses **clean** decorrelated input):
        W_l ← W_l - η N δL * (ã_l - a_l) (x*_{l-1})^T / ||δa||²
    where x*_{l-1} is from the clean forward; ã_l, a_l come from noisy vs clean passes.

    Reward sign convention:
        compute_reward → higher is better.
        L = -reward  (lower is better, matches CE convention).
        delta_L = L_noisy - L_clean = -R_noisy + R_clean = R_clean - R_noisy
        If noise HELPED  (R_noisy > R_clean): delta_L < 0 → scale < 0 → W -= negative → W increases in noise direction ✓
        If noise HURT    (R_noisy < R_clean): delta_L > 0 → scale > 0 → W -= positive → W moves away from noise direction ✓
    """
    prompt, target = example

    # ------------------------------------------------------------------ #
    # 1. Clean forward pass                                                #
    # ------------------------------------------------------------------ #
    hook.base_seed = base_seed
    hook.attach("clean")
    with torch.no_grad():
        if objective == "ce":
            ids, mask, labs = prepare_batch([example], tokenizer, device, max_length)
            out     = model(input_ids=ids, attention_mask=mask)
            L_clean = F.cross_entropy(
                out.logits.view(-1, model.config.vocab_size),
                labs.view(-1), ignore_index=-100,
            ).item()
        else:
            R_clean, _, _ = generate_reward(
                model, tokenizer, prompt, target, device,
                max_new_tokens, reward_do_sample, reward_continuation_only,
            )
            L_clean = -float(R_clean)
    captured_clean = {k: {kk: v.clone() for kk, v in vv.items()}
                      for k, vv in hook._captured_clean.items()}
    hook.detach()

    # ------------------------------------------------------------------ #
    # 2. Noisy forward passes (population)                                 #
    # ------------------------------------------------------------------ #
    accumulated  = {}
    delta_L_sum  = 0.0
    scale_values = []

    for pop_i in range(n_population):
        pop_seed = base_seed + pop_i * 31337
        hook.base_seed = pop_seed
        hook.attach("noisy")
        with torch.no_grad():
            if objective == "ce":
                ids, mask, labs = prepare_batch([example], tokenizer, device, max_length)
                out     = model(input_ids=ids, attention_mask=mask)
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

        delta_L      = L_noisy - L_clean
        delta_L_sum += delta_L

        # Global norm_sq across ALL target layers (Algorithm 1: ||δa||²)
        delta_a_parts = []
        for name, _ in hook._targets:
            a_c = captured_clean[name]["a_clean"]
            a_n = captured_noisy[name]["a_noisy"]
            delta_a_parts.append((a_n - a_c).flatten())

        delta_a_cat = torch.cat(delta_a_parts)
        norm_sq     = float((delta_a_cat ** 2).sum().item())
        N           = delta_a_cat.numel()
        scale       = eta * N/2 * float(delta_L) / norm_sq
        scale_values.append(scale)

        if verbose and pop_i == 0:
            print(f"\n[DANP v4 grad] L_clean={L_clean:.4f} L_noisy={L_noisy:.4f} "
                  f"delta_L={delta_L:.6f} N={N} norm_sq={norm_sq:.4e} scale={scale:.4e}")

        # Per-layer weight update: grad ∝ (ã_l - a_l) (x*_{l-1})^T with x* from clean pass
        for name, mod in hook._targets:
            a_c     = captured_clean[name]["a_clean"]
            a_n     = captured_noisy[name]["a_noisy"]
            delta_a = (a_n - a_c)
            x_star  = captured_clean[name]["x_star"]

            if x_star.dim() > 2:
                x_star  = x_star.reshape(-1, x_star.shape[-1])
                delta_a = delta_a.reshape(-1, delta_a.shape[-1])

            grad   = delta_a.T @ x_star
            update = scale * grad.float()

            key = name + ".weight"
            if key not in accumulated:
                accumulated[key] = torch.zeros_like(update)
            accumulated[key] += update

    # Average over population
    weight_updates = {}
    for key, acc in accumulated.items():
        avg      = acc / n_population
        mod_path = key[:-7]
        mod      = model.get_submodule(mod_path)
        weight_updates[key] = avg.to(mod.weight.dtype)

    delta_L_mean = delta_L_sum / n_population

    # Decorrelation update (from clean x_star, matching toy danp.py)
    decorrelation_updates = {}
    for name in captured_clean:
        x_star = captured_clean[name]["x_star"]
        if x_star.dim() > 2:
            x_star = x_star.reshape(-1, x_star.shape[-1])
        decorrelation_updates[name] = decorrelation_delta_r(
            x_star, hook.R[name], alpha
        )

    scale_mean = float(np.mean(scale_values))
    upd_frob_sq = sum(
        float(torch.sum(u.float() ** 2).item()) for u in weight_updates.values()
    )
    update_norm = float(np.sqrt(upd_frob_sq))

    return (
        weight_updates,
        decorrelation_updates,
        float(L_clean),
        float(delta_L_mean),
        scale_mean,
        update_norm,
    )


# ===========================================================================
# Batch step
# ===========================================================================
def danp_batch_step(model, tokenizer, batch, device, hook,
                    eta, max_length, objective,
                    max_new_tokens, reward_do_sample, reward_continuation_only,
                    n_population, alpha,
                    base_seed, verbose=False):
    total_L, total_dL       = 0.0, 0.0
    total_scale, total_updn = 0.0, 0.0
    accumulated             = {}
    accumulated_dec         = {}

    for i, example in enumerate(batch):
        example_seed = base_seed + i * 997

        updates, dec_updates, L_clean, dL, sc_m, up_n = danp_grad_single(
            model, tokenizer, example, device, hook,
            eta=eta, max_length=max_length, objective=objective,
            max_new_tokens=max_new_tokens, reward_do_sample=reward_do_sample,
            reward_continuation_only=reward_continuation_only,
            n_population=n_population,
            alpha=alpha, base_seed=example_seed,
            verbose=(verbose and i == 0),
        )
        total_L     += L_clean
        total_dL    += dL
        total_scale += sc_m
        total_updn  += up_n

        for key, upd in updates.items():
            if key not in accumulated:
                accumulated[key] = torch.zeros_like(upd)
            accumulated[key] += upd

        for name, dR in dec_updates.items():
            if name not in accumulated_dec:
                accumulated_dec[name] = torch.zeros_like(dR)
            accumulated_dec[name] += dR

    B = len(batch)
    with torch.no_grad():
        for key, acc in accumulated.items():
            avg = acc / B
            mod_path = key[:-7]
            mod      = model.get_submodule(mod_path)
            mod.weight.sub_(avg.to(mod.weight.dtype))

        for name, acc in accumulated_dec.items():
            avg = acc / B
            hook.R[name].sub_(avg.to(device=hook.R[name].device,
                                      dtype=hook.R[name].dtype))

    return total_L / B, total_dL / B, total_scale / B, total_updn / B


# ===========================================================================
# Evaluation
# ===========================================================================
@torch.no_grad()
def eval_ce(model, tokenizer, data, device, max_length, batch_size):
    losses = []
    for i in range(0, len(data), batch_size):
        b    = data[i:i + batch_size]
        ids, mask, labs = prepare_batch(b, tokenizer, device, max_length)
        out  = model(input_ids=ids, attention_mask=mask)
        l    = F.cross_entropy(
            out.logits.view(-1, model.config.vocab_size),
            labs.view(-1), ignore_index=-100,
        ).item()
        losses.append(l)
    return float(np.mean(losses))

@torch.no_grad()
def eval_reward(model, tokenizer, data, device, max_new_tokens, do_sample,
                reward_continuation_only=True):
    rewards = []
    for prompt, target in data:
        r, _, _ = generate_reward(model, tokenizer, prompt, target, device,
                                  max_new_tokens, do_sample, reward_continuation_only)
        rewards.append(r)
    return float(np.mean(rewards))


def print_generations_after_epoch(hook, model, tokenizer, pairs, device,
                                   max_new_tokens, do_sample, max_chars,
                                   epoch_idx, objective, reward_continuation_only=True):
    hook.detach()
    model.eval()
    print(f"\n========== Epoch {epoch_idx + 1} — generations (no hooks) ==========", flush=True)
    for i, (prompt, target) in enumerate(pairs):
        r, full_text, scored = generate_reward(
            model, tokenizer, prompt, target, device,
            max_new_tokens, do_sample, reward_continuation_only)
        shown = full_text if len(full_text) <= max_chars else full_text[:max_chars] + "..."
        line  = (f"  [{i}] prompt: {prompt!r}\n"
                 f"      target: {target!r}\n"
                 f"      generated ({len(full_text)} chars full")
        if objective == "reward" and reward_continuation_only:
            line += f", {len(scored)} chars continuation"
        line += f"): {shown!r}"
        if objective == "reward":
            line += f"\n      compute_reward: {r:.4f}"
        print(line, flush=True)
    print("============================================================\n", flush=True)


def log_reward_generation_sample(model, tokenizer, batch, device,
                                  max_new_tokens, do_sample, max_chars,
                                  epoch_idx, step_in_epoch, global_step,
                                  reward_continuation_only=True):
    prompt, target = batch[0]
    model.eval()
    r, full_text, scored = generate_reward(
        model, tokenizer, prompt, target, device,
        max_new_tokens, do_sample, reward_continuation_only)
    shown    = full_text if len(full_text) <= max_chars else full_text[:max_chars] + "..."
    len_note = f"len(full)={len(full_text)}"
    if reward_continuation_only:
        len_note += f" len(continuation)={len(scored)}"
    print(
        f"\n[gen_log] epoch={epoch_idx+1} batch_step={step_in_epoch} global_step={global_step}\n"
        f"  target (repr): {target!r}\n"
        f"  reward: {r:.4f}  |  {len_note}  len(target)={len(target)}\n"
        f"  generated (repr, truncated): {shown!r}\n", flush=True,
    )


# ===========================================================================
# Main
# ===========================================================================
def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    train_data = list(WP_DUMMY_EXAMPLES)
    eval_data  = list(WP_DUMMY_EXAMPLES)
    reward_continuation_only = not args.reward_full_string
    print(f"Train: {len(train_data)} fixed WP examples.")

    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    dtype     = dtype_map[args.precision]
    print(f"Loading {args.model_name} ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, cache_dir=args.hf_cache_dir,
        torch_dtype=dtype, low_cpu_mem_usage=True,
    )
    tok = AutoTokenizer.from_pretrained(args.model_name, cache_dir=args.hf_cache_dir)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    tok.padding_side = "left"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = model.to(device)
    model.eval()

    hook = DANPHook(model, args.np_include, args.last_k,
                    sigma=args.sigma, base_seed=args.seed)
    for name in hook.R:
        hook.R[name] = hook.R[name].to(device)

    n_targets    = len(hook._targets)
    total_params = sum(mod.weight.numel() for _, mod in hook._targets)
    print(f"Perturbing {n_targets} Linear layers, {total_params:,} weight parameters")
    print(f"R decorrelation: {'enabled alpha='+str(args.alpha) if args.alpha > 0 else 'disabled'}")

    if args.objective == "ce":
        baseline = eval_ce(model, tok, eval_data, device, args.max_length, args.batch_size)
        print(f"[BASELINE] Eval CE loss: {baseline:.4f}")
    else:
        baseline = eval_reward(model, tok, eval_data, device, args.max_new_tokens,
                               args.reward_do_sample, reward_continuation_only)
        print(f"[BASELINE] Eval mean reward: {baseline:.4f}")

    train_metric_history = []
    eval_metric_history  = []
    global_step          = 0
    log_reward = args.objective == "reward" and (
        args.log_generations_every > 0 or args.log_first_batch_each_epoch)

    for epoch in range(args.epochs):
        model.eval()
        indices  = np.arange(len(train_data))
        epoch_dL = []
        epoch_scale, epoch_upd_norm = [], []

        pbar = tqdm(range(0, len(train_data), args.batch_size),
                    desc=f"Epoch {epoch+1}/{args.epochs}")
        for step, start in enumerate(pbar):
            batch_idx  = indices[start:start + args.batch_size]
            batch      = [train_data[i] for i in batch_idx]
            base_seed  = args.seed + epoch * 100_000 + step * 1000

            _L, dL, sc_b, up_b = danp_batch_step(
                model, tok, batch, device, hook,
                eta=args.eta, max_length=args.max_length,
                objective=args.objective,
                max_new_tokens=args.max_new_tokens,
                reward_do_sample=args.reward_do_sample,
                reward_continuation_only=reward_continuation_only,
                n_population=args.n_population,
                alpha=args.alpha, base_seed=base_seed,
                verbose=(args.verbose and step == 0 and epoch == 0),
            )
            epoch_dL.append(dL)
            epoch_scale.append(sc_b)
            epoch_upd_norm.append(up_b)
            pbar.set_postfix({"scale": f"{sc_b:.4e}", "||upd||": f"{up_b:.4e}"})

            if log_reward:
                should_log = (args.log_first_batch_each_epoch and step == 0) or \
                             (args.log_generations_every > 0 and
                              global_step % args.log_generations_every == 0)
                if should_log:
                    log_reward_generation_sample(
                        model, tok, batch, device, args.max_new_tokens,
                        args.reward_do_sample, args.log_generations_max_chars,
                        epoch, step, global_step, reward_continuation_only)
            global_step += 1
            if device.type == "cuda":
                torch.cuda.empty_cache()

        mean_dL = float(np.mean(epoch_dL))
        mean_scale_ep = float(np.mean(epoch_scale))
        mean_upd_ep   = float(np.mean(epoch_upd_norm))
        train_metric_history.append(mean_scale_ep)

        print(f"[Epoch {epoch+1}] mean scale: {mean_scale_ep:.4e} | "
              f"mean ||update||_F (per-example applied): {mean_upd_ep:.4e} | "
              f"mean δL: {mean_dL:.4e}")

        if args.print_generation_each_epoch:
            print_generations_after_epoch(
                hook, model, tok, train_data, device,
                args.max_new_tokens, args.reward_do_sample,
                args.log_generations_max_chars, epoch, args.objective,
                reward_continuation_only)

        if (epoch + 1) % args.eval_interval == 0:
            if args.objective == "ce":
                ev  = eval_ce(model, tok, eval_data, device,
                              args.max_length, args.batch_size)
            else:
                ev  = eval_reward(model, tok, eval_data, device,
                                  args.max_new_tokens, args.reward_do_sample,
                                  reward_continuation_only)
            eval_metric_history.append(ev)
            tag = "CE" if args.objective == "ce" else "reward"
            print(f"  Eval {tag}: {ev:.4f}")
            if device.type == "cuda":
                print(f"  GPU: {torch.cuda.memory_allocated()/1024**2:.1f}MB alloc, "
                      f"{torch.cuda.max_memory_allocated()/1024**2:.1f}MB peak")

    save_dir = os.path.join(args.output_dir, "final_model")
    print(f"Saving to {save_dir} ...")
    model.save_pretrained(save_dir)
    tok.save_pretrained(save_dir)
    print("Done.")


if __name__ == "__main__":
    main()