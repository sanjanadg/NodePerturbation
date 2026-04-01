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

USAGE:
  python danp_llm_v1.py --objective ce --n_train 8 --n_eval 4 --epochs 2 --batch_size 2 --verbose
  python danp_llm_v1.py --objective reward --n_train 8 --n_eval 4 --epochs 2 --batch_size 2 --verbose
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
DEFAULT_ALPHA = 1e-4        # decorrelation rate (frozen in v1, used in v2+)
DEFAULT_MAX_NEW_TOKENS = 100
DEFAULT_POPULATION = 1      # increase for lower-variance gradient estimates

# For reward objective, activations are captured via a TEACHER-FORCING pass
# on (prompt + generated_text), NOT during generate() itself.
# generate() is called hook-free to get the text/reward; then we do one
# teacher-forcing forward with the hook to get x_star and a_clean/a_noisy.


# ===========================================================================
# Argument parsing
# ===========================================================================
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", default=DEFAULT_MODEL)
    p.add_argument("--hf_cache_dir", default="huggingface_cache")
    p.add_argument("--output_dir", default="./out_danp_v1")
    p.add_argument("--n_train", type=int, default=64)
    p.add_argument("--n_eval",  type=int, default=32)
    p.add_argument("--epochs",  type=int, default=5)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--max_length",    type=int, default=512,
                   help="Sequence length cap for teacher-forcing (CE objective)")
    p.add_argument("--max_new_tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS,
                   help="Generation cap (reward objective)")
    p.add_argument("--eta",   type=float, default=DEFAULT_ETA)
    p.add_argument("--sigma", type=float, default=DEFAULT_SIGMA)
    p.add_argument("--alpha", type=float, default=DEFAULT_ALPHA,
                   help="Decorrelation rate (unused in v1, reserved for v2)")
    p.add_argument("--np_include", default="mlp",
                   choices=["head", "attn", "mlp", "all"],
                   help="Which Linear layers to perturb")
    p.add_argument("--last_k", type=int, default=0,
                   help="If >0, only perturb the last k transformer blocks")
    p.add_argument("--n_population", type=int, default=DEFAULT_POPULATION,
                   help="Noisy-forward samples to average gradient over")
    p.add_argument("--max_scale",  type=float, default=1e2,
                   help="Hard clip on |eta*N*delta_L/norm_sq| to prevent explosion")
    p.add_argument("--max_update", type=float, default=1e-3,
                   help="Hard clip on per-element weight update. "
                        "Pretrained weights are O(0.01) so 1e-3 is ~10%% of a typical weight.")
    p.add_argument("--objective", default="ce", choices=["ce", "reward"])
    p.add_argument("--reward_do_sample", action="store_true")
    p.add_argument("--precision", default="bf16", choices=["fp16", "bf16", "fp32"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval_interval", type=int, default=1)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


# ===========================================================================
# Dataset helpers  (same dummy data as wp_conciseness / danp_llm_conciseness)
# ===========================================================================
_DUMMY_PAIRS = [
    ("Summarise in one sentence: The cat sat on the mat and looked around.", " A cat sat on a mat."),
    ("Summarise in one sentence: She went to the store and bought milk.", " She bought milk."),
    ("Summarise in one sentence: The sun rose over the mountains at dawn.", " The sun rose at dawn."),
    ("Summarise in one sentence: He read a book by the fireplace all evening.", " He read by the fireplace."),
]

def build_dataset(n_train, n_eval, seed=42):
    rng = np.random.default_rng(seed)
    total = n_train + n_eval
    pairs = (_DUMMY_PAIRS * ((total // len(_DUMMY_PAIRS)) + 1))[:total]
    rng.shuffle(pairs)  # type: ignore[arg-type]
    return list(pairs[:n_train]), list(pairs[n_train:n_train + n_eval])


def compute_reward(generated_text: str, target_text: str) -> float:
    """Same reward as wp_conciseness: negative absolute length difference."""
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
        Initialised to identity; frozen in v1.
        Shape: (in_features, in_features) — decorrelates the input.
        In v2 we add: R -= alpha * (cov - diag) @ R using clean x (not x_star).

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
        mod.forward = _make_hooked(name, mod, orig, R, sig, uid, self)

    # Make mode_ref point at the hook object so it's always current
    # (the list-cell trick above is fragile if attach() is called multiple times)
    def _make_hooked(name, mod, orig, R, sig, uid, hook_self):
        def hooked_forward(x):
            orig_shape = x.shape
            x2 = x.reshape(-1, orig_shape[-1]).float()
            R_dev = R.to(x2.device)
            x_star = x2 @ R_dev.T
            with torch.no_grad():
                a = orig(x_star.to(mod.weight.dtype))
                a32 = a.float()
                if hook_self._mode == "noisy":
                    g = torch.Generator(device=a32.device)
                    g.manual_seed(_seed_for(uid, hook_self.base_seed, x2.shape[0]))
                    eps = torch.randn_like(a32, generator=g) * sig
                    a_out = a32 + eps
                    hook_self._captured_noisy[name] = {
                        "x_star": x_star.detach().clone(),
                        "a_noisy": a_out.detach().clone(),
                    }
                else:
                    a_out = a32
                    hook_self._captured_clean[name] = {
                        "x_star": x_star.detach().clone(),
                        "a_clean": a32.detach().clone(),
                    }
            a_out = a_out.to(x.dtype).reshape(*orig_shape[:-1], a_out.shape[-1])
            return a_out
        return hooked_forward


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
def generate_reward(model, tokenizer, prompt, target, device,
                    max_new_tokens, do_sample):
    inp  = tokenizer(prompt, return_tensors="pt", padding=True,
                     padding_side="left")
    ids  = inp["input_ids"].to(device)
    amsk = inp["attention_mask"].to(device)
    out  = model.generate(
        input_ids=ids, attention_mask=amsk,
        max_new_tokens=max_new_tokens, do_sample=do_sample,
        pad_token_id=tokenizer.pad_token_id,
    )
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    return compute_reward(text, target), text


# ===========================================================================
# Core DANP gradient estimate — single sample
# ===========================================================================
def _teacher_force_ce(model, tokenizer, prompt, generated_text, device, max_length):
    """
    Compute CE loss treating `generated_text` as the target continuation.
    Used for reward objective to get a scalar loss from a generation result
    via a single coherent forward pass (so hook activations are meaningful).
    """
    example = (prompt, " " + generated_text.replace(prompt, "").strip())
    ids, mask, labs = prepare_batch([example], tokenizer, device, max_length)
    out = model(input_ids=ids, attention_mask=mask)
    return F.cross_entropy(
        out.logits.view(-1, model.config.vocab_size),
        labs.view(-1), ignore_index=-100,
    ).item()


def danp_grad_single(
    model, tokenizer, example, device, hook,
    eta, max_length,
    objective,
    max_new_tokens, reward_do_sample,
    n_population,
    max_scale, max_update,
    verbose=False,
):
    """
    Compute weight update dict for a single (prompt, target) example.

    CE objective:
        Clean and noisy forwards are teacher-forcing passes on (prompt + target).
        delta_L = CE_noisy - CE_clean.

    Reward objective:
        Step 1: generate() WITHOUT hook to get the clean generated text + reward.
        Step 2: teacher-forcing pass WITH hook (clean) on (prompt + generated_text)
                to capture activations at the weights that produced that generation.
        Step 3: same teacher-forcing pass WITH hook (noisy) to get noisy activations
                and a perturbed CE loss on the same sequence.
        delta_L = CE_noisy_tf - CE_clean_tf  (both teacher-forcing on clean generation)
        L_clean reported = -R_clean (for logging; actual update uses delta_L from TF passes)

    WHY NOT hook during generate():
        generate() calls forward() once per output token. Captured activations would
        be from the LAST token step only, not a coherent single forward pass.
        The teacher-forcing trick gives us one pass over the full sequence,
        making x_star and a_clean/a_noisy consistent with each other.

    Returns:
        weight_updates: dict  name+".weight" -> update tensor (same dtype as param)
        L_clean:        float  clean loss / reward (for logging)
        delta_L_mean:   float  mean(CE_noisy_tf - CE_clean_tf) over population
    """
    prompt, target = example

    # ------------------------------------------------------------------ #
    # STEP 1: get scalar loss and (for reward) the generated text          #
    # ------------------------------------------------------------------ #
    if objective == "reward":
        # Generate without hook — keeps generation clean and fast
        R_clean, gen_text = generate_reward(
            model, tokenizer, prompt, target,
            device, max_new_tokens, reward_do_sample,
        )
        L_clean = -float(R_clean)
        # Teacher-forcing example for hook passes: (prompt, generated_continuation)
        gen_continuation = gen_text[len(prompt):].strip()
        tf_example = (prompt, " " + gen_continuation) if gen_continuation else (prompt, target)
    else:
        tf_example = example
        L_clean = None   # will be filled during clean hook pass

    # ------------------------------------------------------------------ #
    # STEP 2: Clean hook pass (teacher-forcing)                            #
    # ------------------------------------------------------------------ #
    hook.attach("clean")
    with torch.no_grad():
        ids, mask, labs = prepare_batch([tf_example], tokenizer, device, max_length)
        out = model(input_ids=ids, attention_mask=mask)
        L_clean_tf = F.cross_entropy(
            out.logits.view(-1, model.config.vocab_size),
            labs.view(-1), ignore_index=-100,
        ).item()
    if objective == "ce":
        L_clean = L_clean_tf
    captured_clean = {k: {kk: v.clone() for kk, v in vv.items()}
                      for k, vv in hook._captured_clean.items()}
    hook.detach()

    # ------------------------------------------------------------------ #
    # STEP 3: Noisy forward(s) — accumulate gradient estimates             #
    # ------------------------------------------------------------------ #
    accumulated = {}
    delta_L_sum = 0.0

    for pop_i in range(n_population):
        hook.base_seed = hook.base_seed + pop_i * 31337
        hook.attach("noisy")
        with torch.no_grad():
            ids, mask, labs = prepare_batch([tf_example], tokenizer, device, max_length)
            out = model(input_ids=ids, attention_mask=mask)
            L_noisy_tf = F.cross_entropy(
                out.logits.view(-1, model.config.vocab_size),
                labs.view(-1), ignore_index=-100,
            ).item()
        captured_noisy = {k: {kk: v.clone() for kk, v in vv.items()}
                          for k, vv in hook._captured_noisy.items()}
        hook.detach()

        if not (np.isfinite(L_clean_tf) and np.isfinite(L_noisy_tf)):
            continue

        # delta_L is always computed on teacher-forcing CE, regardless of objective.
        # For CE: this is the natural loss diff.
        # For reward: this is a proxy — CE on the same generated sequence,
        #   perturbed vs clean. The reward itself is reported in L_clean for logging.
        delta_L = L_noisy_tf - L_clean_tf
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
            print(f"\n[DANP grad] L_clean_tf={L_clean_tf:.4f} L_noisy_tf={L_noisy_tf:.4f} "
                  f"delta_L={delta_L:.6f} N={N} norm_sq={norm_sq:.4e} "
                  f"scale_raw={scale_raw:.4e} scale={scale:.4e} "
                  f"(L_clean reported={L_clean:.4f})")

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
            update = update.float()

            # Absolute clip (prevents catastrophic updates on pretrained weights)
            update = torch.clamp(update, -max_update, max_update)

            # Sanity: print update vs weight magnitude on first pop of first layer
            if verbose and pop_i == 0 and name == hook._targets[0][0]:
                w_mag = mod.weight.data.float().abs().mean().item()
                u_mag = update.abs().mean().item()
                print(f"  [{name}] weight_mean_abs={w_mag:.4e} update_mean_abs={u_mag:.4e} "
                      f"ratio={u_mag/max(w_mag,1e-9):.4e}")

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
    return weight_updates, float(L_clean), float(delta_L_mean)


# ===========================================================================
# Batch step — accumulate then apply (like train_danp_batch in synthetic_data.py)
# ===========================================================================
def danp_batch_step(model, tokenizer, batch, device, hook,
                    eta, max_length, objective,
                    max_new_tokens, reward_do_sample,
                    n_population, max_scale, max_update,
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

    for i, example in enumerate(batch):
        hook.base_seed = base_seed + i * 997   # different seed per example

        updates, L_clean, dL = danp_grad_single(
            model, tokenizer, example, device, hook,
            eta=eta, max_length=max_length,
            objective=objective,
            max_new_tokens=max_new_tokens,
            reward_do_sample=reward_do_sample,
            n_population=n_population,
            max_scale=max_scale, max_update=max_update,
            verbose=(verbose and i == 0),
        )
        total_L  += L_clean
        total_dL += dL

        for key, upd in updates.items():
            if key not in accumulated:
                accumulated[key] = torch.zeros_like(upd)
            accumulated[key] += upd

    # Apply averaged updates
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
def eval_reward(model, tokenizer, data, device, max_new_tokens, do_sample):
    rewards = []
    for prompt, target in data:
        r, _ = generate_reward(model, tokenizer, prompt, target,
                                device, max_new_tokens, do_sample)
        rewards.append(r)
    return float(np.mean(rewards))


# ===========================================================================
# Main training loop
# ===========================================================================
def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    train_data, eval_data = build_dataset(args.n_train, args.n_eval, args.seed)
    print(f"Train: {len(train_data)}, Eval: {len(eval_data)}")

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

    # Baseline
    if args.objective == "ce":
        baseline = eval_ce(model, tok, eval_data, device, args.max_length, args.batch_size)
        print(f"[BASELINE] Eval CE loss: {baseline:.4f}")
    else:
        baseline = eval_reward(model, tok, eval_data, device, args.max_new_tokens, args.reward_do_sample)
        print(f"[BASELINE] Eval mean reward: {baseline:.4f}")

    rng = np.random.default_rng(args.seed)
    train_metric_history = []
    eval_metric_history  = []

    for epoch in range(args.epochs):
        model.eval()   # keep BN/dropout off; DANP doesn't use gradients
        indices = rng.permutation(len(train_data))
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
                n_population=args.n_population,
                max_scale=args.max_scale,
                max_update=args.max_update,
                base_seed=base_seed,
                verbose=(args.verbose and step == 0 and epoch == 0),
            )
            epoch_L.append(L)
            epoch_dL.append(dL)
            pbar.set_postfix({"L": f"{L:.4f}", "dL": f"{dL:.4e}"})

            if device.type == "cuda":
                torch.cuda.empty_cache()

        mean_L  = float(np.mean(epoch_L))
        mean_dL = float(np.mean(epoch_dL))
        train_metric_history.append(mean_L)

        if (epoch + 1) % args.eval_interval == 0:
            if args.objective == "ce":
                ev = eval_ce(model, tok, eval_data, device, args.max_length, args.batch_size)
            else:
                ev = eval_reward(model, tok, eval_data, device,
                                 args.max_new_tokens, args.reward_do_sample)
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
