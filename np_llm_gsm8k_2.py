#!/usr/bin/env python3

"""
Node Perturbation (NP) — STRICT PARITY with weight-perturbation script
Only the perturbation/update path differs:
- process_seed(): wraps forwards with activation-noise (Pass A) instead of weight noise.
- After rewards are collected & z-scored, we REPLAY the generated texts per candidate
  with the same seed to accumulate NP grads, then do one gradient-ascent update.

Everything else (CLI, dataset loading, baseline eval, batching, threads, DDP,
CSV logging, progress messages) matches the WP script.

Parity/speed knobs (minimal diffs):
  • --np_fast_noise (default ON): vectorized noise per layer call (deterministic; A/B match)
  • --np_micro_batch_size 0 => auto (kept for completeness; cached replay ignores it)
  • --np_sync_every (default 1): all-reduce grads every K iterations (1 = every iter, original)

accelerate launch --num_processes=8 np_llm_gsm8k.py \
  --model_name Qwen/Qwen2.5-1.5B-Instruct \
  --hf_cache_dir $HOME/hf_cache \
  --mixed_precision bf16 \
  --n_train 10 --n_eval 100 \
  --n_iters 10 --pop_size 30 \
  --sigma 0.005 --alpha 0.001 \
  --np_include mlp --last_k 1 \
  --np_fast_noise \
  --np_sync_every 1 \
  --visualization_dir ./out_np_gsm8k_parity_cached

"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging
from transformers.cache_utils import DynamicCache
import numpy as np
import os
import argparse
from accelerate import Accelerator
import time
import torch.multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
import gc
import random
import csv
import torch.distributed as dist
import re
from datasets import load_dataset
from tqdm import tqdm
import hashlib

logging.set_verbosity_error()
torch.backends.cuda.matmul.allow_tf32 = True

# Global store for parity debug (only used on rank0, iter0, cand0)
_PARITY_A = []
_PARITY_B = []

# -------------------- CLI (kept same as WP, plus 3 NP knobs) --------------------
parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='Qwen/Qwen2.5-1.5B-Instruct')
parser.add_argument('--hf_cache_dir', type=str,default=os.path.expanduser('~/hf_cache'))
parser.add_argument('--mixed_precision', type=str, default='fp16', choices=['no', 'fp16', 'bf16'])
parser.add_argument('--gpu_threads', type=int, default=1, help='Parallel threads per GPU (set to 1 if issues)')
parser.add_argument('--verbose', action='store_true')
parser.add_argument('--aggressive_gc', action='store_true')
parser.add_argument('--eval_interval', type=int, default=20)
parser.add_argument('--skip_eval', action='store_true')
parser.add_argument('--visualization_dir', type=str, default='./out_np_gsm8k')
parser.add_argument('--weight_sample_interval', type=int, default=10)
parser.add_argument('--meta_seed', type=int, default=42)
parser.add_argument('--n_train', type=int, default=64, help='Number of training samples')
parser.add_argument('--n_eval', type=int, default=100, help='Number of eval samples')
parser.add_argument('--n_iters', type=int, default=100, help='Number of ES iterations')
parser.add_argument('--pop_size', type=int, default=30, help='Population size')
parser.add_argument('--sigma', type=float, default=0.001, help='Activation noise std (NP)')
parser.add_argument('--alpha', type=float, default=0.001, help='Learning rate / step size')

# NP-specific (same semantics as before)
parser.add_argument('--np_include', type=str, default='all', choices=['head','attn','mlp','all'])
parser.add_argument('--last_k', type=int, default=0, help='restrict to last K blocks (0 = all)')
parser.add_argument('--max_length', type=int, default=512, help='(kept for CLI parity; not used in baseline/parity paths)')

# NP replay knobs (kept; cached replay ignores chunking but we keep flags for parity with prior runs)
parser.add_argument('--np_micro_batch_size', type=int, default=4,
                    help='NP replay micro-batch size; 0 = auto (unused in cached replay)')
parser.add_argument('--np_max_tok_chunk', type=int, default=256,
                    help='NP replay time-chunk size (unused in cached replay)')

# NEW: vectorized noise switch (default ON)
parser.add_argument('--np_fast_noise', action='store_true', default=True,
                    help='Use vectorized per-call noise (deterministic, faster).')
parser.add_argument('--no-np_fast_noise', dest='np_fast_noise', action='store_false')

# NEW: periodic grad sync (default 1 = every iter, original behavior)
parser.add_argument('--np_sync_every', type=int, default=1,
                    help='All-reduce gradient tensors every K iterations (1 = every iter).')

args = parser.parse_args()

NUM_ITERATIONS = args.n_iters
POPULATION_SIZE = args.pop_size
SIGMA = args.sigma
ALPHA = args.alpha
max_new_tokens = 256
do_sample = True  # keep greedy like WP --> **but greedy decoding kills NP signal. small activation noise almost never changes the selected token.
temperature = 1.0

# -------------------- Dataset (unchanged) --------------------
print("Loading GSM8K dataset...")
ds_train = load_dataset("gsm8k", "main", split=f"train[:{args.n_train}]")
ds_eval  = load_dataset("gsm8k", "main", split=f"test[:{args.n_eval}]")

dataset_train = []
for ex in ds_train:
    prompt = f"Q: {ex['question']}\nA: "
    m = re.search(r"####\s*([0-9\.\-]+)", ex["answer"])
    answer = m.group(1).strip() if m else None
    dataset_train.append((prompt, answer))

dataset_eval = []
for ex in ds_eval:
    prompt = f"Q: {ex['question']}\nA: "
    m = re.search(r"####\s*([0-9\.\-]+)", ex["answer"])
    answer = m.group(1).strip() if m else None
    dataset_eval.append((prompt, answer))

print(f"Loaded {len(dataset_train)} train samples, {len(dataset_eval)} eval samples")
dataset = dataset_train  # WP uses train set for optimization

# -------------------- Helpers (unchanged where possible) --------------------
def compute_reward(generated_text, target_text):
    if target_text is None:
        return 0.0
    numbers = re.findall(r"[-+]?\d*\.?\d+", generated_text.replace(",", ""))
    if len(numbers) > 0:
        final_number = numbers[-1]
        # ** sparse rewards
        return 1.0 if final_number == target_text else 0.0
    else:
        return 0.0

def force_memory_cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.synchronize()

def evaluate_model(model, tokenizer, input_text, target_text, accelerator, seed_idx=None, thread_id=None, verbose=False, return_text=False, return_ids=False):
    # IDENTICAL to WP path; no hooks active for baseline
    if verbose:
        print(f"Process {accelerator.process_index} Thread {thread_id} evaluating seed {seed_idx}")

    is_batch = isinstance(input_text, list)
    input_texts = input_text if is_batch else [input_text]
    target_texts = target_text if is_batch else [target_text]

    tokenized_inputs = tokenizer(input_texts, return_tensors="pt", padding=True, padding_side="left")  # no truncation
    input_ids = tokenized_inputs["input_ids"].to(accelerator.device)
    attention_mask = tokenized_inputs["attention_mask"].to(accelerator.device)
    with torch.inference_mode():
        unwrapped_model = accelerator.unwrap_model(model)
        outputs = unwrapped_model.generate(input_ids, attention_mask=attention_mask,
                                           max_new_tokens=max_new_tokens, do_sample=do_sample, temperature=temperature)
        if torch.cuda.is_available():
            torch.cuda.synchronize(accelerator.device)

    generated_texts = []
    generated_ids = []
    for i in range(len(input_texts)):
        try:
            generated_text = tokenizer.decode(outputs[i], skip_special_tokens=True)
        except TypeError:
            tokens = tokenizer.convert_ids_to_tokens(outputs[i], skip_special_tokens=True)
            filtered = [t for t in tokens if t is not None]
            generated_text = tokenizer.convert_tokens_to_string(filtered)
        generated_texts.append(generated_text)
        if return_ids:
            generated_ids.append(outputs[i].detach().cpu())

    del input_ids, outputs
    torch.cuda.empty_cache()

    rewards = [compute_reward(gen_text, tgt_text) for gen_text, tgt_text in zip(generated_texts, target_texts)]

    if return_ids:
        return rewards, generated_texts, generated_ids
    elif return_text:
        return rewards, generated_texts
    else:
        return rewards

def setup_reproducibility(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

def _ddp(accelerator: Accelerator) -> bool:
    return accelerator.num_processes > 1 and dist.is_available() and dist.is_initialized()

# -------------------- NP internals --------------------
def _stable_uid(s: str) -> int:
    h = hashlib.blake2s(s.encode('utf-8'), digest_size=8).digest()
    return int.from_bytes(h[:4], 'little') & 0x7FFFFFFF

def _infer_total_blocks(model) -> int:
    names = [n for n,_ in model.named_modules() if re.search(r'\.layers\.(\d+)\.', n)]
    if not names: return 0
    idxs = [int(re.search(r'\.layers\.(\d+)\.', n).group(1)) for n in names]
    return max(idxs)+1

def _is_target_linear(include: str, name: str, mod, last_k: int, total_blocks: int) -> bool:
    if not isinstance(mod, torch.nn.Linear): return False
    lname = name.lower()
    if last_k > 0 and total_blocks > 0:
        m = re.search(r'\.layers\.(\d+)\.', name)
        if m and int(m.group(1)) < total_blocks - last_k:
            return False
    if include == 'head':  return lname.endswith('lm_head')
    if include == 'attn':  return any(k in lname for k in ('attn','attention','q_proj','k_proj','v_proj','o_proj','in_proj'))
    if include == 'mlp':   return any(k in lname for k in ('mlp','ffn','gate_proj','up_proj','down_proj','fc1','fc2'))
    return True  # 'all'

class _NPActivationHook:
    """
    track=False: y_noisy = y + xi (no autograd)
    track=True : manual grads: W.grad += xi^T @ x; b.grad += sum(xi)

    When np_fast_noise=True, draw a full (N,D) xi with a single seeded RNG per call
    using (base_seed, layer_uid, row_counter_start). Deterministic per call.
    """
    def __init__(self, model, include, last_k, sigma, base_seed, fast_noise=True,
                 parity_enable=False, parity_phase='A', parity_limit=20):
        self.model = model
        self.include = include
        self.last_k = last_k
        self.sigma = float(sigma)
        self.base_seed = int(base_seed)
        self.fast_noise = bool(fast_noise)
        
        self.parity_enable = bool(parity_enable)
        self.parity_phase = parity_phase  # 'A' or 'B'
        self.parity_limit = int(parity_limit)
        self.parity_calls = 0

        self._orig = {}
        self._targets = []
        self._row_counters = {}
        self._layer_uids = {}
        self._track = False
        self._grad_scale = 1.0
        self._total_blocks = _infer_total_blocks(model)
        self._first_layer = None

        for name, mod in model.named_modules():
            if _is_target_linear(include, name, mod, last_k, self._total_blocks):
                self._targets.append((name, mod))
        for name, _ in self._targets:
            self._row_counters[name] = 0
            self._layer_uids[name] = _stable_uid(name)
        if self._targets:
            self._first_layer = self._targets[0][0]

    def _seed_for_row(self, layer_uid: int, row_index: int) -> int:
        x = (self.base_seed ^ layer_uid) + 0x9e3779b9 + (row_index << 6) + (row_index >> 2)
        return x & 0x7FFFFFFF

    def _seed_for_call(self, layer_uid: int, row_start: int) -> int:
        x = (self.base_seed ^ layer_uid) + 0x9e3779b9 + (row_start * 2654435761 & 0x7FFFFFFF)
        return x & 0x7FFFFFFF

    def attach(self, track: bool, grad_scale: float = 1.0):
        self._track = bool(track)
        self._grad_scale = float(grad_scale)
        for name, mod in self._targets:
            self._wrap(name, mod)

    def detach(self):
        for name, mod in self._targets:
            if name in self._orig:
                mod.forward = self._orig[name]
        self._orig.clear()

    def _wrap(self, name, mod: torch.nn.Linear):
        orig = mod.forward
        layer_uid = self._layer_uids[name]
        grad_scale = self._grad_scale
        first = (name == self._first_layer)

        def fwd(x):
            x_in = x
            if x_in.dim() > 2:
                N = int(np.prod(x_in.shape[:-1])); C = x_in.shape[-1]
                x2 = x_in.reshape(N, C); reshape_back = True
            else:
                x2 = x_in; N, C = x2.shape; reshape_back = False

            with torch.no_grad():
                y = orig(x2)
                y_dtype = y.dtype
                y32 = y.float()
                out_dim = y32.shape[-1]
                dev = y32.device

                # vectorized noise per call (fast) OR original per-row
                start = self._row_counters[name]
                if self.fast_noise:
                    g = torch.Generator(device=dev); g.manual_seed(self._seed_for_call(layer_uid, start))
                    xi = torch.randn((N, out_dim), generator=g, device=dev, dtype=torch.float32) * self.sigma
                else:
                    xi = torch.empty_like(y32)
                    for r in range(N):
                        g = torch.Generator(device=dev); g.manual_seed(self._seed_for_row(layer_uid, start + r))
                        xi[r] = torch.randn((out_dim,), generator=g, device=dev, dtype=torch.float32) * self.sigma

                # Parity trace (first targeted layer only, limited calls)
                if self.parity_enable and first and self.parity_calls < self.parity_limit:
                    h = hashlib.sha1(xi[0].float().cpu().numpy().tobytes()).hexdigest()
                    rec = (self.parity_calls, int(N), h)
                    if self.parity_phase == 'A':
                        _PARITY_A.append(rec)
                    else:
                        _PARITY_B.append(rec)
                    self.parity_calls += 1

                if self._track:
                    if mod.weight.grad is None:
                        mod.weight.grad = torch.zeros_like(mod.weight)
                    grad_w = xi.t().mm(x2.float())
                    mod.weight.grad.add_((grad_scale * grad_w).to(mod.weight.dtype))
                    if mod.bias is not None:
                        if mod.bias.grad is None:
                            mod.bias.grad = torch.zeros_like(mod.bias)
                        mod.bias.grad.add_((grad_scale * xi.sum(dim=0)).to(mod.bias.dtype))

                # ** this noise is applied after the linear layer. even if noise is present, it rarely propagates to logits
                y_noisy = (y32 + xi).to(y_dtype)

            self._row_counters[name] += N
            if reshape_back:
                new_shape = list(x_in.shape[:-1]) + [y_noisy.shape[-1]]
                y_noisy = y_noisy.reshape(*new_shape)
            return y_noisy

        mod.forward = fwd
        self._orig[name] = orig

def _named_linear_params(model, include, last_k):
    total_blocks = _infer_total_blocks(model)
    for name, mod in model.named_modules():
        if _is_target_linear(include, name, mod, last_k, total_blocks):
            if getattr(mod, "weight", None) is not None:
                yield name + ".weight", mod.weight
            if getattr(mod, "bias", None) is not None:
                yield name + ".bias", mod.bias

# ---------- Cache utilities for Qwen2 compatibility ----------
def to_dynamic_cache(pkv):
    """Convert past_key_values to DynamicCache for Qwen2 compatibility."""
    if isinstance(pkv, DynamicCache):
        return pkv
    dc = DynamicCache()
    for k, v in pkv:
        dc.key_cache.append(k)
        dc.value_cache.append(v)
    if dc.key_cache:
        dc.seen_tokens = dc.key_cache[0].shape[-2]  # [B,H,S,D] -> S
    return dc

def cache_index_select(cache, indices, device):
    """Select batch indices from cache (DynamicCache or legacy tuple)."""
    if cache is None:
        return None
    idx = torch.as_tensor(indices, device=device, dtype=torch.long)
    if isinstance(cache, DynamicCache):
        return cache.batch_select_indices(idx)
    # Legacy tuple fallback
    return tuple((k.index_select(0, idx), v.index_select(0, idx)) for k, v in cache)

# ---------- Exact A/B parity replay: prefill + cached decoding ----------
def _teacher_forced_replay_np_cached(accelerator, model, tokenizer, gen_ids, seed,
                                     include, last_k, grad_scale,
                                     parity_enable=False, parity_limit=20):
    """
    Exact parity with HF generate():
    1. One batched prefill over prompts
    2. Token-by-token decoding with KV cache
    3. Batch shrinking as sequences finish
    
    Uses exact token IDs from Pass A (no re-encoding) to ensure perfect parity.
    """
    base = accelerator.unwrap_model(model)
    device = accelerator.device

    hook = _NPActivationHook(base, include=include, last_k=last_k,
                             sigma=SIGMA, base_seed=int(seed),
                             fast_noise=args.np_fast_noise,
                             parity_enable=parity_enable, parity_phase='B',
                             parity_limit=parity_limit)
    hook.attach(track=True, grad_scale=grad_scale)

    old_cache = getattr(base.config, "use_cache", None)
    base.config.use_cache = True

    try:
        # Get prompts from dataset and tokenize
        prompts = [p for p, _ in dataset]
        prm_toks = tokenizer(prompts, return_tensors="pt", padding=True, padding_side="left")
        prm_ids = prm_toks["input_ids"].to(device)
        prm_mask = prm_toks["attention_mask"].to(device)
        prm_len = prm_mask.sum(dim=1).tolist()
        
        B = len(gen_ids)
        
        # Extract continuation tokens from exact IDs generated in Pass A
        cont_tokens = []
        for i, ids in enumerate(gen_ids):
            ids_tensor = ids.to(device) if not ids.is_cuda else ids
            li = prm_len[i]
            cont_tokens.append(ids_tensor[li:])  # exact continuation tokens from A --> ** no counterfactual trajectories, NP degenerates
        
        with torch.no_grad():
            # Step 1: Batched prefill over all prompts
            out = base(input_ids=prm_ids, attention_mask=prm_mask, use_cache=True, output_hidden_states=False)
            past = to_dynamic_cache(out.past_key_values)
            
            # Step 2: Token-by-token decoding with batch shrinking
            active_idx = [i for i in range(B) if cont_tokens[i].numel() > 0]
            if len(active_idx) == 0:
                return
            
            past_active = cache_index_select(past, active_idx, device)
            cont_lists = [cont_tokens[i] for i in active_idx]
            cursors = [0] * len(active_idx)
            
            while len(active_idx) > 0:
                # Next token for each active sequence
                next_ids = torch.stack([cont_lists[j][cursors[j]] for j in range(len(active_idx))], dim=0).unsqueeze(1)
                
                out = base(input_ids=next_ids, past_key_values=past_active, use_cache=True, output_hidden_states=False)
                past_new = to_dynamic_cache(out.past_key_values)
                
                # Update cursors and find finished sequences
                finished = []
                for j in range(len(active_idx)):
                    cursors[j] += 1
                    if cursors[j] >= cont_lists[j].numel():
                        finished.append(j)
                
                # Shrink batch by removing finished sequences
                if finished:
                    keep = [j for j in range(len(active_idx)) if j not in finished]
                    if keep:
                        past_active = cache_index_select(past_new, keep, device)
                        active_idx = [active_idx[j] for j in keep]
                        cont_lists = [cont_lists[j] for j in keep]
                        cursors = [cursors[j] for j in keep]
                    else:
                        break
                else:
                    past_active = past_new
                    
    finally:
        hook.detach()
        if old_cache is not None:
            base.config.use_cache = old_cache

# -------------------- NP-modified seed processing --------------------
def process_seed(seed_args):
    """
    Identical threading shape as WP, but:
      - We DO NOT add weight noise.
      - We wrap the model in an activation-noise hook for Pass-A generation only.
    Returns: (seed_idx, average_reward, generated_texts, generated_ids, parity_A_or_None)
    """
    seed_idx, seed, model, tokenizer, accelerator, thread_id, verbose, do_parity_trace = seed_args
    if verbose:
        print(f"Process {accelerator.process_index} Thread {thread_id} processing seed {seed_idx} (value: {seed})")

    # Pass A: activation noise during generate()
    base = accelerator.unwrap_model(model)
    hook = _NPActivationHook(base, include=args.np_include, last_k=args.last_k,
                             sigma=SIGMA, base_seed=int(seed), fast_noise=args.np_fast_noise,
                             parity_enable=do_parity_trace, parity_phase='A',
                             parity_limit=20)
    hook.attach(track=False)
    try:
        input_texts = [inp for inp, _ in dataset]
        target_texts = [tgt for _, tgt in dataset]
        rewards, generated_texts, generated_ids = evaluate_model(
            model, tokenizer, input_texts, target_texts, accelerator,
            seed_idx=seed_idx, thread_id=thread_id, verbose=verbose, return_text=True, return_ids=True
        )
    finally:
        hook.detach()

    average_reward = float(np.mean(rewards))
    if verbose:
        print(f"Process {accelerator.process_index} Thread {thread_id} completed seed {seed_idx} with reward {average_reward:.4f}")

    # Copy out parity A only if we traced
    parity_A = list(_PARITY_A) if do_parity_trace else None
    # Clear global store for next usage
    if do_parity_trace:
        _PARITY_A.clear()
    
    return seed_idx, average_reward, generated_texts, generated_ids, parity_A

# -------------------- Main (unchanged except NP update path) --------------------
def main():
    setup_reproducibility(args.meta_seed)
    accelerator = Accelerator(mixed_precision=args.mixed_precision)

    if accelerator.is_main_process:
        print(f"Meta seed: {args.meta_seed}")
        print(f"Total GPUs: {accelerator.num_processes}, Threads per GPU: {args.gpu_threads}")
        print(f"Using mixed precision: {args.mixed_precision}")
        print(f"Population size: {POPULATION_SIZE}, Iterations: {NUM_ITERATIONS}")
        print(f"Sigma: {SIGMA}, Alpha: {ALPHA}")
        print(f"Train samples: {len(dataset)}, Eval samples: {len(dataset_eval)}")
        print(f"Expected evaluations per iteration: {POPULATION_SIZE} (distributed across {accelerator.num_processes} GPUs)")

    reward_history_path = os.path.join(args.visualization_dir, "reward_history.csv")
    if accelerator.is_main_process:
        os.makedirs(args.visualization_dir, exist_ok=True)
        with open(reward_history_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["iteration", "candidate_index", "seed", "train_reward"])

    model_name = args.model_name
    hf_cache_dir = args.hf_cache_dir

    if accelerator.is_main_process:
        print(f"Loading model {model_name}...")

    # Multiple model copies for threads (unchanged)
    model_list = []
    for model_index in range(args.gpu_threads):
        model_list.append(AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=hf_cache_dir,
            device_map={"": accelerator.process_index},
            torch_dtype=torch.float16 if args.mixed_precision == 'fp16' else (torch.bfloat16 if args.mixed_precision == 'bf16' else torch.float32),
        ))

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, cache_dir=hf_cache_dir)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = 'left'

    if accelerator.is_main_process:
        print("Model loaded successfully")

    for i in range(len(model_list)):
        model_list[i].eval()
        model_list[i] = accelerator.prepare(model_list[i])

    force_memory_cleanup()

    # Baseline eval (unchanged; no hooks)
    if accelerator.is_main_process:
        train_inputs = [prompt for prompt, _ in dataset_train]
        train_targets = [ans for _, ans in dataset_train]
        eval_inputs = [prompt for prompt, _ in dataset_eval]
        eval_targets = [ans for _, ans in dataset_eval]

        train_rewards, train_texts = evaluate_model(
            model_list[0], tokenizer, train_inputs, train_targets,
            accelerator, verbose=False, return_text=True
        )
        eval_rewards = evaluate_model(
            model_list[0], tokenizer, eval_inputs, eval_targets,
            accelerator, verbose=False, return_text=False
        )

        ## testing to see if the issue is that the reward variance is still too low
        baseline_unique, baseline_counts = np.unique(train_rewards, return_counts=True)
        print("Unique rewards and their counts (baseline):")
        print(dict(zip(baseline_unique, baseline_counts)))

        baseline_train = float(np.mean(train_rewards))
        baseline_eval = float(np.mean(eval_rewards))
        print(f"[BASELINE] train_acc={baseline_train:.3f} eval_acc={baseline_eval:.3f}")

        with open("/tmp/wp_baseline_completions.txt", "w") as f:
            f.write("=== Baseline Train Completions (NP parity) ===\n\n")
            for idx, (text, prompt, gold) in enumerate(zip(train_texts, train_inputs, train_targets)):
                nums = re.findall(r"[-+]?\d*\.?\d+", text.replace(",", ""))
                final_num = nums[-1] if nums else ""
                correct = "✓" if final_num == gold else "✗"
                f.write(f"--- Train {idx} [{correct}] ---\n")
                f.write(f"Prompt: {prompt[:80]}...\n")
                f.write(f"Gold: {gold}\n")
                f.write(f"Final Number: {final_num}\n")
                f.write(f"Completion: {text}\n\n")

    training_start_time = time.time()

    if accelerator.is_main_process:
        pbar = tqdm(range(NUM_ITERATIONS), desc="ES Training",
                   bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
    else:
        pbar = range(NUM_ITERATIONS)

    for (i, iteration) in enumerate(pbar):
        iter_start_time = time.time()
        force_memory_cleanup()

        iteration_seed = args.meta_seed + iteration

        # Seeds (unchanged)
        if accelerator.is_main_process:
            np.random.seed(iteration_seed)
            seeds = np.random.randint(0, 2**31, size=POPULATION_SIZE, dtype=np.int64).tolist()
            seeds_tensor = torch.tensor(seeds, device=accelerator.device)
        else:
            seeds_tensor = torch.zeros(POPULATION_SIZE, dtype=torch.long, device=accelerator.device)

        if _ddp(accelerator):
            dist.broadcast(seeds_tensor, src=0)

        seeds = seeds_tensor.cpu().tolist()

        # Assign seeds to processes (unchanged)
        local_seeds = []
        for seed_idx, seed in enumerate(seeds):
            if seed_idx % accelerator.num_processes == accelerator.process_index:
                local_seeds.append((seed_idx, seed))

        # Pass A across local seeds (unchanged thread pool pattern)
        local_results = []  # (seed_idx, reward, texts)
        batch_size = max(1, min(args.gpu_threads, len(local_seeds)))

        for batch_start in range(0, len(local_seeds), batch_size):
            batch_end = min(batch_start + batch_size, len(local_seeds))
            batch_batch = local_seeds[batch_start:batch_end]
            
            # Enable parity trace only for iter 0, cand 0, rank 0
            def _trace_for(sid: int) -> bool:
                return accelerator.is_main_process and iteration == 0 and sid == 0

            with ThreadPoolExecutor(max_workers=len(batch_batch)) as executor:
                thread_args = []
                for thread_id, (seed_idx, seed) in enumerate(batch_batch):
                    thread_args.append((seed_idx, seed, model_list[thread_id], tokenizer, accelerator, thread_id, args.verbose, _trace_for(seed_idx)))
                results = list(executor.map(process_seed, thread_args))
                local_results.extend(results)

            if args.aggressive_gc:
                force_memory_cleanup()

        # Collect rewards, IDs, and parity traces
        all_rewards = torch.zeros(POPULATION_SIZE, device=accelerator.device)
        texts_by_seed_idx = [None] * POPULATION_SIZE
        ids_by_seed_idx = [None] * POPULATION_SIZE
        parity_A_by_sid = {}

        for seed_idx, reward, texts, gen_ids, parity_A in local_results:
            all_rewards[seed_idx] = reward
            texts_by_seed_idx[seed_idx] = texts
            ids_by_seed_idx[seed_idx] = gen_ids
            parity_A_by_sid[seed_idx] = parity_A

        if _ddp(accelerator):
            dist.all_reduce(all_rewards, op=dist.ReduceOp.SUM)

        rewards = all_rewards.cpu().tolist()

        ## testing to see if the issue is that the reward variance is still too low
        iter_rewards, iter_counts = np.unique(rewards, return_counts=True)
        print("[UNIQUE REWARDS:]")
        print(dict(zip(iter_rewards, iter_counts)))

        del all_rewards
        force_memory_cleanup()

        # Save rewards CSV (unchanged)
        if accelerator.is_main_process:
            with open(reward_history_path, "a", newline="") as f:
                writer = csv.writer(f)
                for cand_idx, (seed, r) in enumerate(zip(seeds, rewards)):
                    writer.writerow([iteration + 1, cand_idx, int(seed), float(r)])

        # Normalize rewards (unchanged)
        rewards_tensor = np.array(rewards, dtype=np.float32)
        rewards_normalized = (rewards_tensor - rewards_tensor.mean()) / (rewards_tensor.std() + 1e-8)

        ## testing to see if the issue is that the reward variance is still too low --> with the normalized reward
        iter_rewards_normalized, iter_counts_normalized = np.unique(rewards_tensor, return_counts=True)
        print("[UNIQUE REWARDS - NORMALIZED:]")
        print(dict(zip(iter_rewards_normalized, iter_counts_normalized)))

        # cheecking to see if std = 0.03
        if i == 1 or i == 2:
            print("mean reward:", rewards.mean())
            print("reward std:", rewards.std())
            print("min/max:", rewards.min(), rewards.max())

        # ---------- NP UPDATE (only part that differs from WP) ----------
        original_model = accelerator.unwrap_model(model_list[0])

        # zero grads
        for _, p in _named_linear_params(original_model, include=args.np_include, last_k=args.last_k):
            if p.grad is not None:
                p.grad.zero_()
        
        # ** checking double-scaling
        for name, p in _named_linear_params(original_model, include=args.np_include, last_k=args.last_k):
            if p.grad is not None:
                print(name, p.grad.norm().item())
                break

        # replay only the candidates handled by this process (EXACT cached replay with exact IDs)
        for seed_idx, seed in local_seeds:
            gen_ids = ids_by_seed_idx[seed_idx]
            if gen_ids is None:
                print("reached gen_ids None")
                continue
            grad_scale = float(rewards_normalized[seed_idx]) / float(POPULATION_SIZE)
            
            # Enable parity tracing only if we captured A for this sid
            parity_A = parity_A_by_sid.get(seed_idx, None)
            parity_enable = (parity_A is not None)
            
            # Clear previous B traces
            if parity_enable:
                _PARITY_B.clear()
            
            _teacher_forced_replay_np_cached(
                accelerator, model_list[0], tokenizer, gen_ids, int(seed),
                include=args.np_include, last_k=args.last_k, grad_scale=grad_scale,
                parity_enable=parity_enable, parity_limit=20
            )
            
            # Parity check print (only once: iter 0, cand 0, rank 0)
            if parity_enable and accelerator.is_main_process and iteration == 0 and seed_idx == 0:
                A, B = parity_A, list(_PARITY_B)
                _PARITY_B.clear()
                ok = True
                L = min(len(A), len(B))
                mismatch_at = None
                for i in range(L):
                    if A[i] != B[i]:
                        ok = False
                        mismatch_at = (i, A[i], B[i])
                        break
                if ok and len(A) != len(B):
                    ok = False
                    mismatch_at = ("length", len(A), len(B))
                if ok:
                    print(f"[PARITY] PASS: first-layer call trace (A vs B) matches for {min(20, len(A))} calls.")
                else:
                    print(f"[PARITY] FAIL: {mismatch_at}")

        # all-reduce grads so every process sees the same update (keep every iter for parity)
        do_sync = (args.np_sync_every <= 1) or ((iteration + 1) % max(1, args.np_sync_every) == 0)
        if _ddp(accelerator) and do_sync:
            for _, p in _named_linear_params(original_model, include=args.np_include, last_k=args.last_k):
                if p.grad is not None:
                    dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)

        # Normalized global step (stable step size)
        params = [p for _, p in _named_linear_params(original_model, include=args.np_include, last_k=args.last_k) if p.grad is not None]
        if params:
            with torch.no_grad():
                total = torch.zeros([], device=params[0].device, dtype=torch.float32)
                for p in params:
                    total += (p.grad.float().norm(2) ** 2)
                gnorm = total.sqrt().clamp_min(1e-12)
                step_scale = ALPHA / gnorm  
                for p in params:
                    p.add_(step_scale * p.grad)  # ascent
                    p.grad = None

        # Copy weights to other replicas (unchanged)
        for model_idx in range(1, len(model_list)):
            original_model_tmp = accelerator.unwrap_model(model_list[model_idx])
            for name, param in original_model_tmp.named_parameters():
                param.data.copy_(original_model.get_parameter(name).data.clone())

        if torch.cuda.is_available():
            torch.cuda.synchronize(accelerator.device)

        force_memory_cleanup()

        iter_time = time.time() - iter_start_time
        mean_reward = rewards_tensor.mean().item()
        min_reward = rewards_tensor.min().item()
        max_reward = rewards_tensor.max().item()

        # Periodic eval (unchanged)
        if iteration % args.eval_interval == 0 and not args.skip_eval:
            eval_input_texts = [input_text for input_text, _ in dataset_eval]
            eval_target_texts = [target_text for _, target_text in dataset_eval]
            eval_rewards = evaluate_model(
                model_list[0], tokenizer, eval_input_texts, eval_target_texts, accelerator,
                seed_idx=None, thread_id=None, verbose=False, return_text=False
            )
            eval_reward = np.mean(eval_rewards)
            if accelerator.is_main_process:
                tqdm.write(f"[ES] iter={iteration:03d} Time: {iter_time:.1f}s | TRAIN: {mean_reward:.3f} | EVAL: {eval_reward:.3f} | Min: {min_reward:.3f}, Max: {max_reward:.3f}")
        else:
            if accelerator.is_main_process:
                tqdm.write(f"[ES] iter={iteration:03d} Time: {iter_time:.1f}s | TRAIN: {mean_reward:.3f} | Min: {min_reward:.3f}, Max: {max_reward:.3f}")

        del rewards_tensor, rewards_normalized
        force_memory_cleanup()

    if accelerator.is_main_process and isinstance(pbar, tqdm):
        pbar.close()

    total_time = time.time() - training_start_time

    if accelerator.is_main_process:
        print(f"Training completed in {total_time:.2f}s ({total_time/60:.2f} minutes)")
        save_dir = os.path.join(args.visualization_dir, "final_model")
        print(f"Saving model to {save_dir}...")
        original_model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
        print(f"Model saved successfully.")

    # optional: clean DDP shutdown to silence warning
    try:
        accelerator.wait_for_everyone()
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
    except Exception:
        pass

if __name__ == "__main__":
    os.environ["PYTHONWARNINGS"] = "ignore"
    mp.set_start_method('spawn', force=True)
    main()
