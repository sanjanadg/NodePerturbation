 #!/usr/bin/env python3

"""
accelerate launch --num_processes=8 wp_llm_gsm8k.py \
  --model_name Qwen/Qwen2.5-1.5B-Instruct \
  --hf_cache_dir $HOME/hf_cache \
  --mixed_precision bf16 \
  --n_train 10 --n_eval 100 \
  --n_iters 10 --pop_size 30 \
  --sigma 0.001 --alpha 0.001 \
  --meta_seed 42 \
  --gpu_threads 1 \
  --eval_interval 20 \
  --visualization_dir ./out_wp_gsm8k
"""

# need to mkdir -p $HOME/hf_cache

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging
import numpy as np
import copy
import os
import argparse
from accelerate import Accelerator
import time
import torch.multiprocessing as mp
import threading
from queue import Queue
from concurrent.futures import ThreadPoolExecutor
import math
import gc
import random
import csv
import torch.distributed as dist
import re
from datasets import load_dataset
from tqdm import tqdm

logging.set_verbosity_error()
torch.backends.cuda.matmul.allow_tf32 = True

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='Qwen/Qwen2.5-1.5B-Instruct')
parser.add_argument('--hf_cache_dir', type=str,default=os.path.expanduser('~/hf_cache'))
parser.add_argument('--mixed_precision', type=str, default='fp16', choices=['no', 'fp16', 'bf16'])
parser.add_argument('--gpu_threads', type=int, default=1, help='Parallel threads per GPU (set to 1 if issues)')
parser.add_argument('--verbose', action='store_true')
parser.add_argument('--aggressive_gc', action='store_true')
parser.add_argument('--eval_interval', type=int, default=20)
parser.add_argument('--skip_eval', action='store_true')
parser.add_argument('--visualization_dir', type=str, default='./out_wp_gsm8k')
parser.add_argument('--weight_sample_interval', type=int, default=10)
parser.add_argument('--meta_seed', type=int, default=42)
parser.add_argument('--n_train', type=int, default=64, help='Number of training samples')
parser.add_argument('--n_eval', type=int, default=100, help='Number of eval samples')
parser.add_argument('--n_iters', type=int, default=100, help='Number of ES iterations')
parser.add_argument('--pop_size', type=int, default=30, help='Population size')
parser.add_argument('--sigma', type=float, default=0.001, help='Perturbation noise std')
parser.add_argument('--alpha', type=float, default=0.001, help='Learning rate')
args = parser.parse_args()

NUM_ITERATIONS = args.n_iters
POPULATION_SIZE = args.pop_size
SIGMA = args.sigma
ALPHA = args.alpha
max_new_tokens = 256
do_sample = False

# Load GSM8K dataset
print("Loading GSM8K dataset...")
ds_train = load_dataset("gsm8k", "main", split=f"train[:{args.n_train}]")
ds_eval = load_dataset("gsm8k", "main", split=f"test[:{args.n_eval}]")

# Format dataset as (prompt, answer) tuples
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

# Use train set for ES optimization
dataset = dataset_train

# Dump first 10 prompts to verify dataset consistency with NP
with open("/tmp/wp_train_prompts.txt", "w") as f:
    f.write("=== WP Script - First 10 Training Prompts ===\n\n")
    for idx, (prompt, answer) in enumerate(dataset_train[:10]):
        f.write(f"--- Prompt {idx} ---\n")
        f.write(f"Text: {repr(prompt)}\n")
        f.write(f"Answer: {answer}\n\n")
print("Dumped first 10 WP training prompts to /tmp/wp_train_prompts.txt")

def compute_reward(generated_text, target_text):
    """
    Reward function for GSM8K: extract final number and compare with target.
    Returns 1.0 if correct, 0.0 otherwise.
    """
    if target_text is None:
        return 0.0
    
    # Extract all numbers from generated text
    numbers = re.findall(r"[-+]?\d*\.?\d+", generated_text.replace(",", ""))
    
    if len(numbers) > 0:
        # Compare last number with target
        final_number = numbers[-1]
        return 1.0 if final_number == target_text else 0.0
    else:
        return 0.0

def force_memory_cleanup():
    """Force aggressive memory cleanup"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.synchronize()

def evaluate_model(model, tokenizer, input_text, target_text, accelerator, seed_idx=None, thread_id=None, verbose=False, return_text=False):
    """
    Generate a response from the model given an input (single or batch) and compute rewards.
    """
    if verbose:
        print(f"Process {accelerator.process_index} Thread {thread_id} evaluating seed {seed_idx}")

    # Handle both single input and batch input
    is_batch = isinstance(input_text, list)
    input_texts = input_text if is_batch else [input_text]
    target_texts = target_text if is_batch else [target_text]

    # Batch tokenization
    tokenized_inputs = tokenizer(input_texts, return_tensors="pt", padding=True, padding_side="left")
    input_ids = tokenized_inputs["input_ids"].to(accelerator.device)
    attention_mask = tokenized_inputs["attention_mask"].to(accelerator.device)
    with torch.inference_mode():
        unwrapped_model = accelerator.unwrap_model(model)
        outputs = unwrapped_model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=max_new_tokens, do_sample=do_sample)
        if torch.cuda.is_available():
            torch.cuda.synchronize(accelerator.device)

    # Decode batch outputs
    generated_texts = []
    for i in range(len(input_texts)):
        try:
            generated_text = tokenizer.decode(outputs[i], skip_special_tokens=True)
        except TypeError:
            tokens = tokenizer.convert_ids_to_tokens(outputs[i], skip_special_tokens=True)
            filtered = [t for t in tokens if t is not None]
            generated_text = tokenizer.convert_tokens_to_string(filtered)
        generated_texts.append(generated_text)

    del input_ids, outputs
    torch.cuda.empty_cache()

    # Compute rewards for batch texts
    rewards = [compute_reward(gen_text, tgt_text) for gen_text, tgt_text in zip(generated_texts, target_texts)]

    if return_text:
        return rewards, generated_texts
    else:
        return rewards

def process_seed(seed_args):
    """Function to process a single seed, used for thread pool"""
    seed_idx, seed, model, tokenizer, accelerator, thread_id, verbose = seed_args

    if verbose:
        print(f"Process {accelerator.process_index} Thread {thread_id} processing seed {seed_idx} (value: {seed})")

    # Apply perturbation
    original_model = accelerator.unwrap_model(model)
    for name, param in original_model.named_parameters():
        gen = torch.Generator(device=param.device)
        gen.manual_seed(int(seed))
        noise = torch.randn(
            param.shape,
            generator=gen,
            device=param.device,
            dtype=param.dtype
        )
        param.data.add_(SIGMA * noise)

    # Ensure weights are fully loaded before evaluation
    if torch.cuda.is_available():
        torch.cuda.synchronize(accelerator.device)

    # Evaluate all prompts with perturbed weights in batch
    input_texts = [input_text for input_text, _ in dataset]
    target_texts = [target_text for _, target_text in dataset]
    rewards = evaluate_model(
        model, tokenizer, input_texts, target_texts, accelerator,
        seed_idx=seed_idx, thread_id=thread_id, verbose=verbose, return_text=False
    )
    total_reward = sum(rewards)

    # Restore original weights
    for name, param in original_model.named_parameters():
        gen = torch.Generator(device=param.device)
        gen.manual_seed(int(seed))
        noise = torch.randn(
            param.shape,
            generator=gen,
            device=param.device,
            dtype=param.dtype
        )
        param.data.add_(-SIGMA * noise)

    if torch.cuda.is_available():
        torch.cuda.synchronize(accelerator.device)

    average_reward = total_reward / len(dataset)

    force_memory_cleanup()

    if verbose:
        print(f"Process {accelerator.process_index} Thread {thread_id} completed seed {seed_idx} with reward {average_reward:.4f}")

    return seed_idx, average_reward

def setup_reproducibility(seed):
    """Setup reproducibility for a specific seed"""
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

    # Initialize reward history CSV
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

    # Load multiple model copies for parallel threads
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

    # Prepare models
    for i in range(len(model_list)):
        model_list[i].eval()
        model_list[i] = accelerator.prepare(model_list[i])

    force_memory_cleanup()

    # Baseline (no perturbation) accuracy before ES updates
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
        baseline_train = float(np.mean(train_rewards))
        baseline_eval = float(np.mean(eval_rewards))
        print(f"[BASELINE] train_acc={baseline_train:.3f} eval_acc={baseline_eval:.3f}")
        
        # Dump baseline completions to compare with NP
        with open("/tmp/wp_baseline_completions.txt", "w") as f:
            f.write("=== WP Baseline Train Completions ===\n\n")
            for idx, (text, prompt, gold) in enumerate(zip(train_texts, train_inputs, train_targets)):
                nums = re.findall(r"[-+]?\d*\.?\d+", text.replace(",", ""))
                final_num = nums[-1] if nums else ""
                correct = "✓" if final_num == gold else "✗"
                f.write(f"--- Train {idx} [{correct}] ---\n")
                f.write(f"Prompt: {prompt[:80]}...\n")
                f.write(f"Gold: {gold}\n")
                f.write(f"Final Number: {final_num}\n")
                f.write(f"Completion: {text}\n\n")
        print("Dumped WP baseline completions to /tmp/wp_baseline_completions.txt")

    training_start_time = time.time()

    # Create progress bar on main process only
    if accelerator.is_main_process:
        pbar = tqdm(range(NUM_ITERATIONS), desc="ES Training",
                   bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
    else:
        pbar = range(NUM_ITERATIONS)

    for iteration in pbar:
        iter_start_time = time.time()
        force_memory_cleanup()

        iteration_seed = args.meta_seed + iteration
        
        # Generate seeds on main process
        if accelerator.is_main_process:
            np.random.seed(iteration_seed)
            seeds = np.random.randint(0, 2**31, size=POPULATION_SIZE, dtype=np.int64).tolist()
            seeds_tensor = torch.tensor(seeds, device=accelerator.device)
        else:
            seeds_tensor = torch.zeros(POPULATION_SIZE, dtype=torch.long, device=accelerator.device)

        # Broadcast seeds
        if _ddp(accelerator):
            dist.broadcast(seeds_tensor, src=0)

        seeds = seeds_tensor.cpu().tolist()

        # Assign seeds to processes
        local_seeds = []
        for seed_idx, seed in enumerate(seeds):
            if seed_idx % accelerator.num_processes == accelerator.process_index:
                local_seeds.append((seed_idx, seed))

        # Process seeds in batches
        local_rewards = []
        batch_size = max(1, min(args.gpu_threads, len(local_seeds)))

        for batch_start in range(0, len(local_seeds), batch_size):
            batch_end = min(batch_start + batch_size, len(local_seeds))
            batch_seeds = local_seeds[batch_start:batch_end]

            with ThreadPoolExecutor(max_workers=len(batch_seeds)) as executor:
                thread_args = []
                for thread_id, (seed_idx, seed) in enumerate(batch_seeds):
                    thread_args.append((seed_idx, seed, model_list[thread_id], tokenizer, accelerator, thread_id, args.verbose))

                results = list(executor.map(process_seed, thread_args))
                local_rewards.extend(results)

            force_memory_cleanup()

        # Collect rewards from all processes
        all_rewards = torch.zeros(POPULATION_SIZE, device=accelerator.device)

        for seed_idx, reward in local_rewards:
            all_rewards[seed_idx] = reward

        if _ddp(accelerator):
            dist.all_reduce(all_rewards, op=dist.ReduceOp.SUM)

        rewards = all_rewards.cpu().tolist()
        del all_rewards
        force_memory_cleanup()

        # Save rewards to CSV
        if accelerator.is_main_process:
            with open(reward_history_path, "a", newline="") as f:
                writer = csv.writer(f)
                for cand_idx, (seed, r) in enumerate(zip(seeds, rewards)):
                    writer.writerow([iteration + 1, cand_idx, int(seed), float(r)])

        # Normalize rewards
        rewards_tensor = np.array(rewards, dtype=np.float32)
        rewards_normalized = (rewards_tensor - rewards_tensor.mean()) / (rewards_tensor.std() + 1e-8)

        # Update model weights
        original_model = accelerator.unwrap_model(model_list[0])
        for name, param in original_model.named_parameters():
            gen = torch.Generator(device=param.device)
            update = torch.zeros_like(param)
            for seed_idx in range(POPULATION_SIZE):
                r_norm = rewards_normalized[seed_idx]
                seed = seeds[seed_idx]
                gen.manual_seed(int(seed))
                noise = torch.randn(
                    param.shape,
                    generator=gen,
                    device=param.device,
                    dtype=param.dtype
                )
                noise.mul_(float(r_norm))
                update.add_(noise)
                del noise
            update.div_(POPULATION_SIZE)
            param.data.add_(ALPHA * update)
            torch.cuda.empty_cache()

        # Copy weights to other replicas
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

        # Periodic eval on held-out test set
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
    
    # Close progress bar
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

if __name__ == "__main__":
    os.environ["PYTHONWARNINGS"] = "ignore"
    mp.set_start_method('spawn', force=True)
    main()
