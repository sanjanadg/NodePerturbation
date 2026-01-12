import argparse
import random
import numpy as np
import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator
from datasets import load_dataset

# ----------------------------
# Utils
# ----------------------------

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def gsm8k_reward(outputs, answers):
    """
    Exact-match final-number accuracy.
    Returns scalar reward in [0,1].
    """
    preds = []
    for out in outputs:
        text = out
        nums = [
            x for x in text.replace(",", "").split()
            if x.replace(".", "").isdigit()
        ]
        preds.append(nums[-1] if nums else "")

    correct = sum(p == a for p, a in zip(preds, answers))
    return correct / len(answers)

# ----------------------------
# NP Activation Hook
# ----------------------------

class NPActivationHook:
    def __init__(self, model, include, sigma, seed):
        self.model = model
        self.include = include
        self.sigma = sigma
        self.seed = seed
        self.handles = []
        self.track = False
        self.grad_scale = 1.0

    def _should_hook(self, name, module):
        if not isinstance(module, nn.Linear):
            return False
        if self.include == "head":
            return name.endswith("lm_head")
        if self.include == "mlp":
            return "mlp" in name
        return False

    def attach(self, track=False, grad_scale=1.0):
        self.track = track
        self.grad_scale = grad_scale
        torch.manual_seed(self.seed)

        for name, module in self.model.named_modules():
            if self._should_hook(name, module):
                self.handles.append(
                    module.register_forward_hook(self._hook_fn(module))
                )

    def detach(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()

    def _hook_fn(self, module):
        def hook(module, inp, out):
            noise = torch.randn_like(out) * self.sigma

            if not self.track:
                return out + noise

            # --- Node Perturbation gradient estimator ---
            x = inp[0].detach()

            if module.weight.grad is None:
                module.weight.grad = torch.zeros_like(module.weight)

            grad_w = (
                noise.reshape(-1, noise.size(-1)).T
                @ x.reshape(-1, x.size(-1))
            )
            module.weight.grad.add_(self.grad_scale * grad_w)

            if module.bias is not None:
                if module.bias.grad is None:
                    module.bias.grad = torch.zeros_like(module.bias)
                module.bias.grad.add_(self.grad_scale * noise.sum(dim=0))

            return out + noise

        return hook

# ----------------------------
# NP-compatible generation
# ----------------------------

def np_generate(model, tokenizer, prompts, seed, max_new_tokens, temperature):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    toks = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        padding_side="left",
    ).to(model.device)

    input_ids = toks["input_ids"]
    attention_mask = toks["attention_mask"]

    generated = input_ids
    past = None

    for _ in range(max_new_tokens):
        out = model(
            input_ids=generated[:, -1:] if past is not None else generated,
            attention_mask=attention_mask,
            past_key_values=past,
            use_cache=True,
        )

        logits = out.logits[:, -1, :] / temperature
        probs = torch.softmax(logits, dim=-1)
        next_ids = torch.multinomial(probs, num_samples=1)

        generated = torch.cat([generated, next_ids], dim=1)
        attention_mask = torch.cat(
            [attention_mask, torch.ones_like(next_ids)], dim=1
        )

        past = out.past_key_values

    return tokenizer.batch_decode(generated, skip_special_tokens=True)

# ----------------------------
# Main
# ----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--np_include", choices=["head", "mlp"], default="head")
    parser.add_argument("--sigma", type=float, default=0.005)
    parser.add_argument("--alpha", type=float, default=1e-3)
    parser.add_argument("--pop_size", type=int, default=30)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    accelerator = Accelerator()
    set_seed(args.seed)

    # Dataset
    ds = load_dataset("gsm8k", "main", split="train[:8]")
    prompts = ds["question"]
    answers = [a.split("####")[-1].strip() for a in ds["answer"]]

    # Model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map={"": accelerator.process_index},
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.p = 0.0

    model = accelerator.prepare(model)

    # ----------------------------
    # Training loop
    # ----------------------------

    for it in range(args.iters):
        rewards = []
        seeds = []

        # ---------- Pass A ----------
        for i in range(args.pop_size):
            seed_i = args.seed + it * 1000 + i

            hook = NPActivationHook(model, args.np_include, args.sigma, seed_i)
            hook.attach(track=False)

            outs = np_generate(
                model,
                tokenizer,
                prompts,
                seed_i,
                args.max_new_tokens,
                args.temperature,
            )

            hook.detach()

            r = gsm8k_reward(outs, answers)
            rewards.append(r)
            seeds.append(seed_i)

        rewards = torch.tensor(rewards, device=model.device)
        std = rewards.std().item()

        accelerator.print(
            f"[iter {it}] mean={rewards.mean():.3f} std={std:.4f}"
        )

        if std == 0.0:
            continue

        rewards = (rewards - rewards.mean()) / (std + 1e-8)

        # ---------- Pass B ----------
        for p in model.parameters():
            if p.grad is not None:
                p.grad.zero_()

        for seed_i, r in zip(seeds, rewards):
            hook = NPActivationHook(model, args.np_include, args.sigma, seed_i)
            hook.attach(track=True, grad_scale=r.item())

            _ = np_generate(
                model,
                tokenizer,
                prompts,
                seed_i,
                args.max_new_tokens,
                args.temperature,
            )

            hook.detach()

        # ---------- Update ----------
        with torch.no_grad():
            for p in model.parameters():
                if p.grad is not None:
                    p.add_(args.alpha * p.grad)
                    p.grad.zero_()

    accelerator.print("Training done.")

if __name__ == "__main__":
    main()