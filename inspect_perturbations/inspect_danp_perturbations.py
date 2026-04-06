import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from conciseness_task.danp_llm_v3 import DANPHook, get_target_linears

# python3 -m inspect_perturbations.inspect_danp_perturbations

SIGMA = 0.001
POPULATION_SIZE = 30
initial_seed = 33
model_name = "Qwen/Qwen2.5-3B-Instruct"

dataset = [
    ("Solve: 3 + 5 =", "8"),
    ("If all birds can fly and penguins are birds, can penguins fly?", "No"),
]

model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
tok = AutoTokenizer.from_pretrained(model_name)
if tok.pad_token_id is None:
    tok.pad_token_id = tok.eos_token_id
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

hook = DANPHook(model, include="all", last_k=0, sigma=SIGMA, base_seed=initial_seed)
for name in hook.R:
    hook.R[name] = hook.R[name].to(device)

# pick on representative layer to inspect
target_name, target_mod = hook._targets[0]
print(f"Inspecting layer: {target_name}, shape: {target_mod.weight.shape}\n")

prompt, target = dataset[0]
inp = tok(prompt, return_tensors="pt").to(device)

print(f"{'Cand':>5} {'seed_offset':>12} {'eps_norm':>10} {'eps_mean':>10} "
      f"{'eps_std':>10} {'delta_a_norm':>10}")

for pop_i in range(POPULATION_SIZE):
    hook.base_seed = initial_seed + pop_i * 31337

    # clean forward
    hook.attach("clean")
    with torch.no_grad():
        model(**inp)
    a_clean = hook._captured_clean.get(target_name, {}).get("a_clean")
    hook.detach()

    # noisy forward
    hook.attach("noisy")
    with torch.no_grad():
        model(**inp)
    a_noisy = hook._captured_noisy.get(target_name, {}).get("a_noisy")
    hook.detach()

    if a_clean is None or a_noisy is None:
        print(f"Skipping candidate {pop_i} due to missing activations")
        continue
    
    eps = a_noisy - a_clean
    # this IS the perturbation in activation space
    print(f"{pop_i:>5} {initial_seed + pop_i*31337:>12} "
          f"{eps.norm().item():>10.6f} {eps.mean().item():>10.6f} "
          f"{eps.std().item():>10.6f} {eps.norm().item():>10.6f}")
    
