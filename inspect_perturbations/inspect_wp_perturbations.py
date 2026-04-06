import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

SIGMA = 0.001
POPULATION_SIZE = 30
initial_seed = 33
model_name = "Qwen/Qwen2.5-3B-Instruct"

model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
name, param = next((n, p) for n, p in model.named_parameters() if "mlp" in n)
print(f"Inspecting param: {name}, shape: {param.shape}\n")

np.random.seed(initial_seed)
seeds = np.random.randint(0, 2**30, POPULATION_SIZE, np.int64).tolist()

print(f"{'Cand':>5} {'seed':>12} {'norm':>10} {'mean':>10} {'std':>10} {'min':>10} {'max':>10}")
for i, seed in enumerate(seeds):
    gen = torch.Generator(device=param.device)
    gen.manual_seed(int(seed))
    noise = torch.randn(param.shape, generator=gen, device=param.device, dtype=param.dtype)
    perturbation =SIGMA*noise
    print(f"{i:>5} {seed:>12} {perturbation.norm().item():>10.6f} "
          f"{perturbation.mean().item():>10.6f} {perturbation.std().item():>10.6f} "
          f"{perturbation.min().item():>10.6f} {perturbation.max().item():>10.6f}")
