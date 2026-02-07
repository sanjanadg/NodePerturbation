import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

class BPMLP(nn.Module):
    def __init__(self, layer_sizes):
        super().__init__()
        self.layers = nn.ModuleList()
        
        for i in range(len(layer_sizes) - 1):
            layer = nn.Linear(layer_sizes[i], layer_sizes[i+1])
            # Initialize weights using Xavier initialization to match DANP/WP
            # This ensures deterministic initialization when torch.manual_seed is set
            # Uses the same pattern as DANP/WP: torch.randn(...) / fan_in**0.5
            fan_in = layer_sizes[i]
            with torch.no_grad():
                layer.weight.data = torch.randn(layer_sizes[i+1], layer_sizes[i]) / fan_in**0.5
                if layer.bias is not None:
                    layer.bias.data.zero_()
            self.layers.append(layer)
            
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            # ReLU on all layers except output
            if i < len(self.layers) - 1:
                x = torch.relu(x)
        return x