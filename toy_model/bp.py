import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

class BPMLP(nn.Module):
    def __init__(self, layer_sizes):
        super().__init__()
        self.layers = nn.ModuleList()
        
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            # ReLU on all layers except output
            if i < len(self.layers) - 1:
                x = torch.relu(x)
        return x