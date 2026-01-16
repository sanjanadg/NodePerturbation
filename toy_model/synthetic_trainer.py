import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Synthetic teacher task (regression)
def make_teacher(input_dim, hidden_dim, output_dim):
    W1 = torch.randn(hidden_dim, input_dim) / input_dim**0.5
    W2 = torch.randn(output_dim, hidden_dim) / hidden_dim**0.5

    def teacher(x):
        h = torch.relu(W1 @ x)
        y = W2 @ h
        return y

    return teacher