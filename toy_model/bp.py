import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from synthetic_trainer import make_teacher

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


# TRAINING
def train_bp():
    torch.manual_seed(42)
    input_dim = 10
    hidden_dim = 32
    output_dim = 1

    teacher = make_teacher(input_dim, hidden_dim, output_dim)
    
    # Try different optimizers
    configs = [
        {"optimizer": "SGD", "lr": 1e-2, "name": "BP (SGD)"},
        {"optimizer": "Adam", "lr": 1e-3, "name": "BP (Adam)"},
    ]
    
    for config in configs:
        print(f"\n{'='*60}")
        print(f"Config: {config['name']}")
        print(f"optimizer={config['optimizer']}, lr={config['lr']}")
        print(f"{'='*60}")
        
        model = BPMLP([input_dim, hidden_dim, output_dim])
        
        # Create optimizer
        if config['optimizer'] == "SGD":
            optimizer = torch.optim.SGD(model.parameters(), lr=config['lr'])
        elif config['optimizer'] == "Adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
        
        losses = []
        for step in range(5000):
            x = torch.randn(input_dim)
            with torch.no_grad():
                y = teacher(x)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            pred = model(x)
            
            # Compute loss
            loss = F.mse_loss(pred, y)
            
            # Backward pass
            loss.backward()
            
            # Update weights
            optimizer.step()
            
            # Track loss
            losses.append(loss.item())
            
            if step % 500 == 0:
                avg_loss = sum(losses[-100:]) / min(len(losses), 100)
                print(f"Step {step:5d} | loss {loss.item():8.4f} | avg_loss {avg_loss:8.4f}")
        
        final_avg = sum(losses[-100:]) / 100
        print(f"Final average loss: {final_avg:.6f}")


if __name__ == "__main__":
    train_bp()