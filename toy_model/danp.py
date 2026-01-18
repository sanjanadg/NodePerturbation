import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

class DANPMLP:
    def __init__(self, layer_sizes, sigma=0.001, eta=1e-3, alpha=1e-4):
        """
        Args:
            layer_sizes: list of layer dimensions
            sigma: noise std (paper uses σ² = 10⁻⁶, so σ ≈ 0.001)
            eta: weight learning rate
            alpha: decorrelation learning rate
        """
        self.L = len(layer_sizes) - 1
        self.sigma = sigma
        self.eta = eta
        self.alpha = alpha

        self.W = []
        self.R = []

        # Initialize weights and decorrelation matrices
        for l in range(self.L):
            fan_in = layer_sizes[l]
            fan_out = layer_sizes[l + 1]

            W = torch.randn(fan_out, fan_in) / fan_in**0.5
            R = torch.eye(fan_out)

            self.W.append(W)
            self.R.append(R)

        # R[0] for input layer
        self.R.insert(0, torch.eye(layer_sizes[0]))

    def forward_clean(self, x0):
        x = [x0]
        a = []

        for l in range(1, self.L + 1):
            x_star = self.R[l - 1] @ x[l - 1]
            a_l = self.W[l - 1] @ x_star
            x_l = torch.relu(a_l) if l < self.L else a_l  # No activation on output
            
            a.append(a_l)
            x.append(x_l)

        return x, a

    def forward_noisy(self, x0):
        x = [x0]
        a = []

        for l in range(1, self.L + 1):
            x_star = self.R[l - 1] @ x[l - 1]
            
            # Sample noise
            eps_l = torch.randn_like(self.W[l - 1] @ x_star) * self.sigma
            a_l = self.W[l - 1] @ x_star + eps_l
            x_l = torch.relu(a_l) if l < self.L else a_l  # No activation on output

            a.append(a_l)
            x.append(x_l)

        return x, a

    def step(self, x0, target):
        # Clean forward pass
        x_clean, a_clean = self.forward_clean(x0)

        # Noisy forward pass
        x_noisy, a_noisy = self.forward_noisy(x0)

        # Compute loss difference (Eq. in Algorithm 1)
        L_clean = F.mse_loss(x_clean[-1], target)
        L_noisy = F.mse_loss(x_noisy[-1], target)
        delta_L = (L_noisy - L_clean).item()

        # Compute total activity difference norm for normalization
        delta_a_all = []
        for l in range(self.L):
            delta_a_l = a_noisy[l] - a_clean[l] 
            delta_a_all.append(delta_a_l)
        
        # Concatenate and compute norm (as per Eq. 6)
        delta_a_concat = torch.cat([d.flatten() for d in delta_a_all])
        norm_sq = (delta_a_concat**2).sum() + 1e-8

        # Get total number of units N
        N = sum(a.numel() for a in a_clean)

        # Weight updates (Eq. 6 in paper)
        for l in range(self.L):
            delta_a = delta_a_all[l]
            x_star = self.R[l] @ x_clean[l]
            
            # Compute gradient
            grad = torch.outer(delta_a, x_star)
            
            # Update with gradient clipping
            update = self.eta * N * delta_L * grad / norm_sq
            update = torch.clamp(update, -1.0, 1.0)  # Clip for stability
            
            self.W[l] -= update

        # Decorrelation updates (as per Ahmad et al. 2023)
        for l in range(1, self.L + 1):
            x_star = self.R[l] @ x_clean[l]
            cov = torch.outer(x_star, x_star)
            diag = torch.diag(x_star**2)
            
            # Update decorrelation matrix
            dec_update = self.alpha * (cov - diag) @ self.R[l]
            self.R[l] -= dec_update

        return L_clean.item(), delta_L