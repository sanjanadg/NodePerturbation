import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

class DANPMLP:
    def __init__(self, layer_sizes, sigma=0.001, eta=1e-3, alpha=1e-4, relu_clip=10.0):
        """
        Args:
            layer_sizes: list of layer dimensions
            sigma: noise std (paper uses σ² = 10⁻⁶, so σ ≈ 0.001)
            eta: weight learning rate
            alpha: decorrelation learning rate
            relu_clip: max value for ReLU outputs to prevent explosion (None = no clipping)
        """
        self.L = len(layer_sizes) - 1
        self.sigma = sigma
        self.eta = eta
        self.alpha = alpha
        self.relu_clip = relu_clip

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
            if l < self.L:
                x_l = torch.relu(a_l)
                if self.relu_clip is not None:
                    x_l = torch.clamp(x_l, max=self.relu_clip)
            else:
                x_l = a_l  # No activation on output
            
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
            if l < self.L:
                x_l = torch.relu(a_l)
                if self.relu_clip is not None:
                    x_l = torch.clamp(x_l, max=self.relu_clip)
            else:
                x_l = a_l  # No activation on output

            a.append(a_l)
            x.append(x_l)

        return x, a

    def compute_grad_estimate(self, x0, target, debug=False):
        """
        Compute gradient estimate for a single sample without applying updates.
        Returns tuple: (weight_gradients, decorrelation_updates, loss_clean)
        
        Args:
            debug: If True, returns additional diagnostics for NaN tracking
        """
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
        
        # Check for very small norm_sq (could cause explosion)
        if norm_sq < 1e-10:
            if debug:
                return None, None, L_clean.item(), {'error': 'norm_sq too small', 'norm_sq': norm_sq.item()}
            # Use a larger epsilon to prevent explosion
            norm_sq = torch.clamp(norm_sq, min=1e-6)

        # Get total number of units N
        N = sum(a.numel() for a in a_clean)
        
        # Compute scale factor and check for potential explosion
        # Note: N is part of the DANP algorithm (Eq. 6 in paper) -> and getting rid of it didn't do anything
        scale_factor = self.eta * N * abs(delta_L) / norm_sq
        
        # Early detection: if delta_L is extremely large, the model is likely diverging
        if abs(delta_L) > 1e6:
            if debug:
                print(f"CRITICAL: delta_L={delta_L:.2e} is extremely large - model may be diverging!")
                print(f"  eta={self.eta:.6f}, N={N}, norm_sq={norm_sq:.6e}, scale_factor={scale_factor:.2e}")
            # Clamp delta_L to prevent explosion
            delta_L = torch.clamp(torch.tensor(delta_L), -1e6, 1e6).item()
            # Recompute scale factor with clamped delta_L
            scale_factor = self.eta * N * abs(delta_L) / norm_sq
        
        # Adaptive clipping: if scale factor is too large, reduce eta to prevent NaN
        # Much more conservative threshold to prevent explosion
        max_scale_factor = 1e3  # Very conservative threshold
        if scale_factor > max_scale_factor:
            # Reduce eta proportionally to keep scale factor bounded
            effective_eta = self.eta * (max_scale_factor / scale_factor)
            if debug:
                print(f"WARNING: Scale factor {scale_factor:.2e} exceeds {max_scale_factor:.0e}")
                print(f"  Using effective_eta={effective_eta:.6e} (reduced from {self.eta:.6f})")
                print(f"  (eta={self.eta:.6f}, N={N}, delta_L={delta_L:.6f}, norm_sq={norm_sq:.6e})")
        else:
            effective_eta = self.eta

        # Weight gradient estimates (Eq. 6 in paper)
        weight_gradients = []
        diagnostics = {}
        for l in range(self.L):
            delta_a = delta_a_all[l]
            x_star = self.R[l] @ x_clean[l]
            
            # Compute gradient
            grad = torch.outer(delta_a, x_star)

            
            # Compute update with adaptive scaling
            update = effective_eta * N * delta_L * grad / norm_sq
            
            # Clip update magnitude to prevent explosion
            update = torch.clamp(update, -1.0, 1.0)
            weight_gradients.append(update)
            
            if debug:
                diagnostics[f'layer_{l}_update_max'] = update.abs().max().item()
                diagnostics[f'layer_{l}_grad_max'] = grad.abs().max().item()
        
        if debug:
            diagnostics['scale_factor'] = scale_factor.item()
            diagnostics['norm_sq'] = norm_sq.item()
            diagnostics['delta_L'] = delta_L
            diagnostics['N'] = N

        # Decorrelation updates (as per Ahmad et al. 2023)
        decorrelation_updates = []
        for l in range(1, self.L + 1):
            x_star = self.R[l] @ x_clean[l]
            cov = torch.outer(x_star, x_star)
            diag = torch.diag(x_star**2)
            
            # Compute decorrelation update
            dec_update = self.alpha * (cov - diag) @ self.R[l]
            
            decorrelation_updates.append(dec_update)
        
        if debug:
            return weight_gradients, decorrelation_updates, L_clean.item(), diagnostics
        return weight_gradients, decorrelation_updates, L_clean.item()

    def step(self, x0, target):
        """Apply updates for a single sample (original implementation)."""
        # Clean forward pass
        x_clean, a_clean = self.forward_clean(x0)

        # Noisy forward pass
        x_noisy, a_noisy = self.forward_noisy(x0)

        # Compute loss difference (Eq. in Algorithm 1)
        L_clean = F.mse_loss(x_clean[-1], target)
        L_noisy = F.mse_loss(x_noisy[-1], target)
        delta_L = (L_noisy - L_clean).item()
        
        # Early detection: if delta_L is extremely large, clamp it to prevent explosion
        # This happens when the model starts diverging - clamp early to prevent NaN
        if abs(delta_L) > 1e4:
            delta_L = max(-1e4, min(1e4, delta_L))  # Clamp to reasonable range

        # Compute total activity difference norm for normalization
        delta_a_all = []
        delta_a_all = []
        for l in range(self.L):
            delta_a_l = a_noisy[l] - a_clean[l] 
            delta_a_all.append(delta_a_l)
        
        # Concatenate and compute norm (as per Eq. 6)
        delta_a_concat = torch.cat([d.flatten() for d in delta_a_all])
        norm_sq = (delta_a_concat**2).sum() + 1e-8
        
        # Check for very small norm_sq (could cause explosion)
        if norm_sq < 1e-10:
            norm_sq = torch.clamp(norm_sq, min=1e-6)

        # Get total number of units N
        N = sum(a.numel() for a in a_clean)
        
        # Compute scale factor and apply adaptive scaling
        # Note: N is part of the DANP algorithm (Eq. 6 in paper)
        scale_factor = self.eta * N * abs(delta_L) / norm_sq
        
        # Early detection: if delta_L is extremely large, the model is likely diverging
        if abs(delta_L) > 1e6:
            # Clamp delta_L to prevent explosion
            delta_L = torch.clamp(torch.tensor(delta_L), -1e6, 1e6).item()
            # Recompute scale factor with clamped delta_L
            scale_factor = self.eta * N * abs(delta_L) / norm_sq
        
        # Adaptive clipping: if scale factor is too large, reduce eta to prevent NaN
        max_scale_factor = 1e3  # Very conservative threshold
        if scale_factor > max_scale_factor:
            effective_eta = self.eta * (max_scale_factor / scale_factor)
        else:
            effective_eta = self.eta

        # Weight updates (Eq. 6 in paper)
        for l in range(self.L):
            delta_a = delta_a_all[l]
            x_star = self.R[l] @ x_clean[l]
            
            # Compute gradient
            grad = torch.outer(delta_a, x_star)
            
            
            # Update with adaptive scaling and gradient clipping
            update = effective_eta * N * delta_L * grad / norm_sq
            
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