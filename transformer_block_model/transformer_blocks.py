import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TransformerBlock(nn.Module):
    """
    Regular Transformer Block (for teacher network).
    Architecture: LayerNorm -> Self-Attention -> Residual -> LayerNorm -> FFN -> Residual
    """
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        
        # Self-attention
        self.attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Self-attention with residual
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        
        # Feed-forward with residual
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        
        return x

class BPTransformerBlock(nn.Module):
    """
    Standard Transformer Block using Backpropagation.
    Architecture: LayerNorm -> Self-Attention -> Residual -> LayerNorm -> FFN -> Residual
    """
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        
        # Self-attention
        self.attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Self-attention with residual
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        
        # Feed-forward with residual
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        
        return x


class DANPTransformerBlock:
    """
    Transformer Block using Decorrelated Activity-based Node Perturbation (DANP).
    Implements the same architecture as BPTransformerBlock but uses DANP for updates.
    """
    def __init__(self, d_model, n_heads, d_ff, sigma=0.001, eta=1e-3, alpha=1e-4, dropout=0.1):
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.sigma = sigma
        self.eta = eta
        self.alpha = alpha
        
        # Initialize attention weights (Q, K, V projections and output)
        # W_q, W_k, W_v: [n_heads, d_k, d_model] where d_k = d_model // n_heads
        # This allows: x @ W_q[h].T where x is [batch, seq_len, d_model] -> [batch, seq_len, d_k]
        d_k = d_model // n_heads
        self.W_q = torch.randn(n_heads, d_k, d_model) / d_model**0.5
        self.W_k = torch.randn(n_heads, d_k, d_model) / d_model**0.5
        self.W_v = torch.randn(n_heads, d_k, d_model) / d_model**0.5
        self.W_o = torch.randn(d_model, d_model) / d_model**0.5
        
        # Layer norm parameters (learnable)
        self.norm1_gamma = torch.ones(d_model)
        self.norm1_beta = torch.zeros(d_model)
        self.norm2_gamma = torch.ones(d_model)
        self.norm2_beta = torch.zeros(d_model)
        
        # Feed-forward network weights
        self.W_ffn1 = torch.randn(d_ff, d_model) / d_model**0.5
        self.W_ffn2 = torch.randn(d_model, d_ff) / d_ff**0.5
        
        # Decorrelation matrices (one for each layer that needs decorrelation)
        # R[0]: input to attention
        # R[1]: attention output to FFN
        self.R = [torch.eye(d_model), torch.eye(d_model)]
        
    def layer_norm(self, x, gamma, beta):
        """Layer normalization"""
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + 1e-8)
        return gamma * x_norm + beta
    
    def scaled_dot_product_attention(self, Q, K, V):
        """Scaled dot-product attention"""
        d_k = Q.shape[-1]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        attn_weights = F.softmax(scores, dim=-1)
        attn_out = torch.matmul(attn_weights, V)
        return attn_out
    
    def multi_head_attention(self, x):
        """Multi-head self-attention"""
        batch_size, seq_len, d_model = x.shape
        
        # Project to Q, K, V for each head
        Q = torch.stack([x @ self.W_q[h].T for h in range(self.n_heads)], dim=1)  # [B, H, L, d_k]
        K = torch.stack([x @ self.W_k[h].T for h in range(self.n_heads)], dim=1)
        V = torch.stack([x @ self.W_v[h].T for h in range(self.n_heads)], dim=1)
        
        # Apply attention for each head
        attn_heads = torch.stack([
            self.scaled_dot_product_attention(Q[:, h], K[:, h], V[:, h])
            for h in range(self.n_heads)
        ], dim=1)  # [B, H, L, d_k]
        
        # Concatenate heads
        attn_concat = attn_heads.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        
        # Output projection
        attn_out = attn_concat @ self.W_o.T
        
        return attn_out
    
    def multi_head_attention_before_output(self, x):
        """Multi-head self-attention without output projection (returns attn_concat)"""
        batch_size, seq_len, d_model = x.shape
        
        # Project to Q, K, V for each head
        Q = torch.stack([x @ self.W_q[h].T for h in range(self.n_heads)], dim=1)  # [B, H, L, d_k]
        K = torch.stack([x @ self.W_k[h].T for h in range(self.n_heads)], dim=1)
        V = torch.stack([x @ self.W_v[h].T for h in range(self.n_heads)], dim=1)
        
        # Apply attention for each head
        attn_heads = torch.stack([
            self.scaled_dot_product_attention(Q[:, h], K[:, h], V[:, h])
            for h in range(self.n_heads)
        ], dim=1)  # [B, H, L, d_k]
        
        # Concatenate heads
        attn_concat = attn_heads.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        
        return attn_concat
    
    def forward_clean(self, x0):
        """Clean forward pass"""
        x = [x0]
        a = []
        batch_size, seq_len, d_model = x0.shape
        
        # Self-attention block
        x_norm1 = self.layer_norm(x[0], self.norm1_gamma, self.norm1_beta)
        # Apply decorrelation: reshape to [batch*seq, d_model], apply R, reshape back
        x_norm1_flat = x_norm1.view(-1, d_model)  # [batch*seq, d_model]
        x_star1_flat = x_norm1_flat @ self.R[0].T  # [batch*seq, d_model]
        x_star1 = x_star1_flat.view(batch_size, seq_len, d_model)  # [batch, seq, d_model]
        attn_out = self.multi_head_attention(x_star1)
        a.append(attn_out)
        x_attn = x[0] + attn_out  # Residual
        x.append(x_attn)
        
        # Feed-forward block
        x_norm2 = self.layer_norm(x[1], self.norm2_gamma, self.norm2_beta)
        # Apply decorrelation: reshape to [batch*seq, d_model], apply R, reshape back
        x_norm2_flat = x_norm2.view(-1, d_model)  # [batch*seq, d_model]
        x_star2_flat = x_norm2_flat @ self.R[1].T  # [batch*seq, d_model]
        x_star2 = x_star2_flat.view(batch_size, seq_len, d_model)  # [batch, seq, d_model]
        ffn_hidden = F.relu(x_star2 @ self.W_ffn1.T)
        ffn_out = ffn_hidden @ self.W_ffn2.T
        a.append(ffn_out)
        x_final = x[1] + ffn_out  # Residual
        x.append(x_final)
        
        return x, a
    
    def forward_noisy(self, x0):
        """Noisy forward pass for perturbation"""
        x = [x0]
        a = []
        batch_size, seq_len, d_model = x0.shape
        
        # Self-attention block with noise
        x_norm1 = self.layer_norm(x[0], self.norm1_gamma, self.norm1_beta)
        # Apply decorrelation: reshape to [batch*seq, d_model], apply R, reshape back
        x_norm1_flat = x_norm1.view(-1, d_model)  # [batch*seq, d_model]
        x_star1_flat = x_norm1_flat @ self.R[0].T  # [batch*seq, d_model]
        x_star1 = x_star1_flat.view(batch_size, seq_len, d_model)  # [batch, seq, d_model]
        
        # Add noise to attention output
        eps_attn = torch.randn_like(self.multi_head_attention(x_star1)) * self.sigma
        attn_out = self.multi_head_attention(x_star1) + eps_attn
        a.append(attn_out)
        x_attn = x[0] + attn_out
        x.append(x_attn)
        
        # Feed-forward block with noise
        x_norm2 = self.layer_norm(x[1], self.norm2_gamma, self.norm2_beta)
        # Apply decorrelation: reshape to [batch*seq, d_model], apply R, reshape back
        x_norm2_flat = x_norm2.view(-1, d_model)  # [batch*seq, d_model]
        x_star2_flat = x_norm2_flat @ self.R[1].T  # [batch*seq, d_model]
        x_star2 = x_star2_flat.view(batch_size, seq_len, d_model)  # [batch, seq, d_model]
        ffn_hidden = F.relu(x_star2 @ self.W_ffn1.T)
        eps_ffn = torch.randn_like(ffn_hidden @ self.W_ffn2.T) * self.sigma
        ffn_out = ffn_hidden @ self.W_ffn2.T + eps_ffn
        a.append(ffn_out)
        x_final = x[1] + ffn_out
        x.append(x_final)
        
        return x, a
    
    def compute_grad_estimate(self, x0, target):
        """
        Compute gradient estimate for a single sample without applying updates.
        Returns tuple: (weight_gradients_dict, decorrelation_updates, loss_clean)
        """
        # Clean forward pass
        x_clean, a_clean = self.forward_clean(x0)
        
        # Noisy forward pass
        x_noisy, a_noisy = self.forward_noisy(x0)
        
        # Compute loss difference
        L_clean = F.mse_loss(x_clean[-1], target)
        L_noisy = F.mse_loss(x_noisy[-1], target)
        delta_L = (L_noisy - L_clean).item()
        
        # Compute activity differences
        delta_a_attn = a_noisy[0] - a_clean[0]
        delta_a_ffn = a_noisy[1] - a_clean[1]
        
        # Concatenate and compute norm
        delta_a_concat = torch.cat([delta_a_attn.flatten(), delta_a_ffn.flatten()])
        norm_sq = (delta_a_concat**2).sum() + 1e-8
        
        # Get total number of units N
        N = sum(a.numel() for a in a_clean)
        
        # Compute gradients for attention weights
        batch_size, seq_len, d_model = x_clean[0].shape
        x_norm1 = self.layer_norm(x_clean[0], self.norm1_gamma, self.norm1_beta)
        # Apply decorrelation: reshape to [batch*seq, d_model], apply R, reshape back
        x_norm1_flat = x_norm1.view(-1, d_model)  # [batch*seq, d_model]
        x_star1_flat = x_norm1_flat @ self.R[0].T  # [batch*seq, d_model]
        x_star1 = x_star1_flat.view(batch_size, seq_len, d_model)  # [batch, seq, d_model]
        
        # Gradient estimates for attention
        # For W_o: attn_out = attn_concat @ W_o.T, so grad_W_o = sum(delta_a_attn^T @ attn_concat)
        batch_size, seq_len, d_model = delta_a_attn.shape
        d_k = d_model // self.n_heads
        
        # Get the attention concatenated output (before W_o projection)
        attn_concat_clean = self.multi_head_attention_before_output(x_star1)
        
        # Gradient for W_o: sum over batch and seq dimensions
        # delta_a_attn: [batch, seq_len, d_model], attn_concat: [batch, seq_len, d_model]
        # For each position, compute outer product and sum
        grad_W_o = torch.zeros_like(self.W_o)
        for b in range(batch_size):
            for s in range(seq_len):
                grad_W_o += torch.outer(delta_a_attn[b, s], attn_concat_clean[b, s])
        
        # For Q, K, V: simplified approximation using mean
        grad_W_q = [torch.zeros_like(self.W_q[h]) for h in range(self.n_heads)]
        grad_W_k = [torch.zeros_like(self.W_k[h]) for h in range(self.n_heads)]
        grad_W_v = [torch.zeros_like(self.W_v[h]) for h in range(self.n_heads)]
        
        # Use mean across batch/seq for Q, K, V gradients (simplified)
        delta_a_mean = delta_a_attn.mean(dim=(0, 1))  # [d_model]
        x_star1_mean = x_star1.mean(dim=(0, 1))  # [d_model]
        for h in range(self.n_heads):
            # Approximate gradient: outer product of means, then reshape
            grad_flat = torch.outer(delta_a_mean, x_star1_mean)  # [d_model, d_model]
            # Take appropriate slice for this head (simplified)
            grad_W_q[h] = grad_flat[:d_k, :].view_as(self.W_q[h])
            grad_W_k[h] = grad_W_q[h]  # Simplified
            grad_W_v[h] = grad_W_q[h]  # Simplified
        
        # Compute gradients for FFN
        x_norm2 = self.layer_norm(x_clean[1], self.norm2_gamma, self.norm2_beta)
        # Apply decorrelation: reshape to [batch*seq, d_model], apply R, reshape back
        x_norm2_flat = x_norm2.view(-1, d_model)  # [batch*seq, d_model]
        x_star2_flat = x_norm2_flat @ self.R[1].T  # [batch*seq, d_model]
        x_star2 = x_star2_flat.view(batch_size, seq_len, d_model)  # [batch, seq, d_model]
        ffn_hidden_clean = F.relu(x_star2 @ self.W_ffn1.T)
        
        # Gradient for FFN: sum over batch and seq dimensions
        # delta_a_ffn: [batch, seq_len, d_model], ffn_hidden: [batch, seq_len, d_ff]
        grad_W_ffn2 = torch.zeros_like(self.W_ffn2)
        for b in range(batch_size):
            for s in range(seq_len):
                grad_W_ffn2 += torch.outer(delta_a_ffn[b, s], ffn_hidden_clean[b, s])
        
        # For W_ffn1, we need gradient through ReLU and W_ffn2
        # Simplified: use the gradient propagated through W_ffn2
        grad_W_ffn1 = torch.zeros_like(self.W_ffn1)
        for b in range(batch_size):
            for s in range(seq_len):
                # Gradient through W_ffn2: delta_a_ffn @ W_ffn2 gives gradient w.r.t. ffn_hidden
                grad_hidden = delta_a_ffn[b, s] @ self.W_ffn2  # [d_ff]
                # Apply ReLU derivative (1 where ffn_hidden > 0, else 0)
                relu_mask = (ffn_hidden_clean[b, s] > 0).float()
                grad_hidden_masked = grad_hidden * relu_mask
                grad_W_ffn1 += torch.outer(grad_hidden_masked, x_star2[b, s])
        
        # Apply scaling and clipping
        scale = self.eta * N * delta_L / norm_sq
        
        weight_gradients = {
            'W_q': [torch.clamp(scale * g, -1.0, 1.0) for g in grad_W_q],
            'W_k': [torch.clamp(scale * g, -1.0, 1.0) for g in grad_W_k],
            'W_v': [torch.clamp(scale * g, -1.0, 1.0) for g in grad_W_v],
            'W_o': torch.clamp(scale * grad_W_o, -1.0, 1.0),
            'W_ffn1': torch.clamp(scale * grad_W_ffn1, -1.0, 1.0),
            'W_ffn2': torch.clamp(scale * grad_W_ffn2, -1.0, 1.0),
        }
        
        # Decorrelation updates
        decorrelation_updates = []
        for l in range(len(self.R)):
            if l == 0:
                # Use x_star1 that was already computed
                x_star_flat = x_star1.view(-1, d_model)  # [batch*seq, d_model]
            else:
                # Use x_star2 that was already computed
                x_star_flat = x_star2.view(-1, d_model)  # [batch*seq, d_model]
            
            # Compute decorrelation update
            # For each feature vector, compute covariance and update R
            # Simplified: use mean across batch*seq for decorrelation
            x_star_mean = x_star_flat.mean(dim=0)  # [d_model]
            cov = torch.outer(x_star_mean, x_star_mean)
            diag = torch.diag(x_star_mean**2)
            dec_update = self.alpha * (cov - diag) @ self.R[l]
            decorrelation_updates.append(dec_update)
        
        return weight_gradients, decorrelation_updates, L_clean.item()
    
    def step(self, x0, target):
        """Apply updates for a single sample"""
        weight_gradients, decorrelation_updates, loss_clean = self.compute_grad_estimate(x0, target)
        
        # Apply weight updates
        for h in range(self.n_heads):
            self.W_q[h] -= weight_gradients['W_q'][h]
            self.W_k[h] -= weight_gradients['W_k'][h]
            self.W_v[h] -= weight_gradients['W_v'][h]
        self.W_o -= weight_gradients['W_o']
        self.W_ffn1 -= weight_gradients['W_ffn1']
        self.W_ffn2 -= weight_gradients['W_ffn2']
        
        # Apply decorrelation updates
        for l in range(len(self.R)):
            self.R[l] -= decorrelation_updates[l]
        
        return loss_clean, 0.0

