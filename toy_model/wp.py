import torch
import torch.nn as nn
import torch.nn.functional as F

class WPMLP:
    """
    Weight Perturbation (WP) MLP implementation.
    
    WP directly perturbs the weights rather than the node activities. 
    This implementation follows the algorithm from wp_llm_gsm8k.py.
    
    Single-sample algorithm (step method):
    1. Perturbs weights with noise: W' = W + sigma * epsilon
    2. Evaluates loss with perturbed weights
    3. Updates weights: W += -alpha * (L_noisy - L_clean) * epsilon
    
    Population-based algorithm (step_population method, matches wp_llm_gsm8k.py):
    1. Evaluate multiple perturbations (population_size)
    2. Convert losses to rewards: reward = -loss (lower loss = higher reward)
    3. Normalize rewards: (reward - mean) / std
    4. Update: W += alpha * mean(normalized_reward * noise)
    
    Multiple weight perturbations are evaluated and the best ones are used to update the weights.
    """
    
    def __init__(self, layer_sizes, sigma=0.001, alpha=1e-3):
        """
        Args:
            layer_sizes: list of layer dimensions [input_dim, hidden1, ..., hiddenN, output_dim]
            sigma: noise standard deviation for weight perturbations
            alpha: learning rate for weight updates
        """
        self.L = len(layer_sizes) - 1
        self.sigma = sigma
        self.alpha = alpha
        
        # Initialize weights
        self.W = []
        for l in range(self.L):
            fan_in = layer_sizes[l]
            fan_out = layer_sizes[l + 1]
            
            # Xavier initialization
            W = torch.randn(fan_out, fan_in) / fan_in**0.5
            self.W.append(W)
    
    def forward(self, x0):
        """
        Forward pass through the network.
        
        Args:
            x0: input tensor of shape (batch_size, input_dim) or (input_dim,)
        
        Returns:
            x: list of activations at each layer [x0, x1, ..., x_L]
        """
        x = [x0]
        
        for l in range(self.L):
            a_l = self.W[l] @ x[l]
            # ReLU activation on all layers except output
            x_l = torch.relu(a_l) if l < self.L - 1 else a_l
            x.append(x_l)
        
        return x
    
    def forward_with_perturbed_weights(self, x0, noise_list):
        """
        Forward pass with perturbed weights.
        
        Args:
            x0: input tensor
            noise_list: list of noise tensors, one per weight matrix
        
        Returns:
            x: list of activations at each layer
        """
        x = [x0]
        
        for l in range(self.L):
            # Apply perturbed weights: W_perturbed = W + sigma * noise
            W_perturbed = self.W[l] + self.sigma * noise_list[l]
            a_l = W_perturbed @ x[l]
            # ReLU activation on all layers except output
            x_l = torch.relu(a_l) if l < self.L - 1 else a_l
            x.append(x_l)
        
        return x
    
    def compute_grad_estimate(self, x0, target):
        """
        Compute gradient estimate using weight perturbation for a single sample.
        
        This method:
        1. Evaluates loss with clean weights
        2. Generates noise and evaluates loss with perturbed weights
        3. Computes weight update based on loss difference
        
        Args:
            x0: input tensor
            target: target tensor
        
        Returns:
            weight_gradients: list of gradient estimates for each weight matrix
            loss_clean: loss with clean weights
        """
        # Forward pass with clean weights
        x_clean = self.forward(x0)
        loss_clean = F.mse_loss(x_clean[-1], target)
        
        # Generate noise for each weight matrix
        noise_list = []
        for l in range(self.L):
            noise = torch.randn_like(self.W[l])
            noise_list.append(noise)
        
        # Forward pass with perturbed weights
        x_noisy = self.forward_with_perturbed_weights(x0, noise_list)
        loss_noisy = F.mse_loss(x_noisy[-1], target)
        
        # Compute loss difference
        delta_L = (loss_noisy - loss_clean).item()
        
        # Compute weight gradients: gradient = -alpha * delta_L * noise
        # The negative sign is because we want to decrease loss
        weight_gradients = []
        for l in range(self.L):
            grad = -self.alpha * delta_L * noise_list[l]
            # Clip gradients for stability
            grad = torch.clamp(grad, -1.0, 1.0)
            weight_gradients.append(grad)
        
        return weight_gradients, loss_clean.item()
    
    def step(self, x0, target):
        """
        Apply weight perturbation update for a single sample.
        
        This is a simplified version for single-sample updates:
        - If loss_noisy < loss_clean (better), move toward noise (positive update)
        - If loss_noisy > loss_clean (worse), move away from noise (negative update)
        Update: W += -alpha * (loss_noisy - loss_clean) * noise
        
        For population-based updates matching wp_llm_gsm8k.py, use step_population().
        
        Args:
            x0: input tensor
            target: target tensor
        
        Returns:
            loss_clean: loss with clean weights before update
            delta_L: loss difference (noisy - clean)
        """
        # Forward pass with clean weights
        x_clean = self.forward(x0)
        loss_clean = F.mse_loss(x_clean[-1], target)
        
        # Generate noise for each weight matrix
        noise_list = []
        for l in range(self.L):
            noise = torch.randn_like(self.W[l])
            noise_list.append(noise)
        
        # Forward pass with perturbed weights
        x_noisy = self.forward_with_perturbed_weights(x0, noise_list)
        loss_noisy = F.mse_loss(x_noisy[-1], target)
        
        # Compute loss difference
        delta_L = (loss_noisy - loss_clean).item()
        
        # Update weights: W = W - alpha * delta_L * noise
        # The negative sign is because we want to decrease loss
        for l in range(self.L):
            update = -self.alpha * delta_L * noise_list[l]
            # Clip updates for stability
            update = torch.clamp(update, -1.0, 1.0)
            self.W[l] += update
        
        return loss_clean.item(), delta_L
    
    def step_population(self, x0, target, population_size=10):
        """
        Apply weight perturbation update using a population of perturbations.
        
        This follows the algorithm from wp_llm_gsm8k.py:
        1. Evaluate each perturbation and compute loss
        2. Convert losses to rewards (negative loss, since lower loss = higher reward)
        3. Normalize rewards: (reward - mean) / std
        4. Update: W += alpha * mean(normalized_reward * noise)
        
        This is similar to evolutionary strategies (ES) where multiple perturbations
        are evaluated and the best ones are used to update weights. This can be more
        stable than single-sample updates.
        
        Args:
            x0: input tensor
            target: target tensor
            population_size: number of perturbations to evaluate
        
        Returns:
            loss_clean: loss with clean weights before update
            mean_delta_L: average loss difference across population
        """
        # Forward pass with clean weights
        x_clean = self.forward(x0)
        loss_clean = F.mse_loss(x_clean[-1], target)
        
        # Evaluate population of perturbations
        losses = []
        noise_population = []
        
        for _ in range(population_size):
            # Generate noise for each weight matrix
            noise_list = []
            for l in range(self.L):
                noise = torch.randn_like(self.W[l])
                noise_list.append(noise)
            noise_population.append(noise_list)
            
            # Forward pass with perturbed weights
            x_noisy = self.forward_with_perturbed_weights(x0, noise_list)
            loss_noisy = F.mse_loss(x_noisy[-1], target)
            losses.append(loss_noisy.item())
        
        # Convert losses to rewards (negative loss, since lower loss = higher reward)
        # This matches wp_llm_gsm8k.py which uses rewards
        rewards = [-loss for loss in losses]
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
        
        # Normalize rewards: (reward - mean) / std (matching wp_llm_gsm8k.py line 407)
        rewards_normalized = (rewards_tensor - rewards_tensor.mean()) / (rewards_tensor.std() + 1e-8)
        
        # Compute update: mean(normalized_reward * noise) for each layer
        # This matches wp_llm_gsm8k.py lines 411-428
        for l in range(self.L):
            update = torch.zeros_like(self.W[l])
            for i in range(population_size):
                r_norm = rewards_normalized[i].item()
                update += r_norm * noise_population[i][l]
            update = update / population_size
            
            # Apply update: W += alpha * update (matching wp_llm_gsm8k.py line 428)
            # No negative sign needed because we're using rewards (higher is better)
            self.W[l] += self.alpha * update
        
        # Compute mean delta for reporting (loss_noisy - loss_clean)
        mean_delta_L = sum([loss - loss_clean.item() for loss in losses]) / population_size
        return loss_clean.item(), mean_delta_L


def test_wp_learning():
    """
    Test function to verify WP model learns on a synthetic dataset.
    Uses architecture [10, 32, 1] for both dataset and model.
    """
    import sys
    import os
    sys.path.insert(0, os.path.dirname(__file__))
    from sythentic_data import SyntheticDataset
    
    print("="*80)
    print("WP LEARNING TEST")
    print("="*80)
    
    # Create synthetic dataset with architecture [10, 32, 1]
    # Dataset uses its own seed internally, so we set it first
    layer_sizes = [10, 32, 1]
    print(f"\nCreating synthetic dataset with teacher architecture: {layer_sizes}")
    dataset = SyntheticDataset(
        input_dim=layer_sizes[0],
        hidden_dim=layer_sizes[1],
        output_dim=layer_sizes[-1],
        n_samples=1000,
        seed=42,
        teacher_layer_sizes=layer_sizes
    )
    print(f"Dataset size: {len(dataset)}")
    
    # Check dataset output statistics
    sample_targets = torch.stack([dataset[i][1] for i in range(min(100, len(dataset)))])
    print(f"Target statistics: mean={sample_targets.mean().item():.4f}, std={sample_targets.std().item():.4f}, "
          f"min={sample_targets.min().item():.4f}, max={sample_targets.max().item():.4f}")
    
    # Set seed for model initialization AFTER dataset is created
    # This ensures model has different initialization from teacher
    torch.manual_seed(1042)
    
    # Create WP model with same architecture
    print(f"\nCreating WPMLP model with architecture: {layer_sizes}")
    model = WPMLP(
        layer_sizes=layer_sizes,
        sigma=0.001,
        alpha=1e-3
    )
    print(f"Model created with sigma={model.sigma}, alpha={model.alpha}")
    
    # Check initial model predictions
    sample_inputs = torch.stack([dataset[i][0] for i in range(min(100, len(dataset)))])
    with torch.no_grad():
        initial_preds = []
        for i in range(min(100, len(dataset))):
            x, _ = dataset[i]
            pred = model.forward(x)[-1]
            initial_preds.append(pred)
        initial_preds = torch.stack(initial_preds)
        print(f"Initial prediction statistics: mean={initial_preds.mean().item():.4f}, "
              f"std={initial_preds.std().item():.4f}, min={initial_preds.min().item():.4f}, "
              f"max={initial_preds.max().item():.4f}")
    
    # Compute initial loss on a sample
    sample_x, sample_y = dataset[0]
    initial_loss_sample = F.mse_loss(model.forward(sample_x)[-1], sample_y).item()
    print(f"Initial loss on first sample: {initial_loss_sample:.6f}")
    
    # Training parameters
    n_epochs = 20
    train_split = 0.8
    train_size = int(train_split * len(dataset))
    
    # Split dataset
    train_indices = list(range(train_size))
    test_indices = list(range(train_size, len(dataset)))
    
    print(f"\nTraining for {n_epochs} epochs...")
    print(f"Train samples: {len(train_indices)}, Test samples: {len(test_indices)}")
    print("-"*80)
    
    # Training loop
    train_losses = []
    test_losses = []
    
    for epoch in range(n_epochs):
        # Training
        epoch_train_loss = 0.0
        for idx in train_indices:
            x, y = dataset[idx]
            loss, delta_L = model.step(x, y)
            epoch_train_loss += loss
        
        avg_train_loss = epoch_train_loss / len(train_indices)
        train_losses.append(avg_train_loss)
        
        # Evaluation on test set
        epoch_test_loss = 0.0
        with torch.no_grad():
            for idx in test_indices:
                x, y = dataset[idx]
                x_forward = model.forward(x)
                loss = F.mse_loss(x_forward[-1], y).item()
                epoch_test_loss += loss
        
        avg_test_loss = epoch_test_loss / len(test_indices)
        test_losses.append(avg_test_loss)
        
        # Print progress every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{n_epochs} | Train Loss: {avg_train_loss:.6f} | Test Loss: {avg_test_loss:.6f}")
    
    print("-"*80)
    print(f"\nInitial Train Loss: {train_losses[0]:.6f}")
    print(f"Final Train Loss:   {train_losses[-1]:.6f}")
    print(f"Initial Test Loss:  {test_losses[0]:.6f}")
    print(f"Final Test Loss:    {test_losses[-1]:.6f}")
    
    # Check if model learned
    train_improvement = train_losses[0] - train_losses[-1]
    test_improvement = test_losses[0] - test_losses[-1]
    
    print(f"\nTrain Loss Reduction: {train_improvement:.6f} ({train_improvement/train_losses[0]*100:.2f}%)")
    print(f"Test Loss Reduction:  {test_improvement:.6f} ({test_improvement/test_losses[0]*100:.2f}%)")
    
    if train_improvement > 0 and test_improvement > 0:
        print("\nSUCCESS: Model is learning! Loss decreased over training.")
    else:
        print("\nWARNING: Model may not be learning. Loss did not decrease.")
    
    print("="*80)
    
    return {
        'train_losses': train_losses,
        'test_losses': test_losses,
        'initial_train': train_losses[0],
        'final_train': train_losses[-1],
        'initial_test': test_losses[0],
        'final_test': test_losses[-1]
    }


if __name__ == '__main__':
    test_wp_learning()

