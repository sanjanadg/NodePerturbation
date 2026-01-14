import torch
import torch.nn.functional as F

# synthetic teacher task (regression)
# DANP needs a learnable task
def make_teacher(input_dim, hidden_dim, output_dim):
    W1 = torch.randn(hidden_dim, input_dim)/input_dim**0.5
    W2 = torch.randn(output_dim, hidden_dim)/hidden_dim**0.5

    def teacher(x):
        h = torch.relu(W1@x)
        y = W2@h
        return y

    return teacher

# DANP MLP Model
class DANPMLP:

    def __init__(self, layer_sizes, sigma = 0.05, eta = 1e-3, alpha=1e-6):
        self.L = len(layer_sizes) -1
        self.sigma = sigma
        self.eta = eta
        self.alpha = alpha

        self.W = []
        self.R = []

        for l in range(self.L):
            fan_in = layer_sizes[l]
            fan_out = layer_sizes[l+1]

            W = torch.randn(fan_out, fan_in)/fan_in **0.5
            R = torch.eye(fan_out)

            self.W.append(W)
            self.R.append(R)
        
        self.R.insert(0, torch.eye(layer_sizes[0])) # R[0] = I
    
    def forward_clean(self, x0):
        x = [x0]
        a = []

        for l in range(1, self.L + 1):
            x_star = self.R[l-1]@x[l-1]
            a_l = self.W[l-1]@x_star
            x_l = torch.relu(a_l)

            a.append(a_l)
            x.append(x_l)
        
        return x, a

    def forward_noisy(self, x0):
        x = [x0]
        a = []
        eps = []

        for l in range(1, self.L + 1):
            x_star = self.R[l-1]@x[l-1]

            eps_l = torch.randn_like(self.W[l-1]@x_star)*self.sigma
            a_l = self.W[l-1]@x_star + eps_l
            x_l = torch.relu(a_l)

            eps.append(eps_l)
            a.append(a_l)
            x.append(x_l)
        
        return x, a, eps
    
    def step(self, x0, target):
        # Clean forward
        x_clean, a_clean = self.forward_clean(x0)

        # Noisy forward
        x_noisy, a_noisy, eps = self.forward_noisy(x0)

        # Loss difference 
        # just the last layer? is what the alg. calls for MSE?
        L_clean = F.mse_loss(x_clean[-1], target)
        L_noisy = F.mse_loss(x_noisy[-1], target)
        delta_L = (L_noisy - L_clean).item()

        # Weight updates
        for l in range(1, self.L + 1):
            delta_a = a_noisy[l-1] + a_clean[l-1] # = eps[l-1]
            norm = (delta_a**2).sum() + 1e-8  

            x_star = self.R[l-1] @ x_clean[l-1]
            outer = torch.outer(delta_a, x_star)

            # what is numel
            N = delta_a.numel()
            self.W[l-1] -= self.eta*N*delta_L*outer/norm

        # decorrelation updates
        for l in range(1, self.L + 1):
            x_star = self.R[l]@x_clean[l]
            cov = torch.outer(x_star, x_star)
            diag = torch.diag(x_star**2)
            self.R[l] -= self.alpha * (cov-diag)@self.R[l]
        
        return L_clean.item()

# TRAINING!!
torch.manual_seed(0)
input_dim = 10
hidden_dim = 32
output_dim = 1


teacher = make_teacher(input_dim, hidden_dim, output_dim)
model = DANPMLP([input_dim, hidden_dim, output_dim])

for step in range(5000):
    x = torch.randn(input_dim)
    with torch.no_grad():
        y = teacher(x)
    
    loss = model.step(x, y)
    if step%500 == 0:
        print(f"Step{step:5d} | loss {loss: .4f}")



