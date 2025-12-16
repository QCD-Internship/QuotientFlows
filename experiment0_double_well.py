#!/usr/bin/env python3
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance 
# Device selection: CUDA -> MPS (Apple Metal) -> CPU
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS (Apple Metal)")
else:
    device = torch.device("cpu")
    print("Using CPU")
# Target: 1D Double-Well as Mixture of Two Gaussians
LOG_SQRT_2PI = 0.5 * math.log(2.0 * math.pi)

def log_normal(x, mean, log_std):
    """
    Log N(x | mean, exp(log_std)) elementwise.
    mean, log_std can be floats or tensors; we promote to tensors on x.device.
    """
    if not torch.is_tensor(mean):
        mean = torch.tensor(mean, device=x.device, dtype=x.dtype)
    if not torch.is_tensor(log_std):
        log_std = torch.tensor(log_std, device=x.device, dtype=x.dtype)
    var = torch.exp(2.0 * log_std)
    return -0.5 * ((x - mean) ** 2) / var - log_std - LOG_SQRT_2PI

def log_p_double_well(x, m=2.0, sigma=0.5):
    """
    Double well target as mixture:
      p(x) = 0.5 N(x | -m, sigma^2) + 0.5 N(x | +m, sigma^2).
    Returns log p(x) for 1D tensor x.
    """
    x = x.view(-1)
    log_std = math.log(sigma)  # float; log_normal will convert to tensor
    logN1 = log_normal(x, -m, log_std)
    logN2 = log_normal(x,  m, log_std)
    # log[0.5 exp(logN1) + 0.5 exp(logN2)] = logaddexp(logN1, logN2) - log(2)
    log_mix = torch.logaddexp(logN1, logN2) - math.log(2.0)
    return log_mix

def sample_double_well(n, m=2.0, sigma=0.5, torch_device=None):
    """
    Sample from the mixture double well using an explicit mixture.
    """
    if torch_device is None:
        torch_device = device
    with torch.no_grad():
        # mixture component: ±1 with prob 1/2
        signs = torch.randint(0, 2, (n,), device=torch_device) * 2 - 1  # {0,1} -> {-1, +1}
        base = torch.randn(n, device=torch_device) * sigma + m          # N(m, sigma^2)
        x = signs * base
    return x
# Base distribution for flows: standard Normal
base_dist = torch.distributions.Normal(
    torch.tensor(0.0, device=device),
    torch.tensor(1.0, device=device),
)

# Residual Tanh Layers (1D)

class ResidualTanhLayer1D(nn.Module):
    """
    1D residual layer:
      y = x + a * tanh(b * x + c)
    with analytic log|dy/dx|.
    Used for vanilla and quotient flows.
    """
    def __init__(self):
        super().__init__()
        # small initial weights help stability
        self.a = nn.Parameter(torch.zeros(()))
        self.b = nn.Parameter(torch.zeros(()))
        self.c = nn.Parameter(torch.zeros(()))

    def forward(self, x):
        u = self.b * x + self.c
        t = torch.tanh(u)
        y = x + self.a * t
        # dy/dx = 1 + a*b * (1 - tanh(u)^2)
        dydx = 1.0 + self.a * self.b * (1.0 - t**2)
        dydx = torch.clamp(dydx, min=1e-6)
        log_abs_det = torch.log(torch.abs(dydx))
        return y, log_abs_det

class ResidualTanhOddLayer1D(nn.Module):
    """
    1D residual layer constrained to be an odd map:
      y = x + a * tanh(b * x)
    so that y(-x) = -y(x). Used for the equivariant flow.
    """
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.zeros(()))
        self.b = nn.Parameter(torch.zeros(()))

    def forward(self, x):
        u = self.b * x
        t = torch.tanh(u)
        y = x + self.a * t
        dydx = 1.0 + self.a * self.b * (1.0 - t**2)
        dydx = torch.clamp(dydx, min=1e-6)
        log_abs_det = torch.log(torch.abs(dydx))
        return y, log_abs_det
# Vanilla 1D Flow: z -> x (no symmetry constraints)

class VanillaFlow1D(nn.Module):
    def __init__(self, num_layers=8):
        super().__init__()
        self.layers = nn.ModuleList([ResidualTanhLayer1D() for _ in range(num_layers)])

    def forward_flow(self, z):
        """
        Forward map z -> x, returns x, log_abs_det_jacobian.
        """
        x = z
        log_det = torch.zeros_like(z)
        for layer in self.layers:
            x, ldet = layer(x)
            log_det = log_det + ldet
        return x, log_det

    def sample(self, n):
        """
        Sample x ~ q_theta by pushing base z through the flow.
        """
        with torch.no_grad():
            z = base_dist.sample((n,)).to(device)
            x, _ = self.forward_flow(z)
        return x

    def log_q_given_z(self, z, x=None, log_det=None):
        """
        Compute log q_theta(x) given z and log_det from forward_flow.
        If x, log_det not provided, we recompute them.
        """
        if x is None or log_det is None:
            x, log_det = self.forward_flow(z)
        log_q = base_dist.log_prob(z) - log_det
        return x, log_q

# Equivariant Z2 Flow: odd map z -> x, x(-z) = -x(z)
class EquivariantFlow1D(nn.Module):
    def __init__(self, num_layers=8):
        super().__init__()
        self.layers = nn.ModuleList([ResidualTanhOddLayer1D() for _ in range(num_layers)])

    def forward_flow(self, z):
        """
        Forward map z -> x, odd function of z.
        """
        x = z
        log_det = torch.zeros_like(z)
        for layer in self.layers:
            x, ldet = layer(x)
            log_det = log_det + ldet
        return x, log_det

    def sample(self, n):
        with torch.no_grad():
            z = base_dist.sample((n,)).to(device)
            x, _ = self.forward_flow(z)
        return x

    def log_q_given_z(self, z, x=None, log_det=None):
        if x is None or log_det is None:
            x, log_det = self.forward_flow(z)
        log_q = base_dist.log_prob(z) - log_det
        return x, log_q

# Quotient Flow: Flow on |x| >= 0, then random sign lifting
class QuotientFlow1D(nn.Module):
    def __init__(self, num_layers=8):
        super().__init__()
        self.layers = nn.ModuleList([ResidualTanhLayer1D() for _ in range(num_layers)])

    def forward_flow_y(self, z):
        """
        Forward map z -> y >= 0 (on quotient), with log-det for y.
        """
        u = z
        log_det = torch.zeros_like(z)
        for layer in self.layers:
            u, ldet = layer(u)
            log_det = log_det + ldet

        # enforce y >= 0 via softplus
        y = F.softplus(u) + 1e-6  # avoid exactly 0
        # dy/du = sigmoid(u)
        dy_du = torch.sigmoid(u)
        dy_du = torch.clamp(dy_du, min=1e-6)
        log_det_y = log_det + torch.log(dy_du)
        return y, log_det_y

    def sample(self, n):
        """
        Sample x from the lifted distribution:
          1) z -> y >= 0
          2) random sign s in {-1, +1}
          3) x = s * y
        """
        with torch.no_grad():
            z = base_dist.sample((n,)).to(device)
            y, _ = self.forward_flow_y(z)
            signs = torch.randint(0, 2, (n,), device=device) * 2 - 1
            x = signs * y
        return x

    def log_q_X_given_z_and_signs(self, z, signs=None):
        """
        Given z and random signs s, compute:
          x = s * y(z), log q_X(x).
        For Z2 lifting: q_X(x) = (1/2) q_Y(|x|), so
          log q_X(x) = log q_Y(y) - log 2.
        """
        batch = z.shape[0]
        if signs is None:
            signs = torch.randint(0, 2, (batch,), device=device) * 2 - 1

        y, log_det_y = self.forward_flow_y(z)
        x = signs * y

        # density on y (quotient)
        log_q_y = base_dist.log_prob(z) - log_det_y
        # lifted density on X:
        log_q_x = log_q_y - math.log(2.0)
        return x, log_q_x
# Training utilities
def train_flow_reverse_kl(
    model,
    name,
    num_epochs=2000,
    batch_size=4096,
    lr=2e-4,
    clip_grad=True,
    eval_every=20,
    eval_samples=4000,
    target_eval_np=None,
):
    """
    Train a 1D generative flow model using reverse-KL:
      L(theta) = E_z [ log q_theta(x(z)) - log p(x(z)) ],  x(z) from model.
    Also periodically evaluates 1D Wasserstein distance vs target_eval_np.

    Returns:
      model, history where history is a dict with:
        'epoch': list of all epochs
        'loss': list of per-epoch loss values
        'eval_epoch': epochs where W1 was measured
        'wass': corresponding W1 values
    """
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    print(f"\nTraining {name}...")

    history = {
        "epoch": [],
        "loss": [],
        "eval_epoch": [],
        "wass": [],
    }

    for epoch in range(1, num_epochs + 1):
        # Sample base noise
        z = base_dist.sample((batch_size,)).to(device)

        if isinstance(model, QuotientFlow1D):
            # Quotient: z -> y, random sign, x = s*y
            signs = torch.randint(0, 2, (batch_size,), device=device) * 2 - 1
            x, log_q = model.log_q_X_given_z_and_signs(z, signs=signs)
        else:
            # Vanilla / Equivariant: z -> x
            x, log_det = model.forward_flow(z)
            _, log_q = model.log_q_given_z(z, x=x, log_det=log_det)

        # Target log p(x)
        log_p = log_p_double_well(x)
        # clamp for safety (avoid insane exponents)
        log_p = torch.clamp(log_p, min=-1e6, max=1e6)

        # Reverse KL: E[log q - log p]
        loss = (log_q - log_p).mean()

        optimizer.zero_grad()
        loss.backward()

        if clip_grad:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)

        optimizer.step()

        history["epoch"].append(epoch)
        history["loss"].append(float(loss.item()))

        # Print less frequently than every epoch if you want; here every epoch.
        print(f"[{name.lower()}] Epoch {epoch}/{num_epochs} | Loss {loss.item():+.4f}")

        # Periodic Wasserstein evaluation
        if (target_eval_np is not None) and (epoch % eval_every == 0):
            with torch.no_grad():
                S = model.sample(eval_samples).detach().cpu().numpy()
            w = wasserstein_distance(S, target_eval_np)
            history["eval_epoch"].append(epoch)
            history["wass"].append(float(w))
            print(f"  -> {name}: W1(target, model) ≈ {w:.4f}")

    return model, history

# Main experiment
def main():
    torch.manual_seed(1234)
    np.random.seed(1234)

    print("Sampling target distribution...")
    target_samples_t = sample_double_well(10000)
    target_samples = target_samples_t.detach().cpu().numpy()

    # Instantiate models (deeper than before)
    vanilla = VanillaFlow1D(num_layers=8)
    equivariant = EquivariantFlow1D(num_layers=8)
    quotient = QuotientFlow1D(num_layers=8)

    # Train flows with more epochs and larger batch size
    vanilla, hist_v = train_flow_reverse_kl(
        vanilla,
        "Vanilla NF",
        num_epochs=2000,
        batch_size=4096,
        lr=2e-4,
        eval_every=20,
        eval_samples=4000,
        target_eval_np=target_samples,
    )

    equivariant, hist_e = train_flow_reverse_kl(
        equivariant,
        "Equivariant NF",
        num_epochs=2000,
        batch_size=4096,
        lr=2e-4,
        eval_every=20,
        eval_samples=4000,
        target_eval_np=target_samples,
    )

    quotient, hist_q = train_flow_reverse_kl(
        quotient,
        "Quotient NF",
        num_epochs=2000,
        batch_size=4096,
        lr=2e-4,
        eval_every=20,
        eval_samples=4000,
        target_eval_np=target_samples,
    )

    # Final sampling for histograms
    with torch.no_grad():
        S_v = vanilla.sample(5000).detach().cpu().numpy()
        S_e = equivariant.sample(5000).detach().cpu().numpy()
        S_q = quotient.sample(5000).detach().cpu().numpy()

    # Final Wasserstein distances vs target (SciPy)
    w_v = wasserstein_distance(S_v, target_samples)
    w_e = wasserstein_distance(S_e, target_samples)
    w_q = wasserstein_distance(S_q, target_samples)

    print("\n=== Final Wasserstein distances to target (empirical, SciPy) ===")
    print(f"Vanilla NF    : {w_v:.4f}")
    print(f"Equivariant NF: {w_e:.4f}")
    print(f"Quotient NF   : {w_q:.4f}")

    # Plot histograms (target vs final models)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)
    bins = 100

    ax = axes[0, 0]
    ax.hist(target_samples, bins=bins, density=True, alpha=0.8)
    ax.set_title("Target (Double Well)")
    ax.set_ylabel("Density")

    ax = axes[0, 1]
    ax.hist(S_v, bins=bins, density=True, alpha=0.8)
    ax.set_title(f"Vanilla NF (W={w_v:.3f})")

    ax = axes[1, 0]
    ax.hist(S_e, bins=bins, density=True, alpha=0.8)
    ax.set_title(f"Equivariant NF (W={w_e:.3f})")
    ax.set_xlabel("x")
    ax.set_ylabel("Density")

    ax = axes[1, 1]
    ax.hist(S_q, bins=bins, density=True, alpha=0.8)
    ax.set_title(f"Quotient NF (W={w_q:.3f})")
    ax.set_xlabel("x")

    for ax in axes.flatten():
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("experiment0_double_well_histograms.png", dpi=200)
    print("Saved histogram plot to experiment0_double_well_histograms.png")

    # Plot training curves: Wasserstein vs epoch

    plt.figure(figsize=(8, 6))
    if len(hist_v["eval_epoch"]) > 0:
        plt.plot(hist_v["eval_epoch"], hist_v["wass"], label="Vanilla NF")
    if len(hist_e["eval_epoch"]) > 0:
        plt.plot(hist_e["eval_epoch"], hist_e["wass"], label="Equivariant NF")
    if len(hist_q["eval_epoch"]) > 0:
        plt.plot(hist_q["eval_epoch"], hist_q["wass"], label="Quotient NF")

    plt.xlabel("Epoch")
    plt.ylabel("W1(target, model)")
    plt.title("Wasserstein-1 Distance vs Epoch")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("experiment0_double_well_training_curves.png", dpi=200)
    print("Saved training-curve plot to experiment0_double_well_training_curves.png")

if __name__ == "__main__":
    main()
