#!/usr/bin/env python3
"""
experiment_u1_ring_1d_flows.py

Train normalizing flows on 1D U(1) ring configurations:

    - Baseline: NF on the raw angle space X (no quotient).
    - Quotient NF: same architecture, but trained on canonical representatives in X/U(1),
      and sampled by lifting with a random global phase.

Assumes an HMC-generated dataset from `experiment_u1_ring_1d_hmc.py`:

    u1_ring_L{L}_beta{beta:.2f}.npz with keys:
        - "theta_raw": (N, L) raw samples
        - "theta_can": (N, L) canonical reps

Outputs:
    - Training NLL curves
    - Histograms of |M| for HMC, vanilla NF, quotient NF
"""

import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt


# ----------------------
#  Helpers / observables
# ----------------------

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = get_device()
torch.manual_seed(1234)
np.random.seed(1234)


def angle_wrap_torch(x: torch.Tensor) -> torch.Tensor:
    """
    Wrap angles to (-pi, pi] in PyTorch.
    """
    return (x + math.pi) % (2.0 * math.pi) - math.pi


def magnetisation_complex_torch(theta: torch.Tensor) -> torch.Tensor:
    """
    Complex magnetisation M = (1/L) sum_j e^{i theta_j}.

    Args:
        theta: tensor shape (N, L)

    Returns:
        M: tensor shape (N,), complex dtype
    """
    # convert to complex: e^{i theta}
    U = torch.exp(1j * theta)
    return U.mean(dim=-1)


def energy_density_torch(theta: torch.Tensor, beta: float) -> torch.Tensor:
    """
    Energy density e = S / L for a batch of configurations.

    Args:
        theta: (N, L)
        beta:  float

    Returns:
        e: (N,)
    """
    diff = torch.roll(theta, shifts=-1, dims=1) - theta  # (N, L)
    S = -beta * torch.cos(diff).sum(dim=1)
    L = theta.shape[1]
    return S / L


def compute_observables_torch(theta: torch.Tensor, beta: float):
    """
    Simple observables averaged over a batch.

    Args:
        theta: (N, L)
        beta:  float

    Returns:
        dict with scalar floats
    """
    with torch.no_grad():
        M = magnetisation_complex_torch(theta)  # (N,)
        M_abs_mean = M.abs().mean().item()
        E_mean = energy_density_torch(theta, beta).mean().item()
    return {
        "M_abs_mean": M_abs_mean,
        "E_mean": E_mean,
    }


# ----------------------
#  RealNVP building blocks
# ----------------------

class AffineCoupling1D(nn.Module):
    """
    Affine RealNVP coupling layer for 1D vectors with circular Conv1d conditioner.

    We use a binary mask m \in {0,1}^L. For input x:

        x1 = m * x
        x2 = (1-m) * x

        [log_s, t] = NN(x1)  # shape (..., 2*L), we only use on unmasked dims
        y1 = x1
        y2 = x2 * exp(log_s) + t

    The network is implemented with Conv1d with circular padding to respect
    translation structure on the ring.
    """

    def __init__(self, dim: int, hidden_channels: int, mask: torch.Tensor):
        """
        Args:
            dim:             lattice size L
            hidden_channels: width of hidden Conv1d layers
            mask:            1D tensor of shape (dim,), {0,1}, float
        """
        super().__init__()
        self.dim = dim
        self.register_buffer("mask", mask.view(1, dim))  # (1, L)

        self.net = nn.Sequential(
            nn.Conv1d(1, hidden_channels, kernel_size=3, padding=1, padding_mode="circular"),
            nn.ReLU(),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1, padding_mode="circular"),
            nn.ReLU(),
            nn.Conv1d(hidden_channels, 2, kernel_size=3, padding=1, padding_mode="circular"),
        )

    def forward(self, x: torch.Tensor):
        """
        Forward transform.

        Args:
            x: (N, L)

        Returns:
            y:         (N, L)
            log_detJ:  (N,)
        """
        m = self.mask  # (1, L)
        x1 = m * x
        x1_net = x1.unsqueeze(1)  # (N, 1, L)
        h = self.net(x1_net)      # (N, 2, L)
        log_s, t = h[:, 0, :], h[:, 1, :]
        log_s = torch.tanh(log_s)  # squash for stability
        s = torch.exp(log_s)

        y = x1 + (1.0 - m) * (x * s + t)
        log_detJ = ((1.0 - m) * log_s).sum(dim=1)
        return y, log_detJ

    def inverse(self, y: torch.Tensor):
        """
        Inverse transform.

        Args:
            y: (N, L)

        Returns:
            x:         (N, L)
            log_detJ:  (N,)
        """
        m = self.mask
        y1 = m * y
        y1_net = y1.unsqueeze(1)  # (N,1,L)
        h = self.net(y1_net)
        log_s, t = h[:, 0, :], h[:, 1, :]
        log_s = torch.tanh(log_s)
        s = torch.exp(log_s)

        # x = (y2 - t) / s on unmasked dims
        x2 = (1.0 - m) * ((y - t) / (s + 1e-8))
        x = y1 + x2
        log_detJ = -((1.0 - m) * log_s).sum(dim=1)
        return x, log_detJ


class RealNVPFlow1D(nn.Module):
    """
    Multi-layer RealNVP flow for 1D vectors, with alternating masks.
    """

    def __init__(self, dim: int, n_coupling_layers: int = 6, hidden_channels: int = 64):
        super().__init__()
        self.dim = dim
        self.n_coupling_layers = n_coupling_layers

        masks = []
        for i in range(n_coupling_layers):
            if i % 2 == 0:
                # mask evens
                mask_np = np.zeros(dim, dtype=np.float32)
                mask_np[::2] = 1.0
            else:
                # mask odds
                mask_np = np.zeros(dim, dtype=np.float32)
                mask_np[1::2] = 1.0
            masks.append(torch.from_numpy(mask_np))

        self.coupling_layers = nn.ModuleList(
            [
                AffineCoupling1D(dim, hidden_channels, mask=m)
                for m in masks
            ]
        )

        # base distribution: standard normal N(0, I)
        self.register_buffer("base_mean", torch.zeros(dim))
        self.register_buffer("base_log_std", torch.zeros(dim))

    def f(self, x: torch.Tensor):
        """
        Forward mapping x -> z, accumulating log_detJ.
        """
        log_det_total = torch.zeros(x.shape[0], device=x.device)
        z = x
        for layer in self.coupling_layers:
            z, log_det = layer(z)
            log_det_total += log_det
        return z, log_det_total

    def f_inv(self, z: torch.Tensor):
        """
        Inverse mapping z -> x, accumulating log_detJ.
        """
        log_det_total = torch.zeros(z.shape[0], device=z.device)
        x = z
        for layer in reversed(self.coupling_layers):
            x, log_det = layer.inverse(x)
            log_det_total += log_det
        return x, log_det_total

    def log_prob(self, x: torch.Tensor):
        """
        Log-density log q(x) via change of variables from base N(0,I).
        """
        z, log_det = self.f(x)
        mean = self.base_mean
        log_std = self.base_log_std
        std = torch.exp(log_std)

        log_base = -0.5 * (((z - mean) / std) ** 2).sum(dim=1) \
                   - 0.5 * self.dim * math.log(2.0 * math.pi) \
                   - log_std.sum()
        return log_base + log_det

    def sample(self, n: int):
        """
        Sample x ~ q(x) by drawing z ~ N(0,I) and applying inverse flow.
        """
        mean = self.base_mean
        log_std = self.base_log_std
        std = torch.exp(log_std)
        z = mean + std * torch.randn(n, self.dim, device=mean.device)
        x, _ = self.f_inv(z)
        return x


# ----------------------
#  Training utilities
# ----------------------

def make_dataloader(theta: np.ndarray, batch_size: int, shuffle: bool = True) -> DataLoader:
    """
    Turn a numpy array (N, L) into a PyTorch DataLoader on DEVICE.
    """
    theta_t = torch.from_numpy(theta).float()
    dataset = TensorDataset(theta_t)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def train_flow(
    flow: RealNVPFlow1D,
    dataloader: DataLoader,
    n_epochs: int = 100,
    lr: float = 1e-3,
    print_every: int = 10,
):
    """
    Train a RealNVPFlow1D by maximising log-likelihood.

    Returns:
        nll_history: list of mean NLL per epoch
    """
    flow.to(DEVICE)
    flow.train()
    optimizer = torch.optim.Adam(flow.parameters(), lr=lr)

    nll_history = []
    for epoch in range(1, n_epochs + 1):
        total_nll = 0.0
        total_count = 0

        for (batch_theta,) in dataloader:
            batch_theta = batch_theta.to(DEVICE)
            # For angles, we can optionally wrap, but HMC already gives (-pi, pi]
            batch_theta = angle_wrap_torch(batch_theta)

            log_q = flow.log_prob(batch_theta)  # (B,)
            loss = -log_q.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_nll += loss.item() * batch_theta.shape[0]
            total_count += batch_theta.shape[0]

        mean_nll = total_nll / total_count
        nll_history.append(mean_nll)

        if epoch % print_every == 0 or epoch == 1 or epoch == n_epochs:
            print(f"[Train] Epoch {epoch:4d}/{n_epochs} | NLL ≈ {mean_nll:.4f}")

    return nll_history


# ----------------------
#  Main experiment
# ----------------------

def main():
    # Match this to the HMC file
    L = 32
    beta = 2.0
    data_file = f"u1_ring_L{L}_beta{beta:.2f}.npz"

    print("=== U(1) 1D Ring: Flows on X vs X/U(1) ===")
    print(f"Loading HMC ensemble from {data_file}")

    data = np.load(data_file)
    theta_raw = data["theta_raw"]   # (N, L)
    theta_can = data["theta_can"]   # (N, L)
    N = theta_raw.shape[0]
    print(f"Loaded {N} samples, L={theta_raw.shape[1]}")

    batch_size = 128
    n_epochs = 200
    lr = 1e-3

    dl_raw = make_dataloader(theta_raw, batch_size=batch_size, shuffle=True)
    dl_can = make_dataloader(theta_can, batch_size=batch_size, shuffle=True)

    # --- Flow on full X (baseline) ---
    flow_raw = RealNVPFlow1D(dim=L, n_coupling_layers=6, hidden_channels=64)
    print("\n[Baseline] Training NF on raw angles (full X)...")
    nll_raw = train_flow(flow_raw, dl_raw, n_epochs=n_epochs, lr=lr, print_every=10)

    # --- Flow on quotient X/U(1) (canonical reps) ---
    flow_q = RealNVPFlow1D(dim=L, n_coupling_layers=6, hidden_channels=64)
    print("\n[Quotient] Training NF on canonical reps (X/U(1))...")
    nll_q = train_flow(flow_q, dl_can, n_epochs=n_epochs, lr=lr, print_every=10)

    # --- Plot NLL curves ---
    plt.figure()
    plt.plot(nll_raw, label="NF on X (raw)")
    plt.plot(nll_q, label="NF on X/U(1) (canonical)")
    plt.xlabel("Epoch")
    plt.ylabel("NLL (mean per config)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("u1_ring_training_nll.png", dpi=150)
    plt.close()
    print("[PLOT] Saved training NLL curves to u1_ring_training_nll.png")

    # --- Sampling and lifting ---

    n_eval = 2000
    print(f"\nSampling {n_eval} configs from each flow...")

    flow_raw.eval()
    flow_q.eval()

    with torch.no_grad():
        # Baseline NF on X
        theta_raw_nf = flow_raw.sample(n_eval)        # (n_eval, L)
        theta_raw_nf = angle_wrap_torch(theta_raw_nf).to(torch.complex64)

        # Quotient NF: sample canonical reps, then lift with random global phase
        theta_q_can = flow_q.sample(n_eval)           # canonical in X/U(1)
        theta_q_can = angle_wrap_torch(theta_q_can)

        # Sample random global phases alpha ~ Uniform(-pi, pi]
        alpha = (2.0 * math.pi * torch.rand(n_eval, 1, device=DEVICE) - math.pi)
        theta_q = angle_wrap_torch(theta_q_can + alpha)

        theta_raw_nf = theta_raw_nf.to(torch.float32)
        theta_q = theta_q.to(torch.float32)

    # Convert HMC samples to torch
    theta_hmc = torch.from_numpy(theta_raw).float().to(DEVICE)

    # --- Observables comparison ---

    obs_hmc = compute_observables_torch(theta_hmc, beta)
    obs_nf_raw = compute_observables_torch(theta_raw_nf, beta)
    obs_nf_q = compute_observables_torch(theta_q, beta)

    print("\n=== Observables comparison ===")
    print("HMC      :", obs_hmc)
    print("NF (X)   :", obs_nf_raw)
    print("NF (X/G) :", obs_nf_q)

    # --- Magnetisation histograms |M| ---

    with torch.no_grad():
        M_hmc = magnetisation_complex_torch(theta_hmc).abs().cpu().numpy()
        M_nf_raw = magnetisation_complex_torch(theta_raw_nf).abs().cpu().numpy()
        M_nf_q = magnetisation_complex_torch(theta_q).abs().cpu().numpy()

    plt.figure()
    bins = 40
    plt.hist(M_hmc, bins=bins, density=True, alpha=0.5, label="HMC")
    plt.hist(M_nf_raw, bins=bins, density=True, alpha=0.5, label="NF on X")
    plt.hist(M_nf_q, bins=bins, density=True, alpha=0.5, label="NF on X/U(1) lifted")
    plt.xlabel(r"$|M|$")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig("u1_ring_Mabs_hist.png", dpi=150)
    plt.close()
    print("[PLOT] Saved |M| histograms to u1_ring_Mabs_hist.png")

        # --- Save data for external plotting scripts ---
    np.savez(
        f"u1_ring_L{L}_beta{beta:.2f}_flows_results.npz",
        nll_raw=np.array(nll_raw),
        nll_q=np.array(nll_q),
        M_hmc=M_hmc,
        M_nf_raw=M_nf_raw,
        M_nf_q=M_nf_q,
    )
    print(
        f"[SAVE] Stored training curves and |M| samples to "
        f"u1_ring_L{L}_beta{beta:.2f}_flows_results.npz"
    )



if __name__ == "__main__":
    main()
