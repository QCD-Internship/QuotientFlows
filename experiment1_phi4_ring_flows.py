#!/usr/bin/env python3
import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Device


def get_device():
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        print("[Device] Using CUDA")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        dev = torch.device("mps")
        print("[Device] Using MPS (Apple Metal)")
    else:
        dev = torch.device("cpu")
        print("[Device] Using CPU")
    return dev


DEVICE = get_device()


# RealNVP components for R^L


class MLPConditioner(nn.Module):
    """Simple fully-connected conditioner: R^dim -> R^{2 dim}."""
    def __init__(self, dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * dim),
        )

    def forward(self, x):
        return self.net(x)


class RealNVPCoupling(nn.Module):
    """Mask-based RealNVP coupling layer in R^dim with shared conditioner."""
    def __init__(self, dim, mask, hidden_dim=128, s_scale=0.8):
        super().__init__()
        self.dim = dim
        self.register_buffer("mask", mask.view(1, -1))  # (1, dim)
        self.conditioner = MLPConditioner(dim, hidden_dim=hidden_dim)
        self.s_scale = s_scale

    def forward(self, x, reverse=False):
        """
        Args:
            x: (B, dim)
            reverse: if True, apply inverse transform.

        Returns:
            y: (B, dim)
            log_det: (B,)
        """
        x_masked = x * self.mask
        h = self.conditioner(x_masked)
        s_raw, t = torch.chunk(h, 2, dim=-1)
        s = torch.tanh(s_raw) * self.s_scale

        # Only act on unmasked components
        s = s * (1.0 - self.mask)
        t = t * (1.0 - self.mask)

        if not reverse:
            y = x_masked + (1.0 - self.mask) * (x * torch.exp(s) + t)
            log_det = s.sum(dim=-1)
        else:
            y = x_masked + (1.0 - self.mask) * ((x - t) * torch.exp(-s))
            log_det = -s.sum(dim=-1)
        return y, log_det


class RealNVPFlow1D(nn.Module):
    """RealNVP flow on R^dim with standard Gaussian base."""
    def __init__(self, dim, n_couplings=8, hidden_dim=128, s_scale=0.8):
        super().__init__()
        self.dim = dim
        masks = []
        for i in range(n_couplings):
            # Alternating 0101... / 1010... masks
            m = (torch.arange(dim) % 2 == (i % 2)).float()
            masks.append(m)
        self.couplings = nn.ModuleList(
            [RealNVPCoupling(dim, m, hidden_dim=hidden_dim, s_scale=s_scale)
             for m in masks]
        )

    def x_to_z(self, x):
        log_det = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        z = x
        for cp in self.couplings:
            z, ld = cp(z, reverse=False)
            log_det = log_det + ld
        return z, log_det

    def z_to_x(self, z):
        log_det = torch.zeros(z.shape[0], device=z.device, dtype=z.dtype)
        x = z
        for cp in reversed(self.couplings):
            x, ld = cp(x, reverse=True)
            log_det = log_det + ld
        return x, log_det

    def log_prob(self, x):
        """Log p(x) under the flow model."""
        z, log_det = self.x_to_z(x)
        log_pz = -0.5 * torch.sum(z ** 2 + math.log(2 * math.pi), dim=-1)
        return log_pz + log_det

    def sample(self, n, device=None):
        if device is None:
            device = next(self.parameters()).device
        z = torch.randn(n, self.dim, device=device)
        x, _ = self.z_to_x(z)
        return x


# Training utilities

def train_flow(x_data, dim, n_epochs=200, batch_size=128, lr=1e-3,
               n_couplings=8, hidden_dim=128, label=""):
    """
    Train a RealNVP flow on given data.

    x_data: numpy array (N, dim)
    Returns: (model, nll_history)
    """
    x_tensor = torch.from_numpy(x_data).float().to(DEVICE)
    dataset = torch.utils.data.TensorDataset(x_tensor)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=False
    )

    model = RealNVPFlow1D(dim, n_couplings=n_couplings,
                          hidden_dim=hidden_dim).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    nll_history = []

    print(f"[Train] Training flow for {label} with {len(dataset)} samples...")
    for epoch in range(1, n_epochs + 1):
        model.train()
        epoch_logprob = 0.0
        n_total = 0

        for (batch_x,) in loader:
            optimizer.zero_grad()
            logp = model.log_prob(batch_x)
            loss = -logp.mean()
            loss.backward()
            optimizer.step()

            bs = batch_x.shape[0]
            epoch_logprob += logp.detach().sum().item()
            n_total += bs

        mean_logp = epoch_logprob / n_total
        nll = -mean_logp
        nll_history.append(nll)

        if epoch == 1 or epoch % 10 == 0 or epoch == n_epochs:
            print(f"[Train][{label}] Epoch {epoch:4d}/{n_epochs} | NLL ≈ {nll:.4f}")

    return model, np.array(nll_history, dtype=np.float64)


def compute_observables(phi_np: np.ndarray):
    """Compute φ², φ⁴, M, |M|, Binder U4 for numpy array (N, L)."""
    phi = torch.from_numpy(phi_np).float()
    phi2 = (phi ** 2).mean(dim=1)
    phi4 = (phi ** 4).mean(dim=1)
    M = phi.mean(dim=1)
    M2 = M ** 2
    M4 = M ** 4

    obs = {
        "<phi^2>": phi2.mean().item(),
        "<phi^4>": phi4.mean().item(),
        "<M>": M.mean().item(),
        "<|M|>": M.abs().mean().item(),
        "U4": (1.0 - M4.mean().item() / (3.0 * (M2.mean().item() ** 2 + 1e-12))),
    }
    M_abs = M.abs().numpy()
    return obs, M_abs

# Main experiment

def main():
    # Match the HMC script parameters / filenames
    L = 32
    m2 = -0.5
    lam = 3.0

    data_file = f"phi4_ring_1d_L{L}_m2{m2:.2f}_lam{lam:.2f}.npz"
    if not os.path.exists(data_file):
        raise FileNotFoundError(
            f"Could not find {data_file}. Run experiment1_phi4_ring_hmc.py first."
        )

    print("=== 1D phi^4 ring: Flows on X, X/Z2, X/(Z2×C_L) ===")
    print(f"[LOAD] {data_file}")
    data = np.load(data_file)
    phi_raw = data["raw"]   # (N, L)
    phi_Z2 = data["Z2"]
    phi_Z2C = data["Z2C"]

    N, L_check = phi_raw.shape
    assert L_check == L, f"L mismatch: expected {L}, got {L_check}"
    print(f"[DATA] Loaded {N} configurations of length L={L}")

    # HMC observables (raw ensemble)
    obs_hmc, M_abs_hmc = compute_observables(phi_raw)
    print("\n[HMC] Observables (raw ensemble):")
    for k, v in obs_hmc.items():
        print(f"  {k:8s} = {v:+.6f}")

    dim = L
    n_epochs = 200
    batch_size = 128
    lr = 1e-3
    n_couplings = 8
    hidden_dim = 128

    # Baseline: flow on full X

    flow_X, nll_X = train_flow(
        phi_raw, dim,
        n_epochs=n_epochs,
        batch_size=batch_size,
        lr=lr,
        n_couplings=n_couplings,
        hidden_dim=hidden_dim,
        label="X (raw)",
    )

    # Quotient: flow on X/Z2

    flow_Z2, nll_Z2 = train_flow(
        phi_Z2, dim,
        n_epochs=n_epochs,
        batch_size=batch_size,
        lr=lr,
        n_couplings=n_couplings,
        hidden_dim=hidden_dim,
        label="X/Z2",
    )
    # Quotient: flow on X/(Z2×C_L)
    flow_Z2C, nll_Z2C = train_flow(
        phi_Z2C, dim,
        n_epochs=n_epochs,
        batch_size=batch_size,
        lr=lr,
        n_couplings=n_couplings,
        hidden_dim=hidden_dim,
        label="X/(Z2×C_L)",
    )

    # Draw samples from each flow and lift to full X
    n_eval = 2000

    # Baseline flow on X already lives in X: no lifting
    with torch.no_grad():
        phi_X_nf = flow_X.sample(n_eval, device=DEVICE).cpu().numpy()

    # Flow on X/Z2: apply random global sign
    with torch.no_grad():
        phi_Z2_can = flow_Z2.sample(n_eval, device=DEVICE)
    signs = torch.randint(0, 2, (n_eval, 1), device=DEVICE) * 2 - 1  # ±1
    phi_Z2_full = (phi_Z2_can * signs).cpu().numpy()

    # Flow on X/(Z2×C_L): apply random sign and random translation
    with torch.no_grad():
        phi_Z2C_can = flow_Z2C.sample(n_eval, device=DEVICE).cpu()  # (n_eval, L)

    phi_Z2C_full_list = []
    for i in range(n_eval):
        cfg = phi_Z2C_can[i]  # (L,)
        sign = 1.0 if np.random.rand() < 0.5 else -1.0
        shift = np.random.randint(0, L)
        cfg2 = torch.roll(cfg * sign, shifts=shift, dims=0)
        phi_Z2C_full_list.append(cfg2.numpy())
    phi_Z2C_full = np.stack(phi_Z2C_full_list, axis=0)

    # Observables for NF ensembles
    obs_X_nf, M_abs_nf_X = compute_observables(phi_X_nf)
    obs_Z2_nf, M_abs_nf_Z2 = compute_observables(phi_Z2_full)
    obs_Z2C_nf, M_abs_nf_Z2C = compute_observables(phi_Z2C_full)

    print("\n[NF] Observables from trained flows:")
    print("  Flow on X:")
    for k, v in obs_X_nf.items():
        print(f"    {k:8s} = {v:+.6f}")
    print("  Flow on X/Z2 (lifted):")
    for k, v in obs_Z2_nf.items():
        print(f"    {k:8s} = {v:+.6f}")
    print("  Flow on X/(Z2×C_L) (lifted):")
    for k, v in obs_Z2C_nf.items():
        print(f"    {k:8s} = {v:+.6f}")

    # Quick inline NLL plot + save arrays for plotting script
    epochs = np.arange(1, n_epochs + 1)
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, nll_X, label="NF on X", lw=2)
    plt.plot(epochs, nll_Z2, label="NF on X/Z2", lw=2)
    plt.plot(epochs, nll_Z2C, label="NF on X/(Z2×C_L)", lw=2)
    plt.xlabel("Epoch")
    plt.ylabel("NLL per configuration")
    plt.title(f"1D phi^4 (L={L}, m2={m2}, lam={lam}): Training NLL")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        f"phi4_ring_1d_L{L}_m2{m2:.2f}_lam{lam:.2f}_training_nll.png",
        dpi=200,
    )
    plt.close()

    # Save data for external plotting script
    out_file = f"phi4_ring_1d_L{L}_m2{m2:.2f}_lam{lam:.2f}_flows_results.npz"
    np.savez(
        out_file,
        nll_X=nll_X,
        nll_Z2=nll_Z2,
        nll_Z2C=nll_Z2C,
        M_hmc=M_abs_hmc,
        M_nf_X=M_abs_nf_X,
        M_nf_Z2=M_abs_nf_Z2,
        M_nf_Z2C=M_abs_nf_Z2C,
    )
    print(f"\n[SAVE] Stored training curves and |M| samples to {out_file}")


if __name__ == "__main__":
    main()
