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
        print("[Device] Using CUDA")
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("[Device] Using MPS (Apple Metal)")
        return torch.device("mps")
    print("[Device] Using CPU")
    return torch.device("cpu")


DEVICE = get_device()


# RealNVP flow on R^D

class MLPConditioner(nn.Module):
    def __init__(self, dim, hidden_dim=512):
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
    def __init__(self, dim, mask, hidden_dim=512, s_scale=0.8):
        super().__init__()
        self.dim = dim
        self.register_buffer("mask", mask.view(1, -1))
        self.conditioner = MLPConditioner(dim, hidden_dim=hidden_dim)
        self.s_scale = s_scale

    def forward(self, x, reverse=False):
        x_masked = x * self.mask
        h = self.conditioner(x_masked)
        s_raw, t = torch.chunk(h, 2, dim=-1)
        s = torch.tanh(s_raw) * self.s_scale

        s = s * (1.0 - self.mask)
        t = t * (1.0 - self.mask)

        if not reverse:
            y = x_masked + (1.0 - self.mask) * (x * torch.exp(s) + t)
            log_det = s.sum(dim=-1)
        else:
            y = x_masked + (1.0 - self.mask) * ((x - t) * torch.exp(-s))
            log_det = -s.sum(dim=-1)
        return y, log_det


class RealNVPFlow(nn.Module):
    def __init__(self, dim, n_couplings=10, hidden_dim=512, s_scale=0.8):
        super().__init__()
        self.dim = dim
        masks = []
        for i in range(n_couplings):
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
        z, log_det = self.x_to_z(x)
        log_pz = -0.5 * torch.sum(z ** 2 + math.log(2 * math.pi), dim=-1)
        return log_pz + log_det

    def sample(self, n, device=None):
        if device is None:
            device = next(self.parameters()).device
        z = torch.randn(n, self.dim, device=device)
        x, _ = self.z_to_x(z)
        return x

# Helpers

def wrap_angles_np(theta: np.ndarray) -> np.ndarray:
    """
    Wrap angles to (-pi, pi] elementwise.
    """
    return ((theta + np.pi) % (2.0 * np.pi)) - np.pi


def train_flow(
    x_data,
    dim,
    n_epochs=300,
    batch_size=256,
    lr=1e-3,
    n_couplings=10,
    hidden_dim=512,
    label="",
):
    x_tensor = torch.from_numpy(x_data).float().to(DEVICE)
    dataset = torch.utils.data.TensorDataset(x_tensor)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=False
    )

    model = RealNVPFlow(dim, n_couplings=n_couplings,
                        hidden_dim=hidden_dim).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    nll_history = []

    print(f"[Train] Training flow on {label} with {len(dataset)} samples...")
    for epoch in range(1, n_epochs + 1):
        model.train()
        total_logp = 0.0
        n_total = 0

        for (batch_x,) in loader:
            optimizer.zero_grad()
            logp = model.log_prob(batch_x)
            loss = -logp.mean()
            loss.backward()
            optimizer.step()

            total_logp += logp.detach().sum().item()
            n_total += batch_x.shape[0]

        mean_logp = total_logp / n_total
        nll = -mean_logp
        nll_history.append(nll)

        if epoch == 1 or epoch % 10 == 0 or epoch == n_epochs:
            print(f"[Train][{label}] Epoch {epoch:4d}/{n_epochs} | NLL ≈ {nll:.4f}")

    return model, np.array(nll_history, dtype=np.float64)


def compute_observables(theta_flat: np.ndarray, L: int):
    """
    theta_flat: (N, 2*L*L) flattened (theta_x, theta_y).
    Returns:
      obs dict
      P_abs array per configuration
    """
    N, D = theta_flat.shape
    assert D == 2 * L * L, f"Dim mismatch: expected {2*L*L}, got {D}"

    theta = torch.from_numpy(theta_flat).float().view(N, 2, L, L)

    P_abs_list = []
    E_list = []

    for n in range(N):
        cfg = theta[n]
        theta_x = cfg[0]
        theta_y = cfg[1]

        theta_x_ip = torch.roll(theta_x, shifts=-1, dims=0)
        theta_y_jp = torch.roll(theta_y, shifts=-1, dims=1)
        theta_p = theta_x + theta_y_jp - theta_x_ip - theta_y
        E_p = 1.0 - torch.cos(theta_p)
        E_list.append(E_p.mean().item())

        poly_angle = theta_x.sum(dim=0)
        cosA = torch.cos(poly_angle)
        sinA = torch.sin(poly_angle)
        P_re = cosA.mean()
        P_im = sinA.mean()
        P_abs = torch.sqrt(P_re * P_re + P_im * P_im)
        P_abs_list.append(P_abs.item())

    E_arr = np.array(E_list, dtype=np.float64)
    P_abs_arr = np.array(P_abs_list, dtype=np.float64)

    obs = {
        "E_plaquette_mean": E_arr.mean(),
        "P_abs_mean": P_abs_arr.mean(),
    }
    return obs, P_abs_arr


# Main experiment: flows on X and X/Λ

def main():
    L = 16
    beta = 2.0

    data_file = f"u1_2d_L{L}_beta{beta:.2f}.npz"
    if not os.path.exists(data_file):
        raise FileNotFoundError(
            f"{data_file} not found. Run experiment_u1_2d_hmc.py first."
        )

    print("=== 2D U(1) gauge: Flows on X and X/Λ ===")
    print(f"[LOAD] {data_file}")
    data = np.load(data_file)
    theta_raw = data["raw"]
    theta_trans = data["trans"]

    N, D = theta_raw.shape
    assert D == 2 * L * L, f"Dim mismatch: expected {2*L*L}, got {D}"
    print(f"[DATA] Loaded {N} configs of dimension {D} (L={L})")

    obs_hmc, P_abs_hmc = compute_observables(theta_raw, L)
    print("\n[HMC] Observables (raw ensemble):")
    for k, v in obs_hmc.items():
        print(f"  {k:16s} = {v:+.6f}")

    dim = D
    n_epochs = 300
    batch_size = 256
    lr = 1e-3
    n_couplings = 10
    hidden_dim = 512

    flow_X, nll_X = train_flow(
        theta_raw,
        dim,
        n_epochs=n_epochs,
        batch_size=batch_size,
        lr=lr,
        n_couplings=n_couplings,
        hidden_dim=hidden_dim,
        label="X (raw)",
    )

    flow_trans, nll_trans = train_flow(
        theta_trans,
        dim,
        n_epochs=n_epochs,
        batch_size=batch_size,
        lr=lr,
        n_couplings=n_couplings,
        hidden_dim=hidden_dim,
        label="X/Λ",
    )

    n_eval = 2000

    with torch.no_grad():
        theta_nf_X = flow_X.sample(n_eval, device=DEVICE).cpu().numpy()
    theta_nf_X = wrap_angles_np(theta_nf_X)

    with torch.no_grad():
        theta_can = flow_trans.sample(n_eval, device=DEVICE).cpu().numpy()
    theta_can = wrap_angles_np(theta_can)

    theta_nf_trans_list = []
    for i in range(n_eval):
        cfg_flat = theta_can[i]
        cfg = cfg_flat.reshape(2, L, L)

        shift_i = np.random.randint(0, L)
        shift_j = np.random.randint(0, L)

        cfg_shift = np.roll(cfg, shift=(-shift_i, -shift_j), axis=(1, 2))
        theta_nf_trans_list.append(cfg_shift.reshape(-1))
    theta_nf_trans = np.stack(theta_nf_trans_list, axis=0)

    obs_X_nf, P_abs_nf_X = compute_observables(theta_nf_X, L)
    obs_trans_nf, P_abs_nf_trans = compute_observables(theta_nf_trans, L)

    print("\n[NF] Observables from trained flows:")
    print("  Flow on X:")
    for k, v in obs_X_nf.items():
        print(f"    {k:16s} = {v:+.6f}")
    print("  Flow on X/Λ (lifted):")
    for k, v in obs_trans_nf.items():
        print(f"    {k:16s} = {v:+.6f}")

    epochs = np.arange(1, n_epochs + 1)
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, nll_X, label="NF on X", lw=2)
    plt.plot(epochs, nll_trans, label="NF on X/Λ", lw=2)

    all_nll = np.concatenate([nll_X, nll_trans])
    y_min, y_max = all_nll.min(), all_nll.max()
    pad = 0.1 * (y_max - y_min + 1e-6)
    plt.ylim(y_min - pad, y_max + pad)

    plt.xlabel("Epoch")
    plt.ylabel("Negative log-likelihood per configuration")
    plt.title(f"2D U(1) (L={L}, beta={beta:.2f}): Training NLL")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"u1_2d_L{L}_beta{beta:.2f}_training_nll.png", dpi=200)
    plt.close()

    out_file = f"u1_2d_L{L}_beta{beta:.2f}_flows_results.npz"
    np.savez(
        out_file,
        nll_X=nll_X,
        nll_trans=nll_trans,
        P_hmc=P_abs_hmc,
        P_nf_X=P_abs_nf_X,
        P_nf_trans=P_abs_nf_trans,
    )
    print(f"\n[SAVE] Stored training curves and |P| samples to {out_file}")


if __name__ == "__main__":
    main()
