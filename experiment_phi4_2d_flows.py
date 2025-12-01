#!/usr/bin/env python3
import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# ============================================================
# Device
# ============================================================

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

# ============================================================
# RealNVP on R^D
# ============================================================

class MLPConditioner(nn.Module):
    def __init__(self, dim, hidden_dim=256):
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
    def __init__(self, dim, mask, hidden_dim=256, s_scale=0.8):
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
    def __init__(self, dim, n_couplings=8, hidden_dim=256, s_scale=0.8):
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


# ============================================================
# Training helpers
# ============================================================

def train_flow(
    x_data,
    dim,
    n_epochs=200,
    batch_size=128,
    lr=1e-3,
    n_couplings=8,
    hidden_dim=256,
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


def compute_observables(phi_np: np.ndarray):
    """
    phi_np: (N, L^2) flattened.
    Returns obs dict and |M| array.
    """
    phi = torch.from_numpy(phi_np).float()
    N, D = phi.shape
    L2 = D
    # M per configuration: average over all sites
    M = phi.mean(dim=1)
    M2 = M ** 2
    M4 = M ** 4

    phi2 = (phi ** 2).mean(dim=1)
    phi4 = (phi ** 4).mean(dim=1)

    obs = {
        "<phi^2>": phi2.mean().item(),
        "<phi^4>": phi4.mean().item(),
        "<M>": M.mean().item(),
        "<|M|>": M.abs().mean().item(),
        "U4": (1.0 - M4.mean().item() / (3.0 * (M2.mean().item() ** 2 + 1e-12))),
    }
    M_abs = M.abs().numpy()
    return obs, M_abs


# ============================================================
# Main experiment
# ============================================================

def main():
    # Match HMC script
    L = 16
    m2 = -0.5
    lam = 3.0

    data_file = f"phi4_2d_L{L}_m2{m2:.2f}_lam{lam:.2f}.npz"
    if not os.path.exists(data_file):
        raise FileNotFoundError(
            f"{data_file} not found. Run experiment_phi4_2d_hmc.py first."
        )

    print("=== 2D phi^4 (LxL): Flows on X, X/Z2, X/(Z2×Λ) ===")
    print(f"[LOAD] {data_file}")
    data = np.load(data_file)
    phi_raw = data["raw"]   # (N, L^2)
    phi_Z2 = data["Z2"]
    phi_Z2T = data["Z2T"]

    N, D = phi_raw.shape
    assert D == L * L, f"Dim mismatch: expected {L*L}, got {D}"
    print(f"[DATA] Loaded {N} configs of dimension {D} (L={L})")

    # HMC observables (raw)
    obs_hmc, M_abs_hmc = compute_observables(phi_raw)
    print("\n[HMC] Observables (raw ensemble):")
    for k, v in obs_hmc.items():
        print(f"  {k:8s} = {v:+.6f}")

    dim = D
    n_epochs = 200
    batch_size = 128
    lr = 1e-3
    n_couplings = 8
    hidden_dim = 256

    # ------------------------------
    # Flow on full X
    # ------------------------------
    flow_X, nll_X = train_flow(
        phi_raw, dim,
        n_epochs=n_epochs,
        batch_size=batch_size,
        lr=lr,
        n_couplings=n_couplings,
        hidden_dim=hidden_dim,
        label="X (raw)",
    )

    # ------------------------------
    # Flow on X/Z2
    # ------------------------------
    flow_Z2, nll_Z2 = train_flow(
        phi_Z2, dim,
        n_epochs=n_epochs,
        batch_size=batch_size,
        lr=lr,
        n_couplings=n_couplings,
        hidden_dim=hidden_dim,
        label="X/Z2",
    )

    # ------------------------------
    # Flow on X/(Z2×Λ)
    # ------------------------------
    flow_Z2T, nll_Z2T = train_flow(
        phi_Z2T, dim,
        n_epochs=n_epochs,
        batch_size=batch_size,
        lr=lr,
        n_couplings=n_couplings,
        hidden_dim=hidden_dim,
        label="X/(Z2×Λ)",
    )

    # ------------------------------------------------
    # Sample from each flow and lift to full X
    # ------------------------------------------------
    n_eval = 2000

    # 1) Flow on X: already full space
    with torch.no_grad():
        phi_nf_X = flow_X.sample(n_eval, device=DEVICE).cpu().numpy()

    # 2) Flow on X/Z2: apply random global sign
    with torch.no_grad():
        phi_can_Z2 = flow_Z2.sample(n_eval, device=DEVICE)  # (n_eval, D)
    signs = torch.randint(0, 2, (n_eval, 1), device=DEVICE) * 2 - 1  # ±1
    phi_nf_Z2 = (phi_can_Z2 * signs).cpu().numpy()

    # 3) Flow on X/(Z2×Λ): random sign + random 2D shift
    with torch.no_grad():
        phi_can_Z2T = flow_Z2T.sample(n_eval, device=DEVICE).cpu()  # (n_eval, D)

    phi_nf_Z2T_list = []
    for i in range(n_eval):
        cfg_flat = phi_can_Z2T[i]  # (D,)
        cfg = cfg_flat.view(L, L)

        sign = 1.0 if np.random.rand() < 0.5 else -1.0
        shift_x = np.random.randint(0, L)
        shift_y = np.random.randint(0, L)

        cfg2 = cfg * sign
        cfg2 = torch.roll(cfg2, shifts=(shift_y, shift_x), dims=(0, 1))  # y,x
        phi_nf_Z2T_list.append(cfg2.view(-1).numpy())

    phi_nf_Z2T = np.stack(phi_nf_Z2T_list, axis=0)

    # ------------------------------------------------
    # Observables for NF samples
    # ------------------------------------------------
    obs_X_nf, M_abs_nf_X = compute_observables(phi_nf_X)
    obs_Z2_nf, M_abs_nf_Z2 = compute_observables(phi_nf_Z2)
    obs_Z2T_nf, M_abs_nf_Z2T = compute_observables(phi_nf_Z2T)

    print("\n[NF] Observables from trained flows:")
    print("  Flow on X:")
    for k, v in obs_X_nf.items():
        print(f"    {k:8s} = {v:+.6f}")
    print("  Flow on X/Z2 (lifted):")
    for k, v in obs_Z2_nf.items():
        print(f"    {k:8s} = {v:+.6f}")
    print("  Flow on X/(Z2×Λ) (lifted):")
    for k, v in obs_Z2T_nf.items():
        print(f"    {k:8s} = {v:+.6f}")

    # Quick sanity NLL plot
    epochs = np.arange(1, n_epochs + 1)
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, nll_X, label="NF on X", lw=2)
    plt.plot(epochs, nll_Z2, label="NF on X/Z2", lw=2)
    plt.plot(epochs, nll_Z2T, label="NF on X/(Z2×Λ)", lw=2)
    plt.xlabel("Epoch")
    plt.ylabel("NLL per configuration")
    plt.title(f"2D phi^4 (L={L}, m2={m2}, lam={lam}): Training NLL")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        f"phi4_2d_L{L}_m2{m2:.2f}_lam{lam:.2f}_training_nll.png",
        dpi=200,
    )
    plt.close()

    # Save arrays for plotting script
    out_file = f"phi4_2d_L{L}_m2{m2:.2f}_lam{lam:.2f}_flows_results.npz"
    np.savez(
        out_file,
        nll_X=nll_X,
        nll_Z2=nll_Z2,
        nll_Z2T=nll_Z2T,
        M_hmc=M_abs_hmc,
        M_nf_X=M_abs_nf_X,
        M_nf_Z2=M_abs_nf_Z2,
        M_nf_Z2T=M_abs_nf_Z2T,
    )
    print(f"\n[SAVE] Stored training curves and |M| samples to {out_file}")


if __name__ == "__main__":
    main()
