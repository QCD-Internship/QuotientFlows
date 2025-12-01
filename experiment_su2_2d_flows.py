#!/usr/bin/env python
"""
experiment_su2_2d_flows.py

Train RealNVP flows on:
  - X: raw 2D SU(2) gauge configurations
  - X/Λ: translation–canonicalised configurations

Then sample from both flows, lift back to SU(2) by normalising each
4-vector block, and compute plaquette energy and |P| observables.

Results (training curves and observables from flow samples) are stored
in an .npz file for plotting.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import trange

# ---------------------------
# Settings
# ---------------------------

in_file = "su2_2d_L8_beta2.20.npz"   # adjust if you changed L or beta
out_file = "su2_2d_L8_beta2.20_flows_results.npz"

device = torch.device("mps" if torch.backends.mps.is_available()
                      else "cuda" if torch.cuda.is_available()
                      else "cpu")

torch.manual_seed(1234)
np.random.seed(1234)

# ---------------------------
# SU(2) helpers (same as in HMC file, but in torch)
# ---------------------------

def su2_quat_normalize(q):
    return q / q.norm(dim=-1, keepdim=True)

def su2_quat_mul(a, b):
    a0 = a[..., 0:1]
    av = a[..., 1:]
    b0 = b[..., 0:1]
    bv = b[..., 1:]
    scalar = a0 * b0 - (av * bv).sum(dim=-1, keepdim=True)
    vec = a0 * bv + b0 * av + torch.cross(av, bv, dim=-1)
    return torch.cat([scalar, vec], dim=-1)

def su2_quat_inv(a):
    a0 = a[..., 0:1]
    av = a[..., 1:]
    return torch.cat([a0, -av], dim=-1)

def su2_plaquette_field(U):
    L = U.shape[1]
    Ux = U[0]
    Uy = U[1]
    Ux_xp = torch.roll(Ux, shifts=-1, dims=0)
    Uy_yp = torch.roll(Uy, shifts=-1, dims=1)
    up = su2_quat_mul(Ux, su2_quat_mul(Ux_xp, su2_quat_mul(
        su2_quat_inv(Uy_yp),
        su2_quat_inv(Uy)
    )))
    return up[..., 0]

def polyakov_loop_abs(U):
    L = U.shape[1]
    Ux = U[0]
    P = torch.ones(L, 4, device=Ux.device)
    P[:, 0] = 1.0
    P[:, 1:] = 0.0
    for x in range(L):
        P = su2_quat_mul(P, Ux[x])
    a0 = P[..., 0]
    return a0.abs().mean()


# ---------------------------
# RealNVP components
# ---------------------------

class AffineCoupling(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.dim = dim
        self.hidden = nn.Sequential(
            nn.Linear(dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.s = nn.Linear(hidden_dim, dim - dim // 2)
        self.t = nn.Linear(hidden_dim, dim - dim // 2)

        # scale output smaller for stability
        self.scale = 0.8

    def forward(self, x, reverse=False):
        x1 = x[..., : self.dim // 2]
        x2 = x[..., self.dim // 2:]

        h = self.hidden(x1)
        log_s = self.scale * torch.tanh(self.s(h))
        t = self.t(h)

        if not reverse:
            y2 = x2 * torch.exp(log_s) + t
            y = torch.cat([x1, y2], dim=-1)
            log_det = log_s.sum(dim=-1)
        else:
            y2 = (x2 - t) * torch.exp(-log_s)
            y = torch.cat([x1, y2], dim=-1)
            log_det = -log_s.sum(dim=-1)
        return y, log_det


class RealNVP(nn.Module):
    def __init__(self, dim, hidden_dim=512, n_layers=8):
        super().__init__()
        self.dim = dim
        self.layers = nn.ModuleList(
            [AffineCoupling(dim, hidden_dim) for _ in range(n_layers)]
        )
        # simple fixed mask by flipping ordering each layer
        self.register_buffer(
            "mask", torch.tensor([0, 1] * (dim // 2), dtype=torch.bool)
        )

    def f(self, x):
        """Forward map z -> x, and log |det J_f(z)|."""
        log_det_total = torch.zeros(x.shape[0], device=x.device)
        z = x
        for i, layer in enumerate(self.layers):
            if i % 2 == 0:
                z_perm = z
            else:
                z_perm = torch.flip(z, dims=[-1])
            y, log_det = layer(z_perm, reverse=False)
            if i % 2 == 0:
                z = y
            else:
                z = torch.flip(y, dims=[-1])
            log_det_total += log_det
        return z, log_det_total

    def f_inv(self, x):
        """Inverse map x -> z, and log |det J_f^{-1}(x)|."""
        log_det_total = torch.zeros(x.shape[0], device=x.device)
        z = x
        for i, layer in reversed(list(enumerate(self.layers))):
            if i % 2 == 0:
                z_perm = z
            else:
                z_perm = torch.flip(z, dims=[-1])
            y, log_det = layer(z_perm, reverse=True)
            if i % 2 == 0:
                z = y
            else:
                z = torch.flip(y, dims=[-1])
            log_det_total += log_det
        return z, log_det_total

    def log_prob(self, x):
        z, log_det = self.f_inv(x)
        log_pz = -0.5 * (z ** 2).sum(dim=-1) - 0.5 * self.dim * np.log(2 * np.pi)
        return log_pz + log_det

    def sample(self, n):
        z = torch.randn(n, self.dim, device=device)
        x, _ = self.f(z)
        return x


# ---------------------------
# Training helpers
# ---------------------------

def train_flow(data, dim, n_epochs=200, batch_size=256, lr=1e-3, label="X"):
    data = torch.from_numpy(data).float().to(device)
    dataset = torch.utils.data.TensorDataset(data)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )

    model = RealNVP(dim, hidden_dim=512, n_layers=8).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    nll_hist = []
    print(f"[Train] Training flow on {label} with {len(dataset)} samples...")

    for epoch in trange(1, n_epochs + 1, desc=f"[Train][{label}]"):
        total_nll = 0.0
        n_batches = 0
        for (xbatch,) in loader:
            opt.zero_grad()
            logp = model.log_prob(xbatch)
            loss = -logp.mean()
            loss.backward()
            opt.step()

            total_nll += loss.item()
            n_batches += 1
        avg_nll = total_nll / n_batches
        nll_hist.append(avg_nll)
        if epoch % 10 == 0 or epoch == 1:
            print(f"[Train][{label}] Epoch {epoch:4d} / {n_epochs} | NLL ≈ {avg_nll: .4f}")

    return model, np.array(nll_hist, dtype=np.float64)


def samples_to_su2(x, L):
    """
    Map flat R^D samples back to SU(2) link variables by grouping into
    4-vectors and normalising each block.
    x: (N, D)
    Returns: U: (N, 2, L, L, 4)
    """
    N, D = x.shape
    assert D == 2 * L * L * 4
    U = x.view(N, 2, L, L, 4)
    U = su2_quat_normalize(U)
    return U


def measure_observables(U):
    """
    U: (N, 2, L, L, 4)
    Returns dict of observables averaged over samples.
    """
    N = U.shape[0]
    L = U.shape[2]
    E_list = []
    P_list = []

    for i in range(N):
        Ucfg = U[i].to(device)
        plaq = su2_plaquette_field(Ucfg)
        E_list.append((1.0 - plaq).mean().item())
        P_list.append(polyakov_loop_abs(Ucfg).item())

    E = np.array(E_list, dtype=np.float64)
    P = np.array(P_list, dtype=np.float64)
    return {
        "E_mean": E.mean(),
        "E_std": E.std() / np.sqrt(N),
        "|P|_mean": P.mean(),
        "|P|_std": P.std() / np.sqrt(N),
        "E_all": E,
        "P_all": P,
    }


# ---------------------------
# Main
# ---------------------------

def main():
    print(f"[Device] Using {device}")
    data = np.load(in_file)
    L = int(data["L"])
    beta = float(data["beta"])
    configs_raw = data["configs_raw"]       # (N, 2, L, L, 4)
    configs_can = data["configs_can"]       # (N, 2, L, L, 4)

    N = configs_raw.shape[0]
    D = 2 * L * L * 4

    print(f"[LOAD] {in_file}")
    print(f"[DATA] Loaded {N} configs of dimension {D} (L={L})")

    # Flatten for flow
    x_raw = configs_raw.reshape(N, D)
    x_can = configs_can.reshape(N, D)

    # Train flows
    flow_X, nll_X = train_flow(x_raw, D, n_epochs=200, label="X (raw)")
    flow_Xq, nll_Xq = train_flow(x_can, D, n_epochs=200, label="X/Λ (canon)")

    # Sample from flows and map to SU(2)
    n_samples_flow = 5000
    with torch.no_grad():
        x_samp_X = flow_X.sample(n_samples_flow)
        x_samp_Xq = flow_Xq.sample(n_samples_flow)

    U_X = samples_to_su2(x_samp_X, L)
    U_Xq = samples_to_su2(x_samp_Xq, L)

    # Measure observables
    obs_X = measure_observables(U_X)
    obs_Xq = measure_observables(U_Xq)

    print("\n[NF] Observables from trained flows (sampled):")
    print("  Flow on X:")
    print(f"    <E_plaquette> = {obs_X['E_mean']:+.6f} ± {obs_X['E_std']:.6f}")
    print(f"    <|P|>          = {obs_X['|P|_mean']:+.6f} ± {obs_X['|P|_std']:.6f}")
    print("  Flow on X/Λ (canon, lifted):")
    print(f"    <E_plaquette> = {obs_Xq['E_mean']:+.6f} ± {obs_Xq['E_std']:.6f}")
    print(f"    <|P|>          = {obs_Xq['|P|_mean']:+.6f} ± {obs_Xq['|P|_std']:.6f}")

    # Save results
    np.savez(
        out_file,
        L=L,
        beta=beta,
        nll_X=nll_X,
        nll_Xq=nll_Xq,
        obs_X_E=obs_X["E_all"],
        obs_X_P=obs_X["P_all"],
        obs_Xq_E=obs_Xq["E_all"],
        obs_Xq_P=obs_Xq["P_all"],
    )
    print(f"[SAVE] Stored training curves and flow observables to {out_file}")


if __name__ == "__main__":
    main()
