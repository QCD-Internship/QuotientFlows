#!/usr/bin/env python3
"""
2D SU(2) gauge (gauge quotient): train RealNVP flows on

- X (raw link-angle quaternion configs)
- X / G_gauge (gauge-fixed canonical reps)

and save training histories + sampled observables.

Input:
  su2_2d_L{L}_beta{beta:.2f}_gaugequotient.npz

Output:
  su2_2d_L{L}_beta{beta:.2f}_gaugequotient_flows_results.npz
"""

import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import trange


# ------------------------------
# Device
# ------------------------------
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


device = get_device()
print(f"[Device] Using {device}")


# ------------------------------
# SU(2) quaternion utilities (same as in HMC file)
# ------------------------------
def quat_normalize(q: torch.Tensor) -> torch.Tensor:
    norm = torch.linalg.norm(q, dim=-1, keepdim=True)
    return q / norm.clamp_min(1e-8)


def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    w1, x1, y1, z1 = torch.unbind(q1, dim=-1)
    w2, x2, y2, z2 = torch.unbind(q2, dim=-1)
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack((w, x, y, z), dim=-1)


def quat_conj(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = torch.unbind(q, dim=-1)
    return torch.stack((w, -x, -y, -z), dim=-1)


def compute_plaquette_field(U: torch.Tensor) -> torch.Tensor:
    L = U.shape[1]
    Ux = U[0]
    Uy = U[1]

    E = torch.zeros(L, L, device=U.device)
    for x in range(L):
        xp = (x + 1) % L
        for y in range(L):
            yp = (y + 1) % L
            u1 = Ux[x, y]
            u2 = Uy[xp, y]
            u3 = quat_conj(Ux[x, yp])
            u4 = quat_conj(Uy[x, y])
            up = quat_mul(quat_mul(u1, u2), quat_mul(u3, u4))
            a0 = up[..., 0]
            E[x, y] = 1.0 - a0
    return E


def compute_observables(U: torch.Tensor):
    L = U.shape[1]
    Ux = U[0]

    # plaquette
    E = compute_plaquette_field(U)
    E_mean = E.mean().item()

    # Polyakov along x
    P_vals = []
    for y in range(L):
        P = torch.tensor([1.0, 0.0, 0.0, 0.0], device=U.device)
        for x in range(L):
            P = quat_mul(P, Ux[x, y])
        a0 = P[0]
        P_vals.append(torch.abs(a0))
    P_vals = torch.stack(P_vals)
    P_abs_mean = P_vals.mean().item()

    return E_mean, P_abs_mean


def random_gauge_transform(U: torch.Tensor):
    """
    Apply a random local SU(2) gauge transformation:

      U'_mu(x) = g(x) U_mu(x) g^\dagger(x+mu),

    where g(x) ~ Haar(SU(2)) approximated by normalised random quats.
    """
    L = U.shape[1]
    device = U.device
    # random g(x)
    g = torch.randn(L, L, 4, device=device)
    g = quat_normalize(g)

    Ux = U[0]
    Uy = U[1]

    U_new = torch.empty_like(U)
    for x in range(L):
        xp = (x + 1) % L
        for y in range(L):
            yp = (y + 1) % L
            gx = g[x, y]

            gx_next_x = g[xp, y]
            gx_next_y = g[x, yp]

            U_new[0, x, y] = quat_mul(quat_mul(gx, Ux[x, y]), quat_conj(gx_next_x))
            U_new[1, x, y] = quat_mul(quat_mul(gx, Uy[x, y]), quat_conj(gx_next_y))

    U_new = quat_normalize(U_new)
    return U_new


# ------------------------------
# RealNVP components
# ------------------------------
class AffineCoupling(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim

        self.net = nn.Sequential(
            nn.Linear(dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x, mask):
        """
        x: (batch, D)
        mask: (D,) 0/1, with 1 = pass-through, 0 = transformed
        """
        x1 = x * mask
        h = self.net(x1[:, mask.bool()])
        s, t = torch.chunk(h, 2, dim=-1)
        s = torch.tanh(s) * 0.8  # scaled log-scale
        x2 = x[:, ~mask.bool()]
        y2 = x2 * torch.exp(s) + t

        y = torch.zeros_like(x)
        y[:, mask.bool()] = x1[:, mask.bool()]
        y[:, ~mask.bool()] = y2

        log_det = s.sum(dim=-1)
        return y, log_det

    def inverse(self, y, mask):
        y1 = y * mask
        h = self.net(y1[:, mask.bool()])
        s, t = torch.chunk(h, 2, dim=-1)
        s = torch.tanh(s) * 0.8
        y2 = y[:, ~mask.bool()]
        x2 = (y2 - t) * torch.exp(-s)

        x = torch.zeros_like(y)
        x[:, mask.bool()] = y1[:, mask.bool()]
        x[:, ~mask.bool()] = x2

        log_det = -s.sum(dim=-1)
        return x, log_det


class RealNVP(nn.Module):
    def __init__(self, dim, hidden_dim=512, n_coupling_layers=8):
        super().__init__()
        self.dim = dim
        self.masks = nn.ParameterList()
        for i in range(n_coupling_layers):
            if i % 2 == 0:
                mask = torch.cat(
                    [torch.ones(dim // 2), torch.zeros(dim - dim // 2)]
                )
            else:
                mask = torch.cat(
                    [torch.zeros(dim // 2), torch.ones(dim - dim // 2)]
                )
            self.masks.append(nn.Parameter(mask, requires_grad=False))

        self.couplings = nn.ModuleList(
            [AffineCoupling(dim, hidden_dim) for _ in range(n_coupling_layers)]
        )

        # Standard Normal prior parameters (diagonal, factorised)
        self.dim = dim

    def f(self, x):
        """Forward map x -> z with log-det."""
        log_det = torch.zeros(x.shape[0], device=x.device)
        z = x
        for mask, c in zip(self.masks, self.couplings):
            z, ld = c(z, mask.to(x.device))
            log_det += ld
        return z, log_det

    def f_inv(self, z):
        """Inverse map z -> x with log-det."""
        log_det = torch.zeros(z.shape[0], device=z.device)
        x = z
        for mask, c in reversed(list(zip(self.masks, self.couplings))):
            x, ld = c.inverse(x, mask.to(z.device))
            log_det += ld
        return x, log_det

    def log_prob(self, x):
        """
        log p(x) = log p(z) + log |det ∂z/∂x|
        with p(z) = N(0, I).
        """
        z, log_det = self.f(x)
        d = self.dim
        log_pz = -0.5 * (z.pow(2).sum(dim=-1) + d * np.log(2.0 * np.pi))
        return log_pz + log_det

    def sample(self, n_samples, device):
        """
        Sample z ~ N(0, I), then x = f^{-1}(z).
        """
        z = torch.randn(n_samples, self.dim, device=device)
        x, _ = self.f_inv(z)
        return x


# ------------------------------
# Helper: batched sampling on chosen device
# ------------------------------
def sample_flow_in_batches(flow, n_samples, batch_size, device):
    """
    Sample from a flow in small batches to avoid huge allocations.
    Returns a tensor on CPU of shape (n_samples, D).
    """
    flow.eval()
    samples = []
    with torch.no_grad():
        for start in range(0, n_samples, batch_size):
            k = min(batch_size, n_samples - start)
            x_batch = flow.sample(k, device=device)
            samples.append(x_batch.cpu())
    return torch.cat(samples, dim=0)


# ------------------------------
# Main training + sampling
# ------------------------------
def main():
    # --------------------------
    # Load data
    # --------------------------
    L = 16
    beta = 2.20
    data_fname = f"su2_2d_L{L}_beta{beta:.2f}_gaugequotient.npz"
    data = np.load(data_fname)
    configs_raw = data["configs_raw"]      # (N, 2, L, L, 4)
    configs_canon = data["configs_canon"]  # (N, 2, L, L, 4)

    N = configs_raw.shape[0]
    D = configs_raw.reshape(N, -1).shape[1]

    print(f"[DATA] Loaded {N} configs of dimension {D} (L={L}) from {data_fname}")

    X_raw = torch.from_numpy(configs_raw.reshape(N, D)).float()
    X_can = torch.from_numpy(configs_canon.reshape(N, D)).float()

    dataset_raw = TensorDataset(X_raw)
    dataset_can = TensorDataset(X_can)

    batch_size = 256
    loader_raw = DataLoader(dataset_raw, batch_size=batch_size, shuffle=True, drop_last=True)
    loader_can = DataLoader(dataset_can, batch_size=batch_size, shuffle=True, drop_last=True)

    # --------------------------
    # Define flows
    # --------------------------
    flow_X = RealNVP(D, hidden_dim=512, n_coupling_layers=8).to(device)
    flow_XG = RealNVP(D, hidden_dim=512, n_coupling_layers=8).to(device)

    opt_X = torch.optim.Adam(flow_X.parameters(), lr=1e-3)
    opt_XG = torch.optim.Adam(flow_XG.parameters(), lr=1e-3)

    n_epochs = 300

    nll_hist_X = []
    nll_hist_XG = []

    # --------------------------
    # Train flow on X
    # --------------------------
    print("=== 2D SU(2) gauge (gauge quotient): Flows on X and X/G_gauge ===")

    for epoch in trange(1, n_epochs + 1, desc="[Train X]"):
        flow_X.train()
        epoch_nll = 0.0
        n_batches = 0

        for (batch,) in loader_raw:
            batch = batch.to(device)
            opt_X.zero_grad()
            log_p = flow_X.log_prob(batch)
            loss = -log_p.mean()
            loss.backward()
            opt_X.step()

            epoch_nll += loss.item()
            n_batches += 1

        epoch_nll /= max(1, n_batches)
        nll_hist_X.append(epoch_nll)

        if epoch % 50 == 0 or epoch == 1:
            print(f"[Train][X] Epoch {epoch:4d}/{n_epochs} | NLL ≈ {epoch_nll:0.4f}")

    # --------------------------
    # Train flow on X / G_gauge
    # --------------------------
    for epoch in trange(1, n_epochs + 1, desc="[Train X/G_gauge]"):
        flow_XG.train()
        epoch_nll = 0.0
        n_batches = 0

        for (batch,) in loader_can:
            batch = batch.to(device)
            opt_XG.zero_grad()
            log_p = flow_XG.log_prob(batch)
            loss = -log_p.mean()
            loss.backward()
            opt_XG.step()

            epoch_nll += loss.item()
            n_batches += 1

        epoch_nll /= max(1, n_batches)
        nll_hist_XG.append(epoch_nll)

        if epoch % 50 == 0 or epoch == 1:
            print(f"[Train][X/G_gauge] Epoch {epoch:4d}/{n_epochs} | NLL ≈ {epoch_nll:0.4f}")

    # --------------------------
    # Sample from trained flows and measure observables
    # --------------------------
    print("[INFO] Finished training. Moving flows to CPU for evaluation...")
    eval_device = torch.device("cpu")
    flow_X.to(eval_device)
    flow_XG.to(eval_device)

    n_eval = 2000
    eval_batch = 128

    # Flow on X
    print("[INFO] Sampling from flow on X...")
    samples_X = sample_flow_in_batches(flow_X, n_eval, eval_batch, eval_device)
    samples_X = samples_X.view(n_eval, 2, L, L, 4)

    E_X_vals = []
    P_X_vals = []
    print("[INFO] Computing observables for flow on X...")
    for i in range(n_eval):
        U = samples_X[i]
        U = quat_normalize(U)
        E_mean, P_abs_mean = compute_observables(U)
        E_X_vals.append(E_mean)
        P_X_vals.append(P_abs_mean)
    E_X_vals = np.array(E_X_vals)
    P_X_vals = np.array(P_X_vals)

    # Flow on X/G_gauge + random gauge lifting
    print("[INFO] Sampling from flow on X/G_gauge...")
    samples_can = sample_flow_in_batches(flow_XG, n_eval, eval_batch, eval_device)
    samples_can = samples_can.view(n_eval, 2, L, L, 4)

    E_XG_vals = []
    P_XG_vals = []
    print("[INFO] Computing observables for flow on X/G_gauge (with random gauge lifting)...")
    for i in range(n_eval):
        U_can = samples_can[i]
        U_can = quat_normalize(U_can)
        U_lift = random_gauge_transform(U_can)
        E_mean, P_abs_mean = compute_observables(U_lift)
        E_XG_vals.append(E_mean)
        P_XG_vals.append(P_abs_mean)
    E_XG_vals = np.array(E_XG_vals)
    P_XG_vals = np.array(P_XG_vals)

    print("\n[NF] Observables from trained flows (sampled):")
    print("  Flow on X:")
    print(f"    <E_plaquette> = {E_X_vals.mean():+0.6f} ± {E_X_vals.std() / np.sqrt(n_eval):0.6f}")
    print(f"    <|P|>          = {P_X_vals.mean():+0.6f} ± {P_X_vals.std() / np.sqrt(n_eval):0.6f}")
    print("  Flow on X/G_gauge (canon, lifted):")
    print(f"    <E_plaquette> = {E_XG_vals.mean():+0.6f} ± {E_XG_vals.std() / np.sqrt(n_eval):0.6f}")
    print(f"    <|P|>          = {P_XG_vals.mean():+0.6f} ± {P_XG_vals.std() / np.sqrt(n_eval):0.6f}")

    # --------------------------
    # Save results
    # --------------------------
    out_fname = f"su2_2d_L{L}_beta{beta:.2f}_gaugequotient_flows_results.npz"
    np.savez(
        out_fname,
        nll_X=np.array(nll_hist_X),
        nll_XG=np.array(nll_hist_XG),
        E_X=E_X_vals,
        P_X=P_X_vals,
        E_XG=E_XG_vals,
        P_XG=P_XG_vals,
        L=L,
        beta=beta,
    )
    print(f"[SAVE] Stored training curves and flow observables to {out_fname}")
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
