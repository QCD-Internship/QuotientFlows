#!/usr/bin/env python3
"""
2D SU(2) gauge theory on an LxL lattice (periodic BCs).
Ensemble generation + gauge-quotient canonicalisation.

We:
- generate configs U_mu(x) \in SU(2) via a simple global Metropolis update,
- compute basic observables (plaquette energy and |Polyakov loop|),
- gauge-fix to a canonical representative on each configuration
  using a simple tree gauge (axial-like),
- save both raw and gauge-fixed configs for later flow training.

Output: su2_2d_L{L}_beta{beta:.2f}_gaugequotient.npz
  - configs_raw:   shape (N, 2, L, L, 4)
  - configs_canon: shape (N, 2, L, L, 4)
  - E_plaquette_mean
  - P_abs_mean
  - L, beta
"""

import numpy as np
import torch
from tqdm import trange


# ------------------------------
# Device selection
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
# SU(2) quaternion utilities
# ------------------------------
def quat_normalize(q: torch.Tensor) -> torch.Tensor:
    """Normalize quaternions to unit norm along last dim."""
    norm = torch.linalg.norm(q, dim=-1, keepdim=True)
    return q / norm.clamp_min(1e-8)


def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """
    Quaternion product (q1 * q2).

    q = (w, x, y, z) with SU(2) ~ unit quaternions.
    Shapes: (..., 4)
    """
    w1, x1, y1, z1 = torch.unbind(q1, dim=-1)
    w2, x2, y2, z2 = torch.unbind(q2, dim=-1)

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack((w, x, y, z), dim=-1)


def quat_conj(q: torch.Tensor) -> torch.Tensor:
    """Quaternion conjugate for SU(2) (inverse for unit quats)."""
    w, x, y, z = torch.unbind(q, dim=-1)
    return torch.stack((w, -x, -y, -z), dim=-1)


def su2_random_quat(eps: float, shape, device):
    """
    Small random SU(2) element near identity:

    U = cos(theta) + sin(theta) * (n_hat · iσ),
    where theta ~ eps * |n|, n ~ N(0, I_3).
    """
    n = torch.randn(*shape, 3, device=device)
    theta = eps * torch.linalg.norm(n, dim=-1, keepdim=True)  # (..., 1)
    n_hat = torch.where(
        theta > 1e-8,
        n / theta.clamp_min(1e-8),
        torch.zeros_like(n),
    )
    half_theta = theta
    cos_t = torch.cos(half_theta)
    sin_t = torch.sin(half_theta)
    # Quaternion: (w, x, y, z) with imaginary part along n_hat
    q = torch.cat([cos_t, sin_t * n_hat], dim=-1)
    return quat_normalize(q)


# ------------------------------
# Lattice utilities
# ------------------------------
def init_identity_config(L: int, device):
    """
    Identity SU(2) config: all links = 1 (quaternion (1,0,0,0)).

    Shape: (2, L, L, 4)  -- directions mu=0,1.
    """
    U = torch.zeros(2, L, L, 4, device=device)
    U[..., 0] = 1.0  # w=1, x=y=z=0
    return U


def index_shift(i, L, shift):
    """Periodic index: (i + shift) mod L."""
    return (i + shift) % L


def compute_plaquette_field(U: torch.Tensor) -> torch.Tensor:
    """
    Compute plaquette SU(2) matrices (as quats) and return plaquette
    energy E_p = 1 - 0.5 * Re Tr(U_p) = 1 - a0, where a0 is the w component.

    U shape: (2, L, L, 4) with directions 0=x, 1=y.
    """
    L = U.shape[1]
    Ux = U[0]  # (L, L, 4)
    Uy = U[1]

    E = torch.zeros(L, L, device=U.device)

    for x in range(L):
        xp = (x + 1) % L
        for y in range(L):
            yp = (y + 1) % L
            # Plaquette: Ux(x,y) Uy(x+1,y) Ux^\dagger(x,y+1) Uy^\dagger(x,y)
            u1 = Ux[x, y]
            u2 = Uy[xp, y]
            u3 = quat_conj(Ux[x, yp])
            u4 = quat_conj(Uy[x, y])
            up = quat_mul(quat_mul(u1, u2), quat_mul(u3, u4))
            a0 = up[..., 0]
            E[x, y] = 1.0 - a0  # 1 - 1/2 Tr(U_p) since Tr(U_p) = 2 a0
    return E


def compute_observables(U: torch.Tensor):
    """
    Compute:
    - mean plaquette energy <E_p>,
    - Polyakov loop magnitude |P| along x direction:
      P_y = (1/L^2) sum_y |Tr( Π_x U_x(x,y) ) / 2|.
    """
    L = U.shape[1]
    Ux = U[0]

    # Plaquette energy
    E = compute_plaquette_field(U)
    E_mean = E.mean().item()

    # Polyakov loop along x
    P_vals = []
    for y in range(L):
        # product of Ux(x,y) along x
        P = torch.tensor([1.0, 0.0, 0.0, 0.0], device=U.device)
        for x in range(L):
            P = quat_mul(P, Ux[x, y])
        a0 = P[0]  # Re Tr(P)/2 = a0
        P_vals.append(torch.abs(a0))
    P_vals = torch.stack(P_vals)
    P_abs_mean = P_vals.mean().item()

    return E_mean, P_abs_mean


def action(U: torch.Tensor, beta: float) -> torch.Tensor:
    """
    Wilson action: S = beta * sum E_p
    with E_p = 1 - 1/2 Tr(U_p) = 1 - a0.
    """
    E = compute_plaquette_field(U)
    return beta * E.sum()


def metropolis_global_step(U: torch.Tensor, beta: float, eps: float):
    """
    Global Metropolis step:
    propose U' = R * U (elementwise), with R ~ SU(2) near identity.

    This is not super efficient but sufficient at small lattice size;
    it keeps the code simple and mirrors the structure used in the
    U(1) gauge-quotient experiments.
    """
    L = U.shape[1]
    S_old = action(U, beta)
    R = su2_random_quat(eps, (2, L, L), U.device)
    U_prop = quat_mul(R, U)
    U_prop = quat_normalize(U_prop)
    S_new = action(U_prop, beta)
    dS = S_new - S_old
    accept_prob = torch.exp(-dS).clamp(max=1.0).item()

    if np.random.rand() < accept_prob:
        return U_prop, True, S_new
    else:
        return U, False, S_old


# ------------------------------
# Gauge fixing (tree gauge)
# ------------------------------
def gauge_fix_tree(U: torch.Tensor) -> torch.Tensor:
    """
    Simple axial / tree gauge:

    - Choose reference site (0,0) with g(0,0) = 1.
    - Define gauge transform g(x,y) recursively along a spanning tree:
      * First along x from (0,0),
      * then along y for each x.
    - Apply U'_mu(x) = g(x) U_mu(x) g^\dagger(x + \hat{mu}).

    This gives a deterministic (though crude) representative of each
    gauge orbit. Sufficient for our 'gauge quotient' experiment.
    """
    L = U.shape[1]
    device = U.device

    # g(x,y) as quats, start with identity
    g = torch.zeros(L, L, 4, device=device)
    g[..., 0] = 1.0  # identity

    Ux = U[0].clone()
    Uy = U[1].clone()

    # Build g along x at y=0
    for x in range(1, L):
        # g(x,0) = U_x(x-1,0) g(x-1,0)
        g[x, 0] = quat_mul(Ux[x - 1, 0], g[x - 1, 0])
        g[x, 0] = quat_normalize(g[x, 0])

    # Then build g along y for each x
    for x in range(L):
        for y in range(1, L):
            # g(x,y) = U_y(x,y-1) g(x,y-1)
            g[x, y] = quat_mul(Uy[x, y - 1], g[x, y - 1])
            g[x, y] = quat_normalize(g[x, y])

    # Apply gauge transform to all links
    U_gf = torch.empty_like(U)
    for x in range(L):
        xp = (x + 1) % L
        for y in range(L):
            yp = (y + 1) % L

            gx = g[x, y]
            # mu=0: link from (x,y) to (x+1,y)
            gx_next = g[xp, y]
            U_gf[0, x, y] = quat_mul(quat_mul(gx, Ux[x, y]), quat_conj(gx_next))

            # mu=1: link from (x,y) to (x,y+1)
            gy_next = g[x, yp]
            U_gf[1, x, y] = quat_mul(quat_mul(gx, Uy[x, y]), quat_conj(gy_next))

    # Re-project to SU(2)
    U_gf = quat_normalize(U_gf)
    return U_gf


# ------------------------------
# Main driver
# ------------------------------
def main():
    L = 16
    beta = 2.20

    n_therm = 2000
    n_samples = 2000
    n_skip = 5
    eps_prop = 0.3

    print("=== 2D SU(2) gauge (gauge quotient): Metropolis MC ===")

    U = init_identity_config(L, device)
    S_current = action(U, beta)

    # Thermalisation
    print("[Therm] Thermalising...")
    for _ in trange(n_therm, desc="[Therm]", leave=False):
        U, accepted, S_current = metropolis_global_step(U, beta, eps_prop)

    # Sampling
    samples_raw = []
    samples_gf = []
    E_vals = []
    P_vals = []

    print("[Sample] Generating configs...")
    total_steps = n_samples * n_skip
    accepted_count = 0
    for step in trange(total_steps, desc="[Sample]", leave=False):
        U, accepted, S_current = metropolis_global_step(U, beta, eps_prop)
        if accepted:
            accepted_count += 1

        if (step + 1) % n_skip == 0:
            # Measure & store
            E_mean, P_abs_mean = compute_observables(U)
            E_vals.append(E_mean)
            P_vals.append(P_abs_mean)

            U_gf = gauge_fix_tree(U)
            samples_raw.append(U.detach().cpu().numpy())
            samples_gf.append(U_gf.detach().cpu().numpy())

    acc_rate = accepted_count / total_steps
    print(f"[MC] Acceptance rate ≈ {acc_rate:.3f}")

    samples_raw = np.stack(samples_raw, axis=0)  # (N, 2, L, L, 4)
    samples_gf = np.stack(samples_gf, axis=0)

    E_vals = np.array(E_vals)
    P_vals = np.array(P_vals)

    print("\n[HMC] Observables (raw ensemble):")
    print(f"  <E_plaquette> = {E_vals.mean():+0.6f}")
    print(f"  <|P|>          = {P_vals.mean():+0.6f}")

    fname = f"su2_2d_L{L}_beta{beta:.2f}_gaugequotient.npz"
    np.savez(
        fname,
        configs_raw=samples_raw,
        configs_canon=samples_gf,
        E_plaquette_mean=E_vals.mean(),
        P_abs_mean=P_vals.mean(),
        L=L,
        beta=beta,
    )
    print(f"[SAVE] Wrote datasets to {fname}")


if __name__ == "__main__":
    main()
