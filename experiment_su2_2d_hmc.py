#!/usr/bin/env python
"""
experiment_su2_2d_hmc.py

Simple Metropolis MC for 2D SU(2) gauge theory with Wilson action.
Generates an ensemble of configurations plus a translation–canonicalised
version (quotient by the spatial translation group Λ), then stores them
in an .npz file for flow training.

This is deliberately written in the same style as the phi^4 and U(1)
experiments.
"""

import numpy as np
import torch
from torch import tensor
from tqdm import trange

# ---------------------------
# Global settings
# ---------------------------

L = 8                  # lattice linear size (LxL)
beta = 2.2             # gauge coupling
n_therm = 2000         # thermalisation steps
n_samples = 2000       # number of saved configurations
n_skip = 5             # MC steps between saved configs

eps_prop = 0.25        # proposal size in group space
seed = 1234
out_file = f"su2_2d_L{L}_beta{beta:.2f}.npz"
device = torch.device("mps" if torch.backends.mps.is_available()
                      else "cuda" if torch.cuda.is_available()
                      else "cpu")

torch.manual_seed(seed)
np.random.seed(seed)

# ---------------------------
# SU(2) helper routines
# Representation: 4-vector (a0, a1, a2, a3) with ||a|| = 1.
# ---------------------------

def su2_quat_normalize(q):
    """Normalize 4-vector to unit length."""
    return q / q.norm(dim=-1, keepdim=True)

def su2_quat_random(device=None):
    """Haar-random SU(2) element via normalised Gaussian 4-vector."""
    if device is None:
        device = torch.device("cpu")
    q = torch.randn(4, device=device)
    return su2_quat_normalize(q)

def su2_quat_exp(eps, n):
    """
    Small group element R = exp(i eps * n·sigma).
    n: (..., 3)
    Returns R as (..., 4) in quaternion form.
    """
    # Norm of the su(2) algebra element
    n_norm = n.norm(dim=-1, keepdim=True)          # (..., 1)

    # Unit direction (safe division)
    n_hat = n / (n_norm + 1e-8)                    # (..., 3)

    # Rotation angle
    theta = eps * n_norm                           # (..., 1)
    half_theta = theta / 2.0                       # (..., 1)

    a0 = torch.cos(half_theta)                     # (..., 1)
    sin_half = torch.sin(half_theta)               # (..., 1)

    # Vector part
    avec = n_hat * sin_half                        # (..., 3)

    return torch.cat([a0, avec], dim=-1)           # (..., 4)


def su2_quat_mul(a, b):
    """
    Quaternion (SU(2)) multiplication.
    a, b: (..., 4)
    """
    a0 = a[..., 0:1]
    av = a[..., 1:]
    b0 = b[..., 0:1]
    bv = b[..., 1:]
    scalar = a0 * b0 - (av * bv).sum(dim=-1, keepdim=True)
    vec = a0 * bv + b0 * av + torch.cross(av, bv, dim=-1)
    return torch.cat([scalar, vec], dim=-1)

def su2_quat_inv(a):
    """Inverse (conjugate) of unit quaternion."""
    a0 = a[..., 0:1]
    av = a[..., 1:]
    return torch.cat([a0, -av], dim=-1)

# ---------------------------
# Lattice geometry
# ---------------------------

def init_lattice(L, device):
    """
    Random SU(2) configuration on LxL lattice.
    Shape: (2, L, L, 4)  # 2 directions, each link is a 4-vector.
    """
    U = torch.randn(2, L, L, 4, device=device)
    U = su2_quat_normalize(U)
    return U

def su2_plaquette_field(U):
    """
    Compute scalar part (a0) of plaquette variable at each site.
    U: (2, L, L, 4)
    Returns: plaq_scalar: (L, L)
    """
    L = U.shape[1]
    # Links
    Ux = U[0]                   # (L, L, 4)  x-direction
    Uy = U[1]                   # (L, L, 4)  y-direction

    # Periodic shifts
    Ux_xp = torch.roll(Ux, shifts=-1, dims=0)    # Ux(x+1,y)
    Uy_yp = torch.roll(Uy, shifts=-1, dims=1)    # Uy(x,y+1)

    # Plaquette: Ux(x,y) Uy(x+1,y) Ux^{-1}(x,y+1) Uy^{-1}(x,y)
    up = su2_quat_mul(Ux, su2_quat_mul(Ux_xp, su2_quat_mul(
        su2_quat_inv(Uy_yp),
        su2_quat_inv(Uy)
    )))
    return up[..., 0]  # scalar part a0

def su2_action(U, beta):
    """
    Wilson action: S = - (beta / 2) sum Re Tr(U_p) = - beta * sum a0_p
    since Tr(U) = 2 a0 for SU(2).
    """
    plaq = su2_plaquette_field(U)
    return -beta * plaq.sum()

def polyakov_loop_abs(U):
    """
    Very simple Polyakov-loop-like observable along the x-direction:
    product of Ux links around the ring at fixed y, then average over y,
    then take complex phase magnitude ~ a0 of quaternion product.
    """
    L = U.shape[1]
    Ux = U[0]  # (L, L, 4)
    # product around x for each y
    P = torch.ones(L, 4, device=Ux.device)
    P[:, 0] = 1.0
    P[:, 1:] = 0.0
    for x in range(L):
        P = su2_quat_mul(P, Ux[x])  # broadcast over y
    # trace ~ 2 a0
    a0 = P[..., 0]
    # Map to "magnitude" in [0,1].
    return a0.abs().mean()

# ---------------------------
# Metropolis updates
# ---------------------------

def metropolis_step(U, beta, eps, device):
    """
    Global Metropolis update: propose R * U with small random SU(2) rotation R.
    Returns new U and accepted flag.
    """
    L = U.shape[1]
    # random su(2) algebra noises (3 components)
    n = torch.randn_like(U[..., 1:], device=device)  # (2, L, L, 3)
    R = su2_quat_exp(eps, n.view(-1, 3)).view(2, L, L, 4)
    U_prop = su2_quat_mul(R, U)

    S_old = su2_action(U, beta)
    S_new = su2_action(U_prop, beta)
    dS = (S_new - S_old).item()
    if dS < 0.0 or np.random.rand() < np.exp(-dS):
        return U_prop, True
    else:
        return U, False

# ---------------------------
# Canonicalisation (translation quotient)
# ---------------------------

def canonicalise_translation(U):
    """
    Fix a translation by shifting so that the site of maximal plaquette
    scalar a0 is moved to (0,0).  This mirrors the "argmax" canonicalisation
    used in the scalar models.
    U: (2, L, L, 4)
    Returns: U_can with the same shape.
    """
    L = U.shape[1]
    plaq = su2_plaquette_field(U)           # (L, L)
    idx = plaq.view(-1).argmax().item()
    x0, y0 = divmod(idx, L)
    # roll so that (x0,y0) -> (0,0)
    U_can = torch.clone(U)
    U_can = torch.roll(U_can, shifts=(-x0, -y0), dims=(1, 2))
    return U_can

# ---------------------------
# Main routine
# ---------------------------

def main():
    print(f"[Device] Using {device}")
    print(f"=== 2D SU(2) gauge: Metropolis MC ===")

    U = init_lattice(L, device)
    acc = 0
    # thermalisation
    for step in trange(n_therm, desc="[Therm]"):
        U, accepted = metropolis_step(U, beta, eps_prop, device)
        acc += int(accepted)
    print(f"[MC] Thermalisation acceptance ≈ {acc / n_therm:.3f}")
    acc = 0

    configs_raw = []
    configs_can = []
    E_plaquette = []
    P_abs = []

    total_steps = n_therm + n_samples * n_skip
    step = 0
    saved = 0

    pbar = trange(n_samples * n_skip, desc="[MC] Production")
    for _ in pbar:
        U, accepted = metropolis_step(U, beta, eps_prop, device)
        acc += int(accepted)
        step += 1

        if step % n_skip == 0:
            # measure
            plaq = su2_plaquette_field(U)
            E = (1.0 - plaq).mean().item()
            P = polyakov_loop_abs(U).item()
            configs_raw.append(U.cpu().numpy())
            configs_can.append(canonicalise_translation(U).cpu().numpy())
            E_plaquette.append(E)
            P_abs.append(P)
            saved += 1
            pbar.set_postfix({
                "saved": saved,
                "acc": acc / step
            })

    acc_rate = acc / (n_samples * n_skip)
    print(f"[MC] Final acceptance rate ≈ {acc_rate:.3f}")
    print(f"[MC] Generated {len(configs_raw)} samples of size 2x{L}x{L}.")

    # Convert to arrays
    configs_raw = np.stack(configs_raw, axis=0)       # (N, 2, L, L, 4)
    configs_can = np.stack(configs_can, axis=0)       # (N, 2, L, L, 4)
    E_plaquette = np.array(E_plaquette, dtype=np.float64)
    P_abs = np.array(P_abs, dtype=np.float64)

    print("\n[MC] Observables (raw ensemble):")
    print(f"  <E_plaquette> = {E_plaquette.mean():+.6f}")
    print(f"  <|P|>          = {P_abs.mean():+.6f}")

    np.savez(
        out_file,
        L=L,
        beta=beta,
        configs_raw=configs_raw,
        configs_can=configs_can,
        E_plaquette=E_plaquette,
        P_abs=P_abs,
    )
    print(f"[SAVE] Wrote datasets to {out_file}")


if __name__ == "__main__":
    main()
