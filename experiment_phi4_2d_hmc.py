#!/usr/bin/env python3
import math
import numpy as np
import torch

# ============================================================
# Device selection
# ============================================================
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("[Device] Using CUDA")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
    print("[Device] Using MPS (Apple Metal)")
else:
    device = torch.device("cpu")
    print("[Device] Using CPU")


class Phi4Lattice2D:
    r"""
    2D lattice φ^4 scalar field with periodic BCs (LxL).

    Action:
        S[φ] = β ∑_x [
            1/2 ∑_{μ=1,2} (φ_{x+μ} − φ_x)^2
            + m²/2 φ_x²
            + λ/4 φ_x⁴
        ]

    where x+μ is understood modulo L.
    """

    def __init__(self, L: int, m2: float, lam: float, beta: float = 1.0):
        self.L = L
        self.m2 = float(m2)
        self.lam = float(lam)
        self.beta = float(beta)

    def action(self, phi: torch.Tensor) -> torch.Tensor:
        """
        phi: (L, L) tensor on device.
        Returns: scalar tensor S[φ].
        """
        # periodic neighbours
        phi_xp = torch.roll(phi, shifts=-1, dims=1)  # +x
        phi_yp = torch.roll(phi, shifts=-1, dims=0)  # +y

        kinetic = 0.5 * torch.sum((phi_xp - phi) ** 2 + (phi_yp - phi) ** 2)
        mass_term = 0.5 * self.m2 * torch.sum(phi ** 2)
        quartic = 0.25 * self.lam * torch.sum(phi ** 4)
        return self.beta * (kinetic + mass_term + quartic)

    def grad_action(self, phi: torch.Tensor) -> torch.Tensor:
        """
        Gradient dS/dφ, same shape as phi (L, L).
        """
        phi_xp = torch.roll(phi, shifts=-1, dims=1)
        phi_xm = torch.roll(phi, shifts=1, dims=1)
        phi_yp = torch.roll(phi, shifts=-1, dims=0)
        phi_ym = torch.roll(phi, shifts=1, dims=0)

        # from 1/2(φ_{x+μ}-φ_x)^2 + 1/2(φ_x-φ_{x-μ})^2
        grad_kin = (phi - phi_xm) - (phi_xp - phi) \
                 + (phi - phi_ym) - (phi_yp - phi)
        grad_mass = self.m2 * phi
        grad_quartic = self.lam * phi ** 3
        return self.beta * (grad_kin + grad_mass + grad_quartic)


# ============================================================
# HMC sampler
# ============================================================

def hmc_sample(
    model: Phi4Lattice2D,
    n_samples: int = 2000,
    n_burnin: int = 2000,
    n_thin: int = 5,
    eps: float = 0.05,
    n_leapfrog: int = 10,
    init_phi: torch.Tensor | None = None,
    print_every: int = 1000,
):
    """
    Basic HMC for 2D φ^4.

    Returns:
        samples: tensor of shape (n_samples, L, L) on CPU.
    """
    L = model.L
    if init_phi is None:
        phi = torch.zeros(L, L, device=device)
    else:
        phi = init_phi.to(device)

    def kinetic(p):
        return 0.5 * torch.sum(p * p)

    total_steps = n_burnin + n_samples * n_thin
    kept = []
    accepts = 0
    proposals = 0

    for step in range(1, total_steps + 1):
        p = torch.randn_like(phi)

        S_current = model.action(phi)
        K_current = kinetic(p)
        H_current = S_current + K_current

        # leapfrog
        phi_prop = phi.clone()
        p_prop = p.clone()

        grad_S = model.grad_action(phi_prop)
        p_prop = p_prop - 0.5 * eps * grad_S
        for lf_step in range(n_leapfrog):
            phi_prop = phi_prop + eps * p_prop
            grad_S = model.grad_action(phi_prop)
            if lf_step != n_leapfrog - 1:
                p_prop = p_prop - eps * grad_S
        p_prop = p_prop - 0.5 * eps * grad_S

        p_prop = -p_prop

        S_prop = model.action(phi_prop)
        K_prop = kinetic(p_prop)
        H_prop = S_prop + K_prop

        dH = (H_prop - H_current).item()
        accept_prob = math.exp(-dH) if dH > 0 else 1.0

        proposals += 1
        if np.random.rand() < accept_prob:
            phi = phi_prop
            accepts += 1

        if step > n_burnin and (step - n_burnin) % n_thin == 0:
            kept.append(phi.detach().cpu().clone())

        if step % print_every == 0 or step == 1 or step == total_steps:
            acc_rate = accepts / proposals
            print(f"[HMC] Step {step}/{total_steps} | acc ≈ {acc_rate:.3f} | dH = {dH:+.3f}")

    samples = torch.stack(kept, dim=0)
    acc_rate = accepts / max(1, proposals)
    print(f"[HMC] Final acceptance rate ≈ {acc_rate:.3f}")
    print(f"[HMC] Generated {samples.shape[0]} samples of size {model.L}x{model.L}")
    return samples


# ============================================================
# Canonicalisation
# ============================================================

def canonicalise_Z2(phi: torch.Tensor) -> torch.Tensor:
    """
    Canonicalise under global Z2: if mean φ < 0, flip φ -> −φ.
    phi: (L, L)
    """
    M = phi.mean()
    return -phi if M < 0 else phi


def canonicalise_Z2_translations(phi: torch.Tensor) -> torch.Tensor:
    """
    Canonicalise under Z2 × translations on LxL torus.

    Steps:
      1) Find index (i*, j*) where |φ| is maximal.
      2) If φ[i*, j*] < 0, flip φ -> −φ.
      3) Roll so that (i*, j*) is moved to (0, 0).
    """
    L = phi.shape[0]
    abs_phi = phi.abs()
    flat_idx = torch.argmax(abs_phi).item()
    i_max = flat_idx // L
    j_max = flat_idx % L

    phi_can = phi.clone()
    if phi_can[i_max, j_max] < 0:
        phi_can = -phi_can

    # bring (i_max, j_max) to (0,0)
    phi_can = torch.roll(phi_can, shifts=(-i_max, -j_max), dims=(0, 1))
    return phi_can


# ============================================================
# Observables
# ============================================================

def compute_observables(samples: torch.Tensor):
    """
    samples: (N, L, L)
    Returns dict, using Binder cumulant of magnetisation.
    """
    N, L, _ = samples.shape
    # per-configuration averages
    phi2 = (samples ** 2).view(N, -1).mean(dim=1)
    phi4 = (samples ** 4).view(N, -1).mean(dim=1)

    M = samples.view(N, -1).mean(dim=1)
    M2 = M ** 2
    M4 = M ** 4

    obs = {
        "<phi^2>": phi2.mean().item(),
        "<phi^4>": phi4.mean().item(),
        "<M>": M.mean().item(),
        "<|M|>": M.abs().mean().item(),
        "U4": (1.0 - M4.mean().item() / (3.0 * (M2.mean().item() ** 2 + 1e-12))),
    }
    return obs


# ============================================================
# Main driver
# ============================================================

def main():
    # Lattice / action parameters
    L = 16               # you can bump to 32 later
    m2 = -0.5
    lam = 3.0
    beta = 1.0

    model = Phi4Lattice2D(L=L, m2=m2, lam=lam, beta=beta)

    # HMC parameters
    n_samples = 2000
    n_burnin = 2000
    n_thin = 5
    eps = 0.05
    n_leapfrog = 10

    print("=== 2D phi^4 (LxL ring): HMC ===")
    samples = hmc_sample(
        model,
        n_samples=n_samples,
        n_burnin=n_burnin,
        n_thin=n_thin,
        eps=eps,
        n_leapfrog=n_leapfrog,
        init_phi=None,
        print_every=1000,
    )

    obs_raw = compute_observables(samples)
    print("\n[HMC] Observables (raw):")
    for k, v in obs_raw.items():
        print(f"  {k:8s} = {v:+.6f}")

    # Z2 canonical
    samples_Z2 = torch.stack(
        [canonicalise_Z2(cfg) for cfg in samples], dim=0
    )
    obs_Z2 = compute_observables(samples_Z2)
    print("\n[HMC] Observables after Z2 canonicalisation:")
    for k, v in obs_Z2.items():
        print(f"  {k:8s} = {v:+.6f}")

    # Z2 × translations canonical
    samples_Z2T = torch.stack(
        [canonicalise_Z2_translations(cfg) for cfg in samples], dim=0
    )
    obs_Z2T = compute_observables(samples_Z2T)
    print("\n[HMC] Observables after Z2 × translations canonicalisation:")
    for k, v in obs_Z2T.items():
        print(f"  {k:8s} = {v:+.6f}")

    # Save flattened configurations for flows
    out_file = f"phi4_2d_L{L}_m2{m2:.2f}_lam{lam:.2f}.npz"
    np.savez(
        out_file,
        raw=samples.numpy().reshape(samples.shape[0], -1),
        Z2=samples_Z2.numpy().reshape(samples_Z2.shape[0], -1),
        Z2T=samples_Z2T.numpy().reshape(samples_Z2T.shape[0], -1),
    )
    print(f"\n[SAVE] Wrote datasets to {out_file}")


if __name__ == "__main__":
    main()
