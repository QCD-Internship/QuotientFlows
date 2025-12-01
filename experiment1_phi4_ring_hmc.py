#!/usr/bin/env python3
import math
import numpy as np
import torch

# ============================================================
# Device selection: CUDA -> MPS (Apple Metal) -> CPU
# ============================================================
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS (Apple Metal)")
else:
    device = torch.device("cpu")
    print("Using CPU")


class Phi4Ring1D:
    r"""
    1D lattice φ^4 scalar field on a ring (periodic BCs).

    Action:
        S[φ] = ∑_i [ ½ (φ_{i+1} − φ_i)^2
                    + m²/2 φ_i²
                    + λ/4 φ_i⁴ ],
    with i+1 understood modulo L, and an overall factor β.
    """

    def __init__(self, L: int, m2: float, lam: float, beta: float = 1.0):
        self.L = L
        self.m2 = float(m2)
        self.lam = float(lam)
        self.beta = float(beta)

    def action(self, phi: torch.Tensor) -> torch.Tensor:
        """Compute S[φ] (scalar tensor). φ has shape (L,) on device."""
        phi_right = torch.roll(phi, shifts=-1, dims=0)
        kinetic = 0.5 * torch.sum((phi_right - phi) ** 2)
        mass_term = 0.5 * self.m2 * torch.sum(phi ** 2)
        quartic = 0.25 * self.lam * torch.sum(phi ** 4)
        return self.beta * (kinetic + mass_term + quartic)

    def grad_action(self, phi: torch.Tensor) -> torch.Tensor:
        """Gradient dS/dφ (shape (L,)) using periodic BCs."""
        phi_right = torch.roll(phi, shifts=-1, dims=0)
        phi_left = torch.roll(phi, shifts=1, dims=0)

        # d/dφ_i of ½(φ_{i+1}-φ_i)² + ½(φ_i-φ_{i-1})²
        grad_kin = (phi - phi_left) - (phi_right - phi)
        grad_mass = self.m2 * phi
        grad_quartic = self.lam * phi ** 3
        return self.beta * (grad_kin + grad_mass + grad_quartic)


# ============================================================
# HMC sampler
# ============================================================

def hmc_sample(
    model: Phi4Ring1D,
    n_samples: int = 5000,
    n_burnin: int = 2000,
    n_thin: int = 5,
    eps: float = 0.08,
    n_leapfrog: int = 10,
    init_phi: torch.Tensor | None = None,
    print_every: int = 1000,
):
    """
    Basic HMC sampler for 1D φ^4 on a ring.

    Returns:
        samples: tensor of shape (n_samples, L) on CPU.
    """
    L = model.L
    if init_phi is None:
        phi = torch.zeros(L, device=device)
    else:
        phi = init_phi.to(device)

    def kinetic(p):
        return 0.5 * torch.sum(p * p)

    total_steps = n_burnin + n_samples * n_thin
    kept = []
    accepts = 0
    proposals = 0

    for step in range(1, total_steps + 1):
        # fresh momentum
        p = torch.randn_like(phi)

        # current Hamiltonian
        S_current = model.action(phi)
        K_current = kinetic(p)
        H_current = S_current + K_current

        # leapfrog integration
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

        # negate momentum for reversibility
        p_prop = -p_prop

        # proposed Hamiltonian
        S_prop = model.action(phi_prop)
        K_prop = kinetic(p_prop)
        H_prop = S_prop + K_prop

        dH = (H_prop - H_current).item()
        accept_prob = math.exp(-dH) if dH > 0 else 1.0

        proposals += 1
        if np.random.rand() < accept_prob:
            phi = phi_prop
            accepts += 1

        # record sample after burn-in and thinning
        if step > n_burnin and (step - n_burnin) % n_thin == 0:
            kept.append(phi.detach().cpu().clone())

        if step % print_every == 0 or step == 1 or step == total_steps:
            acc_rate = accepts / proposals
            print(f"[HMC] Step {step}/{total_steps} | acc ≈ {acc_rate:.3f} | dH = {dH:+.3f}")

    samples = torch.stack(kept, dim=0)
    acc_rate = accepts / max(1, proposals)
    print(f"[HMC] Final acceptance rate ≈ {acc_rate:.3f}")
    print(f"[HMC] Generated {samples.shape[0]} samples of length {model.L}")
    return samples


# ============================================================
# Canonicalisation utilities
# ============================================================

def canonicalise_Z2(phi: torch.Tensor) -> torch.Tensor:
    """Canonicalise under global Z2: if M < 0 flip φ → −φ."""
    M = phi.mean()
    return -phi if M < 0 else phi


def canonicalise_Z2_translations(phi: torch.Tensor) -> torch.Tensor:
    """
    Canonicalise under Z2 × C_L (global sign × periodic shifts).

    1) Find i_max where |φ_i| is maximal.
    2) If φ[i_max] < 0, flip φ → −φ.
    3) Roll so that i_max → 0.
    """
    L = phi.shape[0]
    abs_phi = phi.abs()
    i_max = torch.argmax(abs_phi).item()
    phi_can = phi.clone()

    if phi_can[i_max] < 0:
        phi_can = -phi_can

    shift = -i_max  # bring i_max to index 0
    phi_can = torch.roll(phi_can, shifts=shift, dims=0)
    return phi_can


# ============================================================
# Observables
# ============================================================

def compute_observables(samples: torch.Tensor):
    """
    Compute <φ²>, <φ⁴>, M, |M|, Binder U4 for samples of shape (N, L).
    """
    N, L = samples.shape
    phi2 = (samples ** 2).mean(dim=1)
    phi4 = (samples ** 4).mean(dim=1)
    M = samples.mean(dim=1)
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
    # Model parameters
    L = 32
    m2 = -0.5
    lam = 3.0
    beta = 1.0

    model = Phi4Ring1D(L=L, m2=m2, lam=lam, beta=beta)

    # HMC parameters
    n_samples = 5000
    n_burnin = 2000
    n_thin = 5
    eps = 0.08
    n_leapfrog = 10

    print("=== 1D phi^4 ring: HMC ===")
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

    # Raw observables
    obs_raw = compute_observables(samples)
    print("\nObservables (raw HMC):")
    for k, v in obs_raw.items():
        print(f"  {k:8s} = {v:+.6f}")

    # Canonical Z2
    samples_Z2 = torch.stack(
        [canonicalise_Z2(cfg) for cfg in samples], dim=0
    )
    obs_Z2 = compute_observables(samples_Z2)
    print("\nObservables after Z2 canonicalisation:")
    for k, v in obs_Z2.items():
        print(f"  {k:8s} = {v:+.6f}")

    # Canonical Z2 × translations
    samples_Z2C = torch.stack(
        [canonicalise_Z2_translations(cfg) for cfg in samples], dim=0
    )
    obs_Z2C = compute_observables(samples_Z2C)
    print("\nObservables after Z2 × C_L canonicalisation:")
    for k, v in obs_Z2C.items():
        print(f"  {k:8s} = {v:+.6f}")

    # Save to disk for flow training
    out_file = f"phi4_ring_1d_L{L}_m2{m2:.2f}_lam{lam:.2f}.npz"
    np.savez(
        out_file,
        raw=samples.numpy(),
        Z2=samples_Z2.numpy(),
        Z2C=samples_Z2C.numpy(),
    )
    print(f"\n[SAVE] Wrote datasets to {out_file}")


if __name__ == "__main__":
    main()
