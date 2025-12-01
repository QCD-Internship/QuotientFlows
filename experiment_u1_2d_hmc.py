#!/usr/bin/env python3
import math
import numpy as np
import torch

# ============================================================
# Device selection
# ============================================================

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


# ============================================================
# 2D U(1) gauge theory with Wilson plaquette action
# ============================================================

class U1Gauge2D:
    """
    2D U(1) gauge theory on an LxL periodic lattice with link angles
    theta_mu(x) in (-pi, pi], mu=0,1.

    Action:
        S[theta] = -beta * sum_p cos(theta_p)

    where theta_p is the oriented plaquette angle.
    """

    def __init__(self, L: int, beta: float):
        self.L = L
        self.beta = float(beta)

    def plaquette_angles(self, theta: torch.Tensor) -> torch.Tensor:
        """
        theta: (2, L, L)
          theta[0] = theta_x (μ=0)
          theta[1] = theta_y (μ=1)
        Returns theta_p: (L, L) plaquette angles.
        """
        theta_x = theta[0]
        theta_y = theta[1]

        theta_x_ip = torch.roll(theta_x, shifts=-1, dims=0)  # i+1,j
        theta_y_jp = torch.roll(theta_y, shifts=-1, dims=1)  # i,j+1

        theta_p = theta_x + theta_y_jp - theta_x_ip - theta_y
        return theta_p

    def action(self, theta: torch.Tensor) -> torch.Tensor:
        """
        theta: (2, L, L)
        Returns scalar S[theta].
        """
        theta_p = self.plaquette_angles(theta)
        return -self.beta * torch.sum(torch.cos(theta_p))

    def grad_action(self, theta: torch.Tensor) -> torch.Tensor:
        """
        Gradient dS/dtheta, same shape as theta.

        dS/dtheta_x(i,j) = beta [ sin(theta_p(i,j)) - sin(theta_p(i-1,j)) ]
        dS/dtheta_y(i,j) = beta [ sin(theta_p(i,j-1)) - sin(theta_p(i,j)) ]
        """
        theta_p = self.plaquette_angles(theta)
        sin_p = torch.sin(theta_p)

        grad_x = self.beta * (sin_p - torch.roll(sin_p, shifts=1, dims=0))
        grad_y = self.beta * (torch.roll(sin_p, shifts=1, dims=1) - sin_p)

        grad = torch.stack([grad_x, grad_y], dim=0)
        return grad


# ============================================================
# Utility: angle wrapping
# ============================================================

def wrap_angles(theta: torch.Tensor) -> torch.Tensor:
    """
    Wrap angles to (-pi, pi].
    """
    return (theta + math.pi) % (2 * math.pi) - math.pi


# ============================================================
# HMC sampler
# ============================================================

def hmc_sample(
    model: U1Gauge2D,
    n_samples: int = 2000,
    n_burnin: int = 4000,
    n_thin: int = 4,
    eps: float = 0.05,
    n_leapfrog: int = 10,
    init_theta: torch.Tensor | None = None,
    print_every: int = 1000,
):
    """
    Basic HMC for 2D U(1) gauge theory.

    Returns:
        samples: tensor of shape (n_samples, 2, L, L) on CPU.
    """
    L = model.L
    if init_theta is None:
        theta = torch.zeros(2, L, L, device=DEVICE)
    else:
        theta = wrap_angles(init_theta.to(DEVICE))

    def kinetic(p):
        return 0.5 * torch.sum(p * p)

    total_steps = n_burnin + n_samples * n_thin
    kept = []
    accepts = 0
    proposals = 0

    for step in range(1, total_steps + 1):
        p = torch.randn_like(theta)

        S_current = model.action(theta)
        K_current = kinetic(p)
        H_current = S_current + K_current

        theta_prop = theta.clone()
        p_prop = p.clone()

        grad_S = model.grad_action(theta_prop)
        p_prop = p_prop - 0.5 * eps * grad_S
        for lf_step in range(n_leapfrog):
            theta_prop = theta_prop + eps * p_prop
            theta_prop = wrap_angles(theta_prop)

            grad_S = model.grad_action(theta_prop)
            if lf_step != n_leapfrog - 1:
                p_prop = p_prop - eps * grad_S
        p_prop = p_prop - 0.5 * eps * grad_S
        p_prop = -p_prop  # momentum flip

        S_prop = model.action(theta_prop)
        K_prop = kinetic(p_prop)
        H_prop = S_prop + K_prop

        dH = (H_prop - H_current).item()
        accept_prob = math.exp(-dH) if dH > 0 else 1.0

        proposals += 1
        if np.random.rand() < accept_prob:
            theta = theta_prop
            accepts += 1

        if step > n_burnin and (step - n_burnin) % n_thin == 0:
            kept.append(theta.detach().cpu().clone())

        if step % print_every == 0 or step == 1 or step == total_steps:
            acc_rate = accepts / proposals
            print(f"[HMC] Step {step}/{total_steps} | acc ≈ {acc_rate:.3f} | dH = {dH:+.3f}")

    samples = torch.stack(kept, dim=0)
    acc_rate = accepts / max(1, proposals)
    print(f"[HMC] Final acceptance rate ≈ {acc_rate:.3f}")
    print(f"[HMC] Generated {samples.shape[0]} samples of size 2x{L}x{L}")
    return samples


# ============================================================
# Canonicalisation under translations (refined)
# ============================================================

def canonicalise_translations(theta: torch.Tensor, model: U1Gauge2D) -> torch.Tensor:
    """
    Canonicalise under lattice translations Λ on the torus using a *smooth* rule:

      1) Compute plaquette action density E(i,j) = 1 - cos(theta_p(i,j)).
      2) Shift E to be strictly positive: w = E - E_min + eps.
      3) Compute center-of-mass:
            i_cm = sum_{i,j} i * w(i,j) / sum_{i,j} w(i,j)
            j_cm = sum_{i,j} j * w(i,j) / sum_{i,j} w(i,j)
      4) Round to nearest lattice site and roll configuration
         so that (i_cm, j_cm) moves to (0,0).

    This is much less discontinuous than "argmax E".
    """
    L = model.L
    theta_p = model.plaquette_angles(theta)          # (L, L)
    E = 1.0 - torch.cos(theta_p)                     # (L, L)

    # Make weights strictly positive
    E_min = torch.min(E)
    w = E - E_min + 1e-6

    # Coordinates grid
    i_coords = torch.arange(L, device=theta.device).view(L, 1).expand(L, L)
    j_coords = torch.arange(L, device=theta.device).view(1, L).expand(L, L)

    denom = torch.sum(w)
    i_cm = torch.sum(i_coords * w) / denom
    j_cm = torch.sum(j_coords * w) / denom

    shift_i = int(torch.round(i_cm).item()) % L
    shift_j = int(torch.round(j_cm).item()) % L

    theta_can = torch.roll(theta, shifts=(-shift_i, -shift_j), dims=(1, 2))
    return theta_can


# ============================================================
# Observables
# ============================================================

def compute_observables(model: U1Gauge2D, samples: torch.Tensor):
    """
    samples: (N, 2, L, L)
    Returns:
      obs: dict with plaquette energy and |Polyakov loop|
      P_abs: numpy array of |P| per configuration
    """
    N, _, L, _ = samples.shape

    theta_p_all = []
    P_abs_list = []

    for n in range(N):
        cfg = samples[n].to(DEVICE)  # (2, L, L)
        theta_p = model.plaquette_angles(cfg)  # (L, L)
        theta_p_all.append(theta_p.detach().cpu())

        theta_x = cfg[0]  # (L, L)
        poly_angle = theta_x.sum(dim=0)  # (L,)
        cosA = torch.cos(poly_angle)
        sinA = torch.sin(poly_angle)
        P_re = cosA.mean()
        P_im = sinA.mean()
        P_abs = torch.sqrt(P_re * P_re + P_im * P_im)
        P_abs_list.append(P_abs.item())

    theta_p_all = torch.stack(theta_p_all, dim=0)  # (N, L, L)
    E_p = 1.0 - torch.cos(theta_p_all)

    E_mean = E_p.mean().item()
    P_abs_arr = np.array(P_abs_list, dtype=np.float64)
    P_mean = P_abs_arr.mean()

    obs = {
        "E_plaquette_mean": E_mean,
        "P_abs_mean": P_mean,
    }
    return obs, P_abs_arr


# ============================================================
# Main driver
# ============================================================

def main():
    L = 16
    beta = 2.0

    model = U1Gauge2D(L=L, beta=beta)

    n_samples = 2000
    n_burnin = 4000
    n_thin = 4
    eps = 0.05
    n_leapfrog = 10

    print("=== 2D U(1) gauge: HMC ===")
    samples = hmc_sample(
        model,
        n_samples=n_samples,
        n_burnin=n_burnin,
        n_thin=n_thin,
        eps=eps,
        n_leapfrog=n_leapfrog,
        init_theta=None,
        print_every=1000,
    )

    obs_raw, P_abs_raw = compute_observables(model, samples)
    print("\n[HMC] Observables (raw):")
    for k, v in obs_raw.items():
        print(f"  {k:16s} = {v:+.6f}")

    samples_trans = torch.stack(
        [canonicalise_translations(cfg, model) for cfg in samples], dim=0
    )
    obs_trans, P_abs_trans = compute_observables(model, samples_trans)
    print("\n[HMC] Observables after translation canonicalisation:")
    for k, v in obs_trans.items():
        print(f"  {k:16s} = {v:+.6f}")

    raw_flat = samples.numpy().reshape(samples.shape[0], -1)
    trans_flat = samples_trans.numpy().reshape(samples_trans.shape[0], -1)

    out_file = f"u1_2d_L{L}_beta{beta:.2f}.npz"
    np.savez(
        out_file,
        raw=raw_flat,
        trans=trans_flat,
        P_abs_raw=P_abs_raw,
        P_abs_trans=P_abs_trans,
    )
    print(f"\n[SAVE] Wrote datasets to {out_file}")


if __name__ == "__main__":
    main()
