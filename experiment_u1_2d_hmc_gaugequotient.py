# experiment_u1_2d_hmc_gaugequotient.py
import argparse
import numpy as np
import torch
from math import pi
from tqdm import trange

# -------------------------------------------------------
# Device
# -------------------------------------------------------

def get_device():
    if torch.backends.mps.is_available():
        print("[Device] Using MPS (Apple Metal)")
        return torch.device("mps")
    if torch.cuda.is_available():
        print("[Device] Using CUDA")
        return torch.device("cuda")
    print("[Device] Using CPU")
    return torch.device("cpu")

# -------------------------------------------------------
# U(1) gauge theory: plaquettes, action, observables
# -------------------------------------------------------

def angle_wrap(theta):
    """
    Map angles to (-pi, pi].
    theta: torch tensor
    """
    return (theta + pi) % (2 * pi) - pi


def plaquette_angles(theta):
    """
    Compute plaquette angles for 2D U(1).
    theta: (2, L, L) tensor, theta[0] = x-links, theta[1] = y-links
    plaquette at (x,y) = theta_x(x,y) + theta_y(x+1,y)
                         - theta_x(x,y+1) - theta_y(x,y)
    """
    t0 = theta[0]  # (L, L)
    t1 = theta[1]  # (L, L)
    # shift indices with periodic boundary conditions
    t1_xp_y = torch.roll(t1, shifts=-1, dims=0)   # y-links at (x+1,y)
    t0_x_yp = torch.roll(t0, shifts=-1, dims=1)   # x-links at (x,y+1)
    theta_p = t0 + t1_xp_y - t0_x_yp - t1
    return angle_wrap(theta_p)


def action(theta, beta):
    """
    Wilson plaquette action: S = - beta * sum cos(theta_p).
    theta: (2, L, L)
    """
    theta_p = plaquette_angles(theta)
    return -beta * torch.cos(theta_p).sum()


def gauge_force(theta, beta):
    """
    Gradient of S w.r.t. link angles theta, shape (2, L, L).
    Uses analytic derivatives:
      dS/dtheta0(x,y) = beta [ sin θ_p(x,y) - sin θ_p(x, y-1) ]
      dS/dtheta1(x,y) = beta [ sin θ_p(x-1,y) - sin θ_p(x,y) ]
    """
    theta_p = plaquette_angles(theta)
    sin_p = torch.sin(theta_p)

    # For x-links (mu=0)
    # sin θ_p(x,y) - sin θ_p(x, y-1)
    sin_p_y_minus = torch.roll(sin_p, shifts=1, dims=1)
    grad0 = beta * (sin_p - sin_p_y_minus)

    # For y-links (mu=1)
    # sin θ_p(x-1,y) - sin θ_p(x,y)
    sin_p_x_minus = torch.roll(sin_p, shifts=1, dims=0)
    grad1 = beta * (sin_p_x_minus - sin_p)

    grad = torch.stack([grad0, grad1], dim=0)
    return grad


def measure_observables(theta_batch, beta):
    """
    theta_batch: (N, 2, L, L) tensor of angles
    Returns:
      E_plaquette_mean, E_plaquette_err,
      P_abs_mean, P_abs_err
    """
    N, _, L, _ = theta_batch.shape
    theta_batch = theta_batch.clone()

    E_plaquettes = []
    P_abs_vals = []

    for i in range(N):
        theta = theta_batch[i]

        theta_p = plaquette_angles(theta)
        E_p = (1.0 - torch.cos(theta_p)).mean().item()
        E_plaquettes.append(E_p)

        # Polyakov loop along y-direction
        theta_y = theta[1]                     # (L, L)
        U_y = torch.exp(1j * theta_y)         # (L, L) complex
        P_x = U_y.prod(dim=1)                 # product over y, shape (L,)
        P = P_x.mean()                        # average over x
        P_abs_vals.append(torch.abs(P).item())

    E_arr = np.array(E_plaquettes)
    P_arr = np.array(P_abs_vals)

    def mean_err(x):
        m = x.mean()
        s = x.std(ddof=1) / np.sqrt(len(x))
        return m, s

    E_mean, E_err = mean_err(E_arr)
    P_mean, P_err = mean_err(P_arr)

    return E_mean, E_err, P_mean, P_err


# -------------------------------------------------------
# HMC step
# -------------------------------------------------------

def hmc_step(theta, beta, eps, n_steps):
    """
    Single HMC step.
    theta: (2, L, L) tensor
    """
    theta = theta.clone()
    p = torch.randn_like(theta)

    S = action(theta, beta)
    K = 0.5 * (p ** 2).sum()
    H = S + K

    theta_new = theta.clone()
    p_new = p.clone()

    # Half step in momentum
    grad = gauge_force(theta_new, beta)
    p_new = p_new - 0.5 * eps * grad

    # Full leapfrog steps
    for i in range(n_steps):
        theta_new = theta_new + eps * p_new
        theta_new = angle_wrap(theta_new)

        if i != n_steps - 1:
            grad = gauge_force(theta_new, beta)
            p_new = p_new - eps * grad

    # Final half step
    grad = gauge_force(theta_new, beta)
    p_new = p_new - 0.5 * eps * grad

    # Reverse momentum
    p_new = -p_new

    S_new = action(theta_new, beta)
    K_new = 0.5 * (p_new ** 2).sum()
    H_new = S_new + K_new

    dH = H_new - H
    accept_prob = torch.exp(-torch.clamp(dH, min=0.0))
    if torch.rand(()) < accept_prob:
        return theta_new, True, dH.item()
    else:
        return theta, False, dH.item()


# -------------------------------------------------------
# Gauge quotient: maximal-tree gauge fixing (local U(1))
# -------------------------------------------------------

def np_angle_wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def gauge_fix_maximal_tree(theta_np):
    """
    Maximal-tree gauge fixing for 2D U(1) configuration.
    theta_np: numpy array of shape (2, L, L), angles in (-pi, pi].
              theta_np[0,x,y] = x-link from (x,y) to (x+1,y)
              theta_np[1,x,y] = y-link from (x,y) to (x,y+1)
    We choose a spanning tree (all +x links except last column, all +y links
    in the first column) and define site phases alpha(x) so that tree links
    are gauge-fixed to zero. Remaining links carry the gauge-invariant flux.
    """
    _, L, _ = theta_np.shape
    alpha = np.zeros((L, L), dtype=np.float64)
    visited = np.zeros((L, L), dtype=bool)

    # BFS on a tree that does NOT use periodic wrap for tree edges
    from collections import deque
    q = deque()
    q.append((0, 0))
    visited[0, 0] = True

    # Directions: (mu, dx, dy)
    directions = [(0, 1, 0),  # x-direction
                  (1, 0, 1)]  # y-direction

    while q:
        x, y = q.popleft()
        for mu, dx, dy in directions:
            nx = x + dx
            ny = y + dy
            if nx < L and ny < L and not visited[nx, ny]:
                theta_link = theta_np[mu, x, y]
                # We want theta'_link = theta_link + alpha(x) - alpha(nx) = 0
                # => alpha(nx) = alpha(x) + theta_link
                alpha[nx, ny] = alpha[x, y] + theta_link
                visited[nx, ny] = True
                q.append((nx, ny))

    # Now apply gauge transform for ALL links including periodic ones
    theta_gf = np.zeros_like(theta_np)
    for mu, (dx, dy) in enumerate([(1, 0), (0, 1)]):
        for x in range(L):
            for y in range(L):
                nx = (x + dx) % L
                ny = (y + dy) % L
                theta_link = theta_np[mu, x, y]
                theta_prime = theta_link + alpha[x, y] - alpha[nx, ny]
                theta_gf[mu, x, y] = np_angle_wrap(theta_prime)

    return theta_gf


# -------------------------------------------------------
# Main
# -------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--L", type=int, default=16)
    parser.add_argument("--beta", type=float, default=2.0)
    parser.add_argument("--n-therm", type=int, default=2000)
    parser.add_argument("--n-samples", type=int, default=2000)
    parser.add_argument("--n-skip", type=int, default=5)
    parser.add_argument("--eps", type=float, default=0.1)
    parser.add_argument("--n-leapfrog", type=int, default=10)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    L = args.L
    beta = args.beta
    n_therm = args.n_therm
    n_samples = args.n_samples
    n_skip = args.n_skip
    eps = args.eps
    n_leapfrog = args.n_leapfrog

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = get_device()

    # Initial configuration: all links = 0
    theta = torch.zeros(2, L, L, device=device)

    total_steps = n_therm + n_samples * n_skip
    samples = []
    accepts = 0
    dH_list = []

    print(f"=== 2D U(1) gauge (gauge quotient via maximal-tree): HMC ===")
    for step in trange(1, total_steps + 1):
        theta, accepted, dH = hmc_step(theta, beta, eps, n_leapfrog)
        if accepted:
            accepts += 1
        dH_list.append(dH)

        if step % 1000 == 0 or step == 1:
            acc_rate = accepts / step
            print(f"[HMC] Step {step}/{total_steps} | acc ≈ {acc_rate:.3f} | dH = {dH:+.3f}")

        if step > n_therm and (step - n_therm) % n_skip == 0:
            samples.append(theta.detach().cpu().clone())

    acc_rate = accepts / total_steps
    print(f"[HMC] Final acceptance rate ≈ {acc_rate:.3f}")
    samples = torch.stack(samples, dim=0)  # (N, 2, L, L)
    N = samples.shape[0]
    print(f"[HMC] Generated {N} samples of size 2x{L}x{L}")

    # Observables on raw ensemble
    E_mean, E_err, P_mean, P_err = measure_observables(samples, beta)
    print("\n[HMC] Observables (raw):")
    print(f"  E_plaquette_mean = {E_mean:+.6f} ± {E_err:.6f}")
    print(f"  P_abs_mean       = {P_mean:+.6f} ± {P_err:.6f}")

    # Gauge-quotient canonicalisation (maximal-tree gauge)
    samples_np = samples.numpy()
    samples_gf_np = np.zeros_like(samples_np)
    for i in range(N):
        samples_gf_np[i] = gauge_fix_maximal_tree(samples_np[i])

    # Verify observables unchanged (gauge invariance)
    samples_gf = torch.from_numpy(samples_gf_np)
    E_mean_gf, E_err_gf, P_mean_gf, P_err_gf = measure_observables(samples_gf, beta)
    print("\n[HMC] Observables after gauge canonicalisation (should match raw):")
    print(f"  E_plaquette_mean = {E_mean_gf:+.6f} ± {E_err_gf:.6f}")
    print(f"  P_abs_mean       = {P_mean_gf:+.6f} ± {P_err_gf:.6f}")

    # Flatten for flows
    X_raw = samples_np.reshape(N, -1)
    X_can = samples_gf_np.reshape(N, -1)

    out_file = f"u1_2d_L{L}_beta{beta:.2f}_gaugequotient.npz"
    np.savez(
        out_file,
        X_raw=X_raw,
        X_can=X_can,
        E_plaquette_mean=E_mean,
        E_plaquette_err=E_err,
        P_abs_mean=P_mean,
        P_abs_err=P_err,
        L=L,
        beta=beta,
    )
    print(f"[SAVE] Wrote datasets to {out_file}")


if __name__ == "__main__":
    main()
