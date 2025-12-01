#!/usr/bin/env python3
"""
experiment_u1_ring_1d_hmc.py

Generate reference ensembles for 1D U(1) lattice gauge theory on a ring using HMC.
We store both the raw angles and their canonical representatives (quotient by global U(1)).

Model: 1D U(1) gauge theory on an L-site ring with action
    S[theta] = -beta * sum_i cos(theta_{i+1} - theta_i),

where theta_i \in (-pi, pi] are angular variables.

Output:
    u1_ring_L{L}_beta{beta:.2f}.npz with keys:
        - "theta_raw":   shape (N_samples, L), raw HMC samples
        - "theta_can":   shape (N_samples, L), canonical reps (magnetisation real, >= 0)
"""

import numpy as np


def angle_wrap(x: np.ndarray) -> np.ndarray:
    """
    Wrap angles to the interval (-pi, pi].

    Args:
        x: array of angles (any shape)

    Returns:
        array with the same shape, wrapped into (-pi, pi]
    """
    return (x + np.pi) % (2.0 * np.pi) - np.pi


def action_u1_1d(theta: np.ndarray, beta: float) -> float:
    """
    U(1) action on a 1D ring:
        S[theta] = -beta * sum_i cos(theta_{i+1} - theta_i).

    Args:
        theta: array of shape (L,)
        beta:  coupling

    Returns:
        scalar action value
    """
    diff = np.roll(theta, -1) - theta  # theta_{i+1} - theta_i, periodic
    return -beta * np.sum(np.cos(diff))


def grad_action_u1_1d(theta: np.ndarray, beta: float) -> np.ndarray:
    """
    Gradient of the U(1) action wrt theta_j:

        S[theta] = -beta * sum_i cos(theta_{i+1} - theta_i)
        ∂S/∂θ_j = beta [ sin(θ_j - θ_{j-1}) - sin(θ_{j+1} - θ_j) ].

    Args:
        theta: array of shape (L,)
        beta:  coupling

    Returns:
        grad: array of shape (L,)
    """
    theta_plus = np.roll(theta, -1)   # θ_{j+1}
    theta_minus = np.roll(theta, 1)   # θ_{j-1}
    delta_minus = theta - theta_minus     # θ_j - θ_{j-1}
    delta_plus = theta_plus - theta       # θ_{j+1} - θ_j

    return beta * (np.sin(delta_minus) - np.sin(delta_plus))


def hmc_u1_1d(
    L: int,
    beta: float,
    n_therm: int,
    n_samples: int,
    n_skip: int,
    step_size: float,
    n_leapfrog: int,
    seed: int = 1234,
):
    """
    Run HMC for the 1D U(1) model.

    Args:
        L:           lattice size
        beta:        coupling
        n_therm:     number of thermalisation steps
        n_samples:   number of saved configurations
        n_skip:      number of HMC steps between saved configs
        step_size:   leapfrog step size
        n_leapfrog:  number of leapfrog steps per trajectory
        seed:        RNG seed

    Returns:
        theta_samples: array shape (n_samples, L)
        accept_rate:   float in [0, 1]
    """
    rng = np.random.default_rng(seed)
    theta = rng.uniform(low=-np.pi, high=np.pi, size=L)  # initial angles

    samples = []
    accepts = 0
    total_traj = n_therm + n_samples * n_skip

    for t in range(total_traj):
        # Sample momenta
        p = rng.normal(loc=0.0, scale=1.0, size=L)

        # Current state
        theta_curr = theta.copy()
        p_curr = p.copy()
        S_curr = action_u1_1d(theta_curr, beta)
        K_curr = 0.5 * np.sum(p_curr**2)

        # Leapfrog
        p = p - 0.5 * step_size * grad_action_u1_1d(theta, beta)
        for _ in range(n_leapfrog):
            theta = angle_wrap(theta + step_size * p)
            grad = grad_action_u1_1d(theta, beta)
            p = p - step_size * grad
        # Final half step
        p = p + 0.5 * step_size * grad_action_u1_1d(theta, beta)
        p = -p  # momentum flip for detailed balance

        S_prop = action_u1_1d(theta, beta)
        K_prop = 0.5 * np.sum(p**2)

        log_accept = (S_curr - S_prop) + (K_curr - K_prop)
        if np.log(rng.uniform()) < log_accept:
            # accept
            accepts += 1
        else:
            # reject; restore old state
            theta = theta_curr

        # Save after thermalisation, every n_skip steps
        if t >= n_therm and (t - n_therm) % n_skip == 0:
            samples.append(theta.copy())

        if (t + 1) % 1000 == 0 or t == total_traj - 1:
            print(
                f"[HMC] Step {t+1}/{total_traj} | "
                f"Accept so far ≈ {accepts / (t+1):.3f}"
            )

    theta_samples = np.stack(samples, axis=0)
    accept_rate = accepts / total_traj
    return theta_samples, accept_rate


def magnetisation_complex(theta: np.ndarray) -> complex:
    """
    Complex magnetisation M = (1/L) sum_j e^{i theta_j} for a single configuration.

    Args:
        theta: array shape (L,)

    Returns:
        complex magnetisation M
    """
    return np.exp(1j * theta).mean()


def canonicalise_u1(theta: np.ndarray) -> np.ndarray:
    """
    Canonical representative for global U(1):

    - Compute complex magnetisation M.
    - Rotate the configuration so that arg(M) = 0, and Re(M) >= 0.
      i.e., multiply by exp(-i arg(M)).

    This corresponds to quotienting by the global U(1) phase.

    Args:
        theta: array shape (L,)

    Returns:
        theta_can: array shape (L,), canonicalised angles
    """
    M = magnetisation_complex(theta)
    phase = np.angle(M)  # in (-pi, pi]
    theta_can = angle_wrap(theta - phase)
    # After this, M_can should have zero phase; if numerical issues, it's fine.
    return theta_can


def energy_density(theta: np.ndarray, beta: float) -> float:
    """
    Energy density e = S / L for a single configuration.

    Args:
        theta: array shape (L,)
        beta:  coupling

    Returns:
        scalar energy density
    """
    L = theta.shape[0]
    return action_u1_1d(theta, beta) / L


def compute_observables(theta_samples: np.ndarray, beta: float):
    """
    Compute simple observables over an ensemble.

    Args:
        theta_samples: array shape (N, L)
        beta:          coupling

    Returns:
        dict with:
            - "M_abs_mean": mean |M|
            - "E_mean":     mean energy density
    """
    N, L = theta_samples.shape
    M_abs = []
    E = []
    for n in range(N):
        th = theta_samples[n]
        M = magnetisation_complex(th)
        M_abs.append(np.abs(M))
        E.append(energy_density(th, beta))
    return {
        "M_abs_mean": float(np.mean(M_abs)),
        "E_mean": float(np.mean(E)),
    }


def main():
    # --- Hyperparameters ---
    L = 32
    beta = 2.0
    n_therm = 5000
    n_samples = 2000
    n_skip = 5
    step_size = 0.15
    n_leapfrog = 10
    seed = 1234

    print("=== HMC for 1D U(1) ring ===")
    print(
        f"L={L}, beta={beta}, n_therm={n_therm}, "
        f"n_samples={n_samples}, n_skip={n_skip}"
    )

    theta_raw, acc = hmc_u1_1d(
        L=L,
        beta=beta,
        n_therm=n_therm,
        n_samples=n_samples,
        n_skip=n_skip,
        step_size=step_size,
        n_leapfrog=n_leapfrog,
        seed=seed,
    )
    print(f"[HMC] Final acceptance rate ≈ {acc:.3f}")
    print(f"[HMC] Generated {theta_raw.shape[0]} samples of length {theta_raw.shape[1]}")

    # Canonicalise each configuration
    theta_can = np.stack([canonicalise_u1(th) for th in theta_raw], axis=0)

    # Quick observables
    obs_raw = compute_observables(theta_raw, beta)
    obs_can = compute_observables(theta_can, beta)
    print("Observables (raw):", obs_raw)
    print("Observables (canonical):", obs_can)

    # Save to npz
    out_name = f"u1_ring_L{L}_beta{beta:.2f}.npz"
    np.savez(out_name, theta_raw=theta_raw, theta_can=theta_can)
    print(f"[SAVE] Wrote HMC ensemble to {out_name}")


if __name__ == "__main__":
    main()
