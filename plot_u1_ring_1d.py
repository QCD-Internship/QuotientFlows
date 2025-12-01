#!/usr/bin/env python3
r"""
plot_u1_ring_1d.py

Plots for the 1D U(1) ring experiment:

  1) Training NLL for NF on X vs NF on X/U(1)
  2) NLL gap:  NLL_X - NLL_{X/G}
  3) |M| histogram: HMC (step) vs flows (filled) + mean lines
  4) |M| histogram differences: (NF - HMC)

Assumes the following files exist in the current directory:

  - u1_ring_L{L}_beta{beta:.2f}.npz
      * "theta_raw" : (N, L)
      * "theta_can" : (N, L)   [not used here]

  - u1_ring_L{L}_beta{beta:.2f}_flows_results.npz
      * "nll_raw"   : (n_epochs,)
      * "nll_q"     : (n_epochs,)
      * "M_hmc"     : (N,)
      * "M_nf_raw"  : (N_eval,)
      * "M_nf_q"    : (N_eval,)
"""

import os
import numpy as np
import matplotlib.pyplot as plt


def smooth(y: np.ndarray, k: int = 3) -> np.ndarray:
    """Simple moving-average smoothing with window size k."""
    if k <= 1:
        return y
    kernel = np.ones(k, dtype=float) / float(k)
    return np.convolve(y, kernel, mode="same")


def main():
    # --- match these to your HMC / flow scripts ---
    L = 32
    beta = 2.0

    hmc_file = f"u1_ring_L{L}_beta{beta:.2f}.npz"
    flows_file = f"u1_ring_L{L}_beta{beta:.2f}_flows_results.npz"

    # --- figure output directory ---
    figdir = f"figures_u1_L{L}_beta{beta:.2f}"
    os.makedirs(figdir, exist_ok=True)
    print(f"[INFO] Saving all figures to: {figdir}/")

    # ------------------------
    # Load HMC + flow results
    # ------------------------
    print(f"[INFO] Loading HMC data from {hmc_file}")
    hmc = np.load(hmc_file)
    theta_raw = hmc["theta_raw"]   # (N, L)
    N, L_check = theta_raw.shape
    if L_check != L:
        raise ValueError(f"L mismatch: expected {L}, got {L_check}")

    print(f"[INFO] Loading flow results from {flows_file}")
    res = np.load(flows_file)
    nll_raw = res["nll_raw"]       # (n_epochs,)
    nll_q = res["nll_q"]           # (n_epochs,)
    M_hmc = res["M_hmc"]           # (N,)
    M_nf_raw = res["M_nf_raw"]     # (N_eval,)
    M_nf_q = res["M_nf_q"]         # (N_eval,)

    # ==========================
    # 1) Training NLL vs epochs
    # ==========================
    epochs = np.arange(1, len(nll_raw) + 1)

    plt.figure(figsize=(6, 4))
    plt.plot(epochs, nll_raw, label="NF on X (raw)", lw=2)
    plt.plot(epochs, nll_q, label="NF on X/U(1) (canonical)", lw=2)

    all_nll = np.concatenate([nll_raw, nll_q])
    y_min, y_max = all_nll.min(), all_nll.max()
    pad = 0.1 * (y_max - y_min + 1e-6)
    plt.ylim(y_min - pad, y_max + pad)

    plt.xlabel("Epoch")
    plt.ylabel("Negative log-likelihood per configuration")
    plt.title(rf"U(1) ring (L={L}, $\beta={beta:.2f}$): Training NLL")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    out_nll = os.path.join(figdir, f"u1_L{L}_beta{beta:.2f}_training_nll.png")
    plt.savefig(out_nll, dpi=200)
    plt.close()
    print(f"[PLOT] Saved training NLL plot to {out_nll}")

    # ==============================
    # 2) NLL gap plot: NLL_X - NLL_G
    # ==============================
    delta_nll = np.array(nll_raw) - np.array(nll_q)

    plt.figure(figsize=(6, 4))
    plt.plot(epochs, delta_nll, lw=2)
    plt.xlabel("Epoch")
    plt.ylabel(r"$\Delta\mathrm{NLL} = \mathrm{NLL}_X - \mathrm{NLL}_{X/G}$")
    plt.title(rf"U(1) ring (L={L}, $\beta={beta:.2f}$): NLL gain from quotient")
    plt.axhline(0.0, color="k", linestyle="--", linewidth=1)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    out_gap = os.path.join(figdir, f"u1_L{L}_beta{beta:.2f}_training_nll_gap.png")
    plt.savefig(out_gap, dpi=200)
    plt.close()
    print(f"[PLOT] Saved NLL gap plot to {out_gap}")

    # ==========================================
    # 3) |M| histogram with HMC as step + means
    # ==========================================
    plt.figure(figsize=(6, 4))
    bins = 40
    bin_range = (0.0, 1.0)

    # HMC as reference step histogram
    plt.hist(
        M_hmc,
        bins=bins,
        range=bin_range,
        density=True,
        histtype="step",
        linewidth=2,
        label="HMC",
        color="black",
    )

    # Flows as filled histograms
    plt.hist(
        M_nf_raw,
        bins=bins,
        range=bin_range,
        density=True,
        alpha=0.4,
        label="NF on X",
    )
    plt.hist(
        M_nf_q,
        bins=bins,
        range=bin_range,
        density=True,
        alpha=0.4,
        label="NF on X/U(1) lifted",
    )

    # Vertical lines for means
    mean_hmc = M_hmc.mean()
    mean_raw = M_nf_raw.mean()
    mean_q = M_nf_q.mean()

    plt.axvline(mean_hmc, color="black", linestyle="--", linewidth=1.5)
    plt.axvline(mean_raw, color="C1", linestyle="--", linewidth=1.5)
    plt.axvline(mean_q, color="C2", linestyle="--", linewidth=1.5)

    plt.xlim(*bin_range)
    plt.xlabel(r"$|M|$")
    plt.ylabel("Density")
    plt.title(rf"U(1) ring (L={L}, $\beta={beta:.2f}$): $|M|$ distribution")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    out_hist = os.path.join(figdir, f"u1_L{L}_beta{beta:.2f}_Mabs_hist.png")
    plt.savefig(out_hist, dpi=200)
    plt.close()
    print(f"[PLOT] Saved |M| histogram to {out_hist}")

    # ==========================================
    # 4) |M| histogram differences vs HMC (smoothed)
    # ==========================================
    hist_hmc, edges = np.histogram(M_hmc, bins=bins, range=bin_range, density=True)
    hist_raw, _ = np.histogram(M_nf_raw, bins=bins, range=bin_range, density=True)
    hist_q, _ = np.histogram(M_nf_q, bins=bins, range=bin_range, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])

    # Smooth differences a bit to reduce jaggedness
    diff_raw = smooth(hist_raw - hist_hmc, k=3)
    diff_q = smooth(hist_q - hist_hmc, k=3)

    plt.figure(figsize=(6, 4))
    plt.plot(centers, diff_raw, label="NF on X - HMC", lw=2)
    plt.plot(centers, diff_q, label="NF on X/U(1) - HMC", lw=2)

    diffs_all = np.concatenate([diff_raw, diff_q])
    dmax = np.max(np.abs(diffs_all))
    plt.ylim(-1.1 * dmax, 1.1 * dmax)

    plt.axhline(0.0, color="k", linestyle="--", linewidth=1)
    plt.xlabel(r"$|M|$")
    plt.ylabel("Density difference")
    plt.title(rf"U(1) ring (L={L}, $\beta={beta:.2f}$): $|M|$ difference vs HMC")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    out_diff = os.path.join(figdir, f"u1_L{L}_beta{beta:.2f}_Mabs_hist_diff.png")
    plt.savefig(out_diff, dpi=200)
    plt.close()
    print(f"[PLOT] Saved |M| histogram difference to {out_diff}")


if __name__ == "__main__":
    main()
