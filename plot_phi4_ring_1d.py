#!/usr/bin/env python3
"""
plot_phi4_ring_1d.py

Plots for the 1D φ^4 ring experiment:

  1) Training NLL for NF on X, X/Z2, X/(Z2×C_L)
  2) NLL gaps relative to X
  3) |M| histograms (HMC + 3 flows) with mean markers
  4) |M| histogram differences vs HMC

Assumes the following files exist:

  - phi4_ring_1d_L{L}_m2{m2:.2f}_lam{lam:.2f}.npz
      * "raw" : (N, L)
      * "Z2"  : (N, L)  [not used here]
      * "Z2C" : (N, L)  [not used here]

  - phi4_ring_1d_L{L}_m2{m2:.2f}_lam{lam:.2f}_flows_results.npz
      * "nll_X"    : (n_epochs,)
      * "nll_Z2"   : (n_epochs,)
      * "nll_Z2C"  : (n_epochs,)
      * "M_hmc"    : (N_hmc,)
      * "M_nf_X"   : (N_eval,)
      * "M_nf_Z2"  : (N_eval,)
      * "M_nf_Z2C" : (N_eval,)
"""

import os
import numpy as np
import matplotlib.pyplot as plt


def smooth(y, k=3):
    """Simple moving-average smoothing with window size k."""
    if k <= 1:
        return y
    kernel = np.ones(k, dtype=float) / float(k)
    return np.convolve(y, kernel, mode="same")


def main():
    # Match parameters used in HMC/flow scripts
    L = 32
    m2 = -0.5
    lam = 3.0

    hmc_file = f"phi4_ring_1d_L{L}_m2{m2:.2f}_lam{lam:.2f}.npz"
    flows_file = f"phi4_ring_1d_L{L}_m2{m2:.2f}_lam{lam:.2f}_flows_results.npz"

    figdir = f"figures_phi4_1d_L{L}_m2{m2:.2f}_lam{lam:.2f}"
    os.makedirs(figdir, exist_ok=True)
    print(f"[INFO] Saving all figures to: {figdir}/")

    print(f"[INFO] Loading HMC data from {hmc_file}")
    hmc = np.load(hmc_file)
    phi_raw = hmc["raw"]   # (N, L)
    N, L_check = phi_raw.shape
    assert L_check == L, f"L mismatch: expected {L}, got {L_check}"

    print(f"[INFO] Loading flow results from {flows_file}")
    res = np.load(flows_file)
    nll_X = res["nll_X"]
    nll_Z2 = res["nll_Z2"]
    nll_Z2C = res["nll_Z2C"]
    M_hmc = res["M_hmc"]
    M_nf_X = res["M_nf_X"]
    M_nf_Z2 = res["M_nf_Z2"]
    M_nf_Z2C = res["M_nf_Z2C"]

    # ==========================
    # 1) Training NLL vs epochs
    # ==========================
    epochs = np.arange(1, len(nll_X) + 1)

    plt.figure(figsize=(6, 4))
    plt.plot(epochs, nll_X, label="NF on X", lw=2)
    plt.plot(epochs, nll_Z2, label="NF on X/Z2", lw=2)
    plt.plot(epochs, nll_Z2C, label="NF on X/(Z2×C_L)", lw=2)

    all_nll = np.concatenate([nll_X, nll_Z2, nll_Z2C])
    y_min, y_max = all_nll.min(), all_nll.max()
    pad = 0.1 * (y_max - y_min + 1e-6)
    plt.ylim(y_min - pad, y_max + pad)

    plt.xlabel("Epoch")
    plt.ylabel("Negative log-likelihood per configuration")
    plt.title(
        rf"1D $\phi^4$ ring (L={L}, $m^2={m2:.2f}$, $\lambda={lam:.2f}$): Training NLL"
    )
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    out_nll = os.path.join(
        figdir, f"phi4_L{L}_m2{m2:.2f}_lam{lam:.2f}_training_nll.png"
    )
    plt.savefig(out_nll, dpi=200)
    plt.close()
    print(f"[PLOT] Saved training NLL plot to {out_nll}")

    # ==============================
    # 2) NLL gaps relative to X
    # ==============================
    delta_Z2 = nll_X - nll_Z2
    delta_Z2C = nll_X - nll_Z2C

    plt.figure(figsize=(6, 4))
    plt.plot(epochs, delta_Z2, label="X − X/Z2", lw=2)
    plt.plot(epochs, delta_Z2C, label="X − X/(Z2×C_L)", lw=2)
    plt.axhline(0.0, color="k", linestyle="--", linewidth=1)

    all_delta = np.concatenate([delta_Z2, delta_Z2C])
    pad = 0.1 * (all_delta.max() - all_delta.min() + 1e-6)
    plt.ylim(all_delta.min() - pad, all_delta.max() + pad)

    plt.xlabel("Epoch")
    plt.ylabel(r"$\Delta\mathrm{NLL} = \mathrm{NLL}_X - \mathrm{NLL}_{\text{quotient}}$")
    plt.title(
        rf"1D $\phi^4$ ring (L={L}, $m^2={m2:.2f}$, $\lambda={lam:.2f}$): "
        r"NLL gain from quotient"
    )
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    out_gap = os.path.join(
        figdir, f"phi4_L{L}_m2{m2:.2f}_lam{lam:.2f}_training_nll_gap.png"
    )
    plt.savefig(out_gap, dpi=200)
    plt.close()
    print(f"[PLOT] Saved NLL gap plot to {out_gap}")

    # ==========================================
    # 3) |M| histogram with means
    # ==========================================
    plt.figure(figsize=(6, 4))
    bins = 40
    bin_range = (0.0, 1.0)

    # HMC as reference step
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
        M_nf_X,
        bins=bins,
        range=bin_range,
        density=True,
        alpha=0.4,
        label="NF on X",
    )
    plt.hist(
        M_nf_Z2,
        bins=bins,
        range=bin_range,
        density=True,
        alpha=0.4,
        label="NF on X/Z2 (lifted)",
    )
    plt.hist(
        M_nf_Z2C,
        bins=bins,
        range=bin_range,
        density=True,
        alpha=0.4,
        label="NF on X/(Z2×C_L) (lifted)",
    )

    # Means
    mean_hmc = M_hmc.mean()
    mean_X = M_nf_X.mean()
    mean_Z2 = M_nf_Z2.mean()
    mean_Z2C = M_nf_Z2C.mean()

    plt.axvline(mean_hmc, color="black", linestyle="--", linewidth=1.5)
    plt.axvline(mean_X, color="C1", linestyle="--", linewidth=1.5)
    plt.axvline(mean_Z2, color="C2", linestyle="--", linewidth=1.5)
    plt.axvline(mean_Z2C, color="C3", linestyle="--", linewidth=1.5)

    plt.xlim(*bin_range)
    plt.xlabel(r"$|M|$")
    plt.ylabel("Density")
    plt.title(
        rf"1D $\phi^4$ ring (L={L}, $m^2={m2:.2f}$, $\lambda={lam:.2f}$): "
        r"$|M|$ distribution"
    )
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    out_hist = os.path.join(
        figdir, f"phi4_L{L}_m2{m2:.2f}_lam{lam:.2f}_Mabs_hist.png"
    )
    plt.savefig(out_hist, dpi=200)
    plt.close()
    print(f"[PLOT] Saved |M| histogram to {out_hist}")

    # ==========================================
    # 4) |M| histogram differences vs HMC
    # ==========================================
    hist_hmc, edges = np.histogram(
        M_hmc, bins=bins, range=bin_range, density=True
    )
    hist_X, _ = np.histogram(
        M_nf_X, bins=bins, range=bin_range, density=True
    )
    hist_Z2, _ = np.histogram(
        M_nf_Z2, bins=bins, range=bin_range, density=True
    )
    hist_Z2C, _ = np.histogram(
        M_nf_Z2C, bins=bins, range=bin_range, density=True
    )
    centers = 0.5 * (edges[:-1] + edges[1:])

    diff_X = smooth(hist_X - hist_hmc, k=3)
    diff_Z2 = smooth(hist_Z2 - hist_hmc, k=3)
    diff_Z2C = smooth(hist_Z2C - hist_hmc, k=3)

    plt.figure(figsize=(6, 4))
    plt.plot(centers, diff_X, label="NF on X - HMC", lw=2)
    plt.plot(centers, diff_Z2, label="NF on X/Z2 - HMC", lw=2)
    plt.plot(centers, diff_Z2C, label="NF on X/(Z2×C_L) - HMC", lw=2)

    diffs_all = np.concatenate([diff_X, diff_Z2, diff_Z2C])
    dmax = np.max(np.abs(diffs_all))
    plt.ylim(-1.1 * dmax, 1.1 * dmax)

    plt.axhline(0.0, color="k", linestyle="--", linewidth=1)
    plt.xlabel(r"$|M|$")
    plt.ylabel("Density difference")
    plt.title(
        rf"1D $\phi^4$ ring (L={L}, $m^2={m2:.2f}$, $\lambda={lam:.2f}$): "
        r"$|M|$ difference vs HMC"
    )
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    out_diff = os.path.join(
        figdir, f"phi4_L{L}_m2{m2:.2f}_lam{lam:.2f}_Mabs_hist_diff.png"
    )
    plt.savefig(out_diff, dpi=200)
    plt.close()
    print(f"[PLOT] Saved |M| histogram difference to {out_diff}")


if __name__ == "__main__":
    main()
