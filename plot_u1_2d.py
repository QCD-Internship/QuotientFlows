#!/usr/bin/env python3
"""
plot_u1_2d.py

Plots for the 2D U(1) gauge experiment:

  1) Training NLL for NF on X and X/Λ
  2) NLL gap relative to X
  3) |P| histograms (HMC + 2 flows) with mean markers
  4) |P| histogram differences vs HMC

Expected files:

  - u1_2d_L{L}_beta{beta:.2f}.npz
      * "raw"         : (N, 2*L*L)
      * "trans"       : (N, 2*L*L)
      * "P_abs_raw"   : (N,)
      * "P_abs_trans" : (N,)

  - u1_2d_L{L}_beta{beta:.2f}_flows_results.npz
      * "nll_X", "nll_trans"
      * "P_hmc", "P_nf_X", "P_nf_trans"
"""

import os
import numpy as np
import matplotlib.pyplot as plt


def smooth(y, k=3):
    if k <= 1:
        return y
    kernel = np.ones(k, dtype=float) / float(k)
    return np.convolve(y, kernel, mode="same")


def main():
    L = 16
    beta = 2.0

    hmc_file = f"u1_2d_L{L}_beta{beta:.2f}.npz"
    flows_file = f"u1_2d_L{L}_beta{beta:.2f}_flows_results.npz"

    figdir = f"figures_u1_2d_L{L}_beta{beta:.2f}"
    os.makedirs(figdir, exist_ok=True)
    print(f"[INFO] Saving figures in: {figdir}/")

    print(f"[INFO] Loading HMC data from {hmc_file}")
    hmc = np.load(hmc_file)
    raw = hmc["raw"]
    P_abs_raw = hmc["P_abs_raw"]

    N, D = raw.shape
    assert D == 2 * L * L, f"Dim mismatch: expected {2*L*L}, got {D}"

    print(f"[INFO] Loading flow results from {flows_file}")
    res = np.load(flows_file)
    nll_X = res["nll_X"]
    nll_trans = res["nll_trans"]
    P_hmc = res["P_hmc"]
    P_nf_X = res["P_nf_X"]
    P_nf_trans = res["P_nf_trans"]

    # Sanity: P_hmc should match P_abs_raw
    # (small numerical differences are fine)
    # ==========================
    # 1) Training NLL vs epochs
    # ==========================
    epochs = np.arange(1, len(nll_X) + 1)

    plt.figure(figsize=(6, 4))
    plt.plot(epochs, nll_X, label="NF on X", lw=2)
    plt.plot(epochs, nll_trans, label="NF on X/Λ", lw=2)

    all_nll = np.concatenate([nll_X, nll_trans])
    y_min, y_max = all_nll.min(), all_nll.max()
    pad = 0.1 * (y_max - y_min + 1e-6)
    plt.ylim(y_min - pad, y_max + pad)

    plt.xlabel("Epoch")
    plt.ylabel("Negative log-likelihood per configuration")
    plt.title(f"2D U(1) (L={L}, beta={beta:.2f}): Training NLL")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    out_nll = os.path.join(
        figdir, f"u1_2d_L{L}_beta{beta:.2f}_training_nll.png"
    )
    plt.savefig(out_nll, dpi=200)
    plt.close()
    print(f"[PLOT] Saved training NLL plot to {out_nll}")

    # ==============================
    # 2) NLL gap relative to X
    # ==============================
    delta_trans = nll_X - nll_trans

    plt.figure(figsize=(6, 4))
    plt.plot(epochs, delta_trans, label="X − X/Λ", lw=2)
    plt.axhline(0.0, color="k", linestyle="--", linewidth=1)

    all_delta = delta_trans
    pad = 0.1 * (all_delta.max() - all_delta.min() + 1e-6)
    plt.ylim(all_delta.min() - pad, all_delta.max() + pad)

    plt.xlabel("Epoch")
    plt.ylabel(r"$\Delta \mathrm{NLL} = \mathrm{NLL}_X - \mathrm{NLL}_{X/\Lambda}$")
    plt.title(
        f"2D U(1) (L={L}, beta={beta:.2f}): NLL gain from quotient"
    )
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    out_gap = os.path.join(
        figdir, f"u1_2d_L{L}_beta{beta:.2f}_training_nll_gap.png"
    )
    plt.savefig(out_gap, dpi=200)
    plt.close()
    print(f"[PLOT] Saved NLL gap plot to {out_gap}")

    # ==========================================
    # 3) |P| histogram with mean markers
    # ==========================================
    plt.figure(figsize=(6, 4))
    bins = 40
    bin_range = (0.0, 1.0)

    # HMC reference
    plt.hist(
        P_hmc,
        bins=bins,
        range=bin_range,
        density=True,
        histtype="step",
        linewidth=2,
        label="HMC",
        color="black",
    )

    # Flows
    plt.hist(
        P_nf_X,
        bins=bins,
        range=bin_range,
        density=True,
        alpha=0.4,
        label="NF on X",
    )
    plt.hist(
        P_nf_trans,
        bins=bins,
        range=bin_range,
        density=True,
        alpha=0.4,
        label="NF on X/Λ (lifted)",
    )

    # Means
    mean_hmc = P_hmc.mean()
    mean_X = P_nf_X.mean()
    mean_trans = P_nf_trans.mean()

    plt.axvline(mean_hmc, color="black", linestyle="--", linewidth=1.5)
    plt.axvline(mean_X, color="C1", linestyle="--", linewidth=1.5)
    plt.axvline(mean_trans, color="C2", linestyle="--", linewidth=1.5)

    plt.xlim(*bin_range)
    plt.xlabel(r"$|P|$")
    plt.ylabel("Density")
    plt.title(f"2D U(1) (L={L}, beta={beta:.2f}): $|P|$ distribution")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    out_hist = os.path.join(
        figdir, f"u1_2d_L{L}_beta{beta:.2f}_Pabs_hist.png"
    )
    plt.savefig(out_hist, dpi=200)
    plt.close()
    print(f"[PLOT] Saved |P| histogram to {out_hist}")

    # ==========================================
    # 4) |P| histogram differences vs HMC
    # ==========================================
    hist_hmc, edges = np.histogram(
        P_hmc, bins=bins, range=bin_range, density=True
    )
    hist_X, _ = np.histogram(
        P_nf_X, bins=bins, range=bin_range, density=True
    )
    hist_trans, _ = np.histogram(
        P_nf_trans, bins=bins, range=bin_range, density=True
    )
    centers = 0.5 * (edges[:-1] + edges[1:])

    diff_X = smooth(hist_X - hist_hmc, k=3)
    diff_trans = smooth(hist_trans - hist_hmc, k=3)

    plt.figure(figsize=(6, 4))
    plt.plot(centers, diff_X, label="NF on X - HMC", lw=2)
    plt.plot(centers, diff_trans, label="NF on X/Λ - HMC", lw=2)

    diffs_all = np.concatenate([diff_X, diff_trans])
    dmax = np.max(np.abs(diffs_all))
    plt.ylim(-1.1 * dmax, 1.1 * dmax)

    plt.axhline(0.0, color="k", linestyle="--", linewidth=1)
    plt.xlabel(r"$|P|$")
    plt.ylabel("Density difference")
    plt.title(
        f"2D U(1) (L={L}, beta={beta:.2f}): $|P|$ difference vs HMC"
    )
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    out_diff = os.path.join(
        figdir, f"u1_2d_L{L}_beta{beta:.2f}_Pabs_hist_diff.png"
    )
    plt.savefig(out_diff, dpi=200)
    plt.close()
    print(f"[PLOT] Saved |P| histogram difference to {out_diff}")


if __name__ == "__main__":
    main()
