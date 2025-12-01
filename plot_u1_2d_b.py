#!/usr/bin/env python3
import os
import numpy as np
import matplotlib.pyplot as plt

def main():
    L = 16
    beta = 2.0

    data_file = f"u1_2d_L{L}_beta{beta:.2f}_b.npz"
    results_file = f"u1_2d_L{L}_beta{beta:.2f}_b_flows_results.npz"

    if not os.path.exists(data_file):
        raise FileNotFoundError(f"{data_file} not found. Run experiment_u1_2d_hmc_b.py first.")
    if not os.path.exists(results_file):
        raise FileNotFoundError(f"{results_file} not found. Run experiment_u1_2d_flows_b.py first.")

    print(f"[INFO] Saving figures in: figures_u1_2d_L{L}_beta{beta:.2f}_b/")
    out_dir = f"figures_u1_2d_L{L}_beta{beta:.2f}_b"
    os.makedirs(out_dir, exist_ok=True)

    data = np.load(data_file)
    results = np.load(results_file)

    nll_X = results["nll_X"]
    nll_trans = results["nll_trans"]
    P_hmc = results["P_hmc"]
    P_nf_X = results["P_nf_X"]
    P_nf_trans = results["P_nf_trans"]

    epochs = np.arange(1, len(nll_X) + 1)

    # ------------------------------------------------------------
    # Training NLL curves
    # ------------------------------------------------------------
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, nll_X, label="NF on X", lw=2)
    plt.plot(epochs, nll_trans, label="NF on X/Λ (Fourier slice)", lw=2)

    all_nll = np.concatenate([nll_X, nll_trans])
    y_min, y_max = all_nll.min(), all_nll.max()
    pad = 0.1 * (y_max - y_min + 1e-6)
    plt.ylim(y_min - pad, y_max + pad)

    plt.xlabel("Epoch")
    plt.ylabel("Negative log-likelihood per configuration")
    plt.title(f"2D U(1) (L={L}, β={beta:.2f}): Training NLL (Fourier slice)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    fname = f"{out_dir}/u1_2d_L{L}_beta{beta:.2f}_b_training_nll.png"
    plt.savefig(fname, dpi=200)
    plt.close()
    print(f"[PLOT] Saved training NLL plot to {fname}")

    # ------------------------------------------------------------
    # NLL gap: X - X/Λ
    # ------------------------------------------------------------
    plt.figure(figsize=(6, 4))
    gap = nll_X - nll_trans
    plt.plot(epochs, gap, lw=2)

    plt.axhline(0.0, color="k", lw=1, linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("ΔNLL = NLL(X) - NLL(X/Λ)")
    plt.title(f"2D U(1) (L={L}, β={beta:.2f}): NLL gap (Fourier slice)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    fname = f"{out_dir}/u1_2d_L{L}_beta{beta:.2f}_b_training_nll_gap.png"
    plt.savefig(fname, dpi=200)
    plt.close()
    print(f"[PLOT] Saved NLL gap plot to {fname}")

    # ------------------------------------------------------------
    # |P| histogram comparison
    # ------------------------------------------------------------
    bins = 40
    plt.figure(figsize=(6, 4))
    plt.hist(P_hmc, bins=bins, density=True, alpha=0.5, label="HMC")
    plt.hist(P_nf_X, bins=bins, density=True, alpha=0.5, label="NF on X")
    plt.hist(P_nf_trans, bins=bins, density=True, alpha=0.5,
             label="NF on X/Λ (Fourier slice)")

    plt.xlabel("|P|")
    plt.ylabel("Density")
    plt.title(f"2D U(1) (L={L}, β={beta:.2f}): |P| histogram")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    fname = f"{out_dir}/u1_2d_L{L}_beta{beta:.2f}_b_Pabs_hist.png"
    plt.savefig(fname, dpi=200)
    plt.close()
    print(f"[PLOT] Saved |P| histogram to {fname}")

    # ------------------------------------------------------------
    # |P| histogram differences (NF - HMC)
    # ------------------------------------------------------------
    hist_hmc, edges = np.histogram(P_hmc, bins=bins, range=(0.0, 1.0), density=True)
    hist_nf_X, _ = np.histogram(P_nf_X, bins=bins, range=(0.0, 1.0), density=True)
    hist_nf_trans, _ = np.histogram(P_nf_trans, bins=bins, range=(0.0, 1.0), density=True)

    centers = 0.5 * (edges[:-1] + edges[1:])

    plt.figure(figsize=(6, 4))
    plt.plot(centers, hist_nf_X - hist_hmc, label="NF on X - HMC", lw=2)
    plt.plot(centers, hist_nf_trans - hist_hmc,
             label="NF on X/Λ (Fourier) - HMC", lw=2)

    plt.axhline(0.0, color="k", lw=1, linestyle="--")
    plt.xlabel("|P|")
    plt.ylabel("Δ density")
    plt.title(f"2D U(1) (L={L}, β={beta:.2f}): |P| density difference")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    fname = f"{out_dir}/u1_2d_L{L}_beta{beta:.2f}_b_Pabs_hist_diff.png"
    plt.savefig(fname, dpi=200)
    plt.close()
    print(f"[PLOT] Saved |P| histogram difference to {fname}")


if __name__ == "__main__":
    main()
