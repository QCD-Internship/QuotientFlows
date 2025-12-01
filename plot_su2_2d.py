#!/usr/bin/env python
"""
plot_su2_2d.py

Produce training NLL plots and |P| histograms for the 2D SU(2) study.
"""

import numpy as np
import matplotlib.pyplot as plt

in_hmc = "su2_2d_L8_beta2.20.npz"
in_flows = "su2_2d_L8_beta2.20_flows_results.npz"

def main():
    hmc = np.load(in_hmc)
    flows = np.load(in_flows)

    L = int(hmc["L"])
    beta = float(hmc["beta"])

    # HMC observables
    P_hmc = hmc["P_abs"]

    # Flow observables
    P_X = flows["obs_X_P"]
    P_Xq = flows["obs_Xq_P"]

    nll_X = flows["nll_X"]
    nll_Xq = flows["nll_Xq"]

    # ---------------- NLL curves ----------------
    plt.figure(figsize=(6,4))
    epochs = np.arange(1, len(nll_X) + 1)
    plt.plot(epochs, nll_X, label="NF on X")
    plt.plot(epochs, nll_Xq, label="NF on X/Λ (canon)")
    plt.xlabel("Epoch")
    plt.ylabel("Training NLL per configuration")
    plt.title(f"2D SU(2) (L={L}, β={beta:.2f}): Training NLL")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"su2_2d_L{L}_beta{beta:.2f}_training_nll.png", dpi=200)

    # NLL gap
    plt.figure(figsize=(6,4))
    plt.plot(epochs, nll_X - nll_Xq)
    plt.axhline(0.0, color="k", linewidth=0.8)
    plt.xlabel("Epoch")
    plt.ylabel("ΔNLL = NLL_X − NLL_{X/Λ}")
    plt.title(f"2D SU(2) (L={L}, β={beta:.2f}): NLL gap")
    plt.tight_layout()
    plt.savefig(f"su2_2d_L{L}_beta{beta:.2f}_training_nll_gap.png", dpi=200)

    # ---------------- |P| histograms ----------------
    bins = 40
    plt.figure(figsize=(6,4))
    plt.hist(P_hmc, bins=bins, density=True, alpha=0.5, label="HMC")
    plt.hist(P_X, bins=bins, density=True, alpha=0.5, label="NF on X")
    plt.hist(P_Xq, bins=bins, density=True, alpha=0.5, label="NF on X/Λ")
    plt.xlabel(r"$|P|$")
    plt.ylabel("Density")
    plt.title(f"2D SU(2) (L={L}, β={beta:.2f}): |P| histogram")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"su2_2d_L{L}_beta{beta:.2f}_Pabs_hist.png", dpi=200)

    # Difference w.r.t HMC
    hist_hmc, edges = np.histogram(P_hmc, bins=bins, range=(0.0, 1.0), density=True)
    hist_X, _ = np.histogram(P_X, bins=edges, density=True)
    hist_Xq, _ = np.histogram(P_Xq, bins=edges, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])

    plt.figure(figsize=(6,4))
    plt.plot(centers, hist_X - hist_hmc, label="NF on X − HMC")
    plt.plot(centers, hist_Xq - hist_hmc, label="NF on X/Λ − HMC")
    plt.axhline(0.0, color="k", linewidth=0.8)
    plt.xlabel(r"$|P|$")
    plt.ylabel("Δ density")
    plt.title(f"2D SU(2) (L={L}, β={beta:.2f}): |P| density difference")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"su2_2d_L{L}_beta{beta:.2f}_Pabs_hist_diff.png", dpi=200)

    print("[PLOT] Saved NLL and |P| plots for 2D SU(2).")

if __name__ == "__main__":
    main()
