# plot_u1_2d_gaugequotient.py
import argparse
import numpy as np
import matplotlib.pyplot as plt
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-file", type=str, default="u1_2d_L16_beta2.00_gaugequotient.npz")
    parser.add_argument("--flows-file", type=str, default="u1_2d_L16_beta2.00_gaugequotient_flows_results.npz")
    args = parser.parse_args()

    data = np.load(args.data_file)
    flows = np.load(args.flows_file)

    L = int(data["L"])
    beta = float(data["beta"])

    X_raw = data["X_raw"]
    X_can = data["X_can"]
    N, D = X_raw.shape

    # We didn't store per-sample |P| in the flows file above; if you want the
    # full histogram comparison, you can easily extend the flows script to save them.
    # Here we'll just plot the training NLL curves.

    nll_X = flows["nll_X"]
    nll_Q = flows["nll_Q"]

    out_dir = f"figures_u1_2d_L{L}_beta{beta:.2f}_gaugequotient"
    os.makedirs(out_dir, exist_ok=True)

    epochs = np.arange(1, len(nll_X) + 1)

    # ---------------------------------------------------
    # Training NLL
    # ---------------------------------------------------
    plt.figure()
    plt.plot(epochs, nll_X, label="Flow on X")
    plt.plot(epochs, nll_Q, label="Flow on X/G_gauge")
    plt.xlabel("Epoch")
    plt.ylabel("NLL (train)")
    plt.title(f"2D U(1) gauge (L={L}, beta={beta:.2f})\nGauge-quotient vs baseline")
    plt.legend()
    plt.tight_layout()
    fname = os.path.join(out_dir, f"u1_2d_L{L}_beta{beta:.2f}_gaugequotient_training_nll.png")
    plt.savefig(fname, dpi=200)
    plt.close()
    print(f"[PLOT] Saved training NLL plot to {fname}")

    # ---------------------------------------------------
    # NLL gap (quotient - baseline)
    # ---------------------------------------------------
    plt.figure()
    gap = nll_Q - nll_X
    plt.plot(epochs, gap)
    plt.axhline(0.0, color="k", linestyle="--", linewidth=0.8)
    plt.xlabel("Epoch")
    plt.ylabel("NLL[X/G_gauge] - NLL[X]")
    plt.title(f"2D U(1) gauge (L={L}, beta={beta:.2f})\nNLL gap (gauge quotient vs baseline)")
    plt.tight_layout()
    fname = os.path.join(out_dir, f"u1_2d_L{L}_beta{beta:.2f}_gaugequotient_training_nll_gap.png")
    plt.savefig(fname, dpi=200)
    plt.close()
    print(f"[PLOT] Saved NLL gap plot to {fname}")

    # If you later add |P| samples from flows + HMC into the npz, you can add
    # histogram and diff plots here in the same style as your existing scripts.


if __name__ == "__main__":
    main()
