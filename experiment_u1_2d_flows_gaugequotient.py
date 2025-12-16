# experiment_u1_2d_flows_gaugequotient.py
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from math import pi
from tqdm import trange

# Device

def get_device():
    if torch.backends.mps.is_available():
        print("[Device] Using MPS (Apple Metal)")
        return torch.device("mps")
    if torch.cuda.is_available():
        print("[Device] Using CUDA")
        return torch.device("cuda")
    print("[Device] Using CPU")
    return torch.device("cpu")

# U(1) observables (same as in HMC code)

def angle_wrap(theta):
    return (theta + pi) % (2 * pi) - pi


def plaquette_angles(theta):
    """
    theta: (N, 2, L, L) or (2, L, L)
    """
    if theta.dim() == 3:
        theta = theta.unsqueeze(0)
    N, _, L, _ = theta.shape
    t0 = theta[:, 0]  # (N, L, L)
    t1 = theta[:, 1]  # (N, L, L)

    t1_xp_y = torch.roll(t1, shifts=-1, dims=1)
    t0_x_yp = torch.roll(t0, shifts=-1, dims=2)
    theta_p = t0 + t1_xp_y - t0_x_yp - t1
    return angle_wrap(theta_p)


def measure_observables(theta_batch, beta):
    """
    theta_batch: (N, 2, L, L) tensor
    """
    N, _, L, _ = theta_batch.shape
    E_plaquettes = []
    P_abs_vals = []

    for i in range(N):
        theta = theta_batch[i]

        theta_p = plaquette_angles(theta)  # (1, L, L)
        E_p = (1.0 - torch.cos(theta_p)).mean().item()
        E_plaquettes.append(E_p)

        theta_y = theta[1]                  # (L, L)
        U_y = torch.exp(1j * theta_y)       # (L, L)
        P_x = U_y.prod(dim=1)               # (L,)
        P = P_x.mean()
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


# RealNVP flow

class CouplingNet(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class AffineCoupling(nn.Module):
    def __init__(self, dim, mask, hidden_dim=512, scale=0.8):
        super().__init__()
        self.dim = dim
        self.register_buffer("mask", mask)
        self.net = CouplingNet(dim, hidden_dim, 2 * dim)
        self.scale = scale

    def forward(self, x):
        """
        x: (N, D)
        Returns: y, log_det_J
        """
        m = self.mask
        x_masked = x * m
        h = self.net(x_masked)
        log_s, t = h.chunk(2, dim=-1)
        log_s = torch.tanh(log_s) * self.scale
        s = torch.exp(log_s)

        y = x_masked + (1 - m) * (x * s + t)
        log_det = ((1 - m) * log_s).sum(dim=-1)
        return y, log_det

    def inverse(self, y):
        m = self.mask
        y_masked = y * m
        h = self.net(y_masked)
        log_s, t = h.chunk(2, dim=-1)
        log_s = torch.tanh(log_s) * self.scale
        s = torch.exp(log_s)

        x = y_masked + (1 - m) * ((y - t) / s)
        log_det = -((1 - m) * log_s).sum(dim=-1)
        return x, log_det


class RealNVP(nn.Module):
    def __init__(self, dim, n_coupling_layers=8, hidden_dim=512):
        super().__init__()
        self.dim = dim

        # Alternating binary masks
        masks = []
        for i in range(n_coupling_layers):
            if i % 2 == 0:
                m = torch.cat(
                    [torch.ones(dim // 2), torch.zeros(dim - dim // 2)]
                )
            else:
                m = torch.cat(
                    [torch.zeros(dim // 2), torch.ones(dim - dim // 2)]
                )
            masks.append(m)
        self.masks = nn.ParameterList(
            [nn.Parameter(m, requires_grad=False) for m in masks]
        )

        self.coupling_layers = nn.ModuleList(
            [AffineCoupling(dim, self.masks[i], hidden_dim)
             for i in range(n_coupling_layers)]
        )

        # constant log(2π) as a buffer (device-aware)
        import math
        self.register_buffer("log_2pi", torch.tensor(math.log(2 * math.pi)))

    def f(self, x):
        """
        Forward flow: x -> z, returns (z, log_det_J).
        """
        log_det_total = torch.zeros(x.shape[0], device=x.device)
        z = x
        for layer in self.coupling_layers:
            z, log_det = layer(z)
            log_det_total += log_det
        return z, log_det_total

    def f_inv(self, z):
        """
        Inverse flow: z -> x, returns (x, log_det_J_inv).
        """
        log_det_total = torch.zeros(z.shape[0], device=z.device)
        x = z
        for layer in reversed(self.coupling_layers):
            x, log_det = layer.inverse(x)
            log_det_total += log_det
        return x, log_det_total

    def log_prob(self, x):
        """
        log p(x) = log p(z) + log|det J_f(x)|
        with p(z) = N(0, I).
        """
        z, log_det = self.f(x)
        # standard normal log density in D dims
        # log p(z) = -0.5 * (||z||^2 + D log(2π))
        quad = z.pow(2).sum(dim=-1)
        log_pz = -0.5 * (quad + self.dim * self.log_2pi)
        return log_pz + log_det

    def sample(self, n, device):
        """
        Sample x by drawing z ~ N(0, I) and inverting the flow.
        """
        z = torch.randn(n, self.dim, device=device)
        x, _ = self.f_inv(z)
        return x


# Main

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default=None,
                        help="Input npz file from experiment_u1_2d_hmc_gaugequotient.py")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--n-coupling", type=int, default=8)
    parser.add_argument("--n-samples-eval", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = get_device()

    # Load data
    if args.file is None:
        # default file name for L=16,beta=2.0
        fname = "u1_2d_L16_beta2.00_gaugequotient.npz"
    else:
        fname = args.file

    data = np.load(fname)
    X_raw = data["X_raw"].astype(np.float32)   # (N, D)
    X_can = data["X_can"].astype(np.float32)   # (N, D)
    L = int(data["L"])
    beta = float(data["beta"])

    N, D = X_raw.shape
    print(f"[DATA] Loaded {N} configs of dimension {D} (L={L}) from {fname}")

    X_raw_t = torch.from_numpy(X_raw).to(device)
    X_can_t = torch.from_numpy(X_can).to(device)

    dataset_raw = TensorDataset(X_raw_t)
    loader_raw = DataLoader(dataset_raw, batch_size=args.batch_size, shuffle=True)

    dataset_can = TensorDataset(X_can_t)
    loader_can = DataLoader(dataset_can, batch_size=args.batch_size, shuffle=True)

    # Build flows
    flow_X = RealNVP(dim=D, n_coupling_layers=args.n_coupling, hidden_dim=args.hidden_dim).to(device)
    flow_Q = RealNVP(dim=D, n_coupling_layers=args.n_coupling, hidden_dim=args.hidden_dim).to(device)

    opt_X = torch.optim.Adam(flow_X.parameters(), lr=args.lr)
    opt_Q = torch.optim.Adam(flow_Q.parameters(), lr=args.lr)


    # Training loops
    nll_hist_X = []
    nll_hist_Q = []

    print("=== 2D U(1) gauge (gauge quotient): Flows on X and X/G_gauge ===")
    for epoch in trange(1, args.epochs + 1):
        # Flow on X
        flow_X.train()
        epoch_loss_X = 0.0
        n_batches_X = 0

        for (batch,) in loader_raw:
            opt_X.zero_grad()
            log_p = flow_X.log_prob(batch)
            loss = -log_p.mean()
            loss.backward()
            opt_X.step()

            epoch_loss_X += loss.item()
            n_batches_X += 1

        mean_nll_X = epoch_loss_X / max(1, n_batches_X)
        nll_hist_X.append(mean_nll_X)

        # Flow on quotient (canonicalised)
        flow_Q.train()
        epoch_loss_Q = 0.0
        n_batches_Q = 0

        for (batch,) in loader_can:
            opt_Q.zero_grad()
            log_p = flow_Q.log_prob(batch)
            loss = -log_p.mean()
            loss.backward()
            opt_Q.step()

            epoch_loss_Q += loss.item()
            n_batches_Q += 1

        mean_nll_Q = epoch_loss_Q / max(1, n_batches_Q)
        nll_hist_Q.append(mean_nll_Q)

        if epoch % 10 == 0 or epoch == 1:
            print(f"[Train] Epoch {epoch:4d}/{args.epochs} | "
                  f"NLL[X] ≈ {mean_nll_X:+.4f} | NLL[X/G_gauge] ≈ {mean_nll_Q:+.4f}")

    nll_hist_X = np.array(nll_hist_X, dtype=np.float32)
    nll_hist_Q = np.array(nll_hist_Q, dtype=np.float32)

    # Evaluate observables from flows (with lifting)
    def sample_and_measure(flow, is_quotient=False):
        flow.eval()
        with torch.no_grad():
            x = flow.sample(args.n_samples_eval, device)   # (N_eval, D)

        # Interpret x as angles, wrap to (-pi,pi]
        x = x.cpu().numpy()
        x = (x + np.pi) % (2 * np.pi) - np.pi
        x = torch.from_numpy(x).float()

        if is_quotient:
            # Lifting: apply random local U(1) gauge transformations
            N_eval = x.shape[0]
            theta = x.view(N_eval, 2, L, L)  # canonical reps

            # Random site phases alpha(x) ~ U(-pi, pi)
            alpha = (2 * np.pi * torch.rand(N_eval, L, L) - np.pi)
            # For each link, theta'_mu = theta_mu + alpha(x) - alpha(x+mu)
            theta_lift = torch.zeros_like(theta)
            for mu, (dx, dy) in enumerate([(1, 0), (0, 1)]):
                for ix in range(L):
                    for iy in range(L):
                        nx = (ix + dx) % L
                        ny = (iy + dy) % L
                        theta_link = theta[:, mu, ix, iy]
                        theta_prime = theta_link + alpha[:, ix, iy] - alpha[:, nx, ny]
                        theta_lift[:, mu, ix, iy] = ((theta_prime + pi) % (2 * pi) - pi)
            theta_use = theta_lift
        else:
            theta_use = x.view(-1, 2, L, L)

        theta_use = theta_use.to(device)
        E_mean, E_err, P_mean, P_err = measure_observables(theta_use, beta)
        return E_mean, E_err, P_mean, P_err

    print("\n[NF] Observables from trained flows (sampled):")
    Ep_X, dEp_X, P_X, dP_X = sample_and_measure(flow_X, is_quotient=False)
    print("  Flow on X:")
    print(f"    <E_plaquette> = {Ep_X:+.6f} ± {dEp_X:.6f}")
    print(f"    <|P|>          = {P_X:+.6f} ± {dP_X:.6f}")

    Ep_Q, dEp_Q, P_Q, dP_Q = sample_and_measure(flow_Q, is_quotient=True)
    print("  Flow on X/G_gauge (canon, lifted):")
    print(f"    <E_plaquette> = {Ep_Q:+.6f} ± {dEp_Q:.6f}")
    print(f"    <|P|>          = {P_Q:+.6f} ± {dP_Q:.6f}")

    # Save results
    out_file = fname.replace(".npz", "_flows_results.npz")
    np.savez(
        out_file,
        nll_X=nll_hist_X,
        nll_Q=nll_hist_Q,
        Ep_X=Ep_X,
        Ep_X_err=dEp_X,
        P_X=P_X,
        P_X_err=dP_X,
        Ep_Q=Ep_Q,
        Ep_Q_err=dEp_Q,
        P_Q=P_Q,
        P_Q_err=dP_Q,
        L=L,
        beta=beta,
    )
    print(f"[SAVE] Stored training curves and observables to {out_file}")


if __name__ == "__main__":
    main()
