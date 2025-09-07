# exp1_vi.py
"""
Variational Inference (VI) baseline on batch datasets (same layout as exp1_train_batch / exp1_test_batch):
- Finds all *_test_* graphs under --setting_dir
- For each graph, fits a VI latent-position model by maximizing ELBO (Gaussian variational approx)
- Saves inferred LPs Z_hat (posterior means mu) and reports:
    • Gromov–Wasserstein distance (GWD) vs. ground-truth positions  [NOTE: not squared]
    • LP-RMSE via orthogonal Procrustes

Usage:
  python exp1_vi.py --setting_dir sim_data_batch/A1 --out_dir results_vi/A1 \
    --latent_dim 16 --epochs 50 --lr 5e-2 --batch_size 65536 --seed 0
"""

import os
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import random
import ot  # POT

# Borrow helpers to match IO and metrics used elsewhere
from exp1_train_batch import (
    list_graph_dirs,
    load_true_positions,
    _subsample_indices,
    _pairwise_euclidean,
    procrustes_rmse,
)
from models.utils import load_graph_dir


# --------------------------- VI Model ---------------------------

class LatentPositionModel_VI(torch.nn.Module):
    """
    Simple VI for latent positions z_i in R^d with a standard normal prior.
    Variational family: factorized Gaussians q(z_i) = N(mu_i, diag(sigma_i^2)).

    Likelihood (edges only): for edge (i,j),
      p(A_ij=1 | z) = sigmoid( -||z_i - z_j||^2 )
    We optimize ELBO = E_q[log p(edges|z)] - KL[q(z)||p(z)]
    using the mean parameters as a proxy in the likelihood term (common in amortized VI baselines).
    """
    def __init__(self, N: int, d: int, device: torch.device):
        super().__init__()
        self.N, self.d = N, d
        self.mu = torch.nn.Parameter(0.01 * torch.randn(N, d, device=device))
        # log_sigma parameterizes std via softplus for positivity and numerical stability
        self.log_sigma = torch.nn.Parameter(torch.full((N, d), -2.0, device=device))

    def pair_logits(self, pairs_2xM: torch.Tensor) -> torch.Tensor:
        """
        pairs_2xM: LongTensor with shape [2, M] containing (i, j) per column.
        Returns logits for Bernoulli edges: -||mu_i - mu_j||^2 (shape [M]).
        """
        i, j = pairs_2xM[0], pairs_2xM[1]
        di = self.mu[i] - self.mu[j]
        d2 = (di * di).sum(-1)
        return -d2  # logits

    def kl_divergence(self) -> torch.Tensor:
        """
        KL[q(z)||p(z)] with p(z)=N(0,I), q(z)=N(mu, diag(sigma^2)).
        Closed form: 0.5 * sum( sigma^2 + mu^2 - 1 - 2*log(sigma) )
        where sigma = softplus(log_sigma) + eps.
        """
        sigma = F.softplus(self.log_sigma) + 1e-6
        kl = 0.5 * (sigma.pow(2) + self.mu.pow(2) - 1.0 - 2.0 * sigma.log()).sum()
        return kl

    def elbo(self, edge_index_2xE: torch.Tensor, batch_size: int = 0) -> torch.Tensor:
        """
        Compute ELBO = log-likelihood(edges; using mu) - KL.
        If batch_size > 0, we use a minibatch estimator of the log-likelihood term
        with appropriate scaling by (E / m).
        """
        E = edge_index_2xE.size(1)
        if E == 0:
            # No edges: ELBO is just -KL
            return -self.kl_divergence()

        if batch_size and batch_size < E:
            # minibatch
            idx = torch.randint(0, E, (batch_size,), device=edge_index_2xE.device)
            eb = edge_index_2xE[:, idx]
            logits = self.pair_logits(eb)
            loglik_b = F.logsigmoid(logits).sum()
            scale = E / float(batch_size)
            log_likelihood = scale * loglik_b
        else:
            logits = self.pair_logits(edge_index_2xE)
            log_likelihood = F.logsigmoid(logits).sum()

        kl = self.kl_divergence()
        return log_likelihood - kl  # maximize ELBO


# --------------------------- Utils ---------------------------

def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@torch.no_grad()
def gwd_from_positions(
    Z_true: np.ndarray,
    Z_hat: np.ndarray,
    max_nodes: int = 2000,
    seed: int = 0,
    center: bool = False,
    max_iter: int = 200,
    tol: float = 1e-9,
) -> float:
    """
    POT non-entropic Gromov–Wasserstein **distance** (GW) between metric spaces induced by Z_true and Z_hat.
    (Compute GW^2 then return sqrt.)
    """
    N = Z_true.shape[0]
    idx = _subsample_indices(N, max_nodes, seed)
    Zt = Z_true[idx].astype(np.float64)
    Zh = Z_hat[idx].astype(np.float64)
    if center:
        Zt -= Zt.mean(0, keepdims=True)
        Zh -= Zh.mean(0, keepdims=True)

    Ct = _pairwise_euclidean(Zt)
    Ch = _pairwise_euclidean(Zh)
    k = Ct.shape[0]
    p = np.ones((k,), dtype=np.float64) / k
    q = p
    G0 = np.outer(p, q)  # deterministic init

    try:
        gw2 = ot.gromov_wasserstein2(
            Ct, Ch, p, q, loss_fun="square_loss", max_iter=max_iter, tol=tol, verbose=False, G0=G0
        )
    except AttributeError:
        gw2 = ot.gromov.gromov_wasserstein2(
            Ct, Ch, p, q, loss_fun="square_loss", max_iter=max_iter, tol=tol, verbose=False, G0=G0
        )
    return float(np.sqrt(max(gw2, 0.0)))


def edges_to_edgeindex_2xE(edges_np: np.ndarray, device: torch.device, undirected: bool) -> torch.Tensor:
    """
    Convert edge list (E,2) to edge_index [2,E]. If undirected, keep as provided.
    Assumes edges already reflect the observed graph (no self-loops).
    """
    if edges_np.size == 0:
        return torch.empty(2, 0, dtype=torch.long, device=device)
    ei = torch.from_numpy(edges_np.astype(np.int64)).to(device)
    if ei.ndim == 2 and ei.shape[1] == 2:
        ei = ei.t().contiguous()  # [2,E]
    return ei


# --------------------------- Training ---------------------------

def vi_fit_single_graph(
    edges_np: np.ndarray,
    N: int,
    device: torch.device,
    d: int = 16,
    epochs: int = 50,
    lr: float = 5e-2,
    batch_size: int = 0,
    seed: int = 0,
    undirected: bool = True,
) -> np.ndarray:
    """
    Fit VI on a single graph by maximizing ELBO with Adam.
    Returns Z_hat (posterior means mu) as np.ndarray [N,d].
    """
    set_all_seeds(seed)
    model = LatentPositionModel_VI(N, d, device=device).to(device)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)

    edge_index = edges_to_edgeindex_2xE(edges_np, device=device, undirected=undirected)

    for _ in range(epochs):
        model.train()
        elbo = model.elbo(edge_index, batch_size=batch_size)
        loss = -elbo  # maximize ELBO
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    return model.mu.detach().cpu().numpy()


# --------------------------- CLI ---------------------------

def parse_args():
    ap = argparse.ArgumentParser(description="Per-graph VI baseline (Gaussian VI) for latent positions.")
    ap.add_argument("--setting_dir", required=True, help="Folder like sim_data_batch/A1")
    ap.add_argument("--out_dir", required=True, help="Directory to save Z_hat and metrics")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)

    # VI / optimization
    ap.add_argument("--latent_dim", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--batch_size", type=int, default=0, help="Minibatch size over edges for ELBO (0=full-batch)")

    # Metrics
    ap.add_argument("--gwd_nodes", type=int, default=2000, help="Max nodes used in GWD (subsample)")
    ap.add_argument("--lp_nodes", type=int, default=5000, help="Max nodes used in LP-RMSE (subsample)")
    ap.add_argument("--center", action="store_true", help="Center embeddings before metrics")

    # Control
    ap.add_argument("--max_graphs", type=int, default=0, help="0=all test graphs, else limit")
    return ap.parse_args()


# --------------------------- Main ---------------------------

def main():
    args = parse_args()
    set_all_seeds(args.seed)
    device = torch.device(args.device)

    os.makedirs(args.out_dir, exist_ok=True)
    z_dir = os.path.join(args.out_dir, "Zhat_vi")
    os.makedirs(z_dir, exist_ok=True)

    # *_test_* graphs to mirror other baselines
    graph_dirs = [g for g in list_graph_dirs(args.setting_dir) if "_test_" in os.path.basename(g)]
    if args.max_graphs and len(graph_dirs) > args.max_graphs:
        graph_dirs = graph_dirs[:args.max_graphs]
    if not graph_dirs:
        raise SystemExit(f"No *_test_* graphs found in {args.setting_dir}")

    metrics = []
    print(f"Running VI (Gaussian) on {len(graph_dirs)} test graphs. Saving Z_hat to: {z_dir}")

    for gdir in graph_dirs:
        edges, N, directed, base = load_graph_dir(gdir)
        undirected = not directed

        Z_true = load_true_positions(gdir)
        if Z_true is None or Z_true.shape[0] != N:
            print(f"[skip] {base}: missing/size-mismatch true positions.")
            continue

        # -------- Train VI per graph (single inference) --------
        Z_hat = vi_fit_single_graph(
            edges_np=edges,
            N=N,
            device=device,
            d=args.latent_dim,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            seed=args.seed,
            undirected=undirected,
        )

        if args.center:
            Z_hat = Z_hat - Z_hat.mean(0, keepdims=True)

        # Save LPs
        zpath = os.path.join(z_dir, f"{base}_Zhat_vi.npy")
        np.save(zpath, Z_hat)

        # -------- Metrics (GWD and LP-RMSE) --------
        k = min(Z_true.shape[0], args.lp_nodes) if args.lp_nodes > 0 else Z_true.shape[0]
        idx = np.arange(Z_true.shape[0]) if k == Z_true.shape[0] else np.sort(
            np.random.default_rng(args.seed).choice(Z_true.shape[0], size=k, replace=False)
        )
        lp_rmse, Z_reduced = procrustes_rmse(Z_true[idx], Z_hat[idx], center=True, scale=False)

        gwd = gwd_from_positions(
            Z_true=Z_true, Z_hat=Z_reduced, max_nodes=args.gwd_nodes, seed=args.seed, center=args.center
        )

        print(f"{base}: GWD={gwd:.6f} | LP-RMSE={lp_rmse:.6f} | Z_hat={zpath}")
        metrics.append({
            "graph": base,
            "n_nodes": int(N),
            "gwd": float(gwd),
            "lp_rmse": float(lp_rmse),
            "zhat_path": zpath,
        })

    # -------- Summary --------
    if metrics:
        mean_gwd = float(np.mean([m["gwd"] for m in metrics]))
        mean_lprmse = float(np.mean([m["lp_rmse"] for m in metrics]))
        summary = {
            "num_graphs": len(metrics),
            "mean_gwd": mean_gwd,
            "mean_lp_rmse": mean_lprmse,
            "args": {
                "latent_dim": args.latent_dim,
                "epochs": args.epochs,
                "lr": args.lr,
                "batch_size": args.batch_size,
            },
        }
        with open(os.path.join(args.out_dir, "test_metrics_vi.json"), "w") as f:
            json.dump({"summary": summary, "details": metrics}, f, indent=2)
        print(f"\nSummary over {summary['num_graphs']} graphs: "
              f"mean GWD={mean_gwd:.6f} | mean LP-RMSE={mean_lprmse:.6f}")
    else:
        print("No graphs evaluated.")

if __name__ == "__main__":
    main()
