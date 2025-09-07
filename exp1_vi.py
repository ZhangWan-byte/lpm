# exp1_vi.py
"""
Per-graph VI baseline on batch datasets (same layout as exp1_train_batch / exp1_test_batch):
- Finds all *_test_* graphs under --setting_dir
- For each graph, fits LatentPositionModel with a **Gaussian variational approximation**:
    ELBO = log-likelihood (using variational means) - KL[q(z)||p(z)]
- Saves inferred LPs Z_hat (variational means mu) and reports:
    • Gromov–Wasserstein distance (GWD) vs. ground-truth positions  [NOTE: not squared]
    • LP-RMSE via orthogonal Procrustes

Usage:
  python exp1_vi.py --setting_dir ./sim_data_batch/A1_poly_feats \
    --out_dir results_vi/A1_poly_feats --latent_dim 2 --epochs 500 --lr 1e-2 --center
"""

import os
import json
import argparse
import numpy as np
import torch
import random
import torch.nn.functional as F

# Borrow helpers to match IO and metrics used elsewhere
from exp1_train_batch import (
    list_graph_dirs,
    load_true_positions,
    procrustes_rmse,
)
from models.utils import load_graph_dir
from exp1_test_batch import gwd_from_positions

# Use the LatentPositionModel definition provided in the repo (for consistent logits etc.)
try:
    from mle_arxiv_ogb import LatentPositionModel  # type: ignore
except Exception:
    from mle_arxiv_ogb_C3 import LatentPositionModel  # type: ignore


# --------------------------- Utils ---------------------------

def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _pair_logits_with_means(model, Zi: torch.Tensor, Zj: torch.Tensor) -> torch.Tensor:
    """
    Use the model's pairwise logits if available; otherwise default to -||Zi - Zj||^2
    with optional bias/scale if detected on the model.
    Zi, Zj: [M, d]
    """
    if hasattr(model, "logits"):
        try:
            return model.logits(Zi, Zj)
        except TypeError:
            pass  # fall back
    d2 = torch.sum((Zi - Zj) ** 2, dim=-1)
    bias = getattr(model, "bias", None)
    log_alpha = getattr(model, "log_alpha", None)
    if bias is None and log_alpha is None:
        return -d2
    b = bias if isinstance(bias, torch.Tensor) else torch.tensor(0.0, device=d2.device)
    if isinstance(log_alpha, torch.Tensor):
        alpha = F.softplus(log_alpha) + 1e-4
    else:
        alpha = torch.tensor(1.0, device=d2.device)
    return b - alpha * d2


# --------------------------- VI core ---------------------------

def vi_kl(mu: torch.Tensor, log_sigma: torch.Tensor) -> torch.Tensor:
    """
    KL[q(z)||p(z)] with p(z)=N(0,I), q(z)=N(mu, diag(sigma^2)), node-wise then summed.
    sigma = softplus(log_sigma) + eps for positivity.
    Returns a scalar tensor.
    """
    sigma = F.softplus(log_sigma) + 1e-6
    return 0.5 * (sigma.pow(2) + mu.pow(2) - 1.0 - 2.0 * sigma.log()).sum()


def vi_block_nll(
    base_model,  # LatentPositionModel (for consistent logits / parameters)
    MU: torch.Tensor,          # [N, d] variational means
    i_slice: slice,
    j_slice: slice,
    edge_blocks: tuple,
    undirected: bool,
) -> torch.Tensor:
    """
    Negative log-likelihood over all (i,j) in this block using variational means MU.
    If i_slice == j_slice and undirected: only use upper triangle (i<j).
    edge_blocks: (rows, cols) 0-based indices *relative to the block* for existing edges.
    """
    Zi = MU[i_slice]  # [bi, d]
    Zj = MU[j_slice]  # [bj, d]
    bi, bj = Zi.shape[0], Zj.shape[0]

    # Build pair views to reuse model's logits if it expects [M,d]
    zi = Zi[:, None, :].expand(bi, bj, -1).reshape(-1, Zi.shape[1])
    zj = Zj[None, :, :].expand(bi, bj, -1).reshape(-1, Zj.shape[1])
    logits = _pair_logits_with_means(base_model, zi, zj).reshape(bi, bj)

    Y = torch.zeros((bi, bj), device=MU.device, dtype=logits.dtype)
    rows, cols = edge_blocks
    if rows.numel() > 0:
        Y[rows, cols] = 1.0

    if undirected and (i_slice.start == j_slice.start) and (i_slice.stop == j_slice.stop):
        tri_mask = torch.ones((bi, bj), device=MU.device, dtype=torch.bool).triu(diagonal=1)
        Y = Y[tri_mask]
        logits = logits[tri_mask]

    # NLL over all pairs in this block
    nll = -(F.logsigmoid(logits) * Y + F.logsigmoid(-logits) * (1.0 - Y)).sum()
    return nll


def vi_fit_single_graph_full(
    edges_np: np.ndarray,
    N: int,
    undirected: bool,
    device: torch.device,
    d: int = 16,
    epochs: int = 50,
    lr: float = 5e-2,
    block_size: int = 512,
    seed: int = 0,
    kl_weight: float = 1.0,
) -> np.ndarray:
    """
    Fit LatentPositionModel with **Gaussian VI** using block processing over all O(N^2) pairs.
    ELBO = log-likelihood (using variational means) - kl_weight * KL[q(z)||p(z)].
    Returns Z_hat (variational means mu) as np.ndarray [N,d].
    """
    set_all_seeds(seed)
    # Base model supplies potential bias/scale/logit function; we optimize only VI params here.
    base = LatentPositionModel(N=N, d=d).to(device)
    # Variational parameters
    mu = torch.nn.Parameter(0.01 * torch.randn(N, d, device=device))
    log_sigma = torch.nn.Parameter(torch.full((N, d), -2.0, device=device))
    opt = torch.optim.Adam([mu, log_sigma] + [p for p in base.parameters() if p.requires_grad], lr=lr)

    # Prepare edge buckets per block
    if undirected:
        u = np.minimum(edges_np[:, 0], edges_np[:, 1])
        v = np.maximum(edges_np[:, 0], edges_np[:, 1])
        E = np.stack([u, v], axis=1)
    else:
        E = edges_np.copy()

    def edge_block_map(N, block):
        nb = (N + block - 1) // block
        buckets = {}
        for (a, b) in E:
            i_blk = a // block
            j_blk = b // block
            key = (min(i_blk, j_blk), max(i_blk, j_blk)) if undirected else (i_blk, j_blk)
            lst = buckets.get(key)
            if lst is None:
                buckets[key] = [(a, b)]
            else:
                lst.append((a, b))
        return buckets, nb

    buckets, nblocks = edge_block_map(N, block_size)

    # Number of optimized blocks (upper triangle if undirected)
    block_index_list = (
        [(i, j) for i in range(nblocks) for j in range(i, nblocks)] if undirected
        else [(i, j) for i in range(nblocks) for j in range(nblocks)]
    )
    num_blocks_effective = len(block_index_list)

    for _ in range(epochs):
        # Shuffle blocks for a bit of SGD flavor
        random.shuffle(block_index_list)

        for (ib, jb) in block_index_list:
            i_start, i_end = ib * block_size, min((ib + 1) * block_size, N)
            j_start, j_end = jb * block_size, min((jb + 1) * block_size, N)
            if undirected and (jb < ib):
                continue  # safety; though we only enumerated upper-tri

            # Collect edges inside this block
            pairs = buckets.get((min(ib, jb), max(ib, jb)) if undirected else (ib, jb), [])
            if len(pairs) > 0:
                arr = np.array(pairs, dtype=np.int64)
                r = torch.from_numpy(arr[:, 0] - i_start).to(device)
                c = torch.from_numpy(arr[:, 1] - j_start).to(device)
            else:
                r = torch.tensor([], dtype=torch.long, device=device)
                c = torch.tensor([], dtype=torch.long, device=device)

            nll = vi_block_nll(
                base_model=base,
                MU=mu,
                i_slice=slice(i_start, i_end),
                j_slice=slice(j_start, j_end),
                edge_blocks=(r, c),
                undirected=undirected,
            )

            # Add a fraction of KL each block to keep gradients well-scaled
            kl = vi_kl(mu, log_sigma) * (kl_weight / num_blocks_effective)

            loss = nll + kl  # minimize negative ELBO
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

    return mu.detach().cpu().numpy()


# --------------------------- CLI ---------------------------

def parse_args():
    ap = argparse.ArgumentParser(description="Per-graph VI baseline (LatentPositionModel, Gaussian VI) for latent positions.")
    ap.add_argument("--setting_dir", required=True, help="Folder like sim_data_batch/A1")
    ap.add_argument("--out_dir", required=True, help="Directory to save Z_hat and metrics")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)

    # Model / optimization
    ap.add_argument("--latent_dim", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--block_size", type=int, default=512, help="Pairwise VI block size")
    ap.add_argument("--kl_weight", type=float, default=1.0, help="Weight on KL term in ELBO")

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
    z_dir = os.path.join(args.out_dir, "Zhat_vi_full")
    os.makedirs(z_dir, exist_ok=True)

    # *_test_* graphs to mirror exp1_mle.py
    graph_dirs = [g for g in list_graph_dirs(args.setting_dir) if "_test_" in os.path.basename(g)]
    if args.max_graphs and len(graph_dirs) > args.max_graphs:
        graph_dirs = graph_dirs[:args.max_graphs]
    if not graph_dirs:
        raise SystemExit(f"No *_test_* graphs found in {args.setting_dir}")

    metrics = []
    print(f"Running VI (LatentPositionModel, Gaussian VI) on {len(graph_dirs)} test graphs. Saving Z_hat to: {z_dir}")

    for gdir in graph_dirs:
        edges, N, directed, base = load_graph_dir(gdir)
        undirected = not directed

        Z_true = load_true_positions(gdir)
        if Z_true is None or Z_true.shape[0] != N:
            print(f"[skip] {base}: missing/size-mismatch true positions.")
            continue

        # -------- Train per graph (VI; single inference) --------
        Z_hat = vi_fit_single_graph_full(
            edges_np=edges,
            N=N,
            undirected=undirected,
            device=device,
            d=args.latent_dim,
            epochs=args.epochs,
            lr=args.lr,
            block_size=args.block_size,
            seed=args.seed,
            kl_weight=args.kl_weight,
        )

        if args.center:
            Z_hat = Z_hat - Z_hat.mean(0, keepdims=True)

        # Save LPs
        zpath = os.path.join(z_dir, f"{base}_Zhat_vi.npy")
        np.save(zpath, Z_hat)

        # -------- Metrics (LP-RMSE and GWD) --------
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
        }
        with open(os.path.join(args.out_dir, "test_metrics_vi.json"), "w") as f:
            json.dump({"summary": summary, "details": metrics}, f, indent=2)
        print(f"\nSummary over {summary['num_graphs']} graphs: mean GWD={mean_gwd:.6f} | mean LP-RMSE={mean_lprmse:.6f}")
        print(f"Metrics saved to {os.path.join(args.out_dir, 'test_metrics_vi.json')}")
    else:
        print("No graphs evaluated.")

if __name__ == "__main__":
    main()
