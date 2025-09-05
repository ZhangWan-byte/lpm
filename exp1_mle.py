# exp1_mle.py
"""
Per-graph MLE baseline on batch datasets (same layout as exp1_train_batch / exp1_test_batch):
- Finds all *_test_* graphs under --setting_dir
- For each graph, fits LatentPositionModel by **full MLE (no negative sampling)**
- Saves inferred LPs Z_hat as .npy and reports:
    • Gromov–Wasserstein distance (GWD) vs. ground-truth positions  [NOTE: not squared]
    • LP-RMSE via orthogonal Procrustes

Usage:
  python exp1_mle.py --setting_dir ./sim_data_batch/A1_poly_feats \
    --out_dir results_mle/A1_poly_feats --latent_dim 2 --epochs 500 --lr 1e-2 --center
"""

import os
import json
import argparse
import numpy as np
import torch
import random
import torch.nn.functional as F
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

# Use the LatentPositionModel definition provided in the repo
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
    (We compute GW^2 and return its square root.)
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


def _get_embedding_parameter(model: torch.nn.Module, N: int, d: int, device: torch.device) -> torch.nn.Parameter:
    """
    Be robust to variations: prefer model.mu (as in LatentPositionModel),
    otherwise fall back to model.Z (if present) or register our own.
    """
    emb = None
    if hasattr(model, "mu") and isinstance(getattr(model, "mu"), torch.Tensor):
        emb = getattr(model, "mu")
        if not isinstance(emb, torch.nn.Parameter):
            emb = torch.nn.Parameter(emb.to(device))
            setattr(model, "mu", emb)
    elif hasattr(model, "Z") and isinstance(getattr(model, "Z"), torch.Tensor):
        emb = getattr(model, "Z")
        if not isinstance(emb, torch.nn.Parameter):
            emb = torch.nn.Parameter(emb.to(device))
            setattr(model, "Z", emb)
    else:
        emb = torch.nn.Parameter(0.01 * torch.randn(N, d, device=device))
        setattr(model, "mu", emb)
    emb.requires_grad_(True)
    return emb


def _pair_logits(model, zi: torch.Tensor, zj: torch.Tensor) -> torch.Tensor:
    """
    Use the model's pairwise logits if available; otherwise default to -||zi - zj||^2
    with optional bias/scale if detected on the model.
    """
    if hasattr(model, "logits"):
        try:
            return model.logits(zi, zj)
        except TypeError:
            pass  # fall back
    # generic fallback: bias - alpha * ||zi - zj||^2 if attributes exist
    d2 = torch.sum((zi - zj) ** 2, dim=-1)
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


def full_mle_loss_block(
    model,
    Z: torch.Tensor,
    i_slice: slice,
    j_slice: slice,
    edge_blocks: tuple,
    undirected: bool,
) -> torch.Tensor:
    """
    Compute negative log-likelihood for the (i_slice, j_slice) block.
    If i_slice == j_slice and undirected: only use upper triangle (i<j).
    edge_blocks: (rows, cols) tensor indices (relative to block) for existing edges within the block.
    """
    Zi = Z[i_slice]  # [bi, d]
    Zj = Z[j_slice]  # [bj, d]
    bi, bj = Zi.shape[0], Zj.shape[0]

    # compute pairwise logits for block
    # broadcast to bi*bj pairs efficiently
    # (Zi[:,None,:] - Zj[None,:,:])^2 -> sum over dim
    # We'll build zi and zj "pairs" views to reuse model logits if it expects [M,d]
    zi = Zi[:, None, :].expand(bi, bj, -1).reshape(-1, Zi.shape[1])
    zj = Zj[None, :, :].expand(bi, bj, -1).reshape(-1, Zj.shape[1])
    logits = _pair_logits(model, zi, zj).reshape(bi, bj)

    # labels matrix Y in this block
    Y = torch.zeros((bi, bj), device=Z.device, dtype=logits.dtype)
    rows, cols = edge_blocks
    if rows.numel() > 0:
        Y[rows, cols] = 1.0

    # mask for undirected diagonal block: use upper triangle only (i<j)
    if undirected and (i_slice.start == j_slice.start) and (i_slice.stop == j_slice.stop):
        tri_mask = torch.ones((bi, bj), device=Z.device, dtype=torch.bool).triu(diagonal=1)
        Y = Y[tri_mask]
        logits = logits[tri_mask]

    # Bernoulli log-likelihood over all pairs in this block
    # sum [ Y * log(sigmoid(l)) + (1-Y) * log(sigmoid(-l)) ]
    nll = -(F.logsigmoid(logits) * Y + F.logsigmoid(-logits) * (1.0 - Y)).sum()
    return nll


def mle_fit_single_graph_full(
    edges_np: np.ndarray,
    N: int,
    undirected: bool,
    device: torch.device,
    d: int = 16,
    epochs: int = 50,
    lr: float = 5e-2,
    block_size: int = 512,
    seed: int = 0,
) -> np.ndarray:
    """
    Fit LatentPositionModel on a single graph using **full MLE** over all (i,j) pairs.
    No negative sampling. Uses block processing for O(N^2) pairs.
    Returns Z_hat as np.ndarray [N,d].
    """
    set_all_seeds(seed)
    model = LatentPositionModel(N=N, d=d).to(device)
    Zparam = _get_embedding_parameter(model, N, d, device)

    # Optimize all trainable parameters of the model (embeddings + any bias/scale)
    params = [p for p in model.parameters() if p.requires_grad]
    if Zparam not in params:
        params.append(Zparam)
    opt = torch.optim.Adam(params, lr=lr)

    # Build edge lookup per block for fast labeling
    # create directed edge list for convenience
    if undirected:
        # normalize to i<j to avoid duplicates
        u = np.minimum(edges_np[:, 0], edges_np[:, 1])
        v = np.maximum(edges_np[:, 0], edges_np[:, 1])
        E = np.stack([u, v], axis=1)
    else:
        E = edges_np.copy()
    # bucket edges by block indices to avoid scanning the full edge list each step
    def edge_block_map(N, block):
        nb = (N + block - 1) // block
        buckets = {}
        for (a, b) in E:
            i_blk = a // block
            j_blk = b // block if not undirected else b // block
            key = (min(i_blk, j_blk), max(i_blk, j_blk)) if undirected else (i_blk, j_blk)
            lst = buckets.get(key)
            if lst is None:
                buckets[key] = [(a, b)]
            else:
                lst.append((a, b))
        return buckets, nb

    buckets, nblocks = edge_block_map(N, block_size)

    for _ in range(epochs):
        model.train()
        # optional: shuffle block order for a bit of SGD flavor
        block_order = [(i, j) for i in range(nblocks) for j in range(i, nblocks)] if undirected else \
                      [(i, j) for i in range(nblocks) for j in range(nblocks)]
        random.shuffle(block_order)

        total_nll = 0.0
        for (ib, jb) in block_order:
            i_start, i_end = ib * block_size, min((ib + 1) * block_size, N)
            j_start, j_end = jb * block_size, min((jb + 1) * block_size, N)
            if undirected and (jb < ib):
                continue  # only upper (including diagonal) blocks

            # collect edges inside this block
            pairs = buckets.get((min(ib, jb), max(ib, jb)) if undirected else (ib, jb), [])
            if len(pairs) > 0:
                arr = np.array(pairs, dtype=np.int64)
                # convert to block-local indices (rows & cols)
                r = torch.from_numpy(arr[:, 0] - i_start).to(device)
                c = torch.from_numpy(arr[:, 1] - j_start).to(device)
            else:
                r = torch.tensor([], dtype=torch.long, device=device)
                c = torch.tensor([], dtype=torch.long, device=device)

            nll = full_mle_loss_block(
                model,
                Zparam,
                slice(i_start, i_end),
                slice(j_start, j_end),
                (r, c),
                undirected=undirected,
            )

            opt.zero_grad(set_to_none=True)
            nll.backward()
            opt.step()
            total_nll += float(nll.detach().cpu().item())

        # (optional) print per-epoch NLL for debugging
        # print(f"Epoch nll: {total_nll:.3f}")

    return Zparam.detach().cpu().numpy()


# --------------------------- CLI ---------------------------

def parse_args():
    ap = argparse.ArgumentParser(description="Per-graph MLE baseline (LatentPositionModel, full MLE) for latent positions.")
    ap.add_argument("--setting_dir", required=True, help="Folder like sim_data_batch/A1")
    ap.add_argument("--out_dir", required=True, help="Directory to save Z_hat and metrics")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)

    # Model / optimization
    ap.add_argument("--latent_dim", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--block_size", type=int, default=512, help="Pairwise MLE block size")
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
    z_dir = os.path.join(args.out_dir, "Zhat_mle_full")
    os.makedirs(z_dir, exist_ok=True)

    # *_test_* graphs to mirror exp1_test_batch
    graph_dirs = [g for g in list_graph_dirs(args.setting_dir) if "_test_" in os.path.basename(g)]
    if args.max_graphs and len(graph_dirs) > args.max_graphs:
        graph_dirs = graph_dirs[:args.max_graphs]
    if not graph_dirs:
        raise SystemExit(f"No *_test_* graphs found in {args.setting_dir}")

    metrics = []
    print(f"Running MLE (LatentPositionModel, full MLE) on {len(graph_dirs)} test graphs. Saving Z_hat to: {z_dir}")

    for gdir in graph_dirs:
        edges, N, directed, base = load_graph_dir(gdir)
        undirected = not directed

        Z_true = load_true_positions(gdir)
        if Z_true is None or Z_true.shape[0] != N:
            print(f"[skip] {base}: missing/size-mismatch true positions.")
            continue

        # -------- Train per graph (full MLE; single inference) --------
        Z_hat = mle_fit_single_graph_full(
            edges_np=edges,
            N=N,
            undirected=undirected,
            device=device,
            d=args.latent_dim,
            epochs=args.epochs,
            lr=args.lr,
            block_size=args.block_size,
            seed=args.seed,
        )

        if args.center:
            Z_hat = Z_hat - Z_hat.mean(0, keepdims=True)

        # Save LPs
        zpath = os.path.join(z_dir, f"{base}_Zhat_mle.npy")
        np.save(zpath, Z_hat)

        # -------- Metrics (GWD and LP-RMSE) --------
        gwd = gwd_from_positions(
            Z_true=Z_true, Z_hat=Z_hat, max_nodes=args.gwd_nodes, seed=args.seed, center=args.center
        )

        k = min(Z_true.shape[0], args.lp_nodes) if args.lp_nodes > 0 else Z_true.shape[0]
        idx = np.arange(Z_true.shape[0]) if k == Z_true.shape[0] else np.sort(
            np.random.default_rng(args.seed).choice(Z_true.shape[0], size=k, replace=False)
        )
        lp_rmse = procrustes_rmse(Z_true[idx], Z_hat[idx], center=True, scale=False)

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
        with open(os.path.join(args.out_dir, "test_metrics_mle.json"), "w") as f:
            json.dump({"summary": summary, "details": metrics}, f, indent=2)
        print(f"\nSummary over {summary['num_graphs']} graphs: mean GWD={mean_gwd:.6f} | mean LP-RMSE={mean_lprmse:.6f}")
        print(f"Metrics saved to {os.path.join(args.out_dir, 'test_metrics_mle.json')}")
    else:
        print("No graphs evaluated.")

if __name__ == "__main__":
    main()
