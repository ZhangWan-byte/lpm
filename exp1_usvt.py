# exp1_usvt.py
"""
USVT baseline on batch datasets (same layout as exp1_train_batch / exp1_test_batch):
- For every *_train_* and *_test_* graph under --setting_dir:
    1) Build a dense (symmetrized) adjacency from edges.
    2) Run USVT to estimate latent positions (Z_hat) via eigenthresholding.
    3) Save Z_hat to disk.
- For *_test_* graphs, also report:
    • Gromov–Wasserstein distance (GWD) to ground-truth positions (not squared)
    • LP-RMSE via orthogonal Procrustes

Usage:
  python exp1_usvt.py --setting_dir sim_data_batch/A1 --out_dir results_usvt/A1 \
    --gamma 0.01 --d_max 16 --energy 1.0 --seed 0
"""

import os
import json
import argparse
import numpy as np
import torch
import random
import ot  # POT

# Borrow helpers to match I/O and metrics used elsewhere
from exp1_train_batch import (
    list_graph_dirs,
    load_true_positions,
    procrustes_rmse,
)
from models.utils import load_graph_dir
from exp1_test_batch import gwd_from_positions

# Use the USVT functions provided in the repo
try:
    from USVT_arxiv_ogb_C3 import usvt_and_embed_gpu  # type: ignore
except Exception:
    from USVT_arxiv_ogb_C2_v4 import usvt_and_embed_gpu  # type: ignore


# --------------------------- Utils ---------------------------

def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def edges_to_dense_adj(N: int, edges: np.ndarray, undirected: bool, device: torch.device) -> torch.Tensor:
    """
    Build a dense, float64, symmetrized adjacency with zero diagonal.
    """
    A = torch.zeros((N, N), dtype=torch.float64, device=device)
    if edges.size > 0:
        idx = torch.from_numpy(edges.astype(np.int64)).to(device)
        A[idx[:, 0], idx[:, 1]] = 1.0
        if undirected:
            A[idx[:, 1], idx[:, 0]] = 1.0
    A.fill_diagonal_(0.0)
    return A

# --------------------------- CLI ---------------------------

def parse_args():
    ap = argparse.ArgumentParser(description="USVT baseline (train/test) for latent positions on batch datasets.")
    ap.add_argument("--setting_dir", required=True, help="Folder like sim_data_batch/A1")
    ap.add_argument("--out_dir", required=True, help="Directory to save Z_hat and metrics")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)

    # USVT hyperparams
    ap.add_argument("--gamma", type=float, default=0.01, help="USVT threshold multiplier")
    ap.add_argument("--d_max", type=int, default=2, help="Max kept eigen-components")
    ap.add_argument("--energy", type=float, default=1.0, help="Energy fraction to keep (<=1.0)")
    ap.add_argument("--eps", type=float, default=1e-10, help="Drop eigenvalues below this after scaling")

    # Metrics
    ap.add_argument("--gwd_nodes", type=int, default=2000, help="Max nodes used in GWD (subsample)")
    ap.add_argument("--lp_nodes", type=int, default=5000, help="Max nodes used in LP-RMSE (subsample)")
    ap.add_argument("--center", action="store_true", help="Center embeddings before metrics")

    # Control
    ap.add_argument("--max_graphs", type=int, default=0, help="0=all graphs, else limit per split")
    return ap.parse_args()


# --------------------------- Main ---------------------------

def main():
    args = parse_args()
    set_all_seeds(args.seed)
    device = torch.device(args.device)

    # Output dirs
    os.makedirs(args.out_dir, exist_ok=True)
    z_dir_train = os.path.join(args.out_dir, "Zhat_usvt_train")
    z_dir_test = os.path.join(args.out_dir, "Zhat_usvt_test")
    os.makedirs(z_dir_train, exist_ok=True)
    os.makedirs(z_dir_test, exist_ok=True)

    # Discover graphs
    all_dirs = list_graph_dirs(args.setting_dir)
    train_dirs = [g for g in all_dirs if "_train_" in os.path.basename(g)]
    test_dirs = [g for g in all_dirs if "_test_" in os.path.basename(g)]
    if args.max_graphs:
        train_dirs = train_dirs[:args.max_graphs]
        test_dirs = test_dirs[:args.max_graphs]

    # -------- TRAIN SPLIT: fit & save Z_hat (no metrics expected) --------
    if train_dirs:
        print(f"USVT on {len(train_dirs)} train graphs...")
        for gdir in train_dirs:
            edges, N, directed, base = load_graph_dir(gdir)
            A = edges_to_dense_adj(N, edges, undirected=not directed, device=device)
            Z_hat = usvt_and_embed_gpu(
                A_t=A,
                gamma=args.gamma,
                d_max=args.d_max,
                energy=args.energy,
                eps=args.eps,
                verbose=False,
                device=device,
                dtype=torch.float64,
                return_numpy=True,
            )
            if args.center:
                Z_hat = Z_hat - Z_hat.mean(0, keepdims=True)
            np.save(os.path.join(z_dir_train, f"{base}_Zhat_usvt.npy"), Z_hat)
        print("Train graphs processed.\n")

    # -------- TEST SPLIT: fit, save Z_hat, compute metrics --------
    metrics = []
    if test_dirs:
        print(f"USVT on {len(test_dirs)} test graphs...")
        for gdir in test_dirs:
            edges, N, directed, base = load_graph_dir(gdir)
            Z_true = load_true_positions(gdir)
            if Z_true is None or Z_true.shape[0] != N:
                print(f"[skip] {base}: missing/size-mismatch true positions.")
                continue

            # Fit USVT on this graph
            A = edges_to_dense_adj(N, edges, undirected=not directed, device=device)
            Z_hat = usvt_and_embed_gpu(
                A_t=A,
                gamma=args.gamma,
                d_max=args.d_max,
                energy=args.energy,
                eps=args.eps,
                verbose=False,
                device=device,
                dtype=torch.float64,
                return_numpy=True,
            )
            if args.center:
                Z_hat = Z_hat - Z_hat.mean(0, keepdims=True)

            # Save
            zpath = os.path.join(z_dir_test, f"{base}_Zhat_usvt.npy")
            np.save(zpath, Z_hat)

            # Metrics: LP-RMSE + GWD
            k = min(Z_true.shape[0], args.lp_nodes) if args.lp_nodes > 0 else Z_true.shape[0]
            idx = np.arange(Z_true.shape[0]) if k == Z_true.shape[0] else np.sort(
                np.random.default_rng(args.seed).choice(Z_true.shape[0], size=k, replace=False)
            )
            lp_rmse, Z_reduced = procrustes_rmse(Z_true[idx], Z_hat[idx], center=True, scale=False)
            
            gwd = gwd_from_positions(
                Z_true=Z_true, Z_hat=Z_reduced, max_nodes=args.gwd_nodes, seed=args.seed, center=args.center
            )
            
            print(f"{base}: GWD={gwd:.6f} | LP-RMSE={lp_rmse:.6f}")
            metrics.append({
                "graph": base,
                "n_nodes": int(N),
                "gwd": float(gwd),
                "lp_rmse": float(lp_rmse),
                "zhat_path": zpath,
            })

        # Save summary
        if metrics:
            mean_gwd = float(np.mean([m["gwd"] for m in metrics]))
            mean_lprmse = float(np.mean([m["lp_rmse"] for m in metrics]))
            summary = {
                "num_graphs": len(metrics),
                "mean_gwd": mean_gwd,
                "mean_lp_rmse": mean_lprmse,
                "gamma": args.gamma,
                "d_max": args.d_max,
                "energy": args.energy,
            }
            with open(os.path.join(args.out_dir, "test_metrics_usvt.json"), "w") as f:
                json.dump({"summary": summary, "details": metrics}, f, indent=2)
            print(f"\nSummary over {summary['num_graphs']} graphs: "
                  f"mean GWD={mean_gwd:.6f} | mean LP-RMSE={mean_lprmse:.6f}")
    else:
        print("No *_test_* graphs found.")

if __name__ == "__main__":
    main()
