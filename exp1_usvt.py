# exp1_usvt.py
"""
USVT baseline:
  1) For *_train_* graphs: run USVT to obtain Z_hat and save.
  2) For *_test_* graphs: run USVT to obtain Z_hat, then report:
       • GWD (vs. true positions)
       • LP-RMSE after Procrustes
       • AUC-ROC (1:1 pos:neg, sampled)
       • AP (1:1 pos:neg, sampled)
       • AP over ALL non-edges (chunked)

Kernel for scoring (USVT): W(x,y) = <x, y>

Usage:
  python exp1_usvt.py --setting_dir sim_data_batch/A1 --out_dir results_usvt/A1 \
    --gamma 0.01 --d_max 16 --energy 1.0 --ap_chunk_size 2000000
"""

import os, json, argparse, numpy as np, torch, random
from exp1_train_batch import list_graph_dirs, load_true_positions, procrustes_rmse, pca_rmse
from exp1_test_batch import gwd_from_positions, _all_non_edges_pairs  # reuse helpers
from models.utils import load_graph_dir, negative_sampling, auc_ap

# USVT backends provided in repo
try:
    from USVT_arxiv_ogb_C3 import usvt_and_embed_gpu  # type: ignore
except Exception:
    from USVT_arxiv_ogb_C2_v4 import usvt_and_embed_gpu  # type: ignore


# --------------------------- utils ---------------------------

def set_all_seeds(seed:int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


def edges_to_dense_adj(N:int, edges:np.ndarray, undirected:bool, device) -> torch.Tensor:
    A = torch.zeros((N, N), dtype=torch.float64, device=device)
    if edges.size:
        ij = torch.from_numpy(edges.astype(np.int64)).to(device)
        A[ij[:,0], ij[:,1]] = 1.0
        if undirected: A[ij[:,1], ij[:,0]] = 1.0
    A.fill_diagonal_(0.0); return A


# --------------------------- kernel metrics (dot-product) ---------------------------

def _pair_scores_dot(Zt: torch.Tensor, pairs_t: torch.Tensor) -> np.ndarray:
    """Sigmoid(<x_i, x_j>) scores as numpy."""
    with torch.no_grad():
        zi = Zt[pairs_t[:, 0]]   # [M, d]
        zj = Zt[pairs_t[:, 1]]   # [M, d]
        logits = (zi * zj).sum(dim=-1)  # dot product
        return torch.sigmoid(logits).detach().cpu().numpy()


def kernel_metrics_usvt(Z_hat: np.ndarray, edges: np.ndarray, N:int, undirected:bool,
                        device, ap_chunk_size:int=2_000_000):
    """AUC(1:1), AP(1:1), AP(all non-edges) using W(x,y)=<x,y>."""
    Zt = torch.from_numpy(Z_hat).float().to(device)
    pos = torch.from_numpy(edges.astype(np.int64)).long().to(device)
    n_pos = pos.size(0)

    # 1:1 sampled negatives
    exclude = set((int(a), int(b)) for a, b in edges)
    neg_np = negative_sampling(N, max(1, int(n_pos)), exclude=exclude, undirected=undirected, device=device)
    neg = torch.from_numpy(neg_np).long().to(device)

    pos_scores = _pair_scores_dot(Zt, pos)
    neg_scores = _pair_scores_dot(Zt, neg)
    scores_11 = np.concatenate([pos_scores, neg_scores], 0)
    labels_11 = np.concatenate([np.ones_like(pos_scores, dtype=np.int32),
                                np.zeros_like(neg_scores, dtype=np.int32)], 0)
    auc_11, ap_11 = auc_ap(scores_11, labels_11)

    # AP over all non-edges (chunked)
    all_neg_np = _all_non_edges_pairs(N, edges, undirected=undirected)
    neg_all_scores = []
    for s in range(0, all_neg_np.shape[0], ap_chunk_size):
        e = min(s + ap_chunk_size, all_neg_np.shape[0])
        neg_all_scores.append(_pair_scores_dot(Zt, torch.from_numpy(all_neg_np[s:e]).long().to(device)))
    neg_all_scores = np.concatenate(neg_all_scores, 0) if neg_all_scores else np.empty((0,), dtype=np.float32)

    scores_all = np.concatenate([pos_scores, neg_all_scores], 0)
    labels_all = np.concatenate([np.ones_like(pos_scores, dtype=np.int32),
                                 np.zeros_like(neg_all_scores, dtype=np.int32)], 0)
    ap_all = auc_ap(scores_all, labels_all)[1]

    return {
        "auc_1to1": float(auc_11),
        "ap_1to1": float(ap_11),
        "ap_all": float(ap_all),
        "n_pos": int(n_pos),
        "n_neg_1to1": int(neg.size(0)),
        "n_neg_all": int(all_neg_np.shape[0]),
    }


# --------------------------- CLI ---------------------------

def parse_args():
    ap = argparse.ArgumentParser("USVT baseline (dot-product kernel) with concise metrics")
    ap.add_argument("--setting_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)

    # USVT hyperparams
    ap.add_argument("--gamma", type=float, default=0.01)
    ap.add_argument("--d_max", type=int, default=2)
    ap.add_argument("--energy", type=float, default=1.0)
    ap.add_argument("--eps", type=float, default=1e-10)

    # Metrics
    ap.add_argument("--rmse", default="procrustes", choices=["procrustes","pca"],
                    help="How to align before LP-RMSE: Procrustes (default) or PCA.")
    ap.add_argument("--gwd_nodes", type=int, default=2000)
    ap.add_argument("--lp_nodes", type=int, default=5000)
    ap.add_argument("--center", action="store_true")
    ap.add_argument("--ap_chunk_size", type=int, default=2_000_000)

    # Control
    ap.add_argument("--max_graphs", type=int, default=0)
    return ap.parse_args()


# --------------------------- main ---------------------------

def main():
    args = parse_args()
    set_all_seeds(args.seed)
    device = torch.device(args.device)

    os.makedirs(args.out_dir, exist_ok=True)
    z_dir_train = os.path.join(args.out_dir, "Zhat_usvt_train")
    z_dir_test  = os.path.join(args.out_dir, "Zhat_usvt_test")
    os.makedirs(z_dir_train, exist_ok=True); os.makedirs(z_dir_test, exist_ok=True)

    all_dirs   = list_graph_dirs(args.setting_dir)
    train_dirs = [g for g in all_dirs if "_train_" in os.path.basename(g)]
    test_dirs  = [g for g in all_dirs if "_test_"  in os.path.basename(g)]
    if args.max_graphs:
        train_dirs, test_dirs = train_dirs[:args.max_graphs], test_dirs[:args.max_graphs]

    # ---- TRAIN (save Z_hat only) ----
    for gdir in train_dirs:
        edges, N, directed, base = load_graph_dir(gdir)
        A = edges_to_dense_adj(N, edges, undirected=not directed, device=device)
        Z_hat = usvt_and_embed_gpu(A_t=A, gamma=args.gamma, d_max=args.d_max, energy=args.energy,
                                   eps=args.eps, verbose=False, device=device, dtype=torch.float64, return_numpy=True)
        if args.center: Z_hat = Z_hat - Z_hat.mean(0, keepdims=True)
        np.save(os.path.join(z_dir_train, f"{base}_Zhat_usvt.npy"), Z_hat)

    # ---- TEST (fit, save, metrics) ----
    metrics = []
    for gdir in test_dirs:
        edges, N, directed, base = load_graph_dir(gdir); undirected = not directed
        Z_true = load_true_positions(gdir)
        if Z_true is None or Z_true.shape[0] != N:
            print(f"[skip] {base}: missing/size-mismatch true positions."); continue

        A = edges_to_dense_adj(N, edges, undirected=undirected, device=device)
        Z_hat = usvt_and_embed_gpu(A_t=A, gamma=args.gamma, d_max=args.d_max, energy=args.energy,
                                   eps=args.eps, verbose=False, device=device, dtype=torch.float64, return_numpy=True)
        if args.center: Z_hat = Z_hat - Z_hat.mean(0, keepdims=True)
        zpath = os.path.join(z_dir_test, f"{base}_Zhat_usvt.npy"); np.save(zpath, Z_hat)

        # LP-RMSE (subsample) & GWD
        k = min(Z_true.shape[0], args.lp_nodes) if args.lp_nodes > 0 else Z_true.shape[0]
        idx = np.arange(Z_true.shape[0]) if k==Z_true.shape[0] else np.sort(
            np.random.default_rng(args.seed).choice(Z_true.shape[0], k, replace=False)
        )
        # lp_rmse, Z_red = procrustes_rmse(Z_true[idx], Z_hat[idx], center=False, scale=False)
        if args.rmse == 'procrustes':
            lp_rmse, Z_red = procrustes_rmse(Z_true[idx], Z_hat[idx], center=True, scale=True)
        elif args.rmse == 'pca':
            lp_rmse, Z_red = pca_rmse(Z_true[idx], Z_hat[idx], center=False, scale=False)
        else:
            raise ValueError(f"Unknown --rmse {args.rmse}")
        gwd = gwd_from_positions(Z_true, Z_red, max_nodes=args.gwd_nodes, seed=args.seed, center=args.center)

        # Kernel metrics with W(x,y)=<x,y>
        edges_np = np.asarray(edges, dtype=np.int64)
        kern = kernel_metrics_usvt(Z_hat, edges_np, N, undirected, device, args.ap_chunk_size)

        print(f"{base}: GWD={gwd:.6f} | LP-RMSE={lp_rmse:.6f} | "
              f"AUC(1:1)={kern['auc_1to1']:.6f} | AP(1:1)={kern['ap_1to1']:.6f} | AP(all)={kern['ap_all']:.6f} "
              f"| Z_hat={zpath} (pos={kern['n_pos']}, neg_1to1={kern['n_neg_1to1']}, neg_all={kern['n_neg_all']})")

        metrics.append({
            "graph": base, "n_nodes": int(N),
            "gwd": float(gwd), "lp_rmse": float(lp_rmse),
            "auc_1to1": float(kern["auc_1to1"]), "ap_1to1": float(kern["ap_1to1"]),
            "ap_all": float(kern["ap_all"]), "n_pos": int(kern["n_pos"]),
            "n_neg_1to1": int(kern["n_neg_1to1"]), "n_neg_all": int(kern["n_neg_all"]),
            "zhat_path": zpath
        })

    # ---- summary ----
    if metrics:
        out = {
            "num_graphs": len(metrics),
            "mean_gwd":      float(np.mean([m["gwd"] for m in metrics])),
            "mean_lp_rmse":  float(np.mean([m["lp_rmse"] for m in metrics])),
            "mean_auc_1to1": float(np.mean([m["auc_1to1"] for m in metrics])),
            "mean_ap_1to1":  float(np.mean([m["ap_1to1"] for m in metrics])),
            "mean_ap_all":   float(np.mean([m["ap_all"] for m in metrics])),
            "details": metrics,
            "gamma": args.gamma, "d_max": args.d_max, "energy": args.energy
        }
        with open(os.path.join(args.out_dir, "test_metrics_usvt.json"), "w") as f: json.dump(out, f, indent=2)
        print(f"\nSummary over {out['num_graphs']} graphs: "
              f"mean GWD={out['mean_gwd']:.6f} | mean LP-RMSE={out['mean_lp_rmse']:.6f} | "
              f"mean AUC(1:1)={out['mean_auc_1to1']:.6f} | mean AP(1:1)={out['mean_ap_1to1']:.6f} | "
              f"mean AP(all)={out['mean_ap_all']:.6f}")
    else:
        print("No *_test_* graphs found.")

if __name__ == "__main__":
    main()
