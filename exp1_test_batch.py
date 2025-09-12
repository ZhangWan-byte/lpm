# exp1_test_batch.py
"""
Infer latent positions ONCE per test graph, save them next to the checkpoint,
then compute:
  (i)  GWD using POT on the saved Z_hat,
  (ii) LP-RSE (RMSE after Procrustes),
  (iii) Kernel reconstruction metrics:
       - AUC-ROC with pos:neg = 1:1 (sampled)
       - AP with pos:neg = 1:1 (sampled)
       - AP over ALL non-edges (extreme case)

Usage:
  python exp1_test_batch.py --setting_dir sim_data_batch/A1 --ckpt runs/A1/ckpt.pt \
    --model RG-G-VAE --latent_dim 16 \
    --ap_chunk_size 2000000
"""

import os, json, argparse, numpy as np, torch, random
import ot  # POT

# Reuse helpers from your training code
from exp1_train_batch import (
    list_graph_dirs, load_node_features, load_true_positions, procrustes_rmse,
    _posterior_sample_latents, build_model, _subsample_indices, _pairwise_euclidean
)
from models.utils import load_graph_dir, build_sparse_adj, negative_sampling, auc_ap


def set_all_seeds(seed:int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

def gwd_from_positions(Z_true: np.ndarray, Z_hat: np.ndarray, max_nodes:int=2000, seed:int=0, center:bool=False,
                       max_iter:int=200, tol:float=1e-9) -> float:
    """GW between metric spaces induced by Z_true and Z_hat (no re-inference)."""
    N = Z_true.shape[0]
    idx = _subsample_indices(N, max_nodes, seed)
    Zt = Z_true[idx].astype(np.float64); Zh = Z_hat[idx].astype(np.float64)
    if center:
        Zt -= Zt.mean(0, keepdims=True); Zh -= Zh.mean(0, keepdims=True)
    Ct = _pairwise_euclidean(Zt); Ch = _pairwise_euclidean(Zh)
    k = Ct.shape[0]; p = np.ones(k)/k; q = p
    try:
        gw2 = ot.gromov_wasserstein2(Ct, Ch, p, q, loss_fun='square_loss', max_iter=max_iter, tol=tol, verbose=False)
    except AttributeError:
        gw2 = ot.gromov.gromov_wasserstein2(Ct, Ch, p, q, loss_fun='square_loss', max_iter=max_iter, tol=tol, verbose=False)
    return float(np.sqrt(max(gw2, 0.0)))

def _edges_to_exclude_set(edges: np.ndarray) -> set:
    """Make a set of pairs (i,j) to exclude when sampling negatives."""
    return set((int(a), int(b)) for a, b in edges)

def _all_non_edges_pairs(N: int, edges: np.ndarray, undirected: bool) -> np.ndarray:
    """
    Enumerate ALL non-edge pairs.
    - For undirected graphs: return unordered pairs (i<j), excluding existing edges (treated as unordered).
    - For directed graphs: return ordered pairs (i,j), i!=j, excluding existing directed edges.
    NOTE: This can be very large (O(N^2)). Use with care.
    """
    if undirected:
        M = np.ones((N, N), dtype=bool)
        np.fill_diagonal(M, False)
        for a, b in edges:
            i, j = int(a), int(b)
            if i == j:
                M[i, j] = False
            else:
                ii, jj = (i, j) if i < j else (j, i)
                M[ii, jj] = False
                M[jj, ii] = False
        iu, ju = np.triu_indices(N, k=1)
        keep = M[iu, ju]
        return np.stack([iu[keep], ju[keep]], axis=1).astype(np.int64)
    else:
        M = np.ones((N, N), dtype=bool)
        np.fill_diagonal(M, False)
        for a, b in edges:
            M[int(a), int(b)] = False
        ii, jj = np.where(M)
        return np.stack([ii, jj], axis=1).astype(np.int64)


# ------------------------------- Concise kernel metrics -------------------------------

def _pair_scores(model, z_t: torch.Tensor, pairs: torch.Tensor) -> np.ndarray:
    """Sigmoid scores for (i,j) pairs as numpy."""
    with torch.no_grad():
        logits = model.pair_logits(z_t, pairs)
        return torch.sigmoid(logits).detach().cpu().numpy()


def kernel_metrics_with_Z(
    model,
    Z_hat: np.ndarray,
    edges: np.ndarray,
    N: int,
    undirected: bool,
    device: torch.device,
    ap_chunk_size: int = 2_000_000,
) -> dict:
    """
    Compute kernel reconstruction metrics given fixed node embeddings Z_hat.

    Reports:
      - AUC-ROC with pos:neg = 1:1 (sampled)
      - AP with pos:neg = 1:1 (sampled)
      - AP over ALL non-edges
    """
    z_t = torch.from_numpy(Z_hat).float().to(device)
    pos = torch.from_numpy(edges.astype(np.int64)).long().to(device)
    n_pos = pos.size(0)
    exclude = _edges_to_exclude_set(edges)

    # ----- AUC & AP with 1:1 sampled negatives -----
    n_neg_sampled = max(1, int(n_pos))
    neg_np = negative_sampling(N, n_neg_sampled, exclude=exclude, undirected=undirected, device=device)
    neg = torch.from_numpy(neg_np).long().to(device)

    pos_scores = _pair_scores(model, z_t, pos)
    neg_scores = _pair_scores(model, z_t, neg)
    scores_11 = np.concatenate([pos_scores, neg_scores], axis=0)
    labels_11 = np.concatenate(
        [np.ones(pos_scores.shape[0], dtype=np.int32),
         np.zeros(neg_scores.shape[0], dtype=np.int32)],
        axis=0
    )
    auc_11, ap_11 = auc_ap(scores_11, labels_11)

    # ----- AP over ALL non-edges -----
    all_neg_np = _all_non_edges_pairs(N, edges, undirected=undirected)
    # chunked scoring for memory friendliness
    neg_all_scores_list = []
    total = all_neg_np.shape[0]
    for s in range(0, total, ap_chunk_size):
        e = min(s + ap_chunk_size, total)
        chunk = torch.from_numpy(all_neg_np[s:e]).long().to(device)
        neg_all_scores_list.append(_pair_scores(model, z_t, chunk))
    neg_all_scores = np.concatenate(neg_all_scores_list, axis=0) if neg_all_scores_list else np.empty((0,), dtype=np.float32)

    scores_all = np.concatenate([pos_scores, neg_all_scores], axis=0)
    labels_all = np.concatenate(
        [np.ones(pos_scores.shape[0], dtype=np.int32),
         np.zeros(neg_all_scores.shape[0], dtype=np.int32)],
        axis=0
    )
    ap_all = auc_ap(scores_all, labels_all)[1]

    return {
        "auc": float(auc_11),
        "ap_11": float(ap_11),
        "ap_all": float(ap_all),
        "n_pos": int(n_pos),
        "n_neg_11": int(n_neg_sampled),
        "n_neg_all": int(all_neg_np.shape[0]),
    }


# --------------------------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--setting_dir", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--deterministic_inference", action="store_true", help="Posterior mean for Z_hat")
    ap.add_argument("--center", action="store_true", help="Center embeddings before metrics")

    # model config (mirror your train args as needed)
    ap.add_argument("--model", default="RG-G-VAE", choices=["RG-G-VAE","RG-P-VAE"])
    ap.add_argument("--latent_dim", type=int, default=16)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--decoder", default="radial",
                    choices=["radial","dot","bilinear","indefinite","mlp","dc_radial","rff"])
    ap.add_argument("--decoder_kwargs", default="{}")
    ap.add_argument("--feat_dec_hidden", type=int, default=64)
    ap.add_argument("--use_struct_feats", action="store_true")

    # eval knobs
    ap.add_argument("--gwd_nodes", type=int, default=2000)
    ap.add_argument("--lp_nodes", type=int, default=5000)
    ap.add_argument("--max_graphs", type=int, default=0)

    # memory control for AP over all non-edges
    ap.add_argument("--ap_chunk_size", type=int, default=2_000_000,
                    help="Chunk size when scoring all non-edges for AP.")

    return ap.parse_args()


def main():
    args = parse_args()
    set_all_seeds(args.seed)
    device = torch.device(args.device)
    ckpt_dir = os.path.dirname(os.path.abspath(args.ckpt))
    print(ckpt_dir)

    graph_dirs = [g for g in list_graph_dirs(args.setting_dir) if "_test_" in os.path.basename(g)]
    if args.max_graphs and len(graph_dirs) > args.max_graphs:
        graph_dirs = graph_dirs[:args.max_graphs]
    if not graph_dirs:
        raise SystemExit(f"No *_test_* graphs found in {args.setting_dir}")

    # build + load
    feats0 = load_node_features(graph_dirs[0], standardize=True)
    model = build_model(feats0.shape[1], args).to(device)
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state["state_dict"], strict=True)
    model.eval()

    metrics = []
    print(f"Evaluating {len(graph_dirs)} test graphs; saving Z_hat to: {ckpt_dir}")
    for gdir in graph_dirs:
        edges, N, directed, base = load_graph_dir(gdir)
        undirected = not directed
        A = build_sparse_adj(N, edges, directed=directed, device=device, self_loops=True)
        X = torch.from_numpy(load_node_features(gdir, standardize=True)).float().to(device)
        Z_true = load_true_positions(gdir)
        if Z_true is None or Z_true.shape[0] != N:
            print(f"[skip] {base}: missing/size-mismatch true positions."); continue

        # ---- infer ONCE ----
        set_all_seeds(args.seed)  # stable per-graph
        Z_hat = _posterior_sample_latents(
            model, A, X, model_name=args.model,
            seed=args.seed, deterministic_inference=args.deterministic_inference
        )
        if args.center:
            Z_hat = Z_hat - Z_hat.mean(0, keepdims=True)
        # save
        os.makedirs(os.path.join(ckpt_dir, "Zhat"), exist_ok=True)
        zpath = os.path.join(ckpt_dir, "Zhat", f"{base}_Zhat.npy"); np.save(zpath, Z_hat)

        # LP-RMSE (RMSE after Procrustes) on (optionally) subsampled nodes
        k = min(Z_true.shape[0], args.lp_nodes) if args.lp_nodes > 0 else Z_true.shape[0]
        idx = np.arange(Z_true.shape[0]) if k==Z_true.shape[0] else np.sort(np.random.default_rng(args.seed).choice(Z_true.shape[0], size=k, replace=False))
        rmse, Z_reduced = procrustes_rmse(Z_true[idx], Z_hat[idx], center=False, scale=False)

        # ---- metrics reusing the saved inference ----
        gwd = gwd_from_positions(Z_true, Z_hat=Z_reduced, max_nodes=args.gwd_nodes, seed=args.seed, center=args.center)

        # ---- kernel reconstruction metrics ----
        kern = kernel_metrics_with_Z(
            model, Z_hat, edges=np.asarray(edges, dtype=np.int64), N=N, undirected=undirected,
            device=device, ap_chunk_size=args.ap_chunk_size
        )

        print(
            f"{base}: GWD={gwd:.6f} | LP-RMSE={rmse:.6f} | "
            f"AUC(1:1)={kern['auc']:.6f} | AP(1:1)={kern['ap_11']:.6f} | AP(all)={kern['ap_all']:.6f} "
            f"| Z_hat={zpath} (pos={kern['n_pos']}, neg_1to1={kern['n_neg_11']}, neg_all={kern['n_neg_all']})"
        )
        metrics.append({
            "graph": base,
            "n_nodes": int(N),
            "gwd": float(gwd),
            "lp_rmse": float(rmse),
            "auc_1to1": float(kern["auc"]),
            "ap_1to1": float(kern["ap_11"]),
            "ap_all": float(kern["ap_all"]),
            "n_pos": int(kern["n_pos"]),
            "n_neg_1to1": int(kern["n_neg_11"]),
            "n_neg_all": int(kern["n_neg_all"]),
            "zhat_path": zpath
        })

    # summary
    if metrics:
        mean_gwd   = float(np.mean([m["gwd"] for m in metrics]))
        mean_rmse  = float(np.mean([m["lp_rmse"] for m in metrics]))
        mean_auc11 = float(np.mean([m["auc_1to1"] for m in metrics]))
        mean_ap11  = float(np.mean([m["ap_1to1"] for m in metrics]))
        mean_apall = float(np.mean([m["ap_all"] for m in metrics]))
        print(
            f"\nSummary over {len(metrics)} graphs: "
            f"mean GWD={mean_gwd:.6f} | mean LP-RMSE={mean_rmse:.6f} | "
            f"mean AUC(1:1)={mean_auc11:.6f} | mean AP(1:1)={mean_ap11:.6f} | mean AP(all)={mean_apall:.6f}"
        )
        out = {
            "mean_gwd": mean_gwd,
            "mean_lp_rmse": mean_rmse,
            "mean_auc_1to1": mean_auc11,
            "mean_ap_1to1": mean_ap11,
            "mean_ap_all": mean_apall,
            "details": metrics
        }
        with open(os.path.join(ckpt_dir, "test_metrics.json"), "w") as f:
            json.dump(out, f, indent=2)
    else:
        print("No graphs evaluated.")

if __name__ == "__main__":
    main()
