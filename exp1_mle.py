# exp1_mle.py
"""
MLE baseline on *_test_* graphs:
  • Fit LatentPositionModel via full MLE (no neg sampling).
  • Save Z_hat and report GWD, LP-RMSE, AUC(1:1), AP(1:1), AP(all non-edges).

Usage:
  python exp1_mle.py --setting_dir sim_data_batch/A1 --out_dir results_mle/A1 \
    --latent_dim 2 --epochs 500 --lr 1e-2 --ap_chunk_size 2000000
"""

import os, json, argparse, random, numpy as np, torch, torch.nn.functional as F
from exp1_train_batch import list_graph_dirs, load_true_positions, procrustes_rmse
from exp1_test_batch import gwd_from_positions, _all_non_edges_pairs  # reuse helpers
from models.utils import load_graph_dir, negative_sampling, auc_ap

# Try both names used in the repo for the baseline model
try:
    from mle_arxiv_ogb import LatentPositionModel  # type: ignore
except Exception:
    from mle_arxiv_ogb_C3 import LatentPositionModel  # type: ignore


# --------------------------- small utils ---------------------------

def set_all_seeds(seed:int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


def _get_Z_param(model, N, d, device):
    if hasattr(model, "mu"): 
        Z = getattr(model, "mu")
        if not isinstance(Z, torch.nn.Parameter):
            Z = torch.nn.Parameter(Z.to(device)); setattr(model, "mu", Z)
    elif hasattr(model, "Z"):
        Z = getattr(model, "Z")
        if not isinstance(Z, torch.nn.Parameter):
            Z = torch.nn.Parameter(Z.to(device)); setattr(model, "Z", Z)
    else:
        Z = torch.nn.Parameter(0.01 * torch.randn(N, d, device=device)); setattr(model, "mu", Z)
    Z.requires_grad_(True); return Z


def _pair_logits(model, zi: torch.Tensor, zj: torch.Tensor) -> torch.Tensor:
    # Prefer model.logits(zi,zj); else fallback to -alpha||zi-zj||^2 + bias
    if hasattr(model, "logits"):
        try: return model.logits(zi, zj)
        except TypeError: pass
    d2 = torch.sum((zi - zj) ** 2, dim=-1)
    b = getattr(model, "bias", None); la = getattr(model, "log_alpha", None)
    b = b if isinstance(b, torch.Tensor) else torch.tensor(0.0, device=d2.device)
    a = (F.softplus(la) + 1e-4) if isinstance(la, torch.Tensor) else torch.tensor(1.0, device=d2.device)
    return b - a * d2


def _pair_scores_from_pairs(model, Z: torch.Tensor, pairs: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        zi = Z[pairs[:, 0]]; zj = Z[pairs[:, 1]]
        return torch.sigmoid(_pair_logits(model, zi, zj)).detach().cpu().numpy()


# --------------------------- full MLE (blockwise) ---------------------------

def mle_fit_full(edges_np: np.ndarray, N:int, undirected:bool, device, d:int, epochs:int, lr:float, block:int, seed:int):
    set_all_seeds(seed)
    model = LatentPositionModel(N=N, d=d).to(device)
    Z = _get_Z_param(model, N, d, device)
    params = [p for p in model.parameters() if p.requires_grad]
    if Z not in params: params.append(Z)
    opt = torch.optim.Adam(params, lr=lr)

    # Edge buckets per (block,block)
    E = edges_np.copy()
    if undirected:
        u, v = np.minimum(E[:,0], E[:,1]), np.maximum(E[:,0], E[:,1]); E = np.stack([u, v], 1)
    nb = (N + block - 1) // block
    buckets = {}
    for a, b in E:
        ib, jb = a // block, b // block
        key = (min(ib, jb), max(ib, jb)) if undirected else (ib, jb)
        (buckets.setdefault(key, [])).append((a, b))

    for _ in range(epochs):
        # Block order
        order = ([(i, j) for i in range(nb) for j in range(i, nb)] if undirected
                 else [(i, j) for i in range(nb) for j in range(nb)])
        random.shuffle(order)

        for ib, jb in order:
            if undirected and jb < ib: continue
            i0, i1 = ib*block, min((ib+1)*block, N)
            j0, j1 = jb*block, min((jb+1)*block, N)
            Zi, Zj = Z[i0:i1], Z[j0:j1]
            bi, bj = Zi.size(0), Zj.size(0)

            zi = Zi[:, None, :].expand(bi, bj, -1).reshape(-1, Zi.size(1))
            zj = Zj[None, :, :].expand(bi, bj, -1).reshape(-1, Zj.size(1))
            logits = _pair_logits(model, zi, zj).reshape(bi, bj)

            Y = torch.zeros((bi, bj), device=device, dtype=logits.dtype)
            pairs = buckets.get((min(ib, jb), max(ib, jb)) if undirected else (ib, jb), [])
            if pairs:
                arr = np.asarray(pairs, dtype=np.int64)
                Y[torch.from_numpy(arr[:,0]-i0).to(device), torch.from_numpy(arr[:,1]-j0).to(device)] = 1.0
            if undirected and ib == jb:
                tri = torch.ones((bi, bj), dtype=torch.bool, device=device).triu(1)
                logits, Y = logits[tri], Y[tri]

            nll = -(F.logsigmoid(logits)*Y + F.logsigmoid(-logits)*(1-Y)).sum()
            opt.zero_grad(set_to_none=True); nll.backward(); opt.step()

    return model, Z.detach().cpu().numpy()


# --------------------------- metrics: AUC/AP ---------------------------

def kernel_metrics(Z_hat: np.ndarray, edges: np.ndarray, N:int, undirected:bool, device, ap_chunk_size:int=2_000_000):
    model_eval = LatentPositionModel(N=N, d=Z_hat.shape[1]).to(device)  # for logits structure
    Zt = torch.from_numpy(Z_hat).float().to(device)
    pos = torch.from_numpy(edges.astype(np.int64)).long().to(device)
    n_pos = pos.size(0)
    # 1:1 sampled negatives
    exclude = set((int(a), int(b)) for a, b in edges)
    neg_np = negative_sampling(N, max(1, int(n_pos)), exclude=exclude, undirected=undirected, device=device)
    neg = torch.from_numpy(neg_np).long().to(device)

    pos_scores = _pair_scores_from_pairs(model_eval, Zt, pos)
    neg_scores = _pair_scores_from_pairs(model_eval, Zt, neg)
    scores_11 = np.concatenate([pos_scores, neg_scores], 0)
    labels_11 = np.concatenate([np.ones_like(pos_scores, dtype=np.int32),
                                np.zeros_like(neg_scores, dtype=np.int32)], 0)
    auc_11, ap_11 = auc_ap(scores_11, labels_11)

    # AP over all non-edges
    all_neg_np = _all_non_edges_pairs(N, edges, undirected=undirected)
    neg_all_scores = []
    for s in range(0, all_neg_np.shape[0], ap_chunk_size):
        e = min(s + ap_chunk_size, all_neg_np.shape[0])
        chunk = torch.from_numpy(all_neg_np[s:e]).long().to(device)
        neg_all_scores.append(_pair_scores_from_pairs(model_eval, Zt, chunk))
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


# --------------------------- CLI & main ---------------------------

def parse_args():
    ap = argparse.ArgumentParser("MLE baseline with concise metrics")
    ap.add_argument("--setting_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)

    # model/optim
    ap.add_argument("--latent_dim", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--block_size", type=int, default=512)

    # metrics
    ap.add_argument("--gwd_nodes", type=int, default=2000)
    ap.add_argument("--lp_nodes", type=int, default=5000)
    ap.add_argument("--center", action="store_true")
    ap.add_argument("--ap_chunk_size", type=int, default=2_000_000)

    ap.add_argument("--max_graphs", type=int, default=0)
    return ap.parse_args()


def main():
    args = parse_args()
    set_all_seeds(args.seed)
    device = torch.device(args.device)

    os.makedirs(args.out_dir, exist_ok=True)
    z_dir = os.path.join(args.out_dir, "Zhat_mle_full"); os.makedirs(z_dir, exist_ok=True)

    graph_dirs = [g for g in list_graph_dirs(args.setting_dir) if "_test_" in os.path.basename(g)]
    if args.max_graphs and len(graph_dirs) > args.max_graphs:
        graph_dirs = graph_dirs[:args.max_graphs]
    if not graph_dirs: raise SystemExit(f"No *_test_* graphs found in {args.setting_dir}")

    metrics = []
    for gdir in graph_dirs:
        edges, N, directed, base = load_graph_dir(gdir); undirected = not directed
        Z_true = load_true_positions(gdir)
        if Z_true is None or Z_true.shape[0] != N:
            print(f"[skip] {base}: missing/size-mismatch true positions."); continue

        model, Z_hat = mle_fit_full(edges, N, undirected, device, args.latent_dim, args.epochs, args.lr, args.block_size, args.seed)
        if args.center: Z_hat = Z_hat - Z_hat.mean(0, keepdims=True)
        zpath = os.path.join(z_dir, f"{base}_Zhat_mle.npy"); np.save(zpath, Z_hat)

        # LP-RMSE (subsample if needed)
        k = min(Z_true.shape[0], args.lp_nodes) if args.lp_nodes > 0 else Z_true.shape[0]
        idx = np.arange(Z_true.shape[0]) if k==Z_true.shape[0] else np.sort(np.random.default_rng(args.seed).choice(Z_true.shape[0], k, replace=False))
        lp_rmse, Z_red = procrustes_rmse(Z_true[idx], Z_hat[idx], center=False, scale=False)

        gwd = gwd_from_positions(Z_true, Z_red, max_nodes=args.gwd_nodes, seed=args.seed, center=args.center)
        kern = kernel_metrics(Z_hat, np.asarray(edges, dtype=np.int64), N, undirected, device, args.ap_chunk_size)

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

    if metrics:
        out = {
            "num_graphs": len(metrics),
            "mean_gwd": float(np.mean([m["gwd"] for m in metrics])),
            "mean_lp_rmse": float(np.mean([m["lp_rmse"] for m in metrics])),
            "mean_auc_1to1": float(np.mean([m["auc_1to1"] for m in metrics])),
            "mean_ap_1to1": float(np.mean([m["ap_1to1"] for m in metrics])),
            "mean_ap_all": float(np.mean([m["ap_all"] for m in metrics])),
            "details": metrics
        }
        with open(os.path.join(args.out_dir, "test_metrics_mle.json"), "w") as f: json.dump(out, f, indent=2)
        print(f"\nSummary over {out['num_graphs']} graphs: "
              f"mean GWD={out['mean_gwd']:.6f} | mean LP-RMSE={out['mean_lp_rmse']:.6f} | "
              f"mean AUC(1:1)={out['mean_auc_1to1']:.6f} | mean AP(1:1)={out['mean_ap_1to1']:.6f} | "
              f"mean AP(all)={out['mean_ap_all']:.6f}")
        print(f"Saved metrics to {os.path.join(args.out_dir, 'test_metrics_mle.json')}")
    else:
        print("No graphs evaluated.")

if __name__ == "__main__":
    main()
