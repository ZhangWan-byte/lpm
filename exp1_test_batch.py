# exp1_test_batch.py
"""
Infer latent positions ONCE per test graph, save them next to the checkpoint,
then compute (i) GWD using POT on the saved Z_hat and (ii) LP-RSE (RMSE after Procrustes).

Usage:
  python exp1_test_batch.py --setting_dir sim_data_batch/A1 --ckpt runs/A1/ckpt.pt \
    --model RG-G-VAE --latent_dim 16
"""

import os, json, argparse, numpy as np, torch, random
import ot  # POT

# Reuse helpers from your training code
from exp1_train_batch import (
    list_graph_dirs, load_node_features, load_true_positions, procrustes_rmse,
    _posterior_sample_latents, build_model, _subsample_indices, _pairwise_euclidean
)
from models.utils import load_graph_dir, build_sparse_adj


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
    # numerical guard in case of tiny negative due to round-off
    return float(np.sqrt(max(gw2, 0.0)))


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
    return ap.parse_args()


def main():
    args = parse_args()
    set_all_seeds(args.seed)
    device = torch.device(args.device)
    ckpt_dir = os.path.dirname(os.path.abspath(args.ckpt))

    graph_dirs = [g for g in list_graph_dirs(args.setting_dir) if "_test_" in os.path.basename(g)]
    if args.max_graphs and len(graph_dirs) > args.max_graphs:
        graph_dirs = graph_dirs[:args.max_graphs]
    if not graph_dirs:
        raise SystemExit(f"No *_test_* graphs found in {args.setting_dir}")

    # build + load
    feats0 = load_node_features(graph_dirs[0], standardize=True)
    model = build_model(feats0.shape[1], args).to(device)
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state.get("model", state), strict=False)
    model.eval()

    metrics = []
    print(f"Evaluating {len(graph_dirs)} test graphs; saving Z_hat to: {ckpt_dir}")
    for gdir in graph_dirs:
        edges, N, directed, base = load_graph_dir(gdir)
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

        # ---- metrics reusing the saved inference ----
        gwd = gwd_from_positions(Z_true, Z_hat, max_nodes=args.gwd_nodes, seed=args.seed, center=args.center)

        # LP-RMSE (RMSE after Procrustes) on (optionally) subsampled nodes
        k = min(Z_true.shape[0], args.lp_nodes) if args.lp_nodes > 0 else Z_true.shape[0]
        idx = np.arange(Z_true.shape[0]) if k==Z_true.shape[0] else np.sort(np.random.default_rng(args.seed).choice(Z_true.shape[0], size=k, replace=False))
        rmse = procrustes_rmse(Z_true[idx], Z_hat[idx], center=True, scale=False)

        print(f"{base}: GWD={gwd:.6f} | LP-RMSE={rmse:.6f} | Z_hat={zpath}")
        metrics.append({"graph": base, "n_nodes": int(N), "gwd": float(gwd), "lp_rmse": float(rmse), "zhat_path": zpath})

    # summary
    if metrics:
        mean_gwd = float(np.mean([m["gwd"] for m in metrics]))
        mean_rmse  = float(np.mean([m["lp_rmse"] for m in metrics]))
        print(f"\nSummary over {len(metrics)} graphs: mean GWD={mean_gwd:.6f} | mean LP-RMSE={mean_rmse:.6f}")
        with open(os.path.join(ckpt_dir, "test_metrics.json"), "w") as f:
            json.dump({"mean_gwd": mean_gwd, "mean_lp_rmse": mean_rmse, "details": metrics}, f, indent=2)
    else:
        print("No graphs evaluated.")

if __name__ == "__main__":
    main()
