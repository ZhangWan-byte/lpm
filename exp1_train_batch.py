# File: exp1_train_batch.py
"""
Train RG-VAE across many graphs of varying sizes under one SETTING (A1/A2/B1).
- Reads graphs from sim_data_batch/<SETTING>/* folders.
- Re-uses models/rg_vae.RG_VAE.
- Aggregates losses as per-graph averages each epoch.
- Computes val AUC/AP each epoch (balanced negatives).
- Periodically computes IGNR-style GWD^2 (POT, non-entropic) and LP-RMSE.
- Saves all metrics, checkpoints, and the full command line into results/<MMDD_HHMM>/

Dependencies:
  pip install POT
"""

import os
import sys
import json
import argparse
from datetime import datetime
from typing import List, Dict, Tuple, Optional

from tqdm import tqdm

# --- force line-buffered, auto-flushing prints (good for Slurm) ---
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    os.environ.setdefault("PYTHONUNBUFFERED", "1")

import numpy as np
import torch
import ot  # Python Optimal Transport (POT)

from models.rg_vae import RG_VAE, RG_P_VAE
from models.utils import (
    load_graph_dir,
    build_sparse_adj,
    split_edges,
    negative_sampling,
    auc_ap,
)

# ----------------- Helpers: listing & features -----------------

def list_graph_dirs(root: str) -> List[str]:
    subs = []
    for name in sorted(os.listdir(root)):
        p = os.path.join(root, name)
        if os.path.isdir(p) and any(f.endswith("_edges.txt") for f in os.listdir(p)):
            subs.append(p)
    return subs


def load_node_features(graph_dir: str, standardize: bool = True) -> np.ndarray:
    node_npzs = [os.path.join(graph_dir, f) for f in os.listdir(graph_dir) if f.endswith("_nodes.npz")]
    if not node_npzs:
        raise FileNotFoundError(f"No *_nodes.npz found in {graph_dir}")
    npz = np.load(node_npzs[0])
    if "node_features" in npz.files:
        feats = np.array(npz["node_features"], dtype=np.float32)
    else:
        # fallback: synthesize from positions
        Z = np.array(npz["positions"], dtype=np.float32)
        N, d = Z.shape
        rng = np.random.default_rng(7)
        m = min(4 * d, 16)
        W = rng.normal(0.0, 1.0, size=(d, m)).astype(np.float32)
        P = Z @ W
        feats = np.concatenate([Z, Z**2, np.sin(P), np.cos(P)], axis=1).astype(np.float32)
    if standardize and feats.size > 0:
        mu = feats.mean(axis=0, keepdims=True)
        sd = feats.std(axis=0, keepdims=True) + 1e-6
        feats = (feats - mu) / sd
    return feats

# ----------------- Training utils -----------------

def current_lambda_kl(base_lambda_kl: float, epoch: int, warmup_epochs: int) -> float:
    if warmup_epochs <= 0 or epoch >= warmup_epochs:
        return base_lambda_kl
    return base_lambda_kl * (epoch / float(warmup_epochs))

def build_model(input_dim: int, args) -> RG_VAE:
    try:
        dec_kwargs = json.loads(args.decoder_kwargs)
        if not isinstance(dec_kwargs, dict):
            raise ValueError
    except Exception:
        raise SystemExit("--decoder_kwargs must be a JSON object string, e.g. '{\"num_features\":1024}'")
    if args.model=='RG-G-VAE':
        model = RG_VAE(
            input_dim=input_dim,
            latent_dim=args.latent_dim,
            hidden=args.hidden,
            enc_layers=2,
            use_struct_feats=args.use_struct_feats,
            decoder=args.decoder,
            decoder_kwargs=dec_kwargs,
            feat_dec_hidden=args.feat_dec_hidden
        )
    elif args.model=='RG-P-VAE':
        model = RG_P_VAE(
            input_dim=input_dim,
            latent_dim=args.latent_dim,
            hidden=args.hidden,
            enc_layers=2,
            use_struct_feats=args.use_struct_feats,
            decoder=args.decoder,
            decoder_kwargs=dec_kwargs,
            feat_dec_hidden=args.feat_dec_hidden
        )
    return model

def step_graph(model: RG_VAE,
               A_norm: torch.Tensor,
               feats_t: torch.Tensor,
               pos_edges: np.ndarray,
               neg_ratio: int,
               undirected: bool,
               exclude_pairs: Optional[set],
               device: torch.device,
               lambda_feat: float,
               lambda_kl: float) -> Tuple[torch.Tensor, Dict[str, float]]:
    n_pos = pos_edges.shape[0]
    neg_edges = negative_sampling(
        A_norm.size(0), max(1, n_pos * neg_ratio), exclude=exclude_pairs, undirected=undirected, device=device
    )
    pos = torch.from_numpy(pos_edges).long().to(device)
    neg = torch.from_numpy(neg_edges).long().to(device)
    loss, stats = model.elbo(A_norm, pos, neg, feats=feats_t, lambda_feat=lambda_feat, lambda_kl=lambda_kl)
    return loss, stats

def val_metrics(model: RG_VAE,
                A_norm: torch.Tensor,
                feats_t: torch.Tensor,
                val_edges: np.ndarray,
                undirected: bool,
                exclude_pairs: Optional[set],
                device: torch.device,
                neg_ratio_for_auc: int = 1) -> Tuple[float, float]:
    model.eval()
    with torch.no_grad():
        Z = model.embed(A_norm, feats=feats_t)  # [N, D]
        pos = torch.from_numpy(val_edges).long().to(device)
        neg_edges = negative_sampling(
            A_norm.size(0), max(1, val_edges.shape[0] * neg_ratio_for_auc), exclude=exclude_pairs, undirected=undirected, device=device
        )
        neg = torch.from_numpy(neg_edges).long().to(device)
        pairs = torch.cat([pos, neg], dim=0)
        logits = model.pair_logits(Z, pairs)
        scores = torch.sigmoid(logits).detach().cpu().numpy()
        labels = np.concatenate([np.ones(pos.size(0), dtype=np.int32), np.zeros(neg.size(0), dtype=np.int32)])
        auc, ap = auc_ap(scores, labels)
        return auc, ap

# ----------------- IGNR-style GWD & LP-RMSE (POT, non-entropic) -----------------

def _subsample_indices(N: int, max_nodes: int, seed: int = 0) -> np.ndarray:
    if max_nodes <= 0 or max_nodes >= N:
        return np.arange(N, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(N, size=max_nodes, replace=False).astype(np.int64))

def _pairwise_euclidean(X: np.ndarray) -> np.ndarray:
    # X: [n,d] -> D: [n,n], Euclidean distances
    G = X @ X.T
    d = np.clip(np.diag(G), 0.0, None)
    D2 = np.clip(d[:, None] + d[None, :] - 2.0 * G, 0.0, None)
    return np.sqrt(D2, dtype=np.float64)

def load_true_positions(graph_dir: str) -> Optional[np.ndarray]:
    files = [f for f in os.listdir(graph_dir) if f.endswith("_nodes.npz")]
    if not files:
        return None
    npz = np.load(os.path.join(graph_dir, files[0]))
    return np.array(npz["positions"], dtype=np.float32) if "positions" in npz.files else None

# def _posterior_sample_latents(model, A_norm: torch.Tensor, feats_t: torch.Tensor, seed: int = 0) -> np.ndarray:
#     """Draw one sample z ~ q_phi(z|x,A) via reparameterization using model.encode(...)."""
#     torch.manual_seed(seed)
#     try:
#         mu, logvar = model.encode(A_norm, feats=feats_t)  # preferred signature
#     except TypeError:
#         mu, logvar = model.encode(A_norm)  # fallback if older signature
#     std = torch.exp(0.5 * logvar)
#     eps = torch.randn_like(std)
#     z = mu + eps * std
#     return z.detach().cpu().numpy()

def _posterior_sample_latents(
    model,
    A_norm: torch.Tensor,
    feats_t: torch.Tensor,
    seed: int = 0,
    model_name: str = "RG-G-VAE",
) -> np.ndarray:
    """
    Draw one posterior sample of node-level latent *positions*, branching by `model_name`.

    Args:
        model: The instantiated model.
        A_norm: [N,N] normalized sparse adjacency (torch.sparse_coo_tensor).
        feats_t: [N,Dx] node feature tensor.
        seed: RNG seed for reproducibility.
        model_name: "RG-G-VAE" (Gaussian) or "RG-P-VAE" (Poisson-latent).

    Returns:
        np.ndarray of shape [N, D_embed] — the same representation the edge decoder consumes.
    """
    torch.manual_seed(seed)

    if model_name == "RG-P-VAE":
        # Poisson-latent branch: rsample relaxed counts and map to latent positions.
        with torch.no_grad():
            lambda_q = model.encode(A_norm, feats_t)        # [N, Dz]
            z_relaxed = model.poisson_rsample(lambda_q)                    # [N, Dz]
            z_embed = model._to_embedding(z_relaxed) if hasattr(model, "_to_embedding") else z_relaxed
        return z_embed.detach().cpu().numpy()
    
    elif model_name == "RG-G-VAE":

        # Default: Gaussian-latent branch (RG-G-VAE)
        try:
            mu, logvar = model.encode(A_norm, feats=feats_t)
        except TypeError:
            mu, logvar = model.encode(A_norm)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z.detach().cpu().numpy()
    
    else:
        raise ValueError(f"Unknown model name: {model_name}")


def compute_gwd_ignr(model: "RG_VAE",
                     A_norm: torch.Tensor,
                     feats_t: torch.Tensor,
                     Z_true: np.ndarray,
                     max_nodes: int = 2000,
                     num_samples: int = 5,
                     seed: int = 0,
                     max_iter: int = 200,
                     tol: float = 1e-9,
                     model_name: str = "RG-G-VAE") -> float:
    """
    IGNR-style GWD^2: compare metric spaces (Z_true, Z_hat) using POT's ot.gromov_wasserstein2.
    - Z_true: ground-truth latent positions (numpy [N,d_true])
    - Z_hat: samples from amortised posterior q(z|x,A) (torch -> numpy)
    We subsample nodes for tractability and average GW^2 over 'num_samples' posterior draws.
    """
    N = A_norm.size(0)
    idx = _subsample_indices(N, max_nodes, seed)
    Zt = Z_true[idx].astype(np.float64)
    Ct = _pairwise_euclidean(Zt)  # observed cost via true LPs

    k = Ct.shape[0]
    p = np.ones((k,), dtype=np.float64) / k
    q = p  # same k on both since we compare same node subset

    gws = []
    for s in range(num_samples):
        Zs = _posterior_sample_latents(model, A_norm, feats_t, model_name=model_name, seed=seed + s)
        Zh = Zs[idx].astype(np.float64)
        Ch = _pairwise_euclidean(Zh)
        # POT non-entropic GW^2 with squared loss
        try:
            gw2 = ot.gromov_wasserstein2(
                Ct, Ch, p, q, loss_fun='square_loss', max_iter=max_iter, tol=tol, verbose=False
            )
        except AttributeError:
            gw2 = ot.gromov.gromov_wasserstein2(
                Ct, Ch, p, q, loss_fun='square_loss', max_iter=max_iter, tol=tol, verbose=False
            )
        gws.append(float(gw2))
    return float(np.mean(gws))

def procrustes_rmse(Z_true: np.ndarray, Z_hat: np.ndarray, center: bool = True, scale: bool = False) -> float:
    """Optional LP-RMSE for reference (uses orthogonal Procrustes)."""
    X = Z_true.astype(np.float64)
    Y = Z_hat.astype(np.float64)
    if center:
        X -= X.mean(axis=0, keepdims=True)
        Y -= Y.mean(axis=0, keepdims=True)
    if scale:
        X /= max(np.linalg.norm(X), 1e-12)
        Y /= max(np.linalg.norm(Y), 1e-12)
    U, _, Vt = np.linalg.svd(Y.T @ X, full_matrices=False)
    R = U @ Vt
    aligned = Y @ R
    return float(np.sqrt(np.mean((aligned - X) ** 2)))

# ----------------- CLI -----------------

def parse_args():
    ap = argparse.ArgumentParser(description="Train RG-VAE on a batch of graphs (A1/A2/B1).")
    ap.add_argument("--setting_dir", required=True, help="Folder like sim_data_batch/A1 (contains many graphs)")
    ap.add_argument("--results_dir", default="results/")

    # model & training
    ap.add_argument("--model", default="RG-G-VAE", choices=["RG-G-VAE", "RG-P-VAE"], help="Model to train")
    # ap.add_argument("--feature_likelihood", default="gaussian", choices=["gaussian", "poisson"], help="Feature likelihood type")

    # training
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--latent_dim", type=int, default=16)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--neg_ratio", type=int, default=10)
    ap.add_argument("--split_seed", type=int, default=42)
    ap.add_argument("--val_frac", type=float, default=0.10)
    ap.add_argument("--test_frac", type=float, default=0.10)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")    

    # decoder
    ap.add_argument("--decoder", default="radial", choices=["radial", "dot", "bilinear", "indefinite", "mlp", "dc_radial", "rff"])
    ap.add_argument("--decoder_kwargs", default="{}", help='JSON dict for decoder (e.g., \'{"num_features":1024,"lengthscale":1.2,"ard":true}\')')

    # feature & loss
    ap.add_argument("--use_struct_feats", action="store_true")
    ap.add_argument("--feat_dec_hidden", type=int, default=64)
    ap.add_argument("--lambda_feat", type=float, default=1.0)
    ap.add_argument("--lambda_kl", type=float, default=5e-3)
    ap.add_argument("--kl_warmup_epochs", type=int, default=50)

    # val metrics
    ap.add_argument("--val_auc_neg_ratio", type=int, default=1)

    # per-graph pair budget (optional downsampling of pos edges per graph for speed)
    ap.add_argument("--max_pos_per_graph", type=int, default=0, help="0 = use all train positives; otherwise cap per graph per epoch")

    # IGNR-style GWD & LP-RMSE evaluation cadence
    ap.add_argument("--gwd_every", type=int, default=10, help="Every N epochs compute IGNR-style POT GWD (0=off)")
    ap.add_argument("--gwd_nodes", type=int, default=2000, help="Max nodes per graph for GWD (subsample)")
    ap.add_argument("--gwd_samples", type=int, default=5, help="# posterior samples to average GWD over")
    ap.add_argument("--gwd_max_graphs", type=int, default=3, help="Max # val graphs for each GWD pass")

    ap.add_argument("--lp_every", type=int, default=10, help="Every N epochs compute LP-RMSE (0=off)")
    ap.add_argument("--lp_nodes", type=int, default=5000, help="Max nodes per graph for LP-RMSE (subsample)")

    return ap.parse_args()

# ----------------- Main -----------------

def main():
    args = parse_args()

    # Create results directory with timestamp
    stamp = datetime.now().strftime("%m%d_%H%M")
    results_root = os.path.join(args.results_dir, stamp+"_"+os.path.basename(os.path.normpath(args.setting_dir)))
    os.makedirs(results_root, exist_ok=True)

    # Save full command line
    with open(os.path.join(results_root, "command.txt"), "w") as f:
        f.write(" ".join(sys.argv) + "\n")

    # Save args for reproducibility
    with open(os.path.join(results_root, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    device = torch.device(args.device)

    graph_dirs = list_graph_dirs(args.setting_dir)
    if not graph_dirs:
        raise SystemExit(f"No graphs found in {args.setting_dir}")

    # Inspect first graph to get input feature dim
    edges0, N0, directed0, base0 = load_graph_dir(graph_dirs[0])
    feats0 = load_node_features(graph_dirs[0], standardize=True)
    input_dim = feats0.shape[1]
    model = build_model(input_dim, args).to(device)
    print("Params: total: {}, encoder: {}, edge decoder: {}, node decoder: {}".format(
        sum(p.numel() for p in model.parameters()),
        sum(p.numel() for p in model.encoder.parameters()),
        sum(p.numel() for p in model.decoder.parameters()),
        sum(p.numel() for p in model.feature_decoder.parameters())
    ))
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Preload splits and adjacency/feats into memory
    loaded = []
    for gdir in graph_dirs:
        edges, N, directed, base = load_graph_dir(gdir)
        undirected = not directed
        split_path = os.path.join(gdir, f"{base}_splits_seed{args.split_seed}.npz")
        if os.path.exists(split_path):
            data = np.load(split_path)
            tr, va, te = data["train"], data["val"], data["test"]
        else:
            splits = split_edges(edges, val_frac=args.val_frac, test_frac=args.test_frac, seed=args.split_seed, undirected=undirected)
            tr, va, te = splits["train"], splits["val"], splits["test"]
            np.savez(split_path, train=tr, val=va, test=te)
        if args.max_pos_per_graph and tr.shape[0] > args.max_pos_per_graph:
            idx = np.random.default_rng(123).choice(tr.shape[0], size=args.max_pos_per_graph, replace=False)
            tr = tr[idx]
        A_norm = build_sparse_adj(N, edges, directed=directed, device=device, self_loops=True)
        feats = torch.from_numpy(load_node_features(gdir, standardize=True)).float().to(device)
        exclude = set((int(a), int(b)) for a, b in edges)
        loaded.append((gdir, base, N, undirected, A_norm, feats, tr, va, exclude))

    print(f"Loaded {len(loaded)} graphs from {args.setting_dir}. Results → {results_root}", flush=True)

    # Metric storage (per-epoch)
    tr_total_hist, tr_edge_hist, tr_feat_hist, tr_kl_hist = [], [], [], []
    va_total_hist, va_edge_hist, va_feat_hist, va_kl_hist = [], [], [], []
    va_auc_hist, va_ap_hist = [], []
    gwd_hist, lprmse_hist = [], []

    best_val_total = float("inf")
    best_epoch = -1
    best_ckpt = None

    for epoch in range(1, args.epochs + 1):
        lam_kl = current_lambda_kl(args.lambda_kl, epoch, args.kl_warmup_epochs)

        model.train()
        tr_totals = tr_edges = tr_feats = tr_kls = 0.0

        for (_, _, _, undirected, A_norm, feats_t, tr, _, exclude) in tqdm(loaded, desc=f"Epoch {epoch:03d}", ncols=80, file=sys.stdout):
            loss, stats = step_graph(
                model, A_norm, feats_t, tr, args.neg_ratio, undirected, exclude, device,
                lambda_feat=args.lambda_feat, lambda_kl=lam_kl
            )
            opt.zero_grad()
            loss.backward()
            opt.step()

            tr_totals += float(loss.item())
            tr_edges  += stats["recon_edge"]
            tr_feats  += stats["recon_feat"]
            tr_kls    += stats["kl"]

        # Validation (loss-like eval + AUC/AP)
        model.eval()
        va_totals = va_edges = va_feats = va_kls = 0.0
        va_aucs: List[float] = []
        va_aps:  List[float] = []
        with torch.no_grad():
            for (_, _, _, undirected, A_norm, feats_t, _, va, exclude) in loaded:
                loss, stats = step_graph(
                    model, A_norm, feats_t, va, args.neg_ratio, undirected, exclude, device,
                    lambda_feat=args.lambda_feat, lambda_kl=lam_kl
                )
                va_totals += float(loss.item())
                va_edges  += stats["recon_edge"]
                va_feats  += stats["recon_feat"]
                va_kls    += stats["kl"]

                auc, ap = val_metrics(
                    model, A_norm, feats_t, va, undirected, exclude, device, neg_ratio_for_auc=args.val_auc_neg_ratio
                )
                va_aucs.append(auc); va_aps.append(ap)

        nG = float(len(loaded))
        tr_total = tr_totals / nG
        tr_edge  = tr_edges  / nG
        tr_feat  = tr_feats  / nG
        tr_kl    = tr_kls    / nG
        va_total = va_totals / nG
        va_edge  = va_edges  / nG
        va_feat  = va_feats  / nG
        va_kl    = va_kls    / nG
        va_auc   = float(np.nanmean(va_aucs)) if len(va_aucs) else np.nan
        va_ap    = float(np.nanmean(va_aps))  if len(va_aps)  else np.nan

        # Append to histories
        tr_total_hist.append(tr_total); tr_edge_hist.append(tr_edge); tr_feat_hist.append(tr_feat); tr_kl_hist.append(tr_kl)
        va_total_hist.append(va_total); va_edge_hist.append(va_edge); va_feat_hist.append(va_feat); va_kl_hist.append(va_kl)
        va_auc_hist.append(va_auc); va_ap_hist.append(va_ap)

        # Periodic IGNR-style GWD & LP-RMSE on validation graphs
        gwd_mean = np.nan
        lprmse_mean = np.nan

        if args.gwd_every > 0 and (epoch % args.gwd_every == 0):
            gwd_vals = []
            picked = 0
            for (gdir, base, N, undirected, A_norm, feats_t, tr, va, exclude) in loaded:
                if "_val_" not in base:
                    continue
                Z_true = load_true_positions(gdir)
                if Z_true is None:
                    continue
                g = compute_gwd_ignr(
                    model, A_norm, feats_t, Z_true,
                    max_nodes=args.gwd_nodes,
                    num_samples=args.gwd_samples,
                    seed=epoch,
                    model_name=args.model
                )
                gwd_vals.append(g)
                picked += 1
                if picked >= args.gwd_max_graphs:
                    break
            if gwd_vals:
                gwd_mean = float(np.mean(gwd_vals))
        gwd_hist.append(gwd_mean)

        if args.lp_every > 0 and (epoch % args.lp_every == 0):
            lp_vals = []
            picked = 0
            for (gdir, base, N, undirected, A_norm, feats_t, tr, va, exclude) in loaded:
                if "_val_" not in base:
                    continue
                Z_true = load_true_positions(gdir)
                if Z_true is None:
                    continue
                Z_hat = _posterior_sample_latents(model, A_norm, feats_t, model_name=args.model, seed=epoch)
                idx = _subsample_indices(Z_true.shape[0], args.lp_nodes, seed=epoch)
                rmse = procrustes_rmse(Z_true[idx], Z_hat[idx], center=True, scale=False)
                lp_vals.append(rmse)
                picked += 1
                if picked >= args.gwd_max_graphs:
                    break
            if lp_vals:
                lprmse_mean = float(np.mean(lp_vals))
        lprmse_hist.append(lprmse_mean)

        print(
            f"Epoch {epoch:03d} | "
            f"train: total={tr_total:.4f}, edge={tr_edge:.4f}, feat={tr_feat:.4f}, KL={tr_kl:.4f}  ||  "
            f"val: total={va_total:.4f}, edge={va_edge:.4f}, feat={va_feat:.4f}, KL={va_kl:.4f}, "
            f"AUC={va_auc:.4f}, AP={va_ap:.4f}  ||  λ_KL={lam_kl:.4g}"
            + (f"  ||  GWD^2={gwd_mean:.6f}" if not np.isnan(gwd_mean) else "")
            + (f"  ||  LP-RMSE={lprmse_mean:.6f}" if not np.isnan(lprmse_mean) else ""),
            flush=True
        )

        # Save metrics so far (append-safe)
        np.savez(
            os.path.join(results_root, "metrics.npz"),
            tr_total=np.array(tr_total_hist, dtype=np.float32),
            tr_edge=np.array(tr_edge_hist, dtype=np.float32),
            tr_feat=np.array(tr_feat_hist, dtype=np.float32),
            tr_kl=np.array(tr_kl_hist, dtype=np.float32),
            va_total=np.array(va_total_hist, dtype=np.float32),
            va_edge=np.array(va_edge_hist, dtype=np.float32),
            va_feat=np.array(va_feat_hist, dtype=np.float32),
            va_kl=np.array(va_kl_hist, dtype=np.float32),
            va_auc=np.array(va_auc_hist, dtype=np.float32),
            va_ap=np.array(va_ap_hist, dtype=np.float32),
            gwd2=np.array(gwd_hist, dtype=np.float64),
            lp_rmse=np.array(lprmse_hist, dtype=np.float64),
            epochs=np.arange(1, len(tr_total_hist) + 1, dtype=np.int32),
            setting_dir=args.setting_dir,
        )

        # Track & save best by val total
        if va_total < best_val_total:
            best_val_total = va_total
            best_epoch = epoch
            ckpt = {
                "state_dict": model.state_dict(),
                "latent_dim": args.latent_dim,
                "hidden": args.hidden,
                "input_dim": input_dim,
                "decoder": args.decoder,
                "decoder_kwargs": json.loads(args.decoder_kwargs),
                "use_struct_feats": args.use_struct_feats,
                "feat_dec_hidden": args.feat_dec_hidden,
                "lambda_feat": args.lambda_feat,
                "lambda_kl": args.lambda_kl,
                "epoch": epoch,
                "best_val_total": best_val_total,
            }
            # Save to results/<stamp>/best.pt
            best_path_results = os.path.join(results_root, f"rg_vae_{os.path.basename(os.path.normpath(args.setting_dir))}_best.pt")
            torch.save(ckpt, best_path_results)
            print(f"Saved BEST model (val total={best_val_total:.4f}) to {best_path_results}", flush=True)

    # Save last checkpoint
    last_ckpt = {
        "state_dict": model.state_dict(),
        "latent_dim": args.latent_dim,
        "hidden": args.hidden,
        "input_dim": input_dim,
        "decoder": args.decoder,
        "decoder_kwargs": json.loads(args.decoder_kwargs),
        "use_struct_feats": args.use_struct_feats,
        "feat_dec_hidden": args.feat_dec_hidden,
        "lambda_feat": args.lambda_feat,
        "lambda_kl": args.lambda_kl,
        "epoch": args.epochs,
        "best_val_total": best_val_total,
        "best_epoch": best_epoch,
    }
    last_path_results = os.path.join(results_root, f"rg_vae_{os.path.basename(os.path.normpath(args.setting_dir))}_last.pt")
    torch.save(last_ckpt, last_path_results)

    print(f"Saved LAST model to {last_path_results}", flush=True)
    print(f"All metrics saved to {os.path.join(results_root, 'metrics.npz')}", flush=True)

if __name__ == "__main__":
    main()
