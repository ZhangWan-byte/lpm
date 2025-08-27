# File: exp1_train.py

import os
import json
import argparse
from typing import Optional, Dict, List
import numpy as np
import torch

from models.rg_vae import RG_VAE
from models.utils import (
    load_graph_dir,
    build_sparse_adj,
    split_edges,
    negative_sampling,
)

# ------------------------
# Helpers
# ------------------------

def find_dataset_dir(root: str, rgm_type: str) -> str:
    """
    Find the first subdirectory of `root` whose name contains `rgm_type` (case-insensitive).
    If `root` itself is a dataset directory (contains *_edges.txt), return root.
    """
    root = os.path.abspath(root)
    if os.path.isdir(root) and any(f.endswith("_edges.txt") for f in os.listdir(root)):
        return root

    rgm_type_lower = rgm_type.lower()
    candidates = []
    for name in os.listdir(root):
        p = os.path.join(root, name)
        if os.path.isdir(p) and rgm_type_lower in name.lower():
            if any(f.endswith("_edges.txt") for f in os.listdir(p)):
                candidates.append(p)
    if not candidates:
        raise FileNotFoundError(
            f"Could not find dataset for rgm_type='{rgm_type}' under {root}. "
            f"Make sure there is a subfolder like .../{rgm_type}_... containing *_edges.txt."
        )
    candidates.sort()
    return candidates[0]


def load_node_features(graph_dir: str, standardize: bool = True) -> np.ndarray:
    """
    Load node features from *_nodes.npz if 'node_features' exists; otherwise synthesize
    from latent positions via a nonlinear random Fourier map (x = f(z)).
    """
    node_npzs = [os.path.join(graph_dir, f) for f in os.listdir(graph_dir) if f.endswith("_nodes.npz")]
    if not node_npzs:
        raise FileNotFoundError(f"No *_nodes.npz found in {graph_dir}")
    npz = np.load(node_npzs[0])

    if "node_features" in npz.files:
        feats = np.array(npz["node_features"], dtype=np.float32)
    else:
        # Synthesize observable features from latent positions
        X = np.array(npz["positions"], dtype=np.float32)  # [N, d]
        N, d = X.shape
        rng = np.random.default_rng(7)
        m = min(4 * d, 32)
        W = rng.normal(0.0, 1.0, size=(d, m)).astype(np.float32)
        P = X @ W
        feats = np.concatenate([X, X**2, np.sin(P), np.cos(P)], axis=1).astype(np.float32)

    if standardize and feats.size > 0:
        mu = feats.mean(axis=0, keepdims=True)
        sd = feats.std(axis=0, keepdims=True) + 1e-6
        feats = (feats - mu) / sd
    return feats


def print_dataset_info(graph_dir: str, N: int, edges: np.ndarray, directed: bool, feats: np.ndarray, splits: Dict[str, np.ndarray]):
    M = edges.shape[0]
    avg_deg = (M * (2 if not directed else 1)) / max(1, N)
    density = (M / (N * (N - 1))) if directed else (M / (N * (N - 1) / 2))
    base = os.path.basename(graph_dir)
    print("=" * 70)
    print(f"Dataset: {base}")
    print(f"Path:    {graph_dir}")
    print(f"N nodes: {N:,} | M edges: {M:,} | directed={directed}")
    print(f"Avg degree: {avg_deg:.2f} | density: {density:.6f}")
    print(f"Features: dim={feats.shape[1] if feats.ndim == 2 else 0}")
    print(f"Split sizes: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")
    print("=" * 70)


# ------------------------
# Training / Validation
# ------------------------

def step_elbo(
    model: RG_VAE,
    A_norm: torch.Tensor,
    pos_edges: np.ndarray,
    neg_ratio: int,
    N: int,
    undirected: bool,
    exclude_pairs: Optional[set],
    device: torch.device,
    feats: torch.Tensor,
    lambda_feat: float,
    lambda_kl: float,
):
    n_pos = pos_edges.shape[0]
    neg_edges = negative_sampling(
        N,
        n_pos * neg_ratio,
        exclude=exclude_pairs,
        undirected=undirected,
    )

    pos = torch.from_numpy(pos_edges).long().to(device)
    neg = torch.from_numpy(neg_edges).long().to(device)

    loss, stats = model.elbo(A_norm, pos, neg, feats=feats, lambda_feat=lambda_feat, lambda_kl=lambda_kl)
    return loss, stats


def parse_args():
    ap = argparse.ArgumentParser(description="Train RG-VAE (two-branch) on a SINGLE generated RGM dataset (Experiment 1)")
    ap.add_argument("--data_root", required=True, help="Root folder containing all generated sets (e.g., D:/rebuttal2025/sim_data)")
    ap.add_argument("--rgm_type", required=True, help="Which dataset to train on (e.g., A1, A2, B1, B2, ...). Matches subfolder name.")
    ap.add_argument("--models_dir", default="models/checkpoints", help="Where to save checkpoints (default ./models/checkpoints)")

    # model/training
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--latent_dim", type=int, default=16)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--neg_ratio", type=int, default=5, help="#negatives per positive (train & val)")
    ap.add_argument("--val_frac", type=float, default=0.10)
    ap.add_argument("--test_frac", type=float, default=0.10)
    ap.add_argument("--split_seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    # decoder options
    ap.add_argument("--decoder", default="radial",
                    choices=["radial", "dot", "bilinear", "indefinite", "mlp", "dc_radial", "rff"])
    ap.add_argument("--decoder_kwargs", default="{}",
                    help='JSON dict of kwargs for the edge decoder (e.g., \'{"num_features":512,"lengthscale":2.0}\')')

    # feature branch & encoder options
    ap.add_argument("--use_struct_feats", action="store_true",
                    help="If set, concatenate simple structural features to node features for the encoder.")
    ap.add_argument("--feat_dec_hidden", type=int, default=64,
                    help="Hidden width of the feature decoder MLP.")

    # loss weights
    ap.add_argument("--lambda_feat", type=float, default=1.0, help="Weight for feature reconstruction loss.")
    ap.add_argument("--lambda_kl", type=float, default=1e-3, help="Weight for KL term.")

    return ap.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.models_dir, exist_ok=True)
    device = torch.device(args.device)

    # Locate dataset directory by rgm_type
    ds_dir = find_dataset_dir(args.data_root, args.rgm_type)

    # Load graph
    edges, N, directed, base = load_graph_dir(ds_dir)
    undirected = not directed
    A_norm = build_sparse_adj(N, edges, directed=directed, device=device, self_loops=True)

    # Load/synthesize node features x and standardize
    feats_np = load_node_features(ds_dir, standardize=True)  # [N, F]
    if feats_np.shape[0] != N:
        raise ValueError(f"Node features rows ({feats_np.shape[0]}) != N ({N}) in {ds_dir}")
    feats = torch.from_numpy(feats_np).float().to(device)

    # Make splits (persist so eval can reuse)
    split_path = os.path.join(ds_dir, f"{base}_splits_seed{args.split_seed}.npz")
    if os.path.exists(split_path):
        data = np.load(split_path)
        train_edges = data["train"]
        val_edges = data["val"]
        test_edges = data["test"]
    else:
        splits = split_edges(edges, val_frac=args.val_frac, test_frac=args.test_frac,
                             seed=args.split_seed, undirected=undirected)
        train_edges, val_edges, test_edges = splits["train"], splits["val"], splits["test"]
        np.savez(split_path, train=train_edges, val=val_edges, test=test_edges)

    # Show dataset info
    print_dataset_info(ds_dir, N, edges, directed, feats_np, {"train": train_edges, "val": val_edges, "test": test_edges})

    # Build model
    try:
        dec_kwargs = json.loads(args.decoder_kwargs)
        if not isinstance(dec_kwargs, dict):
            raise ValueError
    except Exception:
        raise SystemExit("--decoder_kwargs must be a JSON object string, e.g. '{\"num_features\":512}'")

    model = RG_VAE(
        input_dim=feats.shape[1],
        latent_dim=args.latent_dim,
        hidden=args.hidden,
        enc_layers=2,
        use_struct_feats=args.use_struct_feats,
        decoder=args.decoder,
        decoder_kwargs=dec_kwargs,
        feat_dec_hidden=args.feat_dec_hidden,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Prepare exclude set for negatives
    exclude = set((int(a), int(b)) for a, b in edges)

    # Training loop (single dataset)
    for epoch in range(1, args.epochs + 1):
        # ---- Train
        model.train()
        tr_loss, tr_recon_edge, tr_recon_feat, tr_kl = 0.0, 0.0, 0.0, 0.0

        loss, stats = step_elbo(
            model, A_norm, train_edges, args.neg_ratio, N, undirected, exclude, device, feats,
            lambda_feat=args.lambda_feat, lambda_kl=args.lambda_kl
        )
        opt.zero_grad()
        loss.backward()
        opt.step()

        tr_loss += float(loss.item())
        tr_recon_edge += stats["recon_edge"]
        tr_recon_feat += stats["recon_feat"]
        tr_kl += stats["kl"]

        # ---- Validation
        model.eval()
        with torch.no_grad():
            val_loss, val_stats = step_elbo(
                model, A_norm, val_edges, args.neg_ratio, N, undirected, exclude, device, feats,
                lambda_feat=args.lambda_feat, lambda_kl=args.lambda_kl
            )

        print(
            f"Epoch {epoch:03d} | "
            f"train: total={tr_loss:.4f}, edge={tr_recon_edge:.4f}, feat={tr_recon_feat:.4f}, KL={tr_kl:.4f}  ||  "
            f"val: total={float(val_loss.item()):.4f}, edge={val_stats['recon_edge']:.4f}, feat={val_stats['recon_feat']:.4f}, KL={val_stats['kl']:.4f}"
        )

    # Save checkpoint for this dataset
    ckpt_path = os.path.join(args.models_dir, f"rg_vae_{os.path.basename(ds_dir)}.pt")
    torch.save(
        {
            "state_dict": model.state_dict(),
            "latent_dim": args.latent_dim,
            "hidden": args.hidden,
            "input_dim": feats.shape[1],
            "decoder": args.decoder,
            "decoder_kwargs": dec_kwargs,
            "use_struct_feats": args.use_struct_feats,
            "feat_dec_hidden": args.feat_dec_hidden,
            "lambda_feat": args.lambda_feat,
            "lambda_kl": args.lambda_kl,
        },
        ckpt_path,
    )
    print(f"Saved model to {ckpt_path}")


if __name__ == "__main__":
    main()
