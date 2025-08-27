# File: exp1_eval.py

import os
import argparse
from typing import List
import numpy as np
import torch
from models.rg_vae import RG_VAE
from models.utils import load_graph_dir, build_sparse_adj, negative_sampling, auc_ap


def parse_args():
    ap = argparse.ArgumentParser(description="Evaluate Random Graph VAE on held-out edges")
    ap.add_argument("--data_root", required=True,
                    help="Root folder containing graph subfolders (e.g., D:/rebuttal2025/out)")
    ap.add_argument("--models_dir", default="models")
    ap.add_argument("--split_seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--neg_ratio", type=int, default=10)
    return ap.parse_args()


def list_graph_dirs(root: str) -> List[str]:
    subs = []
    for name in os.listdir(root):
        p = os.path.join(root, name)
        if os.path.isdir(p):
            if any(f.endswith("_edges.txt") for f in os.listdir(p)):
                subs.append(p)
    subs.sort()
    return subs


def main():
    args = parse_args()
    device = torch.device(args.device)

    # load model
    ckpt_path = os.path.join(args.models_dir, "rg_vae.pt")
    ckpt = torch.load(ckpt_path, map_location=device)
    model = RG_VAE(latent_dim=ckpt.get("latent_dim", 16),
                   hidden=ckpt.get("hidden", 64))
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()

    graph_dirs = list_graph_dirs(args.data_root)
    all_auc, all_ap = [], []

    with torch.no_grad():
        for gdir in graph_dirs:
            edges, N, directed, base = load_graph_dir(gdir)
            undirected = not directed
            split_path = os.path.join(gdir, f"{base}_splits_seed{args.split_seed}.npz")
            if not os.path.exists(split_path):
                print(f"Split not found for {base}, skipping. (Train first with same seed)")
                continue
            data = np.load(split_path)
            val_edges = data["val"]
            test_edges = data["test"]

            A_norm = build_sparse_adj(N, edges, directed=directed,
                                      device=device, self_loops=True)
            Z = model.embed(A_norm)

            # evaluate on val+test combined
            pos_edges = np.vstack([val_edges, test_edges])
            exclude = set((int(a), int(b)) for a, b in edges)
            neg_edges = negative_sampling(N,
                                          num_samples=len(pos_edges) * args.neg_ratio,
                                          exclude=exclude,
                                          undirected=undirected)

            pairs = np.vstack([pos_edges, neg_edges])
            labels = np.concatenate([np.ones(len(pos_edges)),
                                     np.zeros(len(neg_edges))]).astype(int)

            pairs_t = torch.from_numpy(pairs).long().to(device)
            logits = model.pair_logits(Z, pairs_t).cpu().numpy()
            scores = 1.0 / (1.0 + np.exp(-logits))  # sigmoid
            auc, ap = auc_ap(scores, labels)
            all_auc.append(auc)
            all_ap.append(ap)
            print(f"{os.path.basename(gdir)}: AUC={auc:.4f}, AP={ap:.4f}")

    if all_auc:
        print(f"Overall: mean AUC={np.mean(all_auc):.4f} ± {np.std(all_auc):.4f}; "
              f"mean AP={np.mean(all_ap):.4f} ± {np.std(all_ap):.4f}")


if __name__ == "__main__":
    main()
