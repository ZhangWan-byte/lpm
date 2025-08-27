# File: models/utils.py

import os
import json
import math
from typing import Tuple, List, Dict, Optional
import numpy as np
import torch


def load_graph_dir(graph_dir: str) -> Tuple[np.ndarray, int, bool, str]:
    """Load edges and metadata from a generator output directory.
    Expects files like *_edges.txt and *_meta.json.
    Returns: (edges ndarray [M,2], N, directed, base_name)
    """
    # find first *_edges.txt
    edge_files = [f for f in os.listdir(graph_dir) if f.endswith("_edges.txt")]
    if not edge_files:
        raise FileNotFoundError(f"No *_edges.txt found in {graph_dir}")
    edge_file = os.path.join(graph_dir, edge_files[0])
    base = edge_files[0].replace("_edges.txt", "")

    edges = np.loadtxt(edge_file, dtype=int)
    if edges.ndim == 1 and edges.size == 0:
        edges = edges.reshape(0, 2)
    elif edges.ndim == 1:
        edges = edges.reshape(1, 2)

    meta_file = os.path.join(graph_dir, f"{base}_meta.json")
    N = int(edges.max()) + 1 if edges.size > 0 else 0
    directed = False
    if os.path.exists(meta_file):
        with open(meta_file, "r") as f:
            meta = json.load(f)
            N = int(meta.get("N", N))
            directed = bool(meta.get("directed", False))
    return edges, N, directed, base


def build_sparse_adj(N: int, edges: np.ndarray, directed: bool, device: torch.device, self_loops: bool=True) -> torch.Tensor:
    """Return normalized adjacency (D^{-1/2} A D^{-1/2}) as torch.sparse_coo_tensor."""
    if N == 0:
        return torch.sparse_coo_tensor(torch.zeros((2,0), dtype=torch.long),
                                       torch.zeros(0, dtype=torch.float32),
                                       (0, 0)).to(device)

    # Build edge list (symmetrize if undirected)
    if edges.size == 0:
        idx = torch.arange(N)
        values = torch.ones(N, dtype=torch.float32)
        A = torch.sparse_coo_tensor(torch.stack([idx, idx]), values, (N, N))
    else:
        if not directed:
            und = np.vstack([edges, edges[:, [1, 0]]])
            edges = und
        i = torch.from_numpy(edges.T).long()            # [2, M]
        v = torch.ones(edges.shape[0], dtype=torch.float32)
        if self_loops:
            loop_idx = torch.arange(N, dtype=torch.long)
            i = torch.cat([i, torch.stack([loop_idx, loop_idx])], dim=1)
            v = torch.cat([v, torch.ones(N, dtype=torch.float32)])
        A = torch.sparse_coo_tensor(i, v, (N, N))

    # Coalesce before using indices/values
    A = A.coalesce()
    idx = A.indices()
    val = A.values()

    # Degree from coalesced A
    deg = torch.sparse.sum(A, dim=1).to_dense()  # [N]
    deg_inv_sqrt = (deg + 1e-8).pow(-0.5)

    # Scale edge values: v_ij <- v_ij * d_i^{-1/2} * d_j^{-1/2}
    row, col = idx[0], idx[1]
    val_norm = val * deg_inv_sqrt[row] * deg_inv_sqrt[col]

    A_norm = torch.sparse_coo_tensor(idx, val_norm, (N, N)).coalesce().to(device)
    return A_norm


def split_edges(edges: np.ndarray, val_frac: float, test_frac: float, seed: int, undirected: bool=True) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    E = edges.copy()
    if undirected:
        # keep only i<j
        mask = E[:,0] < E[:,1]
        E = E[mask]
    rng.shuffle(E)
    m = E.shape[0]
    n_test = int(test_frac * m)
    n_val = int(val_frac * m)
    test_edges = E[:n_test]
    val_edges = E[n_test:n_test+n_val]
    train_edges = E[n_test+n_val:]
    return {"train": train_edges, "val": val_edges, "test": test_edges}


def negative_sampling(N: int, num_samples: int, exclude: Optional[set]=None, undirected: bool=True, rng: Optional[np.random.Generator]=None) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()
    negs = set()
    tries = 0
    max_tries = num_samples * 20 + 1000
    while len(negs) < num_samples and tries < max_tries:
        i = int(rng.integers(0, N))
        j = int(rng.integers(0, N))
        if i == j:
            tries += 1
            continue
        a, b = (i, j) if (i < j or not undirected) else (j, i)
        if exclude and ((a, b) in exclude or (b, a) in exclude):
            tries += 1
            continue
        negs.add((a, b))
        tries += 1
    if len(negs) < num_samples:
        # fall back: fill remaining by simple grid walk (rare)
        i = 0
        while len(negs) < num_samples:
            a = i % N
            b = (i*7 + 3) % N
            if a != b:
                tup = (a, b) if (a < b or not undirected) else (b, a)
                if not exclude or (tup not in exclude and (tup[1], tup[0]) not in exclude):
                    negs.add(tup)
            i += 1
    return np.array(list(negs), dtype=int)


def bce_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)


def auc_ap(scores: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
    # AUC via rank statistic (Mann–Whitney U)
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan"), float("nan")
    ranks = np.argsort(np.argsort(np.concatenate([neg, pos]))) + 1
    r_pos = ranks[len(neg):].sum()
    auc = (r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
    # AP (average precision)
    order = np.argsort(-scores)
    y = labels[order]
    cum_tp = np.cumsum(y)
    precision = cum_tp / (np.arange(len(y)) + 1)
    ap = (precision[y == 1]).mean() if (y == 1).any() else float("nan")
    return float(auc), float(ap)