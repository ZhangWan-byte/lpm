import numpy as np
import torch
from torch_geometric.utils import to_scipy_sparse_matrix, subgraph
from ogb.nodeproppred import PygNodePropPredDataset
import os
import argparse

# -----------------------------
# USVT bits (yours, with helpers)
# -----------------------------
def cutoff(A, vmin=0.0, vmax=1.0):
    B = A.clone()
    B[B > vmax] = vmax
    B[B < vmin] = vmin
    return B

def get_subgraph_by_year(data, target_year):
    years = data.node_year.squeeze()
    node_mask = (years == target_year)
    selected_nodes = node_mask.nonzero(as_tuple=True)[0]

    if selected_nodes.numel() == 0:
        raise ValueError(f"No nodes found for year {target_year}.")

    sub_edge_index, _ = subgraph(selected_nodes, data.edge_index, relabel_nodes=True)

    sub_data = data.clone()
    sub_data.edge_index = sub_edge_index
    sub_data.x = data.x[selected_nodes]
    sub_data.y = data.y[selected_nodes]
    sub_data.node_year = data.node_year[selected_nodes]
    sub_data.num_nodes = selected_nodes.size(0)
    return sub_data

# -----------------------------
# New: Sliced Wasserstein distance (rotation-robust)
# -----------------------------
def sliced_w2_distance(X, Y, n_projections=256, n_quantiles=512, seed=0):
    """
    Approximate W2 distance between empirical measures supported on rows of X and Y
    using Sliced Wasserstein. Works well when embeddings may be arbitrarily rotated.
    """
    rng = np.random.RandomState(seed)
    d = X.shape[1]
    # If dimensions differ (can happen if #kept eigs differ), pad the smaller with zeros
    if Y.shape[1] != d:
        if Y.shape[1] < d:
            Y = np.hstack([Y, np.zeros((Y.shape[0], d - Y.shape[1]))])
        else:
            X = np.hstack([X, np.zeros((X.shape[0], Y.shape[1] - d))])

    W2_sq = 0.0
    qs = np.linspace(0.0, 1.0, n_quantiles)
    for _ in range(n_projections):
        u = rng.randn(d)
        norm = np.linalg.norm(u)
        if norm < 1e-12:
            continue
        u /= norm
        x_proj = X @ u
        y_proj = Y @ u
        # 1D W2 via quantile matching
        xq = np.quantile(x_proj, qs, method='linear')
        yq = np.quantile(y_proj, qs, method='linear')
        W2_sq += np.mean((xq - yq) ** 2)
    W2 = np.sqrt(W2_sq / n_projections)
    return W2

# -----------------------------
# Utilities for Wasserstein distances between embeddings
# -----------------------------
def _pad_to_common_dim(X: np.ndarray, Y: np.ndarray):
    """Pad both embeddings with zero-columns so they share the same feature dimension."""
    d1, d2 = X.shape[1], Y.shape[1]
    d = max(d1, d2)
    if d1 < d:
        X = np.hstack([X, np.zeros((X.shape[0], d - d1), dtype=X.dtype)])
    if d2 < d:
        Y = np.hstack([Y, np.zeros((Y.shape[0], d - d2), dtype=Y.dtype)])
    return X, Y, d

def sliced_w2_distance_numpy(X, Y, n_projections=256, n_quantiles=512, seed=0):
    """
    Rotation-robust Sliced W2 using random 1D projections (NumPy fallback).
    Handles different feature dims by zero-padding to a common dim first.
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    X, Y, d = _pad_to_common_dim(X, Y)

    rng = np.random.RandomState(seed)
    qs = np.linspace(0.0, 1.0, n_quantiles)
    W2_sq = 0.0

    for _ in range(n_projections):
        u = rng.randn(d)
        norm = np.linalg.norm(u)
        if norm < 1e-12:
            continue
        u /= norm
        x_proj = X @ u
        y_proj = Y @ u
        # 1D W2 via quantile matching on a fixed grid (works for unequal sample sizes)
        xq = np.quantile(x_proj, qs)
        yq = np.quantile(y_proj, qs)
        W2_sq += np.mean((xq - yq) ** 2)

    return float(np.sqrt(W2_sq / n_projections))

def sliced_w2_distance_pot(X, Y, n_projections=256, seed=0):
    """
    Prefer POT's implementation if available.
    pip install POT
    """
    try:
        import ot
        from ot.sliced import sliced_wasserstein_distance  # POT >= 0.8
    except Exception:
        return None  # will trigger fallback

    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    X, Y, _ = _pad_to_common_dim(X, Y)
    # POT returns a distance (not squared); we keep that convention.
    return float(sliced_wasserstein_distance(X, Y, n_projections=n_projections, seed=seed))

def wasserstein_distance_between_embeddings(X, Y, n_projections=256, n_quantiles=512, seed=0, use_pot=True):
    """
    Wrapper: try POT's sliced W2; otherwise use the NumPy fallback above.
    """
    if use_pot:
        val = sliced_w2_distance_pot(X, Y, n_projections=n_projections, seed=seed)
        if val is not None:
            return val
    return sliced_w2_distance_numpy(X, Y, n_projections=n_projections, n_quantiles=n_quantiles, seed=seed)

# -----------------------------
# New: USVT + embedding
# -----------------------------
def usvt_and_embed(A_t, gamma, d_max=None, energy=None, eps=1e-10, verbose=True):
    """
    1) Eigendecompose A (symmetric)
    2) USVT-threshold eigenvalues at gamma * sqrt(rho * n)
    3) Scale by 1/rho (since eigvals of \hat P = s_thr / rho)
    4) Build latent positions X = V_kept * sqrt(Lambda_kept)
       - keep by top-k (d_max) and/or energy fraction (on post-scaled positive eigvals)

    Returns:
        X: np.ndarray of shape (n, k_kept)
    """
    # usvt - phase 1 - eigendecomposition
    n = A_t.shape[0]
    rho = A_t.mean().item() if n > 0 else 1.0
    rho = max(rho, 1e-12)
    evals, evecs = torch.linalg.eigh(A_t)

    # usvt - phase 2 - thresholding
    thr = gamma * torch.sqrt(torch.tensor(rho * n, dtype=evals.dtype, device=evals.device))
    suppressed = (evals < thr).sum().item()
    if verbose:
        print(f"{suppressed / n:.3f} of eigenvalues suppressed")
    evals = torch.where(evals < thr, torch.zeros_like(evals), evals)

    evals = evals / rho

    # keep positive eigenvalues
    pos = evals > eps
    if pos.sum().item() == 0:
        if verbose:
            print(f"[USVT] rho={rho:.6e}; n={n}; all components suppressed, returning zeros.")
        return np.zeros((n, 1), dtype=np.float64)
    evals_pos = evals[pos]
    evecs_pos = evecs[:, pos]

    # usvt - phase 3 - embedding
    idx = torch.argsort(evals_pos, descending=True)
    evals_sorted = evals_pos[idx]
    evecs_sorted = evecs_pos[:, idx]

    k = evals_sorted.numel()

    if energy is not None:
        cum = torch.cumsum(evals_sorted, dim=0)
        total = evals_sorted.sum()
        k = int(torch.searchsorted(cum, energy * total).item() + 1)

    if d_max is not None:
        k = min(k, d_max)

    evals_k = evals_sorted[:k]
    evecs_k = evecs_sorted[:, :k]
    X = evecs_k * torch.sqrt(torch.clamp(evals_k, min=0.0))

    if verbose:
        print(f"[USVT] rho={rho:.6e}; n={n}; kept dims={k}")

    return X.cpu().numpy().astype(np.float64)


def usvt_and_embed_gpu(
    A_t: torch.Tensor,
    gamma: float,
    d_max: int | None = None,
    energy: float | None = None,
    eps: float = 1e-10,
    verbose: bool = True,
    *,
    device: str | torch.device | None = None,
    dtype: torch.dtype = torch.float64,
    return_numpy: bool = True
):
    """
    USVT + embedding in ONE eigendecomposition (GPU-optimized).

    Args:
        A_t: (n, n) symmetric adjacency (dense). Will be moved to `device`/`dtype`.
        gamma: threshold multiplier.
        d_max: keep at most this many components (after energy rule).
        energy: if set in (0,1], keep the smallest k s.t. sum(lam[:k]) >= energy * sum(lam).
        eps: discard non-positive eigenvalues below this.
        verbose: log suppression stats.
        device: e.g. "cuda" or "cpu". If None, use A_t.device.
        dtype: torch.float64 (default) or torch.float64 (slower on GPU).
        return_numpy: if True, returns np.ndarray; else returns torch.Tensor on `device`.

    Returns:
        X: (n, k_kept) latent positions for the USVT-denoised P-hat.
    """
    # ---- device / dtype placement
    if device is None:
        device = A_t.device
    A_t = A_t.to(device=device, dtype=dtype, non_blocking=True)

    n = A_t.shape[0]
    n_t = torch.tensor(n, device=device, dtype=dtype)

    # rho as tensor (avoid .item() CPU sync)
    rho = A_t.mean() if n > 0 else torch.tensor(1.0, device=device, dtype=dtype)
    rho = torch.clamp(rho, min=torch.tensor(1e-12, device=device, dtype=dtype))

    # ---- single eigendecomposition (ascending eigenvalues)
    evals, evecs = torch.linalg.eigh(A_t)  # CUDA-supported for float64/64

    # ---- USVT threshold (all on device)
    thr = torch.sqrt(rho * n_t) * float(gamma)  # gamma is Python float; cast once
    mask = evals >= thr
    if verbose:
        suppressed = (evals.numel() - mask.sum()).to(dtype=dtype) / n_t
        print(f"{suppressed.item():.3f} of eigenvalues suppressed")

    # zero-out below threshold (in-place where possible)
    evals = evals.masked_fill(~mask, 0)
    # scale to get eigenvalues of P_hat
    lam = evals / rho  # may contain zeros

    # keep strictly positive eigenvalues
    pos = lam > eps
    if pos.sum() == 0:
        if verbose:
            print(f"[USVT] rho={rho.item():.6e}; n={n}; all components suppressed, returning zeros.")
        X_zero = torch.zeros((n, 1), device=device, dtype=dtype)
        return X_zero.cpu().numpy() if return_numpy else X_zero

    lam_pos = lam[pos]
    V_pos = evecs[:, pos]

    # sort by descending eigenvalue once
    lam_sorted, sort_idx = torch.sort(lam_pos, descending=True)
    V_sorted = V_pos[:, sort_idx]

    # choose k via energy and/or d_max
    k = lam_sorted.numel()
    if energy is not None:
        # smallest k with cumulative energy >= target
        cum = torch.cumsum(lam_sorted, dim=0)
        total = cum[-1]
        target = total * float(energy)
        # searchsorted expects ascending, so flip signs or use this trick:
        k = int(torch.searchsorted(cum, target).item() + 1)

    if d_max is not None:
        k = min(k, int(d_max))

    lam_k = lam_sorted[:k]
    V_k = V_sorted[:, :k]

    # latent coordinates X = V_k * sqrt(lam_k)
    X = V_k * torch.sqrt(torch.clamp(lam_k, min=0))

    if verbose:
        # note: rho.item() triggers a sync only if you print; fine under verbose
        print(f"[USVT] rho={rho.item():.6e}; n={n}; kept dims={k}")

    if return_numpy:
        return X.detach().cpu().numpy()
    return X  # keep on GPU


# -----------------------------
# Main: baseline 2010 vs 2011-2020
# -----------------------------
if __name__ == "__main__":
    # Reproducibility
    np.random.seed(0)
    torch.manual_seed(0)

    parser = argparse.ArgumentParser(description="USVT on OGB arXiv dataset")
    parser.add_argument("--gamma", type=float, default=0.01, help="USVT threshold multiplier")
    parser.add_argument("--energy", type=float, default=None, help="Energy preservation ratio")
    parser.add_argument("--d_max", type=int, default=2, help="Maximum dimensionality")
    parser.add_argument("--max_nodes", type=int, default=20000, help="Maximum number of nodes to use")
    parser.add_argument("--device", type=str, help="Device to use (e.g., 'cuda' or 'cpu')")
    args = parser.parse_args()

    # Device configuration
    device = torch.device(args.device)

     # USVT sweep grid (tweak if you like)
    GAMMA = args.gamma
    ENERGY_KEEP = args.energy
    D_MAX = args.d_max
    MAX_NODES = args.max_nodes

    # Output dirs
    os.makedirs("usvt_C2_v4", exist_ok=True)

    # Load data
    dataset = PygNodePropPredDataset(name='ogbn-arxiv')
    data = dataset[0]

    # Helper to build dense symmetrized adjacency for a year
    def build_dense_adj_for_year(year, max_nodes, device):
        sub_data = get_subgraph_by_year(data, year)
        n = sub_data.num_nodes
        
        # Decide which nodes to keep (optionally downsample)
        if n > max_nodes:
            rng = np.random.default_rng(seed=42)
            keep = rng.choice(n, size=max_nodes, replace=False).astype(np.int64)
            keep_idx = torch.from_numpy(keep)
            sub_edge_index, _ = subgraph(keep_idx, sub_data.edge_index, relabel_nodes=True, num_nodes=n)
            edge_index = sub_edge_index
            n_use = max_nodes
        else:
            edge_index = sub_data.edge_index
            n_use = n

        A_sparse = to_scipy_sparse_matrix(edge_index, num_nodes=n_use).astype(np.float64)
        A_sparse = 0.5 * (A_sparse + A_sparse.T)   # symmetrize
        A_sparse.setdiag(0.0)
        A_sparse.eliminate_zeros()  # remove zero entries
        A_dense = A_sparse.toarray()

        return torch.from_numpy(A_dense).to(device).type(torch.float64)

    # Build baseline (2010)
    baseline_year = 2010
    print(f"=== Baseline year {baseline_year} ===")
    A2010 = build_dense_adj_for_year(baseline_year, MAX_NODES, device)
    X2010 = usvt_and_embed_gpu(
        A_t=A2010,
        gamma=GAMMA,
        d_max=D_MAX,
        energy=ENERGY_KEEP,
        eps=1e-10,
        verbose=True,
        device=device,
        dtype=torch.float64,
        return_numpy=True
    )

    np.save("usvt_C2_v4/X_2010.npy", X2010)

    # Compare 2011..2020 to 2010 via Sliced W2
    results = {}
    for year in range(2011, 2021):
        print(f"\n=== Processing year {year} ===")
        A_year = build_dense_adj_for_year(year, MAX_NODES, device)
        X_year = usvt_and_embed_gpu(
            A_t=A_year,
            gamma=GAMMA,
            d_max=D_MAX,
            energy=ENERGY_KEEP,
            eps=1e-10,
            verbose=True,
            device=device,
            dtype=torch.float64,
            return_numpy=True
        )
        np.save(f"usvt_C2_v4/X_{year}.npy", X_year)

        # Sliced W2 distance to baseline latent positions
        dist = wasserstein_distance_between_embeddings(
            X2010, X_year,
            n_projections=256,   # you can lower to 128 for speed
            n_quantiles=512,     # only used by the NumPy fallback
            seed=42,
            use_pot=True         # tries POT first, then falls back
        )

        results[year] = {
            "sliced_W2_to_2010": float(dist)
        }
        print(f"[{year}] Sliced W2 to 2010: {dist:.6f}")

    # Save the summary table
    np.save("usvt_C2_v4/sliced_W2_results.npy", results)
    # Also write a human-readable text file
    with open("usvt_C2_v4/sliced_W2_results.txt", "w") as f:
        for y in range(2011, 2021):
            r = results[y]
            f.write(f"{y}\tW2={r['sliced_W2_to_2010']:.6f}\n")

    print("\n=== Done. Summary (Sliced W2 to 2010) ===")
    for y in range(2011, 2021):
        print(f"{y}: {results[y]['sliced_W2_to_2010']:.6f}")
