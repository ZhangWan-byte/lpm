import numpy as np
import torch
from torch_geometric.utils import to_scipy_sparse_matrix, subgraph
from ogb.nodeproppred import PygNodePropPredDataset
from tqdm import tqdm
import os

# -----------------------------
# USVT bits (yours, with helpers)
# -----------------------------
def cutoff(A, vmin=0.0, vmax=1.0):
    B = A.clone()
    B[B > vmax] = vmax
    B[B < vmin] = vmin
    return B

def USVT_threshold(s, v, n, gamma=0.5, rho=None, cut=True, vmin=0.0, vmax=1.0, verbose=False):
    thr = gamma * torch.sqrt(torch.tensor(rho * n, dtype=s.dtype, device=s.device))
    suppressed = (s < thr).sum().item()
    if verbose:
        print(f"{suppressed / n:.3f} of eigenvalues suppressed")

    s = torch.where(s < thr, torch.zeros_like(s), s)

    Ahat = (v * s) @ v.T
    Ahat = Ahat / rho

    if cut:
        Ahat = cutoff(Ahat, vmin, vmax)

    return Ahat

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
# New: from USVT matrix -> latent positions
# -----------------------------
def gram_to_latent_positions(W_hat_t, d_max=None, energy=None, eps=1e-10):
    """
    From symmetric (denoised) Gram/probability matrix W_hat_t to latent node embeddings.
    Keep only positive eigenvalues; then X = U * sqrt(Lambda).

    Args:
        W_hat_t: torch.Tensor (n x n), symmetric.
        d_max: int or None. If set, cap the number of components to this.
        energy: float in (0,1] or None. If set, choose smallest k explaining this fraction of sum of positive eigenvalues.
        eps: threshold for "positive" eigenvalues.
    Returns:
        X: np.ndarray (n x d_kept)
        kept_eigs: np.ndarray of kept eigenvalues
    """
    # eigh returns ascending eigenvalues
    evals, evecs = torch.linalg.eigh(W_hat_t)
    pos = evals > eps
    evals = evals[pos]
    evecs = evecs[:, pos]

    if evals.numel() == 0:
        # fallback: everything was suppressed; return zeros with 1 dim
        n = W_hat_t.shape[0]
        return np.zeros((n, 1), dtype=np.float64), np.array([], dtype=np.float64)

    # sort descending (largest signal first)
    idx = torch.argsort(evals, descending=True)
    evals = evals[idx]
    evecs = evecs[:, idx]

    if energy is not None:
        cum = torch.cumsum(evals, dim=0)
        total = evals.sum()
        k = int(torch.searchsorted(cum, energy * total).item() + 1)
    else:
        k = evals.numel()

    if d_max is not None:
        k = min(k, d_max)

    evals_k = evals[:k]
    evecs_k = evecs[:, :k]
    X = evecs_k * torch.sqrt(torch.clamp(evals_k, min=0.0))

    return X.cpu().numpy().astype(np.float64), evals_k.cpu().numpy().astype(np.float64)

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
# New: USVT wrapper to pick best gamma and return W_hat + embedding
# -----------------------------
def usvt_best_denoise_and_embed(A_dense_t, gammas, cut=True, vmin=0.0, vmax=1.0,
                                energy_keep=None, d_max=None, verbose=False, save_each=None):
    """
    1) eigendecompose A
    2) USVT-threshold sweep over gammas; pick best by Frobenius reconstruction loss
    3) return best W_hat, its gamma, and latent embedding X

    save_each: if a folder path is provided, saves each candidate W_hat as .npy (optional).
    """
    n = A_dense_t.shape[0]
    rho = A_dense_t.mean().item() if n > 0 else 1.0
    rho = max(rho, 1e-12)
    s, v = torch.linalg.eigh(A_dense_t)

    best_loss, best_W, best_gamma = np.inf, None, None
    if verbose:
        print(f"[USVT] rho={rho:.6e}; n={n}")

    for gamma in tqdm(gammas, desc="USVT gamma sweep", leave=False):
        W_hat_t = USVT_threshold(s, v, n, gamma=gamma, rho=rho, cut=cut, vmin=vmin, vmax=vmax, verbose=False)
        loss = torch.linalg.matrix_norm(A_dense_t - W_hat_t, ord='fro').item()
        if loss < best_loss:
            best_loss = loss
            best_W = W_hat_t.clone()
            best_gamma = gamma
        if save_each is not None:
            np.save(os.path.join(save_each, f"W_gamma{gamma:.4f}.npy"), W_hat_t.cpu().numpy())

    if verbose:
        print(f"[USVT] best gamma={best_gamma:.4f}, best Fro loss={best_loss:.4f}")

    # latent positions from best W
    X, kept = gram_to_latent_positions(best_W, d_max=d_max, energy=energy_keep)
    return best_W.cpu().numpy(), X, best_gamma, kept

# -----------------------------
# Main: baseline 2010 vs 2011-2020
# -----------------------------
if __name__ == "__main__":
    # Reproducibility
    np.random.seed(0)
    torch.manual_seed(0)

    # Output dirs
    os.makedirs("usvt_C2_v2", exist_ok=True)

    # Load data
    dataset = PygNodePropPredDataset(name='ogbn-arxiv')
    data = dataset[0]

    # Helper to build dense symmetrized adjacency for a year
    def build_dense_adj_for_year(year):
        sub_data = get_subgraph_by_year(data, year)
        n = sub_data.num_nodes
        A_sparse = to_scipy_sparse_matrix(sub_data.edge_index, num_nodes=n).astype(np.float64)
        A_sparse = 0.5 * (A_sparse + A_sparse.T)   # symmetrize
        A_sparse.setdiag(0.0)
        A_dense = A_sparse.toarray()
        return torch.from_numpy(A_dense).to(dtype=torch.float64)

    # USVT sweep grid (tweak if you like)
    GAMMAS = np.linspace(0.01, 0.20, 20)

    # Choose embedding truncation:
    # - use energy_keep=0.9 to keep enough components to explain 90% of positive-eigenvalue mass
    # - or fix d_max, e.g., d_max=32
    ENERGY_KEEP = 0.90
    D_MAX = None  # set to an int (e.g., 32) if you want a fixed dimensionality cap

    # Build baseline (2010)
    baseline_year = 2010
    print(f"=== Baseline year {baseline_year} ===")
    A2010 = build_dense_adj_for_year(baseline_year)
    W2010, X2010, gamma2010, kept2010 = usvt_best_denoise_and_embed(
        A2010, GAMMAS, cut=True, vmin=0.0, vmax=1.0,
        energy_keep=ENERGY_KEEP, d_max=D_MAX, verbose=True,
        save_each=None  # e.g., "usvt_outputs/2010_candidates"
    )
    np.save("usvt_outputs/W_2010.npy", W2010)
    np.save("usvt_outputs/X_2010.npy", X2010)

    # Compare 2011..2020 to 2010 via Sliced W2
    results = {}
    for year in range(2011, 2021):
        print(f"\n=== Processing year {year} ===")
        A_year = build_dense_adj_for_year(year)
        save_dir = None  # e.g., f"usvt_outputs/{year}_candidates"; os.makedirs(save_dir, exist_ok=True)

        W_y, X_y, gamma_y, kept_y = usvt_best_denoise_and_embed(
            A_year, GAMMAS, cut=True, vmin=0.0, vmax=1.0,
            energy_keep=ENERGY_KEEP, d_max=D_MAX, verbose=True,
            save_each=save_dir
        )
        np.save(f"usvt_outputs/W_{year}.npy", W_y)
        np.save(f"usvt_outputs/X_{year}.npy", X_y)

        # Sliced W2 distance to baseline latent positions
        # dist = sliced_w2_distance(X2010, X_y, n_projections=256, n_quantiles=512, seed=42)
        dist = wasserstein_distance_between_embeddings(
            X2010, X_y,
            n_projections=256,   # you can lower to 128 for speed
            n_quantiles=512,     # only used by the NumPy fallback
            seed=42,
            use_pot=True         # tries POT first, then falls back
        )

        results[year] = {
            "sliced_W2_to_2010": float(dist),
            "best_gamma": float(gamma_y),
            "kept_eigs": kept_y.tolist()
        }
        print(f"[{year}] Sliced W2 to 2010: {dist:.6f} (best gamma={gamma_y:.4f}; kept dims={len(kept_y)})")

    # Save the summary table
    np.save("usvt_outputs/sliced_W2_results.npy", results)
    # Also write a human-readable text file
    with open("usvt_outputs/sliced_W2_results.txt", "w") as f:
        for y in range(2011, 2021):
            r = results[y]
            f.write(f"{y}\tW2={r['sliced_W2_to_2010']:.6f}\tgamma={r['best_gamma']:.4f}\tdims={len(r['kept_eigs'])}\n")

    print("\n=== Done. Summary (Sliced W2 to 2010) ===")
    for y in range(2011, 2021):
        print(f"{y}: {results[y]['sliced_W2_to_2010']:.6f}  (gamma={results[y]['best_gamma']:.4f}, dims={len(results[y]['kept_eigs'])})")
