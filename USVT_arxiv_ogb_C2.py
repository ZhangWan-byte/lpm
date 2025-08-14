import numpy as np
import torch
from torch_geometric.utils import to_scipy_sparse_matrix, subgraph
from ogb.nodeproppred import PygNodePropPredDataset
from tqdm import tqdm

def cutoff(A, vmin=0.0, vmax=1.0):
    # A is a torch.Tensor
    B = A.clone()
    B[B > vmax] = vmax
    B[B < vmin] = vmin
    return B

def USVT(A_dense_t, gamma=0.5, rho=None, cut=True, vmin=0.0, vmax=1.0, verbose=False):
    """
    A_dense_t: torch.Tensor, shape (n, n), symmetric
    rho: edge density. If None, estimate as A.mean().item()
    """
    n = A_dense_t.shape[0]
    if rho is None:
        rho = A_dense_t.mean().item() if n > 0 else 1.0
        rho = max(rho, 1e-12)  # avoid division by zero

    # eigendecomposition (A is symmetric)
    # returns eigenvalues in ascending order and eigenvectors in columns
    s, v = torch.linalg.eigh(A_dense_t)

    thr = gamma * torch.sqrt(torch.tensor(rho * n, dtype=s.dtype, device=s.device))
    suppressed = (s < thr).sum().item()
    if verbose:
        print(f"{suppressed / n:.3f} of eigenvalues suppressed")

    s = torch.where(s < thr, torch.zeros_like(s), s)

    # Reconstruct: V diag(s) V^T, then de-bias by rho
    Ahat = (v * s) @ v.T
    Ahat = Ahat / rho

    if cut:
        Ahat = cutoff(Ahat, vmin, vmax)
    return Ahat


def USVT_threshold(s, v, n, gamma=0.5, rho=None, cut=True, vmin=0.0, vmax=1.0, verbose=False):
    thr = gamma * torch.sqrt(torch.tensor(rho * n, dtype=s.dtype, device=s.device))
    suppressed = (s < thr).sum().item()
    if verbose:
        print(f"{suppressed / n:.3f} of eigenvalues suppressed")

    s = torch.where(s < thr, torch.zeros_like(s), s)

    # Reconstruct: V diag(s) V^T, then de-bias by rho
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

    # Relabel nodes in the edge_index to range 0..len(selected_nodes)-1
    sub_edge_index, _ = subgraph(selected_nodes, data.edge_index, relabel_nodes=True)

    # Build subgraph data object
    sub_data = data.clone()
    sub_data.edge_index = sub_edge_index
    sub_data.x = data.x[selected_nodes]
    sub_data.y = data.y[selected_nodes]
    sub_data.node_year = data.node_year[selected_nodes]
    sub_data.num_nodes = selected_nodes.size(0)  # <-- Fix: correct graph size

    return sub_data


if __name__ == "__main__":
    # --- Data prep
    dataset = PygNodePropPredDataset(name='ogbn-arxiv')
    data = dataset[0]

    for year in range(2016, 2021):

        sub_data = get_subgraph_by_year(data, year)
        n = sub_data.num_nodes

        # sparse adjacency (float64)
        A_sparse = to_scipy_sparse_matrix(sub_data.edge_index, num_nodes=n).astype(np.float64)

        # symmetrize for USVT (since ogbn-arxiv is directed)
        A_sparse = 0.5 * (A_sparse + A_sparse.T)
        A_sparse.setdiag(0.0)  # optional: clear self-loops

        # dense torch tensor (float64 for numerical stability)
        A_dense_np = A_sparse.toarray()
        A_dense_t = torch.from_numpy(A_dense_np).to(dtype=torch.float64)

        # USVT - phase 1 svd
        print("USVT - phase 1 svd at year {}".format(year))
        n = A_dense_t.shape[0]
        rho = A_dense_t.mean().item() if n > 0 else 1.0
        rho = max(rho, 1e-12)  # avoid division by zero
        print(rho)

        # eigendecomposition (A is symmetric)
        # returns eigenvalues in ascending order and eigenvectors in columns
        s, v = torch.linalg.eigh(A_dense_t)

        # USVT - phase 2 thresholding
        best_loss, best_W = np.inf, None
        log = []
        for gamma in tqdm(np.linspace(0.01, 0.2, 20), desc=f"USVT gamma sweep at year {year}"):
            W_hat_t = USVT_threshold(s, v, n, gamma=gamma, rho=rho, cut=True, vmin=0.0, vmax=1.0, verbose=True)

            # Compute loss
            loss = torch.norm(A_dense_t - W_hat_t, p='fro').item()
            if loss < best_loss:
                best_loss = loss
                best_W = W_hat_t

            log.append((gamma, loss))
            np.save(f"usvt_C2_v1/year{year}_gamma{gamma}.npy", W_hat_t.numpy())

        print("best gamma:", log[np.argmin([l[1] for l in log])][0])
        best_W_np = best_W.numpy()
        np.save(f"usvt_C2/year{year}_best_W.npy", best_W_np)

