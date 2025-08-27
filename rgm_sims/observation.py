# File: rgm_sims/observation.py

from __future__ import annotations
import numpy as np
import networkx as nx
from typing import Tuple, Optional, Iterable

def drop_missing_edges(edges: np.ndarray, rate: float, rng) -> np.ndarray:
    if rate <= 0: return edges
    m = edges.shape[0]
    keep = rng.random(m) > rate
    return edges[keep]

def ego_net_induced(N: int, edges: np.ndarray, k_nodes: Optional[int], radius_hops: int, rng) -> Tuple[np.ndarray, np.ndarray]:
    if not k_nodes or k_nodes >= N:
        return np.arange(N), edges
    G = nx.Graph()
    G.add_nodes_from(range(N))
    G.add_edges_from(map(tuple, edges))
    seeds = rng.choice(N, size=min(k_nodes, N), replace=False)
    keep = set()
    for s in seeds:
        nbrs = nx.single_source_shortest_path_length(G, s, cutoff=radius_hops).keys()
        keep.update(nbrs)
    keep = np.array(sorted(list(keep)))
    mask = np.isin(edges[:,0], keep) & np.isin(edges[:,1], keep)
    sub_edges = edges[mask]
    # reindex nodes to 0..k-1
    remap = {int(u):i for i,u in enumerate(keep)}
    sub_edges = np.vectorize(lambda x: remap[int(x)])(sub_edges)
    return keep, sub_edges

