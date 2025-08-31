# File: rgm_sims/generator.py

from __future__ import annotations
import numpy as np
from typing import Tuple, Dict, Any, Optional
from rgm_sims.config_schema import SimConfig
from rgm_sims import distributions as dist
from rgm_sims import kernels as K
from rgm_sims import degree as DC

SIGMOID = lambda x: 1.0 / (1.0 + np.exp(-x))

def _edge_sample_from_probs(rng, P: np.ndarray, directed: bool, self_loops: bool) -> np.ndarray:
    N = P.shape[0]
    if not directed:
        iu = np.triu_indices(N, k=0 if self_loops else 1)
        probs = P[iu]
        draws = rng.random(size=probs.shape[0]) < probs
        E = np.vstack((iu[0][draws], iu[1][draws])).T
        if not self_loops:
            E = E[E[:,0] != E[:,1]]
        return E.astype(int)
    else:
        idx = np.where(np.ones_like(P, dtype=bool))
        if not self_loops:
            mask = idx[0] != idx[1]
            idx = (idx[0][mask], idx[1][mask])
        probs = P[idx]
        draws = rng.random(size=probs.shape[0]) < probs
        E = np.vstack((idx[0][draws], idx[1][draws])).T
        return E.astype(int)


# === REPLACE the _emit_node_features function with this ===
def _emit_node_features(rng, Z: np.ndarray, blocks, node_feat_cfg) -> np.ndarray:
    """
    Map latent positions Z -> observed node features X with a target dim x_dim.
    Uses a simple mix: [Z, Z^2, RFF(Z)] (+ optional block one-hot), then projects to x_dim.

    node_feat_cfg.params:
      - x_dim: int (default 40)
      - poly: bool (default True)          # include [Z, Z^2]
      - rff_features: int (default max(2*d, x_dim))
      - rff_lengthscale: float (default 1.0)
      - include_blocks_onehot: bool (default False)
      - standardize: bool (default True)
      - seed: int (optional; uses rng if absent)
    """
    if not getattr(node_feat_cfg, "enabled", False):
        return None
    if Z is None:
        return None

    N, d = Z.shape
    params = getattr(node_feat_cfg, "params", {}) or {}
    x_dim = int(params.get("x_dim", 40))
    poly = bool(params.get("poly", True))
    m_rff = int(params.get("rff_features", max(2 * d, x_dim)))
    ell = float(params.get("rff_lengthscale", 1.0))
    incl_blocks = bool(params.get("include_blocks_onehot", False))
    standardize = bool(params.get("standardize", True))

    # local RNG (optional seed) derived from top-level rng for reproducibility
    seed = params.get("seed", None)
    local_rng = np.random.default_rng(rng.integers(0, 2**31 - 1) if seed is None else int(seed))

    feats = []
    if poly:
        feats.append(Z)
        feats.append(Z**2)
        feats.append(np.exp(Z))
    # print(len(feats))

    if m_rff > 0:
        # Random Fourier Features (RFF) for a stationary bump in feature space
        W = local_rng.normal(loc=0.0, scale=1.0 / max(ell, 1e-6), size=(d, m_rff))
        b = local_rng.uniform(low=0.0, high=2 * np.pi, size=(m_rff,))
        Phi = np.sqrt(2.0 / float(m_rff)) * np.cos(Z @ W + b)  # [N, m_rff]
        feats.append(Phi)

    if incl_blocks and blocks is not None:
        K = int(np.max(blocks)) + 1
        onehot = np.eye(K, dtype=np.float32)[blocks]
        feats.append(onehot)

    F = np.concatenate(feats, axis=1).astype(np.float32)  # [N, D_total]

    # Project (or pad) to target x_dim
    D_total = F.shape[1]
    if D_total > x_dim:
        P = local_rng.normal(loc=0.0, scale=1.0 / np.sqrt(D_total), size=(D_total, x_dim)).astype(np.float32)
        X = F @ P
    elif D_total < x_dim:
        pad = np.zeros((N, x_dim - D_total), dtype=np.float32)
        X = np.hstack([F, pad])
    else:
        X = F
    # print(X.shape)
    # print(X)
    if standardize:
        mu = X.mean(axis=0, keepdims=True)
        sd = X.std(axis=0, keepdims=True) + 1e-6
        X = (X - mu) / sd

    return X.astype(np.float32)


def generate_graph(cfg: SimConfig) -> Dict[str, Any]:
    rng = np.random.default_rng(cfg.seed)
    N = cfg.graph.N
    d = cfg.latent_space.dimension

    # Latent positions
    z_blocks = None
    X = None
    X_out = X_in = None

    bm = cfg.latent_space.base_measure
    if bm.type == "isotropic_gaussian":
        mean = np.array(bm.params.get("mean", [0.0]*d))
        sigma = float(bm.params.get("sigma", 1.0))
        X = dist.isotropic_gaussian(rng, N, d, mean, sigma)
    elif bm.type == "gaussian_mixture":
        Kc = int(bm.params.get("K", 3))
        weights = bm.params.get("weights", [1.0/Kc]*Kc)
        means = bm.params.get("means", "random_sphere")
        cov_scale = float(bm.params.get("cov_scale", 0.5))
        X, z_blocks = dist.gaussian_mixture(rng, N, d, Kc, weights, means, cov_scale)
    elif bm.type == "categorical_blocks":
        Kc = int(bm.params.get("K", 4))
        pi = bm.params.get("pi", [1.0/Kc]*Kc)
        z_blocks = dist.categorical_blocks(rng, N, Kc, pi)
        X = np.eye(Kc)[z_blocks][:, :d] if d <= Kc else np.pad(np.eye(Kc)[z_blocks], ((0,0),(0,d-Kc)))
    elif bm.type == "product_gaussian":
        X_out, X_in = dist.product_gaussian(
            rng, N, d,
            bm.params.get("mean_out", 0.0), bm.params.get("mean_in", 0.0),
            bm.params.get("sigma_out", 1.0), bm.params.get("sigma_in", 1.0)
        )
    elif bm.type == "flow_warped":
        proto = bm.params.get("prototype", "isotropic_gaussian")
        sigma = float(bm.params.get("sigma", 1.0))
        X = dist.isotropic_gaussian(rng, N, d, np.zeros(d), sigma)
        if d < 2:
            raise ValueError("flow_warped requires d>=2")
        strength = float(bm.params.get("warp_strength", 0.8))
        x, y = X[:,0], X[:,1]
        y = y + strength * (x**2 - 1.0)
        X[:,0], X[:,1] = x, y
    else:
        raise NotImplementedError(f"Base measure {bm.type}")

    # Kernel / logits
    directed = cfg.graph.directed
    if cfg.kernel.type == "block_constant":
        B = np.array(cfg.kernel.params["B"], dtype=float)
        if z_blocks is None:
            raise ValueError("block_constant kernel requires block assignments")
        P = K.kernel_block_constant(z_blocks, B)
        logit = np.log(P + 1e-9) - np.log(1 - P + 1e-9)
    elif cfg.kernel.type == "radial_smooth":
        values = cfg.kernel.params.get("values", [0.6, 0.2, 0.05, 0.01])
        rmax = float(cfg.kernel.params.get("range", 3.0))
        P = K.kernel_radial_smooth(X, values, rmax)
        logit = np.log(P + 1e-9) - np.log(1 - P + 1e-9)
    elif cfg.kernel.type == "directed_bilinear":
        if X_out is None or X_in is None:
            raise ValueError("directed_bilinear requires product_gaussian base measure")
        A_vals = np.array(cfg.kernel.params.get("A_values"))
        B_vals = np.array(cfg.kernel.params.get("B_values"))
        logit = K.kernel_directed_bilinear(X_out, X_in, A_vals, B_vals, cfg.kernel.params.get("logit_bias", -2.5))
        P = SIGMOID(logit)
        directed = True
    elif cfg.kernel.type == "heterophily_indefinite":
        S_vals = np.array(cfg.kernel.params.get("S_values"))
        logit = K.kernel_indefinite_linear(X, S_vals, cfg.kernel.params.get("logit_bias", -2.2))
        P = SIGMOID(logit)
    elif cfg.kernel.type == "translation_invariant_rff_mixture":
        # Uses latent positions X (undirected, translation-invariant)
        if X is None:
            raise ValueError("translation_invariant_rff_mixture requires point latents X (not product_gaussian).")
        logit = K.kernel_translation_invariant_rff_mixture(X, cfg.kernel.params)
        P = SIGMOID(logit)
    else:
        raise NotImplementedError(f"Kernel {cfg.kernel.type}")

    # Degree correction
    if cfg.degree_correction.enabled:
        mu = float(cfg.degree_correction.params.get("mu", 0.0))
        sigma = float(cfg.degree_correction.params.get("sigma", 1.0))
        s = DC.draw_degree_factors(rng, N, mu, sigma)
        logit = DC.apply_degree_correction(logit, s)
        P = SIGMOID(logit)

    # Scale to target expected degree by global bias tweak
    target_m = cfg.graph.expected_degree * N / (2 if not directed else 1)
    mean_p = P.mean() if directed else P[np.triu_indices(N, 1)].mean()
    if mean_p > 0:
        scale = target_m / (mean_p * (N*(N-1)/2 if not directed else N*(N-1)))
        # adjust by adding scalar to logits
        adj = np.log(scale + 1e-9)
        logit = logit + adj
        P = SIGMOID(logit)

    # Sample edges
    E = _edge_sample_from_probs(rng, P, directed=directed, self_loops=cfg.graph.self_loops)

    # ----- NEW: node features from latent positions -----
    # choose Z for features: use X if present, else concatenate out/in if product-Gaussian
    Z_feat = None
    if X is not None:
        Z_feat = X
    elif (X_out is not None) and (X_in is not None):
        Z_feat = np.concatenate([X_out, X_in], axis=1)

    node_feat_cfg = cfg.attributes.node_features if hasattr(cfg.attributes, "node_features") else None
    node_features = None
    if node_feat_cfg is not None:
        node_features = _emit_node_features(rng, Z_feat, z_blocks, node_feat_cfg)
    # -----------------------------------------------------

    return {
        "edges": E,
        "positions": X if X is not None else np.zeros((N, d)),
        "positions_out": X_out,
        "positions_in": X_in,
        "blocks": z_blocks,
        "prob_matrix_summary": dict(mean=float(P.mean()), min=float(P.min()), max=float(P.max())),
        "node_features": node_features,   # <-- include for saving
    }
