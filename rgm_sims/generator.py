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


def _emit_node_features(rng, positions: np.ndarray, attr_cfg) -> Optional[np.ndarray]:
    """
    Minimal emission for node features to support B1 configs.
    If attributes.node_features.enabled and emission=='gaussian' with map=='nonlinear_random_fourier',
    we synthesize x_i = [positions, positions^2, sin(Wz), cos(Wz)] + noise.
    """
    if not getattr(attr_cfg, "node_features", None):
        return None
    nf = attr_cfg.node_features
    if not nf.enabled:
        return None
    if nf.emission != "gaussian":
        return None
    map_name = nf.params.get("map", "")
    if map_name != "nonlinear_random_fourier":
        return None

    X = np.asarray(positions, dtype=np.float32)  # [N, d]
    N, d = X.shape
    m = int(nf.params.get("num_features", max(8, 4 * d)))
    noise_sigma = float(nf.params.get("noise_sigma", 0.1))

    # Random projection
    W = rng.normal(0.0, 1.0, size=(d, m)).astype(np.float32)
    P = X @ W  # [N, m]
    feats = np.concatenate([X, X**2, np.sin(P), np.cos(P)], axis=1).astype(np.float32)
    feats += rng.normal(0.0, noise_sigma, size=feats.shape).astype(np.float32)
    # Standardize (zero mean / unit var)
    mu = feats.mean(axis=0, keepdims=True)
    sd = feats.std(axis=0, keepdims=True) + 1e-6
    feats = (feats - mu) / sd
    return feats


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
    elif cfg.kernel.type == "translation_invariant":
        # NEW: translation-invariant via RFF
        logit = K.translation_invariant_logits(X, cfg.kernel.params)
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

    # OPTIONAL: synthesize node features if requested by config
    node_feats = _emit_node_features(rng, X if X is not None else np.zeros((N, d)), cfg.attributes)

    extra = {}
    if node_feats is not None:
        extra["node_features"] = node_feats
    if X_out is not None:
        extra["positions_out"] = X_out
        extra["positions_in"] = X_in
    if z_blocks is not None:
        extra["blocks"] = z_blocks

    return {
        "edges": E,
        "positions": X if X is not None else np.zeros((N, d)),
        "positions_out": X_out,
        "positions_in": X_in,
        "blocks": z_blocks,
        "prob_matrix_summary": dict(mean=float(P.mean()), min=float(P.min()), max=float(P.max())),
        **extra,
    }
