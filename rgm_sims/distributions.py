# File: rgm_sims/distributions.py

from __future__ import annotations
import numpy as np
from typing import Dict, Any, Tuple

RNG = np.random.Generator

def isotropic_gaussian(rng: RNG, N: int, d: int, mean: np.ndarray, sigma: float) -> np.ndarray:
    return rng.normal(loc=mean, scale=sigma, size=(N, d))

def gaussian_mixture(rng: RNG, N: int, d: int, K: int, weights, means, cov_scale: float):
    weights = np.array(weights)
    z = rng.choice(K, size=N, p=weights)
    X = np.zeros((N, d))
    M = np.zeros((K, d))
    if isinstance(means, str) and means == "random_sphere":
        M = rng.normal(size=(K, d))
        M = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-8)
        M *= 2.5
    else:
        M = np.array(means)
    for k in range(K):
        idx = np.where(z == k)[0]
        if len(idx) == 0:
            continue
        X[idx] = rng.normal(loc=M[k], scale=cov_scale, size=(len(idx), d))
    return X, z

def categorical_blocks(rng: RNG, N: int, K: int, pi):
    z = rng.choice(K, size=N, p=np.array(pi))
    return z

def product_gaussian(rng: RNG, N: int, d: int, mean_out: float, mean_in: float, sigma_out: float, sigma_in: float):
    x_out = rng.normal(mean_out, sigma_out, size=(N, d))
    x_in = rng.normal(mean_in, sigma_in, size=(N, d))
    return x_out, x_in