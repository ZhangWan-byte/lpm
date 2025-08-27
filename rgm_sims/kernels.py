# File: rgm_sims/kernels.py

from __future__ import annotations
import numpy as np
from typing import Dict, Any, Tuple, Optional

SIGMOID = lambda x: 1.0 / (1.0 + np.exp(-x))


# ---------------------------
# Existing helpers / kernels
# ---------------------------

class RadialSpline:
    def __init__(self, values, rmax):
        self.values = np.array(values, dtype=float)
        self.rmax = float(rmax)
        self.K = len(values) - 1

    def __call__(self, r):
        # piecewise linear on [0, rmax] with K segments, tail decays to last value
        r = np.clip(r, 0.0, None)
        t = np.clip(r / self.rmax, 0.0, 1.0)
        idx = np.minimum((t * self.K).astype(int), self.K - 1)
        frac = t * self.K - idx
        v0 = self.values[idx]
        v1 = self.values[idx + 1]
        return v0 * (1 - frac) + v1 * frac


def kernel_block_constant(z, B):
    return B[z][:, z]


def kernel_radial_smooth(X, values, rmax):
    spline = RadialSpline(values, rmax)
    # distance matrix lazily; caller should avoid dense for large N
    D = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=-1)
    return spline(D)


def kernel_directed_bilinear(X_out, X_in, A_vals, B_vals, logit_bias):
    # A and B are diagonal given values
    A = np.diag(A_vals)
    B = np.diag(B_vals)
    S = X_out @ A
    T = X_in @ B
    return S @ T.T + float(logit_bias)


def kernel_indefinite_linear(X, S_vals, logit_bias):
    S = np.diag(S_vals)
    G = X @ S @ X.T
    return G + float(logit_bias)


# ---------------------------
# NEW: Translation-invariant kernels via Random Fourier Features (RFF)
# ---------------------------

def _rff_omegas_and_phases(
    d: int,
    m: int,
    lengthscales: np.ndarray,
    family: str = "rbf",
    seed: Optional[int] = None,
):
    """
    Draw spectral frequencies Ω and phases b for RFF.
    For 'rbf', ω ~ N(0, Λ) with Λ = diag(ell^{-2}).
    For other families, we fall back to Gaussian draws (reasonable default).
    Returns:
        Omega: [m, d], phases: [m]
    """
    rng = np.random.default_rng(None if seed is None else int(seed))
    family = (family or "rbf").lower()

    # ARD: lengthscales per dimension; allow scalar ell
    if np.isscalar(lengthscales):
        ell = np.full((d,), float(lengthscales))
    else:
        ell = np.array(lengthscales, dtype=float).reshape(-1)
        if ell.size != d:
            raise ValueError(f"lengthscales must have size {d}, got {ell.size}")

    inv_ell = 1.0 / np.maximum(ell, 1e-12)

    if family in ("rbf", "gaussian"):
        # ω_k ~ N(0, diag(inv_ell^2))
        eps = rng.normal(loc=0.0, scale=1.0, size=(m, d))
        Omega = eps * inv_ell.reshape(1, d)
    elif family in ("laplace", "exponential"):
        # Approximate Laplace kernel spectral density via independent Cauchy per dim
        # ω_k ~ Cauchy(0, inv_ell)
        # using tangent trick: Cauchy can be sampled via tan(pi*(U-0.5))
        U = rng.uniform(low=0.0, high=1.0, size=(m, d))
        cauchy = np.tan(np.pi * (U - 0.5))
        Omega = cauchy * inv_ell.reshape(1, d)
    elif family.startswith("matern"):
        # Simple Gaussian scale-mixture proxy: ω ~ N(0, diag(inv_ell^2) * s) with s ~ LogNormal(0, tau)
        # (This is an approximation; for many purposes a fixed Gaussian works well.)
        tau = 0.25
        s = rng.lognormal(mean=0.0, sigma=tau, size=(m, 1))
        eps = rng.normal(size=(m, d))
        Omega = (eps * np.sqrt(s)) * inv_ell.reshape(1, d)
    else:
        # Fallback to Gaussian draws
        eps = rng.normal(loc=0.0, scale=1.0, size=(m, d))
        Omega = eps * inv_ell.reshape(1, d)

    phases = rng.uniform(low=0.0, high=2 * np.pi, size=(m,))
    return Omega.astype(np.float64), phases.astype(np.float64)


def _rff_features(X: np.ndarray, Omega: np.ndarray, phases: np.ndarray) -> np.ndarray:
    """
    Compute φ(X) = sqrt(2/m) * cos(X Ω^T + b), where:
        X: [N, d], Omega: [m, d], phases b: [m]
    Returns:
        Phi: [N, m]
    """
    N, d = X.shape
    m, d2 = Omega.shape
    assert d == d2, "Omega shape incompatible with X"
    proj = X @ Omega.T  # [N, m]
    Phi = np.cos(proj + phases[None, :]) * np.sqrt(2.0 / float(m))
    return Phi


def kernel_translation_invariant_rff(
    X: np.ndarray,
    params: Dict[str, Any],
) -> np.ndarray:
    """
    Approximate a translation-invariant kernel k(x - y) using Random Fourier Features.

    Expected params:
        family: str, one of {"rbf","laplace","matern"} (default "rbf")
        lengthscales: list|float, ARD lengthscales per latent dim (or scalar)
        num_features: int, number of RFF features (default 512)
        amplitude: float, overall kernel scale (default 1.0)
        logit_bias: float, global logit bias (default -2.5)
        seed: int (optional), for reproducible Ω, phases

    Returns:
        logit matrix L ∈ R^{N×N} such that P = sigmoid(L)
    """
    family = params.get("family", "rbf")
    lengthscales = params.get("lengthscales", 1.0)
    num_features = int(params.get("num_features", 512))
    amplitude = float(params.get("amplitude", 1.0))
    logit_bias = float(params.get("logit_bias", -2.5))
    seed = params.get("seed", None)

    N, d = X.shape
    Omega, phases = _rff_omegas_and_phases(d=d, m=num_features, lengthscales=np.array(lengthscales), family=family, seed=seed)
    Phi = _rff_features(X, Omega, phases)  # [N, m]

    # Kernel approximation: K ≈ amplitude * Phi Phi^T
    # NOTE: This forms a dense N×N matrix; for very large N consider chunking.
    K = amplitude * (Phi @ Phi.T)

    # Return logits = bias + K
    return (logit_bias + K).astype(np.float64)


# ---------------------------
# Convenience dispatcher (optional)
# ---------------------------

def translation_invariant_logits(X: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
    """Alias helper for clarity."""
    return kernel_translation_invariant_rff(X, params)
