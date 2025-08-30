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


# ====== RFF utilities and TI-mixture kernel (append to kernels.py) ======
from typing import Dict, Any, Optional, List
import numpy as np

def _rff_omegas_and_phases(
    d: int,
    m: int,
    lengthscales,
    family: str = "rbf",
    seed: Optional[int] = None,
):
    """
    Draw spectral frequencies Ω and phases b for Random Fourier Features.
    For stationary kernels k(h), Bochner's theorem => ω sampled from spectral density.
    - family='rbf': ω ~ N(0, diag(ell^{-2}))  (ARD allowed)
    - family='laplace'|'exponential': ω ~ Cauchy(0, ell^{-1}) independently (approx.)
    - family='matern': simple lognormal scale mixture proxy around Gaussian
    Returns:
        Omega: [m, d], phases: [m]
    """
    rng = np.random.default_rng(None if seed is None else int(seed))

    ell = np.array(lengthscales, dtype=float).reshape(-1) if np.ndim(lengthscales) > 0 else np.full((d,), float(lengthscales))
    if ell.size != d:
        raise ValueError(f"lengthscales must have size {d}, got {ell.size}")
    inv_ell = 1.0 / np.maximum(ell, 1e-12)

    fam = (family or "rbf").lower()
    if fam in ("rbf", "gaussian"):
        eps = rng.normal(loc=0.0, scale=1.0, size=(m, d))
        Omega = eps * inv_ell.reshape(1, d)
    elif fam in ("laplace", "exponential"):
        U = rng.uniform(low=0.0, high=1.0, size=(m, d))
        cauchy = np.tan(np.pi * (U - 0.5))
        Omega = cauchy * inv_ell.reshape(1, d)
    elif fam.startswith("matern"):
        tau = 0.25  # small dispersion for scale mixture
        s = rng.lognormal(mean=0.0, sigma=tau, size=(m, 1))
        eps = rng.normal(size=(m, d))
        Omega = (eps * np.sqrt(s)) * inv_ell.reshape(1, d)
    else:
        # fallback: Gaussian draws (works reasonably in practice)
        eps = rng.normal(loc=0.0, scale=1.0, size=(m, d))
        Omega = eps * inv_ell.reshape(1, d)

    phases = rng.uniform(low=0.0, high=2 * np.pi, size=(m,))
    return Omega.astype(np.float64), phases.astype(np.float64)


def _rff_features(X: np.ndarray, Omega: np.ndarray, phases: np.ndarray) -> np.ndarray:
    """
    φ(X) = sqrt(2/m) * cos(X Ω^T + b)
    X: [N, d], Omega: [m, d], phases b: [m]  ->  Φ: [N, m]
    """
    proj = X @ Omega.T  # [N, m]
    Phi = np.cos(proj + phases[None, :]) * np.sqrt(2.0 / float(Omega.shape[0]))
    return Phi


def kernel_translation_invariant_rff_mixture(
    X: np.ndarray,
    params: Dict[str, Any],
) -> np.ndarray:
    """
    Translation-invariant kernel mixture via Random Fourier Features.
    Builds a kernel K ≈ Σ_c w_c * Φ_c Φ_c^T  and returns logits = logit_bias + amplitude * K.

    Expected params:
      components: List[{
        "weight": float,
        "family": "rbf"|"laplace"|"matern"|...,
        "lengthscales": float | List[float] (ARD),
        "num_features": int,
        "seed": int (optional)
      }, ...]
      amplitude: float (default 1.0)
      logit_bias: float (default -2.5)

    Returns:
      logits L ∈ R^{N×N}; use sigmoid(L) for probabilities.
    """
    comps: List[Dict[str, Any]] = params.get("components", [])
    amplitude = float(params.get("amplitude", 1.0))
    logit_bias = float(params.get("logit_bias", -2.5))

    if not comps:
        raise ValueError("translation_invariant_rff_mixture: 'components' list is empty")

    N, d = X.shape
    K = np.zeros((N, N), dtype=np.float64)

    for c in comps:
        w = float(c.get("weight", 1.0))
        fam = c.get("family", "rbf")
        ell = c.get("lengthscales", 1.0)
        m = int(c.get("num_features", 256))
        sd = c.get("seed", None)

        Omega, phases = _rff_omegas_and_phases(d=d, m=m, lengthscales=ell, family=fam, seed=sd)
        Phi = _rff_features(X, Omega, phases)  # [N, m]
        K += w * (Phi @ Phi.T)

    logits = logit_bias + amplitude * K
    return logits
# ====== end: RFF TI-mixture kernel ======

