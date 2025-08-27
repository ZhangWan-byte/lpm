# File: rgm_sims/degree.py

from __future__ import annotations
import numpy as np
from typing import Tuple

SIGMOID = lambda x: 1.0 / (1.0 + np.exp(-x))

def draw_degree_factors(rng, N, mu, sigma):
    return rng.lognormal(mean=mu, sigma=sigma, size=N)

def apply_degree_correction(logit_P, s):
    # multiplicative on probability equals additive in logit
    # p_ij = sigmoid(logit_P + log s_i + log s_j)
    ls = np.log(np.maximum(s, 1e-12))
    return logit_P + ls[:, None] + ls[None, :]