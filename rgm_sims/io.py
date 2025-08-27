# File: rgm_sims/io.py

from __future__ import annotations
import json, os
import numpy as np
from typing import Dict, Any

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def save_graph(outdir: str, name: str, N: int, edges: np.ndarray, directed: bool, positions: np.ndarray, extra: Dict[str, Any]):
    ensure_dir(outdir)
    np.savetxt(os.path.join(outdir, f"{name}_edges.txt"), edges.astype(int), fmt="%d")
    np.savez(os.path.join(outdir, f"{name}_nodes.npz"), positions=positions, **extra)
    meta = {"N": N, "M": int(edges.shape[0]), "directed": directed}
    meta.update({k: v for k, v in extra.items() if isinstance(v, (int, float, str, bool))})
    with open(os.path.join(outdir, f"{name}_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)