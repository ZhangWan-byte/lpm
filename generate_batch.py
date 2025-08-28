# File: sim_batch_generate.py
"""
Generate multiple graphs for A1 / A2 / B1 into sim_data_batch/<SETTING>/<RUN_NAME>.

Re-uses rgm_sims to keep things simple & fast:
- A1: block-constant (SBM-like), 4 blocks
- A2: radial-smooth kernel, isotropic Gaussian latent
- B1: stationary-ish kernel (radial-smooth here as a practical stand-in), isotropic Gaussian latent

Each graph's expected degree is matched via the generator's internal scaling.
"""

import os
import argparse
import json
from typing import Dict, Any, List
import numpy as np

# Make sure local rgm_sims is importable
import sys
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from rgm_sims.config_schema import SimConfig
from rgm_sims.generator import generate_graph
from rgm_sims.io import save_graph, ensure_dir


def make_cfg_A1(name: str, N: int, seed: int) -> Dict[str, Any]:
    """SBM-like, 4 blocks."""
    return {
        "name": name,
        "seed": int(seed),
        "graph": {"N": int(N), "expected_degree": 20, "directed": False, "self_loops": False},
        "latent_space": {
            "dimension": 2,
            "base_measure": {"type": "categorical_blocks", "params": {"K": 4, "pi": [0.25, 0.25, 0.25, 0.25]}},
        },
        "kernel": {
            "type": "block_constant",
            "params": {
                "B": [
                    [0.06, 0.01, 0.01, 0.01],
                    [0.01, 0.06, 0.01, 0.01],
                    [0.01, 0.01, 0.06, 0.01],
                    [0.01, 0.01, 0.01, 0.06],
                ]
            },
        },
        "degree_correction": {"enabled": False, "distribution": "lognormal", "params": {"mu": 0.0, "sigma": 0.0}},
        "attributes": {"node_features": {"enabled": False, "emission": "none", "params": {}}, "edge_features": {"enabled": False, "emission": "none", "params": {}}},
        "observation": {"missing_edges": {"type": "MAR", "rate": 0.10}},
        "evaluation": {"holdout_edge_fraction": 0.10, "negative_sampling_ratio": 5},
        "temporal": {"enabled": False, "steps": 0, "deformation": {"bandwidth_scale_per_step": 1.0}},
    }


def make_cfg_A2(name: str, N: int, seed: int) -> Dict[str, Any]:
    """Radial-smooth kernel, Gaussian latent."""
    return {
        "name": name,
        "seed": int(seed),
        "graph": {"N": int(N), "expected_degree": 12, "directed": False, "self_loops": False},
        "latent_space": {
            "dimension": 2,
            "base_measure": {"type": "isotropic_gaussian", "params": {"mean": [0.0, 0.0], "sigma": 1.0}},
        },
        "kernel": {
            "type": "radial_smooth",
            "params": {
                "range": 3.0,
                # high → low on increasing distance
                "values": [0.55, 0.15, 0.03, 0.005],
            },
        },
        "degree_correction": {"enabled": False, "distribution": "lognormal", "params": {"mu": 0.0, "sigma": 0.0}},
        "attributes": {"node_features": {"enabled": False, "emission": "none", "params": {}}, "edge_features": {"enabled": False, "emission": "none", "params": {}}},
        "observation": {"missing_edges": {"type": "MAR", "rate": 0.10}},
        "evaluation": {"holdout_edge_fraction": 0.10, "negative_sampling_ratio": 10},
        "temporal": {"enabled": False, "steps": 0, "deformation": {"bandwidth_scale_per_step": 1.0}},
    }


def make_cfg_B1(name: str, N: int, seed: int) -> Dict[str, Any]:
    """
    Stationary/translation-invariant-esque.
    We approximate with radial_smooth here for simplicity & speed (decoder can still be RFF).
    """
    return {
        "name": name,
        "seed": int(seed),
        "graph": {"N": int(N), "expected_degree": 15, "directed": False, "self_loops": False},
        "latent_space": {
            "dimension": 2,
            "base_measure": {"type": "isotropic_gaussian", "params": {"mean": [0.0, 0.0], "sigma": 1.0}},
        },
        "kernel": {
            "type": "radial_smooth",
            "params": {
                "range": 3.0,
                "values": [0.6, 0.20, 0.04, 0.01],
            },
        },
        "degree_correction": {"enabled": False, "distribution": "lognormal", "params": {"mu": 0.0, "sigma": 0.0}},
        "attributes": {"node_features": {"enabled": False, "emission": "none", "params": {}}, "edge_features": {"enabled": False, "emission": "none", "params": {}}},
        "observation": {"missing_edges": {"type": "MAR", "rate": 0.10}},
        "evaluation": {"holdout_edge_fraction": 0.10, "negative_sampling_ratio": 10},
        "temporal": {"enabled": False, "steps": 0, "deformation": {"bandwidth_scale_per_step": 1.0}},
    }


def synthesize_node_features(out_npz_path: str, x_dim: int = 16, seed: int = 7):
    """
    Append a simple feature matrix x = [z, z^2, sin(Wz), cos(Wz)] to *_nodes.npz.
    Keeps code self-contained; features are standardized later during training.
    """
    import numpy as np
    data = np.load(out_npz_path)
    Z = data["positions"].astype(np.float32)  # [N, 2]
    N, d = Z.shape
    rng = np.random.default_rng(seed)
    m = min(4 * d, 16)
    W = rng.normal(0.0, 1.0, size=(d, m)).astype(np.float32)
    P = Z @ W
    X = np.concatenate([Z, Z**2, np.sin(P), np.cos(P)], axis=1).astype(np.float32)
    # truncate/expand to x_dim
    if X.shape[1] >= x_dim:
        X = X[:, :x_dim]
    else:
        pad = np.zeros((N, x_dim - X.shape[1]), dtype=np.float32)
        X = np.hstack([X, pad])
    # Save back
    np.savez(out_npz_path, **{k: data[k] for k in data.files}, node_features=X)


def main():
    ap = argparse.ArgumentParser(description="Batch simulate A1/A2/B1 graphs into sim_data_batch/<SETTING>")
    ap.add_argument("--setting", required=True, choices=["A1", "A2", "B1"])
    ap.add_argument("--out_root", default="sim_data_batch")
    ap.add_argument("--sizes", default="2000,5000,10000", help="Comma-separated N per graph")
    ap.add_argument("--per_size", type=int, default=10, help="#graphs per size for TRAIN")
    ap.add_argument("--val_per_size", type=int, default=2, help="#graphs per size for VAL")
    ap.add_argument("--test_per_size", type=int, default=2, help="#graphs per size for TEST")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--x_dim", type=int, default=16, help="Observed node feature dimension to synthesize")
    args = ap.parse_args()

    sizes: List[int] = [int(s) for s in args.sizes.split(",")]
    setting_dir = os.path.join(args.out_root, args.setting)
    splits = [("train", args.per_size), ("val", args.val_per_size), ("test", args.test_per_size)]

    rng = np.random.default_rng(args.seed)
    ensure_dir(setting_dir)

    cfg_fn = {"A1": make_cfg_A1, "A2": make_cfg_A2, "B1": make_cfg_B1}[args.setting]

    total = 0
    for split_name, k in splits:
        for N in sizes:
            for r in range(k):
                seed = int(rng.integers(0, 10_000_000))
                run_name = f"{args.setting}_N{N}_{split_name}_{r:02d}"
                outdir = os.path.join(setting_dir, run_name)
                ensure_dir(outdir)

                cfg_dict = cfg_fn(run_name, N, seed)
                cfg = SimConfig(**cfg_dict)
                res = generate_graph(cfg)

                extra = {}
                if res.get("blocks") is not None:
                    extra["blocks"] = res["blocks"]
                if res.get("positions_out") is not None:
                    extra["positions_out"] = res["positions_out"]
                    extra["positions_in"] = res["positions_in"]

                save_graph(outdir, cfg.name, cfg.graph.N, res["edges"], cfg.graph.directed, res["positions"], extra)

                # add synthetic node features to *_nodes.npz
                nodes_npz = os.path.join(outdir, f"{cfg.name}_nodes.npz")
                synthesize_node_features(nodes_npz, x_dim=args.x_dim, seed=seed)

                total += 1
                print(f"[{args.setting}] saved {run_name} → {outdir}")

    print(f"Done. Generated {total} graphs under {setting_dir}")


if __name__ == "__main__":
    main()
