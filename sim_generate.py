# File: generate.py

import argparse
import json
import os
import sys
from typing import Any

# Ensure the repo root is on sys.path so "import rgm_sims" works when running this file directly.
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from rgm_sims.config_schema import SimConfig  # type: ignore
from rgm_sims.generator import generate_graph  # type: ignore
from rgm_sims.io import save_graph  # type: ignore


def parse_args():
    ap = argparse.ArgumentParser(description="Generate a random graph from a JSON config using rgm_sims.")
    ap.add_argument("--config", required=True, help="Path to JSON config (see configs/ examples)")
    ap.add_argument("--outdir", required=True, help="Output directory")
    ap.add_argument("--name", default=None, help="Override output base name (defaults to config name)")
    return ap.parse_args()


def load_config(path: str) -> SimConfig:
    with open(path, "r") as f:
        data: Any = json.load(f)
    return SimConfig(**data)


def main():
    args = parse_args()
    cfg = load_config(args.config)
    if args.name:
        cfg.name = args.name

    res = generate_graph(cfg)

    extra = {}
    if res.get("blocks") is not None:
        extra["blocks"] = res["blocks"]
    if res.get("positions_out") is not None:
        extra["positions_out"] = res["positions_out"]
        extra["positions_in"] = res["positions_in"]

    save_graph(args.outdir, cfg.name, cfg.graph.N, res["edges"], cfg.graph.directed, res["positions"], extra)

    print(f"Saved graph '{cfg.name}' to {os.path.abspath(args.outdir)}")

# python generate.py --config configs\B1.json --outdir .\out\B1
if __name__ == "__main__":
    main()
