# File: rgm_sims/cli.py

from __future__ import annotations
import argparse, json, os
from rgm_sims.config_schema import SimConfig
from rgm_sims.generator import generate_graph
from rgm_sims.io import save_graph, ensure_dir


def load_config(path: str) -> SimConfig:
    with open(path, "r") as f:
        data = json.load(f)
    return SimConfig(**data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to JSON config")
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    res = generate_graph(cfg)

    name = cfg.name
    outdir = os.path.join(args.outdir)
    extra = {}
    if res.get("blocks") is not None:
        extra["blocks"] = res["blocks"]
    if res.get("positions_out") is not None:
        extra["positions_out"] = res["positions_out"]
        extra["positions_in"] = res["positions_in"]
    save_graph(outdir, name, cfg.graph.N, res["edges"], cfg.graph.directed, res["positions"], extra)
    print(f"Saved to {outdir}")

if __name__ == "__main__":
    main()