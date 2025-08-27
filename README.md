# File: README.md

## Random Graph Simulation Generator (RGM > Graphon)

This repo generates synthetic graphs for Experiment 1 under a minimal starting set:
A1 (SBM), A2 (Radial), B2 (Directed Bilinear), B3 (Heterophily/indefinite), B4 (Degree-corrected heavy tails), B5 (Flow-warped P).

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -U numpy scipy networkx pydantic typing_extensions
```

## Usage

```bash
python -m rgm_sims.cli --config configs/A1_SBM_K4_dense.json --outdir ./out/A1
```

Outputs edge list, node table (latent positions, block/attrs), and a metadata JSON.

---
