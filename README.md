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


## Run exp1

A1 (linux)

```bash
python exp1_train_batch.py --setting_dir sim_data_batch/A1 --models_dir models/checkpoints --epochs 80 --latent_dim 16 --hidden 64 --decoder dot --decoder_kwargs '{}' --neg_ratio 5 --lambda_feat 1.0 --lambda_kl 0.005 --kl_warmup_epochs 50 --val_auc_neg_ratio 1
```

A1 (windows)

```bash
python exp1_train_batch.py --setting_dir sim_data_batch\A1 --models_dir models\checkpoints --epochs 80 --latent_dim 16 --hidden 64 --decoder dot --decoder_kwargs "{}" --neg_ratio 5 --lambda_feat 1.0 --lambda_kl 0.005 --kl_warmup_epochs 50 --val_auc_neg_ratio 1
```

A2 (linux)

```bash
python exp1_train_batch.py --setting_dir sim_data_batch/A2 --models_dir models/checkpoints --epochs 100 --latent_dim 16 --hidden 128 --decoder radial --decoder_kwargs '{}' --neg_ratio 10 --lambda_feat 1.0 --lambda_kl 0.005 --kl_warmup_epochs 80 --val_auc_neg_ratio 1
```

A2 (windows)

```bash
python exp1_train_batch.py --setting_dir sim_data_batch\A2 --models_dir models\checkpoints --epochs 100 --latent_dim 16 --hidden 128 --decoder radial --decoder_kwargs "{}" --neg_ratio 10 --lambda_feat 1.0 --lambda_kl 0.005 --kl_warmup_epochs 80 --val_auc_neg_ratio 1
```

B1 (linux)

```bash
python exp1_train_batch.py --setting_dir sim_data_batch/B1 --models_dir models/checkpoints --epochs 100 --latent_dim 16 --hidden 128 --decoder rff --decoder_kwargs '{"num_features":1024,"lengthscale":1.2,"ard":true,"learn_lengthscale":true,"learn_omegas":false,"seed":0}' --neg_ratio 10 --lambda_feat 1.0 --lambda_kl 0.005 --kl_warmup_epochs 80 --val_auc_neg_ratio 1
```

B1 (windows)

```bash
python exp1_train_batch.py --setting_dir sim_data_batch\B1 --models_dir models\checkpoints --epochs 100 --latent_dim 16 --hidden 128 --decoder rff --decoder_kwargs "{\"num_features\":1024,\"lengthscale\":1.2,\"ard\":true,\"learn_lengthscale\":true,\"learn_omegas\":false,\"seed\":0}" --neg_ratio 10 --lambda_feat 1.0 --lambda_kl 0.005 --kl_warmup_epochs 80 --val_auc_neg_ratio 1
```