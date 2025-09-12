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
python exp1_train_batch.py --setting_dir sim_data_batch/A1 --results_dir results --epochs 200 --latent_dim 16 --hidden 128 --decoder dot --decoder_kwargs '{}' --neg_ratio 10 --lambda_feat 1.0 --lambda_kl 0.005 --kl_warmup_epochs 100 --val_auc_neg_ratio 1 --lr 5e-4
```

```bash
python exp1_train_batch.py --setting_dir sim_data_batch/A1 --results_dir results --epochs 200 --latent_dim 16 --hidden 128 --decoder rff --decoder_kwargs '{"num_features":1024,"lengthscale":1.2,"ard":true,"learn_lengthscale":true,"learn_omegas":false,"seed":0}' --neg_ratio 10 --lambda_feat 1.0 --lambda_kl 0.005 --kl_warmup_epochs 100 --val_auc_neg_ratio 1 --lr 5e-4
```

```bash
python exp1_train_batch.py --setting_dir sim_data_batch/A1_poly_feats --model RG-G-VAE --results_dir results --epochs 2000 --latent_dim 2 --hidden 128 --decoder rff --decoder_kwargs '{"num_features":1024,"lengthscale":1.2,"ard":true,"learn_lengthscale":true,"learn_omegas":false,"seed":0}' --feat_dec_hidden 128 --neg_ratio 20 --lambda_edge 1.0 --lambda_feat 1.0 --lambda_kl 0.005 --kl_warmup_epochs 100 --val_auc_neg_ratio 1 --lr 1e-4 --edge_weighting none --task_weighting fixed
```

A1 (windows)

```bash
python exp1_train_batch.py --setting_dir sim_data_batch\A1 --results_dir results --epochs 200 --latent_dim 16 --hidden 128 --decoder dot --decoder_kwargs "{}" --neg_ratio 10 --lambda_feat 1.0 --lambda_kl 0.005 --kl_warmup_epochs 100 --val_auc_neg_ratio 1 --lr 5e-4
```

```bash
python exp1_train_batch.py --setting_dir sim_data_batch\A1 --results_dir results --epochs 200 --latent_dim 16 --hidden 128 --decoder rff --decoder_kwargs "{\"num_features\":1024,\"lengthscale\":1.2,\"ard\":true,\"learn_lengthscale\":true,\"learn_omegas\":false,\"seed\":0}" --neg_ratio 10 --lambda_feat 1.0 --lambda_kl 0.005 --kl_warmup_epochs 100 --val_auc_neg_ratio 1 --lr 5e-4
```

```bash
python exp1_train_batch.py --setting_dir sim_data_batch/A1_poly_feats --model RG-G-VAE --results_dir results --epochs 2000 --latent_dim 2 --hidden 128 --decoder rff --decoder_kwargs "{\"num_features\":1024,\"lengthscale\":1.2,\"ard\":true,\"learn_lengthscale\":true,\"learn_omegas\":false,\"seed\":0}" --feat_dec_hidden 128 --neg_ratio 20 --lambda_edge 1.0 --lambda_feat 1.0 --lambda_kl 0.005 --kl_warmup_epochs 100 --val_auc_neg_ratio 1 --lr 1e-4 --edge_weighting none --task_weighting fixed
```

A2 (linux)

```bash
python exp1_train_batch.py --setting_dir sim_data_batch/A2 --results_dir results --epochs 200 --latent_dim 16 --hidden 128 --decoder radial --decoder_kwargs '{}' --neg_ratio 10 --lambda_feat 1.0 --lambda_kl 0.005 --kl_warmup_epochs 100 --val_auc_neg_ratio 1 --lr 5e-4
```

```bash
python exp1_train_batch.py --setting_dir sim_data_batch/A2 --results_dir results --epochs 200 --latent_dim 16 --hidden 128 --decoder rff --decoder_kwargs '{"num_features":1024,"lengthscale":1.2,"ard":true,"learn_lengthscale":true,"learn_omegas":false,"seed":0}' --neg_ratio 10 --lambda_feat 1.0 --lambda_kl 0.005 --kl_warmup_epochs 100 --val_auc_neg_ratio 1 --lr 5e-4
```

A2 (windows)

```bash
python exp1_train_batch.py --setting_dir sim_data_batch\A2 --results_dir results --epochs 200 --latent_dim 16 --hidden 128 --decoder radial --decoder_kwargs "{}" --neg_ratio 10 --lambda_feat 1.0 --lambda_kl 0.005 --kl_warmup_epochs 100 --val_auc_neg_ratio 1 --lr 5e-4
```

```bash
python exp1_train_batch.py --setting_dir sim_data_batch\A2 --results_dir results --epochs 200 --latent_dim 16 --hidden 128 --decoder rff --decoder_kwargs "{\"num_features\":1024,\"lengthscale\":1.2,\"ard\":true,\"learn_lengthscale\":true,\"learn_omegas\":false,\"seed\":0}" --neg_ratio 10 --lambda_feat 1.0 --lambda_kl 0.005 --kl_warmup_epochs 100 --val_auc_neg_ratio 1 --lr 5e-4
```

B1 (linux)

```bash
python exp1_train_batch.py --setting_dir sim_data_batch/B1 --results_dir results --epochs 200 --latent_dim 16 --hidden 128 --decoder rff --decoder_kwargs '{"num_features":1024,"lengthscale":1.2,"ard":true,"learn_lengthscale":true,"learn_omegas":false,"seed":0}' --neg_ratio 10 --lambda_feat 1.0 --lambda_kl 0.005 --kl_warmup_epochs 120 --val_auc_neg_ratio 1 --lr 1e-3
```

B1 (windows)

```bash
python exp1_train_batch.py --setting_dir sim_data_batch\B1 --results_dir results --epochs 200 --latent_dim 16 --hidden 128 --decoder rff --decoder_kwargs "{\"num_features\":1024,\"lengthscale\":1.2,\"ard\":true,\"learn_lengthscale\":true,\"learn_omegas\":false,\"seed\":0}" --neg_ratio 10 --lambda_feat 1.0 --lambda_kl 0.005 --kl_warmup_epochs 120 --val_auc_neg_ratio 1 --lr 1e-3
```

General

baseline setting + D=2 + recon feats only
```bash
python exp1_train_batch.py --setting_dir sim_data_batch/A1_poly_feats --model RG-G-VAE --results_dir results --epochs 2000 --latent_dim 2 --hidden 128 --decoder rff --decoder_kwargs '{"num_features":1024,"lengthscale":1.2,"ard":true,"learn_lengthscale":true,"learn_omegas":false,"seed":0}' --feat_dec_hidden 128 --neg_ratio 20 --lambda_feat 1.0 --lambda_edge 0.0 --lambda_kl 0.005 --kl_warmup_epochs 100 --val_auc_neg_ratio 1 --lr 1e-4 --edge_weighting none --task_weighting fixed
```

baseline setting + loss balance + D=2
```bash
python exp1_train_batch.py --setting_dir sim_data_batch/A1_poly_feats --model RG-G-VAE --results_dir results --epochs 2000 --latent_dim 2 --hidden 128 --decoder rff --decoder_kwargs '{"num_features":1024,"lengthscale":1.2,"ard":true,"learn_lengthscale":true,"learn_omegas":false,"seed":0}' --feat_dec_hidden 128 --neg_ratio 20 --lambda_feat 1.0 --lambda_edge 1.0 --lambda_kl 0.005 --kl_warmup_epochs 100 --val_auc_neg_ratio 1 --lr 1e-4 --edge_weighting weighted_renorm --task_weighting uncertainty
```

### Baselines

MLE

```bash
python exp1_mle.py --setting_dir ./sim_data_batch/A1_poly_feats --out_dir ./results_mle/A1_poly_feats_D2 --latent_dim 2
```

VI
```bash
python exp1_vi.py --setting_dir ./sim_data_batch/A1_poly_feats --out_dir ./results_vi/A1_poly_feats_D2 --latent_dim 2
```

USVT

```bash
python exp1_usvt.py --setting_dir ./sim_data_batch/A1_poly_feats --out_dir ./results_usvt/A1_poly_feats_D2 --d_max 2
```

### Visualisation

single run

```bash
python viz_training_logs.py --results results/0829_0021_A1 
```

multiple runs

```bash
python viz_training_logs.py --results results/0829_0021_A1,results/0829_0020_A2,results/0829_0149_B1 --no_show
```


### testing

Windows

recon both
```bash
python exp1_test_batch.py --setting_dir ./sim_data_batch/A1_poly_feats --ckpt ./results/0903_2227_A1_poly_feats/rg_vae_A1_poly_feats_best.pt --model RG-G-VAE --latent_dim 32 --hidden 128 --decoder rff --decoder_kwargs "{\"num_features\":1024,\"lengthscale\":1.2,\"ard\":true,\"learn_lengthscale\":true,\"learn_omegas\":false,\"seed\":0}" --feat_dec_hidden 128
```

recon feats only
```bash
python exp1_test_batch.py --setting_dir ./sim_data_batch/A1_poly_feats --ckpt ./results/0905_2003_A1_poly_feats/rg_vae_A1_poly_feats_best.pt --model RG-G-VAE --latent_dim 2 --hidden 128 --decoder rff --decoder_kwargs "{\"num_features\":1024,\"lengthscale\":1.2,\"ard\":true,\"learn_lengthscale\":true,\"learn_omegas\":false,\"seed\":0}" --feat_dec_hidden 128
```