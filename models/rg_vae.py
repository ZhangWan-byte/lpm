# File: models/rg_vae.py

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import bce_logits


class GraphConv(nn.Module):
    """Simple GCN layer: H' = A_norm @ (H W). A_norm is sparse."""
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)

    def forward(self, A_norm: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
        HW = self.lin(H)
        return torch.sparse.mm(A_norm, HW)


class NodeEncoder(nn.Module):
    """
    Encodes observable node features (optionally concatenated with structural features) to q(z|·).
    Approximates f^{-1}: x_i -> z_i.
    """
    def __init__(self, in_dim: int, hidden: int, latent_dim: int, layers: int = 2):
        super().__init__()
        self.gc1 = GraphConv(in_dim, hidden)
        self.gc2 = GraphConv(hidden, hidden) if layers >= 2 else None
        self.mu = nn.Linear(hidden, latent_dim)
        self.logvar = nn.Linear(hidden, latent_dim)

    def forward(self, A_norm: torch.Tensor, feats: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = F.relu(self.gc1(A_norm, feats))
        if self.gc2 is not None:
            h = F.relu(self.gc2(A_norm, h))
        return self.mu(h), self.logvar(h)


# ---------------------------
# Edge decoders (pairwise)
# ---------------------------

class RadialDecoder(nn.Module):
    """p(A_ij=1 | z_i, z_j) = sigma(bias - alpha * ||z_i - z_j||^2)"""
    def __init__(self, latent_dim: int):
        super().__init__()
        self.log_alpha = nn.Parameter(torch.tensor(0.0))
        self.bias = nn.Parameter(torch.tensor(-2.0))

    def logits(self, zi: torch.Tensor, zj: torch.Tensor) -> torch.Tensor:
        dist2 = ((zi - zj) ** 2).sum(dim=-1)
        alpha = F.softplus(self.log_alpha) + 1e-4
        return self.bias - alpha * dist2


class DotDecoder(nn.Module):
    """sigma(bias + scale * <z_i, z_j>)"""
    def __init__(self, latent_dim: int):
        super().__init__()
        self.log_scale = nn.Parameter(torch.tensor(0.0))
        self.bias = nn.Parameter(torch.tensor(-2.0))

    def logits(self, zi: torch.Tensor, zj: torch.Tensor) -> torch.Tensor:
        scale = F.softplus(self.log_scale) + 1e-4
        return self.bias + scale * (zi * zj).sum(dim=-1)


class BilinearDecoder(nn.Module):
    """sigma(bias + (W_out z_i)·(W_in z_j)) — captures directed-style interactions with two projections."""
    def __init__(self, latent_dim: int, proj_dim: Optional[int] = None):
        super().__init__()
        h = proj_dim or latent_dim
        self.W_out = nn.Linear(latent_dim, h, bias=False)
        self.W_in = nn.Linear(latent_dim, h, bias=False)
        self.bias = nn.Parameter(torch.tensor(-2.0))

    def logits(self, zi: torch.Tensor, zj: torch.Tensor) -> torch.Tensor:
        so = self.W_out(zi)
        ti = self.W_in(zj)
        return self.bias + (so * ti).sum(dim=-1)


class IndefiniteDecoder(nn.Module):
    """sigma(bias + z_i^T S z_j) with learnable diagonal S (allows heterophily via sign flips)."""
    def __init__(self, latent_dim: int):
        super().__init__()
        self.s = nn.Parameter(torch.zeros(latent_dim))
        self.bias = nn.Parameter(torch.tensor(-2.0))

    def logits(self, zi: torch.Tensor, zj: torch.Tensor) -> torch.Tensor:
        return self.bias + (zi * self.s * zj).sum(dim=-1)


class MLPDecoder(nn.Module):
    """Symmetric MLP on pair features: |zi-zj| and zi*zj -> logit."""
    def __init__(self, latent_dim: int, hidden: int = 64):
        super().__init__()
        in_dim = 2 * latent_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def logits(self, zi: torch.Tensor, zj: torch.Tensor) -> torch.Tensor:
        x = torch.cat([torch.abs(zi - zj), zi * zj], dim=-1)
        return self.net(x).squeeze(-1)


class DegreeCorrectedRadialDecoder(nn.Module):
    """Last dim of z carries a per-node degree propensity s (log-scale)."""
    def __init__(self, latent_dim: int):
        super().__init__()
        assert latent_dim >= 2, "dc_radial requires latent_dim >= 2"
        self.core = RadialDecoder(latent_dim - 1)

    def logits(self, zi: torch.Tensor, zj: torch.Tensor) -> torch.Tensor:
        s_i = zi[:, -1]
        s_j = zj[:, -1]
        core_zi = zi[:, :-1]
        core_zj = zj[:, :-1]
        return self.core.logits(core_zi, core_zj) + s_i + s_j


class RFFDecoder(nn.Module):
    """Random Fourier Feature decoder: logit = bias + scale * φ(z_i)^T φ(z_j)."""
    def __init__(
        self,
        latent_dim: int,
        num_features: int = 512,
        lengthscale: float = 1.0,
        ard: bool = False,
        learn_lengthscale: bool = True,
        learn_omegas: bool = False,
        seed: int = 0,
    ):
        super().__init__()
        self.m = int(num_features)
        self.bias = nn.Parameter(torch.tensor(-2.0))
        self.log_scale = nn.Parameter(torch.tensor(0.0))

        # lengthscale parameter(s)
        if ard:
            init = torch.full((latent_dim,), float(lengthscale))
            self._raw_ell = nn.Parameter(init.log()) if learn_lengthscale else nn.Parameter(init.log(), requires_grad=False)
        else:
            init = torch.tensor(float(lengthscale))
            self._raw_ell = nn.Parameter(init.log()) if learn_lengthscale else nn.Parameter(init.log(), requires_grad=False)
        self.ard = ard

        g = torch.Generator().manual_seed(int(seed))
        eps = torch.randn((self.m, latent_dim), generator=g)
        phases = torch.rand((self.m,), generator=g) * 2 * torch.pi

        if learn_omegas:
            self.Omega_eps = nn.Parameter(eps)
            self.phases = nn.Parameter(phases)
        else:
            self.register_buffer("Omega_eps", eps)
            self.register_buffer("phases", phases)

        self.sqrt2_over_m = (2.0 / self.m) ** 0.5

    def _lengthscale_vec(self) -> torch.Tensor:
        ell = F.softplus(self._raw_ell) + 1e-5
        return ell

    def _scaled_omega(self) -> torch.Tensor:
        ell = self._lengthscale_vec()
        if self.ard:
            inv_ell = ell.reciprocal()
            return self.Omega_eps * inv_ell.unsqueeze(0)
        else:
            inv_ell = ell.reciprocal().view(1, 1)
            return self.Omega_eps * inv_ell

    def _phi(self, z: torch.Tensor) -> torch.Tensor:
        Omega = self._scaled_omega()       # [m, D]
        proj = z @ Omega.t()               # [*, m]
        H = torch.cos(proj + self.phases) * self.sqrt2_over_m
        return H

    def logits(self, zi: torch.Tensor, zj: torch.Tensor) -> torch.Tensor:
        phi_i = self._phi(zi)
        phi_j = self._phi(zj)
        scale = F.softplus(self.log_scale) + 1e-4
        return self.bias + scale * (phi_i * phi_j).sum(dim=-1)


# ---------------------------
# Feature decoder (node attributes)
# ---------------------------

class FeatureDecoderGaussian(nn.Module):
    """
    Decodes z_i -> parameters of p(x_i|z_i).
    Here: Gaussian with mean MLP(z_i) and fixed unit variance (MSE loss).
    """
    def __init__(self, latent_dim: int, x_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, x_dim)
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)  # predicted mean


# ---------------------------
# VAE wrapper (two-branch decoder)
# ---------------------------

class RG_VAE(nn.Module):
    def __init__(
        self,
        input_dim: int,                 # observed node feature dimension (x)
        latent_dim: int = 16,
        hidden: int = 64,
        enc_layers: int = 2,
        use_struct_feats: bool = False, # if True, concat simple structural features to observed x
        decoder: str = "radial",        # edge decoder name
        decoder_kwargs: Optional[dict] = None,
        feat_dec_hidden: int = 64,      # hidden width for feature decoder
    ):
        """
        Encoder approximates f^{-1}: x -> z.
        Two-branch decoder:
          (a) p(x|z): FeatureDecoderGaussian
          (b) p(A|z_i,z_j): edge decoder family
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.use_struct_feats = use_struct_feats
        self.struct_dim = 3 if use_struct_feats else 0

        enc_in = input_dim + self.struct_dim
        self.encoder = NodeEncoder(in_dim=enc_in, hidden=hidden, latent_dim=latent_dim, layers=enc_layers)

        # Edge decoder
        self.decoder_name = (decoder or "radial").lower()
        self.decoder = self._make_edge_decoder(self.decoder_name, latent_dim, (decoder_kwargs or {}))

        # Feature decoder (Gaussian)
        self.feature_decoder = FeatureDecoderGaussian(latent_dim=latent_dim, x_dim=input_dim, hidden=feat_dec_hidden)

    @staticmethod
    def _make_edge_decoder(name: str, latent_dim: int, kwargs: dict) -> nn.Module:
        if name == "radial":
            return RadialDecoder(latent_dim)
        if name == "dot":
            return DotDecoder(latent_dim)
        if name == "bilinear":
            return BilinearDecoder(latent_dim, proj_dim=kwargs.get("proj_dim"))
        if name == "indefinite":
            return IndefiniteDecoder(latent_dim)
        if name == "mlp":
            return MLPDecoder(latent_dim, hidden=kwargs.get("hidden", 64))
        if name == "dc_radial":
            return DegreeCorrectedRadialDecoder(latent_dim)
        if name == "rff":
            return RFFDecoder(
                latent_dim,
                num_features=kwargs.get("num_features", 512),
                lengthscale=kwargs.get("lengthscale", 1.0),
                ard=kwargs.get("ard", False),
                learn_lengthscale=kwargs.get("learn_lengthscale", True),
                learn_omegas=kwargs.get("learn_omegas", False),
                seed=int(kwargs.get("seed", 0)),
            )
        raise ValueError(f"Unknown decoder '{name}'")

    @staticmethod
    def structural_features(A_norm: torch.Tensor) -> torch.Tensor:
        """3 simple structural features per node: ones, log(deg+1), (log(deg+1))^2."""
        N = A_norm.size(0)
        ones = torch.ones(N, 1, device=A_norm.device)
        deg = torch.sparse.sum(A_norm, dim=1).to_dense().unsqueeze(1)
        logd = torch.log1p(deg)
        feats = torch.cat([ones, logd, logd ** 2], dim=1)
        return feats

    def _compose_encoder_input(self, A_norm: torch.Tensor, feats: torch.Tensor) -> torch.Tensor:
        if self.use_struct_feats:
            struct = self.structural_features(A_norm)
            return torch.cat([feats, struct], dim=1)
        return feats

    def encode(self, A_norm: torch.Tensor, feats: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self._compose_encoder_input(A_norm, feats)
        mu, logvar = self.encoder(A_norm, x)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    # ----- Edge branch -----

    def pair_logits(self, z: torch.Tensor, pairs: torch.Tensor) -> torch.Tensor:
        zi = z[pairs[:, 0]]
        zj = z[pairs[:, 1]]
        return self.decoder.logits(zi, zj)

    # ----- Losses -----

    @staticmethod
    def kl_normal(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        # KL(q||p) for standard normal prior
        return 0.5 * torch.sum(torch.exp(logvar) + mu**2 - 1.0 - logvar)

    @staticmethod
    def gaussian_mse_loss(x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # Fixed unit-variance Gaussian => MSE/2 up to constant; we use mean MSE for scale stability
        return F.mse_loss(x_hat, x)

    # ----- ELBO with two branches -----

    def elbo(
        self,
        A_norm: torch.Tensor,
        pos_pairs: torch.Tensor,
        neg_pairs: torch.Tensor,
        feats: torch.Tensor,            # observed node features x
        lambda_feat: float = 1.0,
        lambda_kl: float = 1e-3,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        mu, logvar = self.encode(A_norm, feats)
        z = self.reparameterize(mu, logvar)

        # Edge reconstruction on pos+neg pairs
        pairs = torch.cat([pos_pairs, neg_pairs], dim=0)
        labels = torch.cat(
            [torch.ones(pos_pairs.size(0), device=z.device),
             torch.zeros(neg_pairs.size(0), device=z.device)]
        )
        logits = self.pair_logits(z, pairs)
        recon_edge = bce_logits(logits, labels)

        # Feature reconstruction
        x_hat = self.feature_decoder(z)
        recon_feat = self.gaussian_mse_loss(x_hat, feats)

        # KL
        kl = self.kl_normal(mu, logvar) / mu.size(0)  # average per node

        # Total
        loss = recon_edge + lambda_feat * recon_feat + lambda_kl * kl
        stats = {
            "recon_edge": float(recon_edge.item()),
            "recon_feat": float(recon_feat.item()),
            "kl": float(kl.item()),
            "decoder": self.decoder_name,
        }
        return loss, stats

    # ----- Embedding -----

    def embed(self, A_norm: torch.Tensor, feats: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Posterior mean embeddings μ. For Exp1, pass node features (x).
        If feats is None and use_struct_feats=True, returns μ using only structural features (fallback).
        """
        if feats is None:
            x = self.structural_features(A_norm) if self.use_struct_feats else torch.zeros(
                A_norm.size(0), 0, device=A_norm.device
            )
        else:
            x = self._compose_encoder_input(A_norm, feats)
        mu, _ = self.encoder(A_norm, x)
        return mu
