# File: rgm_sims/config_schema.py

from __future__ import annotations
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Literal, Dict, Any

BaseMeasureType = Literal[
    "gaussian_mixture", "isotropic_gaussian", "flow_warped",
    "categorical_blocks", "product_gaussian"
]
KernelType = Literal[
    "block_constant", "radial_smooth", "directed_bilinear",
    "heterophily_indefinite", "translation_invariant"
]

class GraphCfg(BaseModel):
    N: int
    expected_degree: int = Field(..., gt=0)
    directed: bool = False
    self_loops: bool = False

class BaseMeasureCfg(BaseModel):
    type: BaseMeasureType
    params: Dict[str, Any] = {}

class LatentSpaceCfg(BaseModel):
    dimension: int
    base_measure: BaseMeasureCfg

class KernelCfg(BaseModel):
    type: KernelType
    params: Dict[str, Any] = {}

class DegreeCorrectionCfg(BaseModel):
    enabled: bool = False
    distribution: Literal["lognormal"] = "lognormal"
    params: Dict[str, Any] = {"mu": 0.0, "sigma": 0.0}

class NodeFeaturesCfg(BaseModel):
    enabled: bool = False
    emission: Literal["none", "gaussian", "categorical"] = "none"
    params: Dict[str, Any] = {}

class EdgeFeaturesCfg(BaseModel):
    enabled: bool = False
    emission: Literal["none", "gaussian", "categorical"] = "none"
    params: Dict[str, Any] = {}

class AttributesCfg(BaseModel):
    node_features: NodeFeaturesCfg = NodeFeaturesCfg()
    edge_features: EdgeFeaturesCfg = EdgeFeaturesCfg()

class MissingEdgesCfg(BaseModel):
    type: Literal["MAR"] = "MAR"
    rate: float = 0.0

class EgoNetSamplingCfg(BaseModel):
    enabled: bool = False
    k_nodes: Optional[int] = None
    radius_hops: int = 2
    strategy: Literal["seeded", "uniform"] = "seeded"

class NodeSubsamplingCfg(BaseModel):
    enabled: bool = False
    rate: float = 0.0

class ObservationCfg(BaseModel):
    missing_edges: MissingEdgesCfg = MissingEdgesCfg()
    ego_net_sampling: EgoNetSamplingCfg = EgoNetSamplingCfg()
    node_subsampling: NodeSubsamplingCfg = NodeSubsamplingCfg()

class EvaluationCfg(BaseModel):
    holdout_edge_fraction: float = 0.2
    negative_sampling_ratio: int = 5

class TemporalDeformationCfg(BaseModel):
    bandwidth_scale_per_step: float = 1.0
    rotation_deg_per_step: float = 0.0
    density_temperature_per_step: float = 1.0

class TemporalCfg(BaseModel):
    enabled: bool = False
    steps: int = 0
    deformation: TemporalDeformationCfg = TemporalDeformationCfg()

class SimConfig(BaseModel):
    name: str
    seed: int = 0
    graph: GraphCfg
    latent_space: LatentSpaceCfg
    kernel: KernelCfg
    degree_correction: DegreeCorrectionCfg = DegreeCorrectionCfg()
    attributes: AttributesCfg = AttributesCfg()
    observation: ObservationCfg = ObservationCfg()
    evaluation: EvaluationCfg = EvaluationCfg()
    temporal: TemporalCfg = TemporalCfg()