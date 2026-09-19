"""模型注册表：通过 name 选择模型类。"""

from __future__ import annotations

from typing import Type

from models._functional import (
    LOCAL_FEATS,
    build_scalars,
    compress,
    n_params,
    shrink_lfc,
)
from models.base import BaseModelSpec, ModelParams
from models.pert_response import PertResponseModel, PertResponseNet
from models.priors import UNMAPPED_GWPS
from models.zero_fan import ZeroFanModel, ZeroFanNet
from models.zero_fan_v2 import ZeroFanV2Model, ZeroFanV2Net

MODELS: dict[str, Type[BaseModelSpec]] = {
    PertResponseModel.name: PertResponseModel,
    ZeroFanModel.name: ZeroFanModel,
    ZeroFanV2Model.name: ZeroFanV2Model,
}


def get_model(name: str) -> BaseModelSpec:
    if name not in MODELS:
        known = ", ".join(sorted(MODELS))
        raise KeyError(f"未知模型 '{name}'，可选: {known}")
    return MODELS[name]()


def create_model(name: str, n_ctx_genes: int, esm_dim: int,
                 params: ModelParams | None = None) -> PertResponseNet:
    return get_model(name).build(n_ctx_genes, esm_dim, params)


def list_models() -> dict[str, str]:
    return {n: cls.description for n, cls in MODELS.items()}

__all__ = [
    "PertResponseNet",
    "PertResponseModel",
    "ZeroFanNet",
    "ZeroFanModel",
    "ZeroFanV2Net",
    "ZeroFanV2Model",
    "BaseModelSpec",
    "ModelParams",
    "get_model",
    "create_model",
    "list_models",
    "LOCAL_FEATS",
    "compress",
    "build_scalars",
    "shrink_lfc",
    "n_params",
    "UNMAPPED_GWPS",
]
