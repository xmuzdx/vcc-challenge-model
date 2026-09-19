"""模型规格基类：新模型继承此类并注册到 registry。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn


@dataclass
class ModelParams:
    """架构超参，未指定字段使用各模型默认值。"""

    hidden: int | None = None
    depth: int | None = None
    drop: float | None = None
    cond: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)


class BaseModelSpec(ABC):
    """可注册模型：负责构建 nn.Module 与 train.py 兼容的 CLI 参数。"""

    name: str = ""
    description: str = ""

    @abstractmethod
    def defaults(self) -> ModelParams:
        ...

    @abstractmethod
    def build(self, n_ctx_genes: int, esm_dim: int, params: ModelParams | None = None) -> nn.Module:
        ...

    def resolved(self, params: ModelParams | None = None) -> ModelParams:
        d = self.defaults()
        if params is None:
            return d
        return ModelParams(
            hidden=params.hidden if params.hidden is not None else d.hidden,
            depth=params.depth if params.depth is not None else d.depth,
            drop=params.drop if params.drop is not None else d.drop,
            cond=params.cond if params.cond is not None else d.cond,
            extra={**d.extra, **params.extra},
        )

    def train_args(self, params: ModelParams | None = None) -> dict[str, Any]:
        """映射到 train.py 的 --hidden / --depth / --drop。"""
        p = self.resolved(params)
        out: dict[str, Any] = {}
        if p.hidden is not None:
            out["hidden"] = p.hidden
        if p.depth is not None:
            out["depth"] = p.depth
        if p.drop is not None:
            out["drop"] = p.drop
        return out

    def load_from_checkpoint(self, ckpt_path: str, n_ctx_genes: int, esm_dim: int,
                             device: str) -> nn.Module:
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        a = ck["args"]
        params = ModelParams(hidden=a.get("hidden"), depth=a.get("depth"), drop=a.get("drop"))
        model = self.build(n_ctx_genes, esm_dim, params).to(device)
        model.load_state_dict(ck["model"])
        model.eval()
        return model
