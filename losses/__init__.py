"""损失函数注册表。"""

from __future__ import annotations

from typing import Type

from losses._functional import LOSS_W, LOSS_W_V2, MAG_W, MAG_W_V2, SPAN, weights_from_mag
from losses.aux import MMDLoss, ZeroPertLoss
from losses.base import BaseLossSpec
from losses.six_score import SixScoreLoss, SixScoreLossSpec
from losses.six_score_v2 import SixScoreLossV2, SixScoreLossV2Spec

LOSSES: dict[str, Type[BaseLossSpec]] = {
    SixScoreLossSpec.name: SixScoreLossSpec,
    SixScoreLossV2Spec.name: SixScoreLossV2Spec,
}


def get_loss(name: str) -> BaseLossSpec:
    if name not in LOSSES:
        known = ", ".join(sorted(LOSSES))
        raise KeyError(f"未知损失 '{name}'，可选: {known}")
    return LOSSES[name]()


def list_losses() -> dict[str, str]:
    return {n: cls.description for n, cls in LOSSES.items()}

__all__ = [
    "SixScoreLoss",
    "SixScoreLossSpec",
    "SixScoreLossV2",
    "SixScoreLossV2Spec",
    "ZeroPertLoss",
    "MMDLoss",
    "BaseLossSpec",
    "get_loss",
    "list_losses",
    "LOSS_W",
    "LOSS_W_V2",
    "MAG_W",
    "MAG_W_V2",
    "SPAN",
    "weights_from_mag",
]
