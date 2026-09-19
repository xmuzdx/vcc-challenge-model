"""损失函数规格基类。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from torch import nn


class BaseLossSpec(ABC):
    name: str = ""
    description: str = ""

    @abstractmethod
    def defaults(self) -> dict[str, float]:
        """各 metric 权重默认值。"""

    @abstractmethod
    def build(self, weights: dict[str, float] | None = None) -> nn.Module:
        ...

    def resolved(self, weights: dict[str, float] | None = None) -> dict[str, float]:
        d = dict(self.defaults())
        if weights:
            d.update(weights)
        return d
