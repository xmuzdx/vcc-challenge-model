"""辅助损失：零扰动约束与可微 MMD 分布监督。"""

from __future__ import annotations

import torch
from torch import nn


class ZeroPertLoss(nn.Module):
    """输入无扰动时预测应保持 control，即压缩空间 LFC → 0。"""

    def forward(self, lfc, universe):
        return (lfc.square() * universe).sum(1).mean() / universe.size(1)


class MMDLoss(nn.Module):
    """RBF-MMD between softly synthesised cells and real perturbation cells."""

    def __init__(self, sigma: float = 1.0):
        super().__init__()
        self.sigma = sigma

    def _k(self, a, b):
        # a, b: (n, d)
        d2 = (a[:, None, :] - b[None, :, :]).square().sum(-1)
        return torch.exp(-d2 / (2.0 * self.sigma ** 2 + 1e-6))

    def forward(self, pred, real):
        if pred.numel() == 0 or real.numel() == 0:
            return pred.new_zeros(())
        xx = self._k(pred, pred).mean()
        yy = self._k(real, real).mean()
        xy = self._k(pred, real).mean()
        return xx + yy - 2.0 * xy


def soft_synth_cells(ctrl, lfc, scale: float):
    """可微软合成：ctrl 为 log1p(CPM)，跳过 binomial/Poisson 以保留梯度。"""
    return torch.log1p(torch.expm1(ctrl) * torch.exp2(scale * lfc))
