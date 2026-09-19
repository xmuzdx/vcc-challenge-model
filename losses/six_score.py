"""SixScoreLoss：六项官方指标可微代理。"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from losses._functional import COV_FLOOR, DIR_BETA, LOSS_W, PDS_TAU, weights_from_mag
from losses.base import BaseLossSpec


class SixScoreLoss(nn.Module):
    """Differentiable surrogate of the six official metrics, 1/span weighted."""

    def __init__(self, weights: dict[str, float] | None = None):
        super().__init__()
        self.w = dict(LOSS_W if weights is None else weights)

    def forward(self, lfc, sig_logit, true_lfc, true_sig, universe, pds_mask,
                sample_w=None):
        m = universe
        s = torch.sigmoid(sig_logit) * m
        z = true_sig.to(lfc.dtype) * m
        n_z = z.sum(1)
        agree = torch.tanh(DIR_BETA * lfc) * torch.sign(true_lfc)

        l_nmae = (((lfc - true_lfc).abs() * z).sum(1)
                  / (true_lfc.abs() * z).sum(1).clamp_min(1e-6)).mean()

        l_mse = (((lfc - true_lfc) ** 2 * m).sum()
                 / ((true_lfc ** 2) * m).sum().clamp_min(1e-6))

        inter = (s * z).sum(1)
        l_jac = (1.0 - inter / (s.sum(1) + n_z - inter).clamp_min(1e-6)).mean()

        prec = (s * agree).sum(1) / s.sum(1).clamp_min(1e-6)
        cov = (s.sum(1) / n_z.clamp_min(COV_FLOOR)).clamp(max=1.0)
        l_fid = (1.0 - prec.clamp_min(0.0) * cov).mean()

        wc = s * z
        l_reach = (1.0 - ((wc * agree).sum(1) / wc.sum(1).clamp_min(1e-6)).clamp_min(0.0)).mean()

        mm = m * pds_mask[None, :]
        a = F.normalize(lfc * mm, dim=1)
        b = F.normalize(true_lfc * mm, dim=1)
        logits = (a @ b.T) / PDS_TAU
        l_pds = F.cross_entropy(logits, torch.arange(a.shape[0], device=a.device))

        parts = {"nmae": l_nmae, "mse": l_mse, "jac": l_jac,
                 "fid": l_fid, "reach": l_reach, "pds": l_pds}
        total = sum(self.w[k] * v for k, v in parts.items())
        return total, {k: float(v.detach()) for k, v in parts.items()}


class SixScoreLossSpec(BaseLossSpec):
    name = "six_score"
    description = "六项官方指标代理（SixScoreLoss，默认 LOSS_W）"

    def defaults(self) -> dict[str, float]:
        return dict(LOSS_W)

    def build(self, weights: dict[str, float] | None = None) -> nn.Module:
        return SixScoreLoss(weights=self.resolved(weights))
