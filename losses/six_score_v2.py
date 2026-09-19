"""SixScoreLossV2：1/span 全权重 + 可微 top-k，对齐真实 fid/jac。"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from losses._functional import (
    COV_FLOOR,
    DIR_BETA,
    LOSS_W_V2,
    PDS_TAU,
    SOFT_TEMP,
    SOFT_TOPK,
    soft_topk_mask,
)
from losses.base import BaseLossSpec


class SixScoreLossV2(nn.Module):
    """Official-six surrogate with amplitude terms restored and a soft top-k.

    The v1 fid/jac surrogates score every gene the sigmoid lights up.  The
    real metrics only see the k genes that sparsify keeps, so the gradient
    never learned that calling past ~168 is free damage.  The mask here is
    the missing piece: s is gated by a differentiable top-k of sig_logit.
    """

    def __init__(self, weights: dict[str, float] | None = None,
                 topk: int = SOFT_TOPK, temp: float = SOFT_TEMP):
        super().__init__()
        self.w = dict(LOSS_W_V2 if weights is None else weights)
        self.topk = topk
        self.temp = temp

    def forward(self, lfc, sig_logit, true_lfc, true_sig, universe, pds_mask,
                sample_w=None):
        m = universe
        gate = soft_topk_mask(sig_logit, self.topk, self.temp)
        s = torch.sigmoid(sig_logit) * m * gate
        z = true_sig.to(lfc.dtype) * m
        n_z = z.sum(1)
        agree = torch.tanh(DIR_BETA * lfc) * torch.sign(true_lfc)

        nmae_b = (((lfc - true_lfc).abs() * z).sum(1)
                  / (true_lfc.abs() * z).sum(1).clamp_min(1e-6))
        mse_b = (((lfc - true_lfc) ** 2 * m).sum(1)
                 / ((true_lfc ** 2) * m).sum(1).clamp_min(1e-6))

        inter = (s * z).sum(1)
        jac_b = 1.0 - inter / (s.sum(1) + n_z - inter).clamp_min(1e-6)

        prec = (s * agree).sum(1) / s.sum(1).clamp_min(1e-6)
        cov = (s.sum(1) / n_z.clamp_min(COV_FLOOR)).clamp(max=1.0)
        fid_b = 1.0 - prec.clamp_min(0.0) * cov

        wc = s * z
        reach_b = 1.0 - ((wc * agree).sum(1) / wc.sum(1).clamp_min(1e-6)).clamp_min(0.0)

        mm = m * pds_mask[None, :]
        a = F.normalize(lfc * mm, dim=1)
        b = F.normalize(true_lfc * mm, dim=1)
        logits = (a @ b.T) / PDS_TAU
        pds_b = F.cross_entropy(
            logits, torch.arange(a.shape[0], device=a.device), reduction="none")

        parts_b = {"nmae": nmae_b, "mse": mse_b, "jac": jac_b,
                   "fid": fid_b, "reach": reach_b, "pds": pds_b}
        if sample_w is None:
            w = lfc.new_ones(lfc.size(0))
        else:
            w = sample_w.to(lfc.dtype)
        w = w / w.sum().clamp_min(1e-6)
        parts = {k: float((v * w).sum().detach()) for k, v in parts_b.items()}
        total = sum(self.w[k] * (parts_b[k] * w).sum() for k in parts_b)
        return total, parts


class SixScoreLossV2Spec(BaseLossSpec):
    name = "six_score_v2"
    description = "六项代理 v2：1/span 含 MSE + 可微 top-k"

    def defaults(self) -> dict[str, float]:
        return dict(LOSS_W_V2)

    def build(self, weights: dict[str, float] | None = None) -> nn.Module:
        return SixScoreLossV2(weights=self.resolved(weights))
