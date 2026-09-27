"""训练损失。six_v2() 是默认，aligned() 是带直传信任域的另一项。"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

CENTER_W = 0.3


def decenter(lfc, w: float = CENTER_W):
    """减去该 context 内所有扰动的平均响应。"""
    if w <= 0.0:
        return lfc
    return lfc - w * lfc.mean(0, keepdim=True)

"""损失权重与官方 metric span 常量。"""


SPAN = {"fid": 0.295748, "jac": 0.370727, "pds": 0.448742,
        "nmae": 0.601318, "reach": 0.892709, "mse": 0.952500}
MAG_W = {"nmae": 0.06, "mse": 0.0}
# v2 把幅度项按 1/span 放回：提交 raw MSE≈6.1 对应 α≈2.26，幅度过大而非过小。
MAG_W_V2 = {"nmae": 1.0, "mse": 1.0}

DIR_BETA = 2.0
PDS_TAU = 0.10
COV_FLOOR = 400.0
# D1: matching n_real=168 drops H1 score_avg to -0.04; 7000/0.5 is still best.
SOFT_TOPK = 7000
SOFT_TEMP = 0.25


def normalize_weights(raw: dict[str, float]) -> dict[str, float]:
    s = sum(raw.values())
    return {k: v / s for k, v in raw.items()}


def weights_from_mag(mag: dict[str, float] | None = None) -> dict[str, float]:
    mag = dict(MAG_W if mag is None else mag)
    raw = {k: mag.get(k, 1.0 / v) for k, v in SPAN.items()}
    return normalize_weights(raw)


_RAW_W = {k: MAG_W.get(k, 1.0 / v) for k, v in SPAN.items()}
LOSS_W = {k: v / sum(_RAW_W.values()) for k, v in _RAW_W.items()}
LOSS_W_V2 = weights_from_mag(MAG_W_V2)

# Align train weights with eval: PDS/Jaccard/Reach carry more of score_avg.
SCORE_W = dict(nmae=1.0, mse=0.3, jac=2.0, fid=1.0, reach=1.5, pds=2.0)


def soft_topk_mask(logit, k: int, temp: float = SOFT_TEMP):
    """可微 top-k：以第 k 大 logit 为阈值的 sigmoid 掩码。"""
    if k <= 0 or k >= logit.shape[-1]:
        return logit.new_ones(logit.shape)
    thr = logit.topk(k, dim=-1).values[..., -1:]
    return (logit - thr).div(temp).sigmoid()

"""SixScoreLossV2：1/span 全权重 + 可微 top-k，对齐真实 fid/jac。"""


import torch
import torch.nn.functional as F
from torch import nn


__all__ = ["SixScoreLossV2", "de_bg_mask"]

BG_K = 1000


def de_bg_mask(de_mask: torch.Tensor, universe: torch.Tensor,
               k: int = BG_K) -> torch.Tensor:
    """DE 基因并上每条样本均匀采样的 k 个背景基因。"""
    de = de_mask.bool() & universe.bool()
    bg = universe.bool() & ~de
    noise = torch.rand(bg.shape, device=bg.device, dtype=torch.float32)
    noise = noise.masked_fill(~bg, -1.0)
    take = min(int(k), bg.shape[-1])
    idx = noise.topk(take, dim=1).indices
    sampled = torch.zeros_like(bg)
    sampled.scatter_(1, idx, True)
    return de | (sampled & bg)


class SixScoreLossV2(nn.Module):
    """Official-six surrogate with amplitude terms restored and a soft top-k.

    The v1 fid/jac surrogates score every gene the sigmoid lights up.  The
    real metrics only see the k genes that sparsify keeps, so the gradient
    never learned that calling past ~168 is free damage.  The mask here is
    the missing piece: s is gated by a differentiable top-k of sig_logit.
    """

    def __init__(self, weights: dict[str, float] | None = None,
                 topk: int = SOFT_TOPK, temp: float = SOFT_TEMP,
                 bg_k: int = BG_K):
        super().__init__()
        self.w = dict(SCORE_W if weights is None else weights)
        self.topk = topk
        self.temp = temp
        self.bg_k = bg_k

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
        mmse = de_bg_mask(true_sig, m, self.bg_k).to(lfc.dtype)
        mse_b = (((lfc - true_lfc) ** 2 * mmse).sum(1)
                 / ((true_lfc ** 2) * mmse).sum(1).clamp_min(1e-6))

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

"""TransferAlignedLoss：面向 H1 迁移的损失，按官方评估链路对齐。

与 SixScoreLossV2 的区别：

* anchor：相对直传基线的残差能量。零残差基线在 H1 上 +0.128，v2 训练一离开
  起点就掉分，所以把“偏离基线”本身变成代价，残差只在确有收益时才长出来。
* pds 先 decenter（与 Validator 的 CENTER_W 一致），配合同 destination 的 batch，
  负样本是同一 context 的其他扰动，而不是其他细胞系。
* rank / dir 作用在 |lfc| 上而非 sigmoid(sig_logit)：评估里 sig_logit 只做
  topk=7000 粗筛，真正决定 DE 调用和 reach 排序的是合成细胞后的幅度。
* sig 头只留一个轻量 BCE。
"""


import torch
import torch.nn.functional as F
from torch import nn

RANK_TAU = 0.10


def _masked_log_softmax(x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    return torch.log_softmax(x.masked_fill(m <= 0, -1e4), dim=1)


class TransferAlignedLoss(nn.Module):
    def __init__(self, weights: dict[str, float] | None = None,
                 rank_tau: float = RANK_TAU, center_w: float = CENTER_W,
                 bg_k: int = BG_K):
        super().__init__()
        self.w = dict(ALIGNED_W)
        if weights:
            self.w.update(weights)
        self.rank_tau = rank_tau
        self.center_w = center_w
        self.bg_k = bg_k
        # 与 SixScoreLossV2 的日志字段对齐
        self.topk, self.temp = 0, rank_tau

    def forward(self, lfc, sig_logit, true_lfc, true_sig, universe, pds_mask, base):
        m = universe
        z = true_sig.to(lfc.dtype) * m
        n_z = z.sum(1)
        has_de = (n_z > 0).to(lfc.dtype)

        nmae_b = (((lfc - true_lfc).abs() * z).sum(1)
                  / (true_lfc.abs() * z).sum(1).clamp_min(1e-6))
        mmse = de_bg_mask(true_sig, m, self.bg_k).to(lfc.dtype)
        mse_b = (((lfc - true_lfc) ** 2 * mmse).sum(1)
                 / ((true_lfc ** 2) * mmse).sum(1).clamp_min(1e-6))

        # 残差信任域：相对基线能量归一化，与 mse 同一量纲
        anchor_b = (((lfc - base) ** 2 * m).sum(1)
                    / ((base ** 2) * m).sum(1).clamp_min(1e-6))

        # |lfc| 的排序对齐真实显著基因（ListNet，记 KL 以便日志可读）
        logq = _masked_log_softmax(lfc.abs() / self.rank_tau, m)
        tgt = z * true_lfc.abs()
        p = tgt / tgt.sum(1, keepdim=True).clamp_min(1e-12)
        rank_b = (p * (torch.log(p.clamp_min(1e-12)) - logq)).sum(1) * has_de

        # 高幅度基因的软方向精度（替代 fid / reach）
        pi = logq.exp()
        agree = torch.tanh(DIR_BETA * lfc) * torch.sign(true_lfc)
        dir_b = 1.0 - (pi * agree).sum(1)

        # 与评估一致的 decenter 后做 batch 内 InfoNCE
        mm = m * pds_mask[None, :]
        pc = lfc - self.center_w * lfc.mean(0, keepdim=True)
        tc = true_lfc - self.center_w * true_lfc.mean(0, keepdim=True)
        a = F.normalize(pc * mm, dim=1)
        b = F.normalize(tc * mm, dim=1)
        pds_b = F.cross_entropy(
            (a @ b.T) / PDS_TAU, torch.arange(a.shape[0], device=a.device),
            reduction="none")

        bce = F.binary_cross_entropy_with_logits(
            sig_logit, true_sig.to(lfc.dtype), reduction="none")
        sig_b = (bce * m).sum(1) / m.sum(1).clamp_min(1.0)

        parts_b = {"nmae": nmae_b, "mse": mse_b, "pds": pds_b, "rank": rank_b,
                   "dir": dir_b, "anchor": anchor_b, "sig": sig_b}
        parts = {k: float(v.detach().mean()) for k, v in parts_b.items()}
        total = sum(self.w[k] * v.mean() for k, v in parts_b.items())
        return total, parts


def six_v2():
    return SixScoreLossV2()


def aligned():
    return TransferAlignedLoss()
