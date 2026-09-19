"""模型辅助函数：振幅压缩、特征标量、shrinkage。"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

LFC_GAMMA = 0.35
LFC_CAP = 4.0
MUL_CAP = 0.3
ADD_CAP = 0.25
SIG_EPS = 1e-3

LOCAL_FEATS = (
    "lfc_src",
    "zn_src",
    "sig_src",
    "has_src",
    "ctrl_tgt",
    "ctrl_src",
    "dctrl",
    "prior",
)
N_LOCAL = len(LOCAL_FEATS)
N_SCALAR = 8


def compress(lfc: torch.Tensor, gamma: float = LFC_GAMMA,
             cap: float = LFC_CAP) -> torch.Tensor:
    return lfc.sign() * lfc.abs().clamp(max=cap).pow(gamma)


def shrink_lfc(lfc: np.ndarray, z: np.ndarray, c: float = 1.0) -> np.ndarray:
    return (lfc * (z**2 / (z**2 + c))).astype(np.float32)


def build_scalars(sig_src: np.ndarray, lfc_src: np.ndarray, universe: np.ndarray,
                  ctrl_tgt: np.ndarray, n_ctx_univ: int) -> np.ndarray:
    u = universe.astype(np.float32)
    nu = max(u.sum(), 1.0)
    a = np.abs(lfc_src) * u
    n_sig = float((sig_src & universe).sum())
    return np.array([
        np.log1p(n_sig),
        n_sig / nu,
        float(a.sum() / nu),
        float(a.max()) if a.size else 0.0,
        float(np.sqrt((a ** 2).sum())),
        float(np.median(ctrl_tgt[universe])) if universe.any() else 0.0,
        np.log1p(n_ctx_univ) - 9.0,
        float((a > 0.5).sum() / nu),
    ], dtype=np.float32)


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
