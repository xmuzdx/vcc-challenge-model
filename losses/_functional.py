"""损失权重与官方 metric span 常量。"""

from __future__ import annotations

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


def soft_topk_mask(logit, k: int, temp: float = SOFT_TEMP):
    """可微 top-k：以第 k 大 logit 为阈值的 sigmoid 掩码。"""
    if k <= 0 or k >= logit.shape[-1]:
        return logit.new_ones(logit.shape)
    thr = logit.topk(k, dim=-1).values[..., -1:]
    return (logit - thr).div(temp).sigmoid()
