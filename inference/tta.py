"""TTA：MC dropout 不确定性、一致性取优、低置信基因回退 control。"""

from __future__ import annotations

import torch

from engine.train import batchify


@torch.no_grad()
def predict_mean(models, corpus, perts: list[str], dest: str, device: str,
                 chunk: int = 24):
    lfc_out, sig_out = [], []
    for i in range(0, len(perts), chunk):
        sel = [(p, dest) for p in perts[i : i + chunk]]
        loc, ctx, esm, sca, pri, _, _, _, pann = _unpack(corpus, sel, device)
        la, sa = None, None
        for m in models:
            m.eval()
            lfc, sig, _ = m(loc, ctx, esm, sca, pri, pann)
            la = lfc if la is None else la + lfc
            sa = sig if sa is None else sa + sig
        lfc_out.append(la / len(models))
        sig_out.append(sa / len(models))
    return torch.cat(lfc_out), torch.cat(sig_out)


def _unpack(corpus, sel, device):
    out = batchify(corpus, sel, device)
    if len(out) == 8:
        loc, ctx, esm, sca, pri, uni, a, b = out
        pann = loc.new_zeros(loc.size(0), 16)
        return loc, ctx, esm, sca, pri, uni, a, b, pann
    return out


@torch.no_grad()
def mc_dropout_predict(model, corpus, perts: list[str], dest: str, device: str,
                       n_fwd: int = 32, chunk: int = 24):
    """Keep dropout on; return mean lfc/sig and per-gene std as uncertainty."""
    acc_lfc, acc_sig, acc_sq = None, None, None
    model.train()
    for _ in range(n_fwd):
        lfcs, sigs = [], []
        for i in range(0, len(perts), chunk):
            sel = [(p, dest) for p in perts[i : i + chunk]]
            loc, ctx, esm, sca, pri, _, _, _, pann = _unpack(corpus, sel, device)
            lfc, sig, _ = model(loc, ctx, esm, sca, pri, pann)
            lfcs.append(lfc)
            sigs.append(sig)
        lfc, sig = torch.cat(lfcs), torch.cat(sigs)
        acc_lfc = lfc if acc_lfc is None else acc_lfc + lfc
        acc_sig = sig if acc_sig is None else acc_sig + sig
        acc_sq = lfc.square() if acc_sq is None else acc_sq + lfc.square()
    model.eval()
    mean = acc_lfc / n_fwd
    var = (acc_sq / n_fwd - mean.square()).clamp_min(0.0)
    return mean, acc_sig / n_fwd, var.sqrt()


def consistency_pick(lfcs: list[torch.Tensor], sigs: list[torch.Tensor]):
    """无标签取优：选与 ensemble 均值方向最一致的一份。"""
    stack = torch.stack(lfcs)
    mu = stack.mean(0)
    agree = (stack.sign() == mu.sign()).float().mean(dim=(-1, -2))
    i = int(agree.argmax())
    return lfcs[i], sigs[i], float(agree[i])


def confidence_gate(lfc, std, keep_quantile: float):
    """标准差分位超过 keep_quantile 的基因回退到 0（与 control bit-identical）。"""
    if keep_quantile >= 1.0 or std is None:
        return lfc
    # smaller std = more confident; keep the most certain quantile
    thr = std.flatten().quantile(keep_quantile)
    return torch.where(std <= thr, lfc, torch.zeros_like(lfc))
