"""ZeroFanNet：宽条件分支 + 窄深 per-gene 主干的二代响应网络。"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from models._functional import (
    ADD_CAP,
    MUL_CAP,
    N_LOCAL,
    N_SCALAR,
    SIG_EPS,
    compress,
)
from models.base import BaseModelSpec, ModelParams
from models.priors import PERT_ANN_DIM


def _mlp(dims: list[int], drop: float) -> nn.Sequential:
    layers: list[nn.Module] = []
    for a, b in zip(dims[:-1], dims[1:]):
        layers += [nn.Linear(a, b), nn.GELU(), nn.Dropout(drop)]
    return nn.Sequential(*layers)


class _Block(nn.Module):
    def __init__(self, hidden: int, drop: float, drop_path: float):
        super().__init__()
        self.norm = nn.LayerNorm(hidden)
        self.fc1 = nn.Linear(hidden, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.drop = nn.Dropout(drop)
        self.ls = nn.Parameter(torch.full((hidden,), 1e-2))
        self.drop_path = drop_path

    def forward(self, h, scale, shift):
        x = self.norm(h) * (1.0 + scale[:, None, :]) + shift[:, None, :]
        y = self.drop(self.fc2(F.gelu(self.fc1(x)))) * self.ls
        if self.training and self.drop_path > 0:
            keep = (torch.rand(h.size(0), 1, 1, device=h.device) >= self.drop_path)
            y = y * keep.to(y.dtype) / (1.0 - self.drop_path)
        return h + y


class ZeroFanNet(nn.Module):
    """容量加在 (batch, cond) 一侧；per-gene 主干保持窄，配 LayerScale / drop-path。

    Residual heads stay zero-initialised so a freshly built network is exactly
    compress(GWPS) source transfer.  The trunk is free to learn, but it has
    to beat that baseline on held-out H1 before it changes a submission.
    """

    def __init__(self, n_ctx_genes: int, esm_dim: int = 1280, hidden: int = 256,
                 depth: int = 6, cond: int = 1024, drop: float = 0.35,
                 cond_depth: int = 8, esm_depth: int = 6, stoch_depth: float = 0.1,
                 use_checkpoint: bool = True, gene_feat_dim: int = 52,
                 pert_ann_dim: int = PERT_ANN_DIM):
        super().__init__()
        self.depth = depth
        self.hidden = hidden
        self.use_checkpoint = use_checkpoint
        gf_h = 32

        self.ctx_enc = _mlp([n_ctx_genes, 256, 256], drop)
        esm_dims = [esm_dim] + [cond] * esm_depth
        self.esm_enc = _mlp(esm_dims, drop)
        self.pert_enc = _mlp([pert_ann_dim, 128, 128], drop)
        self.gene_enc = nn.Sequential(nn.Linear(gene_feat_dim, gf_h), nn.GELU())
        glob_in = 256 + cond + N_SCALAR + 128
        self.glob = nn.Sequential(
            nn.Linear(glob_in, cond), nn.GELU(), nn.Dropout(drop),
            _mlp([cond] * (cond_depth + 1), drop),
        )
        self.inp = nn.Linear(N_LOCAL + gf_h, hidden)
        self.film = nn.Linear(cond, 2 * hidden * depth)
        dpr = [stoch_depth * i / max(depth - 1, 1) for i in range(depth)]
        self.blocks = nn.ModuleList([_Block(hidden, drop, p) for p in dpr])
        self.out_norm = nn.LayerNorm(hidden)
        self.head_mul = nn.Linear(hidden, 1)
        self.head_add = nn.Linear(hidden, 1)
        self.head_sig = nn.Linear(hidden, 1)
        self.head_gate = nn.Linear(cond, 1)
        self.sig_slope = nn.Parameter(torch.ones(1))
        self.register_buffer("gene_feat", torch.zeros(1, gene_feat_dim), persistent=False)

        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)
        for head in (self.head_mul, self.head_add, self.head_sig, self.head_gate):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        nn.init.constant_(self.head_gate.bias, 5.0)

    def attach_gene_space(self, feat_w: torch.Tensor) -> None:
        """绑定工作基因空间上的面板先验，形状 (n_w, F)。"""
        self.gene_feat = feat_w.to(device=self.gene_feat.device, dtype=torch.float32)

    def _block(self, i: int, h, scale, shift):
        return self.blocks[i](h, scale, shift)

    def forward(self, local, ctx_expr, esm, scalars, prior, pert_ann=None):
        lfc_src = compress(local[..., 0])
        has_src = local[..., 3]
        if pert_ann is None:
            pert_ann = local.new_zeros(local.size(0), PERT_ANN_DIM)
        c = self.glob(torch.cat([
            self.ctx_enc(ctx_expr), self.esm_enc(esm), scalars, self.pert_enc(pert_ann),
        ], -1))
        film = self.film(c).view(-1, self.depth, 2, self.hidden)

        gf = self.gene_enc(self.gene_feat)
        if gf.size(0) == 1:
            gf = gf.expand(local.size(1), -1)
        h = self.inp(torch.cat([local, gf.expand(local.size(0), -1, -1)], -1))
        for i in range(self.depth):
            if self.use_checkpoint and self.training:
                h = checkpoint(self.blocks[i], h, film[:, i, 0], film[:, i, 1],
                               use_reentrant=False)
            else:
                h = self.blocks[i](h, film[:, i, 0], film[:, i, 1])
        h = self.out_norm(h)

        mul = torch.tanh(self.head_mul(h)).squeeze(-1) * MUL_CAP
        add = torch.tanh(self.head_add(h)).squeeze(-1) * ADD_CAP
        lfc = lfc_src * (1.0 + mul) + add
        gate = torch.sigmoid(self.head_gate(c)) * has_src
        lfc = gate * lfc + (1.0 - gate) * compress(prior)
        sig_logit = (self.sig_slope * torch.log(lfc.abs() + SIG_EPS)
                     + self.head_sig(h).squeeze(-1).clamp(-4.0, 4.0))
        return lfc, sig_logit, gate.mean(-1)


class ZeroFanModel(BaseModelSpec):
    name = "zero_fan"
    description = "宽条件分支 + 窄深主干（ZeroFanNet），残差头零初始化"

    def defaults(self) -> ModelParams:
        return ModelParams(
            hidden=256, depth=6, drop=0.35, cond=1024,
            extra={"cond_depth": 8, "esm_depth": 6, "stoch_depth": 0.1,
                   "checkpoint": True},
        )

    def build(self, n_ctx_genes: int, esm_dim: int, params: ModelParams | None = None) -> nn.Module:
        p = self.resolved(params)
        ex = p.extra
        return ZeroFanNet(
            n_ctx_genes, esm_dim,
            hidden=p.hidden, depth=p.depth, drop=p.drop, cond=p.cond or 1024,
            cond_depth=int(ex.get("cond_depth", 8)),
            esm_depth=int(ex.get("esm_depth", 6)),
            stoch_depth=float(ex.get("stoch_depth", 0.1)),
            use_checkpoint=bool(ex.get("checkpoint", True)),
        )

    def train_args(self, params: ModelParams | None = None) -> dict:
        out = super().train_args(params)
        p = self.resolved(params)
        if p.cond is not None:
            out["cond"] = p.cond
        return out
