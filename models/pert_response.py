"""PertResponseNet：FiLM 调制 per-gene 残差网络。"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from models._functional import (
    ADD_CAP,
    MUL_CAP,
    N_LOCAL,
    N_SCALAR,
    SIG_EPS,
    compress,
)
from models.base import BaseModelSpec, ModelParams


class _Block(nn.Module):
    def __init__(self, hidden: int, drop: float):
        super().__init__()
        self.norm = nn.LayerNorm(hidden)
        self.fc1 = nn.Linear(hidden, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.drop = nn.Dropout(drop)

    def forward(self, h, scale, shift):
        x = self.norm(h) * (1.0 + scale[:, None, :]) + shift[:, None, :]
        return h + self.drop(self.fc2(F.gelu(self.fc1(x))))


class PertResponseNet(nn.Module):
    def __init__(self, n_ctx_genes: int, esm_dim: int = 1280, hidden: int = 192,
                 depth: int = 3, cond: int = 128, drop: float = 0.15):
        super().__init__()
        self.depth = depth
        self.hidden = hidden
        self.ctx_enc = nn.Linear(n_ctx_genes, 32)
        self.esm_enc = nn.Sequential(nn.Linear(esm_dim, 96), nn.GELU(), nn.Linear(96, 48))
        self.glob = nn.Sequential(
            nn.Linear(32 + 48 + N_SCALAR, cond), nn.GELU(),
            nn.Dropout(drop), nn.Linear(cond, cond), nn.GELU(),
        )
        self.inp = nn.Linear(N_LOCAL, hidden)
        self.film = nn.Linear(cond, 2 * hidden * depth)
        self.blocks = nn.ModuleList([_Block(hidden, drop) for _ in range(depth)])
        self.out_norm = nn.LayerNorm(hidden)
        self.head_mul = nn.Linear(hidden, 1)
        self.head_add = nn.Linear(hidden, 1)
        self.head_sig = nn.Linear(hidden, 1)
        self.head_gate = nn.Linear(cond, 1)
        self.sig_slope = nn.Parameter(torch.ones(1))

        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)
        for head in (self.head_mul, self.head_add, self.head_sig, self.head_gate):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        nn.init.constant_(self.head_gate.bias, 5.0)

    def forward(self, local, ctx_expr, esm, scalars, prior, pert_ann=None):
        lfc_src = compress(local[..., 0])
        has_src = local[..., 3]
        c = self.glob(torch.cat([self.ctx_enc(ctx_expr), self.esm_enc(esm), scalars], -1))
        film = self.film(c).view(-1, self.depth, 2, self.hidden)
        h = self.inp(local)
        for i, blk in enumerate(self.blocks):
            h = blk(h, film[:, i, 0], film[:, i, 1])
        h = self.out_norm(h)

        mul = torch.tanh(self.head_mul(h)).squeeze(-1) * MUL_CAP
        add = torch.tanh(self.head_add(h)).squeeze(-1) * ADD_CAP
        lfc = lfc_src * (1.0 + mul) + add
        gate = torch.sigmoid(self.head_gate(c)) * has_src
        lfc = gate * lfc + (1.0 - gate) * compress(prior)

        sig_logit = (self.sig_slope * torch.log(lfc.abs() + SIG_EPS)
                     + self.head_sig(h).squeeze(-1).clamp(-4.0, 4.0))
        return lfc, sig_logit, gate.mean(-1)


class PertResponseModel(BaseModelSpec):
    name = "pert_response"
    description = "FiLM 调制 per-gene 残差网络（PertResponseNet）"

    def defaults(self) -> ModelParams:
        return ModelParams(hidden=192, depth=3, drop=0.15, cond=128)

    def build(self, n_ctx_genes: int, esm_dim: int, params: ModelParams | None = None) -> nn.Module:
        p = self.resolved(params)
        kw = dict(hidden=p.hidden, depth=p.depth, drop=p.drop)
        if p.cond is not None:
            kw["cond"] = p.cond
        return PertResponseNet(n_ctx_genes, esm_dim, **kw)
