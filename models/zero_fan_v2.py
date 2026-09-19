"""ZeroFanV2Net：GWPS 程序分解 + 显著性双路融合。"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from models._functional import N_SCALAR, SIG_EPS, compress
from models.base import BaseModelSpec, ModelParams
from models.priors import PERT_ANN_DIM
from models.programs import PROGRAM_K

PROG_CAP = 0.5          # 程序系数最大相对调制
MIX_SLOPE = 4.0         # sig_src -> 融合权重
MIX_BIAS = -1.0


def _mlp(dims: list[int], drop: float) -> nn.Sequential:
    layers: list[nn.Module] = []
    for a, b in zip(dims[:-1], dims[1:]):
        layers += [nn.Linear(a, b), nn.GELU(), nn.Dropout(drop)]
    return nn.Sequential(*layers)


class _Block(nn.Module):
    def __init__(self, hidden: int, drop: float):
        super().__init__()
        self.norm = nn.LayerNorm(hidden)
        self.fc1 = nn.Linear(hidden, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.drop = nn.Dropout(drop)
        self.ls = nn.Parameter(torch.full((hidden,), 1e-2))

    def forward(self, h):
        return h + self.drop(self.fc2(F.gelu(self.fc1(self.norm(h))))) * self.ls


class ZeroFanV2Net(nn.Module):
    """Predict K program coefficients instead of 11961 per-gene residuals.

    Why: the destination corpora run 45-121 cells per perturbation, so their
    per-gene fold changes are mostly sampling error, and direction accuracy
    past the top of the ranking sits at 0.52 -- indistinguishable from chance.
    Regressing every gene therefore fits noise, which is why eight seeds and a
    22M-parameter trunk all failed to beat the zero-residual baseline.

    What survives is (a) the source's own significant genes, where direction
    accuracy is 0.717 but the median count is only 6, and (b) the low-rank
    structure the whole screen shares.  This net keeps (a) verbatim through a
    direct path and routes everything else through (b), where projection onto
    K programs discards per-gene noise while preserving the between-perturbation
    contrast that PDS -- 76% of the local score -- actually measures.
    """

    def __init__(self, n_ctx_genes: int, esm_dim: int = 1280, hidden: int = 256,
                 depth: int = 4, cond: int = 512, drop: float = 0.30,
                 n_prog: int = PROGRAM_K, cond_depth: int = 4,
                 pert_ann_dim: int = PERT_ANN_DIM, use_checkpoint: bool = True):
        super().__init__()
        self.n_prog = n_prog
        self.use_checkpoint = use_checkpoint

        self.ctx_enc = _mlp([n_ctx_genes, 256, 256], drop)
        self.esm_enc = _mlp([esm_dim, cond, cond], drop)
        self.pert_enc = _mlp([pert_ann_dim, 128, 128], drop)
        self.coef_enc = _mlp([n_prog, 256, 256], drop)
        glob_in = 256 + cond + N_SCALAR + 128 + 256
        self.glob = nn.Sequential(
            nn.Linear(glob_in, cond), nn.GELU(), nn.Dropout(drop),
            _mlp([cond] * (cond_depth + 1), drop),
        )
        self.blocks = nn.ModuleList([_Block(cond, drop) for _ in range(depth)])
        self.out_norm = nn.LayerNorm(cond)

        self.head_prog = nn.Linear(cond, n_prog)
        self.head_gate = nn.Linear(cond, 1)
        self.head_sig = nn.Linear(cond, 1)
        self.sig_slope = nn.Parameter(torch.ones(1))
        self.mix_slope = nn.Parameter(torch.tensor(MIX_SLOPE))
        self.mix_bias = nn.Parameter(torch.tensor(MIX_BIAS))
        self.register_buffer("basis", torch.zeros(n_prog, 1), persistent=False)

        for head in (self.head_prog, self.head_gate, self.head_sig):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        nn.init.constant_(self.head_gate.bias, 5.0)

    def attach_programs(self, basis: torch.Tensor) -> None:
        """绑定 (K, n_w) 程序基；必须在 forward 前调用。"""
        self.basis = basis.to(device=self.head_prog.weight.device,
                              dtype=torch.float32)

    def forward(self, local, ctx_expr, esm, scalars, prior, pert_ann=None):
        lfc_src = local[..., 0]
        zn_src = local[..., 1]
        sig_src = local[..., 2]
        has_src = local[..., 3]
        if pert_ann is None:
            pert_ann = local.new_zeros(local.size(0), PERT_ANN_DIM)

        coef = lfc_src @ self.basis.t()                       # (B, K)
        c = self.glob(torch.cat([
            self.ctx_enc(ctx_expr), self.esm_enc(esm), scalars,
            self.pert_enc(pert_ann), self.coef_enc(coef),
        ], -1))
        for blk in self.blocks:
            c = checkpoint(blk, c, use_reentrant=False) if (
                self.use_checkpoint and self.training) else blk(c)
        c = self.out_norm(c)

        delta = torch.tanh(self.head_prog(c)) * PROG_CAP      # zero-init
        lfc_prog = compress((coef * (1.0 + delta)) @ self.basis)
        lfc_direct = compress(lfc_src)

        # the source's own calls are worth 0.717; do not let the projection
        # average them away, so gate the two paths on source significance
        w = torch.sigmoid(self.mix_slope * sig_src + self.mix_bias)
        lfc = w * lfc_direct + (1.0 - w) * lfc_prog

        gate = torch.sigmoid(self.head_gate(c)) * has_src
        lfc = gate * lfc + (1.0 - gate) * compress(prior)

        # rank by |z|, not |lfc|: top-168 direction accuracy 0.559 vs 0.538
        base = torch.where(has_src > 0, zn_src.abs(), lfc.abs().clamp(max=1.0))
        sig_logit = (self.sig_slope * torch.log(base + SIG_EPS)
                     + self.head_sig(c).clamp(-4.0, 4.0))
        return lfc, sig_logit, gate.mean(-1)


class ZeroFanV2Model(BaseModelSpec):
    name = "zero_fan_v2"
    description = "GWPS 程序分解 + 显著性双路融合（ZeroFanV2Net）"

    def defaults(self) -> ModelParams:
        return ModelParams(
            hidden=256, depth=4, drop=0.30, cond=512,
            extra={"n_prog": PROGRAM_K, "cond_depth": 4, "checkpoint": True},
        )

    def build(self, n_ctx_genes: int, esm_dim: int,
              params: ModelParams | None = None) -> nn.Module:
        p = self.resolved(params)
        ex = p.extra
        return ZeroFanV2Net(
            n_ctx_genes, esm_dim, hidden=p.hidden, depth=p.depth,
            drop=p.drop, cond=p.cond or 512,
            n_prog=int(ex.get("n_prog", PROGRAM_K)),
            cond_depth=int(ex.get("cond_depth", 4)),
            use_checkpoint=bool(ex.get("checkpoint", True)),
        )
