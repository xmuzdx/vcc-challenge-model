from __future__ import annotations

"""扰动响应网络。换模型时在 train.py 里调用 pert_response 或 local_response。"""

"""模型辅助函数：振幅压缩、特征标量、shrinkage。"""


import numpy as np
import torch
from torch import nn

LFC_GAMMA = 0.35    # 压缩指数
LFC_CAP = 4.0       # |lfc| 截断上限
LFC_FLOOR = 1e-6    # 防 pow 反向发散，见 compress
MUL_CAP = 0.5       # 乘性残差头幅度上限
ADD_CAP = 0.75      # 加性残差头幅度上限
DPRIOR_CAP = 0.5    # 先验幅度上限
SIG_EPS = 1e-3      # log(|lfc|) 的保护项

LOCAL_FEATS = ("lfc_src", "zn_src", "sig_src", "has_src",
               "ctrl_tgt", "ctrl_src", "dctrl", "prior")
N_LOCAL = len(LOCAL_FEATS)
N_SCALAR = 8
IDX_LFC_SRC = LOCAL_FEATS.index("lfc_src")
IDX_HAS_SRC = LOCAL_FEATS.index("has_src")


def compress(lfc: torch.Tensor, gamma: float = LFC_GAMMA,
             cap: float = LFC_CAP, floor: float = LFC_FLOOR) -> torch.Tensor:
    """signed-power 压缩：sign(x) * clamp(|x|, floor, cap) ** gamma。

    floor 只为数值稳定：gamma < 1 时 pow 在 0 处反向发散，且 sign(0)=0 会把
    inf 变成 NaN。37% 的工作空间没有 GWPS 覆盖，program 重建每批都会碰到精确
    0；floor 让这些位置梯度为 0，同时不影响前向值。
    """
    return lfc.sign() * lfc.abs().clamp(floor, cap).pow(gamma)


def shrink_lfc(lfc: np.ndarray, z: np.ndarray, c: float = 1.0) -> np.ndarray:
    """按 z**2/(z**2+c) 把低 z 的 lfc 收缩到 0。"""
    return (lfc * (z ** 2 / (z ** 2 + c))).astype(np.float32)


def build_scalars(sig_src: np.ndarray, lfc_src: np.ndarray, universe: np.ndarray,
                  ctrl_tgt: np.ndarray, n_ctx_univ: int) -> np.ndarray:
    """8 维扰动全局标量（效应强度 / 规模 / 上下文覆盖）。"""
    u = universe.astype(np.float32)
    nu = max(u.sum(), 1.0)
    a = np.abs(lfc_src) * u
    n_sig = float((sig_src & universe).sum())
    return np.array([
        np.log1p(n_sig),                        # 命中基因数
        n_sig / nu,                             # 命中比例
        a.sum() / nu,                           # 平均 |lfc|
        a.max() if a.size else 0.0,             # 峰值 |lfc|
        np.sqrt((a ** 2).sum()),                # L2 范数
        np.median(ctrl_tgt[universe]) if universe.any() else 0.0,
        np.log1p(n_ctx_univ) - 9.0,             # ctx 覆盖数（去基线）
        (a > 0.5).sum() / nu,                   # 强效应占比
    ], dtype=np.float32)


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


"""PertResponseNet v1.1：针对初代诊断漏洞的完整修补版（可直接替换原文件）。

修补对照（P 编号对应诊断结论）：
  [P1] 残差坍缩   : GATE_INIT 5→2 + 训练期源丢弃 p_src_drop
  [P2] 无源零学习 : 去掉 gate×has_src 硬掩码 + 新增 Δprior 头
  [P3] 残差限幅死 : MUL_CAP 0.5 / ADD_CAP 0.75 / DPRIOR_CAP 0.5（压缩空间）
  [P4] _Cross 未用: 每层 Block 后接一次基因混合，k=64
  [P5] 无基因维度 : FiLM 与 gate 均逐基因（输入 [h; cond]）
  [P6] 无基因身份 : gene embedding（配套单独 weight_decay，见训练脚本）
  [P7] 容量不足   : hidden 384 / depth 6 / cond 256
  [P8] sig 头冗余 : 独立显著性头，去掉 log|lfc| 影子项

依赖 models/base.py 的两处修改：
    MUL_CAP = 0.5
    ADD_CAP = 0.75
    DPRIOR_CAP = 0.5      # 新增一行

调用侧变化：build_pert / PertResponseNet 首参为 n_genes（universe 大小），
forward 返回值形状不变：(lfc, sig_logit, gate.mean(-1))。
"""


import torch
import torch.nn.functional as F
from torch import nn


IDX_ZN_SRC = LOCAL_FEATS.index("zn_src")
IDX_SIG_SRC = LOCAL_FEATS.index("sig_src")
SRC_CHANNELS = (IDX_LFC_SRC, IDX_ZN_SRC, IDX_SIG_SRC, IDX_HAS_SRC)   # 源丢弃时一起置零

CTX_DIM = 64                    # ctx 表达编码维度 [P7]
ESM_MID, ESM_OUT = 256, 128     # ESM 投影 MLP 的中间 / 输出维度 [P7]
GO_IN, GO_OUT = 65, 64          # GO one-hot 维度 / 编码维度
GATE_INIT = 2.0                 # [P1] 5→2：sigmoid(2)≈0.88，仍偏信源但留学习空间
P_SRC_DROP = 0.15               # [P1] 训练期源丢弃概率


class _Block(nn.Module):
    """Pre-norm MLP 残差块；[P5] FiLM 逐基因、每层独立。

    scale 过 tanh 限幅（调制倍率限制在 [0,2]），shift 零初始化，
    保持"初始恒等"性质，但不再卡死输出头的表达范围。
    """

    def __init__(self, hidden: int, cond: int, drop: float):
        super().__init__()
        self.norm = nn.LayerNorm(hidden)
        self.fc1 = nn.Linear(hidden, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.drop = nn.Dropout(drop)
        self.film = nn.Linear(hidden + cond, 2 * hidden)    # [P5] 逐基因 FiLM
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)

    def forward(self, h, c):                                # h [B,G,H], c [B,G,cond]
        scale, shift = self.film(torch.cat([h, c], -1)).chunk(2, -1)
        x = self.norm(h) * (1.0 + torch.tanh(scale)) + shift
        return h + self.drop(self.fc2(F.gelu(self.fc1(x))))


class _Cross(nn.Module):
    """genes -> k 个 slot -> genes 的 Perceiver 式混合，O(G·k)。[P4] 正式启用。"""

    def __init__(self, hidden: int, k: int = 64):
        super().__init__()
        self.q = nn.Parameter(0.02 * torch.randn(k, hidden))
        self.norm = nn.LayerNorm(hidden)
        self.w = nn.Linear(hidden, hidden)          # 零初始化 => 初始恒等
        nn.init.zeros_(self.w.weight)
        nn.init.zeros_(self.w.bias)

    def forward(self, h):                           # [B, G, H]
        x = self.norm(h)
        d = x.shape[-1] ** 0.5
        m = torch.softmax(self.q @ x.transpose(-1, -2) / d, -1) @ x   # [B,k,H]
        return h + self.w(torch.softmax(x @ m.transpose(-1, -2) / d, -1) @ m)


class PertResponseNet(nn.Module):
    def __init__(self, n_genes: int, n_ctx_genes: int, esm_dim: int = 1280,
                 hidden: int = 192, depth: int = 3, cond: int = 256, drop: float = 0.15,
                 gene_emb: int = 64, cross_k: int = 64, p_src_drop: float = P_SRC_DROP,
                 use_go: bool = False, raw_esm: bool = False, no_scalars: bool = False,
                 sig_prior: float = 0.0):
        super().__init__()
        # sig 头零初始化时 topk 在全体平局里随机挑基因；加上源 |z| 先验，
        # 起点就是已验证的 |z| 排序（旧 checkpoint 为 0，行为不变）
        self.sig_prior = sig_prior
        self.depth, self.hidden = depth, hidden
        self.use_go, self.raw_esm, self.no_scalars = use_go, raw_esm, no_scalars
        self.p_src_drop = p_src_drop

        # ---- [P6] 基因身份 ----
        self.n_genes = n_genes
        self.register_buffer("gene_id", torch.arange(n_genes), persistent=False)
        self.gene_emb = nn.Embedding(n_genes, gene_emb) if gene_emb > 0 else None
        self.gene_proj = None
        if self.gene_emb is not None and gene_emb != hidden:
            self.gene_proj = nn.Linear(gene_emb, hidden)
            nn.init.zeros_(self.gene_proj.weight)
            nn.init.zeros_(self.gene_proj.bias)

        # ---- 条件编码：ctx 表达 + ESM(+GO) + 标量 -> [B, cond] ----
        self.ctx_enc = nn.Linear(n_ctx_genes, CTX_DIM)
        self.esm_enc = (nn.Identity() if raw_esm else
                        nn.Sequential(nn.Linear(esm_dim, ESM_MID), nn.GELU(),
                                      nn.Linear(ESM_MID, ESM_OUT)))
        self.go_enc = (nn.Sequential(nn.Linear(GO_IN, GO_OUT), nn.GELU())
                       if use_go else None)
        cond_in = (CTX_DIM + (esm_dim if raw_esm else ESM_OUT)
                   + (GO_OUT if use_go else 0) + (0 if no_scalars else N_SCALAR))
        self.glob = nn.Sequential(
            nn.Linear(cond_in, cond), nn.GELU(),
            nn.Dropout(drop), nn.Linear(cond, cond), nn.GELU(),
        )

        # ---- per-gene 主干 + 逐基因 FiLM + 基因混合 ----
        self.inp = nn.Linear(N_LOCAL, hidden)
        self.blocks = nn.ModuleList(_Block(hidden, cond, drop) for _ in range(depth))
        self.cross = nn.ModuleList(_Cross(hidden, cross_k) for _ in range(depth))  # [P4]
        self.out_norm = nn.LayerNorm(hidden)

        # ---- 输出头 ----
        self.head = nn.Linear(hidden, 4)              # (mul, add, dprior, sig) [P2][P3][P8]
        self.head_gate = nn.Linear(hidden + cond, 1)  # [P5] 逐基因门控

        # 零初始化（训练稳定）；坍缩问题由源丢弃 [P1] 与门偏置下调解决
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        nn.init.zeros_(self.head_gate.weight)
        nn.init.constant_(self.head_gate.bias, GATE_INIT)

    # ---------- 内部 ----------

    def _condition(self, ctx_expr, esm, scalars, go):
        parts = [self.ctx_enc(ctx_expr), self.esm_enc(esm)]
        if self.use_go:
            parts.append(self.go_enc(go if go is not None
                                     else esm.new_zeros(esm.shape[0], GO_IN)))
        if not self.no_scalars:
            parts.append(scalars)
        return self.glob(torch.cat(parts, -1))

    def _trunk(self, local, cond, gene_id=None):
        del gene_id
        h = self.inp(local)
        if self.gene_emb is not None:                              # [P6]
            g = self.gene_emb(self.gene_id)
            if self.gene_proj is not None:
                g = self.gene_proj(g)
            h = h + g
        cb = cond.unsqueeze(1).expand(-1, h.shape[1], -1)          # [B,G,cond]
        for blk, cross in zip(self.blocks, self.cross):            # [P4][P5]
            h = blk(h, cb)
            h = cross(h)
        return self.out_norm(h)

    def _drop_source(self, local):
        """[P1] 训练期源丢弃：以 p 概率把整条样本当作无源样本。

        四个源通道（lfc_src / zn_src / sig_src / has_src）一起置零；旧版只清了
        lfc_src 和 has_src，zn_src、sig_src 仍带着源信息，sig_logit 的 |zn_src| 项也照常起作用。
        注意 scalars（由源 lfc / sig 计算）和 ctx_topk 模式下的 ctx 仍含源信息；
        要做到完全无源，需要在 batchify 阶段按同一掩码处理。
        """
        keep = (torch.rand(local.shape[0], 1, device=local.device)
                >= self.p_src_drop).to(local.dtype)
        local = local.clone()
        for idx in SRC_CHANNELS:
            local[..., idx] = local[..., idx] * keep
        return local

    # ---------- 对外 ----------

    def forward_trunk(self, local, ctx_expr, esm, scalars, prior, go=None, gene_id=None):
        """主干特征 [B, n_genes, hidden]；不做门控、先验回退和残差头。"""
        del prior
        return self._trunk(local, self._condition(ctx_expr, esm, scalars, go), gene_id)

    def forward(self, local, ctx_expr, esm, scalars, prior, pert_ann=None,
                go=None, gene_id=None, return_base=False):
        del pert_ann

        # [P1] 源丢弃：训练期以 p 概率整条样本当"无源"
        if self.training and self.p_src_drop > 0:
            local = self._drop_source(local)
        lfc_src = local[..., IDX_LFC_SRC]
        zn_src = local[..., IDX_ZN_SRC]

        c = self._condition(ctx_expr, esm, scalars, go)
        h = self._trunk(local, c, gene_id)
        hc = torch.cat([h, c.unsqueeze(1).expand(-1, h.shape[1], -1)], -1)   # [B,G,H+cond]

        mul, add, dprior, sig = self.head(h).unbind(-1)
        src_path = (compress(lfc_src) * (1.0 + MUL_CAP * torch.tanh(mul))
                    + ADD_CAP * torch.tanh(add))                              # [P3]
        prior_path = compress(prior) + DPRIOR_CAP * torch.tanh(dprior)        # [P2]
        gate = torch.sigmoid(self.head_gate(hc)).squeeze(-1)                  # [B,G] [P2][P5]
        lfc = gate * src_path + (1.0 - gate) * prior_path
        sig_logit = sig + self.sig_prior * zn_src.abs()                       # [P8] 独立头

        if return_base:
            # 零初始化时的输出（直传基线），供 anchor 损失做信任域
            with torch.no_grad():
                g0 = torch.sigmoid(self.head_gate.bias.new_tensor(GATE_INIT))
                base = g0 * compress(lfc_src) + (1.0 - g0) * compress(prior)
            return lfc, sig_logit, gate.mean(-1), base
        return lfc, sig_logit, gate.mean(-1)


def build_pert(n_genes: int, n_ctx_genes: int, esm_dim: int,
               args: dict | None = None) -> PertResponseNet:
    a = dict(args or {})
    extra = a.get("extra") or {}

    def pick(name, default):
        v = a.get(name, extra.get(name))
        return default if v is None else v

    return PertResponseNet(
        n_genes, n_ctx_genes, esm_dim,
        hidden=int(pick("hidden", 192)), depth=int(pick("depth", 3)),
        cond=int(pick("cond", 256)), drop=float(pick("drop", 0.15)),
        gene_emb=int(pick("gene_emb", 64)),
        cross_k=int(pick("cross_k", 64)),
        p_src_drop=float(pick("p_src_drop", P_SRC_DROP)),
        use_go=bool(a.get("go") or extra.get("go")),
        raw_esm=bool(a.get("raw_esm") or extra.get("raw_esm")),
        no_scalars=bool(a.get("no_scalars") or extra.get("no_scalars")),
        sig_prior=float(pick("sig_prior", 0.0)),
    )





"""LocalResponseNet：只学"基因局部函数"的迁移修正（诊断 D4，方案 P4）。

PertResponseNet 的全局条件（ctx 表达编码、scalars 第 5/6 维）在训练中每个 destination 只有一个取值，
四个训练系就是四个点；再加上跨基因混合（_Cross）和 gene embedding，模型能拼出细胞系指纹，
记住逐基因、逐细胞系的修正，到 H1 上只能外推。这里把这三样都去掉：

  * 可学部分是 f(逐基因可测量特征 | 扰动层面条件)。逐基因特征默认为
    lfc_src, zn_src, sig_src, has_src, ctrl_tgt, ctrl_src, dctrl；
    不含 prior（它近似一个基因指纹），需要时用 --local-feats 加回；
  * 条件只来自扰动：ESM（--no-esm 可关）、GO（--go）、scalars 中随扰动变化的 6 维；
  * 没有 gene embedding，没有跨基因混合，每个基因独立经过同一个 MLP。

ctrl_tgt / dctrl 在同一细胞系内部就有上万个取值，训练样本是（基因, 扰动）对，而不是四个细胞系；
用到 H1 时不需要在"细胞系空间"里外推。

输出头零初始化，起点严格等于直传：lfc = compress(lfc_src)，
sig_logit = sig_prior·|zn_src| + 1e-6·|lfc|。forward 签名与 PertResponseNet 相同。
"""


import torch
import torch.nn.functional as F
from torch import nn

ESM_MID, ESM_OUT = 256, 128
GO_IN, GO_OUT = 65, 64
TIE_EPS = 1e-6
# build_scalars 的第 5 维（destination 对照表达中位数）和第 6 维（destination universe 大小）
# 每个细胞系一个常数，等价于细胞系 ID，这里不用
PERT_SCALAR_IDX = (0, 1, 2, 3, 4, 7)
DEFAULT_FEATS = ("lfc_src", "zn_src", "sig_src", "has_src", "ctrl_tgt", "ctrl_src", "dctrl")
assert N_SCALAR == 8, "PERT_SCALAR_IDX 按 build_scalars 的 8 维写死"


class _LocalBlock(nn.Module):
    """Pre-norm 逐基因 MLP 残差块。FiLM 只看扰动条件，零初始化时不做调制。"""

    def __init__(self, hidden: int, cond: int, drop: float):
        super().__init__()
        self.norm = nn.LayerNorm(hidden)
        self.fc1 = nn.Linear(hidden, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.drop = nn.Dropout(drop)
        self.film = nn.Linear(cond, 2 * hidden)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)

    def forward(self, h, c):                        # h [B,G,H]，c [B,cond]
        scale, shift = self.film(c).unsqueeze(1).chunk(2, -1)
        x = self.norm(h) * (1.0 + torch.tanh(scale)) + shift
        return h + self.drop(self.fc2(F.gelu(self.fc1(x))))


class LocalResponseNet(nn.Module):
    def __init__(self, esm_dim: int, hidden: int = 64, depth: int = 2, cond: int = 64,
                 drop: float = 0.1, use_go: bool = False, use_esm: bool = True,
                 sig_prior: float = 4.0, feats: tuple[str, ...] = DEFAULT_FEATS):
        super().__init__()
        bad = [f for f in feats if f not in LOCAL_FEATS]
        if bad:
            raise ValueError(f"未知的逐基因特征 {bad}，可选 {LOCAL_FEATS}")
        self.feats = tuple(feats)
        self.register_buffer("feat_idx", torch.tensor([LOCAL_FEATS.index(f) for f in self.feats]),
                             persistent=False)
        self.register_buffer("pert_sc", torch.tensor(PERT_SCALAR_IDX), persistent=False)
        self.sig_prior = float(sig_prior)
        self.use_go, self.use_esm = use_go, use_esm

        self.esm_enc = (nn.Sequential(nn.Linear(esm_dim, ESM_MID), nn.GELU(),
                                      nn.Linear(ESM_MID, ESM_OUT)) if use_esm else None)
        self.go_enc = nn.Sequential(nn.Linear(GO_IN, GO_OUT), nn.GELU()) if use_go else None
        cond_in = len(PERT_SCALAR_IDX) + (ESM_OUT if use_esm else 0) + (GO_OUT if use_go else 0)
        self.glob = nn.Sequential(nn.Linear(cond_in, cond), nn.GELU(), nn.Dropout(drop),
                                  nn.Linear(cond, cond), nn.GELU())

        self.inp = nn.Linear(len(self.feats), hidden)
        self.blocks = nn.ModuleList(_LocalBlock(hidden, cond, drop) for _ in range(depth))
        self.out_norm = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, 3)            # (mul, add, sig)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def _condition(self, esm, scalars, go):
        parts = [scalars.index_select(-1, self.pert_sc)]
        if self.use_esm:
            parts.append(self.esm_enc(esm))
        if self.use_go:
            parts.append(self.go_enc(go if go is not None
                                     else scalars.new_zeros(scalars.shape[0], GO_IN)))
        return self.glob(torch.cat(parts, -1))

    def forward(self, local, ctx_expr, esm, scalars, prior, pert_ann=None, go=None,
                gene_id=None, return_base=False):
        del ctx_expr, prior, pert_ann, gene_id      # 细胞系层面的输入一律不用
        base = compress(local[..., IDX_LFC_SRC])
        c = self._condition(esm, scalars, go)
        h = self.inp(local.index_select(-1, self.feat_idx))
        for blk in self.blocks:
            h = blk(h, c)
        mul, add, sig = self.head(self.out_norm(h)).unbind(-1)
        lfc = base * (1.0 + MUL_CAP * torch.tanh(mul)) + ADD_CAP * torch.tanh(add)
        sig_logit = sig + self.sig_prior * local[..., IDX_ZN_SRC].abs() + TIE_EPS * base.abs()
        gate = lfc.new_ones(lfc.shape[0])
        if return_base:
            return lfc, sig_logit, gate, base.detach()
        return lfc, sig_logit, gate


def build_local(n_genes: int, n_ctx_genes: int, esm_dim: int,
                args: dict | None = None) -> LocalResponseNet:
    """与 build_pert 同签名，方便 train.py / loading.py 统一调用；n_genes、n_ctx_genes 不用。"""
    del n_genes, n_ctx_genes
    a = dict(args or {})

    def pick(name, default):
        v = a.get(name)
        return default if v is None else v

    feats = a.get("local_feats") or ""
    if isinstance(feats, str):
        feats = tuple(f.strip() for f in feats.split(",") if f.strip())
    return LocalResponseNet(
        esm_dim,
        hidden=int(pick("hidden", 64)), depth=int(pick("depth", 2)),
        cond=int(pick("cond", 64)), drop=float(pick("drop", 0.1)),
        use_go=bool(a.get("go")), use_esm=not bool(a.get("no_esm")),
        sig_prior=float(pick("sig_prior", 4.0)),
        feats=tuple(feats) or DEFAULT_FEATS,
    )


class Spec:
    """训练入口用的模型规格。真正的网络要等语料维度确定后再 build。"""

    def __init__(self, kind: str, lr: float, batch_size: int, **kw):
        self.kind = kind
        self.lr = float(lr)
        self.batch_size = int(batch_size)
        self.kw = kw

    def build(self, n_genes: int, n_ctx_genes: int, esm_dim: int) -> nn.Module:
        kw = self.kw
        if self.kind == "pert":
            return PertResponseNet(
                n_genes, n_ctx_genes, esm_dim,
                hidden=int(kw.get("hidden", 192)),
                depth=int(kw.get("depth", 3)),
                cond=int(kw.get("cond", 256)),
                drop=float(kw.get("drop", 0.15)),
                gene_emb=int(kw.get("gene_emb", 64)),
                cross_k=int(kw.get("cross_k", 64)),
                p_src_drop=float(kw.get("p_src_drop", P_SRC_DROP)),
                use_go=bool(kw.get("go", False)),
                raw_esm=bool(kw.get("raw_esm", False)),
                no_scalars=bool(kw.get("no_scalars", False)),
                sig_prior=float(kw.get("sig_prior", 0.0)),
            )
        feats = kw.get("local_feats") or ""
        if isinstance(feats, str):
            feats = tuple(f.strip() for f in feats.split(",") if f.strip())
        return LocalResponseNet(
            esm_dim,
            hidden=int(kw.get("hidden", 64)),
            depth=int(kw.get("depth", 2)),
            cond=int(kw.get("cond", 64)),
            drop=float(kw.get("drop", 0.1)),
            use_go=bool(kw.get("go", False)),
            use_esm=not bool(kw.get("no_esm", False)),
            sig_prior=float(kw.get("sig_prior", 4.0)),
            feats=tuple(feats) or DEFAULT_FEATS,
        )


def pert_response(lr: float = 2e-3, batch_size: int = 24, hidden: int = 192,
                  depth: int = 3, drop: float = 0.15, cond: int = 256, **kw) -> Spec:
    return Spec("pert", lr, batch_size, hidden=hidden, depth=depth, drop=drop, cond=cond, **kw)


def local_response(lr: float = 1e-3, batch_size: int = 24, hidden: int = 64,
                   depth: int = 2, drop: float = 0.1, cond: int = 64, **kw) -> Spec:
    return Spec("local", lr, batch_size, hidden=hidden, depth=depth, drop=drop, cond=cond, **kw)
