"""训练。改下面这一段就能换模型、学习率和卡。

    python train.py
"""

from __future__ import annotations

from model import local_response, pert_response
from loss import six_v2

model = pert_response(lr=2e-3, batch_size=24, hidden=192, depth=3)
loss_fn = six_v2()
epochs = 30
gpus = (0,)          # 多卡写成 (0, 1, 2, 3)
tag = "try1"
patience = 8

import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from loss import TransferAlignedLoss
from model import build_scalars, compress, n_params, shrink_lfc

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / "data"
PRIOR = ROOT.parent / "prior"
PREP = DATA / "prep"

ESM_PATH = PRIOR / "gene_priors" / "full_esm_embeddings.npz"
GO_PATH = PRIOR / "gene_features" / "go_onto.npz"
PERT_ANN_PATH = PRIOR / "pert_classification.tsv"

SRC = "gwps"
TRAIN_LINES = ("rpe1", "hepg2", "jurkat", "k562ess")
VAL_LINE = "h1"


def prep_line_npz(name: str) -> Path:
    if name == "h1":
        return PREP / "val" / "h1.npz"
    return PREP / "train" / f"{name}.npz"


def prep_ctx_npz(ctx: str) -> Path:
    return PREP / "test" / f"ctx_{ctx}.npz"


def prep_h1_eval() -> Path:
    return PREP / "val" / "h1_eval.h5"

UNMAPPED_GWPS = frozenset({
    "AC015871.1", "AC118549.1", "AHSA2", "ALG1L", "ARPC4-TTLL3",
    "C14orf178", "C19orf48", "C1orf61", "C22orf46", "CCDC169-SOHLH2",
    "CENPBD1", "FAM86C1", "HHLA3", "NEDD8-MDP1", "NME1-NME2",
    "RBAK-RBAKDN", "RBM14-RBM4", "RPL17-C18orf32", "RPS10-NUDT3",
    "SNHG32", "ST20-MTHFS", "TMEM99", "ZBED6CL",
})


PERT_ANN_DIM = 16
_STRENGTH = {"弱": 0.0, "中": 0.5, "强": 1.0}
_FLAG_COLS = (
    "GO_BP", "GO_MF", "GO_CC", "Reactome", "CORUM", "Complex_Portal",
    "OmniPath", "TRRUST", "DoRothEA", "CollecTRI", "Pfam", "InterPro",
    "UniProt_功能",
)



def load_pert_annotation() -> dict[str, np.ndarray]:
    """扰动基因 → 紧凑注释向量（强度 / 位移 / 通路开关）。"""
    import pandas as pd

    if not PERT_ANN_PATH.exists():
        return {}
    df = pd.read_csv(PERT_ANN_PATH, sep="\t")
    out: dict[str, np.ndarray] = {}
    for gene, g in df.groupby(df.iloc[:, 0].astype(str)):
        strength = max(_STRENGTH.get(str(s), 0.0) for s in g["扰动强度"].astype(str))
        n_move = float(np.log1p(pd.to_numeric(g["n_move_0.10"], errors="coerce").fillna(0).max()))
        flags = []
        for col in _FLAG_COLS:
            if col not in g.columns:
                flags.append(0.0)
                continue
            flags.append(1.0 if g[col].astype(str).replace("nan", "").str.len().max() > 0 else 0.0)
        vec = np.array([strength, n_move, *flags], dtype=np.float32)
        if vec.size < PERT_ANN_DIM:
            vec = np.pad(vec, (0, PERT_ANN_DIM - vec.size))
        out[str(gene)] = vec[:PERT_ANN_DIM]
    return out


def pert_ann_tensor(table: dict[str, np.ndarray], name: str) -> np.ndarray:
    v = table.get(name)
    return v if v is not None else np.zeros(PERT_ANN_DIM, np.float32)



class Corpus:
    """All sources aligned onto one working gene space in the 18533 panel."""

    def __init__(self, device: str, lines: tuple[str, ...]):
        self.device = device
        raw = {n: dict(np.load(prep_line_npz(n), allow_pickle=True))
               for n in (SRC, *lines, VAL_LINE)}
        ctx = {c: dict(np.load(prep_ctx_npz(c))) for c in "ABC"}

        # working space W: genes a 2026 context expresses (control CPM > 5),
        # plus GWPS targets listed in W_12833.txt whose control CPM is at most 5
        w = np.zeros(18533, bool)
        for c in "ABC":
            w |= ctx[c]["universe"]
        self.W = np.flatnonzero(w)
        self.n_w = self.W.size
        w_pos = -np.ones(18533, np.int64)
        w_pos[self.W] = np.arange(self.n_w)

        # context-encoding space C: measured everywhere, so it transfers
        c_mask = w.copy()
        for n, d in raw.items():
            seen = np.zeros(18533, bool)
            seen[d["panel_idx"]] = True
            c_mask &= seen
        self.C = np.flatnonzero(c_mask)
        self.c_in_w = w_pos[self.C]
        if self.c_in_w.size and int(self.c_in_w.min()) < 0:
            raise RuntimeError("C is not a subset of W")
        self.ctx_topk = 0

        self.lines: dict[str, dict] = {}
        for name, d in raw.items():
            self.lines[name] = self._pack(d, w_pos)
        for c in "ABC":
            self.lines[f"ctx_{c}"] = self._pack_ctx(ctx[c], w_pos)

        # leave-one-context-out mean response prior。H1 是留出上下文：它的 mean_lfc
        # 不进训练 destination 的 prior，也不进 prior_all（2026 上下文和未覆盖靶点用）。
        # H1 自己的 prior 与旧版相同（旧版本来就排除了 h1 自身），旧 checkpoint 的 H1 分不变。
        self.prior = {}
        pool = [SRC, *lines]
        for name in (*pool, VAL_LINE):
            others = [self.lines[o]["mean_lfc"] for o in pool if o != name]
            self.prior[name] = np.mean(others, 0).astype(np.float32)
        self.prior_all = np.mean([self.lines[o]["mean_lfc"] for o in pool], 0).astype(np.float32)

        cx = np.stack([self.lines[n]["ctx_expr"] for n in self.lines])
        self.ctx_mu, self.ctx_sd = cx.mean(0), cx.std(0) + 1e-3
        self.esm = self._load_esm()
        self.go = self._load_go()
        self.pert_ann = load_pert_annotation()
        print(f"[corpus] W={self.n_w} genes, C={self.C.size} ctx genes, "
              f"lines={list(self.lines)}", flush=True)

    def _pack(self, d: dict, w_pos: np.ndarray) -> dict:
        pidx = d["panel_idx"].astype(np.int64)
        col = w_pos[pidx]
        keep = col >= 0
        col, src = col[keep], np.flatnonzero(keep)

        lfc = np.zeros((d["lfc"].shape[0], self.n_w), np.float32)
        z = np.zeros_like(lfc)
        sig = np.zeros(lfc.shape, bool)
        lfc[:, col] = shrink_lfc(d["lfc"][:, src], d["z"][:, src])
        z[:, col] = d["z"][:, src]
        sig[:, col] = d["sig"][:, src]

        ctrl = np.zeros(self.n_w, np.float32)
        univ = np.zeros(self.n_w, bool)
        has = np.zeros(self.n_w, bool)
        ctrl[col] = np.log1p(d["ctrl_cpm"][src])
        univ[col] = d["universe"][src]
        has[col] = True

        full = np.zeros(18533, np.float32)
        full[pidx] = np.log1p(d["ctrl_cpm"])
        return {
            "perts": d["perts"].astype(str), "lfc": lfc, "z": z, "sig": sig,
            "lfc_c": compress(torch.from_numpy(lfc)).numpy(),
            "ctrl": ctrl, "univ": univ, "has": has,
            "n_cells": d["n_cells"], "n_ctrl": int(d["n_ctrl"]),
            "mean_lfc": (lfc * univ).mean(0), "ctx_expr": full[self.C],
            "index": {p: i for i, p in enumerate(d["perts"].astype(str))},
        }

    def _pack_ctx(self, d: dict, w_pos: np.ndarray) -> dict:
        ctrl = np.zeros(self.n_w, np.float32)
        univ = np.zeros(self.n_w, bool)
        col = w_pos[np.arange(18533)]
        keep = col >= 0
        ctrl[col[keep]] = np.log1p(d["ctrl_cpm"][keep])
        univ[col[keep]] = d["universe"][keep]
        return {"ctrl": ctrl, "univ": univ, "has": np.ones(self.n_w, bool),
                "ctx_expr": np.log1p(d["ctrl_cpm"])[self.C],
                "ctrl_cpm_full": d["ctrl_cpm"], "mean_lfc": np.zeros(self.n_w, np.float32)}

    def _load_esm(self) -> dict[str, np.ndarray]:
        d = np.load(ESM_PATH, allow_pickle=True)
        sym = d["gene_symbol"].astype(str)
        emb = d["embeddings"].astype(np.float32)
        emb = (emb - emb.mean(0)) / (emb.std(0) + 1e-6)
        out: dict[str, np.ndarray] = {}
        for s, e in zip(sym, emb):
            out.setdefault(s, e)
        self.esm_dim = emb.shape[1]
        self.esm_zero = np.zeros(self.esm_dim, np.float32)
        return out

    def _load_go(self) -> dict[str, np.ndarray]:
        """扰动基因的 OPA2Vec。只在有标注的行上标准化，缺测是 64 个 0 再加 has_go=0。"""
        d = np.load(GO_PATH, allow_pickle=True)
        genes = d["genes"].astype(str)
        emb = d["opa2vec"].astype(np.float32)
        mask = d["opa2vec_mask"].astype(bool)
        present = emb[mask]
        emb = (emb - present.mean(0)) / (present.std(0) + 1e-6)
        emb[~mask] = 0.0
        out: dict[str, np.ndarray] = {}
        for s, e, m in zip(genes, emb, mask):
            if m:
                vec = np.empty(65, np.float32)
                vec[:64] = e
                vec[64] = 1.0
                out.setdefault(s, vec)
        self.go_zero = np.zeros(65, np.float32)
        return out

    def _src_topk_lfc(self, pert: str, k: int) -> np.ndarray:
        """C 上按源 |z| 取 top-k 的 shrink 源 lfc，降序排列，不足补 0。"""
        out = np.zeros(k, np.float32)
        i = self.lines[SRC]["index"].get(pert, -1)
        if i < 0:
            return out
        z = self.lines[SRC]["z"][i, self.c_in_w]
        lfc = self.lines[SRC]["lfc"][i, self.c_in_w]
        n = int(z.size)
        take = min(k, n)
        if take <= 0:
            return out
        if take < n:
            idx = np.argpartition(-np.abs(z), take - 1)[:take]
        else:
            idx = np.arange(n)
        idx = idx[np.argsort(-np.abs(z[idx]), kind="stable")]
        out[:take] = lfc[idx]
        return out

    # -------------------------------------------------------------- featurizer

    def features(self, pert: str, dest: str, prior_key: str | None = None):
        """Per-gene channels, context vector, ESM vector, scalars and prior."""
        s = self.lines[SRC]
        t = self.lines[dest]
        i = s["index"].get(pert, -1)
        lfc_s = s["lfc"][i] if i >= 0 else np.zeros(self.n_w, np.float32)
        z_s = s["z"][i] if i >= 0 else np.zeros(self.n_w, np.float32)
        sig_s = s["sig"][i] if i >= 0 else np.zeros(self.n_w, bool)
        has_s = s["has"] if i >= 0 else np.zeros(self.n_w, bool)

        prior = self.prior[prior_key if prior_key else dest] \
            if (prior_key or dest) in self.prior else self.prior_all
        univ = t["univ"]
        local = np.stack([
            lfc_s,
            np.tanh(z_s / 8.0),
            sig_s.astype(np.float32),
            has_s.astype(np.float32),
            t["ctrl"] / 5.0,
            s["ctrl"] / 5.0,
            (t["ctrl"] - s["ctrl"]) / 5.0,
            prior,
        ], -1).astype(np.float32)
        if self.ctx_topk:
            ctx = self._src_topk_lfc(pert, int(self.ctx_topk))
        else:
            ctx = (t["ctx_expr"] - self.ctx_mu) / self.ctx_sd
        esm = self.esm.get(pert, self.esm_zero)
        go = self.go.get(pert, self.go_zero)
        sc = build_scalars(sig_s, lfc_s, univ, t["ctrl"], int(univ.sum()))
        ann = pert_ann_tensor(self.pert_ann, pert)
        return local, ctx, esm, sc, prior, univ, ann, go

    def pairs(self, lines: tuple[str, ...]) -> list[tuple[str, str]]:
        src = set(self.lines[SRC]["perts"])
        out = []
        for ln in lines:
            for p in self.lines[ln]["perts"]:
                if p in src and p not in UNMAPPED_GWPS:
                    out.append((p, ln))
        return out


def batchify(corpus: Corpus, items: list[tuple[str, str]], device: str):
    loc, ctx, esm, sca, pri, uni, tgt_lfc, tgt_sig, pann, go = (
        [], [], [], [], [], [], [], [], [], [])
    for p, ln in items:
        l, c, e, s, pr, u, ann, g = corpus.features(p, ln)
        loc.append(l); ctx.append(c); esm.append(e); sca.append(s); pri.append(pr); uni.append(u)
        pann.append(ann); go.append(g)
        line = corpus.lines[ln]
        if "index" in line and p in line["index"]:
            j = line["index"][p]
            # the model predicts in the compressed amplitude space, so the target
            # has to live there too or the two are orders of magnitude apart
            tgt_lfc.append(line["lfc_c"][j])
            tgt_sig.append(line["sig"][j])
        else:                       # inference against a control-only context
            tgt_lfc.append(np.zeros(corpus.n_w, np.float32))
            tgt_sig.append(np.zeros(corpus.n_w, np.float32))

    def t(a):
        return torch.as_tensor(np.stack(a), dtype=torch.float32, device=device)

    return (t(loc), t(ctx), t(esm), t(sca), t(pri), t(uni), t(tgt_lfc), t(tgt_sig), t(pann), t(go))




def _target_panel_mask(corpus: Corpus) -> torch.Tensor:
    genes = pd.read_csv(DATA / "challenge_2026/gene_names.csv")["gene_name"].astype(str).to_numpy()
    targets = set(pd.read_csv(DATA / "challenge_2026/pert_counts.csv")["target_gene"].astype(str))
    is_t = np.array([g in targets for g in genes])
    w_pos = -np.ones(18533, np.int64)
    w_pos[corpus.W] = np.arange(corpus.n_w)
    pos = w_pos[np.flatnonzero(is_t)]
    return torch.as_tensor(pos[pos >= 0], dtype=torch.long)


def w_index(corpus) -> dict[str, int]:
    cached = getattr(corpus, "w_of", None)
    if cached is not None:
        return cached
    genes = pd.read_csv(DATA / "challenge_2026/gene_names.csv")["gene_name"].astype(str).to_numpy()
    corpus.w_of = {g: k for k, g in enumerate(genes[corpus.W])}
    return corpus.w_of


def own_target_mask(w_of: dict[str, int], items, like: torch.Tensor) -> torch.Tensor:
    own = torch.zeros_like(like)
    rows, cols = [], []
    for r, (p, _) in enumerate(items):
        j = w_of.get(p)
        if j is not None:
            rows.append(r)
            cols.append(j)
    if rows:
        own[torch.as_tensor(rows, device=like.device),
            torch.as_tensor(cols, device=like.device)] = 1.0
    return own


def _shard(n_items: int, rank: int, world: int, batch: int, seed: int, epoch: int):
    rng = np.random.default_rng(seed + epoch)
    order = rng.permutation(n_items)
    n = (len(order) // world) * world
    order = order[:n]
    mine = order[rank::world]
    n_b = (len(mine) // batch) * batch
    return mine[:n_b]


def _warmup_cosine(optimizer, warmup_steps: int, total_steps: int):
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        t = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        t = min(1.0, max(0.0, t))
        return 0.5 * (1.0 + math.cos(math.pi * t))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _is_aligned() -> bool:
    return isinstance(loss_fn, TransferAlignedLoss)


def train_loop(rank: int, world: int, gpu: int) -> Path:
    use_cuda = torch.cuda.is_available()
    device = f"cuda:{gpu}" if use_cuda else "cpu"
    if use_cuda:
        torch.cuda.set_device(gpu)
    torch.manual_seed(0)
    np.random.seed(0)
    ckpt = ROOT / f"{tag}.pt"

    def log(msg: str) -> None:
        if rank == 0:
            print(f"[{tag}] {msg}", flush=True)

    corpus = Corpus(device, TRAIN_LINES)
    items = corpus.pairs(TRAIN_LINES)
    n_ctx = int(corpus.ctx_topk) if corpus.ctx_topk else int(corpus.C.size)
    net = model.build(corpus.n_w, n_ctx, corpus.esm_dim).to(device)
    if world > 1:
        net = DDP(net, device_ids=[gpu] if use_cuda else None)
    raw = net.module if world > 1 else net
    log(f"pairs={len(items)} params={n_params(raw)} device={device} world={world}")

    loss = loss_fn.to(device)
    emb = [p for n, p in raw.named_parameters() if n.startswith("gene_emb.")]
    rest = [p for n, p in raw.named_parameters() if not n.startswith("gene_emb.")]
    groups = [{"params": rest, "weight_decay": 0.03}]
    if emb:
        groups.append({"params": emb, "weight_decay": 0.1})
    opt = torch.optim.AdamW(groups, lr=model.lr)
    batch = model.batch_size
    steps = max(1, len(_shard(len(items), rank, world, batch, 0, 1)) // batch)
    sched = _warmup_cosine(opt, steps, max(1, epochs) * steps)

    pds_mask = torch.ones(corpus.n_w, device=device)
    pds_mask[_target_panel_mask(corpus).to(device)] = 0.0
    w_of = w_index(corpus)
    aligned = _is_aligned()
    best, best_ep = -1e9, 0

    def evaluate(ep: int) -> None:
        nonlocal best, best_ep
        if rank != 0:
            return
        from eval import Validator
        net.eval()
        result = Validator(corpus, device, 0).run(raw)
        net.train()
        score = float(result["score_avg"])
        log(f"ep {ep} score_avg {score:+.4f} pds {result['raw_pds']:.3f} "
            f"fid {result['raw_fid']:.3f} nmae {result['raw_nmae']:.3f}")
        if score > best:
            best, best_ep = score, ep
            torch.save({
                "model": raw.state_dict(),
                "args": {"kind": model.kind, "lr": model.lr, "batch_size": model.batch_size,
                         "n_genes": corpus.n_w, "n_ctx": n_ctx, "esm_dim": corpus.esm_dim,
                         **model.kw},
                "epoch": ep, "score": score, "val": result,
            }, ckpt)
            log(f"saved {ckpt}")

    for ep in range(1, epochs + 1):
        net.train()
        order = _shard(len(items), rank, world, batch, 0, ep)
        run_loss, seen = 0.0, 0
        for b in range(0, len(order), batch):
            sel = [items[i] for i in order[b:b + batch]]
            loc, ctx, esm, sca, pri, uni, t_lfc, t_sig, pann, go = batchify(corpus, sel, device)
            uni = uni * (1.0 - own_target_mask(w_of, sel, uni))
            if aligned:
                lfc, sig_logit, gate, base = net(
                    loc, ctx, esm, sca, pri, pann, go=go, return_base=True)
                loss_v, _parts = loss(lfc, sig_logit, t_lfc, t_sig, uni, pds_mask, base)
            else:
                lfc, sig_logit, gate = net(loc, ctx, esm, sca, pri, pann, go=go)
                loss_v, _parts = loss(lfc, sig_logit, t_lfc, t_sig, uni, pds_mask)
            opt.zero_grad(set_to_none=True)
            loss_v.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
            seen += len(sel)
            run_loss += float(loss_v.detach()) * len(sel)
        log(f"ep {ep} loss {run_loss / max(seen, 1):.4f}")
        if world > 1:
            dist.barrier()
        evaluate(ep)
        if world > 1:
            dist.barrier()
            flag = torch.tensor([1 if (ep - best_ep) >= patience else 0], device=device)
            dist.broadcast(flag, src=0)
            stop = bool(flag.item())
        else:
            stop = (ep - best_ep) >= patience
        if stop:
            log(f"early stop at {ep}, best {best:+.4f} @ {best_ep}")
            break
    return ckpt


def _entry(local_rank: int, world: int, gpu_ids: tuple[int, ...]) -> None:
    gpu = int(gpu_ids[local_rank])
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "29517")
    dist.init_process_group("nccl", rank=local_rank, world_size=world)
    try:
        train_loop(local_rank, world, gpu)
    finally:
        dist.destroy_process_group()


def main() -> None:
    ids = tuple(int(g) for g in gpus)
    if len(ids) <= 1:
        gpu = ids[0] if ids else 0
        train_loop(0, 1, gpu)
        return
    torch.multiprocessing.spawn(_entry, args=(len(ids), ids), nprocs=len(ids), join=True)


if __name__ == "__main__":
    main()
