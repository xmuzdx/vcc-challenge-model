#!/usr/bin/env python3
"""细胞子样本池：每个 (pert, line) 预存 HVG 上的扰动 / control 细胞。

prep/*.npz 只有 DE 表，MMD 需要真实细胞。这里按 lfc 方差挑 2000 个 HVG，
每个扰动水库抽样 32 个细胞、每个细胞系 4096 个 control，fp16 存盘。
约 1.2 GB，训练时可全量驻留。

    python data/cellpool.py --lines rpe1 hepg2 jurkat k562ess
    python data/cellpool.py --lines h1
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.preprocess import (  # noqa: E402
    CTRL_LABELS,
    SOURCES,
    _column,
    _cpm,
    _gene_symbols,
    _panel_map,
    _read_block,
    _shape,
    load_panel,
)
from paths import PREP  # noqa: E402

N_HVG = 2000
N_PERT_CELLS = 32
N_CTRL_CELLS = 4096
BLOCK = 20_000


def _hvg_from_npz(name: str, n_hvg: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    d = np.load(PREP / f"{name}.npz", allow_pickle=True)
    univ = d["universe"].astype(bool)
    lfc = d["lfc"]
    var = lfc[:, univ].var(0) if univ.any() else lfc.var(0)
    take = np.argsort(-var)[:n_hvg]
    cols = np.flatnonzero(univ)[take] if univ.any() else take
    return d["panel_idx"][cols], d["perts"].astype(str), cols


def _reservoir(path: Path, pert_col: str, perts: np.ndarray, src_cols: np.ndarray,
               n_pert: int, n_ctrl: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    code = {p: i for i, p in enumerate(perts)}
    pbuf = np.zeros((len(perts), n_pert, src_cols.size), np.float16)
    pseen = np.zeros(len(perts), np.int32)
    cbuf = np.zeros((n_ctrl, src_cols.size), np.float16)
    cseen = 0
    with h5py.File(path, "r") as f:
        n_obs, n_gene = _shape(f)
        labels = _column(f["obs"], pert_col)
        t0 = time.time()
        for lo in range(0, n_obs, BLOCK):
            hi = min(lo + BLOCK, n_obs)
            blk = _cpm(_read_block(f, lo, hi, n_gene))[:, src_cols]
            for k, lab in enumerate(labels[lo:hi]):
                s = str(lab)
                row = np.log1p(np.clip(blk[k], 0, 1e6)).astype(np.float16)
                if s.lower() in CTRL_LABELS:
                    cseen += 1
                    if cseen <= n_ctrl:
                        cbuf[cseen - 1] = row
                    else:
                        j = int(rng.integers(cseen))
                        if j < n_ctrl:
                            cbuf[j] = row
                    continue
                i = code.get(s)
                if i is None:
                    continue
                pseen[i] += 1
                n = int(pseen[i])
                if n <= n_pert:
                    pbuf[i, n - 1] = row
                else:
                    j = int(rng.integers(n))
                    if j < n_pert:
                        pbuf[i, j] = row
            if (lo // BLOCK) % 5 == 0:
                print(f"[cellpool] {path.name} {hi}/{n_obs} {time.time() - t0:.0f}s",
                      flush=True)
    return pbuf, cbuf, pseen, cseen


def build_line(name: str, n_hvg: int, n_pert: int, n_ctrl: int) -> Path:
    out = PREP / "cellpool"
    out.mkdir(parents=True, exist_ok=True)
    dest = out / f"{name}.npz"
    if dest.exists():
        print(f"[cellpool] skip existing {dest}", flush=True)
        return dest

    panel_hvg, perts, local_cols = _hvg_from_npz(name, n_hvg)
    if name == "h1":
        return _build_h1(dest, panel_hvg, perts, n_pert, n_ctrl)

    path, pert_col = SOURCES[name]
    _, lut = load_panel()
    with h5py.File(path, "r") as f:
        src, dst = _panel_map(_gene_symbols(f), lut)
    # local_cols indexes the source's panel-mapped gene axis (prep npz columns)
    src_cols = src[local_cols]
    print(f"[cellpool] {name} hvg={src_cols.size} perts={perts.size}", flush=True)
    pbuf, cbuf, pseen, cseen = _reservoir(path, pert_col, perts, src_cols, n_pert, n_ctrl)
    np.savez_compressed(
        dest, perts=perts, panel_idx=panel_hvg, pert_cells=pbuf, ctrl_cells=cbuf,
        n_seen=pseen, n_ctrl_seen=np.int32(cseen),
    )
    print(f"[cellpool] wrote {dest} pert={pbuf.shape} ctrl={cbuf.shape}", flush=True)
    return dest


def _build_h1(dest: Path, panel_hvg: np.ndarray, perts: np.ndarray,
              n_pert: int, n_ctrl: int) -> Path:
    import h5py
    from paths import PREP as P

    ev = P / "h1_eval.h5"
    with h5py.File(ev, "r") as f:
        def csr(name):
            import scipy.sparse as sp
            g = f[name]
            return sp.csr_matrix(
                (g["data"][:], g["indices"][:], g["indptr"][:]),
                shape=tuple(g.attrs["shape"]),
            )
        pert = csr("pert").toarray().astype(np.float32)
        ctrl = csr("ctrl").toarray().astype(np.float32)
        ev_perts = np.array([x.decode() if isinstance(x, bytes) else str(x)
                             for x in f["perts"][:]])
        pert_of = np.array([x.decode() if isinstance(x, bytes) else str(x)
                            for x in f["pert_of"][:]])
        panel_idx = f["panel_idx"][:]

    lut = {int(v): i for i, v in enumerate(panel_idx)}
    cols = np.array([lut.get(int(i), -1) for i in panel_hvg], np.int64)
    keep = cols >= 0
    cols, panel_hvg = cols[keep], panel_hvg[keep]
    rng = np.random.default_rng(0)
    pbuf = np.zeros((len(perts), n_pert, cols.size), np.float16)
    pseen = np.zeros(len(perts), np.int32)
    index = {p: i for i, p in enumerate(perts)}
    for p in ev_perts:
        i = index.get(p)
        if i is None:
            continue
        rows = pert[pert_of == p]
        take = rng.choice(len(rows), size=min(n_pert, len(rows)), replace=False)
        pbuf[i, : len(take)] = np.log1p(np.clip(rows[take][:, cols], 0, 1e6))
        pseen[i] = len(take)
    ctake = rng.choice(len(ctrl), size=min(n_ctrl, len(ctrl)), replace=False)
    cbuf = np.log1p(np.clip(ctrl[ctake][:, cols], 0, 1e6)).astype(np.float16)
    np.savez_compressed(
        dest, perts=perts, panel_idx=panel_hvg, pert_cells=pbuf, ctrl_cells=cbuf,
        n_seen=pseen, n_ctrl_seen=np.int32(len(ctake)),
    )
    print(f"[cellpool] wrote {dest} from h1_eval.h5", flush=True)
    return dest


def _as_log1p(a: np.ndarray) -> np.ndarray:
    x = np.nan_to_num(np.asarray(a, np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    pos = x[x > 0]
    if pos.size and float(np.median(pos)) > 5.0:
        x = np.log1p(np.clip(x, 0, 1e6))
    return x


class CellPool:
    """按 (pert, line) 取 HVG 细胞，映射到工作空间 W 的列。"""

    def __init__(self, lines: tuple[str, ...], W: np.ndarray, device: str):
        self.device = device
        self.W = np.asarray(W)
        w_pos = -np.ones(18533, np.int64)
        w_pos[self.W] = np.arange(self.W.size)
        self.lines: dict[str, dict] = {}
        for name in lines:
            path = PREP / "cellpool" / f"{name}.npz"
            if not path.exists():
                continue
            d = np.load(path, allow_pickle=True)
            pidx = d["panel_idx"].astype(np.int64)
            col = w_pos[pidx]
            keep = col >= 0
            pert = _as_log1p(d["pert_cells"][:, :, keep])
            ctrl = _as_log1p(d["ctrl_cells"][:, keep])
            self.lines[name] = {
                "perts": d["perts"].astype(str),
                "index": {p: i for i, p in enumerate(d["perts"].astype(str))},
                "col": col[keep],
                "pert_cells": pert,
                "ctrl_cells": ctrl,
                "n_hvg": int(keep.sum()),
            }
        print(f"[cellpool] loaded {list(self.lines)}", flush=True)

    def available(self, dest: str) -> bool:
        return dest in self.lines

    def sample(self, pert: str, dest: str, n_pred: int, n_real: int, rng):
        d = self.lines[dest]
        i = d["index"].get(pert, -1)
        ctrl = d["ctrl_cells"]
        cidx = rng.integers(0, ctrl.shape[0], size=n_pred)
        pred_ctrl = torch_from(ctrl[cidx], self.device)
        real = None
        if i >= 0:
            cells = d["pert_cells"][i]
            live = cells[cells.sum(1) > 0]
            if live.shape[0] > 0:
                ridx = rng.integers(0, live.shape[0], size=min(n_real, live.shape[0]))
                real = torch_from(live[ridx], self.device)
        return pred_ctrl, real, d["col"]


def torch_from(a: np.ndarray, device: str):
    import torch
    return torch.as_tensor(np.asarray(a, np.float32), device=device)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lines", nargs="+", default=["rpe1", "hepg2", "jurkat", "k562ess"])
    ap.add_argument("--n-hvg", type=int, default=N_HVG)
    ap.add_argument("--n-pert", type=int, default=N_PERT_CELLS)
    ap.add_argument("--n-ctrl", type=int, default=N_CTRL_CELLS)
    args = ap.parse_args()
    for name in args.lines:
        build_line(name, args.n_hvg, args.n_pert, args.n_ctrl)


if __name__ == "__main__":
    main()
