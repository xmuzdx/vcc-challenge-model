#!/usr/bin/env python3
"""Extract per-(cell_line, perturbation) DE tables from the CRISPRi corpora.

One npz per source into model/prep/, every gene axis indexed into the official
18,533-gene panel (data/challenge_2026/gene_names.csv).

The DE contract of cell-eval2 vcc2026 (rule_version 3) is reproduced exactly:
  * ranked / averaged values are per-cell CPM (each cell normalized to 1e6)
  * lfc = log2((mean_cpm_pert + 1e-9) / (mean_cpm_ctrl + 1e-9))
  * two-sided Wilcoxon rank-sum, normal approximation, zero-block tie correction
  * gene universe = mean control CPM > 5
  * Benjamini-Hochberg within each perturbation, alpha = 0.05

Mann-Whitney U decomposes per cell,
    U = sum_{v in pert} [ #{ctrl < v} + #{ctrl == v} / 2 ],
so U accumulates over a single sequential pass and never needs the perturbation
cells held together.  The bracket is read off Q pre-sorted control quantiles per
gene via a batched torch.searchsorted.

Usage:
    python data/preprocess.py --sources gwps                 # the 65 GB one, run first
    python data/preprocess.py --sources rpe1 hepg2 jurkat k562ess
    python data/preprocess.py --sources h1 contexts
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np
import scipy.sparse as sp
import torch

from paths import DATA, PREP as OUT

CTRL_LABELS = {"non-targeting", "non_targeting", "nontargeting", "nt", "control", "ctrl"}
N_QUANT = 4096          # control quantiles per gene; rank error <= n_ctrl/(2*Q)
CPM_MIN = 5.0           # official low-expression filter, on control mean CPM
ALPHA = 0.05
EPS_LFC = 1e-9          # official epsilon in the fold-change ratio
BLOCK = 40_000          # cells per streaming block

# 2026 evaluation geometry; H1 is downsampled to match so anchors are comparable.
CELLS_PER_PERT = 400
CTRL_CELLS = 18_400
TARGET_MEDIAN_UMI = 20_000

SOURCES = {
    "gwps":    (DATA / "public/replogle_2022/K562_gwps_raw_singlecell_01.h5ad", "gene"),
    "k562ess": (DATA / "public/replogle_2022/K562_essential_raw_singlecell_01.h5ad", "gene"),
    "rpe1":    (DATA / "public/replogle_2022/rpe1_raw_singlecell_01.h5ad", "gene"),
    "hepg2":   (DATA / "public/nadig_2025/GSE264667_hepg2_raw_singlecell_01.h5ad", "gene"),
    "jurkat":  (DATA / "public/nadig_2025/GSE264667_jurkat_raw_singlecell_01.h5ad", "gene"),
}
H1_FILES = [DATA / f"challenge_2025/adata_{s}.h5ad" for s in ("Training", "Validation", "Test")]
CONTEXTS = {c: DATA / f"challenge_2026/context_{c}.h5ad" for c in ("A", "B", "C")}


# --------------------------------------------------------------------------- io

def _to_str(a) -> np.ndarray:
    return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in a])


def _column(grp: h5py.Group, col: str) -> np.ndarray:
    """String column from obs/var, tolerating both anndata categorical layouts."""
    if col not in grp:
        raise KeyError(col)
    node = grp[col]
    if isinstance(node, h5py.Group):
        if "categories" in node:                       # categorical
            return _to_str(node["categories"][:])[node["codes"][:]]
        if "values" in node:                           # nullable-string-array
            return _to_str(node["values"][:])
        raise TypeError(f"unsupported encoding for {col}: {list(node.keys())}")
    if "__categories" in grp and col in grp["__categories"]:
        return _to_str(grp["__categories"][col][:])[node[:]]
    return _to_str(node[:])


def _index(grp: h5py.Group) -> np.ndarray:
    key = grp.attrs.get("_index", grp.attrs.get("index", "_index"))
    return _column(grp, str(key.decode() if isinstance(key, bytes) else key))


def _gene_symbols(f: h5py.File) -> np.ndarray:
    var = f["var"]
    for col in ("gene_name", "gene_symbol"):
        try:
            return _column(var, col)
        except KeyError:
            continue
    return _index(var)


def _shape(f: h5py.File) -> tuple[int, int]:
    X = f["X"]
    if isinstance(X, h5py.Dataset):
        return X.shape
    return tuple(int(v) for v in X.attrs["shape"])


def _read_block(f: h5py.File, lo: int, hi: int, n_gene: int) -> np.ndarray:
    """Dense float32 block [lo, hi) regardless of on-disk density."""
    X = f["X"]
    if isinstance(X, h5py.Dataset):
        return np.asarray(X[lo:hi], dtype=np.float32)
    ptr = X["indptr"][lo : hi + 1]
    a, b = int(ptr[0]), int(ptr[-1])
    m = sp.csr_matrix(
        (X["data"][a:b].astype(np.float32), X["indices"][a:b], ptr - ptr[0]),
        shape=(hi - lo, n_gene),
    )
    return m.toarray()


def _cpm(block: np.ndarray) -> np.ndarray:
    lib = block.sum(1, keepdims=True)
    np.maximum(lib, 1.0, out=lib)
    return block * (1e6 / lib)


# ------------------------------------------------------------------ statistics

def _bh_sig(p: np.ndarray, universe: np.ndarray, alpha: float = ALPHA) -> np.ndarray:
    """Benjamini-Hochberg per row, restricted to `universe` columns."""
    n_pert = p.shape[0]
    m = int(universe.sum())
    sig = np.zeros(p.shape, dtype=bool)
    if m == 0:
        return sig
    cols = np.flatnonzero(universe)
    sub = p[:, cols]
    order = np.argsort(sub, axis=1, kind="stable")
    ranked = np.take_along_axis(sub, order, axis=1)
    thresh = ranked * m / np.arange(1, m + 1)[None, :]
    # step-up: a hypothesis passes if any larger rank passes
    passes = np.minimum.accumulate(thresh[:, ::-1], axis=1)[:, ::-1] < alpha
    hit = np.zeros_like(passes)
    np.put_along_axis(hit, order, passes, axis=1)
    sig[np.arange(n_pert)[:, None], cols[None, :]] = hit
    return sig


def _wilcoxon_p(u: np.ndarray, m: np.ndarray, n_ctrl: int, n_zero: np.ndarray) -> np.ndarray:
    """Two-sided normal-approximation p-values with a zero-block tie correction."""
    m = m[:, None].astype(np.float64)
    total = m + n_ctrl
    tie = (n_zero.astype(np.float64) ** 3 - n_zero) / np.maximum(total * (total - 1.0), 1.0)
    var = m * n_ctrl / 12.0 * ((total + 1.0) - tie)
    z = (u - m * n_ctrl / 2.0) / np.sqrt(np.maximum(var, 1e-12))
    p = torch.erfc(torch.from_numpy(np.abs(z) / np.sqrt(2.0))).numpy()
    return np.clip(p, 1e-300, 1.0), z


# ------------------------------------------------------------------ panel maps

def load_panel() -> tuple[np.ndarray, dict[str, int]]:
    import pandas as pd

    genes = pd.read_csv(DATA / "challenge_2026/gene_names.csv")["gene_name"].astype(str).to_numpy()
    return genes, {g: i for i, g in enumerate(genes)}


def _panel_map(symbols: np.ndarray, lut: dict[str, int]) -> tuple[np.ndarray, np.ndarray]:
    """Columns of this source that live in the panel, and their panel indices."""
    src, dst, seen = [], [], set()
    for j, s in enumerate(symbols):
        i = lut.get(s)
        if i is not None and i not in seen:
            seen.add(i)
            src.append(j)
            dst.append(i)
    return np.asarray(src, np.int64), np.asarray(dst, np.int32)


# ------------------------------------------------------------------- main pass

def _control_quantiles(
    f: h5py.File, ctrl_rows: np.ndarray, n_gene: int, cols: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Control mean CPM, per-gene sorted quantiles, and per-gene zero counts."""
    n_c = ctrl_rows.size
    buf = np.empty((n_c, cols.size), dtype=np.float32)
    at = 0
    for lo in range(0, n_c, BLOCK):
        rows = ctrl_rows[lo : lo + BLOCK]
        blk = _read_block(f, int(rows[0]), int(rows[-1]) + 1, n_gene)
        blk = blk[rows - int(rows[0])]
        buf[at : at + rows.size] = _cpm(blk)[:, cols]
        at += rows.size
    mean_cpm = buf.mean(0, dtype=np.float64)
    n_zero = (buf == 0).sum(0).astype(np.int64)
    buf.sort(axis=0)
    take = np.linspace(0, n_c - 1, N_QUANT).round().astype(np.int64)
    quant = np.ascontiguousarray(buf[take].T)          # (G, Q), sorted along Q
    del buf
    return mean_cpm, quant, n_zero


def prep_source(name: str, path: Path, pert_col: str, lut: dict[str, int]) -> None:
    t0 = time.time()
    with h5py.File(path, "r") as f:
        n_obs, n_gene = _shape(f)
        labels = _column(f["obs"], pert_col)
        cols, panel_idx = _panel_map(_gene_symbols(f), lut)
        g = cols.size

        is_ctrl = np.array([str(x).lower() in CTRL_LABELS for x in labels])
        ctrl_rows = np.flatnonzero(is_ctrl)
        perts = np.array(sorted(set(labels[~is_ctrl])))
        code = {p: i for i, p in enumerate(perts)}
        pcode = np.full(n_obs, -1, np.int64)
        pcode[~is_ctrl] = [code[x] for x in labels[~is_ctrl]]
        print(f"[{name}] {n_obs} cells, {g}/{n_gene} genes in panel, "
              f"{perts.size} perts, {ctrl_rows.size} control", flush=True)
        assert ctrl_rows.size > 0 and perts.size > 0

        ctrl_cpm, quant, ctrl_zero = _control_quantiles(f, ctrl_rows, n_gene, cols)
        print(f"[{name}] control quantiles in {time.time() - t0:.0f}s", flush=True)

        qt = torch.from_numpy(quant)
        p_sum = torch.zeros((perts.size, g), dtype=torch.float64)
        p_u = torch.zeros((perts.size, g), dtype=torch.float64)
        p_zero = torch.zeros((perts.size, g), dtype=torch.float64)
        p_n = torch.zeros(perts.size, dtype=torch.float64)
        n_c = float(ctrl_rows.size)
        seen = 0

        for lo in range(0, n_obs, BLOCK):
            hi = min(lo + BLOCK, n_obs)
            keep = pcode[lo:hi] >= 0
            if not keep.any():
                continue
            cpm = torch.from_numpy(_cpm(_read_block(f, lo, hi, n_gene))[keep][:, cols])
            idx = torch.from_numpy(pcode[lo:hi][keep])
            # (#{ctrl<v} + #{ctrl==v}/2) read off the control quantiles, per gene
            vt = cpm.T.contiguous()                                   # (G, B)
            r = (torch.searchsorted(qt, vt, right=False)
                 + torch.searchsorted(qt, vt, right=True)).to(torch.float64)
            r *= n_c / (2.0 * (N_QUANT - 1))
            p_u.index_add_(0, idx, r.T)
            p_sum.index_add_(0, idx, cpm.to(torch.float64))
            p_zero.index_add_(0, idx, (cpm == 0).to(torch.float64))
            p_n.index_add_(0, idx, torch.ones(idx.numel(), dtype=torch.float64))
            seen += int(keep.sum())
            if (lo // BLOCK) % 10 == 0:
                print(f"[{name}] {hi}/{n_obs} cells  {time.time() - t0:.0f}s", flush=True)

    n_cells = p_n.numpy()
    assert seen == int(n_cells.sum()) == int((pcode >= 0).sum()), "cells dropped"
    mean_cpm = (p_sum.numpy() / np.maximum(n_cells[:, None], 1.0))
    lfc = np.log2((mean_cpm + EPS_LFC) / (ctrl_cpm[None, :] + EPS_LFC)).astype(np.float32)

    universe = ctrl_cpm > CPM_MIN
    n_zero_tot = p_zero.numpy() + ctrl_zero[None, :]
    pval, z = _wilcoxon_p(p_u.numpy(), n_cells, ctrl_rows.size, n_zero_tot)
    sig = _bh_sig(pval, universe) & universe[None, :]

    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT / f"{name}.npz",
        perts=perts, panel_idx=panel_idx, lfc=lfc, z=z.astype(np.float32),
        sig=sig, n_cells=n_cells.astype(np.int32), ctrl_cpm=ctrl_cpm.astype(np.float32),
        universe=universe, n_ctrl=np.int64(ctrl_rows.size),
    )
    print(f"[{name}] done in {time.time() - t0:.0f}s | universe={universe.sum()} "
          f"| median n_sig={np.median(sig.sum(1)):.0f} "
          f"| median |lfc| on sig={np.median(np.abs(lfc[sig])) if sig.any() else 0:.3f}",
          flush=True)


# ------------------------------------------------------------------- H1 as ctx

def prep_h1(lut: dict[str, int], skip_de: bool = False) -> None:
    """H1 across all three 2025 splits, treated as one held-out context.

    Also writes a single-cell evaluation bundle downsampled to the 2026 geometry
    (400 cells per perturbation, 18400 control cells, median 20k UMI) so the
    official six can be reproduced locally on real data.
    """
    t0 = time.time()
    rng = np.random.default_rng(0)
    handles = [h5py.File(p, "r") for p in H1_FILES]
    try:
        cols, panel_idx = _panel_map(_gene_symbols(handles[0]), lut)
        g = cols.size
        n_gene = _shape(handles[0])[1]
        for f in handles[1:]:
            assert _shape(f)[1] == n_gene

        labels, owner, row = [], [], []
        for k, f in enumerate(handles):
            lab = _column(f["obs"], "target_gene")
            labels.append(lab)
            owner.append(np.full(lab.size, k, np.int64))
            row.append(np.arange(lab.size, dtype=np.int64))
        labels = np.concatenate(labels)
        owner = np.concatenate(owner)
        row = np.concatenate(row)

        is_ctrl = np.array([str(x).lower() in CTRL_LABELS for x in labels])
        perts = np.array(sorted(set(labels[~is_ctrl])))
        print(f"[h1] {labels.size} cells, {g}/{n_gene} genes in panel, "
              f"{perts.size} perts, {int(is_ctrl.sum())} control", flush=True)

        ctrl_sel = np.flatnonzero(is_ctrl)
        if skip_de:
            d = np.load(OUT / "h1.npz", allow_pickle=True)
            assert (d["perts"].astype(str) == perts).all()
            universe, n_cells = d["universe"], d["n_cells"].astype(np.float64)
            print(f"[h1] reusing DE table | universe={universe.sum()}", flush=True)
            _h1_eval_bundle(handles, labels, owner, row, perts, n_cells, ctrl_sel,
                            universe, cols, panel_idx, n_gene, rng, t0)
            return

        # ---- full-depth DE table (all cells, matching the public-corpus format)
        ctrl_cpm, quant, ctrl_zero = _h1_control(handles, owner, row, ctrl_sel, n_gene, cols)
        qt = torch.from_numpy(quant)
        p_sum = torch.zeros((perts.size, g), dtype=torch.float64)
        p_u = torch.zeros((perts.size, g), dtype=torch.float64)
        p_zero = torch.zeros((perts.size, g), dtype=torch.float64)
        p_n = torch.zeros(perts.size, dtype=torch.float64)
        code = {p: i for i, p in enumerate(perts)}
        n_c = float(ctrl_sel.size)

        for k, f in enumerate(handles):
            lab = _column(f["obs"], "target_gene")
            pc = np.array([code.get(str(x), -1) for x in lab], np.int64)
            n_obs = lab.size
            for lo in range(0, n_obs, BLOCK):
                hi = min(lo + BLOCK, n_obs)
                keep = pc[lo:hi] >= 0
                if not keep.any():
                    continue
                cpm = torch.from_numpy(_cpm(_read_block(f, lo, hi, n_gene))[keep][:, cols])
                idx = torch.from_numpy(pc[lo:hi][keep])
                vt = cpm.T.contiguous()
                r = (torch.searchsorted(qt, vt, right=False)
                     + torch.searchsorted(qt, vt, right=True)).to(torch.float64)
                r *= n_c / (2.0 * (N_QUANT - 1))
                p_u.index_add_(0, idx, r.T)
                p_sum.index_add_(0, idx, cpm.to(torch.float64))
                p_zero.index_add_(0, idx, (cpm == 0).to(torch.float64))
                p_n.index_add_(0, idx, torch.ones(idx.numel(), dtype=torch.float64))
            print(f"[h1] split {k} scanned  {time.time() - t0:.0f}s", flush=True)

        n_cells = p_n.numpy()
        mean_cpm = p_sum.numpy() / np.maximum(n_cells[:, None], 1.0)
        lfc = np.log2((mean_cpm + EPS_LFC) / (ctrl_cpm[None, :] + EPS_LFC)).astype(np.float32)
        universe = ctrl_cpm > CPM_MIN
        pval, z = _wilcoxon_p(p_u.numpy(), n_cells, ctrl_sel.size,
                              p_zero.numpy() + ctrl_zero[None, :])
        sig = _bh_sig(pval, universe) & universe[None, :]
        OUT.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            OUT / "h1.npz",
            perts=perts, panel_idx=panel_idx, lfc=lfc, z=z.astype(np.float32),
            sig=sig, n_cells=n_cells.astype(np.int32), ctrl_cpm=ctrl_cpm.astype(np.float32),
            universe=universe, n_ctrl=np.int64(ctrl_sel.size),
        )
        print(f"[h1] DE table done {time.time() - t0:.0f}s | universe={universe.sum()} "
              f"| median n_sig={np.median(sig.sum(1)):.0f}", flush=True)

        _h1_eval_bundle(handles, labels, owner, row, perts, n_cells, ctrl_sel,
                        universe, cols, panel_idx, n_gene, rng, t0)
    finally:
        for f in handles:
            f.close()


def _h1_eval_bundle(handles, labels, owner, row, perts, n_cells, ctrl_sel,
                    universe, cols, panel_idx, n_gene, rng, t0) -> None:
    """Single-cell H1 bundle reshaped to the 2026 evaluation geometry."""
    keep_pert = perts[n_cells >= CELLS_PER_PERT]
    print(f"[h1] eval bundle: {keep_pert.size}/{perts.size} perts with "
          f">={CELLS_PER_PERT} cells", flush=True)
    univ_cols = cols[universe]
    univ_panel = panel_idx[universe]

    # Pick every wanted cell first, then fill it in one sequential sweep per
    # file.  Reading a contiguous span per perturbation instead would touch most
    # of the file 255 times over, since a perturbation's cells are scattered.
    pert_of, want = [], []
    for p in keep_pert:
        cand = np.flatnonzero(labels == p)
        for gi in rng.choice(cand, CELLS_PER_PERT, replace=False):
            want.append((owner[gi], row[gi], len(want)))
        pert_of.append(np.full(CELLS_PER_PERT, p))
    n_pert_rows = len(want)
    for gi in rng.choice(ctrl_sel, min(CTRL_CELLS, ctrl_sel.size), replace=False):
        want.append((owner[gi], row[gi], len(want)))
    want = np.asarray(want, np.int64)

    buf = np.zeros((want.shape[0], univ_cols.size), np.float32)
    for k, f in enumerate(handles):
        mine = want[want[:, 0] == k]
        if mine.size == 0:
            continue
        mine = mine[np.argsort(mine[:, 1])]
        n_obs_k = _shape(f)[0]
        at = 0
        for lo in range(0, n_obs_k, BLOCK):
            hi = min(lo + BLOCK, n_obs_k)
            end = at + int(np.searchsorted(mine[at:, 1], hi))
            if end == at:
                continue
            blk = _read_block(f, lo, hi, n_gene)
            sub = mine[at:end]
            buf[sub[:, 2]] = blk[sub[:, 1] - lo][:, univ_cols]
            at = end
        assert at == mine.shape[0], f"split {k}: {at}/{mine.shape[0]} rows filled"
        print(f"[h1] eval sweep split {k} done  {time.time() - t0:.0f}s", flush=True)

    pert_mat = buf[:n_pert_rows]
    ctrl_mat = buf[n_pert_rows:]
    med = float(np.median(np.concatenate([pert_mat.sum(1), ctrl_mat.sum(1)])))
    keep_p = min(1.0, TARGET_MEDIAN_UMI / max(med, 1.0))
    print(f"[h1] median UMI {med:.0f} -> downsample p={keep_p:.4f}", flush=True)
    pert_mat = rng.binomial(pert_mat.astype(np.int64), keep_p).astype(np.int32)
    ctrl_mat = rng.binomial(ctrl_mat.astype(np.int64), keep_p).astype(np.int32)

    with h5py.File(OUT / "h1_eval.h5", "w") as out:
        for nm, mat in (("pert", pert_mat), ("ctrl", ctrl_mat)):
            csr = sp.csr_matrix(mat)
            grp = out.create_group(nm)
            grp.create_dataset("data", data=csr.data.astype(np.int32), compression="lzf")
            grp.create_dataset("indices", data=csr.indices.astype(np.int32), compression="lzf")
            grp.create_dataset("indptr", data=csr.indptr.astype(np.int64))
            grp.attrs["shape"] = csr.shape
        out.create_dataset("pert_of", data=np.concatenate(pert_of).astype("S32"))
        out.create_dataset("perts", data=keep_pert.astype("S32"))
        out.create_dataset("panel_idx", data=univ_panel.astype(np.int32))
        out.attrs["downsample_p"] = keep_p
        out.attrs["median_umi_before"] = med
    print(f"[h1] eval bundle written {time.time() - t0:.0f}s "
          f"| pert {pert_mat.shape} ctrl {ctrl_mat.shape}", flush=True)


def _h1_control(handles, owner, row, ctrl_sel, n_gene, cols):
    buf = np.empty((ctrl_sel.size, cols.size), np.float32)
    at = 0
    for k, f in enumerate(handles):
        mine = ctrl_sel[owner[ctrl_sel] == k]
        if mine.size == 0:
            continue
        rr = np.sort(row[mine])
        for lo in range(0, rr.size, BLOCK):
            rs = rr[lo : lo + BLOCK]
            blk = _read_block(f, int(rs[0]), int(rs[-1]) + 1, n_gene)
            buf[at : at + rs.size] = _cpm(blk[rs - int(rs[0])])[:, cols]
            at += rs.size
    assert at == ctrl_sel.size
    mean_cpm = buf.mean(0, dtype=np.float64)
    n_zero = (buf == 0).sum(0).astype(np.int64)
    buf.sort(axis=0)
    take = np.linspace(0, ctrl_sel.size - 1, N_QUANT).round().astype(np.int64)
    quant = np.ascontiguousarray(buf[take].T)
    return mean_cpm, quant, n_zero


# ------------------------------------------------------------------- contexts

def prep_contexts(panel: np.ndarray, lut: dict[str, int]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    meta = {}
    for ctx, path in CONTEXTS.items():
        with h5py.File(path, "r") as f:
            n_obs, n_gene = _shape(f)
            assert n_gene == panel.size
            sym = _index(f["var"])
            assert (sym == panel).all(), f"context {ctx} gene axis differs from panel"
            tot = np.zeros(n_gene, np.float64)
            nz = np.zeros(n_gene, np.int64)
            for lo in range(0, n_obs, BLOCK):
                hi = min(lo + BLOCK, n_obs)
                cpm = _cpm(_read_block(f, lo, hi, n_gene))
                tot += cpm.sum(0, dtype=np.float64)
                nz += (cpm == 0).sum(0)
            ctrl_cpm = tot / n_obs
        universe = ctrl_cpm > CPM_MIN
        np.savez_compressed(
            OUT / f"ctx_{ctx}.npz",
            ctrl_cpm=ctrl_cpm.astype(np.float32), universe=universe,
            n_zero=nz, n_ctrl=np.int64(n_obs),
        )
        meta[ctx] = {"n_ctrl": int(n_obs), "universe": int(universe.sum())}
        print(f"[ctx {ctx}] n_ctrl={n_obs} universe={universe.sum()}", flush=True)
    (OUT / "contexts.json").write_text(json.dumps(meta, indent=2))


# ----------------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="+", required=True,
                    help="any of: " + " ".join(list(SOURCES) + ["h1", "h1_eval", "contexts"]))
    args = ap.parse_args()
    torch.set_num_threads(max(1, min(16, torch.get_num_threads())))
    panel, lut = load_panel()
    for name in args.sources:
        if name == "contexts":
            prep_contexts(panel, lut)
        elif name == "h1":
            prep_h1(lut)
        elif name == "h1_eval":
            prep_h1(lut, skip_de=True)
        else:
            path, col = SOURCES[name]
            prep_source(name, path, col, lut)


if __name__ == "__main__":
    main()
