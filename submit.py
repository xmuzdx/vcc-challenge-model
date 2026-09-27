#!/usr/bin/env python3
"""写出 VCC 2026 的 h5ad。默认不上传；加上 --send 才调用 vcc submit。

    python submit.py
    python submit.py --calibrate
    python submit.py --send
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

import eval as M
from loss import CENTER_W, decenter
from model import local_response, pert_response
from train import DATA, ROOT, TRAIN_LINES, Corpus, batchify, tag as default_tag

CIS_LFC = float(np.log2(0.06))
CELLS = 400
MAX_STORED = 4_750_000_000
GRID_SCALE = (0.3, 0.5, 0.8, 1.1, 1.5)
GRID_TOPK = (50, 100, 170, 300, 600)


def _spec_from_args(args: dict):
    kind = args.get("kind", "pert")
    factory = local_response if kind == "local" else pert_response
    skip = {"kind", "lr", "batch_size", "n_genes", "n_ctx", "esm_dim"}
    return factory(
        lr=float(args.get("lr", 2e-3)),
        batch_size=int(args.get("batch_size", 24)),
        **{k: v for k, v in args.items() if k not in skip},
    )


def load_models(paths: list[Path], corpus: Corpus, device: str) -> list:
    models = []
    for path in paths:
        ck = torch.load(path, map_location=device, weights_only=False)
        args = ck["args"]
        spec = _spec_from_args(args)
        net = spec.build(int(args["n_genes"]), int(args["n_ctx"]), int(args["esm_dim"])).to(device)
        net.load_state_dict(ck["model"])
        net.eval()
        net._score = ck.get("score")
        net._src = str(path)
        models.append(net)
        print(f"[submit] loaded {path} epoch={ck.get('epoch')} score={ck.get('score')}", flush=True)
    return models


@torch.no_grad()
def predict(models, corpus: Corpus, perts: list[str], dest: str, device: str,
            chunk: int = 24, center_w: float = CENTER_W):
    lfc_out, sig_out = [], []
    for i in range(0, len(perts), chunk):
        sel = [(p, dest) for p in perts[i:i + chunk]]
        loc, ctx, esm, sca, pri, _, _, _, pann, go = batchify(corpus, sel, device)
        la = sa = None
        for net in models:
            lfc, sig, _ = net(loc, ctx, esm, sca, pri, pann, go=go)
            la = lfc if la is None else la + lfc
            sa = sig if sa is None else sa + sig
        lfc_out.append(la / len(models))
        sig_out.append(sa / len(models))
    return decenter(torch.cat(lfc_out), center_w), torch.cat(sig_out)


def calibrate(models, corpus: Corpus, device: str, seed: int = 0) -> dict:
    val = M.Validator(corpus, device, seed)
    pre = predict(models, corpus, val.perts, "h1", device, center_w=0.0)
    best = None
    rows = []
    for topk in GRID_TOPK:
        for scale in GRID_SCALE:
            result = val.run(models[0], scale=scale, topk=topk, pre=pre, center_w=CENTER_W)
            rec = {"scale": scale, "topk": topk, **{k: float(v) for k, v in result.items()}}
            rows.append(rec)
            print(f"[calib] topk={topk:4d} scale={scale:.2f} -> avg {result['score_avg']:+.4f}",
                  flush=True)
            if best is None or result["score_avg"] > best["score_avg"]:
                best = rec
    path = ROOT / f"{default_tag}_calib.json"
    path.write_text(json.dumps({"best": best, "grid": rows}, indent=2))
    print(f"[calib] best topk={best['topk']} scale={best['scale']} "
          f"score_avg={best['score_avg']:+.4f}", flush=True)
    return best


def _context_cells(ctx: str, device: str):
    import h5py
    with h5py.File(DATA / f"challenge_2026/context_{ctx}.h5ad", "r") as handle:
        x = handle["X"]
        n_obs, n_gene = (int(v) for v in x.attrs["shape"])
        mat = sp.csr_matrix((x["data"][:], x["indices"][:], x["indptr"][:]), shape=(n_obs, n_gene))
    cells = torch.as_tensor(mat.toarray().astype(np.float32), device=device)
    lib = cells.sum(1, keepdim=True).clamp_min(1.0)
    mean_cpm = (cells * (M.CPM / lib)).mean(0)
    return cells, mean_cpm


def _assert_contract(x: sp.csr_matrix, obs: pd.DataFrame, panel: np.ndarray, targets: list[str]) -> None:
    if x.shape != (len(targets) * CELLS * 3, panel.size):
        raise RuntimeError(f"shape {x.shape}")
    if x.dtype.kind not in "iu":
        raise RuntimeError(f"X must be integral, got {x.dtype}")
    if x.data.min() < 0:
        raise RuntimeError("negative counts")
    if not np.isfinite(x.data).all():
        raise RuntimeError("non-finite counts")
    total = np.asarray(x.sum(1)).ravel()
    if total.max() > 1_000_000:
        raise RuntimeError(f"cell total {total.max()} exceeds 1e6")
    if x.nnz > MAX_STORED:
        raise RuntimeError(f"{x.nnz} stored entries exceeds {MAX_STORED}")
    counts = obs.groupby(["context", "target_gene"], observed=True).size()
    if set(counts.to_numpy()) != {CELLS}:
        raise RuntimeError(f"cells per perturbation: {set(counts.to_numpy())}")
    if set(obs["context"].unique()) != {"A", "B", "C"}:
        raise RuntimeError("context set mismatch")
    for ctx in "ABC":
        got = set(obs.loc[obs["context"] == ctx, "target_gene"].unique())
        if got != set(targets):
            raise RuntimeError(f"context {ctx} perturbation set mismatch")
    if "non-targeting" in set(obs["target_gene"].astype(str)):
        raise RuntimeError("prediction contains control cells")
    print(f"[submit] contract ok: {x.shape} nnz={x.nnz}", flush=True)


def write_submission(models, corpus: Corpus, device: str, scale: float, topk: int,
                     tag: str, seed: int = 0) -> Path:
    panel = pd.read_csv(DATA / "challenge_2026/gene_names.csv")["gene_name"].astype(str).to_numpy()
    targets = pd.read_csv(DATA / "challenge_2026/pert_counts.csv")["target_gene"].astype(str).tolist()
    if len(targets) != 300 or len(set(targets)) != 300:
        raise RuntimeError("expected 300 unique targets")
    gene_pos = {g: i for i, g in enumerate(panel)}
    work = torch.as_tensor(corpus.W, device=device, dtype=torch.long)
    covered = set(corpus.lines["gwps"]["perts"])
    uncovered = [t for t in targets if t not in covered]
    prior = torch.as_tensor(corpus.prior_all, device=device)
    print(f"[submit] {len(uncovered)}/300 targets absent from the source screen", flush=True)

    blocks, obs_pert, obs_ctx = [], [], []
    t0 = time.time()
    missing = set(uncovered)
    for ctx in "ABC":
        cells, mean_cpm = _context_cells(ctx, device)
        lfc_w, sig_w = predict(models, corpus, targets, f"ctx_{ctx}", device)
        univ = torch.as_tensor(corpus.lines[f"ctx_{ctx}"]["univ"].astype(np.float32), device=device)
        gen = torch.Generator(device=device).manual_seed(seed + ord(ctx))
        pieces = []
        for i, tgt in enumerate(targets):
            value = (prior if tgt in missing else lfc_w[i]) * univ
            value = M.sparsify(value, sig_w[i], topk)
            full = torch.zeros(panel.size, device=device)
            full[work] = value
            j = gene_pos.get(tgt)
            if j is not None:
                full[j] = CIS_LFC if float(mean_cpm[j]) >= 5.0 else 0.0
            counts = M.synthesize_cells(full, cells, mean_cpm, CELLS, scale, gen)
            pieces.append(sp.csr_matrix(counts.to(torch.int32).cpu().numpy()))
            obs_pert.extend([tgt] * CELLS)
            obs_ctx.extend([ctx] * CELLS)
        blocks.append(sp.vstack(pieces, format="csr"))
        del cells
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"[submit] context {ctx} {blocks[-1].shape} ({time.time() - t0:.0f}s)", flush=True)

    matrix = sp.vstack(blocks, format="csr")
    obs = pd.DataFrame({
        "target_gene": pd.Categorical(obs_pert),
        "context": pd.Categorical(obs_ctx),
    })
    _assert_contract(matrix, obs, panel, targets)
    import anndata as ad
    adata = ad.AnnData(X=matrix, obs=obs, var=pd.DataFrame(index=pd.Index(panel, name=None)))
    path = ROOT / f"{tag}.h5ad"
    adata.write_h5ad(path, compression="gzip")
    meta = {"tag": tag, "scale": scale, "topk": topk, "seed": seed,
            "shape": list(matrix.shape), "nnz": int(matrix.nnz)}
    (ROOT / f"{tag}.json").write_text(json.dumps(meta, indent=2))
    print(f"[submit] wrote {path}", flush=True)
    return path


def _vcc_bin() -> Path | None:
    found = shutil.which("vcc")
    if found:
        return Path(found)
    for candidate in (
        Path("/media/data4T/lyw/project_1/envs/vcc/bin/vcc"),
        Path("/data/lyw/project_1/envs/vcc/bin/vcc"),
        Path("/data/lyw/project_2/envs/vcc/bin/vcc"),
    ):
        if candidate.is_file():
            return candidate
    return None


def send(h5ad: Path, tag: str) -> None:
    vcc = _vcc_bin()
    if vcc is None:
        raise RuntimeError("找不到 vcc 命令，h5ad 已写好但没有上传")
    genes = ROOT / "gene_names_headerless.csv"
    pd.read_csv(DATA / "challenge_2026/gene_names.csv")["gene_name"].to_csv(
        genes, index=False, header=False)
    packed = ROOT / f"{tag}.vcc"
    subprocess.run([
        str(vcc), "prep", str(h5ad), "-g", str(genes),
        "--perts", str(DATA / "challenge_2026/pert_counts.csv"),
        "-o", str(packed), "-f",
    ], check=True)
    subprocess.run([
        str(vcc), "submit", str(packed), "-m", tag,
        "-d", "perturbation response model",
        "--wait", "--wait-timeout", "7200", "--json",
    ], check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="写 VCC 2026 h5ad")
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--tag", default=default_tag)
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--topk", type=int, default=7000)
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--send", action="store_true")
    args = ap.parse_args()
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    corpus = Corpus(device, TRAIN_LINES)
    ckpt = Path(args.ckpt) if args.ckpt else ROOT / f"{args.tag}.pt"
    models = load_models([ckpt], corpus, device)
    scale, topk = args.scale, args.topk
    if args.calibrate:
        best = calibrate(models, corpus, device)
        scale, topk = best["scale"], best["topk"]
    path = write_submission(models, corpus, device, scale, topk, args.tag)
    if args.send:
        send(path, args.tag)


if __name__ == "__main__":
    main()
