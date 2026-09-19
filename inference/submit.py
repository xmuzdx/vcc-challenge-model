#!/usr/bin/env python3
"""Calibrate the output layer, then write the 2026 submission.

Two knobs sit between a predicted log2 fold change and a scored submission:

  topk   how many genes are allowed a non-zero fold change at all.  Because a
         gene left unchanged stays bit-identical to its carrier cell, this *is*
         the predicted call count, and fidelity is precision times coverage
         capped at 1 -- so calling past the reference's ~168 buys nothing while
         every extra call contributes a coin-flip direction.
  scale  multiplies the fold change.  For a squared error the optimum is the
         achieved cosine, and for NMAE any scale below 1 helps as long as the
         direction beats chance, so it cannot be left at 1 by assumption.

Both are fitted on H1 against the locally rebuilt official six -- the same
reference the training run selects on -- and not assumed.

Note the target gene itself is excluded from all six metrics (and PDS drops all
300 panel targets), so the cis knockdown below cannot earn any score.  It is
written for biological correctness only; every point comes from downstream genes.

    python run.py submit --calibrate --ckpt ensemble8
    python run.py submit --write --preset final
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

import metrics as M
from engine.train import Corpus, TRAIN_LINES, Validator, batchify
from models import PertResponseNet, get_model
from models.base import ModelParams
from inference.output import CENTER_W, decenter
from models.priors import panel_features_on
from models.programs import PROGRAM_K, load_programs
from paths import DATA, ROOT, SUBMISSIONS as OUT

CIS_LFC = np.log2(0.06)      # alpha = 0.94 knockdown, from results/vcc2026_ontarget
CELLS = 400
MAX_STORED = 4_750_000_000


def attach_priors(model, corpus: Corpus, device: str) -> None:
    if hasattr(model, "attach_gene_space"):
        model.attach_gene_space(panel_features_on(corpus.W).to(device))


def attach_programs(model, corpus: Corpus, device: str,
                    n_prog: int = PROGRAM_K) -> None:
    if hasattr(model, "attach_programs"):
        model.attach_programs(load_programs(corpus, k=n_prog).to(device))


def load_models(paths: list[Path], corpus: Corpus, device: str,
                model_name: str = "pert_response",
                n_prog: int = PROGRAM_K) -> list:
    models = []
    if len(paths) == 1 and str(paths[0]) == "untrained":
        # every head is zero-initialised, so this is exactly source transfer with
        # the calibrated amplitude compression -- the strongest configuration
        # measured on held-out H1, and a safe fallback if training adds nothing
        spec = get_model(model_name)
        m = spec.build(corpus.C.size, corpus.esm_dim).to(device)
        attach_priors(m, corpus, device)
        attach_programs(m, corpus, device, n_prog=n_prog)
        m.eval()
        m._score, m._src = None, "untrained"
        print(f"[submit] using the zero-residual {model_name} "
              "(= calibrated source transfer)", flush=True)
        return [m]
    for p in paths:
        ck = torch.load(p, map_location=device, weights_only=False)
        a = ck["args"]
        name = a.get("model", model_name)
        spec = get_model(name)
        extra = a.get("extra") or {}
        mp = ModelParams(hidden=a.get("hidden"), depth=a.get("depth"),
                         drop=a.get("drop"), cond=a.get("cond"), extra=extra)
        k = int(extra.get("n_prog", n_prog))
        m = spec.build(corpus.C.size, corpus.esm_dim, mp).to(device)
        attach_priors(m, corpus, device)
        attach_programs(m, corpus, device, n_prog=k)
        m.load_state_dict(ck["model"])
        m.eval()
        m._score = ck.get("score")
        m._src = str(p)
        models.append(m)
        print(f"[submit] loaded {p} (epoch {ck.get('epoch')}, score {ck.get('score')})",
              flush=True)
    return models


@torch.no_grad()
def predict(models, corpus: Corpus, perts: list[str], dest: str,
            device: str, chunk: int = 24,
            center_w: float = CENTER_W) -> tuple[torch.Tensor, torch.Tensor]:
    """Ensemble mean lfc and significance logit on the working gene space."""
    lfc_out, sig_out = [], []
    for i in range(0, len(perts), chunk):
        sel = [(p, dest) for p in perts[i : i + chunk]]
        loc, ctx, esm, sca, pri, _, _, _, pann = batchify(corpus, sel, device)
        la, sa = None, None
        for m in models:
            lfc, sig, _ = m(loc, ctx, esm, sca, pri, pann)
            la = lfc if la is None else la + lfc
            sa = sig if sa is None else sa + sig
        lfc_out.append(la / len(models))
        sig_out.append(sa / len(models))
    lfc_w = decenter(torch.cat(lfc_out), center_w)
    return lfc_w, torch.cat(sig_out)


# ------------------------------------------------------------------ calibration

GRID_SCALE = [0.3, 0.5, 0.8, 1.1, 1.5]
GRID_TOPK = [50, 100, 170, 300, 600]


def calibrate(models, corpus: Corpus, device: str, seed: int = 0,
              center_w: float = CENTER_W) -> dict:
    val = Validator(corpus, device, seed)
    # predict 不做 decenter；由 val.run 统一处理，避免二次去均值
    pre = predict(models, corpus, val.perts, "h1", device, center_w=0.0)
    ref_sig = val.panel.real_sig().sum(1).to(torch.float64)
    print(f"[calib] reference n_sig: median {float(ref_sig.median()):.0f} "
          f"mean {float(ref_sig.mean()):.0f}", flush=True)

    rows, best = [], None
    for topk in GRID_TOPK:
        for scale in GRID_SCALE:
            r = val.run(models[0], scale=scale, topk=topk, pre=pre,
                        center_w=center_w)
            rec = {"scale": scale, "topk": topk,
                   **{k: float(v) for k, v in r.items()}}
            rows.append(rec)
            print(f"[calib] topk={topk:4d} scale={scale:.2f} -> avg {r['score_avg']:+.4f} "
                  f"| n_pred {r['n_pred']:6.0f}/{r['n_real']:.0f} fid {r['raw_fid']:.3f} "
                  f"pds {r['raw_pds']:.3f} jac {r['raw_jac']:.4f} nmae {r['raw_nmae']:.3f} "
                  f"mse {r['raw_mse']:.3f} reach {r['raw_reach']:.3f}", flush=True)
            if best is None or r["score_avg"] > best["score_avg"]:
                best = rec
    (ROOT / "runs/calibration.json").write_text(json.dumps(
        {"best": best, "grid": rows}, indent=2))
    print(f"\n[calib] best: topk={best['topk']} scale={best['scale']} "
          f"score_avg={best['score_avg']:+.4f}", flush=True)
    return best


# ----------------------------------------------------------------------- write

def _context_cells(ctx: str, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    import h5py
    with h5py.File(DATA / f"challenge_2026/context_{ctx}.h5ad", "r") as f:
        X = f["X"]
        n_obs, n_gene = (int(v) for v in X.attrs["shape"])
        mat = sp.csr_matrix((X["data"][:], X["indices"][:], X["indptr"][:]),
                            shape=(n_obs, n_gene))
    cells = torch.as_tensor(mat.toarray().astype(np.float32), device=device)
    lib = cells.sum(1, keepdim=True).clamp_min(1.0)
    mean_cpm = (cells * (M.CPM / lib)).mean(0)
    return cells, mean_cpm


def write_submission(models, corpus: Corpus, device: str, scale: float, topk: int,
                     tag: str, seed: int = 0,
                     center_w: float = CENTER_W) -> Path:
    panel = pd.read_csv(DATA / "challenge_2026/gene_names.csv")["gene_name"].astype(str).to_numpy()
    targets = pd.read_csv(DATA / "challenge_2026/pert_counts.csv")["target_gene"].astype(str).tolist()
    assert len(targets) == 300 and len(set(targets)) == 300
    gene_pos = {g: i for i, g in enumerate(panel)}
    w = torch.as_tensor(corpus.W, device=device, dtype=torch.long)

    # 28 of the 300 targets are absent from the source screen, so their per-gene
    # channels are all zero and the trunk has nothing to condition on.  Fall those
    # back to the mean-response prior explicitly: that is the official baseline
    # arm, worth 0, whereas an unconditioned trunk output could be worth much less.
    covered = set(corpus.lines["gwps"]["perts"])
    uncovered = [t for t in targets if t not in covered]
    prior = torch.as_tensor(corpus.prior_all, device=device)
    print(f"[submit] {len(uncovered)}/300 targets absent from the source screen, "
          f"served from the mean-response prior", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    blocks, obs_pert, obs_ctx = [], [], []
    t0 = time.time()
    unc = set(uncovered)
    tta = bool(getattr(models[0], "_tta", False))
    keep_q = float(getattr(models[0], "_keep_q", 0.2))
    for ctx in "ABC":
        cells, mean_cpm = _context_cells(ctx, device)
        if tta:
            from inference.tta import confidence_gate, mc_dropout_predict
            lfc_w, sig_w, std_w = mc_dropout_predict(
                models[0], corpus, targets, f"ctx_{ctx}", device)
            lfc_w = confidence_gate(lfc_w, std_w, keep_q)
        else:
            lfc_w, sig_w = predict(models, corpus, targets, f"ctx_{ctx}", device,
                                   center_w=center_w)
        # a gene this context does not express cannot respond in it
        univ = torch.as_tensor(corpus.lines[f"ctx_{ctx}"]["univ"].astype(np.float32),
                               device=device)
        gen = torch.Generator(device=device).manual_seed(seed + ord(ctx))
        per_ctx = []
        for i, tgt in enumerate(targets):
            v = (prior if tgt in unc else lfc_w[i]) * univ
            v = M.sparsify(v, sig_w[i], topk)
            full = torch.zeros(panel.size, device=device)
            full[w] = v
            j = gene_pos.get(tgt)
            if j is not None:
                full[j] = CIS_LFC if float(mean_cpm[j]) >= 5.0 else 0.0
            y = M.synthesize_cells(full, cells, mean_cpm, CELLS, scale, gen)
            per_ctx.append(sp.csr_matrix(y.to(torch.int32).cpu().numpy()))
            obs_pert.extend([tgt] * CELLS)
            obs_ctx.extend([ctx] * CELLS)
        blocks.append(sp.vstack(per_ctx, format="csr"))
        del cells
        torch.cuda.empty_cache()
        print(f"[submit] context {ctx} done, {blocks[-1].shape} "
              f"nnz={blocks[-1].nnz} ({time.time() - t0:.0f}s)", flush=True)

    x = sp.vstack(blocks, format="csr")
    obs = pd.DataFrame({"target_gene": pd.Categorical(obs_pert),
                        "context": pd.Categorical(obs_ctx)})
    _assert_contract(x, obs, panel, targets)

    import anndata as ad
    adata = ad.AnnData(X=x, obs=obs, var=pd.DataFrame(index=pd.Index(panel, name=None)))
    path = OUT / f"{tag}.h5ad"
    adata.write_h5ad(path, compression="gzip")
    meta = {"tag": tag, "scale": scale, "topk": topk, "seed": seed,
            "uncovered_targets": uncovered,
            "models": [m._src for m in models],
            "model_scores": [m._score for m in models],
            "shape": list(x.shape), "nnz": int(x.nnz),
            "median_umi": float(np.median(np.asarray(x.sum(1)).ravel()))}
    (OUT / f"{tag}.json").write_text(json.dumps(meta, indent=2))
    print(f"[submit] wrote {path} | {x.shape} nnz={x.nnz} "
          f"median UMI {meta['median_umi']:.0f} | {time.time() - t0:.0f}s", flush=True)
    return path


def _assert_contract(x: sp.csr_matrix, obs: pd.DataFrame, panel: np.ndarray,
                     targets: list[str]) -> None:
    """Every upload rule from the metric spec, checked before the file is written."""
    assert x.shape == (len(targets) * CELLS * 3, panel.size), x.shape
    assert x.dtype.kind in "iu", f"X must be integral, got {x.dtype}"
    assert x.data.min() >= 0, "negative counts"
    assert np.isfinite(x.data).all(), "non-finite counts"
    tot = np.asarray(x.sum(1)).ravel()
    assert tot.max() <= 1_000_000, f"cell total {tot.max()} exceeds 1e6"
    assert x.nnz <= MAX_STORED, f"{x.nnz} stored entries exceeds {MAX_STORED}"
    counts = obs.groupby(["context", "target_gene"], observed=True).size()
    assert set(counts.values) == {CELLS}, f"cells per perturbation: {set(counts.values)}"
    assert set(obs["context"].unique()) == {"A", "B", "C"}
    for c in "ABC":
        got = set(obs.loc[obs["context"] == c, "target_gene"].unique())
        assert got == set(targets), f"context {c} perturbation set mismatch"
    assert "non-targeting" not in set(obs["target_gene"].unique()), "control rows present"
    print(f"[submit] contract ok: {x.shape}, nnz={x.nnz} "
          f"({100 * x.nnz / (x.shape[0] * x.shape[1]):.1f}% dense), "
          f"max cell total {tot.max()}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--scale", type=float, default=None)
    ap.add_argument("--topk", type=int, default=None)
    ap.add_argument("--tag", default="six_aligned_v1")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = f"cuda:{args.gpu}"
    corpus = Corpus(device, TRAIN_LINES)
    models = load_models([Path(p) for p in args.ckpt], corpus, device)

    scale, topk = args.scale, args.topk
    if args.calibrate:
        best = calibrate(models, corpus, device, args.seed)
        scale = best["scale"] if scale is None else scale
        topk = best["topk"] if topk is None else topk
    if args.write:
        assert scale is not None and topk is not None, "need --scale/--topk or --calibrate"
        write_submission(models, corpus, device, scale, topk, args.tag, args.seed)


if __name__ == "__main__":
    main()
