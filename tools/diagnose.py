#!/usr/bin/env python3
"""Stage-0 diagnostics for the zero-residual source-transfer baseline.

The official submission (Petit Fan, score_avg = 0.0740) ships untrained
PertResponseNet with scale=0.5 and topk=7000.  Training logs show
n_pred ≈ 3450 against a reference of 168 calls, which pins fidelity at
chance (k / max(n_pred, n_real) ≈ 0.5) and makes MSE structurally
unable to beat the context-mean baseline.  These three probes test that
story on held-out H1 without training anything:

  D1  topk × scale grid of the official six (does cutting calls lift fid/jac/mse?)
  D2  direction conservation of compress(GWPS) vs H1, by rank depth
  D3  calibration of sig_logit rank vs sign accuracy and |LFC| error

    python tools/diagnose.py --gpu 0
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.train import Corpus, TRAIN_LINES, VAL_LINE, Validator
from inference.submit import load_models
from models import compress
from paths import RUNS


GRID_TOPK = [50, 100, 168, 300, 600, 1000, 2000, 4500, 7000]
GRID_SCALE = [0.1, 0.2, 0.3, 0.5, 0.8]
DEPTHS = [6, 50, 168, 2000, 7000]
STRATA = ("all_universe", "h1_sig", "src_sig", "both_sig")
RANKINGS = ("abs_lfc", "abs_z", "sig_then_z")
N_BINS = 10


def _predict(models, corpus: Corpus, perts: list[str], dest: str, device: str):
    from engine.train import batchify

    lfc_out, sig_out = [], []
    for i in range(0, len(perts), 24):
        sel = [(p, dest) for p in perts[i : i + 24]]
        loc, ctx, esm, sca, pri, _, _, _, pann = batchify(corpus, sel, device)
        la, sa = None, None
        for m in models:
            lfc, sig, _ = m(loc, ctx, esm, sca, pri, pann)
            la = lfc if la is None else la + lfc
            sa = sig if sa is None else sa + sig
        lfc_out.append(la / len(models))
        sig_out.append(sa / len(models))
    return torch.cat(lfc_out), torch.cat(sig_out)


# --------------------------------------------------------------------- D1

def run_d1(val: Validator, model, pre, out: Path) -> list[dict]:
    rows, best = [], None
    t0 = time.time()
    for topk in GRID_TOPK:
        for scale in GRID_SCALE:
            r = val.run(model, scale=scale, topk=topk, pre=pre)
            rec = {"topk": topk, "scale": scale,
                   **{k: float(v) for k, v in r.items()}}
            rows.append(rec)
            print(f"[d1] topk={topk:4d} scale={scale:.2f} "
                  f"avg {r['score_avg']:+.4f} n_pred {r['n_pred']:.0f}/{r['n_real']:.0f} "
                  f"fid {r['raw_fid']:.3f} jac {r['raw_jac']:.4f} "
                  f"mse {r['raw_mse']:.3f} pds {r['raw_pds']:.3f} "
                  f"nmae {r['raw_nmae']:.3f} reach {r['raw_reach']:.3f}",
                  flush=True)
            if best is None or r["score_avg"] > best["score_avg"]:
                best = rec
    (out / "d1_topk_scale.json").write_text(json.dumps(
        {"best": best, "grid": rows, "seconds": time.time() - t0}, indent=2))
    print(f"[d1] best topk={best['topk']} scale={best['scale']} "
          f"score_avg={best['score_avg']:+.4f}", flush=True)
    return rows


# --------------------------------------------------------------------- D2

def _stratum_masks(univ, a, b, h1_sig, src_sig):
    fin = np.isfinite(b)
    return {
        "all_universe": univ & fin & (np.abs(a) > 0) & (np.abs(b) > 0),
        "h1_sig": h1_sig & fin,
        "src_sig": src_sig & fin & (np.abs(a) > 0),
        "both_sig": (h1_sig & src_sig) & fin & (np.abs(a) > 0),
    }


def _rank_indices(a, z, src_sig, ranking: str) -> np.ndarray:
    if ranking == "abs_lfc":
        return np.argsort(-np.abs(a))
    if ranking == "abs_z":
        return np.argsort(-np.abs(z))
    key = src_sig.astype(np.float32) * 1e6 + np.abs(z)
    return np.argsort(-key)


def run_d2(corpus: Corpus, perts: list[str], out: Path) -> dict:
    """P(sign match) of compress(GWPS) vs H1 LFC，按显著性分层 + 三种排序 top-k。"""
    src, h1 = corpus.lines["gwps"], corpus.lines[VAL_LINE]
    univ = h1["univ"]
    acc_strata = {s: [] for s in STRATA}
    acc_rank = {r: {k: [] for k in DEPTHS} for r in RANKINGS}
    n_used = 0
    for p in perts:
        i, j = src["index"].get(p, -1), h1["index"].get(p, -1)
        if i < 0 or j < 0:
            continue
        a = compress(torch.from_numpy(src["lfc"][i])).numpy()
        b = h1["lfc"][j]
        z = src["z"][i] if i >= 0 else np.zeros_like(a)
        h1_sig = h1["sig"][j] & univ if "sig" in h1 else np.zeros_like(univ)
        src_sig = src["sig"][i] & univ if "sig" in src else np.zeros_like(univ)
        masks = _stratum_masks(univ, a, b, h1_sig, src_sig)
        if int(masks["h1_sig"].sum()) < 5:
            continue
        n_used += 1
        match = (np.sign(a) == np.sign(b))
        for name, mask in masks.items():
            if int(mask.sum()) >= 1:
                acc_strata[name].append(float(match[mask].mean()))
        for ranking in RANKINGS:
            order = _rank_indices(a, z, src_sig, ranking)
            for k in DEPTHS:
                take = order[:k]
                valid = np.isfinite(b[take])
                if not valid.any():
                    continue
                idx = take[valid]
                acc_rank[ranking][k].append(float(match[idx].mean()))

    summary: dict = {
        "strata": {
            s: {"mean": float(np.mean(v)), "std": float(np.std(v)), "n": len(v)}
            for s, v in acc_strata.items() if v
        },
        "rankings": {
            r: {
                str(k): {"mean": float(np.mean(v)), "std": float(np.std(v)), "n": len(v)}
                for k, v in rk.items() if v
            }
            for r, rk in acc_rank.items()
        },
        "n_perts": n_used,
    }
    (out / "d2_direction.json").write_text(json.dumps(summary, indent=2))
    print(f"[d2] {n_used} H1 perts; sign-match by stratum:", flush=True)
    for s in STRATA:
        if s not in summary["strata"]:
            continue
        t = summary["strata"][s]
        print(f"     {s:<14}  {t['mean']:.3f} ± {t['std']:.3f}", flush=True)
    print("[d2] top-k curves by ranking:", flush=True)
    for r in RANKINGS:
        print(f"  {r}:", flush=True)
        for k in DEPTHS:
            key = str(k)
            if key not in summary["rankings"].get(r, {}):
                continue
            t = summary["rankings"][r][key]
            print(f"     top-{k:<5}  {t['mean']:.3f} ± {t['std']:.3f}", flush=True)
    return summary


# --------------------------------------------------------------------- D3

def run_d3(corpus: Corpus, perts: list[str], lfc_w, sig_w, out: Path) -> dict:
    """Calibration of sig_logit rank vs sign accuracy and |LFC| relative error."""
    src, h1 = corpus.lines["gwps"], corpus.lines[VAL_LINE]
    univ = torch.as_tensor(h1["univ"], device=lfc_w.device)
    bins = [[] for _ in range(N_BINS)]
    mag_err = [[] for _ in range(N_BINS)]
    n_used = 0
    for t, p in enumerate(perts):
        j = h1["index"].get(p, -1)
        if j < 0:
            continue
        true = torch.as_tensor(h1["lfc"][j], device=lfc_w.device)
        pred = lfc_w[t]
        logit = sig_w[t]
        keep = univ.bool() & torch.isfinite(true) & (true.abs() > 0) & (pred.abs() > 0)
        if int(keep.sum()) < 20:
            continue
        n_used += 1
        # rank 0 = most confident; quantile 0 is the top of the list
        rank = torch.zeros_like(logit)
        order = torch.argsort(logit, descending=True)
        rank[order] = torch.arange(logit.numel(), device=logit.device, dtype=logit.dtype)
        q = (rank / max(float(rank.numel()), 1.0)).clamp(max=0.999)
        bi = (q * N_BINS).long()
        same = (pred.sign() == true.sign()).to(torch.float32)
        rel = (pred.abs() - true.abs()).abs() / true.abs().clamp_min(1e-3)
        idx = torch.nonzero(keep, as_tuple=True)[0]
        for i in idx.tolist():
            b = int(bi[i])
            bins[b].append(float(same[i]))
            mag_err[b].append(float(rel[i]))

    curve = []
    for b in range(N_BINS):
        curve.append({
            "bin": b,
            "quantile_lo": b / N_BINS,
            "quantile_hi": (b + 1) / N_BINS,
            "n": len(bins[b]),
            "sign_acc": float(np.mean(bins[b])) if bins[b] else None,
            "rel_lfc_err": float(np.median(mag_err[b])) if mag_err[b] else None,
        })
    # first bin that drops to chance (~0.55) is a natural TTA / post-process cut
    cut = 1.0
    for row in curve:
        if row["sign_acc"] is not None and row["sign_acc"] < 0.55:
            cut = row["quantile_lo"]
            break
    rec = {"n_perts": n_used, "n_bins": N_BINS, "curve": curve,
           "suggested_keep_quantile": cut}
    (out / "d3_calibration.json").write_text(json.dumps(rec, indent=2))
    print(f"[d3] {n_used} H1 perts; sign-acc by sig_logit quantile "
          f"(0 = most confident):", flush=True)
    for row in curve:
        acc = "  n/a" if row["sign_acc"] is None else f"{row['sign_acc']:.3f}"
        err = "  n/a" if row["rel_lfc_err"] is None else f"{row['rel_lfc_err']:.3f}"
        print(f"     q={row['quantile_lo']:.1f}-{row['quantile_hi']:.1f}  "
              f"acc={acc}  |lfc|_rel={err}  n={row['n']}", flush=True)
    print(f"[d3] suggested keep-quantile (acc>=0.55) = {cut:.2f}", flush=True)
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = f"cuda:{args.gpu}"
    out = RUNS / "zero_fan_diag"
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    corpus = Corpus(device, TRAIN_LINES)
    models = load_models([Path("untrained")], corpus, device)
    val = Validator(corpus, device, args.seed)
    print("[diag] predicting once on H1...", flush=True)
    pre = val.predict(models[0])

    d2 = run_d2(corpus, val.perts, out)
    d3 = run_d3(corpus, val.perts, pre[0], pre[1], out)
    d1 = run_d1(val, models[0], pre, out)

    summary = {
        "d1_best": max(d1, key=lambda r: r["score_avg"]) if d1 else None,
        "d1_baseline_7000_0.5": next(
            (r for r in d1 if r["topk"] == 7000 and abs(r["scale"] - 0.5) < 1e-9), None),
        "d2_top168": d2.get("168"),
        "d3_keep_quantile": d3.get("suggested_keep_quantile"),
        "seconds": time.time() - t0,
    }
    (out / "SUMMARY.json").write_text(json.dumps(summary, indent=2))
    print(f"[diag] done in {time.time() - t0:.0f}s -> {out}", flush=True)


if __name__ == "__main__":
    main()
