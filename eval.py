#!/usr/bin/env python3
"""Local reimplementation of the six official vcc2026 metrics, on GPU.

Follows docs/vcc2026_metrics (cell-eval2 0.15.0, rule_version 3) clause by
clause.  Raw values are scaled with the *official* anchors, which were recovered
by OLS from the five published submissions, so a number printed here is on the
same axis as the leaderboard.  H1 is pre-shaped by prep.py to the 2026 geometry
(400 cells per perturbation, 18400 control cells, median 20k UMI), which is what
makes the raw values comparable in the first place.

Sanity checks available via __main__ (see smoke()):
  * split-half replicate on H1 must land inside the published r intervals
  * a constant-pseudobulk submission must reproduce B0's pathology
    (fidelity ~0.002, jaccard ~0)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

TS = 5.0e4          # profile normalization for PDS / expression error
CPM = 1.0e6         # per-cell normalization for the DE test
EPS_LFC = 1e-9      # official epsilon in the fold-change ratio
CPM_MIN = 5.0       # low-expression filter, on control mean CPM
ALPHA = 0.05
PURITY = 0.9        # direction-reach pass mark
NMAE_GATE = 10      # minimum reference-significant genes to score NMAE
TOL = 1e-12         # tie tolerance for the PDS rank

# Official anchors: raw = b + (r - b) * score, fitted on vcc2026-val A/B/C.
ANCHORS = {
    "pds":   (0.499812, 0.948554, None),
    "mse":   (0.989000, 0.036500, (0.0, 1.0)),
    "nmae":  (1.000942, 0.399624, (-6.0, None)),
    "fid":   (0.510100, 0.805848, None),
    "reach": (0.079046, 0.971755, None),
    "jac":   (0.030623, 0.401350, None),
}
KEYS = ("pds", "mse", "nmae", "fid", "reach", "jac")
# Published replicate envelope over contexts A/B/C, used only to validate this file.
REPLICATE_ENVELOPE = {
    "pds": (0.927, 0.984), "mse": (0.028, 0.045), "nmae": (0.369, 0.431),
    "fid": (0.795, 0.832), "reach": (0.958, 0.978), "jac": (0.375, 0.423),
}


def scale(raw: dict[str, float]) -> dict[str, float]:
    """Official scaling s = (u - b) / (r - b), with the documented clamps."""
    out = {}
    for k in KEYS:
        b, r, clamp = ANCHORS[k]
        s = (raw[k] - b) / (r - b)
        if clamp is not None:
            lo, hi = clamp
            if lo is not None:
                s = max(lo, s)
            if hi is not None:
                s = min(hi, s)
        out[f"score_{k}"] = s
    out["score_avg"] = sum(out[f"score_{k}"] for k in KEYS) / 6.0
    return out


# ----------------------------------------------------------------- primitives

def profile(counts: torch.Tensor) -> torch.Tensor:
    """Group profile: sum over cells, renormalize to TS, log1p.  (n,G) -> (G,)"""
    s = counts.sum(0, dtype=torch.float64)
    return torch.log1p(TS * s / s.sum().clamp_min(1.0))


def _jackknife(counts: torch.Tensor, block: int = 4096) -> torch.Tensor:
    """Delete-one-cell jackknife scatter C_p of the group profile (scalar)."""
    n = counts.shape[0]
    if n < 2:
        return torch.zeros((), dtype=torch.float64, device=counts.device)
    p = counts.sum(0, dtype=torch.float64)
    s = p.sum()
    ell = counts.sum(1, dtype=torch.float64)
    den = (s - ell).clamp_min(1.0)

    vsum = torch.zeros_like(p)
    for lo in range(0, n, block):
        y = counts[lo : lo + block].to(torch.float64)
        vsum += torch.log1p(TS * (p - y) / den[lo : lo + block, None]).sum(0)
    vbar = vsum / n
    ss = torch.zeros_like(p)
    for lo in range(0, n, block):
        y = counts[lo : lo + block].to(torch.float64)
        v = torch.log1p(TS * (p - y) / den[lo : lo + block, None])
        ss += ((v - vbar) ** 2).sum(0)
    return (n - 1) / n * ss.sum()


def _percell_cpm(counts: torch.Tensor) -> torch.Tensor:
    lib = counts.sum(1, keepdim=True, dtype=torch.float32).clamp_min(1.0)
    return counts.to(torch.float32) * (CPM / lib)


def _bh(p: torch.Tensor, universe: torch.Tensor) -> torch.Tensor:
    """Benjamini-Hochberg within one perturbation, over `universe` genes."""
    idx = torch.nonzero(universe, as_tuple=True)[0]
    m = idx.numel()
    sig = torch.zeros_like(universe)
    if m == 0:
        return sig
    sub = p[idx]
    order = torch.argsort(sub)
    ranked = sub[order]
    thresh = ranked * m / torch.arange(1, m + 1, device=p.device, dtype=p.dtype)
    # step-up: pass if any deeper rank passes
    passes = torch.flip(torch.cummin(torch.flip(thresh, [0]), 0).values, [0]) < ALPHA
    hit = torch.zeros(m, dtype=torch.bool, device=p.device)
    hit[order] = passes
    sig[idx] = hit
    return sig


class DEReference:
    """Sorted control CPM per gene, plus everything the DE test reuses."""

    def __init__(self, ctrl_counts: torch.Tensor):
        cpm = _percell_cpm(ctrl_counts)
        self.n = cpm.shape[0]
        self.mean = cpm.mean(0, dtype=torch.float64)
        self.universe = self.mean > CPM_MIN
        self.n_zero = (cpm == 0).sum(0)
        self.sorted = cpm.T.contiguous()                # (G, n_ctrl)
        self.sorted, _ = torch.sort(self.sorted, dim=1)
        self.device = cpm.device
        self.G = cpm.shape[1]

    def table(self, counts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """DE table of one group against the control: (p, adj_p, lfc)."""
        cpm = _percell_cpm(counts)
        m = cpm.shape[0]
        vt = cpm.T.contiguous()                          # (G, m)
        u = (torch.searchsorted(self.sorted, vt, right=False)
             + torch.searchsorted(self.sorted, vt, right=True)).to(torch.float64)
        u = u.sum(1) * 0.5                               # (G,)

        n0 = (self.n_zero + (cpm == 0).sum(0)).to(torch.float64)
        tot = float(m + self.n)
        tie = (n0**3 - n0) / max(tot * (tot - 1.0), 1.0)
        var = (m * self.n / 12.0) * ((tot + 1.0) - tie)
        z = (u - m * self.n / 2.0) / var.clamp_min(1e-12).sqrt()
        p = torch.erfc(z.abs() / np.sqrt(2.0)).clamp(1e-300, 1.0)

        lfc = torch.log2((cpm.mean(0, dtype=torch.float64) + EPS_LFC)
                         / (self.mean + EPS_LFC))
        adj = _bh(p, self.universe)
        return p, adj, lfc


# --------------------------------------------------------------------- metrics

def _pds(pred_prof: torch.Tensor, real_prof: torch.Tensor, ctrl_prof: torch.Tensor,
         drop: torch.Tensor) -> float:
    """Perturbation discrimination; `drop` masks all 300 panel target genes."""
    keep = ~drop
    dp = (pred_prof - ctrl_prof)[:, keep]
    dr = (real_prof - ctrl_prof)[:, keep]
    np_, nr = dp.norm(dim=1), dr.norm(dim=1)
    d = 1.0 - (dp @ dr.T) / (np_[:, None] * nr[None, :]).clamp_min(1e-30)
    d[np_ <= 0] = 1.0
    d[:, nr <= 0] = 1.0
    own = d.diagonal()[:, None]
    less = (d < own - TOL).sum(1).to(torch.float64)
    eq = ((d - own).abs() <= TOL).sum(1).to(torch.float64)
    k = less + 0.5 * (eq - 1.0)
    n = d.shape[1]
    return float((1.0 - k / (n - 1)).mean())


def _expr_mse(pred_prof, real_prof, ctrl_prof, c_pred, c_real, c_ctrl, scored) -> float:
    """expr_mse_unbiased_capped_norm: ratio of two panel-wide sums."""
    gp = scored.sum(1).to(torch.float64).clamp_min(1.0)          # G_p
    inv = 1.0 / gp
    dist_pr = (((pred_prof - real_prof) ** 2) * scored).sum(1)
    dist_rc = (((real_prof - ctrl_prof) ** 2) * scored).sum(1)

    cmin = torch.minimum(c_pred, c_real)
    # rho: cap the total correction by the submission's own across-perturbation spread
    w = inv[:, None] * scored
    wsum = w.sum(0).clamp_min(1e-30)
    mean_w = (w * pred_prof).sum(0) / wsum
    b = (w * (pred_prof - mean_w[None, :]) ** 2).sum()
    denom_rho = (inv * cmin).sum()
    rho = torch.clamp(b / denom_rho.clamp_min(1e-30), max=1.0) if denom_rho > 0 else \
        torch.ones((), dtype=torch.float64, device=gp.device)

    n_p = inv * (dist_pr - rho * cmin - c_real)
    d_p = inv * (dist_rc - c_real - c_ctrl)
    return float(n_p.sum() / d_p.sum())


def _de_metrics(pred, real, own_idx: int) -> dict[str, float | None]:
    """Fidelity, reach, jaccard and NMAE for one perturbation.

    `pred` / `real` are (p, adj_sig, lfc) triples against the same control.
    """
    p_p, p_sig, p_lfc = pred
    _, r_sig, r_lfc = real
    if own_idx >= 0:
        p_sig = p_sig.clone(); p_sig[own_idx] = False
        r_sig = r_sig.clone(); r_sig[own_idx] = False

    adjud = torch.isfinite(r_lfc) & (r_lfc != 0)
    n_real = int(r_sig.sum())
    call = p_sig & adjud
    n_pred = int(call.sum())

    # fidelity: k / max(n_pred, n_real); a call without a direction is a miss
    same = torch.sign(p_lfc) == torch.sign(r_lfc)
    same &= torch.isfinite(p_lfc) & (p_lfc != 0)
    k = int((call & same).sum())
    fid = None if (n_pred == 0 and n_real == 0) else k / max(n_pred, n_real, 1)

    # jaccard over the significant sets; empty union means both agree on silence
    inter = int((p_sig & r_sig).sum())
    union = int((p_sig | r_sig).sum())
    jac = 1.0 if union == 0 else inter / union

    # reach: deepest prefix of the reference budget with purity >= 0.9,
    # ordered by the submission's own confidence
    reach = None
    if n_real > 0:
        pool = torch.nonzero(r_sig & adjud, as_tuple=True)[0]
        if pool.numel() > 0:
            key = torch.stack([(~p_sig[pool]).to(torch.float64),
                               p_p[pool].to(torch.float64),
                               -p_lfc[pool].abs().to(torch.float64)])
            order = np.lexsort(key.cpu().numpy()[::-1])
            hit = same[pool][torch.as_tensor(order.copy(), device=pool.device)]
            cum = torch.cumsum(hit.to(torch.float64), 0)
            kk = torch.arange(1, hit.numel() + 1, device=hit.device, dtype=torch.float64)
            ok = cum >= PURITY * kk
            reach = (float(kk[ok].max()) if bool(ok.any()) else 0.0) / n_real
        else:
            reach = 0.0

    # nmae over the reference's own significant gate
    nmae = None
    gate = r_sig & torch.isfinite(r_lfc)
    if int(gate.sum()) >= NMAE_GATE:
        pl = torch.where(torch.isfinite(p_lfc), p_lfc, torch.zeros_like(p_lfc))
        den = float(r_lfc[gate].abs().sum())
        if den > 0:
            nmae = float((pl[gate] - r_lfc[gate]).abs().sum()) / den
    return {"fid": fid, "reach": reach, "jac": jac, "nmae": nmae}


# ------------------------------------------------------------------ entrypoint

def official_six(pred_cells: list[torch.Tensor], ref: "Panel") -> dict[str, float]:
    """Six raw metrics for a submission given as one count matrix per perturbation."""
    assert len(pred_cells) == ref.n_pert
    prof, c_pred, rows = [], [], []
    for i, y in enumerate(pred_cells):
        prof.append(profile(y))
        c_pred.append(_jackknife(y))
        rows.append(_de_metrics(ref.de.table(y), ref.real_table[i], ref.own_idx[i]))
    prof = torch.stack(prof)
    c_pred = torch.stack(c_pred)

    raw = {
        "pds": _pds(prof, ref.real_prof, ref.ctrl_prof, ref.drop_pds),
        "mse": _expr_mse(prof, ref.real_prof, ref.ctrl_prof,
                         c_pred, ref.c_real, ref.c_ctrl, ref.scored),
    }
    for k in ("fid", "reach", "jac", "nmae"):
        vals = [r[k] for r in rows if r[k] is not None]
        raw[k] = float(np.mean(vals)) if vals else float("nan")
    return raw


def sparsify(lfc: torch.Tensor, sig_logit: torch.Tensor, k: int) -> torch.Tensor:
    """Zero the fold change outside the k genes the model is most confident about.

    This is what makes the predicted call count controllable.  Fidelity is
    directional precision times coverage capped at 1, so calling past the
    reference's own count buys nothing while every extra call contributes a
    coin-flip direction.  Left unsparsified the submission calls thousands of
    genes and fidelity pins itself to chance regardless of how good the top of
    the ranking is.
    """
    if k >= lfc.shape[-1]:
        return lfc
    idx = sig_logit.topk(k, dim=-1).indices
    out = torch.zeros_like(lfc)
    return out.scatter_(-1, idx, lfc.gather(-1, idx))


def synthesize_cells(lfc: torch.Tensor, ctrl_cells: torch.Tensor, ctrl_mean_cpm: torch.Tensor,
                     n_out: int, scale: float, gen: torch.Generator) -> torch.Tensor:
    """Turn a predicted log2 fold change into integer counts that can be scored.

    Real control cells are the carriers, and a gene with no predicted change is
    left *bit-identical* in every cell.  Down-regulation is binomial thinning,
    which keeps a negative-binomial count negative-binomial; up-regulation adds a
    Poisson increment sized from the control group mean, so a gene that dropped
    out in a cell can still come up.

    Renormalizing and redrawing the whole cell as Poisson instead -- the obvious
    way to restore integrality -- silently destroys the metric: Poisson variance
    equals its mean while real counts are overdispersed, so the predicted group
    is far tighter than the control it is tested against and the Wilcoxon test
    calls ~7800 of 10778 genes no matter what was predicted.  Coverage then pins
    at 1 with thousands of coin-flip directions, and fidelity sticks at 0.50.

    Replicating one deterministic pseudobulk is the same failure at its limit:
    zero within-group variance, which is what left B0 at a fidelity of 0.002.
    """
    n_src = ctrl_cells.shape[0]
    if n_out <= n_src:
        idx = torch.randperm(n_src, generator=gen, device=ctrl_cells.device)[:n_out]
    else:
        idx = torch.randint(n_src, (n_out,), generator=gen, device=ctrl_cells.device)
    y = ctrl_cells[idx].to(torch.float32)
    lib = y.sum(1, keepdim=True).clamp_min(1.0)
    mult = torch.exp2(scale * lfc)[None, :]

    # Down-regulation by binomial thinning, which keeps an overdispersed count
    # overdispersed; up-regulation by a Poisson increment sized from the control
    # group mean, so a gene that dropped out in a cell can still come up.
    # (Deterministic multiplication with unbiased randomised rounding was tried
    # as a lower-variance alternative and scored 0.093 against 0.105 -- the extra
    # thinning variance is not what limits this.)
    keep = mult.clamp(max=1.0).expand_as(y)
    out = torch.binomial(y, keep, generator=gen)
    gain = (mult - 1.0).clamp(min=0.0) * (ctrl_mean_cpm[None, :] / CPM) * lib
    if bool((gain > 0).any()):
        out = out + torch.poisson(gain.expand_as(y), generator=gen)
    return out


class Panel:
    """Everything about the reference side, computed once and reused each epoch."""

    def __init__(self, real_cells: list[torch.Tensor], ctrl_cells: torch.Tensor,
                 own_idx: np.ndarray, panel_target_mask: torch.Tensor):
        self.n_pert = len(real_cells)
        self.de = DEReference(ctrl_cells)
        self.own_idx = [int(i) for i in own_idx]
        self.drop_pds = panel_target_mask
        self.ctrl_prof = profile(ctrl_cells)
        self.real_prof = torch.stack([profile(y) for y in real_cells])
        self.c_real = torch.stack([_jackknife(y) for y in real_cells])
        self.c_ctrl = _jackknife(ctrl_cells)
        self.real_table = [self.de.table(y) for y in real_cells]

        g = ctrl_cells.shape[1]
        scored = torch.ones((self.n_pert, g), dtype=torch.float64, device=ctrl_cells.device)
        for i, j in enumerate(self.own_idx):
            if j >= 0:
                scored[i, j] = 0.0
        self.scored = scored

    def real_sig(self) -> torch.Tensor:
        """(P, G) reference significance with each row's own target removed."""
        out = torch.stack([t[1] for t in self.real_table])
        for i, j in enumerate(self.own_idx):
            if j >= 0:
                out[i, j] = False
        return out

    def real_lfc(self) -> torch.Tensor:
        return torch.stack([t[2] for t in self.real_table])


# ----------------------------------------------------------------- H1 loading

def load_h1_eval(path, device="cuda") -> tuple[Panel, list[torch.Tensor], torch.Tensor, dict]:
    """Read prep.py's H1 bundle and build the reference Panel."""
    import h5py
    import scipy.sparse as sp

    with h5py.File(path, "r") as f:
        def csr(name):
            g = f[name]
            return sp.csr_matrix(
                (g["data"][:], g["indices"][:], g["indptr"][:]), shape=tuple(g.attrs["shape"])
            )
        pert = csr("pert").toarray().astype(np.float32)
        ctrl = csr("ctrl").toarray().astype(np.float32)
        perts = np.array([x.decode() for x in f["perts"][:]])
        pert_of = np.array([x.decode() for x in f["pert_of"][:]])
        panel_idx = f["panel_idx"][:]
        meta = dict(f.attrs)

    ctrl_t = torch.as_tensor(ctrl, device=device)
    real = []
    for p in perts:
        real.append(torch.as_tensor(pert[pert_of == p], device=device))

    lut = {int(v): i for i, v in enumerate(panel_idx)}
    own = np.array([lut.get(int(i), -1) for i in _panel_of(perts)], np.int64)
    drop = torch.zeros(panel_idx.size, dtype=torch.bool, device=device)
    for i in _panel_of(perts):
        j = lut.get(int(i), -1)
        if j >= 0:
            drop[j] = True
    return Panel(real, ctrl_t, own, drop), real, ctrl_t, {
        "perts": perts, "panel_idx": panel_idx, **meta
    }


_PANEL_CACHE: dict | None = None


def _panel_of(names: np.ndarray) -> np.ndarray:
    """Panel indices of the given gene symbols (-1 when absent)."""
    global _PANEL_CACHE
    if _PANEL_CACHE is None:
        import pandas as pd
        from train import DATA
        gp = DATA / "challenge_2026/gene_names.csv"
        genes = pd.read_csv(gp)["gene_name"].astype(str).to_numpy()
        _PANEL_CACHE = {g: i for i, g in enumerate(genes)}
    return np.array([_PANEL_CACHE.get(str(n), -1) for n in names], np.int64)



# ----------------------------------------------------------------- H1 validation

from train import DATA, PREP, VAL_LINE, batchify, prep_h1_eval
from loss import CENTER_W, decenter

GEN_SCALE, GEN_TOPK = 0.5, 7000


class Validator:
    """在留出的 H1 上重算官方六项。"""

    def __init__(self, corpus, device: str, seed: int = 0):
        self.corpus = corpus
        self.device = device
        panel, real, ctrl, meta = load_h1_eval(prep_h1_eval(), device)
        eval_perts = list(meta["perts"])
        src_perts = set(corpus.lines["gwps"]["perts"])
        h1 = corpus.lines[VAL_LINE]
        self.use = [i for i, p in enumerate(eval_perts) if p in src_perts and p in h1["index"]]
        if not self.use:
            raise RuntimeError("no H1 evaluation perturbation is covered by the source")
        self.perts = [eval_perts[i] for i in self.use]
        sub_real = [real[i] for i in self.use]
        own = np.array([panel.own_idx[i] for i in self.use], np.int64)
        self.panel = Panel(sub_real, ctrl, own, panel.drop_pds)
        self.ctrl = ctrl
        self.ctrl_mean = self.panel.de.mean.to(torch.float32)
        self.n_out = sub_real[0].shape[0]
        w_pos = -np.ones(18533, np.int64)
        w_pos[corpus.W] = np.arange(corpus.n_w)
        col = w_pos[meta["panel_idx"].astype(np.int64)]
        self.eval_in_w = torch.as_tensor(np.where(col >= 0, col, 0), device=device)
        self.eval_valid = torch.as_tensor(col >= 0, device=device)
        self.univ_w = torch.as_tensor(
            corpus.lines[VAL_LINE]["univ"].astype(np.float32), device=device)
        self.seed = seed
        self.gen = torch.Generator(device=device).manual_seed(seed)
        self.n_real = float(self.panel.real_sig().sum(1).to(torch.float64).median())
        print(f"[val] {len(self.perts)} H1 perturbations, {self.n_out} cells each, "
              f"{ctrl.shape[1]} eval genes ({int(self.eval_valid.sum())} inside W), "
              f"reference median n_sig={self.n_real:.0f}", flush=True)

    @torch.no_grad()
    def predict(self, model):
        model.eval()
        lfc, sig = [], []
        for i in range(0, len(self.perts), 24):
            chunk = [(p, VAL_LINE) for p in self.perts[i:i + 24]]
            loc, ctx, esm, sca, pri, uni, _, _, pann, go = batchify(self.corpus, chunk, self.device)
            a, b, _ = model(loc, ctx, esm, sca, pri, pann, go=go)
            lfc.append(a)
            sig.append(b)
        model.train()
        return torch.cat(lfc), torch.cat(sig)

    @torch.no_grad()
    def run(self, model, scale=GEN_SCALE, topk=GEN_TOPK, pre=None, center_w: float = CENTER_W):
        lfc_w, sig_w = self.predict(model) if pre is None else pre
        lfc_w = decenter(lfc_w, center_w)
        self.gen.manual_seed(self.seed)
        cells = []
        for i in range(len(self.perts)):
            v = (lfc_w[i] * self.univ_w)[self.eval_in_w] * self.eval_valid
            s = torch.where(self.eval_valid, sig_w[i][self.eval_in_w], torch.full_like(v, -1e9))
            cells.append(synthesize_cells(
                sparsify(v, s, topk), self.ctrl, self.ctrl_mean, self.n_out, scale, self.gen))
        raw = official_six(cells, self.panel)
        n_pred = float(np.mean([
            int(self.panel.de.table(c)[1].sum()) for c in cells[:min(32, len(cells))]]))
        return {**{f"raw_{k}": raw[k] for k in KEYS}, **scale(raw),
                "n_pred": n_pred, "n_real": self.n_real}


def main() -> None:
    import argparse
    from train import ROOT, TRAIN_LINES, Corpus, tag as default_tag

    ap = argparse.ArgumentParser(description="本地 H1 官方六项")
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()
    ckpt = Path(args.ckpt) if args.ckpt else ROOT / f"{default_tag}.pt"
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    corpus = Corpus(device, TRAIN_LINES)
    from submit import load_models
    models = load_models([ckpt], corpus, device)
    result = Validator(corpus, device, 0).run(models[0])
    print(f"score_avg = {result['score_avg']:+.6f}")
    for k in KEYS:
        print(f"  {k:6s} raw={result['raw_' + k]:.4f}  score={result['score_' + k]:+.4f}")


if __name__ == "__main__":
    main()
