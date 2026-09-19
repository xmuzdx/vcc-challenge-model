#!/usr/bin/env python3
"""Train the cross-context response network and select on the official six.

Protocol.  Source context is always Replogle K562 GWPS, which measures 272 of
the 300 target genes of 2026.  Training destinations are RPE1, HepG2 and Jurkat
(~2390 shared perturbations each); the held-out destination is H1, which is both
the closest context to the 2026 controls and the only one with enough cells per
perturbation to rebuild the official metrics locally.  So validation is a
leave-one-context-out *and* leave-one-perturbation-out estimate of exactly the
transfer the submission has to perform.

Early stopping watches the locally recomputed official score_avg on H1, never the
training loss and never an MAE proxy.

Result, stated plainly: on this data the residual does not help.  Across three
configurations (lr 1e-3..3e-3, then 1e-4..5e-4, then with the amplitude losses
switched off and the rescaling capped at +-30%) and eight seeds each, every run
peaked at epoch 1-3 and the best checkpoint scored +0.121..+0.128 against the
+0.125 of the zero-residual model it starts from.  Averaging all eight gives
+0.124, still below.  The submission therefore ships the zero-residual path.

Two things cause it, and both are properties of the corpus rather than bugs:

  * the destinations run 45-83 cells per perturbation, so their fold changes are
    mostly sampling error, and the amplitude losses are minimised by shrinking
    toward zero -- which destroys the predicted call count that fidelity needs;
  * their own BH significant counts (16-307) are an order of magnitude below the
    ~900 the 400-vs-18400 reference calls, so the coverage term saturates at the
    wrong place and stops asking for more calls.

What did carry the result was the amplitude compression in net.compress and the
call-count control in metrics.sparsify, both calibrated on held-out H1.

Run one process per GPU (see scripts/launch8.sh):
    python run.py train --preset launch8_g0 --gpu 0
`--final` retrains on every context including H1 for a fixed number of epochs.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

import metrics as M
from losses import SixScoreLoss
from models import LOCAL_FEATS, PertResponseNet, build_scalars, compress, n_params, shrink_lfc
from models.priors import UNMAPPED_GWPS, load_pert_annotation, pert_ann_tensor
from inference.output import CENTER_W, decenter
from paths import DATA, PREP, RUNS
ESM_PATH = DATA / "gene_priors/full_esm_embeddings.npz"

SRC = "gwps"
# k562ess is a second screen of the *same* line as the source, so it anchors the
# residual: with nothing to transfer across, the correct correction is zero.
# Without it the model is only ever shown destinations that differ, and it learns
# to always modify the source signal.
TRAIN_LINES = ("rpe1", "hepg2", "jurkat", "k562ess")
VAL_LINE = "h1"
GEN_SCALE, GEN_TOPK = 0.5, 7000     # matched to net.LFC_GAMMA; see runs/shape2.json
TEMP_LIMIT, TEMP_RESUME = 82, 75


# ------------------------------------------------------------------ gpu health

def gpu_temp(idx: int) -> int:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader", "-i", str(idx)],
            timeout=10)
        return int(out.decode().split()[0])
    except Exception:
        return 0


def cool_down(idx: int, log) -> None:
    t = gpu_temp(idx)
    if t < TEMP_LIMIT:
        return
    log(f"gpu {idx} at {t}C, pausing until {TEMP_RESUME}C")
    while gpu_temp(idx) > TEMP_RESUME:
        time.sleep(15.0)
    log(f"gpu {idx} back to {gpu_temp(idx)}C")


# --------------------------------------------------------------------- corpus

class Corpus:
    """All sources aligned onto one working gene space in the 18533 panel."""

    def __init__(self, device: str, lines: tuple[str, ...]):
        self.device = device
        raw = {n: dict(np.load(PREP / f"{n}.npz", allow_pickle=True))
               for n in (SRC, *lines, VAL_LINE)}
        ctx = {c: dict(np.load(PREP / f"ctx_{c}.npz")) for c in "ABC"}

        # working space W: any gene the 2026 evaluation can test
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

        self.lines: dict[str, dict] = {}
        for name, d in raw.items():
            self.lines[name] = self._pack(d, w_pos)
        for c in "ABC":
            self.lines[f"ctx_{c}"] = self._pack_ctx(ctx[c], w_pos)

        # leave-one-context-out mean response prior
        self.prior = {}
        pool = [SRC, *lines, VAL_LINE]
        for name in pool:
            others = [self.lines[o]["mean_lfc"] for o in pool if o != name]
            self.prior[name] = np.mean(others, 0).astype(np.float32)
        self.prior_all = np.mean([self.lines[o]["mean_lfc"] for o in pool], 0).astype(np.float32)

        cx = np.stack([self.lines[n]["ctx_expr"] for n in self.lines])
        self.ctx_mu, self.ctx_sd = cx.mean(0), cx.std(0) + 1e-3
        self.esm = self._load_esm()
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
        ctx = (t["ctx_expr"] - self.ctx_mu) / self.ctx_sd
        esm = self.esm.get(pert, self.esm_zero)
        sc = build_scalars(sig_s, lfc_s, univ, t["ctrl"], int(univ.sum()))
        ann = pert_ann_tensor(self.pert_ann, pert)
        return local, ctx, esm, sc, prior, univ, ann

    def pairs(self, lines: tuple[str, ...]) -> list[tuple[str, str]]:
        src = set(self.lines[SRC]["perts"])
        out = []
        for ln in lines:
            for p in self.lines[ln]["perts"]:
                if p in src and p not in UNMAPPED_GWPS:
                    out.append((p, ln))
        return out


def batchify(corpus: Corpus, items: list[tuple[str, str]], device: str):
    loc, ctx, esm, sca, pri, uni, tgt_lfc, tgt_sig, pann = [], [], [], [], [], [], [], [], []
    for p, ln in items:
        l, c, e, s, pr, u, ann = corpus.features(p, ln)
        loc.append(l); ctx.append(c); esm.append(e); sca.append(s); pri.append(pr); uni.append(u)
        pann.append(ann)
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

    return (t(loc), t(ctx), t(esm), t(sca), t(pri), t(uni), t(tgt_lfc), t(tgt_sig), t(pann))


# ------------------------------------------------------------------ validation

class Validator:
    """Rebuilds the official six on the held-out H1 context each epoch."""

    def __init__(self, corpus: Corpus, device: str, seed: int = 0):
        self.corpus = corpus
        self.device = device
        panel, real, ctrl, meta = M.load_h1_eval(PREP / "h1_eval.h5", device)
        eval_perts = list(meta["perts"])
        src = set(corpus.lines[SRC]["perts"])
        h1 = corpus.lines[VAL_LINE]
        self.use = [i for i, p in enumerate(eval_perts) if p in src and p in h1["index"]]
        assert self.use, "no H1 evaluation perturbation is covered by the source"
        self.perts = [eval_perts[i] for i in self.use]

        # restrict the reference panel to the covered perturbations
        sub_real = [real[i] for i in self.use]
        own = np.array([panel.own_idx[i] for i in self.use], np.int64)
        self.panel = M.Panel(sub_real, ctrl, own, panel.drop_pds)
        self.ctrl = ctrl
        self.ctrl_mean = self.panel.de.mean.to(torch.float32)
        self.n_out = sub_real[0].shape[0]
        # map the H1 evaluation gene axis onto the working space W
        w_pos = -np.ones(18533, np.int64)
        w_pos[corpus.W] = np.arange(corpus.n_w)
        col = w_pos[meta["panel_idx"].astype(np.int64)]
        self.eval_in_w = torch.as_tensor(np.where(col >= 0, col, 0), device=device)
        self.eval_valid = torch.as_tensor(col >= 0, device=device)
        # a gene the destination context does not express cannot respond in it, so
        # calling one only dilutes directional precision
        self.univ_w = torch.as_tensor(
            corpus.lines[VAL_LINE]["univ"].astype(np.float32), device=device)
        self.gen = torch.Generator(device=device).manual_seed(seed)
        self.n_real = float(self.panel.real_sig().sum(1).to(torch.float64).median())
        print(f"[val] {len(self.perts)} H1 perturbations, {self.n_out} cells each, "
              f"{ctrl.shape[1]} eval genes ({int(self.eval_valid.sum())} inside W), "
              f"reference median n_sig={self.n_real:.0f}", flush=True)

    @torch.no_grad()
    def predict(self, model) -> tuple[torch.Tensor, torch.Tensor]:
        model.eval()
        lfc, sig = [], []
        for i in range(0, len(self.perts), 24):
            chunk = [(p, VAL_LINE) for p in self.perts[i : i + 24]]
            loc, ctx, esm, sca, pri, uni, _, _, pann = batchify(self.corpus, chunk, self.device)
            a, b, _ = model(loc, ctx, esm, sca, pri, pann)
            lfc.append(a)
            sig.append(b)
        model.train()
        return torch.cat(lfc), torch.cat(sig)

    @torch.no_grad()
    def run(self, model, scale=GEN_SCALE, topk=GEN_TOPK,
            pre=None, center_w: float = CENTER_W) -> dict[str, float]:
        lfc_w, sig_w = self.predict(model) if pre is None else pre
        lfc_w = decenter(lfc_w, center_w)

        cells = []
        for i in range(len(self.perts)):
            v = (lfc_w[i] * self.univ_w)[self.eval_in_w] * self.eval_valid
            s = torch.where(self.eval_valid, sig_w[i][self.eval_in_w],
                            torch.full_like(v, -1e9))
            cells.append(M.synthesize_cells(M.sparsify(v, s, topk), self.ctrl,
                                            self.ctrl_mean, self.n_out, scale, self.gen))
        raw = M.official_six(cells, self.panel)
        # fidelity is precision x capped coverage, so the call count says which
        # half is binding; without it a low fidelity is uninterpretable
        n_pred = float(np.mean([int(self.panel.de.table(c)[1].sum())
                                for c in cells[: min(32, len(cells))]]))
        return {**{f"raw_{k}": raw[k] for k in M.KEYS}, **M.scale(raw),
                "n_pred": n_pred, "n_real": self.n_real}


# ----------------------------------------------------------------------- train

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--tag", default="s0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--wd", type=float, default=3e-2)
    ap.add_argument("--hidden", type=int, default=192)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--drop", type=float, default=0.15)
    ap.add_argument("--patience", type=int, default=40)
    ap.add_argument("--final", action="store_true",
                    help="train on every context including H1, no early stop")
    ap.add_argument("--final-epochs", type=int, default=0)
    ap.add_argument("--time-budget", type=float, default=1e9, help="seconds")
    args = ap.parse_args()

    device = f"cuda:{args.gpu}"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out = RUNS / args.tag
    out.mkdir(parents=True, exist_ok=True)
    logf = (out / "train_log.jsonl").open("a")

    def log(msg, **kw):
        print(f"[{args.tag}] {msg}", flush=True)
        logf.write(json.dumps({"t": time.time(), "msg": msg, **kw}) + "\n")
        logf.flush()

    lines = (*TRAIN_LINES, VAL_LINE) if args.final else TRAIN_LINES
    corpus = Corpus(device, TRAIN_LINES)
    items = corpus.pairs(lines)
    rng = np.random.default_rng(args.seed)
    log(f"training pairs = {len(items)} over {lines}",
        n_pairs=len(items), lines=list(lines), n_w=corpus.n_w)
    per_line = {ln: sum(1 for _, l in items if l == ln) for ln in lines}
    log(f"pairs per destination: {per_line}", per_line=per_line)

    model = PertResponseNet(corpus.C.size, corpus.esm_dim, args.hidden,
                            args.depth, drop=args.drop).to(device)
    log(f"model has {n_params(model)} parameters", n_params=n_params(model))
    loss_fn = SixScoreLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.epochs * max(1, len(items) // args.batch),
        pct_start=0.15)

    pds_mask = torch.ones(corpus.n_w, device=device)
    tgt_panel = _target_panel_mask(corpus)
    pds_mask[tgt_panel] = 0.0

    validator = None if args.final else Validator(corpus, device, args.seed)
    best, best_ep, wait, t0 = -1e9, -1, 0, time.time()
    stop_ep = args.final_epochs if (args.final and args.final_epochs) else args.epochs

    for ep in range(1, stop_ep + 1):
        order = rng.permutation(len(items))
        seen, run_loss, parts_acc = 0, 0.0, {}
        for b in range(0, len(order) - args.batch + 1, args.batch):
            sel = [items[i] for i in order[b : b + args.batch]]
            loc, ctx, esm, sca, pri, uni, t_lfc, t_sig, pann = batchify(corpus, sel, device)
            lfc, sig_logit, gate = model(loc, ctx, esm, sca, pri, pann)
            loss, parts = loss_fn(lfc, sig_logit, t_lfc, t_sig, uni, pds_mask)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            seen += len(sel)
            run_loss += float(loss.detach()) * len(sel)
            for k, v in parts.items():
                parts_acc[k] = parts_acc.get(k, 0.0) + v * len(sel)
            if (b // args.batch) % 200 == 0:
                cool_down(args.gpu, log)

        # every pair is either used or explicitly accounted for by the drop-last remainder
        dropped = len(items) - seen
        assert 0 <= dropped < args.batch, f"lost {dropped} training pairs"
        rec = {"epoch": ep, "seen": seen, "dropped_last_batch": dropped,
               "loss": run_loss / max(seen, 1), "gate": float(gate.detach().mean()),
               **{f"L_{k}": v / max(seen, 1) for k, v in parts_acc.items()}}

        if validator is not None:
            v = validator.run(model)
            rec.update(v)
            score = v["score_avg"]
            if score > best + 1e-5:
                best, best_ep, wait = score, ep, 0
                torch.save({"model": model.state_dict(), "args": vars(args),
                            "epoch": ep, "score": score, "val": v}, out / "best.pt")
            else:
                wait += 1
            log(f"ep {ep} loss {rec['loss']:.4f} score_avg {score:+.4f} "
                f"(best {best:+.4f} @ {best_ep}) fid {v['raw_fid']:.3f} "
                f"pds {v['raw_pds']:.3f} jac {v['raw_jac']:.4f} nmae {v['raw_nmae']:.3f} "
                f"mse {v['raw_mse']:.3f} n_pred {v['n_pred']:.0f}/{v['n_real']:.0f} "
                f"gate {rec['gate']:.2f}", **rec)
            if wait >= args.patience:
                log(f"early stop at {ep}, best {best:+.4f} @ {best_ep}")
                break
        elif ep % 10 == 0 or ep == 1:
            log(f"ep {ep} loss {rec['loss']:.4f} gate {rec['gate']:.3f}", **rec)

        if time.time() - t0 > args.time_budget:
            log(f"time budget reached at epoch {ep}")
            break

    if args.final:
        torch.save({"model": model.state_dict(), "args": vars(args),
                    "epoch": stop_ep, "score": None}, out / "final.pt")
    log(f"done in {time.time() - t0:.0f}s | best {best:+.4f} @ epoch {best_ep}",
        best=best, best_epoch=best_ep)


def _target_panel_mask(corpus: Corpus) -> torch.Tensor:
    """Positions in W of the 300 panel target genes, excluded from the PDS surrogate."""
    import pandas as pd
    genes = pd.read_csv(DATA / "challenge_2026/gene_names.csv")["gene_name"].astype(str).to_numpy()
    targets = set(pd.read_csv(DATA / "challenge_2026/pert_counts.csv")["target_gene"].astype(str))
    is_t = np.array([g in targets for g in genes])
    w_pos = -np.ones(18533, np.int64)
    w_pos[corpus.W] = np.arange(corpus.n_w)
    pos = w_pos[np.flatnonzero(is_t)]
    return torch.as_tensor(pos[pos >= 0], dtype=torch.long)


if __name__ == "__main__":
    main()
