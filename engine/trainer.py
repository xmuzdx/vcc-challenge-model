"""训练执行器：复用 train.py 组件，支持 registry 中的 loss 权重。"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from config.registry import ExperimentConfig
from engine import ddp as D
from engine.sampler import LineBalancedSampler
from engine.train import (
    Corpus,
    TRAIN_LINES,
    Validator,
    _target_panel_mask,
    batchify,
    cool_down,
    n_params,
)
from inference.output import resolve_topk
from losses import MMDLoss, ZeroPertLoss, get_loss
from losses.aux import soft_synth_cells
from models import get_model
from models.base import ModelParams
from models.priors import panel_features_on
from models.programs import load_programs
from paths import ROOT, RUNS


def _uses_legacy_train(cfg: ExperimentConfig) -> bool:
    return (
        cfg.model == "pert_response"
        and cfg.loss == "six_score"
        and cfg.weight == "default"
        and not cfg.adaptive_topk
        and not cfg.balanced_sampler
        and not cfg.use_ddp
        and not cfg.final
    )


def train_via_subprocess(cfg: ExperimentConfig) -> Path:
    """默认损失时直接调用 train.py，行为与原版完全一致。"""
    cmd = [
        sys.executable, str(ROOT / "engine" / "train.py"),
        "--gpu", str(cfg.gpu),
        "--tag", cfg.tag,
        "--seed", str(cfg.seed),
        "--epochs", str(cfg.epochs),
        "--batch", str(cfg.batch),
        "--lr", str(cfg.lr),
        "--wd", str(cfg.wd),
        "--patience", str(cfg.patience),
        "--time-budget", str(cfg.time_budget),
    ]
    model_spec = get_model(cfg.model)
    mp = ModelParams(hidden=cfg.hidden, depth=cfg.depth, drop=cfg.drop)
    for k, v in model_spec.train_args(mp).items():
        cmd.extend([f"--{k.replace('_', '-')}", str(v)])
    if cfg.final:
        cmd.append("--final")
    if cfg.final_epochs:
        cmd.extend(["--final-epochs", str(cfg.final_epochs)])

    print("[trainer] subprocess:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=str(ROOT))
    return RUNS / cfg.tag / ("final.pt" if cfg.final else "best.pt")


def _sample_w(corpus: Corpus, sel: list[tuple[str, str]], device: str):
    w = []
    for p, ln in sel:
        line = corpus.lines[ln]
        j = line["index"].get(p, -1)
        n = float(line["n_cells"][j]) if j >= 0 and "n_cells" in line else 1.0
        w.append(np.sqrt(max(n, 1.0)))
    return torch.as_tensor(w, dtype=torch.float32, device=device)


def _zero_pert_batch(corpus: Corpus, dests: list[str], device: str, n: int = 2):
    """源响应全零的假扰动，用来钉住零扰动约束。"""
    items = []
    for i, ln in enumerate(dests[:n]):
        # reuse a real dest context so ctrl / universe are valid
        p = next(iter(corpus.lines[ln]["index"]))
        items.append((p, ln))
    loc, ctx, esm, sca, pri, uni, _, _, pann = batchify(corpus, items, device)
    loc = loc.clone()
    loc[..., 0] = 0.0          # lfc_src
    loc[..., 1] = 0.0          # zn_src
    loc[..., 2] = 0.0          # sig_src
    esm = torch.zeros_like(esm)
    pann = torch.zeros_like(pann)
    return loc, ctx, esm, sca, pri, uni, pann


def train_inprocess(cfg: ExperimentConfig) -> Path:
    """zero_fan / 自定义 loss / DDP / 平衡采样 的统一训练环。"""
    if D.enabled():
        cfg.use_ddp = True
    rank, world, device = D.setup()
    if not D.enabled():
        device = torch.device(f"cuda:{cfg.gpu}")
        torch.cuda.set_device(cfg.gpu)
    torch.manual_seed(cfg.seed + rank)
    np.random.seed(cfg.seed + rank)
    out = RUNS / cfg.tag
    if D.is_rank0():
        out.mkdir(parents=True, exist_ok=True)
    D.barrier()
    logf = (out / "train_log.jsonl").open("a") if D.is_rank0() else None

    def log(msg, **kw):
        if not D.is_rank0():
            return
        print(f"[{cfg.tag}] {msg}", flush=True)
        logf.write(json.dumps({"t": time.time(), "msg": msg, **kw}) + "\n")
        logf.flush()

    lines = (*TRAIN_LINES, "h1") if cfg.final else TRAIN_LINES
    corpus = Corpus(str(device), TRAIN_LINES)
    items = corpus.pairs(lines)
    rng = np.random.default_rng(cfg.seed + rank)

    model_spec = get_model(cfg.model)
    extra = {**cfg.extra, "n_prog": cfg.n_prog}
    mp = ModelParams(hidden=cfg.hidden, depth=cfg.depth, drop=cfg.drop,
                     cond=cfg.cond, extra=extra)
    raw = model_spec.build(corpus.C.size, corpus.esm_dim, mp).to(device)
    if hasattr(raw, "attach_gene_space"):
        raw.attach_gene_space(panel_features_on(corpus.W).to(device))
    if hasattr(raw, "attach_programs"):
        raw.attach_programs(load_programs(corpus, k=cfg.n_prog).to(device))
    log(f"model={cfg.model} params={n_params(raw)}", n_params=n_params(raw))
    model = D.wrap(raw) if cfg.use_ddp else raw

    loss_fn = get_loss(cfg.loss).build(cfg.loss_weights())
    zero_fn = ZeroPertLoss()
    mmd_fn = MMDLoss()
    log(f"loss={cfg.loss} weight={cfg.weight}")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    steps_ep = max(1, (len(items) // cfg.batch + world - 1) // world)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=cfg.lr, total_steps=max(cfg.epochs * steps_ep, 1), pct_start=0.15)
    pds_mask = torch.ones(corpus.n_w, device=device)
    pds_mask[_target_panel_mask(corpus)] = 0.0

    pool = None
    if cfg.mmd_w > 0:
        try:
            from data.cellpool import CellPool
            pool = CellPool(lines, corpus.W, str(device))
            if not pool.lines:
                pool = None
                log("cellpool empty, MMD disabled")
        except Exception as e:
            log(f"cellpool load failed ({e}), MMD disabled")
            pool = None

    val_topk = resolve_topk(cfg.topk, corpus, "h1", cfg.adaptive_topk)
    validator = None
    if (not cfg.final) and D.is_rank0():
        validator = Validator(corpus, str(device), cfg.seed)
    best, best_ep, wait, t0 = -1e9, -1, 0, time.time()
    stop_ep = cfg.final_epochs if (cfg.final and cfg.final_epochs) else cfg.epochs
    sampler = LineBalancedSampler(items, cfg.batch, rng) if cfg.balanced_sampler else None

    for ep in range(1, stop_ep + 1):
        model.train()
        if sampler is not None:
            batches = sampler.epoch()
        else:
            order = rng.permutation(len(items))
            batches = [[items[i] for i in order[b:b + cfg.batch]]
                       for b in range(0, len(order) - cfg.batch + 1, cfg.batch)]
        if world > 1:
            batches = batches[rank::world]

        seen, run_loss, parts_acc = 0, 0.0, {}
        for b, sel in enumerate(batches):
            loc, ctx, esm, sca, pri, uni, t_lfc, t_sig, pann = batchify(
                corpus, sel, str(device))
            sw = _sample_w(corpus, sel, str(device))
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                lfc, sig_logit, gate = model(loc, ctx, esm, sca, pri, pann)
                loss, parts = loss_fn(lfc, sig_logit, t_lfc, t_sig, uni, pds_mask, sw)
                if cfg.zero_w > 0:
                    zloc, zctx, zesm, zsca, zpri, zuni, zpann = _zero_pert_batch(
                        corpus, [ln for _, ln in sel], str(device))
                    zlfc, _, _ = model(zloc, zctx, zesm, zsca, zpri, zpann)
                    loss = loss + cfg.zero_w * zero_fn(zlfc, zuni)
                if cfg.mmd_w > 0 and pool is not None:
                    mmd = lfc.new_zeros(())
                    n_m = 0
                    for i, (p, ln) in enumerate(sel[:4]):
                        if not pool.available(ln):
                            continue
                        ctrl, real, col = pool.sample(p, ln, 16, 16, rng)
                        if real is None:
                            continue
                        pred = soft_synth_cells(ctrl, lfc[i, col].float(), cfg.scale)
                        mmd = mmd + mmd_fn(pred, real)
                        n_m += 1
                    if n_m:
                        loss = loss + cfg.mmd_w * (mmd / n_m)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            seen += len(sel)
            run_loss += float(loss.detach()) * len(sel)
            for k, v in parts.items():
                parts_acc[k] = parts_acc.get(k, 0.0) + v * len(sel)
            if b % 200 == 0:
                cool_down(0 if device.index is None else device.index, log)

        rec = {"epoch": ep, "seen": seen,
               "loss": run_loss / max(seen, 1),
               "gate": float(gate.detach().mean()),
               **{f"L_{k}": v / max(seen, 1) for k, v in parts_acc.items()}}

        stop = False
        if validator is not None and D.is_rank0():
            v = validator.run(D.unwrap(model), scale=cfg.scale, topk=val_topk,
                              center_w=cfg.center_w)
            rec.update(v)
            score = v["score_avg"]
            if score > best + 1e-5:
                best, best_ep, wait = score, ep, 0
                D.save_best(out / "best.pt", {
                    "model": D.unwrap(model).state_dict(),
                    "args": {**cfg.to_dict(), "epoch": ep},
                    "epoch": ep, "score": score, "val": v,
                })
            else:
                wait += 1
            log(f"ep {ep} score_avg {score:+.4f} (best {best:+.4f} @ {best_ep}) "
                f"topk={val_topk}", **rec)
            if wait >= cfg.patience:
                log(f"early stop @ {ep}, best {best:+.4f} @ {best_ep}")
                stop = True
        elif D.is_rank0() and (ep % 10 == 0 or ep == 1):
            log(f"ep {ep} loss {rec['loss']:.4f}", **rec)

        if time.time() - t0 > cfg.time_budget:
            log(f"time budget reached @ epoch {ep}")
            stop = True
        if D.broadcast_flag(stop, device):
            break

    if cfg.final:
        D.save_best(out / "final.pt", {
            "model": D.unwrap(model).state_dict(),
            "args": cfg.to_dict(), "epoch": stop_ep, "score": None,
        })
        D.cleanup()
        return out / "final.pt"
    D.cleanup()
    return out / "best.pt"


def train(cfg: ExperimentConfig) -> Path:
    if _uses_legacy_train(cfg):
        return train_via_subprocess(cfg)
    return train_inprocess(cfg)
