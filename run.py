#!/usr/bin/env python3
"""统一 CLI 入口：训练 / 评估 / 提交。

示例::

    python run.py catalog
    python run.py eval --preset final
    python run.py train --preset launch8_g0 --gpu 0
    bash scripts/launch8.sh
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from experiment import Experiment, launch8
from inference.submit import calibrate, load_models
from engine.train import Corpus, TRAIN_LINES


def _add_common(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--preset", default="", help="实验预设名")
    ap.add_argument("--model", default="", help="模型名，默认 pert_response")
    ap.add_argument("--loss", default="", help="损失名，默认 six_score")
    ap.add_argument("--weight", default="", help="损失权重预设，默认 default")
    ap.add_argument("--ckpt", default="", help="checkpoint 预设或路径")

    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--wd", type=float, default=None)
    ap.add_argument("--hidden", type=int, default=None)
    ap.add_argument("--depth", type=int, default=None)
    ap.add_argument("--drop", type=float, default=None)
    ap.add_argument("--cond", type=int, default=None)
    ap.add_argument("--patience", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--final-epochs", type=int, default=None)
    ap.add_argument("--time-budget", type=float, default=None)

    ap.add_argument("--scale", type=float, default=None)
    ap.add_argument("--topk", type=int, default=None)
    ap.add_argument("--gamma", type=float, default=None)
    ap.add_argument("--eval-seed", type=int, default=None)

    ap.add_argument("--submit-tag", default="")
    ap.add_argument("--adaptive-topk", action="store_true")
    ap.add_argument("--balanced-sampler", action="store_true")
    ap.add_argument("--use-ddp", action="store_true")
    ap.add_argument("--zero-w", type=float, default=None)
    ap.add_argument("--mmd-w", type=float, default=None)
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--keep-q", type=float, default=None)
    ap.add_argument("--center-w", type=float, default=None)
    ap.add_argument("--n-prog", type=int, default=None)


_STORE_TRUE = {"adaptive_topk", "balanced_sampler", "use_ddp", "tta", "final"}


def _kwargs(args: argparse.Namespace) -> dict:
    out = {}
    for k, v in vars(args).items():
        if k in ("command", "calibrate", "write", "json"):
            continue
        if v is None or v == "":
            continue
        key = k.replace("-", "_")
        if key in _STORE_TRUE and v is False:
            continue
        out[key] = v
    return out


def _make_experiment(args: argparse.Namespace) -> Experiment:
    if args.preset:
        return Experiment.from_preset(args.preset, **_kwargs(args))
    return Experiment(**_kwargs(args))


def cmd_catalog(_args: argparse.Namespace) -> None:
    print(json.dumps(Experiment.catalog(), indent=2, ensure_ascii=False))


def cmd_train(args: argparse.Namespace) -> None:
    exp = _make_experiment(args)
    print(exp.summary())
    exp.train()


def cmd_eval(args: argparse.Namespace) -> None:
    exp = _make_experiment(args)
    print(exp.summary())
    result = exp.evaluate()
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"score_avg = {result['score_avg']:+.6f}")


def cmd_submit(args: argparse.Namespace) -> None:
    exp = _make_experiment(args)
    print(exp.summary())
    if args.calibrate and args.write:
        path = exp.calibrate_and_submit()
    elif args.calibrate:
        device = f"cuda:{exp.cfg.gpu}"
        corpus = Corpus(device, TRAIN_LINES)
        paths = exp.cfg.resolve_ckpt_paths()
        models = load_models([Path(p) for p in paths], corpus, device)
        calibrate(models, corpus, device, exp.cfg.eval_seed)
        return
    elif args.write:
        path = exp.submit()
    else:
        print("请指定 --calibrate 和/或 --write", file=sys.stderr)
        sys.exit(2)
        return
    print(f"submission -> {path}")


def cmd_launch8(args: argparse.Namespace) -> None:
    launch8(epochs=args.epochs or 300, time_budget=args.time_budget or 11400.0)


def main() -> None:
    ap = argparse.ArgumentParser(description="VCC 模型实验统一入口")
    sub = ap.add_subparsers(dest="command", required=True)

    p_cat = sub.add_parser("catalog", help="列出 model/loss/weight/ckpt/preset")
    p_cat.set_defaults(func=cmd_catalog)

    p_train = sub.add_parser("train", help="训练")
    _add_common(p_train)
    p_train.set_defaults(func=cmd_train)

    p_eval = sub.add_parser("eval", help="H1 官方六项评估")
    _add_common(p_eval)
    p_eval.add_argument("--json", action="store_true")
    p_eval.set_defaults(func=cmd_eval)

    p_sub = sub.add_parser("submit", help="校准 / 写提交 h5ad")
    _add_common(p_sub)
    p_sub.add_argument("--calibrate", action="store_true")
    p_sub.add_argument("--write", action="store_true")
    p_sub.set_defaults(func=cmd_submit)

    p8 = sub.add_parser("launch8", help="八卡并行训练")
    p8.add_argument("--epochs", type=int, default=300)
    p8.add_argument("--time-budget", type=float, default=11400.0)
    p8.set_defaults(func=cmd_launch8)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
