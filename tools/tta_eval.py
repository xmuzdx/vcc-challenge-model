#!/usr/bin/env python3
"""在 H1 上跑 MC dropout TTA，并与确定性前向对比官方六项。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.train import Corpus, TRAIN_LINES, Validator
from inference.output import CALIBRATED_SCALE, CALIBRATED_TOPK
from inference.submit import load_models
from inference.tta import confidence_gate, mc_dropout_predict
from paths import RUNS


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--ckpt", default="untrained")
    ap.add_argument("--model", default="pert_response")
    ap.add_argument("--n-fwd", type=int, default=8)
    ap.add_argument("--keep-q", type=float, default=0.2)
    args = ap.parse_args()

    device = f"cuda:{args.gpu}"
    corpus = Corpus(device, TRAIN_LINES)
    models = load_models([Path(args.ckpt)], corpus, device, model_name=args.model)
    val = Validator(corpus, device, 0)

    det = val.run(models[0], scale=CALIBRATED_SCALE, topk=CALIBRATED_TOPK)
    print(f"[tta] deterministic score_avg={det['score_avg']:+.4f}", flush=True)

    lfc, sig, std = mc_dropout_predict(
        models[0], corpus, val.perts, "h1", device, n_fwd=args.n_fwd)
    gated = confidence_gate(lfc, std, args.keep_q)
    tta = val.run(models[0], scale=CALIBRATED_SCALE, topk=CALIBRATED_TOPK,
                  pre=(gated, sig))
    print(f"[tta] mc-dropout+gate q={args.keep_q} "
          f"score_avg={tta['score_avg']:+.4f}", flush=True)

    out = RUNS / "zero_fan_diag" / "tta_eval.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"det": det, "tta": tta, "keep_q": args.keep_q,
                               "n_fwd": args.n_fwd}, indent=2))
    print(f"[tta] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
