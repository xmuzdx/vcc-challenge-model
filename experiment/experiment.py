"""Experiment 类：一行配置完成训练 / 评估 / 提交。"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

from config.registry import (
    ExperimentConfig,
    build_presets,
    from_preset,
    list_ckpts,
    list_presets,
    list_weights,
)
from engine.train import Corpus, TRAIN_LINES, Validator
from engine.trainer import train
from inference.submit import calibrate, load_models, write_submission
from losses import list_losses
from models import list_models
from paths import ROOT


class Experiment:
    """用法示例::

        exp = Experiment(model="pert_response", loss="six_score",
                         weight="default", ckpt="untrained",
                         epochs=300, scale=0.5, topk=7000)
        exp.train(gpu=0, tag="g0", seed=0)
        metrics = exp.evaluate()
        exp.submit()

        # 或直接使用 FINAL 预设
        Experiment.from_preset("final").evaluate()
    """

    def __init__(self, **kwargs: Any):
        defaults = build_presets()["default"]
        merged = {f.name: getattr(defaults, f.name) for f in defaults.__dataclass_fields__.values()}
        for k, v in kwargs.items():
            if v is None or v == "":
                continue
            if k in merged:
                merged[k] = v
            else:
                merged.setdefault("extra", {})[k] = v
        self.cfg = ExperimentConfig(**merged)

    @classmethod
    def from_preset(cls, name: str, **overrides) -> Experiment:
        return cls(**from_preset(name, **overrides).to_dict())

    # ------------------------------------------------------------------ 训练

    def train(self, **overrides) -> Path:
        for k, v in overrides.items():
            if v is None or v == "":
                continue
            if hasattr(self.cfg, k):
                setattr(self.cfg, k, v)
        ckpt = train(self.cfg)
        print(f"[experiment] 训练完成 -> {ckpt}", flush=True)
        return ckpt

    # ------------------------------------------------------------------ 评估

    def evaluate(self, **overrides) -> dict[str, float]:
        for k, v in overrides.items():
            if v is None or v == "":
                continue
            if hasattr(self.cfg, k):
                setattr(self.cfg, k, v)

        device = f"cuda:{self.cfg.gpu}"
        corpus = Corpus(device, TRAIN_LINES)
        paths = self.cfg.resolve_ckpt_paths()
        models = load_models([Path(p) for p in paths], corpus, device,
                             model_name=self.cfg.model, n_prog=self.cfg.n_prog)
        val = Validator(corpus, device, self.cfg.eval_seed)
        topk = self.cfg.topk
        if getattr(self.cfg, "adaptive_topk", False):
            from inference.output import resolve_topk
            topk = resolve_topk(self.cfg.topk, corpus, "h1", True)
        result = val.run(models[0], scale=self.cfg.scale, topk=topk,
                         center_w=self.cfg.center_w)
        print(f"[experiment] evaluate ckpt={self.cfg.ckpt} "
              f"scale={self.cfg.scale} topk={topk} "
              f"score_avg={result['score_avg']:+.4f}", flush=True)
        return result

    # ------------------------------------------------------------------ 提交

    def submit(self, **overrides) -> Path | None:
        for k, v in overrides.items():
            if v is None or v == "":
                continue
            if hasattr(self.cfg, k):
                setattr(self.cfg, k, v)

        device = f"cuda:{self.cfg.gpu}"
        corpus = Corpus(device, TRAIN_LINES)
        paths = self.cfg.resolve_ckpt_paths()
        models = load_models([Path(p) for p in paths], corpus, device,
                             model_name=self.cfg.model, n_prog=self.cfg.n_prog)
        for m in models:
            m._tta = bool(self.cfg.tta)
            m._keep_q = float(self.cfg.keep_q)

        scale, topk = self.cfg.scale, self.cfg.topk
        if self.cfg.calibrate:
            best = calibrate(models, corpus, device, self.cfg.eval_seed)
            scale = best["scale"] if scale is None else scale
            topk = best["topk"] if topk is None else topk

        return write_submission(
            models, corpus, device, scale, topk,
            self.cfg.submit_tag, self.cfg.eval_seed,
            center_w=self.cfg.center_w,
        )

    def calibrate_and_submit(self, **overrides) -> Path:
        overrides = dict(overrides)
        overrides["calibrate"] = True
        path = self.submit(**overrides)
        assert path is not None
        return path

    # ------------------------------------------------------------------ 工具

    def save_config(self, path: Path | None = None) -> Path:
        path = path or (ROOT / "runs" / f"{self.cfg.tag}_config.json")
        path.write_text(json.dumps(self.cfg.to_dict(), indent=2))
        return path

    def summary(self) -> str:
        c = self.cfg
        return (
            f"model={c.model} loss={c.loss} weight={c.weight} ckpt={c.ckpt}\n"
            f"epochs={c.epochs} lr={c.lr} wd={c.wd} batch={c.batch}\n"
            f"scale={c.scale} topk={c.topk} gamma={c.gamma}\n"
            f"tag={c.tag} gpu={c.gpu} seed={c.seed}"
        )

    @staticmethod
    def catalog() -> dict[str, Any]:
        return {
            "models": list_models(),
            "losses": list_losses(),
            "weights": list(list_weights()),
            "ckpts": list_ckpts(),
            "presets": list_presets(),
        }


def launch8(gpus: tuple[int, ...] = tuple(range(8)), epochs: int = 300,
            time_budget: float = 11400.0) -> None:
    """并行启动 launch8 八组实验（subprocess 调用 run.py train）。"""
    import os
    procs = []
    for i in gpus:
        preset = f"launch8_g{i}"
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(i)
        cmd = [
            sys.executable, str(ROOT / "run.py"), "train",
            "--preset", preset,
            "--gpu", "0",
            "--epochs", str(epochs),
            "--time-budget", str(time_budget),
        ]
        log = ROOT / "logs" / f"train_g{i}.log"
        log.parent.mkdir(exist_ok=True)
        print(f"[launch8] GPU {i} -> {preset}", flush=True)
        with log.open("a") as fh:
            procs.append(subprocess.Popen(cmd, cwd=str(ROOT), env=env,
                                          stdout=fh, stderr=subprocess.STDOUT))
    for p in procs:
        p.wait()
        if p.returncode != 0:
            raise RuntimeError(f"launch8 子进程失败 exit={p.returncode}")
