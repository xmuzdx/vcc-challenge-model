"""实验预设、损失权重与 checkpoint 注册表。

本次实验（FINAL 提交）已全部登记：
  - 模型 pert_response
  - 损失 six_score / six_score_uniform
  - 权重 default / no_mse / uniform
  - checkpoint untrained / g0..g7 / ensemble8
  - 实验预设 final / launch8_g* / launch8_all
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

from losses import LOSS_W, LOSS_W_V2, weights_from_mag
from paths import ROOT, RUNS

# ------------------------------------------------------------------ 损失权重

WEIGHT_PRESETS: dict[str, dict[str, float]] = {
    "default": dict(LOSS_W),
    "no_mse": weights_from_mag({"nmae": 0.06, "mse": 0.0}),
    "uniform": {k: 1.0 / 6 for k in LOSS_W},
    "full_span": weights_from_mag({"nmae": 1.0, "mse": 1.0}),
    "v2": dict(LOSS_W_V2),
}

# ------------------------------------------------------------------ checkpoint

CKPT_PRESETS: dict[str, str | list[str]] = {
    "untrained": "untrained",
    "g0": str(RUNS / "g0" / "best.pt"),
    "g1": str(RUNS / "g1" / "best.pt"),
    "g2": str(RUNS / "g2" / "best.pt"),
    "g3": str(RUNS / "g3" / "best.pt"),
    "g4": str(RUNS / "g4" / "best.pt"),
    "g5": str(RUNS / "g5" / "best.pt"),
    "g6": str(RUNS / "g6" / "best.pt"),
    "g7": str(RUNS / "g7" / "best.pt"),
    "ensemble8": [str(RUNS / f"g{i}" / "best.pt") for i in range(8)],
    "zero_fan": str(RUNS / "zero_fan" / "best.pt"),
    "zero_fan_final": str(RUNS / "zero_fan_final" / "final.pt"),
    "zero_fan_v2": str(RUNS / "zero_fan_v2" / "best.pt"),
}

# launch8.sh 八卡配置
_LAUNCH8 = (
    dict(seed=0, lr=1e-4, hidden=192, depth=3, drop=0.20, wd=0.05, tag="g0"),
    dict(seed=1, lr=2e-4, hidden=192, depth=3, drop=0.20, wd=0.05, tag="g1"),
    dict(seed=2, lr=1e-4, hidden=256, depth=3, drop=0.30, wd=0.10, tag="g2"),
    dict(seed=3, lr=3e-4, hidden=192, depth=2, drop=0.15, wd=0.05, tag="g3"),
    dict(seed=4, lr=2e-4, hidden=128, depth=4, drop=0.20, wd=0.05, tag="g4"),
    dict(seed=5, lr=5e-5, hidden=256, depth=2, drop=0.30, wd=0.10, tag="g5"),
    dict(seed=6, lr=3e-4, hidden=192, depth=3, drop=0.35, wd=0.10, tag="g6"),
    dict(seed=7, lr=5e-4, hidden=192, depth=3, drop=0.20, wd=0.05, tag="g7"),
)


@dataclass
class ExperimentConfig:
    """统一超参；空字符串 / None 表示采用默认值。"""

    # 注册名
    model: str = "pert_response"
    loss: str = "six_score"
    weight: str = "default"
    ckpt: str = "untrained"
    preset: str = ""

    # 训练
    epochs: int = 300
    batch: int = 24
    lr: float = 2e-3
    wd: float = 3e-2
    hidden: int | None = None
    depth: int | None = None
    drop: float | None = None
    cond: int | None = None
    patience: int = 40
    seed: int = 0
    gpu: int = 0
    tag: str = "s0"
    final: bool = False
    final_epochs: int = 0
    time_budget: float = 1e9

    # 评估 / 提交输出层
    scale: float = 0.5
    topk: int = 7000
    gamma: float = 0.35
    eval_seed: int = 0

    # 提交
    submit_tag: str = "six_aligned_v1"
    calibrate: bool = False

    # zero_fan
    adaptive_topk: bool = False
    balanced_sampler: bool = False
    use_ddp: bool = False
    zero_w: float = 0.0
    mmd_w: float = 0.0
    tta: bool = False
    keep_q: float = 0.2
    center_w: float = 0.3
    n_prog: int = 64

    extra: dict[str, Any] = field(default_factory=dict)

    def resolve_ckpt_paths(self) -> list[str]:
        if self.ckpt in CKPT_PRESETS:
            v = CKPT_PRESETS[self.ckpt]
            return [v] if isinstance(v, str) else list(v)
        p = Path(self.ckpt)
        if p.exists():
            return [str(p)]
        raise FileNotFoundError(f"未知 checkpoint '{self.ckpt}'，可选: {', '.join(CKPT_PRESETS)}")

    def loss_weights(self) -> dict[str, float]:
        if self.weight in WEIGHT_PRESETS:
            return dict(WEIGHT_PRESETS[self.weight])
        raise KeyError(f"未知 weight '{self.weight}'，可选: {', '.join(WEIGHT_PRESETS)}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _merge(base: ExperimentConfig, **overrides) -> ExperimentConfig:
    d = base.to_dict()
    for k, v in overrides.items():
        if v is None or v == "":
            continue
        if k in d:
            d[k] = v
        else:
            d.setdefault("extra", {})[k] = v
    return ExperimentConfig(**{f.name: d[f.name] for f in fields(ExperimentConfig)})


def from_preset(name: str, **overrides) -> ExperimentConfig:
    """按预设名构建配置，overrides 覆盖非空字段。"""
    presets = build_presets()
    if name not in presets:
        known = ", ".join(sorted(presets))
        raise KeyError(f"未知 preset '{name}'，可选: {known}")
    cfg = presets[name]
    cfg.preset = name
    return _merge(cfg, **overrides)


def build_presets() -> dict[str, ExperimentConfig]:
    """全部内置预设。"""
    base_train = ExperimentConfig(
        model="pert_response",
        loss="six_score",
        weight="default",
        batch=24,
        patience=25,
        time_budget=11400.0,
    )

    launch8: dict[str, ExperimentConfig] = {}
    for i, kw in enumerate(_LAUNCH8):
        launch8[f"launch8_g{i}"] = _merge(
            base_train,
            preset=f"launch8_g{i}",
            epochs=300,
            **kw,
        )
    launch8["launch8_all"] = _merge(base_train, preset="launch8_all", tag="multi")

    final = ExperimentConfig(
        preset="final",
        model="pert_response",
        loss="six_score",
        weight="default",
        ckpt="untrained",
        scale=0.5,
        topk=7000,
        gamma=0.35,
        submit_tag="six_aligned_v1",
    )

    zf = ExperimentConfig(
        preset="zero_fan",
        model="zero_fan",
        loss="six_score_v2",
        weight="v2",
        hidden=256,
        depth=6,
        drop=0.35,
        cond=1024,
        lr=1e-4,
        wd=0.10,
        batch=8,
        epochs=80,
        patience=12,
        scale=0.5,
        topk=7000,
        gamma=0.35,
        balanced_sampler=True,
        zero_w=0.05,
        mmd_w=0.02,
        tag="zero_fan",
        submit_tag="zero_fan_v1",
    )
    zf_final = _merge(
        zf, preset="zero_fan_final", final=True, final_epochs=8,
        tag="zero_fan_final", ckpt="zero_fan_final",
    )

    zf2 = ExperimentConfig(
        preset="zero_fan_v2",
        model="zero_fan_v2",
        loss="six_score_v2",
        weight="v2",
        hidden=256,
        depth=4,
        drop=0.30,
        cond=512,
        lr=1e-4,
        wd=0.10,
        batch=16,
        epochs=80,
        patience=12,
        scale=0.5,
        topk=7000,
        gamma=0.35,
        balanced_sampler=True,
        zero_w=0.05,
        mmd_w=0.02,
        center_w=0.3,
        n_prog=64,
        tag="zero_fan_v2",
        submit_tag="zero_fan_v2",
    )

    return {
        "default": ExperimentConfig(),
        "final": final,
        "zero_fan": zf,
        "zero_fan_final": zf_final,
        "zero_fan_v2": zf2,
        **launch8,
    }


def list_presets() -> dict[str, str]:
    desc = {
        "default": "全局默认超参",
        "final": "FINAL 提交：zero-residual + scale=0.5 topk=7000",
        "zero_fan": "二代 ZeroFan：H1 留出训练 + six_score_v2",
        "zero_fan_final": "二代 ZeroFan：并入 H1 全量重训",
        "zero_fan_v2": "三代 ZeroFanV2：程序分解 + 双路融合 + |z| 排序",
        "launch8_all": "八卡 sweep 元信息（需 launch8_all 脚本）",
    }
    for i in range(8):
        c = _LAUNCH8[i]
        desc[f"launch8_g{i}"] = (
            f"launch8 GPU{i}: seed={c['seed']} lr={c['lr']} "
            f"hidden={c['hidden']} depth={c['depth']} drop={c['drop']} wd={c['wd']}"
        )
    return desc


def list_weights() -> dict[str, dict[str, float]]:
    return {k: dict(v) for k, v in WEIGHT_PRESETS.items()}


def list_ckpts() -> dict[str, str | list[str]]:
    return dict(CKPT_PRESETS)
