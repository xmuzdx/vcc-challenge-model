"""VCC 扰动响应预测项目（SMP 风格包结构）。

快速开始::

    from experiment import Experiment
    from models import get_model, PertResponseNet
    from losses import get_loss, SixScoreLoss

    Experiment.from_preset("final").evaluate()
"""

from config.registry import ExperimentConfig, build_presets, from_preset
from experiment import Experiment, launch8
from losses import SixScoreLoss, get_loss
from models import PertResponseNet, ZeroFanNet, ZeroFanV2Net, get_model

__all__ = [
    "Experiment",
    "ExperimentConfig",
    "from_preset",
    "build_presets",
    "launch8",
    "get_model",
    "get_loss",
    "PertResponseNet",
    "ZeroFanNet",
    "ZeroFanV2Net",
    "SixScoreLoss",
]
