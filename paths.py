"""项目路径：无论代码位于哪个子包，数据目录始终指向 model/ 根。"""

from __future__ import annotations

from pathlib import Path

# model/ 目录（prep、runs、logs、submissions 所在处）
PROJECT_ROOT = Path(__file__).resolve().parent
ROOT = PROJECT_ROOT
PREP = ROOT / "prep"
RUNS = ROOT / "runs"
LOGS = ROOT / "logs"
SUBMISSIONS = ROOT / "submissions"
DATA = ROOT.parent / "data"
RESULTS = ROOT.parent / "results"
