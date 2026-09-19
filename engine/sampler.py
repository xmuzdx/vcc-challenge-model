"""LineBalancedSampler：按扰动细胞数比例混合四个细胞系。"""

from __future__ import annotations

import numpy as np

# 与四个 destination 扰动细胞数对齐：rpe1 25.5 / hepg2 15.1 / jurkat 27.1 / k562ess 32.3
LINE_WEIGHTS = {"rpe1": 0.255, "hepg2": 0.151, "jurkat": 0.271, "k562ess": 0.323}


class LineBalancedSampler:
    """同一 batch 混合四系：先抽细胞系，系内均匀抽扰动基因。"""

    def __init__(self, items: list[tuple[str, str]], batch: int, rng: np.random.Generator,
                 weights: dict[str, float] | None = None):
        self.batch = batch
        self.rng = rng
        by: dict[str, list[tuple[str, str]]] = {}
        for p, ln in items:
            by.setdefault(ln, []).append((p, ln))
        self.lines = [ln for ln in by if by[ln]]
        self.by = by
        w = dict(LINE_WEIGHTS if weights is None else weights)
        raw = np.array([w.get(ln, 1.0 / len(self.lines)) for ln in self.lines], np.float64)
        self.p = raw / raw.sum()
        self.n_pairs = len(items)

    def epoch(self) -> list[list[tuple[str, str]]]:
        n_step = max(1, self.n_pairs // self.batch)
        batches = []
        for _ in range(n_step):
            pick = self.rng.choice(len(self.lines), size=self.batch, p=self.p)
            batch = []
            for i in pick:
                ln = self.lines[int(i)]
                pool = self.by[ln]
                batch.append(pool[int(self.rng.integers(len(pool)))])
            batches.append(batch)
        return batches
