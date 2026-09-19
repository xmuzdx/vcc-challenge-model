"""基因先验：STRING/GO 面板特征与扰动注释，以及未映射基因清单。"""

from __future__ import annotations

import numpy as np
import torch

from paths import DATA, RESULTS

# GWPS 扰动里找不到蛋白编码 ESM 的 23 个符号。readthrough / lncRNA /
# 非 HGNC 条目喂全零向量只会加噪声，训练时从 pair 列表剔除。
UNMAPPED_GWPS = frozenset({
    "AC015871.1", "AC118549.1", "AHSA2", "ALG1L", "ARPC4-TTLL3",
    "C14orf178", "C19orf48", "C1orf61", "C22orf46", "CCDC169-SOHLH2",
    "CENPBD1", "FAM86C1", "HHLA3", "NEDD8-MDP1", "NME1-NME2",
    "RBAK-RBAKDN", "RBM14-RBM4", "RPL17-C18orf32", "RPS10-NUDT3",
    "SNHG32", "ST20-MTHFS", "TMEM99", "ZBED6CL",
})

GENE_FEAT_PATH = RESULTS / "vcc2026_submit" / "cache" / "gene_features.npz"
PERT_ANN_PATH = RESULTS / "pert_annotation" / "pert_classification.tsv"

PERT_ANN_DIM = 16
_STRENGTH = {"弱": 0.0, "中": 0.5, "强": 1.0}
_FLAG_COLS = (
    "GO_BP", "GO_MF", "GO_CC", "Reactome", "CORUM", "Complex_Portal",
    "OmniPath", "TRRUST", "DoRothEA", "CollecTRI", "Pfam", "InterPro",
    "UniProt_功能",
)


def load_panel_features() -> np.ndarray:
    """18533 × F 数值先验（STRING 2-hop + GO SVD + 功能类）。"""
    d = np.load(GENE_FEAT_PATH, allow_pickle=True)
    x = d["X"].astype(np.float32)
    x = (x - x.mean(0)) / (x.std(0) + 1e-6)
    return x


def load_pert_annotation() -> dict[str, np.ndarray]:
    """扰动基因 → 紧凑注释向量（强度 / 位移 / 通路开关）。"""
    import pandas as pd

    if not PERT_ANN_PATH.exists():
        return {}
    df = pd.read_csv(PERT_ANN_PATH, sep="\t")
    out: dict[str, np.ndarray] = {}
    for gene, g in df.groupby(df.iloc[:, 0].astype(str)):
        strength = max(_STRENGTH.get(str(s), 0.0) for s in g["扰动强度"].astype(str))
        n_move = float(np.log1p(pd.to_numeric(g["n_move_0.10"], errors="coerce").fillna(0).max()))
        flags = []
        for col in _FLAG_COLS:
            if col not in g.columns:
                flags.append(0.0)
                continue
            flags.append(1.0 if g[col].astype(str).replace("nan", "").str.len().max() > 0 else 0.0)
        vec = np.array([strength, n_move, *flags], dtype=np.float32)
        if vec.size < PERT_ANN_DIM:
            vec = np.pad(vec, (0, PERT_ANN_DIM - vec.size))
        out[str(gene)] = vec[:PERT_ANN_DIM]
    return out


def pert_ann_tensor(table: dict[str, np.ndarray], name: str) -> np.ndarray:
    v = table.get(name)
    return v if v is not None else np.zeros(PERT_ANN_DIM, np.float32)


def panel_features_on(W: np.ndarray) -> torch.Tensor:
    feat = load_panel_features()
    return torch.from_numpy(feat[np.asarray(W, np.int64)])
