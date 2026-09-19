"""响应程序基：GWPS 响应矩阵的低秩分解、缓存与加载。"""

from __future__ import annotations

import numpy as np
import torch

from paths import PREP

PROGRAM_K = 64
PROGRAM_CACHE = PREP / "programs.npz"


def build_programs(lfc: np.ndarray, k: int = PROGRAM_K) -> tuple[np.ndarray, np.ndarray]:
    """对 (n_pert, n_w) 源响应做截断 SVD，返回 (K, n_w) 程序基与奇异值。

    The gene axis is what transfers: a program is a direction in expression
    space that many perturbations share.  Centring first keeps the leading
    component from collapsing onto the corpus-wide mean response, which the
    prior channel already carries.
    """
    from sklearn.utils.extmath import randomized_svd

    x = np.nan_to_num(lfc, nan=0.0, posinf=0.0, neginf=0.0)
    x = x - x.mean(0, keepdims=True)
    _, s, vt = randomized_svd(x, n_components=k, random_state=0)
    return vt.astype(np.float32), s.astype(np.float32)


def load_programs(corpus, k: int = PROGRAM_K, rebuild: bool = False) -> torch.Tensor:
    """取工作空间 W 上的程序基，缺失时从 corpus 的源响应重建并缓存。"""
    if PROGRAM_CACHE.exists() and not rebuild:
        d = np.load(PROGRAM_CACHE)
        if int(d["n_w"]) == corpus.n_w and d["basis"].shape[0] >= k:
            return torch.from_numpy(d["basis"][:k].copy())
    basis, sv = build_programs(corpus.lines["gwps"]["lfc"], k)
    PROGRAM_CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(PROGRAM_CACHE, basis=basis, sv=sv,
                        n_w=np.int64(corpus.n_w))
    print(f"[programs] built K={k} basis over {corpus.n_w} genes", flush=True)
    return torch.from_numpy(basis)
