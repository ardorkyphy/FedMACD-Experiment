# -*- coding: utf-8 -*-
"""测试共享工具（不被 unittest discover 收集，仅被 test_*.py 引用）。"""
import numpy as np


def gen_discrete_chain(n: int = 300, seed: int = 0) -> np.ndarray:
    """离散链式结构：X3→X1、X2 独立（真实边 (0,2) 与 (1,2)，方向 0→2、1→2）。"""
    rng = np.random.default_rng(seed)
    x3 = rng.integers(0, 3, size=n)
    x1 = (x3 + rng.integers(0, 3, size=n)) % 3
    x2 = rng.integers(0, 3, size=n)
    return np.stack([x1, x2, x3], axis=1)


def gen_cont_chain(n: int = 300, seed: int = 0) -> np.ndarray:
    """连续链式结构：X0→X1→X2（非高斯 t 分布噪声）。"""
    rng = np.random.default_rng(seed)
    e = rng.standard_t(5, size=(n, 3))
    x0 = e[:, 0]
    x1 = 0.8 * x0 + e[:, 1]
    x2 = 0.7 * x1 + e[:, 2]
    return np.stack([x0, x1, x2], axis=1)


def base_config(extra_params: dict = None, seed: int = 0, alpha: float = 0.05) -> dict:
    """最小算法配置（ALGORITHM_PARAMS/ALPHA/SEED）。"""
    return {"ALGORITHM_PARAMS": extra_params or {}, "ALPHA": alpha, "SEED": seed}
