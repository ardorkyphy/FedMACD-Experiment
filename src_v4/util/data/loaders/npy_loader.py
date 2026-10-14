# -*- coding: utf-8 -*-
"""NPY 数据加载器（从 src/util/data/loaders/npy_loader.py 复刻，备用）。"""
from typing import Any, Dict, List, Set, Tuple

import numpy as np

from util.data.base import DataLoader
from util.data.common import total_n
from util.data.datasets import resolve_path


class NpyLoader(DataLoader):
    """NPY 格式（预先生成好的样本矩阵）。"""

    name = "npy"

    def __init__(self) -> None:
        self._columns: List[str] = []
        self._d: int = 0

    def load(self, data_cfg: Dict[str, Any]) -> np.ndarray:
        path = resolve_path(str(data_cfg.get("name", "")))["data"]
        arr = np.load(path)
        self._d = int(arr.shape[1])
        total = total_n(data_cfg)
        return arr[:total] if total < arr.shape[0] else arr

    def load_true_edges(self, data_cfg: Dict[str, Any], columns: List[str]) -> Set[Tuple[int, int]]:
        return set()

    def columns(self, data_cfg: Dict[str, Any]) -> List[str]:
        if not self._columns:
            if not self._d:
                path = resolve_path(str(data_cfg.get("name", "")))["data"]
                self._d = int(np.load(path, mmap_mode="r").shape[1])
            self._columns = [f"X{i}" for i in range(self._d)]
        return list(self._columns)