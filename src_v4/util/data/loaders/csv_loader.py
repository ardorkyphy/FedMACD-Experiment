# -*- coding: utf-8 -*-
"""CSV 数据加载器（从 src/util/data/loaders/csv_loader.py 复刻，通用备用）。"""
from typing import Any, Dict, List, Set, Tuple

import numpy as np
import pandas as pd

from util.data.base import DataLoader
from util.data.common import factorize_discrete, total_n
from util.data.datasets import resolve_path


class CsvLoader(DataLoader):
    """CSV 格式（无真图/无变量语义时使用）。"""

    name = "csv"

    def __init__(self) -> None:
        self._columns: List[str] = []
        self._data: np.ndarray = np.empty((0, 0))

    def load(self, data_cfg: Dict[str, Any]) -> np.ndarray:
        path = resolve_path(str(data_cfg.get("name", "")))["data"]
        raw = pd.read_csv(path, header=None)
        first = raw.iloc[0]
        if pd.to_numeric(first, errors="coerce").notna().all():
            df = raw
            self._columns = [f"X{i}" for i in range(raw.shape[1])]
        else:
            self._columns = [str(c) for c in raw.iloc[0]]
            df = raw.iloc[1:].reset_index(drop=True)
            df.columns = self._columns
        if bool(data_cfg.get("discrete", True)):
            df = factorize_discrete(df)
        else:
            df = df.apply(pd.to_numeric, errors="coerce").dropna(axis=0)  # Exclude incomplete rows from downstream statistics.
        arr = df.to_numpy()
        total = total_n(data_cfg)
        self._data = arr[:total] if total < arr.shape[0] else arr
        return self._data

    def load_true_edges(self, data_cfg: Dict[str, Any], columns: List[str]) -> Set[Tuple[int, int]]:
        return set()

    def columns(self, data_cfg: Dict[str, Any]) -> List[str]:
        if not self._columns:
            self.load(data_cfg)
        return list(self._columns)
