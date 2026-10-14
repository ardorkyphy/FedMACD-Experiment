# -*- coding: utf-8 -*-
"""BIF 数据加载器（从 src/util/data/loaders/bif_loader.py 复刻）。

pgmpy 解析 + 前向采样（sampling_seed/sample_size）；BIF 自带真图与变量名。
"""
from typing import Any, Dict, List, Set, Tuple

import numpy as np
import pandas as pd

from util.data.base import DataLoader
from util.data.common import factorize_discrete, sampling_seed, total_n
from util.data.datasets import resolve_path


class BifLoader(DataLoader):
    """BIF 格式（含真图与变量名），当前实验主用格式。"""

    name = "bif"

    def __init__(self) -> None:
        self._columns: List[str] = []
        self._bif_edges: Set[Tuple[int, int]] = set()
        self._model: Any = None

    def load(self, data_cfg: Dict[str, Any]) -> np.ndarray:
        from pgmpy.readwrite import BIFReader
        from pgmpy.sampling import BayesianModelSampling
        path = resolve_path(str(data_cfg.get("name", "")))["data"]
        total = total_n(data_cfg)
        if total > 10_000_000:                    # "全量"哨兵：BIF 需要生成样本，默认 1000
            total = 1000
        model = BIFReader(path).get_model()
        df = BayesianModelSampling(model).forward_sample(size=total, seed=sampling_seed(data_cfg))
        self._model = model
        self._columns = list(df.columns)
        name2idx = {name: i for i, name in enumerate(self._columns)}
        self._bif_edges = {(name2idx[s], name2idx[t]) for s, t in model.edges()}
        if bool(data_cfg.get("discrete", True)):
            from util.data.common import encode_by_states
            df = encode_by_states(df, model)   # Stable state encoding must match heterogeneous splits.
        return df.to_numpy()

    def load_true_edges(self, data_cfg: Dict[str, Any], columns: List[str]) -> Set[Tuple[int, int]]:
        return set()                               # BIF 自带边走 bif_edges（DataService 兜底）

    def columns(self, data_cfg: Dict[str, Any]) -> List[str]:
        if not self._columns:                      # 未先 load（防御）：只读变量名不采样
            from pgmpy.readwrite import BIFReader
            path = resolve_path(str(data_cfg.get("name", "")))["data"]
            self._columns = list(BIFReader(path).get_model().nodes())
        return list(self._columns)

    @property
    def bif_edges(self) -> Set[Tuple[int, int]]:
        return set(self._bif_edges)
