# -*- coding: utf-8 -*-
"""同质分片（随机等分，从 src/util/data/splitters/random_split.py 复刻）。"""
from typing import Any, Dict, List, Optional

import numpy as np

from util.data.base import Splitter


class RandomSplitter(Splitter):
    """同质场景：样本随机均分到各客户端。"""

    name = "random"

    def split(self, data: np.ndarray, num_clients: int, seed: int,
              data_cfg: Optional[Dict[str, Any]] = None) -> List[np.ndarray]:
        n = data.shape[0]
        rng = np.random.default_rng(seed)
        idx = rng.permutation(n)
        return [data[idx[i * n // num_clients:(i + 1) * n // num_clients]]
                for i in range(num_clients)]