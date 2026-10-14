# -*- coding: utf-8 -*-
"""数据处理层抽象（从 src/util/data/base.py 复刻）。

DataLoader：加载原始数据 + 真图 + 前向采样（bif 需要；csv/npy 默认原样）。
Splitter：把总样本切成客户端分片（同质随机 / 异质分层）。
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np


@dataclass
class LoadedData:
    """加载+分割完成的数据产物。"""
    data: np.ndarray
    parts: List[np.ndarray]
    columns: List[str]
    discrete: bool
    true_edges: Set[Tuple[int, int]]
    meta: Dict[str, Any] = field(default_factory=dict)


class DataLoader(ABC):
    """数据集格式加载器（BIF / CSV / NPY 各一个子类）。"""

    name: str = ""

    @abstractmethod
    def load(self, data_cfg: Dict[str, Any]) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def load_true_edges(self, data_cfg: Dict[str, Any], columns: List[str]) -> Set[Tuple[int, int]]:
        raise NotImplementedError

    def columns(self, data_cfg: Dict[str, Any]) -> List[str]:
        raise NotImplementedError("框架：默认列名")


class Splitter(ABC):
    """客户端分片策略（同质随机 / 异质分层）。"""

    name: str = ""

    @abstractmethod
    def split(self, data: np.ndarray, num_clients: int, seed: int,
              data_cfg: Optional[Dict[str, Any]] = None) -> List[np.ndarray]:
        raise NotImplementedError