# -*- coding: utf-8 -*-
"""因果发现算法策略基类（从 src/util/algorithms/base.py 复刻）。

策略模式 + 模板方法：每个算法一个子类，只实现 discover（核心调用）；
加算法 = 新子类 + @register，client/协议零改动（OCP）。
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple


class CausalDiscoveryAlgorithm(ABC):
    """算法统一接口：元数据（类属性）+ discover（核心）。"""

    name: str = ""
    family: str = ""
    data_type: str = "any"       # discrete / continuous / any
    cost: str = "fast"           # fast / medium / slow
    libs: Tuple[str, ...] = ()
    requires_module: str = ""    # discover 内部 import 的具体子模块（如 causallearn.search.ScoreBased.DGES），available 时一并探测
    param_schema: Dict[str, Any] = {}
    description: str = ""

    def __init__(self, params: Optional[Dict[str, Any]] = None,
                 alpha: float = 0.05, seed: int = 0) -> None:
        self.params: Dict[str, Any] = dict(params or {})
        self.alpha: float = alpha
        self.seed: int = seed

    @abstractmethod
    def discover(self, data, discrete: bool, columns: Optional[List[str]]) -> List[Any]:
        """跑算法并返回统一 Edge 列表（数据适配/输出适配在子类内）。"""
        raise NotImplementedError

    def available(self) -> bool:
        import importlib
        # Probe the concrete dependency used by discover, not only its top-level package.
        if self.requires_module:
            try:
                importlib.import_module(self.requires_module)
            except Exception:
                return False
        for lib in self.libs:
            try:
                importlib.import_module(lib)
            except Exception:
                return False
        return True
