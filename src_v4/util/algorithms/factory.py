# -*- coding: utf-8 -*-
"""算法门面 AlgorithmRunner（从 src/util/algorithms/factory.py 复刻）。

Client 只依赖本门面：run(name) → 注册表实例化 → discover。
"""
from typing import Any, Dict, List, Optional

from util.algorithms.base import CausalDiscoveryAlgorithm
from util.algorithms.edge import Edge
from util.algorithms.registry import AlgorithmRegistry, registry


class AlgorithmRunner:
    """算法调用门面：Client 出图的唯一入口。"""

    def __init__(self, config: Dict[str, Any], reg: Optional[AlgorithmRegistry] = None) -> None:
        self.config = config or {}
        self.registry = reg or registry

    def run(self, name: str, data, discrete: bool, columns: Optional[List[str]],
            extra_params: Optional[Dict[str, Any]] = None,
            seed: Optional[int] = None) -> List[Edge]:
        alg: CausalDiscoveryAlgorithm = self.registry.create(
            name, self.config, extra_params=extra_params, seed=seed)
        return alg.discover(data, discrete=discrete, columns=columns)

    def available_names(self, discrete: Optional[bool] = None) -> List[str]:
        return self.registry.candidates(discrete)

    def pool_text(self, discrete: bool) -> str:
        return self.registry.pool_text(discrete)

