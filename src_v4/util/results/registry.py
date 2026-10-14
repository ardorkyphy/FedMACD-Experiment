# -*- coding: utf-8 -*-
"""指标注册表（从 src/util/results/registry.py 复刻）。"""
from typing import Dict, List, Optional, Type

from util.results.base import Metric


class MetricRegistry:
    """名字 → 指标类；compute_all 按过滤条件批量计算。"""

    def __init__(self) -> None:
        self._classes: Dict[str, Type[Metric]] = {}

    def register(self, cls: Type[Metric]) -> Type[Metric]:
        if cls.name in self._classes:
            print(f"[results] 重复注册指标覆盖: {cls.name}")
        self._classes[cls.name] = cls
        return cls

    def get(self, name: str) -> Optional[Type[Metric]]:
        return self._classes.get(name)

    def names(self) -> List[str]:
        return list(self._classes.keys())

    def compute_all(self, ctx, enabled: Optional[Dict[str, bool]] = None) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for name, cls in self._classes.items():
            if enabled is not None and not enabled.get(name, True):
                continue
            try:
                out[name] = float(cls().compute(ctx))
            except Exception as exc:
                print(f"[results] 指标 {name} 计算失败: {exc}")
        return out


metric_registry = MetricRegistry()


def register_metric(cls: Type[Metric]) -> Type[Metric]:
    metric_registry.register(cls)
    return cls