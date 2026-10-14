# -*- coding: utf-8 -*-
"""Evidence-layer abstractions.

- Statistic：单个统计量（MI/G²/条件独立/熵/残差/v 结构…），@register_stat 进 stat_registry。
- EvidenceProvider：组合统计量产出某类证据块（存在性/方向性），@register_provider。
- 聚合器（Aggregator）在 evidence/aggregate.py。
"""
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Type


class EvidenceType(Enum):
    EXISTENCE = "existence"
    DIRECTION = "direction"


class Registry:
    """通用类注册表（名字 → 类），@register 装饰器自动收集。"""

    def __init__(self) -> None:
        self._classes: Dict[str, Type] = {}

    def register(self, cls: Type) -> Type:
        name = getattr(cls, "name", "") or cls.__name__
        if name in self._classes:
            print(f"[evidence] 重复注册覆盖: {name}")
        self._classes[name] = cls
        return cls

    def get(self, name: str) -> Optional[Type]:
        return self._classes.get(name)

    def names(self) -> List[str]:
        return list(self._classes.keys())

    def instantiate_all(self, **kwargs) -> List[Any]:
        return [cls(**kwargs) for cls in self._classes.values()]


stat_registry = Registry()
provider_registry = Registry()


def side_of(alg: str, stat_dir: float, llm_ans: str = "") -> str:
    """方向侧分（v4.25 统一口径，辩论包 / DirectionSplit / 中立率共用）：

    三信号 = 算法方向（->/<-）+ 统计方向（stat_dir≠0.5）+ LLM 方向答案（u->v/v->u）；
    行级按多数投票，平局按 算法>统计>LLM 优先序；全中性 → ""（不进任何包、不计方向票）。
    """
    signals = []
    if alg == "->":
        signals.append("uv")
    elif alg == "<-":
        signals.append("vu")
    if stat_dir > 0.5:
        signals.append("uv")
    elif stat_dir < 0.5:
        signals.append("vu")
    if llm_ans == "u->v":
        signals.append("uv")
    elif llm_ans == "v->u":
        signals.append("vu")
    n_uv, n_vu = signals.count("uv"), signals.count("vu")
    if n_uv != n_vu:
        return "uv" if n_uv > n_vu else "vu"
    return signals[0] if signals else ""


def register_stat(cls: Type) -> Type:
    stat_registry.register(cls)
    return cls


def register_provider(cls: Type) -> Type:
    provider_registry.register(cls)
    return cls


class Statistic(ABC):
    """单个统计量：客户端在本地数据上对一条候选边算一个（或一组）数值。"""

    name: str = ""
    kind: str = "existence"        # existence / direction（决定进哪个证据矩阵）

    @abstractmethod
    def compute(self, data, columns: List[str], edge: Tuple[str, str],
                discrete: bool, options: Dict[str, Any]) -> Dict[str, Any]:
        """返回 {字段名: 值}，并入该边的证据块。"""
        raise NotImplementedError


class EvidenceProvider(ABC):
    """某类证据的提供者：组合注册的统计量，产出统一证据块（dataclass 由 config.schema 定义）。"""

    kind: EvidenceType = EvidenceType.EXISTENCE
    stats: List[str] = []          # 本提供者使用的统计量名（stat_registry）

    @abstractmethod
    def compute(self, data, columns: List[str], edges: List[Tuple[str, str]],
                discrete: bool, options: Dict[str, Any]) -> Dict[Tuple[str, str], Dict[str, Any]]:
        """返回 {edge: 证据字段 dict}（调用方转换为 schema dataclass）。"""
        raise NotImplementedError
