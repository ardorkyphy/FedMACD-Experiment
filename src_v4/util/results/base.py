# -*- coding: utf-8 -*-
"""指标抽象（从 src/util/results/base.py 复刻）。"""
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Set, Tuple


@dataclass
class ResultContext:
    """评估输入：估计边/真边/节点数 + 可选效率输入（计时/LLM 调用）。"""
    est_edges: Set[Tuple[int, int]]
    true_edges: Set[Tuple[int, int]]
    n_nodes: int
    timings: Optional[Dict[str, Dict[str, float]]] = None
    llm_calls: Optional[Dict[str, float]] = None
    extra: Dict[str, Any] = field(default_factory=dict)


class Metric:
    """单一指标计算器（可扩展：继承并注册）。"""

    name: str = ""

    def compute(self, ctx: ResultContext) -> float:
        raise NotImplementedError