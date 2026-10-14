# -*- coding: utf-8 -*-
"""约束基算法（causal-learn，从 src/util/algorithms/cl/constraint.py 复刻）：PC / FCI / CDNOD。"""
from typing import Any, List, Optional

import numpy as np

from util.algorithms.base import CausalDiscoveryAlgorithm
from util.algorithms.common import call_kwargs, resolve_cit
from util.algorithms.edge import graph_to_edges
from util.algorithms.registry import register


@register
class PcAlgorithm(CausalDiscoveryAlgorithm):
    """PC（causal-learn；indep_test 按数据特征 auto 选 gsq/fisherz）。"""
    name = "pc"
    family = "constraint"
    data_type = "any"
    libs = ("causallearn",)
    requires_module = "causallearn.search.ConstraintBased.PC"
    param_schema = {
        "indep_test": {"type": "string", "enum": ["auto", "gsq", "fisherz"], "default": "auto"},
        "stable": {"type": "boolean", "default": True},
    }

    description = "约束基：以条件独立性检验（离散 G²/卡方、连续 Fisher-Z）逐对筛边建骨架再定向，假设无隐藏混杂；数据前提：中小维度、机制稀疏、无大量隐藏共因，离散检验可靠性取决于每格计数。"

    def discover(self, data, discrete: bool, columns: Optional[List[str]]) -> List[Any]:
        from causallearn.search.ConstraintBased.PC import pc
        p = dict(self.params)
        cit = resolve_cit(p, discrete)
        cg = call_kwargs(pc, p, data=data, alpha=self.alpha, indep_test=cit, show_progress=False)
        return graph_to_edges(cg.G.graph)


@register
class FciAlgorithm(CausalDiscoveryAlgorithm):
    """FCI（causal-learn，含隐藏变量场景）。"""
    name = "fci"
    family = "constraint"
    data_type = "any"
    libs = ("causallearn",)
    requires_module = "causallearn.search.ConstraintBased.FCI"
    param_schema = {
        "indep_test": {"type": "string", "enum": ["auto", "gsq", "fisherz"], "default": "auto"},
    }

    description = "约束基：PC 的推广，用 m-separation 处理潜在混杂/隐藏变量，输出含部分确定关系的图；数据前提：存在隐藏共因、语义提示有未观测混杂；无隐藏变量时无额外可辨识信息。"

    def discover(self, data, discrete: bool, columns: Optional[List[str]]) -> List[Any]:
        from causallearn.search.ConstraintBased.FCI import fci
        p = dict(self.params)
        cit = resolve_cit(p, discrete)
        g, _ = call_kwargs(fci, p, dataset=data, independence_test_method=cit,
                           alpha=self.alpha, verbose=False)
        return graph_to_edges(g.graph)


@register
class CdnodAlgorithm(CausalDiscoveryAlgorithm):
    """CDNOD（causal-learn，带环境/背景变量）。"""
    name = "cdnod"
    family = "constraint"
    data_type = "any"
    libs = ("causallearn",)
    requires_module = "causallearn.search.ConstraintBased.CDNOD"
    param_schema = {
        "indep_test": {"type": "string", "enum": ["auto", "gsq", "fisherz"], "default": "auto"},
    }

    description = "约束基：利用数据分布随环境/时间的变化（非平稳、异质）识别方向；数据前提：跨环境、异质、分布偏移观测；单一同质分布下无环境变化信息。"

    def discover(self, data, discrete: bool, columns: Optional[List[str]]) -> List[Any]:
        from causallearn.search.ConstraintBased.CDNOD import cdnod
        p = dict(self.params)
        cit = resolve_cit(p, discrete)
        c_indx = p.pop("c_indx", None)
        if c_indx is None:
            c_indx = np.zeros(data.shape[0], dtype=int).reshape(-1, 1)
        cg = call_kwargs(cdnod, p, data=data, c_indx=c_indx, alpha=self.alpha,
                         indep_test=cit, verbose=False, show_progress=False)
        d = int(data.shape[1])
        return graph_to_edges(np.asarray(cg.G.graph)[:d, :d])  # 去背景列（环境变量）
