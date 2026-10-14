# -*- coding: utf-8 -*-
"""论文算法池中的 causal-learn 排列基方法：GRaSP。"""
from typing import Any, List, Optional

from util.algorithms.base import CausalDiscoveryAlgorithm
from util.algorithms.edge import graph_to_edges
from util.algorithms.registry import register


def _result_graph(result):
    """gcastle/causal-learn 返回 dict 或 CausalGraph，统一取 G。"""
    return result["G"] if isinstance(result, dict) else result
@register
class GraspAlgorithm(CausalDiscoveryAlgorithm):
    """GRaSP（排列基，连续数据）。"""
    name = "grasp"
    family = "permutation"
    data_type = "continuous"
    libs = ("causallearn",)
    requires_module = "causallearn.search.PermutationBased.GRaSP"

    description = "排列基：GRaSP 在排序空间贪心搜索；数据前提：连续、中等维度稀疏结构。"

    def discover(self, data, discrete: bool, columns: Optional[List[str]]) -> List[Any]:
        from causallearn.search.PermutationBased.GRaSP import grasp
        p = dict(self.params)
        score_func = p.pop("score_func", None) or "local_score_BIC"   # 连续默认 BIC
        # The installed causallearn version does not implement its documented
        # default covariance score. GRaSP accepts no verbose argument here, so
        # pass an explicit BIC score and leave maxP/parameters at defaults.
        result = grasp(data.astype(float), score_func=score_func,
                       depth=p.pop("depth", 3))
        return graph_to_edges(_result_graph(result).graph)
