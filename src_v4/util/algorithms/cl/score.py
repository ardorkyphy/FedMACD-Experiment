# -*- coding: utf-8 -*-
"""论文算法池中的 causal-learn 评分基方法：GES。"""
from typing import Any, List, Optional

from util.algorithms.base import CausalDiscoveryAlgorithm
from util.algorithms.common import call_kwargs
from util.algorithms.edge import graph_to_edges
from util.algorithms.registry import register


@register
class GesAlgorithm(CausalDiscoveryAlgorithm):
    """GES（离散 local_score_BDeu / 连续 local_score_BIC）。"""
    name = "ges"
    family = "score"
    data_type = "any"
    libs = ("causallearn",)
    requires_module = "causallearn.search.ScoreBased.GES"
    param_schema = {
        "score_func": {"type": "string", "enum": ["auto", "local_score_BDeu", "local_score_BIC"], "default": "auto"},
        "maxP": {"type": "integer", "default": None},
    }

    description = "评分基：对整图做 BIC/BDeu 评分并贪心搜索（加边-删边-转向），假设机制近似线性、误差同分布，连续/离散都可用；数据前提：样本较充足、机制近评分假设（线性高斯/BDeu）。"

    def discover(self, data, discrete: bool, columns: Optional[List[str]]) -> List[Any]:
        from causallearn.search.ScoreBased.GES import ges
        p = dict(self.params)
        score_func = p.pop("score_func", None) or ""
        if score_func in ("", "auto"):   # Select the score from observed data properties.
            score_func = "local_score_BDeu" if discrete else "local_score_BIC"
        maxP = p.pop("maxP", None)
        result = call_kwargs(ges, p, X=data, score_func=score_func, maxP=maxP)
        return graph_to_edges(result["G"].graph)
