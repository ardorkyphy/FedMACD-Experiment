# -*- coding: utf-8 -*-
"""pgmpy 评分基：HillClimbSearch（hc / hc_k2，从 src/util/algorithms/pgmpy_impl/score.py 复刻）。

注意：max_indegree 是 estimate() 的参数（不是构造函数），必须 pop 后传给 estimate。
"""
from typing import Any, List, Optional

from util.algorithms.base import CausalDiscoveryAlgorithm
from util.algorithms.common import pgmpy_edges, pgmpy_frame
from util.algorithms.registry import register


@register
class HillClimbAlgorithm(CausalDiscoveryAlgorithm):
    """HC（pgmpy HillClimbSearch；离散 BDeu/K2、连续 bic-g）。"""
    name = "hc"
    family = "score"
    data_type = "any"
    libs = ("pgmpy",)
    param_schema = {
        "scoring_method": {"type": "string", "enum": ["auto", "BDeu", "K2", "bic-g"], "default": "auto"},
        "max_indegree": {"type": "integer", "default": None},
    }

    description = "评分基：HillClimb 爬山局部搜索（离散 BDeu/K2、连续 bic-g），从空图贪心加/删/转向边；数据前提：中小维度、机制近评分假设（线性高斯/BDeu）。"

    def discover(self, data, discrete: bool, columns: Optional[List[str]]) -> List[Any]:
        from pgmpy.estimators import HillClimbSearch
        p = dict(self.params)
        scoring = p.pop("scoring_method", None) or ""
        if scoring in ("", "auto"):   # auto=离散 BDeu / 连续 bic-g（v4.25）
            scoring = "BDeu" if discrete else "bic-g"
        max_indegree = p.pop("max_indegree", None)
        df = pgmpy_frame(data, columns, discrete)
        dag = HillClimbSearch(df, **p).estimate(scoring_method=scoring, max_indegree=max_indegree)
        return pgmpy_edges(dag, columns)


@register
class HillClimbK2Algorithm(CausalDiscoveryAlgorithm):
    """HC-K2（pgmpy HillClimbSearch + K2 打分，仅离散）。"""
    name = "hc_k2"
    family = "score"
    data_type = "discrete"
    libs = ("pgmpy",)
    param_schema = {"max_indegree": {"type": "integer", "default": None}}

    description = "评分基：K2 评分（仅离散）爬山搜索，贪心加边无骨架剪枝；数据前提：离散。"

    def discover(self, data, discrete: bool, columns: Optional[List[str]]) -> List[Any]:
        from pgmpy.estimators import HillClimbSearch
        p = dict(self.params)
        max_indegree = p.pop("max_indegree", None)
        df = pgmpy_frame(data, columns, discrete)
        dag = HillClimbSearch(df, **p).estimate(scoring_method="K2", max_indegree=max_indegree)
        return pgmpy_edges(dag, columns)