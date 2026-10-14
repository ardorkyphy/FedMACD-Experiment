# -*- coding: utf-8 -*-
"""pgmpy 约束基：PC（从 src/util/algorithms/pgmpy_impl/constraint.py 复刻）。"""
from typing import Any, List, Optional

from util.algorithms.base import CausalDiscoveryAlgorithm
from util.algorithms.common import pgmpy_edges, pgmpy_frame
from util.algorithms.registry import register


@register
class PgmpyPcAlgorithm(CausalDiscoveryAlgorithm):
    """pgmpy 版 PC（与 causal-learn 版并存，入口不同）。"""
    name = "pgmpy_pc"
    family = "constraint"
    data_type = "any"
    libs = ("pgmpy",)
    param_schema = {
        "variant": {"type": "string", "enum": ["orig", "stable"], "default": "orig"},
    }

    description = "约束基：pgmpy 库实现的 PC（逻辑同 causal-learn 版、接口不同）；数据前提：中小维度离散/连续、机制稀疏。"

    def discover(self, data, discrete: bool, columns: Optional[List[str]]) -> List[Any]:
        from pgmpy.estimators import PC
        p = dict(self.params)
        sig = p.pop("significance_level", self.alpha)
        variant = p.pop("variant", "orig")
        df = pgmpy_frame(data, columns, discrete)
        # ci_test 按数据特征显式指定（v4.25 修：pgmpy 对离散 category code 的默认/pearson
        # 检验口径不可靠，离散必须 chi_square，连续才用 pearson）
        ci_test = "chi_square" if discrete else "pearsonr"   # pgmpy 连续检验名为 pearsonr
        dag = PC(df, **p).estimate(variant=variant, ci_test=ci_test, significance_level=sig)
        return pgmpy_edges(dag, columns)
