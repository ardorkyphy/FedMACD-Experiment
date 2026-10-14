# -*- coding: utf-8 -*-
"""gcastle FCM：ANM（非线性加性噪声，从 src/util/algorithms/gcastle_impl/fcm.py 复刻）。"""
from typing import Any, List, Optional

from util.algorithms.base import CausalDiscoveryAlgorithm
from util.algorithms.common import as_numpy
from util.algorithms.edge import fcm_to_edges
from util.algorithms.registry import register


@register
class AnmAlgorithm(CausalDiscoveryAlgorithm):
    """ANM（gcastle ANMNonlinear，连续数据）。"""
    name = "anm"
    family = "fcm"
    data_type = "continuous"
    libs = ("castle",)

    description = "FCM 基：加性噪声模型，假设 Y=f(X)+ε 且 ε 独立于 X，用独立性检验辨识方向；数据前提：连续强非线性机制（饱和/阈值/正反馈）且噪声加性。"

    def discover(self, data, discrete: bool, columns: Optional[List[str]]) -> List[Any]:
        from castle.algorithms import ANMNonlinear
        model = ANMNonlinear(**dict(self.params))
        model.learn(data.astype(float))
        return fcm_to_edges(as_numpy(model.causal_matrix))