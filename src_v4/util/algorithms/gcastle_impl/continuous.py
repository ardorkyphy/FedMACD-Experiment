# -*- coding: utf-8 -*-
"""gcastle 连续优化算法：NOTEARS / GOLEM（连续数据，从 src/util/algorithms/gcastle_impl/continuous.py 复刻）。"""
from typing import Any, List, Optional

from util.algorithms.base import CausalDiscoveryAlgorithm
from util.algorithms.common import as_numpy
from util.algorithms.edge import fcm_to_edges
from util.algorithms.registry import register


@register
class NotearsAlgorithm(CausalDiscoveryAlgorithm):
    """NOTEARS（gcastle，DAG 连续优化）。"""
    name = "notears"
    family = "continuous_optim"
    data_type = "continuous"
    libs = ("castle",)

    description = "连续优化：把 DAG 约束转成光滑惩罚做连续优化，假设线性 SEM 且误差近似高斯；数据前提：连续、中等规模、近似线性高斯机制；非高斯噪声下方向信息弱、强非线性下模型误设。"

    def discover(self, data, discrete: bool, columns: Optional[List[str]]) -> List[Any]:
        from castle.algorithms import Notears
        model = Notears(**dict(self.params))
        model.learn(data.astype(float))
        return fcm_to_edges(as_numpy(model.causal_matrix))


@register
class GolemAlgorithm(CausalDiscoveryAlgorithm):
    """GOLEM（gcastle，DAG 连续优化，稀疏感知）。"""
    name = "golem"
    family = "continuous_optim"
    data_type = "continuous"
    libs = ("castle",)

    description = "连续优化：NOTEARS 的生成式变体，用连续约束+稀疏惩罚学习线性 DAG；数据前提：连续、样本量较大、机制近似线性；强非线性下模型误设。"

    def discover(self, data, discrete: bool, columns: Optional[List[str]]) -> List[Any]:
        from castle.algorithms import GOLEM
        p = dict(self.params)
        p.setdefault("device_type", "cpu")   # 实测 GOLEM GPU 卡死，默认强制 CPU；显式参数可覆盖
        model = GOLEM(seed=self.seed, **p)
        model.learn(data.astype(float))
        return fcm_to_edges(as_numpy(model.causal_matrix))