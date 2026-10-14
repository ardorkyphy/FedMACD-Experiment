# -*- coding: utf-8 -*-
"""高维场景代表：HighDimDirectLiNGAM（lingam 库，从 src/util/algorithms/lingam_impl/highdim.py 复刻）。

文献：Shimizu 等 LiNGAM 高维变体（lingam 库实现）。
场景定位：变量数较多/样本有限时的非高斯因果发现代表。
方向约定：adjacency_matrix_[i][j] 表示 j→i（行=效应、列=因），转置后走 fcm_to_edges。
"""
from typing import Any, List, Optional

import numpy as np

from util.algorithms.base import CausalDiscoveryAlgorithm
from util.algorithms.edge import fcm_to_edges
from util.algorithms.registry import register


@register
class HighDimLingamAlgorithm(CausalDiscoveryAlgorithm):
    """HighDimDirectLiNGAM：高维非高斯因果发现（连续数据）。"""
    name = "highdim_lingam"
    family = "fcm"
    data_type = "continuous"
    cost = "fast"
    libs = ("lingam",)

    description = "FCM 基：HighDimDirectLiNGAM（DirectLiNGAM 的高维变体，基于 Lasso 剪枝）；数据前提：变量数较多（d 大）、机制线性非高斯、样本相对不足。"

    def discover(self, data, discrete: bool, columns: Optional[List[str]]) -> List[Any]:
        from lingam import HighDimDirectLiNGAM
        p = dict(self.params)
        p.pop("random_state", None)
        model = HighDimDirectLiNGAM(random_state=self.seed, **p)
        model.fit(np.asarray(data, dtype=float))
        # HighDimDirectLiNGAM.adjacency_matrix_ uses the same j->i convention
        # as DirectLiNGAM (_estimate_adjacency_matrix assigns
        # B[target, predictors] = 系数赋值，行=效应、列=因；同数据两模型矩阵逐位一致），
        # Transpose to i->j before converting the matrix to edges.
        return fcm_to_edges(np.asarray(model.adjacency_matrix_, dtype=float).T)
