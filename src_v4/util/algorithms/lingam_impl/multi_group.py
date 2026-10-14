# -*- coding: utf-8 -*-
"""异质/多组场景代表：MultiGroupDirectLiNGAM（lingam 库，从 src/util/algorithms/lingam_impl/multi_group.py 复刻）。

文献：Shimizu 等 LiNGAM 系列的多组推广（MultiGroup 变体，lingam 库实现）。
场景定位：客户端分布异质时，本地数据近似"多组同因果机制、不同分布"，
该算法按"样本分组"（list of datasets）直接建模组间异质性。

本地用法（单客户端一份数据）：库要求至少两组，故把本地数据随机切两半作为两组
（同机制、同分布；只取 group-0 的邻接矩阵）。方向约定：adjacency_matrices_[g][i][j] 表示
j→i（行=效应、列=因），转置后走 fcm_to_edges。
"""
from typing import Any, List, Optional

import numpy as np

from util.algorithms.base import CausalDiscoveryAlgorithm
from util.algorithms.edge import fcm_to_edges
from util.algorithms.registry import register


@register
class MultiGroupLingamAlgorithm(CausalDiscoveryAlgorithm):
    """MultiGroupDirectLiNGAM：多组样本的 LiNGAM（异质场景代表，连续数据）。"""
    name = "multigroup_lingam"
    family = "fcm"
    data_type = "continuous"
    cost = "fast"
    libs = ("lingam",)

    description = "FCM 基：MultiGroupDirectLiNGAM，按样本分组建模“同机制不同分布”的异质数据；数据前提：多环境/多组样本、组间分布不同；单一同质分布下无分组信息。"

    def discover(self, data, discrete: bool, columns: Optional[List[str]]) -> List[Any]:
        from lingam import MultiGroupDirectLiNGAM
        p = dict(self.params)
        p.pop("random_state", None)
        arr = np.asarray(data, dtype=float)
        n = int(arr.shape[0])
        if n < 2:
            raise ValueError(f"multigroup_lingam 需要至少 2 组样本，当前 n={n}")
        rng = np.random.default_rng(self.seed)
        idx = rng.permutation(n)
        groups = [arr[idx[: n // 2]], arr[idx[n // 2:]]]   # 随机切两半作两组
        model = MultiGroupDirectLiNGAM(random_state=self.seed, **p)
        model.fit(groups)
        return fcm_to_edges(np.asarray(model.adjacency_matrices_[0], dtype=float).T)
