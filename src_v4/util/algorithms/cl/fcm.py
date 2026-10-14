# -*- coding: utf-8 -*-
"""FCM 基算法（causal-learn，从 src/util/algorithms/cl/fcm.py 复刻）：DirectLiNGAM / ICALiNGAM / RCD（连续数据）。

方向约定（实测确认）：causal-learn/lingam 库的 adjacency_matrix_[i][j] 表示 j→i
（行=效应、列=因），与 fcm_to_edges 的 i→j 约定相反，故统一 .T 转置后再适配。
"""
from typing import Any, List, Optional

import numpy as np

from util.algorithms.base import CausalDiscoveryAlgorithm
from util.algorithms.edge import fcm_to_edges
from util.algorithms.registry import register


def _lingam_edges(model) -> List[Any]:
    """lingam 系 adjacency（j→i 约定）→ Edge 列表（转置为 i→j 后走 fcm_to_edges）。"""
    return fcm_to_edges(np.asarray(model.adjacency_matrix_, dtype=float).T)


@register
class DirectLingamAlgorithm(CausalDiscoveryAlgorithm):
    """DirectLiNGAM（逐对独立性）。"""
    name = "direct_lingam"
    family = "fcm"
    data_type = "continuous"
    libs = ("causallearn",)
    requires_module = "causallearn.search.FCMBased.lingam.direct_lingam"

    description = "FCM 基：假设机制线性、噪声非高斯且相互独立，用逐对独立性排序恢复方向；数据前提：连续非高斯线性机制（偏态/重尾/均匀噪声）；高斯噪声下方向不可辨识。"

    def discover(self, data, discrete: bool, columns: Optional[List[str]]) -> List[Any]:
        from causallearn.search.FCMBased.lingam.direct_lingam import DirectLiNGAM
        p = dict(self.params)
        p.pop("random_state", None)   # 种子统一走 self.seed
        model = DirectLiNGAM(random_state=self.seed, **p)
        model.fit(data.astype(float))
        return _lingam_edges(model)


@register
class IcaLingamAlgorithm(CausalDiscoveryAlgorithm):
    """ICALiNGAM（ICA 分解）。"""
    name = "ica_lingam"
    family = "fcm"
    data_type = "continuous"
    libs = ("causallearn",)
    requires_module = "causallearn.search.FCMBased.lingam.ica_lingam"

    description = "FCM 基：ICA 分解实现 LiNGAM 假设（线性非高斯）；数据前提：连续非高斯线性机制、样本量中等以上；强非线性机制下假设不成立。"

    def discover(self, data, discrete: bool, columns: Optional[List[str]]) -> List[Any]:
        from causallearn.search.FCMBased.lingam.ica_lingam import ICALiNGAM
        p = dict(self.params)
        p.pop("random_state", None)
        model = ICALiNGAM(random_state=self.seed, **p)
        model.fit(data.astype(float))
        return _lingam_edges(model)


@register
class RcdAlgorithm(CausalDiscoveryAlgorithm):
    """RCD（回归依赖，连续数据）。"""
    name = "rcd"
    family = "fcm"
    data_type = "continuous"
    libs = ("causallearn",)
    requires_module = "causallearn.search.FCMBased.lingam.rcd"

    description = "FCM 基：RCD 通过重复选择回归方向，处理潜在混淆下的线性非高斯机制；数据前提：连续、线性非高斯、可能存在潜在混淆。"

    def discover(self, data, discrete: bool, columns: Optional[List[str]]) -> List[Any]:
        from causallearn.search.FCMBased.lingam.rcd import RCD
        model = RCD(**dict(self.params))
        model.fit(data.astype(float))
        return _lingam_edges(model)