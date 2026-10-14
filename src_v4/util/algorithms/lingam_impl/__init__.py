# -*- coding: utf-8 -*-
"""独立 lingam 库实现族（从 src/util/algorithms/lingam_impl 复刻）。

与 cl/fcm.py 的 causal-learn 内置 lingam 区别：本族注册独立 lingam 库独有的方法——
MultiGroupDirectLiNGAM（样本分组异质/多环境场景代表）、HighDimDirectLiNGAM（高维场景代表）。
"""
from util.algorithms.lingam_impl import highdim, multi_group  # noqa: F401