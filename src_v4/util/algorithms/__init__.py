# -*- coding: utf-8 -*-
"""FedMACD 因果发现算法层。

导入本包即注册论文 Table 1 所列的 17 项算法。候选池再按数据类型和可用依赖过滤，
因此呈现给 LLM 的每个候选算法都可以在当前环境中实际运行。
"""
from util.algorithms import cl  # noqa: F401
from util.algorithms import fedecd  # noqa: F401
from util.algorithms import gcastle_impl  # noqa: F401
from util.algorithms import lingam_impl  # noqa: F401
from util.algorithms import pgmpy_impl  # noqa: F401
from util.algorithms.base import CausalDiscoveryAlgorithm  # noqa: F401
from util.algorithms.edge import Edge  # noqa: F401
from util.algorithms.factory import AlgorithmRunner  # noqa: F401
from util.algorithms.registry import AlgorithmRegistry, registry, register  # noqa: F401
from util.algorithms.selector import AlgorithmChoice, AlgorithmSelector  # noqa: F401

__all__ = ["Edge", "CausalDiscoveryAlgorithm", "AlgorithmRunner", "AlgorithmRegistry",
           "registry", "register", "AlgorithmChoice", "AlgorithmSelector"]
