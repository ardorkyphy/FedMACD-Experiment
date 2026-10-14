# -*- coding: utf-8 -*-
"""结果层（从 src/util/results 复刻）：指标注册表 + ResultService。

import 本包即触发指标 @register_metric 自注册（util/results/metrics）。
"""
from util.results import metrics  # noqa: F401  触发指标注册
from util.results.base import Metric, ResultContext  # noqa: F401
from util.results.registry import MetricRegistry, metric_registry, register_metric  # noqa: F401
from util.results.service import ResultService  # noqa: F401

__all__ = ["Metric", "ResultContext", "MetricRegistry", "metric_registry",
           "register_metric", "ResultService"]