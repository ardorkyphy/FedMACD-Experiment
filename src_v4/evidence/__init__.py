# -*- coding: utf-8 -*-
"""Statistical, existence, orientation, and aggregate evidence providers.

import 本包即触发统计量/提供者/聚合器的 @register_* 自注册。
"""
from evidence import base, statistics, existence, direction, aggregate  # noqa: F401
from evidence.base import (  # noqa: F401
    EvidenceType, Registry, stat_registry, provider_registry,
    register_stat, register_provider, Statistic, EvidenceProvider,
)

__all__ = ["EvidenceType", "Registry", "stat_registry", "provider_registry",
           "register_stat", "register_provider", "Statistic", "EvidenceProvider"]
