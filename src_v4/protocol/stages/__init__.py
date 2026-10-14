# -*- coding: utf-8 -*-
"""v4 阶段实现（import 触发 @register_stage 自注册，供 StageFactory 使用）。"""
from protocol.stages import (          # noqa: F401（导入即注册）
    align, local_discovery, existence_evidence, upload,
    aggregate, absence_supplement, debate, judge,
    share_candidate, direction_evidence, assemble,
)
