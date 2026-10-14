# -*- coding: utf-8 -*-
"""Protocol stage and stage-group abstractions.

Stage = 协议的一个可独立替换环节（run(ctx) 定义怎么跑）。
Phase = 一组有序 Stage（v4 两阶段：存在性阶段 / 方向阶段），进入时标记 ctx.current_phase
（debate/judge 按此分发协议与裁判）。
Pipeline 组合 Phase（模板方法：固定顺序，环节内部可并行）。
"""
from abc import ABC, abstractmethod
from typing import List

from protocol.context import ProtocolContext


class Stage(ABC):
    name: str = ""

    @abstractmethod
    def run(self, ctx: ProtocolContext) -> None:
        raise NotImplementedError

    def log(self, ctx: ProtocolContext):
        """阶段日志器（util.log 命名空间 protocol.<name>）。"""
        from util.log import get
        return get(f"protocol.{self.name}")


class Phase:
    """一组有序 Stage：v4 把流水线分为 Phase1（存在性→候选集）与 Phase2（方向→最终图）。"""

    def __init__(self, name: str, stages: List[Stage], kind: str = "") -> None:
        self.name = name
        self.stages = stages
        self.kind = kind                 # existence / direction（写入 ctx.current_phase）

    def run(self, ctx: ProtocolContext) -> None:
        """标记当前阶段 → 顺序执行本阶段全部 Stage。"""
        if self.kind:
            ctx.current_phase = self.kind
        for stage in self.stages:
            stage.run(ctx)
