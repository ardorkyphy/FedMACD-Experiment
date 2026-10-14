# -*- coding: utf-8 -*-
"""Collect local orientation evidence.

每个客户端对候选集所有边：统计方向证据（evidence.direction provider）+ 本地 LLM
方向性分析（答案+分数+理由）。前提：各客户端 LLM 不同，全员语义采集有信息价值。
"""
from protocol.base import Stage
from protocol.context import ProtocolContext
from protocol.pipeline import register_stage


@register_stage
class DirectionEvidenceStage(Stage):
    name = "direction_evidence"

    def run(self, ctx: ProtocolContext) -> None:
        tasks = [lambda c=c: self._compute_one(ctx, c) for c in ctx.clients]
        ctx.executor.run_all(tasks)

    def _compute_one(self, ctx: ProtocolContext, client) -> None:
        client.compute_direction_evidence()   # 统计方向证据（全部候选边）
        client.analyze(kind="direction")      # 本地 LLM 方向性分析
