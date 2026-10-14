# -*- coding: utf-8 -*-
"""Local edge-existence evidence computation.

客户端对本地骨架上的边：算存在性统计量（evidence 层提供者）+ 本地 LLM 存在合理性分析。
候选集里"自己没出的边"由 absence_supplement 阶段补采（方案 B）。
"""
from protocol.base import Stage
from protocol.context import ProtocolContext
from protocol.pipeline import register_stage


@register_stage
class ExistenceEvidenceStage(Stage):
    name = "existence_evidence"

    def run(self, ctx: ProtocolContext) -> None:
        tasks = [lambda c=c: self._compute_one(ctx, c) for c in ctx.clients]
        ctx.executor.run_all(tasks)

    def _compute_one(self, ctx: ProtocolContext, client) -> None:
        client.compute_existence_evidence()   # 统计量（evidence.existence provider）
        client.analyze(kind="existence")      # 本地 LLM 存在合理性分析
