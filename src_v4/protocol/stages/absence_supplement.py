# -*- coding: utf-8 -*-
"""Supplementary existence evidence for initially absent edges.

候选集确定后，客户端对"候选集里自己没出的边"定向补算独立性证据（统计量），
并补本地 LLM 存在性判断（v4.21 修：与统计对称，证据矩阵全客户端覆盖），
让"无边"方有真实证据而非沉默；补采结果立即回写 ctx.edge_records
（upload 在 aggregate 之前已跑过，补采必须重新归档），随后重建聚合块。
"""
from protocol.base import Stage
from protocol.context import ProtocolContext
from protocol.pipeline import register_stage


@register_stage
class AbsenceSupplementStage(Stage):
    name = "absence_supplement"

    def run(self, ctx: ProtocolContext) -> None:
        tasks = [lambda c=c: self._supplement_one(ctx, c) for c in ctx.clients]
        ctx.executor.run_all(tasks)
        from evidence.aggregate import rebuild_aggregate_blocks
        rebuild_aggregate_blocks(ctx)   # 补采回写后刷新聚合块（v4.21 修）

    def _supplement_one(self, ctx: ProtocolContext, client) -> None:
        missing = [k for k in ctx.candidate_set if not client.has_skeleton_edge(k)]
        if missing:
            client.supplement_existence_evidence(missing)      # 统计量补采
            client.analyze(kind="existence", edges=missing)    # 本地 LLM 存在性判断（v4.21 修）
            client.upload_evidence(ctx)                        # 回写证据（幂等，只覆盖该边）
        self.log(ctx).info("%s 补采 %d 条缺失边证据", client.client_id, len(missing))
