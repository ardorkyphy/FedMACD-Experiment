# -*- coding: utf-8 -*-
"""Evidence upload and archival.

客户端把骨架/统计量/置信/LLM 判断上传到 ctx.edge_records（每条候选边挂 ClientEvidence）。
只传证据，不传原始数据（隐私边界）。
"""
from protocol.base import Stage
from protocol.context import ProtocolContext
from protocol.pipeline import register_stage


@register_stage
class UploadStage(Stage):
    name = "upload"

    def run(self, ctx: ProtocolContext) -> None:
        # Phase2 只上传候选集边的方向证据，避免把 Phase1 已删除边重新写回
        only = set(ctx.candidate_set) if ctx.current_phase == "direction" else None
        for client in ctx.clients:
            client.upload_evidence(ctx, only=only, slot=ctx.current_phase)   # 写 ctx.edge_records
        self.log(ctx).info("证据上传完成：候选边 %d（Phase2 限定候选集=%s）",
                           len(ctx.edge_records), only is not None)
