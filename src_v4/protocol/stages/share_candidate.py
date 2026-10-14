# -*- coding: utf-8 -*-
"""Share the candidate edge set for the orientation stage.

服务端把最新候选集通知各客户端（客户端据此对全部候选边做方向性证据采集）。
"""
from protocol.base import Stage
from protocol.context import ProtocolContext
from protocol.pipeline import register_stage


@register_stage
class ShareCandidateStage(Stage):
    name = "share_candidate"

    def run(self, ctx: ProtocolContext) -> None:
        for client in ctx.clients:
            client.candidate_set = list(ctx.candidate_set)
        self.log(ctx).info("候选集已下发：%d 条边", len(ctx.candidate_set))
