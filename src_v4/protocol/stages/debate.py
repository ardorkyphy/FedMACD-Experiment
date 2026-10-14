# -*- coding: utf-8 -*-
"""Server-side edge debate.

统一阶段：按当前 Phase 选择辩论协议（existence / direction），
辩论协议负责构造证据包（debate.packets）、调两辩手 LLM、汇总 DebateOutcome。
"""
from protocol.base import Stage
from protocol.context import ProtocolContext
from protocol.pipeline import register_stage


@register_stage
class DebateStage(Stage):
    name = "debate"

    def run(self, ctx: ProtocolContext) -> None:
        stage = ctx.current_phase   # 由 Pipeline/Phase 标记：existence / direction
        protocol = self._protocol(ctx, stage)
        protocol.debate(ctx, list(ctx.edge_records))

    def _protocol(self, ctx: ProtocolContext, stage: str):
        from debate.base import debate_protocols
        name = {"existence": "existence_debate", "direction": "direction_debate"}[stage]
        proto_cls = debate_protocols.get(name)
        if proto_cls is None:
            raise KeyError(f"辩论协议 {name!r} 未注册")
        return proto_cls()
