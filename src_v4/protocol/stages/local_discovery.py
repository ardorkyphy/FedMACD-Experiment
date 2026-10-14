# -*- coding: utf-8 -*-
"""Local graph discovery and skeleton construction.

每个客户端：LLM 选算法（ALGORITHM_SELECTION）→ 跑因果发现 → 无向骨架（方向抹掉，
第一阶段只保留存在性）。算法方向作为 alg_direction 证据留给第二阶段（v4.5）。
"""
from protocol.base import Stage
from protocol.context import ProtocolContext
from protocol.pipeline import register_stage


@register_stage
class LocalDiscoveryStage(Stage):
    name = "local_discovery"

    def run(self, ctx: ProtocolContext) -> None:
        tasks = [lambda c=c: self._discover_one(ctx, c) for c in ctx.clients]
        ctx.executor.run_all(tasks)

    def _discover_one(self, ctx: ProtocolContext, client) -> None:
        client.select_algorithm()          # 本地 LLM 选算法（复刻 v3 selector）
        client.do_causal_algorithm()       # 跑算法 → 无向骨架 + alg_direction 留档
        self.log(ctx).info("%s 骨架完成：边=%d", client.client_id, len(client.skeleton_edges))
