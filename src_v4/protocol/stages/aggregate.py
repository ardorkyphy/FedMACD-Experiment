# -*- coding: utf-8 -*-
"""Server-side evidence aggregation.

并集确定候选集（Phase1）/ 沿用候选集（Phase2）；对每条候选边计算聚合统计块
（AggregateBlock：支持率/统计支持率/被解释率/联邦 p 值/方向分裂度/异质性，
由 evidence.aggregate 注册的聚合器组合生成）。
"""
from protocol.base import Stage
from protocol.context import ProtocolContext
from protocol.pipeline import register_stage


@register_stage
class AggregateStage(Stage):
    name = "aggregate"

    def run(self, ctx: ProtocolContext) -> None:
        from evidence.aggregate import rebuild_aggregate_blocks
        # Phase1 才并集定候选集；Phase2 沿用 Phase1 产物（空候选集=全裁，正常终止，
        # 绝不重建把已删边复活，v4.19 修）
        if ctx.current_phase == "existence" and not ctx.candidate_set:
            self._union_candidate_set(ctx)
        if not ctx.edge_records:
            self.log(ctx).info("候选集为空：正常终止，跳过聚合")
            return
        rebuild_aggregate_blocks(ctx)
        self.log(ctx).info("聚合完成：候选边 %d", len(ctx.edge_records))

    def _union_candidate_set(self, ctx: ProtocolContext) -> None:
        idx = {name: i for i, name in enumerate(ctx.global_vars)}
        union = set()
        for client in ctx.clients:
            for (u, v) in client.skeleton_edges:
                # key 恒按全局列索引序小端在前（v4.24 修：与客户端 key 基准一致；
                # 之前用字符串 min/max，列名非字典序时服务端候选集 key 与客户端
                # edge_directions（索引序）失配 → Phase2 alg_direction 全部落空）
                if u not in idx or v not in idx:
                    continue
                union.add((u, v) if idx[u] < idx[v] else (v, u))
        ctx.candidate_set = sorted(union, key=lambda kv: (idx[kv[0]], idx[kv[1]]))
        from protocol.state import EdgeRecord
        for key in ctx.candidate_set:
            if key not in ctx.edge_records:
                ctx.edge_records[key] = EdgeRecord(key)
