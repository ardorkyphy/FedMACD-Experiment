# -*- coding: utf-8 -*-
"""Assemble the final graph after orientation adjudication.

按裁决组装最终图：定边（有向）+ 存在但方向不定（无向保留）；
Cycle removal uses Tarjan SCC detection and deletes the directed edge with the
lowest final-judge confidence, without thresholding or conversion to undirected.
"""
from typing import Dict, List, Set, Tuple

from protocol.base import Stage
from protocol.context import ProtocolContext
from protocol.pipeline import register_stage


@register_stage
class AssembleStage(Stage):
    name = "assemble"

    def run(self, ctx: ProtocolContext) -> None:
        from config.schema import FinalGraph
        edges, undirected = [], []
        for key, rec in ctx.edge_records.items():
            if rec.direction == "->":
                edges.append(key)
            elif rec.direction == "<-":
                edges.append((key[1], key[0]))
            elif rec.doubtful:
                undirected.append(key)
        self._enforce_acyclic(ctx, edges)
        # 保序去重（v4.27）：任何路径下同一无向边只出现一次，防审计重复边
        edges = list(dict.fromkeys(edges))
        undirected = list(dict.fromkeys(undirected))
        graph = FinalGraph(nodes=list(ctx.global_vars), edges=edges, undirected=undirected)
        ctx.final_graph = graph
        self.log(ctx).info("最终图：定边=%d 无向=%d", len(edges), len(undirected))

    # ---------------- 去环（精修-A，从 v3 arbitration 复刻） ----------------

    def _enforce_acyclic(self, ctx: ProtocolContext, edges: List[Tuple[str, str]]) -> None:
        """Use Tarjan SCCs and delete the edge with the lowest final-judge confidence.

        循环直到无环；迭代上限 64 防御性兜底（异常时保留残余环并告警）。
        """
        guard = 0
        while guard < 64:
            guard += 1
            members = self._cycle_edges(ctx, edges)
            if not members:
                return
            key = min(members, key=lambda k: self._edge_combined_conf(ctx, k))
            conf = self._edge_combined_conf(ctx, key)
            rec_key = self._rec_key(ctx, key)
            rec = ctx.edge_records.get(rec_key)
            edges.remove(key)
            if rec is not None:
                rec.refine_action = "prune_acyclic"
                rec.decision = "drop"
                rec.direction = None            # 同步清空方向，避免 settled 虚高/审计矛盾（v4.25 修）
                rec.doubtful = False
            self.log(ctx).info("去环：删 %s（综合置信 %.3f）", "->".join(rec_key), conf)
        self.log(ctx).warning("去环达到迭代上限，仍有环残留（异常）")

    @staticmethod
    def _rec_key(ctx: ProtocolContext, key: Tuple[str, str]) -> Tuple[str, str]:
        """方向序 (src,dst) → 记录 key（edge_records 恒为 (u,v) 小端在前；<- 边是 (v,u)）。"""
        return key if key in ctx.edge_records else (key[1], key[0])

    def _edge_combined_conf(self, ctx: ProtocolContext, key: Tuple[str, str]) -> float:
        """去环排序键（升序=最弱优先）：裁判终裁置信（v4.23 评分器已删，
        最弱边=裁判置信最低；无裁判记录（异常）→ 中性 0.5）。"""
        rec = ctx.edge_records.get(self._rec_key(ctx, key))
        if rec is None:
            return 0.0
        v = getattr(rec, "verdict", None)
        if v is None:
            return 0.5
        return float(getattr(v, "confidence", 0.0) or 0.0)

    def _cycle_edges(self, ctx: ProtocolContext, edges: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        """返回当前定边图中所有环相关边（size≥2 的 SCC 内全部有向边）。"""
        adj, edge_map = self._directed(edges)
        nodes = set(adj) | {v for vs in adj.values() for v in vs}
        scc_of = self._scc_map(adj, nodes)
        members = []
        for (src, dst), key in edge_map.items():
            comp = scc_of.get(src) or set()
            if len(comp) >= 2 and scc_of.get(dst) == comp:
                members.append(key)
        return members

    @staticmethod
    def _directed(edges: List[Tuple[str, str]]):
        """有向边列表（key 已是 (src, dst) 序）→ adj、edge_map((src,dst)->key)。"""
        adj: Dict[str, set] = {}
        edge_map: Dict[Tuple[str, str], Tuple[str, str]] = {}
        for src, dst in edges:
            adj.setdefault(src, set()).add(dst)
            edge_map[(src, dst)] = (src, dst)
        return adj, edge_map

    @staticmethod
    def _scc_map(adj: Dict[str, set], nodes: set) -> Dict[str, set]:
        """Tarjan 强连通分量：node -> 所属 SCC 节点集。"""
        index = 0
        indices: Dict[str, int] = {}
        lowlink: Dict[str, int] = {}
        stack: List[str] = []
        on_stack: Set[str] = set()
        comps: Dict[str, set] = {}

        def _strongconnect(v: str) -> None:
            nonlocal index
            indices[v] = lowlink[v] = index
            index += 1
            stack.append(v)
            on_stack.add(v)
            for w in sorted(adj.get(v, set())):
                if w not in indices:
                    _strongconnect(w)
                    lowlink[v] = min(lowlink[v], lowlink[w])
                elif w in on_stack:
                    lowlink[v] = min(lowlink[v], indices[w])
            if lowlink[v] == indices[v]:
                comp = set()
                while stack:
                    w = stack.pop()
                    on_stack.remove(w)
                    comp.add(w)
                    if w == v:
                        break
                for m in comp:
                    comps[m] = comp

        for v in sorted(nodes):
            if v not in indices:
                _strongconnect(v)
        return comps
