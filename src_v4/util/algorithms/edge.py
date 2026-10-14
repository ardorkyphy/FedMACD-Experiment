# -*- coding: utf-8 -*-
"""算法层统一输出类型与适配器（从 src/util/algorithms/edge.py 复刻）。

约定：Edge(u, v) 恒满足 u<v（索引序小端在前），direction 以 u→v 为基准：
"->" = u→v / "<-" = v→u / "none" = 未定向或双向。
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class Edge:
    """一条有向边：u<v（索引序小端在前），direction 以 u→v 为基准。"""
    u: int
    v: int
    direction: str          # "->" / "<-" / "none"


def graph_to_edges(gmat) -> List[Edge]:
    """causal-learn 邻接矩阵 → Edge 列表（g[i][j] = 边 (i,j) 在 i 端的端点）。

    本版本 causal-learn 端点编码（causallearn.graph.Endpoint）：TAIL=-1 / NULL=0 /
    ARROW=1 / CIRCLE=2 / STAR=3 / TAIL_AND_ARROW=4 / ARROW_AND_ARROW=5（v4.24 核实，
    旧文档"1/2/3/6"是历史版本编码）。映射：任一 0 → 无此边；一端 ARROW=1 另一端非 1
    （PC 的 ->/<-，FCI 的 o->、*->）→ 定向保留箭头信号；其余（<->、o-o、-o 等）
    → "none"（出边但方向未定/双向待定，与 schema 四态 undirected 语义一致）。
    """
    n = int(gmat.shape[0])
    out: List[Edge] = []
    for i in range(n):
        for j in range(i + 1, n):
            ei, ej = int(gmat[i][j]), int(gmat[j][i])
            if ei == 0 and ej == 0:
                continue
            if ej == 1 and ei != 1:
                out.append(Edge(i, j, "->"))
            elif ei == 1 and ej != 1:
                out.append(Edge(i, j, "<-"))
            else:
                out.append(Edge(i, j, "none"))
    return out


def fcm_to_edges(adj) -> List[Edge]:
    """LiNGAM 等 FCM 的 adjacency_matrix_（adj[i][j]!=0 ⇒ i→j）→ Edge 列表。"""
    n = int(adj.shape[0])
    out: List[Edge] = []
    for i in range(n):
        for j in range(i + 1, n):
            a, b = adj[i][j], adj[j][i]
            if a == 0 and b == 0:
                continue
            if a != 0 and b == 0:
                out.append(Edge(i, j, "->"))
            elif b != 0 and a == 0:
                out.append(Edge(i, j, "<-"))
            else:
                out.append(Edge(i, j, "none"))
    return out


def edges_to_evidence(edges: List[Edge], columns: List[str]) -> List[Tuple[str, str, str]]:
    """Edge（本地索引）→ (u, v, direction)（全局变量名，u<v），供 client 建骨架与方向索引。"""
    out: List[Tuple[str, str, str]] = []
    for e in edges:
        if e.u >= len(columns) or e.v >= len(columns):
            raise ValueError(
                f"Edge 索引越界: {e} 超出列数 {len(columns)}（算法输出含非观测节点？）")
        out.append((columns[e.u], columns[e.v], e.direction))
    return out
