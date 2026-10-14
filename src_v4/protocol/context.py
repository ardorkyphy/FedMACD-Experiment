# -*- coding: utf-8 -*-
"""Shared state for the two-stage protocol.

一次实验的共享状态容器：资源（clients/server/executor）+ 配置 + 流程状态。
Stage 只依赖 ctx，不自己持有资源（依赖注入）。
v4 流程状态：edge_records（每边 EdgeRecord，含证据矩阵/聚合块/评分/辩论/裁决）、
candidate_set（Phase1 产物）、dropped_edges（Phase1 删除审计）、final_graph（Phase2 产物）。
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

from util.execution import TaskExecutor
from util.timing import PhaseTimings


@dataclass
class ProtocolContext:
    config: Dict[str, Any]
    clients: List[Any]
    server: Any
    executor: TaskExecutor
    loaded: Any                              # util.data.LoadedData
    current_phase: str = "existence"      # 当前阶段：existence / direction（debate/judge 分发用）

    # Server-aligned semantics and evidence-field documentation.
    evidence_pack: Dict[str, Any] = field(default_factory=dict)   # {"semantics": {...}, "field_docs": {select/existence/direction/aggregate}}

    # ---- 对齐产物（AlignStage 写入） ----
    global_vars: List[str] = field(default_factory=list)
    var_mapping: Dict[str, Dict[str, str]] = field(default_factory=dict)

    # ---- 证据与裁决（evidence/debate 层读写；评分器已删 v4.23） ----
    edge_records: Dict[Tuple[str, str], Any] = field(default_factory=dict)   # (u,v) -> EdgeRecord
    candidate_set: List[Tuple[str, str]] = field(default_factory=list)       # Phase1 输出（u<v）
    dropped_edges: List[Tuple[str, str]] = field(default_factory=list)       # Phase1 删除审计
    dropped_records: Dict[Tuple[str, str], Any] = field(default_factory=dict)  # Full records for rejected edges.
    final_graph: Any = None                  # FinalGraph（AssembleStage 写入；Pipeline 返回）
    # ---- 审计与计时 ----
    timings: PhaseTimings = field(default_factory=PhaseTimings)
    llm_calls: int = 0                       # 累计 LLM 调用（成本指标）
