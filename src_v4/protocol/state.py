# -*- coding: utf-8 -*-
"""Protocol state types.

EdgeRecord：一条无向边（全局变量名键，u<v）的流程工作状态——
挂各客户端证据（ClientEvidence）、服务端聚合块、辩论结果、裁判裁决、最终决策。
"""
from typing import Any, Dict, Optional, Tuple


class EdgeRecord:
    """一条候选边的全流程状态（替代 v3 EdgeState）。"""

    def __init__(self, key: Tuple[str, str]) -> None:
        self.key: Tuple[str, str] = key                    # (u, v)，u < v
        self.client_evidence: Dict[str, Any] = {}          # client_id -> ClientEvidence
        self.aggregate: Optional[Any] = None               # AggregateBlock（服务端聚合统计块）
        self.debate: Optional[Any] = None                  # DebateOutcome（两辩手辩论结果）
        self.verdict: Optional[Any] = None                 # Verdict（裁判终裁）

        # ---- Phase1（存在性）决策 ----
        self.exist_decision: str = ""                      # keep / drop / uncertain
        # ---- Phase2（方向）决策 ----
        self.direction: Optional[str] = None               # "->" / "<-"
        self.doubtful: bool = False                        # 存在但方向不定 → 无向保留
        self.decision: str = ""                            # settle / doubtful / drop（审计）
        self.refine_action: str = ""                       # 去环动作审计：prune_acyclic（直接删最弱边）
