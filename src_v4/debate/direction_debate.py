# -*- coding: utf-8 -*-
"""Edge-orientation debate protocol.

辩手 A/B 共享同一份完整证据包（v4.36 不分包），A 既定立场=u->v、B=v->u，
逐边多轮对辩（允许被说服后改立场）；答案 ∈ {u->v, v->u, related}。
结果 → DebateOutcome(stage="direction") 挂 rec.debate（覆盖 Phase1 的存在性辩论记录）。
"""
from typing import Any, Dict, Optional, Tuple

from debate.base import DebateProtocol, register_debate

_ANSWERS = ("u->v", "v->u", "related")


def _parse(out: Dict[str, Any], key: Tuple[str, str]) -> Optional[str]:
    """方向答案归一化（容忍 LLM 写真实变量名，相对基准 key 映射）。"""
    from debate.base import normalize_direction
    ans = normalize_direction(out.get("answer"), key)
    return ans if ans in _ANSWERS else None


@register_debate
class DirectionDebate(DebateProtocol):
    name = "direction_debate"
    stage = "direction"
    prompt_key = "debater_direction"
    batch_prompt_key = "debater_direction_batch"     # 批量模板（v4.28）
    stance_a = "主张方向 u->v（u 是 v 的原因）"          # 系统指定既定立场（v4.36 不分包）
    stance_b = "主张方向 v->u（v 是 u 的原因）"

    def packets_fn(self, ctx, key: Tuple[str, str], rec):
        from debate.packets import direction_packets
        return direction_packets(ctx, key, rec)

    def parse_fn(self, out: Dict[str, Any], key: Tuple[str, str]) -> Optional[str]:
        return _parse(out, key)

    def debate_one(self, ctx, key: Tuple[str, str], rec) -> Any:
        from config.schema import DebateOutcome
        packets = self.packets_fn(ctx, key, rec)
        answers: Dict[str, str] = {}
        confs: Dict[str, float] = {}
        reasons: Dict[str, str] = {}
        rounds = self._run_rounds(ctx, key, self.prompt_key, packets,
                                  self.parse_fn, answers, confs, reasons)
        if rounds == 0:
            return DebateOutcome(stage="direction")          # 无辩论
        return self._outcome(key, "direction", answers, confs, reasons, rounds)
