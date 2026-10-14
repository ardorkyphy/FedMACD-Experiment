# -*- coding: utf-8 -*-
"""Edge-existence debate protocol.

辩手 A/B 共享同一份完整证据包（v4.36 不分包），A 既定立场=存在、B=不存在，
逐边多轮对辩（允许被说服后改立场）；答案 ∈ {exists, none, uncertain}，
输出三件套（v4.9）。结果 → DebateOutcome(stage="existence") 挂 rec.debate。
"""
from typing import Any, Dict, Optional, Tuple

from debate.base import DebateProtocol, register_debate

_ANSWERS = ("exists", "none", "uncertain")


def _parse(out: Dict[str, Any]) -> Optional[str]:
    """存在性答案归一化（与方向辩论同标准，v4.19 修：容忍空格/大小写）。"""
    from debate.base import _norm_ans
    ans = _norm_ans(out.get("answer"))
    return ans if ans in _ANSWERS else None


@register_debate
class ExistenceDebate(DebateProtocol):
    name = "existence_debate"
    stage = "existence"
    prompt_key = "debater_existence"
    batch_prompt_key = "debater_existence_batch"     # 批量模板（v4.28）
    stance_a = "主张该边存在（exists）"                # 系统指定既定立场（v4.36 不分包）
    stance_b = "主张该边不存在（none）"

    def packets_fn(self, ctx, key: Tuple[str, str], rec):
        from debate.packets import existence_packets
        return existence_packets(ctx, key, rec)

    def parse_fn(self, out: Dict[str, Any], key: Tuple[str, str]) -> Optional[str]:
        return _parse(out)

    def debate_one(self, ctx, key: Tuple[str, str], rec) -> Any:
        from config.schema import DebateOutcome
        packets = self.packets_fn(ctx, key, rec)
        answers: Dict[str, str] = {}
        confs: Dict[str, float] = {}
        reasons: Dict[str, str] = {}
        rounds = self._run_rounds(ctx, key, self.prompt_key, packets,
                                  self.parse_fn, answers, confs, reasons)
        if rounds == 0:
            return DebateOutcome(stage="existence")          # 无辩论（模板缺失/调用失败）
        return self._outcome(key, "existence", answers, confs, reasons, rounds)
