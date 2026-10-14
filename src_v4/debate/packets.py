# -*- coding: utf-8 -*-
"""Construct protocol evidence packages.

辩手输入 = 聚合后证据（只含统计量/LLM 判断/全局聚合块，不含原始数据，隐私边界同 prompt.privacy_guard）。
Existence and direction debaters share the same complete evidence package
（全部客户端的本地证据 + 全局聚合块）；对抗性来自系统指定的既定立场（{stance} 由
debate.base injects the assigned stance. Signal-based packet splitting is not used because it gives the sides unequal evidence.
返回 {"_first": "full", "_second": "full", "full": 文本块}，A/B 各持同一份。
"""
from typing import Any, Dict, List, Tuple

from evidence.render import fmt_aggregate, fmt_direction_stats, fmt_llm, fmt_stats


def _evidence_of(rec, cid: str) -> Dict[str, Any]:
    """客户端证据块（ClientEvidence.existence / direction / llm → 扁平 dict）。"""
    ce = (rec.client_evidence or {}).get(cid)
    if ce is None:
        return {}
    out: Dict[str, Any] = {}
    for slot in ("existence", "direction"):
        obj = getattr(ce, slot, None)
        if obj is not None:
            d = _asdict(obj)
            for k, v in d.items():
                out[k] = v
    return out


def _asdict(obj: Any) -> Dict[str, Any]:
    from dataclasses import asdict
    try:
        return asdict(obj)
    except Exception:
        if hasattr(obj, "__dict__"):
            return dict(obj.__dict__)
        return {}



def _full_block(title: str, lines: List[str], agg: str) -> str:
    body = "\n".join(lines) or "（无客户端证据）"
    return f"{title}\n{body}\n{agg}"


def existence_packets(ctx, key: Tuple[str, str], rec) -> Dict[str, str]:
    """存在性完整证据包：{"_first": "full", "_second": "full", "full": txt}。

    v4.36 不分包：全部客户端证据（存在性统计 + 本地 LLM 判断）逐客户端排列，
    末尾附全局聚合块；A/B 拿同一份，各自立场由 {stance} 指定。
    """
    stats_on = bool((ctx.config.get("STATISTICAL_EVIDENCE") or {}).get("enabled", True))
    lines = []
    for cid in sorted((rec.client_evidence or {}).keys()):
        ev = _evidence_of(rec, cid)
        ce = (rec.client_evidence or {}).get(cid) or {}
        if not ev:
            continue
        line = f"[{cid}] {fmt_stats(ev, stats=stats_on)}"
        llm_txt = fmt_llm(getattr(ce, "llm", None))
        if llm_txt:
            line += f"；{llm_txt}"
        lines.append(line)
    agg = fmt_aggregate(getattr(rec, "aggregate", None), "existence", stats=stats_on)
    full = _full_block("【全部客户端证据】", lines, agg)
    return {"_first": "full", "_second": "full", "full": full}


def direction_packets(ctx, key: Tuple[str, str], rec) -> Dict[str, str]:
    """方向性完整证据包：{"_first": "full", "_second": "full", "full": txt}。

    v4.36 不分包：全部客户端证据（方向统计 + 本地 LLM 判断）逐客户端排列，
    末尾附全局聚合块；A/B 拿同一份，各自立场由 {stance} 指定。
    """
    stats_on = bool((ctx.config.get("STATISTICAL_EVIDENCE") or {}).get("enabled", True))
    lines = []
    for cid in sorted((rec.client_evidence or {}).keys()):
        ev = _evidence_of(rec, cid)
        ce = (rec.client_evidence or {}).get(cid) or {}
        if not ev:
            continue
        line = f"[{cid}] {fmt_direction_stats(ev, stats=stats_on)}"
        llm_txt = fmt_llm(getattr(ce, "llm", None))
        if llm_txt:
            line += f"；{llm_txt}"
        lines.append(line)
    agg = fmt_aggregate(getattr(rec, "aggregate", None), "direction", stats=stats_on)
    full = _full_block("【全部客户端证据】", lines, agg)
    return {"_first": "full", "_second": "full", "full": full}
