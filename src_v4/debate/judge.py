# -*- coding: utf-8 -*-
"""Adjudication for the existence and orientation stages.

Every candidate receives a final judgment without score pre-filtering:
所有候选边分批小批量（JUDGE_BATCH_EDGES）过裁判，评分器已删（v4.23）。
- ExistenceJudge：存在性终裁（exists/none/uncertain）→ Phase1 重建新候选集。
- DirectionJudge：方向性终裁（u->v/v->u/related）→ Phase2 落定边/无向。
裁判输入（v4.23）：两辩手最终立场 + 全局聚合统计块 + 结构上下文（环/v 结构提示）+
变量语义。输出三件套 {answer, score, reason}（v4.9）。
"""
from typing import Any, Dict, List, Optional, Tuple

from config.schema import Verdict
from debate.base import (Judge, _clamp_score, _judge_call, norm_edge_key,
                           norm_undirected_key, register_judge)


class _BaseJudge(Judge):
    """两阶段裁判共享：批量输入构造 + LLM 调用 + 逐边解析 + 直接采纳。"""

    def judge_batch(self, ctx, chunk: List[Tuple[str, str]]) -> Dict[Tuple[str, str], Verdict]:
        block = "\n\n".join(self._edge_block(ctx, key, ctx.edge_records.get(key))
                            for key in chunk)
        out = _judge_call(ctx, self.prompt_key, block)
        verdicts: Dict[Tuple[str, str], Verdict] = {}
        if not out:
            return verdicts                      # 整批失败 → 返回空，上层拆半递归重试
        _norm = norm_undirected_key if self.stage == "existence" else norm_edge_key
        by_edge: Dict[str, Dict] = {}
        for it in (out.get("edges", []) or []):
            if isinstance(it, dict) and str(it.get("edge", "") or "").strip():
                by_edge[_norm(it.get("edge"))] = it   # v4.28 修：与匹配端同口径（存在性=无向归一）
        for key in chunk:
            rec = ctx.edge_records.get(key)
            if rec is None:
                continue
            fwd = _norm("->".join(key))
            rev = _norm("->".join(reversed(key)))
            it = by_edge.get(fwd)
            base = key
            if it is None and rev != fwd:        # Reverse matches require flipping the orientation baseline.
                it = by_edge.get(rev)
                base = (key[1], key[0])
            v = self._parse(it, base) if isinstance(it, dict) else None
            if v is None:
                continue                         # 单条缺失/解析失败 → 维持初判
            if base != key:                      # 反序命中：答案相对反序基准，翻回记录 key 基准
                v.answer = {"u->v": "v->u", "v->u": "u->v"}.get(v.answer, v.answer)
            v.applied = True
            self._set_decision(rec, v.answer)    # 直接采纳（v4：裁判是终裁，无改判门）
            verdicts[key] = v
        return verdicts

    def _edge_block(self, ctx, key: Tuple[str, str], rec) -> str:
        """裁判输入块（v4.23）：辩论立场 + 聚合块 + 结构上下文（评分器已删）。"""
        # Render existence-stage candidates as undirected to avoid orientation leakage.
        stats_on = bool((ctx.config.get("STATISTICAL_EVIDENCE") or {}).get("enabled", True))
        lines = [f"【边】{key[0]} - {key[1]}（无向）" if self.stage == "existence"
                 else f"【边】{key[0]}->{key[1]}"]
        db = getattr(rec, "debate", None)
        if db is not None and (getattr(db, "answer_a", "") or getattr(db, "answer_b", "")):
            lines.append("辩手A立场=%s 置信=%.2f 理由：%s" % (
                db.answer_a or "（无）", float(getattr(db, "conf_a", 0.0) or 0.0),
                str(getattr(db, "reason_a", "") or "") or "（无）"))
            lines.append("辩手B立场=%s 置信=%.2f 理由：%s" % (
                db.answer_b or "（无）", float(getattr(db, "conf_b", 0.0) or 0.0),
                str(getattr(db, "reason_b", "") or "") or "（无）"))
        else:
            lines.append("辩论：无有效辩论记录")
        # 裁判补辩论全过程（逐轮立场/置信/理由 + 改立场提示）：终裁参考双方论据演变
        if db is not None:
            tr = getattr(db, "trace", None) or []
            if tr:
                role_txt = ("A=u->v 方 B=v->u 方" if self.stage == "direction"
                            else "A=存在方 B=不存在方")
                lines.append("辩论过程（逐轮，%s）：" % role_txt)
                for t in tr:
                    lines.append("R%s A=%s(%.2f) 理由：%s | B=%s(%.2f) 理由：%s" % (
                        getattr(t, "round", ""),
                        getattr(t, "a_answer", "") or "-", float(getattr(t, "a_conf", 0.0) or 0.0),
                        str(getattr(t, "a_reason", "") or "") or "（无）",
                        getattr(t, "b_answer", "") or "-", float(getattr(t, "b_conf", 0.0) or 0.0),
                        str(getattr(t, "b_reason", "") or "") or "（无）"))
                a0, a1 = getattr(tr[0], "a_answer", ""), getattr(tr[-1], "a_answer", "")
                b0, b1 = getattr(tr[0], "b_answer", ""), getattr(tr[-1], "b_answer", "")
                if a0 and a1 and a0 != a1:
                    lines.append("（A 改立场：%s → %s）" % (a0, a1))
                if b0 and b1 and b0 != b1:
                    lines.append("（B 改立场：%s → %s）" % (b0, b1))
        agg = getattr(rec, "aggregate", None)
        if agg is not None:
            from evidence.render import fmt_aggregate
            lines.append(fmt_aggregate(agg, self.stage, stats=stats_on))   # v4.28 分阶段渲染
        hint = self._context_hint(ctx, key, rec)
        if hint:
            lines.append("结构上下文：" + hint)
        return "\n".join(lines)

    def _parse(self, it: Dict[str, Any], key: Tuple[str, str]) -> Optional[Verdict]:
        if self.stage == "direction":
            from debate.base import normalize_direction
            raw = normalize_direction(it.get("answer"), key)
        else:
            raw = str(it.get("answer", "") or "").strip().lower().replace(" ", "")
        ans = self._valid_answers.get(raw)
        if ans is None:
            return None
        return Verdict(answer=ans, confidence=_clamp_score(it.get("score")),
                       reason=str(it.get("reason", "") or ""))

    # ---------- 子类钩子 ----------

    prompt_key: str = ""
    _valid_answers: Dict[str, str] = {}

    def _context_hint(self, ctx, key, rec) -> str:
        return ""

    def _set_decision(self, rec, answer: str) -> None:
        raise NotImplementedError


@register_judge
class ExistenceJudge(_BaseJudge):
    """存在性裁判：全员终裁 → 新候选集（finalize 过滤 drop）。"""

    name = "existence_judge"
    stage = "existence"
    prompt_key = "judge_existence"
    _valid_answers = {"exists": "exists", "none": "none", "uncertain": "uncertain"}

    def _context_hint(self, ctx, key, rec) -> str:
        degree = 0
        for cid, ce in (rec.client_evidence or {}).items():
            ev = getattr(ce, "existence", None)
            if ev is not None and bool(getattr(ev, "skeleton_has", False)):
                degree += 1
        return "骨架支持客户端数=%d（度数代理）" % degree

    def _set_decision(self, rec, answer: str) -> None:
        rec.exist_decision = {"exists": "keep", "none": "drop",
                              "uncertain": "uncertain"}.get(answer, "uncertain")

    def finalize(self, ctx) -> None:
        """Phase1 产物：新候选集 = 未被判 drop 的边（uncertain 保留进 Phase2）。

        drop 边从 edge_records 剔除（Phase2 只处理存活边），进 ctx.dropped_edges 审计。
        """
        dropped = [k for k in list(ctx.edge_records)
                   if getattr(ctx.edge_records[k], "exist_decision", "") == "drop"]
        drop_set = set(dropped)
        if drop_set:
            ctx.candidate_set = [k for k in ctx.candidate_set if k not in drop_set]
            for k in dropped:
                ctx.dropped_records[k] = ctx.edge_records[k]   # Archive the complete decision record before removal.
                del ctx.edge_records[k]
            ctx.dropped_edges.extend(dropped)
        from util.log import get
        log = get("judge.existence")
        for k in dropped:                      # Log the adjudication rationale for removed edges.
            rec = ctx.dropped_records[k]
            vd = getattr(rec, "verdict", None)
            db = getattr(rec, "debate", None)
            log.warning("DROP %s-%s | 裁判=%s conf=%.2f 理由=%s | 辩A=%s(%.2f) 辩B=%s(%.2f)",
                        k[0], k[1],
                        getattr(vd, "answer", "") or "?", float(getattr(vd, "confidence", 0.0) or 0.0),
                        str(getattr(vd, "reason", "") or ""),
                        getattr(db, "answer_a", "") or "?", float(getattr(db, "conf_a", 0.0) or 0.0),
                        getattr(db, "answer_b", "") or "?", float(getattr(db, "conf_b", 0.0) or 0.0))
        log.info("新候选集：%d 条（删 %d，审计 dropped_edges=%d）",
                 len(ctx.candidate_set), len(dropped), len(ctx.dropped_edges))


@register_judge
class DirectionJudge(_BaseJudge):
    """方向性裁判：全员终裁 → 定边/无向（finalize 汇总）。"""

    name = "direction_judge"
    stage = "direction"
    prompt_key = "judge_direction"
    _valid_answers = {"u->v": "u->v", "v->u": "v->u", "related": "related"}

    def _context_hint(self, ctx, key, rec) -> str:
        # Cycle hints are omitted because batch-local directions would make them order-dependent.
        # 裁决结果随批大小漂移；去环由 assemble 阶段独立 SCC 处理兜底，提示信息冗余。
        return "存在性裁决=%s；方向争议（全员终裁，含弱方向与 related）" % (
            getattr(rec, "exist_decision", ""))

    def _set_decision(self, rec, answer: str) -> None:
        """Apply direction verdicts: forward, reverse, or retain as undirected."""
        if answer == "u->v":
            rec.direction, rec.decision = "->", "settle"
        elif answer == "v->u":
            rec.direction, rec.decision = "<-", "settle"
        else:                                # related：存在但方向不定 → 无向保留
            rec.doubtful, rec.decision = True, "doubtful"

    def finalize(self, ctx) -> None:
        """Phase2 产物：汇总定边/无向/未决统计（assemble 阶段据此组装最终图）。"""
        from util.log import get
        log = get("judge.direction")
        settled = sum(1 for r in ctx.edge_records.values() if getattr(r, "direction", None))
        doubtful = sum(1 for r in ctx.edge_records.values() if getattr(r, "doubtful", False))
        unsettled = len(ctx.edge_records) - settled - doubtful
        log.info("方向裁判完成：定边=%d 无向=%d 未决=%d（未决=解析失败维持 related 兜底）",
                 settled, doubtful, unsettled)
