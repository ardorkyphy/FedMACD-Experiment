# -*- coding: utf-8 -*-
"""Debate and adjudication abstractions.

- DebateProtocol：一次逐边辩论（存在性/方向性两个实现，@register_debate 进 debate_protocols），
  产出 DebateOutcome 挂到 EdgeRecord.debate；证据包构造在 debate/packets.py（协议只调包，不自己拼证据）。
- Judge: every candidate is judged in small batches without score pre-filtering,
  两阶段各一个实现（@register_judge 进 judges）；批次内一次 LLM 调用，返回 {edge: Verdict}，
  批次覆盖不全 → 未覆盖子集拆半递归重试（v3 兜底），单条仍失败才维持评分器初判
  （v4.23 起为确定性统计兜底，applied=False，失败兜底而非改判门）。finalize(ctx) 是阶段产物钩子
  （Phase1=重建候选集 / Phase2=汇总定边）。
- _debater_call / _judge_call：统一 LLM 调用（三件套 {answer, score, reason}，v4.9；模板缺失 → None）。
"""
import json
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Type

debate_protocols: Dict[str, Type["DebateProtocol"]] = {}
judges: Dict[str, Type["Judge"]] = {}


def register_debate(cls: Type["DebateProtocol"]) -> Type["DebateProtocol"]:
    name = getattr(cls, "name", "") or cls.__name__
    if name in debate_protocols:
        print(f"[debate] 重复注册辩论协议覆盖: {name}")
    debate_protocols[name] = cls
    return cls


def register_judge(cls: Type["Judge"]) -> Type["Judge"]:
    name = getattr(cls, "name", "") or cls.__name__
    if name in judges:
        print(f"[debate] 重复注册裁判覆盖: {name}")
    judges[name] = cls
    return cls


def _system_prompt() -> Optional[str]:
    try:
        from config.prompt import PROMPTS
        return str(PROMPTS.get("privacy_guard", "") or "") or None
    except Exception:
        return None


def _template(key: str) -> str:
    try:
        from config.prompt import PROMPTS
        return str(PROMPTS.get(key, "") or "")
    except Exception:
        return ""


def _norm_ans(ans: Any) -> str:
    """LLM 答案归一化小写（容忍 unicode 箭头与空白）。"""
    return str(ans or "").strip().lower().replace("\u2192", "->").replace("\u2190", "<-").replace(" ", "")


def normalize_direction(raw: Any, key: Optional[Tuple[str, str]] = None) -> str:
    """方向答案归一化到基准方向：key=(u,v) 下合法值为 u->v / v->u / related。

    LLM 可能直接写真实变量名（如 "tub->either"，相对基准 either->tub 即 v->u）→ 映射归一化；
    无法识别 → 返回空串（调用方按解析失败处理）。
    """
    a = _norm_ans(raw)
    if a in ("u->v", "v->u", "related"):
        return a
    if key:
        u, v = _norm_ans(key[0]), _norm_ans(key[1])
        if a == f"{u}->{v}":
            return "u->v"
        if a == f"{v}->{u}":
            return "v->u"
    return ""


def norm_edge_key(s: Any) -> str:
    """边标识归一化（LLM 常写 'a -> b'，容忍内部空白/大小写/unicode 箭头，v4.21 修）。
    ASCII '<-' is normalized so 'a<-b' is interpreted as 'b->a'."""
    a = _norm_ans(s)
    if "<-" in a:
        parts = [t.strip() for t in a.split("<-")]
        if len(parts) == 2 and parts[0] and parts[1]:
            a = f"{parts[1]}->{parts[0]}"
    return a


def norm_undirected_key(s: Any) -> str:
    """无向边标识归一化（v4.27 修：存在性阶段无向渲染的解析匹配用）。

    'a->b' / 'b<-a' / 'a-b' / 'a ↔ b' 统一为按变量名排序的 'a-b'，方向不参与。
    """
    import re
    a = _norm_ans(s).replace("->", "-").replace("<-", "-").replace("\u2194", "-")
    toks = [t for t in re.split(r"-", a) if t]
    return "-".join(sorted(toks)) if len(toks) >= 2 else a


def _clamp_score(v: Any) -> float:
    try:
        return float(min(max(float(v), 0.0), 1.0))
    except (TypeError, ValueError):
        return 0.0


def _semantics_txt(ctx) -> str:
    return json.dumps((ctx.config.get("DATA") or {}).get("semantics", {}) or {},
                      ensure_ascii=False) or "（无）"


def _fmt_history(hist, stage: str) -> str:
    """Render the complete round-by-round stance, confidence, and rationale history."""
    if not hist:
        return "（无，首轮）"
    role_txt = ("A=存在方 B=不存在方" if stage == "existence"
                else "A=u->v 方 B=v->u 方")
    out = ["辩论历史（逐轮，%s）：" % role_txt]
    for t in hist:
        out.append("R%s A=%s(%.2f) 理由：%s | B=%s(%.2f) 理由：%s" % (
            getattr(t, "round", ""),
            getattr(t, "a_answer", "") or "-", float(getattr(t, "a_conf", 0.0) or 0.0),
            str(getattr(t, "a_reason", "") or "") or "（无）",
            getattr(t, "b_answer", "") or "-", float(getattr(t, "b_conf", 0.0) or 0.0),
            str(getattr(t, "b_reason", "") or "") or "（无）"))
    return "\n".join(out)


def _debater_call(ctx, role: str, prompt_key: str, edge_txt: str, packet: str,
                  opponent: str = "", stance: str = "",
                  history: str = "") -> Optional[Dict[str, Any]]:
    """单次辩手 LLM 调用（三件套）。模板缺失/调用失败 → None（上层按无辩论处理）。

    stance：系统指定既定立场（v4.36 不分包：A/B 共享同一证据包，对抗来自立场）。
    history：截至上一轮的双方辩论完整历史（v4.36 新增，辩手可见立场演变）。"""
    stats_on = bool((ctx.config.get("STATISTICAL_EVIDENCE") or {}).get("enabled", True))
    tpl = _template(prompt_key)
    if not tpl:
        return None
    from evidence.existence import field_doc_text
    from evidence.direction import field_doc_text as dir_field_doc_text
    from evidence.aggregate import agg_field_doc_text
    _fdocs = (ctx.evidence_pack or {}).get("field_docs", {}) or {}
    prompt = (tpl.replace("{edge}", edge_txt).replace("{packet}", packet or "（无）")
              .replace("{opponent}", opponent or "（首轮，无对方观点）")
              .replace("{history}", history or "（无）")
              .replace("{stance}", stance or "（未指定）")
              .replace("{field_doc}", (_fdocs.get("existence") or field_doc_text()) if stats_on else "")
              .replace("{dir_field_doc}", (_fdocs.get("direction") or dir_field_doc_text()) if stats_on else "")
              .replace("{agg_field_doc}", (_fdocs.get("aggregate") or agg_field_doc_text()) if stats_on else "")
              .replace("{semantics}", _semantics_txt(ctx)))
    llm = getattr(ctx.server, "llm", None)
    if llm is None:
        return None
    try:
        out = llm.chat_json(prompt, role=role, system_prompt=_system_prompt()) or {}
    except Exception as exc:
        from util.log import get
        get("debate").warning("辩手 %s 调用失败（%s）: %s", role, prompt_key, exc)
        return None
    return out if isinstance(out, dict) else None


def _debater_call_batch(ctx, role: str, prompt_key: str, edges_block: str,
                         stance: str = "") -> Optional[Dict[str, Any]]:
    """单次批量辩手 LLM 调用（一批多条边一次调用；输出 {"edges": [三件套×N]}，v4.28）。

    stance：系统指定既定立场（v4.36 不分包，见 _debater_call）。"""
    stats_on = bool((ctx.config.get("STATISTICAL_EVIDENCE") or {}).get("enabled", True))
    tpl = _template(prompt_key)
    if not tpl:
        return None
    from evidence.existence import field_doc_text
    from evidence.direction import field_doc_text as dir_field_doc_text
    from evidence.aggregate import agg_field_doc_text
    _fdocs = (ctx.evidence_pack or {}).get("field_docs", {}) or {}   # v4.33
    prompt = (tpl.replace("{edges_block}", edges_block or "（无）")
              .replace("{stance}", stance or "（未指定）")
              .replace("{field_doc}", (_fdocs.get("existence") or field_doc_text()) if stats_on else "")
              .replace("{dir_field_doc}", (_fdocs.get("direction") or dir_field_doc_text()) if stats_on else "")
              .replace("{agg_field_doc}", (_fdocs.get("aggregate") or agg_field_doc_text()) if stats_on else "")
              .replace("{semantics}", _semantics_txt(ctx)))
    llm = getattr(ctx.server, "llm", None)
    if llm is None:
        return None
    try:
        out = llm.chat_json(prompt, role=role, system_prompt=_system_prompt()) or {}
    except Exception as exc:
        from util.log import get
        get("debate").warning("辩手 %s 批量调用失败（%s）: %s", role, prompt_key, exc)
        return None
    return out if isinstance(out, dict) else None


def _judge_call(ctx, prompt_key: str, edges_block: str) -> Optional[Dict[str, Any]]:
    """单次裁判批调用（一批多条边，一次 LLM；输出 {"edges": [三件套×N]}）。
    模板缺失/调用失败 → None（本批全部按初判兜底，上层逐边维持）。"""
    stats_on = bool((ctx.config.get("STATISTICAL_EVIDENCE") or {}).get("enabled", True))
    tpl = _template(prompt_key)
    if not tpl:
        return None
    from evidence.aggregate import agg_field_doc_text
    _fdocs = (ctx.evidence_pack or {}).get("field_docs", {}) or {}   # v4.33
    prompt = (tpl.replace("{edges_block}", edges_block or "（无）")
              .replace("{agg_field_doc}", (_fdocs.get("aggregate") or agg_field_doc_text()) if stats_on else "")
              .replace("{semantics}", _semantics_txt(ctx)))
    llm = getattr(ctx.server, "llm", None)
    if llm is None:
        return None
    try:
        out = llm.chat_json(prompt, role="judge", system_prompt=_system_prompt()) or {}
    except Exception as exc:
        from util.log import get
        get("debate").warning("裁判 %s 批调用失败: %s", prompt_key, exc)
        return None
    return out if isinstance(out, dict) else None


class DebateProtocol(ABC):
    """分批多轮辩论（v4.28 起默认批量路径：一批多条边一次 LLM 调用；单边 debate_one 保留兼容）。"""

    name: str = ""
    stage: str = ""              # existence / direction（决定进哪个辩论）
    prompt_key: str = ""         # 单边模板 key（debate_one 用）
    batch_prompt_key: str = ""   # 批量模板 key（debate() 用，v4.28）
    stance_a: str = ""           # 系统指定既定立场：A 方（v4.36 不分包，对抗来自立场）
    stance_b: str = ""           # 系统指定既定立场：B 方

    def _stance(self, side: str) -> str:
        """该侧辩手的既定立场文本（填 {stance} 占位符；允许被说服后改立场）。"""
        return self.stance_a if side == "a" else self.stance_b

    def debate(self, ctx, edges: List[Tuple[str, str]]) -> None:
        """对全部候选边分批辩论（DEBATE_BATCH_EDGES 小批量，防注意力稀释），结果写入 rec.debate。"""
        from util.log import get
        log = get(f"debate.{self.stage}")
        batch = max(int(ctx.config.get("DEBATE_BATCH_EDGES", 20) or 20), 1)
        done = 0
        for start in range(0, len(edges), batch):
            chunk = [k for k in edges[start:start + batch] if k in ctx.edge_records]
            if chunk:
                self._debate_batch(ctx, chunk)
                done += len(chunk)
        log.info("%s 辩论完成：%d/%d 条边（批大小=%d）", self.name, done, len(edges), batch)

    def _debate_batch(self, ctx, chunk: List[Tuple[str, str]]) -> None:
        """一批多条边：多轮对辩，每轮每辩手一次批量 LLM 调用（v4.28）。"""
        from config.schema import DebateOutcome
        items = []
        for key in chunk:
            rec = ctx.edge_records.get(key)
            if rec is None:
                continue
            items.append((key, rec, self.packets_fn(ctx, key, rec)))
        if not items:
            return
        answers = {key: {} for key, _, _ in items}
        confs = {key: {} for key, _, _ in items}
        reasons = {key: {} for key, _, _ in items}
        rounds, trace = self._run_rounds_batch(ctx, items, self.batch_prompt_key,
                                               answers, confs, reasons)
        from util.log import get
        log = get(f"debate.{self.stage}")
        for key, rec, _ in items:
            if not answers.get(key):
                rec.debate = DebateOutcome(stage=self.stage)    # 该边无有效辩论（模板缺失/调用失败）
            else:
                rec.debate = self._outcome(key, self.stage, answers[key],
                                           confs[key], reasons[key], rounds,
                                           trace.get(key))
            tr = getattr(rec.debate, "trace", None) or []
            if tr:
                evo_a = "->".join("%s(%.2f)" % (t.a_answer or "-", t.a_conf) for t in tr)
                evo_b = "->".join("%s(%.2f)" % (t.b_answer or "-", t.b_conf) for t in tr)
                log.info("辩论过程 %s-%s: A[%s] B[%s]", key[0], key[1], evo_a, evo_b)

    @abstractmethod
    def debate_one(self, ctx, key: Tuple[str, str], rec) -> Any:
        """对单条边跑完整个辩论，返回 DebateOutcome（无证据/模板缺失 → 空结果）。"""
        raise NotImplementedError

    # ---------- 子类共享：多轮对辩（批量路径 v4.28 / 单边路径兼容） ----------

    def _run_rounds_batch(self, ctx, items, prompt_key: str,
                          answers, confs, reasons) -> int:
        """批量对辩：每轮每辩手一次 LLM 调用覆盖批内全部边，逐边独立维护立场。

        items: [(key, rec, packets)]；answers/confs/reasons: {key: {side: val}}。
        返回实际完成的轮数（模板缺失/调用失败 → 0）。
        """
        max_rounds = max(int(ctx.config.get("DEBATE_MAX_ROUNDS", 2) or 2), 1)
        roles = {"a": "debater_a", "b": "debater_b"}
        roles_rev = {"a": "b", "b": "a"}
        peer = {key: {"a": pk[pk["_first"]], "b": pk[pk["_second"]]}
                for key, _, pk in items}
        edge_txt = {key: (" - ".join(key) if self.stage == "existence" else "->".join(key))
                    for key, _, _ in items}
        from config.schema import DebateRound
        rounds = 0
        trace: Dict[Tuple[str, str], List] = {key: [] for key, _, _ in items}
        for r in range(max_rounds):
            got = 0
            for side in ("a", "b"):
                got += self._debater_batch_recursive(
                    ctx, roles[side], prompt_key, items, answers, confs, reasons,
                    side, r, roles_rev, edge_txt, peer, trace)
            if got == 0:
                break                      # 本轮两方都无有效输出 → 提前终止
            rounds += 1
            for key, _, _ in items:       # Snapshot each round for reproducible auditing.
                trace[key].append(DebateRound(
                    round=rounds,
                    a_answer=answers[key].get("a", ""), a_conf=round(float(confs[key].get("a", 0.0)), 4),
                    a_reason=reasons[key].get("a", "") or "",
                    b_answer=answers[key].get("b", ""), b_conf=round(float(confs[key].get("b", 0.0)), 4),
                    b_reason=reasons[key].get("b", "") or ""))
        return rounds, trace

    def _debater_batch_recursive(self, ctx, role: str, prompt_key: str,
                                 items, answers, confs, reasons,
                                 side: str, r: int, roles_rev, edge_txt, peer,
                                 trace) -> int:
        """Run one debater in batches; recursively split failed or incomplete batches.
        仅对未覆盖子集拆半递归重试；单条仍失败 → 该侧该轮不更新（维持上一轮立场）并记 warning。
        存在性/方向性两阶段共享（_run_rounds_batch 调用）。trace：逐边完整辩论历史（v4.36）。"""
        if not items:
            return 0
        p_side = roles_rev[side]
        lines = []
        for key, rec, _ in items:
            opponent_view = ""
            if r > 0 and answers[key].get(p_side):
                opponent_view = ("对方立场=%s 置信=%.2f 理由：%s"
                                 % (answers[key][p_side], confs[key].get(p_side, 0.0),
                                    reasons[key].get(p_side, "") or "（无）"))
            lines.append("--- 边 %s ---\n完整证据包：\n%s\n对方当前立场：\n%s\n%s"
                         % (edge_txt[key], peer[key][side],
                            opponent_view or "（首轮，无对方观点）",
                            _fmt_history(trace.get(key) or [], self.stage)))
        out = _debater_call_batch(ctx, role, prompt_key, "\n\n".join(lines),
                                  stance=self._stance(side))
        parsed = self._parse_batch_out(out, items) if out else {}
        got = 0
        for key, (ans, score, reason) in parsed.items():
            answers[key][side] = ans
            confs[key][side] = _clamp_score(score)
            reasons[key][side] = reason
            got += 1
        missed = [it for it in items if it[0] not in parsed]
        if not missed:
            return got
        if len(missed) == 1:
            from util.log import get
            get("debate").warning("辩手 %s 单条辩论失败（%s-%s），该侧该轮维持原立场",
                                  role, missed[0][0][0], missed[0][0][1])
            return got
        half = len(missed) // 2
        got += self._debater_batch_recursive(ctx, role, prompt_key, missed[:half],
                                             answers, confs, reasons, side, r,
                                             roles_rev, edge_txt, peer, trace)
        got += self._debater_batch_recursive(ctx, role, prompt_key, missed[half:],
                                             answers, confs, reasons, side, r,
                                             roles_rev, edge_txt, peer, trace)
        return got

    def _parse_batch_out(self, out: Dict[str, Any], items) -> Dict[Tuple[str, str], Any]:
        """批量 LLM 输出 {"edges": [...]} → {key: (answer, score, reason)}（v4.28）。

        按边标识匹配批内 key：存在性用无向归一化（正反同 key）；方向用有向归一化
        （LLM 写 v->u 也认，映射到同一 key）。解析失败/未知边 → 跳过（该边本轮不更新）。
        """
        canon: Dict[str, Tuple[str, str]] = {}
        for key, rec, _ in items:
            canon[norm_undirected_key("->".join(key))] = key
            if self.stage == "direction":
                canon[norm_edge_key("->".join(key))] = key
                canon[norm_edge_key("->".join(reversed(key)))] = key
        res = {}
        for it in (out.get("edges", []) or []):
            if not isinstance(it, dict):
                continue
            edge_s = str(it.get("edge", "") or "").strip()
            if not edge_s:
                continue
            normed = (norm_undirected_key(edge_s) if self.stage == "existence"
                      else norm_edge_key(edge_s))
            key = canon.get(normed)
            if key is None:
                continue
            # 反序命中（方向阶段）：LLM 以反序真名写边（如列表给 "A->B"，LLM 回 "B->A"）
            # Flip the baseline before parsing an answer produced for the reverse edge order.
            base = key
            if self.stage == "direction" and normed == norm_edge_key("->".join(reversed(key))):
                base = (key[1], key[0])
            ans = self.parse_fn(it, base)
            if ans is None:
                continue
            # 反序命中时答案相对反序基准：u->v <-> v->u 翻转回记录 key 基准（related 不变）
            if self.stage == "direction" and base != key:
                ans = {"u->v": "v->u", "v->u": "u->v"}.get(ans, ans)
            res[key] = (ans, it.get("score", 0.0), str(it.get("reason", "") or ""))
        return res


    def _run_rounds(self, ctx, key: Tuple[str, str], prompt_key: str,
                    packets: Dict[str, str], parse_answer,
                    answers: Dict[str, str], confs: Dict[str, float],
                    reasons: Dict[str, str]) -> int:
        """两辩手交替 N 轮：A/B 共享同一完整证据包（v4.36 不分包），
        后续轮次附对方立场 + 双方完整辩论历史（逐轮立场/置信/理由）。

        返回实际完成的轮数（模板缺失/调用失败 → 0，即无辩论）。
        """
        max_rounds = max(int(ctx.config.get("DEBATE_MAX_ROUNDS", 2) or 2), 1)
        # 存在性阶段无向渲染（v4.27 修：带箭头的 "u->v" 会让辩手把"方向反了"当成
        # "边不存在"，Phase1 只论存在性，方向留 Phase2）
        edge_txt = (" - ".join(key) if self.stage == "existence" else "->".join(key))
        roles = {"a": "debater_a", "b": "debater_b"}
        roles_rev = {"a": "b", "b": "a"}
        peer = {"a": packets[packets["_first"]], "b": packets[packets["_second"]]}
        from config.schema import DebateRound
        rounds = 0
        hist: List[Any] = []
        for r in range(max_rounds):
            got = 0
            for side in ("a", "b"):
                opponent_view = ""
                p_side = roles_rev[side]
                if r > 0 and answers.get(p_side):
                    opponent_view = ("对方立场=%s 置信=%.2f 理由：%s"
                                     % (answers[p_side], confs.get(p_side, 0.0),
                                        reasons.get(p_side, "") or "（无）"))
                out = _debater_call(ctx, roles[side], prompt_key, edge_txt,
                                    peer[side], opponent_view,
                                    stance=self._stance(side),
                                    history=_fmt_history(hist, self.stage))
                if out is None:
                    continue
                ans = parse_answer(out, key)   # v4.36 修：单边路径 parse_fn 需 key（既有漏传）
                if ans is None:
                    continue
                answers[side] = ans
                confs[side] = _clamp_score(out.get("score", 0.0))   # v4.21 修：裁剪到 [0,1]
                reasons[side] = str(out.get("reason", "") or "")
                got += 1
            if got == 0:
                break                      # 本轮两方都无有效输出 → 提前终止
            rounds += 1
            hist.append(DebateRound(
                round=rounds,
                a_answer=answers.get("a", ""), a_conf=round(float(confs.get("a", 0.0)), 4),
                a_reason=reasons.get("a", "") or "",
                b_answer=answers.get("b", ""), b_conf=round(float(confs.get("b", 0.0)), 4),
                b_reason=reasons.get("b", "") or ""))
        return rounds

    def _outcome(self, key: Tuple[str, str], stage: str, answers: Dict[str, str],
                 confs: Dict[str, float], reasons: Dict[str, str],
                 rounds: int, trace=None) -> Any:
        """汇总 DebateOutcome（from config.schema；无辩论 → 空结果）。"""
        from config.schema import DebateOutcome
        a, b = answers.get("a", ""), answers.get("b", "")
        return DebateOutcome(
            stage=stage, answer_a=a, answer_b=b,
            conf_a=round(float(confs.get("a", 0.0)), 4),
            conf_b=round(float(confs.get("b", 0.0)), 4),
            reason_a=reasons.get("a", "") or "", reason_b=reasons.get("b", "") or "",
            agreed=bool(a and b and a == b), rounds=rounds,
            trace=list(trace or []))


class Judge(ABC):
    """一次全员终裁（模板方法：全量候选边分批，批次内一次 LLM，子类实现批构造与解析）。"""

    name: str = ""
    stage: str = ""              # existence / direction

    def judge(self, ctx, keys: List[Tuple[str, str]]) -> None:
        """对全部候选边分批终裁（JUDGE_BATCH_EDGES 小批量，防注意力稀释）。

        批次覆盖不全（整批失败/个别缺失/解析失败）→ 仅对未覆盖子集拆半递归重试
        A single-edge failure retains the deterministic scorer's initial decision (applied=False).
        """
        from util.log import get
        log = get(f"judge.{self.stage}")
        batch = max(int(ctx.config.get("JUDGE_BATCH_EDGES", 20) or 20), 1)
        stats = {"calls": 0, "covered": 0, "failed": 0}
        for start in range(0, len(keys), batch):
            self._judge_recursive(ctx, keys[start:start + batch], log, stats)
        log.info("%s 裁判完成：覆盖=%d 失败=%d 边数=%d（批大小=%d，失败维持初判）",
                 self.name, stats["covered"], stats["failed"], len(keys), batch)

    def _judge_recursive(self, ctx, chunk: List[Tuple[str, str]], log, stats) -> None:
        """拆半递归：仅对未覆盖子集重试（子集内仍一次 LLM；单条仍失败才兜底）。"""
        if not chunk:
            return
        verdicts = self.judge_batch(ctx, chunk)
        missed: List[Tuple[str, str]] = []
        for key in chunk:
            rec = ctx.edge_records.get(key)
            if rec is None:
                continue
            v = verdicts.get(key)
            if v is not None:
                rec.verdict = v
                stats["covered"] += 1
            else:
                missed.append(key)
        if not missed:
            return
        if len(missed) == 1:
            stats["failed"] += 1
            rec = ctx.edge_records.get(missed[0])
            if rec is not None:
                rec.verdict = self._fallback_verdict(rec)
                self._set_decision(rec, rec.verdict.answer)   # 兜底同样落决策（v4.23）
            log.warning("裁判单条失败 %s（确定性统计兜底）", "->".join(missed[0]))
            return
        half = len(missed) // 2
        self._judge_recursive(ctx, missed[:half], log, stats)
        self._judge_recursive(ctx, missed[half:], log, stats)

    def finalize(self, ctx) -> None:
        """阶段产物钩子：Phase1=重建候选集；Phase2=确保全部边落决策。默认空操作。"""

    @abstractmethod
    def judge_batch(self, ctx, chunk: List[Tuple[str, str]]) -> Dict[Tuple[str, str], Any]:
        """对一批边一次 LLM 终裁，返回 {key: Verdict}；缺 key = 该边维持初判。"""
        raise NotImplementedError

    # ---------- 子类共享 ----------

    def _fallback_verdict(self, rec) -> Any:
        """失败兜底：确定性统计规则（v4.23 评分器已删；实验场景裁判失败即重跑，极少触发）。

        存在性：统计支持率 ≥0.5 → exists，否则 uncertain（保守不删边）；
        方向性：related（无向保留，不臆造方向）。
        """
        from config.schema import Verdict
        block = getattr(rec, "aggregate", None)
        if self.stage == "existence":
            ssr = float(getattr(block, "stat_support_rate", 0.0) or 0.0) if block is not None else 0.0
            ans = "exists" if ssr >= 0.5 else "uncertain"
        else:
            ans = "related"
        return Verdict(answer=ans, confidence=0.0, reason="", applied=False)

