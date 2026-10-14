# -*- coding: utf-8 -*-
"""Server-side adjudication stage.

Every candidate receives a final judgment without score pre-filtering.
编排：裁判对**所有**候选边分批小批量终裁（JUDGE_BATCH_EDGES，防注意力稀释）
→ finalize 阶段产物。评分器已删（v4.23）：初判/置信改由裁判输出承担，
失败兜底为确定性统计规则（实验场景裁判失败即重跑，极少触发）。
- Phase1（existence）：裁判逐边 exists/none/uncertain → 新候选集（drop 剔除进 dropped_edges 审计）。
- Phase2（direction）：裁判逐边 u->v/v->u/related → 定边图（related 无向保留）。
"""
from protocol.base import Stage
from protocol.context import ProtocolContext
from protocol.pipeline import register_stage


@register_stage
class JudgeStage(Stage):
    name = "judge"

    def run(self, ctx: ProtocolContext) -> None:
        stage = ctx.current_phase
        from debate.base import judges
        judge_cls = judges.get(f"{stage}_judge")
        if judge_cls is None:
            raise KeyError(f"裁判 {stage}_judge 未注册（已注册: {sorted(judges)}）")

        keys = list(ctx.edge_records)                # 全部候选边过裁判（v4：不筛）
        judge = judge_cls()
        judge.judge(ctx, keys)                       # 分批小批量 LLM 终裁
        judge.finalize(ctx)                          # 阶段产物（候选集/定边汇总）
        self._log_evidence(ctx, stage)               # Per-edge aggregate, debate, and adjudication audit.

    # ---------------- v4.29 逐边证据日志（供算法效果分析，不改变判定） ----------------

    def _log_evidence(self, ctx: ProtocolContext, stage: str) -> None:
        """每个裁判阶段结束后，对每条边（含被裁边）打印一行聚合+辩论+裁判、一行证据矩阵。"""
        from util.log import get
        log = get("judge." + stage)
        recs = dict(ctx.edge_records)
        recs.update(ctx.dropped_records)             # 存在性被裁边（finalize 已存档）
        for key, rec in sorted(recs.items(), key=lambda kv: kv[0]):
            line1, line2 = _fmt_edge_log(key, rec, stage)
            log.info("%s", line1)
            log.info("%s", line2)


def _fmt_edge_log(key, rec, stage):
    """格式化单边证据日志：行1=聚合+辩论+裁判；行2=客户端证据矩阵。"""
    u, v = key
    agg = getattr(rec, "aggregate", None)
    db = getattr(rec, "debate", None)
    vd = getattr(rec, "verdict", None)
    ce = getattr(rec, "client_evidence", {}) or {}

    def ga(name, dflt=0.0):
        return getattr(agg, name, dflt) if agg is not None else dflt

    if stage == "existence":
        agg_s = ("support=%.2f stat_sup=%.2f expl=%.2f fisher=%.3g z=%.2f"
                 % (ga("support_rate"), ga("stat_support_rate"),
                    ga("explained_away_rate"), ga("fisher_p", 1.0), ga("stouffer_z")))
        cells = []
        for cid, c in sorted(ce.items()):
            ex = getattr(c, "existence", None)
            ll = getattr(c, "llm_existence", None)
            if ex is None and ll is None:
                cells.append("%s:∅" % cid)
                continue
            sk = "✓" if getattr(ex, "skeleton_has", False) else "✗"
            p = getattr(ex, "pearson_p", None)
            if p is None:
                p = getattr(ex, "g2_p", None)
            ps = "%.3g" % p if p is not None else "-"
            ans = getattr(ll, "answer", "") if ll is not None else ""
            sc = getattr(ll, "score", 0.0) if ll is not None else 0.0
            cells.append("%s:skel%s g2=%s LLM=%s(%.2f)" % (cid, sk, ps, ans or "-", sc))
    else:
        agg_s = ("dir_split_uv=%d vu=%d alg(->=%d,<-=%d,und=%d,abs=%d) llm(uv=%d,vu=%d) neutral=%.2f dirg=%.2f"
                 % (ga("dir_split_uv"), ga("dir_split_vu"),
                    ga("alg_arrow_uv"), ga("alg_arrow_vu"),
                    ga("alg_undirected"), ga("alg_absent"),
                    ga("llm_uv"), ga("llm_vu"),
                    ga("neutral_rate"), ga("stat_dir_global", 0.5)))
        cells = []
        for cid, c in sorted(ce.items()):
            de = getattr(c, "direction", None)
            ll = getattr(c, "llm", None)
            if de is None and ll is None:
                cells.append("%s:∅" % cid)
                continue
            ad = getattr(de, "alg_direction", "") or ""
            sd = getattr(de, "stat_dir", 0.5)
            ans = getattr(ll, "answer", "") if ll is not None else ""
            sc = getattr(ll, "score", 0.0) if ll is not None else 0.0
            cells.append("%s:alg=%s stat=%.2f LLM=%s(%.2f)" % (cid, ad or "-", sd, ans or "-", sc))

    ba = getattr(db, "answer_a", "") if db is not None else ""
    bb = getattr(db, "answer_b", "") if db is not None else ""
    ca = getattr(db, "conf_a", 0.0) if db is not None else 0.0
    cb = getattr(db, "conf_b", 0.0) if db is not None else 0.0
    va = getattr(vd, "answer", "") if vd is not None else ""
    vc = getattr(vd, "confidence", 0.0) if vd is not None else 0.0
    vr = str(getattr(vd, "reason", "") or "") if vd is not None else ""
    line1 = ("[EDGE] %s-%s | %s | 辩论: A=%s(%.2f) B=%s(%.2f) | 裁判: %s conf=%.2f 理由=%s"
             % (u, v, agg_s, ba or "-", ca, bb or "-", cb, va or "-", vc, vr))
    line2 = "[MATRIX] %s-%s | %s" % (u, v, " || ".join(cells) if cells else "（无客户端证据）")
    return line1, line2
