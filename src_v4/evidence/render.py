# -*- coding: utf-8 -*-
"""Render structured evidence for the protocol prompts.

把证据 dataclass/聚合块渲染成 prompt 文本块（聚合值，非原始数据）。
client（本地分析统计块）与 debate/packets（服务端证据包）共用，避免 client 跨层依赖 debate。
"""
from typing import Any, Dict


def fmt_stats(d: Dict[str, Any], stats: bool = True) -> str:
    """单个证据块的统计字段 → 文本（聚合值，非原始数据）。

    连续/离散两套字段分流（v4.30 对齐 v3）：连续数据只渲染 r/边际p/偏相关，
    不渲染离散占位字段（state/cell_min/g2），避免 "MI=0/每格min=0" 误导 LLM。
    """
    if not d:
        return "（无证据）"
    if not stats:
        # With statistical evidence disabled, retain only skeleton structure.
        out = []
        if d.get("skeleton_has"):
            out.append("骨架含边")
        _pl = int(d.get("skel_path_len", -1) or -1)
        if "skel_path_len" in d:
            if _pl == 1:
                out.append("骨架路径=相邻")
            elif _pl > 1:
                _mid = "、".join(d.get("mid_nodes") or [])
                out.append("骨架路径=%d跳(经%s)" % (_pl, _mid))
            elif _pl == -1:
                out.append("骨架路径=不连通")
        return "；".join(out) or "（无统计数值，仅骨架信息）"
    out = []
    continuous = d.get("pearson_p") is not None
    if d.get("skeleton_has"):
        out.append("骨架含边")
    if "n" in d:
        out.append("n=%d" % int(d.get("n", 0)))
    if continuous:
        if "pearson_r" in d:
            out.append("r=%.3f" % float(d.get("pearson_r", 0.0)))
        out.append("边际p=%.3f" % float(d.get("pearson_p", 1.0)))
        if d.get("spearman_r") is not None:
            out.append("rho=%.3f p=%.3f" % (float(d.get("spearman_r", 0.0)),
                                             float(d.get("spearman_p", 1.0))))
        if d.get("ratio_abs_sp_pearson") is not None:
            out.append("|rho|/|r|=%.2f" % float(d.get("ratio_abs_sp_pearson", 0.0)))
        if d.get("partial_all_r") is not None:
            out.append("半偏=%.3f p=%.3f" % (float(d.get("partial_all_r", 0.0)),
                                             float(d.get("partial_all_p", 1.0))))
    else:
        if "state_u" in d or "state_v" in d:
            out.append("状态(%s,%s)" % (d.get("state_u", "?"), d.get("state_v", "?")))
        if "cell_min" in d:
            out.append("每格min=%.0f" % float(d.get("cell_min", 0.0)))
        if d.get("zero_cell_ratio") is not None:
            out.append("零格=%.2f" % float(d.get("zero_cell_ratio", 0.0)))
        if "mi" in d:
            out.append("MI=%.3f" % float(d.get("mi", 0.0)))
        if "nmi" in d:
            out.append("NMI=%.3f" % float(d.get("nmi", 0.0)))
        p = d.get("g2_p")
        if p is not None:
            out.append("边际p=%.3f" % float(p))
    for cd in (d.get("cond") or []):
        # v4.31 修：p=0.0 是合法强显著值，`or 1.0` 会把 0.0 当 falsy 反转成 1.0
        _p = cd.get("p") if cd.get("p") is not None else 1.0
        _d = cd.get("delta")
        # 标签与 explained_away 口径一致（statistics.py：p>alpha 且 delta<0 才是"被解释"；
        # delta>=0 是条件化后关联增强 → v 结构特征，不能标"被解释"）
        if float(_p) >= 0.05 and (_d is None or float(_d) < 0):
            tag = "关联被解释"
        elif float(_p) < 0.05:
            tag = "仍显著"
        else:
            tag = "条件化后增强"
        pr = cd.get("partial_r")
        if continuous and pr is not None:
            out.append("偏相关(%s)=%.3f p=%.3f(%s)" % (cd.get("w", "?"),
                        float(pr), float(_p), tag))
        else:
            out.append("cond(%s)p=%.3f(%s)" % (cd.get("w", "?"), float(_p), tag))
    if d.get("h_u_given_v") is not None:
        out.append("H(U|V)=%.3f H(V|U)=%.3f 熵比=%.3f maxP=%.3f"
                   % (float(d.get("h_u_given_v", 0.0)), float(d.get("h_v_given_u", 0.0)),
                      float(d.get("ent_ratio", 1.0) or 1.0),
                      float(d.get("max_cond_prob", 0.0) or 0.0)))
    if d.get("explained_away"):
        out.append("被解释")
    # 结构上下文（v4.32）：骨架路径/中间节点/条件化覆盖（无向，不含箭头）
    _pl = int(d.get("skel_path_len", -1) or -1)
    if "skel_path_len" in d:
        if _pl == 1:
            out.append("骨架路径=相邻")
        elif _pl > 1:
            _mid = "、".join(d.get("mid_nodes") or [])
            out.append("骨架路径=%d跳(经%s)" % (_pl, _mid))
        elif _pl == -1:
            out.append("骨架路径=不连通")
    _cn = int(d.get("cond_cand_n", 0) or 0)
    if _cn > 0:
        # Only the first max_cond candidates are tested, so report that limit as the denominator.
        # 候选多于上限时标注"（候选N）"，避免 LLM 把未检验候选误读为证据不完整。
        _ck = int(d.get("cond_checked_n", len(d.get("cond") or [])) or 0)
        _mx = int(d.get("cond_max_cond", 0) or 0)
        _den = _mx if _mx > 0 else _cn
        if _cn > _den:
            out.append("条件化=%d/%d（候选%d）" % (_ck, _den, _cn))
        else:
            out.append("条件化=%d/%d" % (_ck, _cn))
    elif "cond_cand_n" in d:
        out.append("条件化=0/0")
    return "；".join(out) or "（无统计）"
def fmt_llm(llm: Any) -> str:
    if llm is None:
        return ""
    ans = getattr(llm, "answer", "") or ""
    if not ans:
        return ""
    return "LLM判断=%s 置信=%.2f 理由：%s" % (
        ans, float(getattr(llm, "score", 0.0) or 0.0),
        str(getattr(llm, "reason", "") or "") or "（无）")


def fmt_aggregate(block: Any, stage: str = "", stats: bool = True) -> str:
    """Render only the evidence fields relevant to the current protocol stage."""
    if block is None:
        return "（无全局聚合块）"
    if not stats:
        # The no-stat view retains skeleton support and algorithm/LLM direction aggregation.
        if stage == "direction":
            return ("全局：算法方向(->%d,<-%d,未定%d,未出%d) LLM方向(uv=%d,vu=%d)"
                    % (int(getattr(block, "alg_arrow_uv", 0) or 0),
                       int(getattr(block, "alg_arrow_vu", 0) or 0),
                       int(getattr(block, "alg_undirected", 0) or 0),
                       int(getattr(block, "alg_absent", 0) or 0),
                       int(getattr(block, "llm_uv", 0) or 0),
                       int(getattr(block, "llm_vu", 0) or 0)))
        return "全局：支持率=%.2f" % float(getattr(block, "support_rate", 0.0) or 0.0)
    # v4.31 修：fisher_p/stat_dir_global 的 0.0 是合法值，不能用 `or default`（0.0 被当 falsy 反转）
    _fp = getattr(block, "fisher_p", 1.0)
    _fp = 1.0 if _fp is None else _fp
    _sd = getattr(block, "stat_dir_global", 0.5)
    _sd = 0.5 if _sd is None else _sd
    if stage == "existence":
        _sc = getattr(block, "sign_consistency", None)
        _nm = getattr(block, "nmi_median", None)
        _ni = getattr(block, "nmi_iqr", None)
        parts = ["全局：支持率=%.2f 统计支持率=%.2f 被解释率=%.2f Fisher_p=%.3f Stouffer_z=%.2f"
                 % (float(getattr(block, "support_rate", 0.0) or 0.0),
                    float(getattr(block, "stat_support_rate", 0.0) or 0.0),
                    float(getattr(block, "explained_away_rate", 0.0) or 0.0),
                    float(_fp),
                    float(getattr(block, "stouffer_z", 0.0) or 0.0)),
                 "条件化覆盖率=%.2f" % float(getattr(block, "cond_coverage_rate", 0.0) or 0.0)]
        if _sc is not None:
            parts.append("r符号一致=%.2f" % float(_sc))
        if _nm is not None:
            parts.append("NMI中位=%.3f IQR=%.3f" % (float(_nm), float(_ni or 0.0)))
        return " ".join(parts)
    if stage == "direction":
        return ("全局：方向分裂(uv=%d,vu=%d) 方向中立率=%.2f 统计方向=%.2f 算法方向(->%d,<-%d,未定%d,未出%d) "
                "LLM方向(uv=%d,vu=%d) 异质=%.3f v结构支持率=%.2f"
                % (int(getattr(block, "dir_split_uv", 0) or 0),
                   int(getattr(block, "dir_split_vu", 0) or 0),
                   float(getattr(block, "neutral_rate", 0.0) or 0.0),
                   float(_sd),
                   int(getattr(block, "alg_arrow_uv", 0) or 0),
                   int(getattr(block, "alg_arrow_vu", 0) or 0),
                   int(getattr(block, "alg_undirected", 0) or 0),
                   int(getattr(block, "alg_absent", 0) or 0),
                   int(getattr(block, "llm_uv", 0) or 0),
                   int(getattr(block, "llm_vu", 0) or 0),
                   float(getattr(block, "dir_hetero", 0.0) or 0.0),
                   float(getattr(block, "v_struct_rate", 0.0) or 0.0)))
    # 默认全量渲染（无 stage，兼容旧调用）
    return ("全局：支持率=%.2f 统计支持率=%.2f 被解释率=%.2f Fisher_p=%.3f 方向分裂(uv=%d,vu=%d) "
            "方向中立率=%.2f 统计方向=%.2f 算法方向(->%d,<-%d,未定%d,未出%d) LLM方向(uv=%d,vu=%d) 异质=%.3f"
            % (float(getattr(block, "support_rate", 0.0) or 0.0),
               float(getattr(block, "stat_support_rate", 0.0) or 0.0),
               float(getattr(block, "explained_away_rate", 0.0) or 0.0),
               float(_fp),
               int(getattr(block, "dir_split_uv", 0) or 0),
               int(getattr(block, "dir_split_vu", 0) or 0),
               float(getattr(block, "neutral_rate", 0.0) or 0.0),
               float(_sd),
               int(getattr(block, "alg_arrow_uv", 0) or 0),
               int(getattr(block, "alg_arrow_vu", 0) or 0),
               int(getattr(block, "alg_undirected", 0) or 0),
               int(getattr(block, "alg_absent", 0) or 0),
               int(getattr(block, "llm_uv", 0) or 0),
               int(getattr(block, "llm_vu", 0) or 0),
               float(getattr(block, "dir_hetero", 0.0) or 0.0)))
_ALG_TXT = {
    "->": "本地算法输出 u->v",
    "<-": "本地算法输出 v->u",
    "undirected": "本地算法输出：边存在但方向未定",
    "": "本地算法输出：无（该客户端未出此边）",
}


def fmt_direction_stats(d: Dict[str, Any], stats: bool = True) -> str:
    """Render identifiability, v-structure, residual, and structural context."""
    if not stats:
        # The no-stat view omits statistical direction, v-structure, and residual evidence.
        out = [_ALG_TXT.get(str(d.get("alg_direction", "") or ""), _ALG_TXT[""])]
        _pl = int(d.get("skel_path_len", -1) or -1)
        if "skel_path_len" in d:
            if _pl == 1:
                out.append("骨架路径=相邻")
            elif _pl > 1:
                _mid = "、".join(d.get("mid_nodes") or [])
                out.append("骨架路径=%d跳(经%s)" % (_pl, _mid))
            elif _pl == -1:
                out.append("骨架路径=不连通")
        return "；".join(out)
    out = [_ALG_TXT.get(str(d.get("alg_direction", "") or ""), _ALG_TXT[""])]
    if d.get("dir_identifiable"):
        out.append("统计方向=%.2f" % float(d.get("stat_dir", 0.5) or 0.5))
    else:
        out.append("统计方向=不可辨识（离散双变量，仅结构线索可辨识）")
    for cd in (d.get("collider") or []):
        out.append("v结构(%s)Δ=%.3f" % (cd.get("w", "?"), float(cd.get("delta", 0.0) or 0.0)))
    _cn = int(d.get("cond_cand_n", 0) or 0)
    if "cond_cand_n" in d:
        # Report max_cond as the denominator to match the tested conditioning set.
        _ck = int(d.get("cond_checked_n", 0) or 0)
        _mx = int(d.get("cond_max_cond", 0) or 0)
        _den = _mx if _mx > 0 else _cn
        if _cn > _den:
            out.append("v结构覆盖=%d/%d（候选%d）" % (_ck, _den, _cn))
        else:
            out.append("v结构覆盖=%d/%d" % (_ck, _cn))
    if d.get("resid_u_to_v") is not None and d.get("resid_v_to_u") is not None:
        out.append("残差(u->v)=%.3f 残差(v->u)=%.3f" % (d["resid_u_to_v"], d["resid_v_to_u"]))
    _pl = int(d.get("skel_path_len", -1) or -1)
    if "skel_path_len" in d:
        if _pl == 1:
            out.append("骨架路径=相邻")
        elif _pl > 1:
            _mid = "、".join(d.get("mid_nodes") or [])
            out.append("骨架路径=%d跳(经%s)" % (_pl, _mid))
        elif _pl == -1:
            out.append("骨架路径=不连通")
    return "；".join(out)
