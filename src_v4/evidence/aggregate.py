# -*- coding: utf-8 -*-
"""Server-side aggregation of client evidence packages.

每条候选边：把各客户端 ClientEvidence 汇总为 AggregateBlock（支持率/统计支持率/
被解释率/联邦 p 值/方向分裂度/异质性）。加聚合指标 = @register_aggregator 一个类。
"""
from typing import Any, Dict, Tuple, Type

import numpy as np
from scipy.stats import chi2, norm

from config.schema import AggregateBlock
from evidence.base import Registry, side_of

AGG_FIELD_DOC: Dict[str, Dict[str, str]] = {
    "support_rate": {"term": "支持率", "dim": "全局聚合", "meaning": "骨架含该边的客户端占比"},
    "stat_support_rate": {"term": "统计支持率", "dim": "全局聚合", "meaning": "边际独立性检验显著（p<α）的客户端占比"},
    "explained_away_rate": {"term": "被解释率", "dim": "全局聚合", "meaning": "条件化后关联不显著的客户端占比（离散=条件 G² 不显著；连续=偏相关不显著）。条件化后关联变弱有两种情况：关联可由条件变量解释（间接路径/共因）；或两端为确定性/机械决定关系（条件化后天然低熵）"},
    "cond_coverage_rate": {"term": "条件化覆盖率", "dim": "全局聚合", "meaning": "做过条件化检验的客户端占比"},
    "sign_consistency": {"term": "r 符号一致性", "dim": "全局聚合", "meaning": "客户端间边际 r 符号的多数占比（0.5=方向分裂，1=一致）"},
    "nmi_median": {"term": "NMI 中位数", "dim": "全局聚合", "meaning": "客户端间关联强度的中位数"},
    "nmi_iqr": {"term": "NMI 四分位距", "dim": "全局聚合", "meaning": "客户端间关联强度的离散度"},
    "fisher_p": {"term": "Fisher 合并 p 值", "dim": "全局聚合", "meaning": "多客户端边际 p 值的联邦合并"},
    "stouffer_z": {"term": "Stouffer z 分数", "dim": "全局聚合", "meaning": "多客户端边际 p 值的联邦合并（正值表示方向一致支持关联）"},
    # 方向性聚合字段（v4.32）
    "dir_split_uv": {"term": "方向分裂（uv 票）", "dim": "全局聚合", "meaning": "统计方向证据支持 u->v 的客户端票数"},
    "dir_split_vu": {"term": "方向分裂（vu 票）", "dim": "全局聚合", "meaning": "统计方向证据支持 v->u 的客户端票数"},
    "neutral_rate": {"term": "方向中立率", "dim": "全局聚合", "meaning": "完全没有方向信号（算法未定向+统计 0.5+LLM 无判断）的客户端占比"},
    "stat_dir_global": {"term": "统计方向全局均值", "dim": "全局聚合", "meaning": "非中性统计方向证据的均值（0.5=全局无统计方向信息，连续数据才有信号）"},
    "alg_arrow_uv": {"term": "算法方向（u->v 票）", "dim": "全局聚合", "meaning": "本地算法输出 u->v 的客户端数"},
    "alg_arrow_vu": {"term": "算法方向（v->u 票）", "dim": "全局聚合", "meaning": "本地算法输出 v->u 的客户端数"},
    "alg_undirected": {"term": "算法方向（未定票）", "dim": "全局聚合", "meaning": "本地算法出边但方向未定的客户端数"},
    "alg_absent": {"term": "算法方向（未出边票）", "dim": "全局聚合", "meaning": "本地算法未出此边的客户端数"},
    "llm_uv": {"term": "LLM 方向票（u->v）", "dim": "全局聚合", "meaning": "客户端本地 LLM 判断 u->v 的票数"},
    "llm_vu": {"term": "LLM 方向票（v->u）", "dim": "全局聚合", "meaning": "客户端本地 LLM 判断 v->u 的票数"},
    "dir_hetero": {"term": "跨客户端异质性", "dim": "全局聚合", "meaning": "客户端间统计方向证据的方差"},
    "v_struct_rate": {"term": "v 结构支持率", "dim": "全局聚合", "meaning": "检测到对撞信号（Δ>0）的客户端占比"},
}


def agg_field_doc_text() -> str:
    """裁判聚合字段解释文本（prompt 渲染用：专业术语 / 判断维度 / 用处）。"""
    lines = []
    for name, doc in AGG_FIELD_DOC.items():
        lines.append("- %s【%s/维度=%s】：%s" % (name, doc["term"], doc["dim"], doc["meaning"]))
    return "\n".join(lines)


aggregator_registry = Registry()


def register_aggregator(cls: Type) -> Type:
    aggregator_registry.register(cls)
    return cls


def _denominator(ctx, client_evidence: Dict[str, Any]) -> int:
    """支持率类分母：全体客户端（ctx 可用时），否则按已有证据数（v4.21 修）。"""
    if ctx is not None and getattr(ctx, "clients", None):
        return max(len(ctx.clients), 1)
    return max(len(client_evidence), 1)


def _p_value(ev) -> float:
    """单个客户端该边的边际独立性 p 值（离散 g2_p / 连续 pearson_p；缺失 → 1.0 中立）。"""
    if ev is None:
        return 1.0
    p = getattr(ev, "pearson_p", None)
    if p is None:
        p = getattr(ev, "g2_p", None)
    if p is None:
        return 1.0
    try:
        return min(max(float(p), 1e-12), 1.0)
    except (TypeError, ValueError):
        return 1.0


def _llm_ans(ce) -> str:
    """该客户端 LLM 方向答案（u->v / v->u / related / 空=无判断）。"""
    llm = getattr(ce, "llm", None)
    if llm is None:
        return ""
    return str(getattr(llm, "answer", "") or "").strip().lower()


def _has_dir_signal(ce) -> bool:
    """该客户端是否有任一方向信号（与 side_of/辩论包同口径，v4.25 统一）。"""
    de = getattr(ce, "direction", None)
    if de is None:
        return False
    return bool(side_of(str(getattr(de, "alg_direction", "") or ""),
                        float(0.5 if (sd := getattr(de, "stat_dir", 0.5)) is None else sd), _llm_ans(ce)))


class Aggregator:
    """单个全局聚合指标：输入全客户端证据（+ctx 读 config），输出该指标值。"""

    name: str = ""

    def aggregate(self, client_evidence: Dict[str, Any], ctx=None) -> Any:
        raise NotImplementedError


@register_aggregator
class SupportRate(Aggregator):
    """骨架含该边的客户端占比（存在性主信号）。"""
    name = "support_rate"

    def aggregate(self, client_evidence, ctx=None):
        total = _denominator(ctx, client_evidence)
        has = sum(1 for ce in client_evidence.values()
                  if getattr(getattr(ce, "existence", None), "skeleton_has", False))
        return round(has / total, 4)


@register_aggregator
class StatSupportRate(Aggregator):
    """边际 p<α 的客户端占比（纯数据，不依赖骨架）。"""
    name = "stat_support_rate"

    def aggregate(self, client_evidence, ctx=None):
        total = _denominator(ctx, client_evidence)
        alpha = float((ctx.config if ctx is not None else {}).get("ALPHA", 0.05) or 0.05)
        strong = sum(1 for ce in client_evidence.values()
                     if _p_value(getattr(ce, "existence", None)) < alpha)
        return round(strong / total, 4)


@register_aggregator
class ExplainedAwayRate(Aggregator):
    """被解释率：条件化后不显著（p_cond>α）的客户端占比（假阳性指纹）。"""
    name = "explained_away_rate"

    def aggregate(self, client_evidence, ctx=None):
        total = _denominator(ctx, client_evidence)
        away = sum(1 for ce in client_evidence.values()
                   if bool(getattr(getattr(ce, "existence", None), "explained_away", False)))
        return round(away / total, 4)


@register_aggregator
class FisherP(Aggregator):
    """联邦 p 值合并（Fisher，客户端数据不相交→检验独立→合并成立，v4.5 方法贡献）。"""
    name = "fisher_p"

    def aggregate(self, client_evidence, ctx=None):
        ps = [_p_value(getattr(ce, "existence", None)) for ce in client_evidence.values()]
        ps = [min(max(p, 1e-12), 1.0) for p in ps]
        if not ps:
            return 1.0
        chi2_stat = -2.0 * sum(np.log(p) for p in ps)
        return round(float(chi2.sf(chi2_stat, 2 * len(ps))), 6)


@register_aggregator
class StoufferZ(Aggregator):
    """联邦 p 值合并（Stouffer z 分数，正=方向一致支持存在）。"""
    name = "stouffer_z"

    def aggregate(self, client_evidence, ctx=None):
        ps = [_p_value(getattr(ce, "existence", None)) for ce in client_evidence.values()]
        if not ps:
            return 0.0
        # p 上限截到 1-1e-12：p=1.0 时 1-p=0 → norm.ppf(0)=-inf 会污染审计 JSON（v4.25 修）
        zs = [norm.ppf(1.0 - min(max(p, 1e-12), 1.0 - 1e-12)) for p in ps]
        return round(float(sum(zs) / np.sqrt(len(zs))), 4)


@register_aggregator
class DirectionSplit(Aggregator):
    """方向分裂：uv / vu 票数（v4.25 统一口径=行级三信号多数，与辩论包/中立率一致；
    评分器已删，不再有"统计层被语义污染"问题，票型即证据包归属）。"""
    name = "direction_split"

    def aggregate(self, client_evidence, ctx=None):
        uv = vu = 0
        for ce in client_evidence.values():
            de = getattr(ce, "direction", None)
            if de is None:
                continue
            side = side_of(str(getattr(de, "alg_direction", "") or ""),
                           float(0.5 if (sd := getattr(de, "stat_dir", 0.5)) is None else sd), _llm_ans(ce))
            if side == "uv":
                uv += 1
            elif side == "vu":
                vu += 1
        return uv, vu


@register_aggregator
class DirectionHetero(Aggregator):
    """跨客户端异质性（统计量方差，差异大→全局置信度下调，v4.5 异质性抓手）。"""
    name = "direction_hetero"

    def aggregate(self, client_evidence, ctx=None):
        vals = []
        for ce in client_evidence.values():
            de = getattr(ce, "direction", None)
            if de is not None and getattr(de, "stat_dir", None) is not None:
                vals.append(float(0.5 if (sd := getattr(de, "stat_dir", 0.5)) is None else sd))
        if not vals:
            return 0.0
        return round(float(np.std(vals)), 4)


@register_aggregator
class DirectionNeutralRate(Aggregator):
    """方向中立率：无方向信号（算法未定向+统计0.5+LLM无/related）客户端占比（v4.23）。"""
    name = "neutral_rate"

    def aggregate(self, client_evidence, ctx=None):
        total = _denominator(ctx, client_evidence)
        neutral = sum(1 for ce in client_evidence.values() if not _has_dir_signal(ce))
        return round(neutral / total, 4)


@register_aggregator
class StatDirGlobal(Aggregator):
    """统计方向聚合强度：非中性 stat_dir 的均值（连续 ANM 残差；0.5=全局无统计方向信息）。"""
    name = "stat_dir_global"

    def aggregate(self, client_evidence, ctx=None):
        vals = []
        for ce in client_evidence.values():
            de = getattr(ce, "direction", None)
            if de is None:
                continue
            sd = float(0.5 if (sd := getattr(de, "stat_dir", 0.5)) is None else sd)
            if sd != 0.5:
                vals.append(sd)
        if not vals:
            return 0.5
        return round(float(np.mean(vals)), 4)


@register_aggregator
class AlgDirProfile(Aggregator):
    """算法方向画像：-> / <- / undirected（出边未定向）/ 未出边 的客户端数（v4.23）。"""
    name = "alg_dir_profile"

    def aggregate(self, client_evidence, ctx=None):
        uv = vu = und = absent = 0
        for ce in client_evidence.values():
            de = getattr(ce, "direction", None)
            alg = str(getattr(de, "alg_direction", "") or "") if de is not None else ""
            if alg == "->":
                uv += 1
            elif alg == "<-":
                vu += 1
            elif alg == "undirected":
                und += 1
            else:
                absent += 1
        return uv, vu, und, absent


@register_aggregator
class LlmDirSplit(Aggregator):
    """LLM 方向票：u->v / v->u（related 与无判断不计；v4.23，独立于统计方向分裂）。"""
    name = "llm_dir_split"

    def aggregate(self, client_evidence, ctx=None):
        uv = vu = 0
        for ce in client_evidence.values():
            ans = _llm_ans(ce)
            if ans == "u->v":
                uv += 1
            elif ans == "v->u":
                vu += 1
        return uv, vu


@register_aggregator
class VStructRate(Aggregator):
    """v 结构支持率：检测到对撞信号（collider 非空，Δ>0）的客户端占比（v4.32）。"""
    name = "v_struct_rate"

    def aggregate(self, client_evidence, ctx=None):
        total = _denominator(ctx, client_evidence)
        hit = sum(1 for ce in client_evidence.values()
                  if bool(getattr(getattr(ce, "direction", None), "collider", None)))
        return round(hit / total, 4)


@register_aggregator
class CondCoverageRate(Aggregator):
    """条件化覆盖率：做了条件化检验（cond 非空）的客户端占比（v4.32）。

    防"无条件化证据"被误读为"条件化后解释不掉"（explained_away 仅在有 cond 时有效）。
    """
    name = "cond_coverage_rate"

    def aggregate(self, client_evidence, ctx=None):
        total = _denominator(ctx, client_evidence)
        done = sum(1 for ce in client_evidence.values()
                   if bool(getattr(getattr(ce, "existence", None), "cond", None)))
        return round(done / total, 4)


@register_aggregator
class SignConsistency(Aggregator):
    """r 符号一致性：连续数据边际 r 符号多数占比（0.5=完全分裂，1=一致；离散=None）。"""
    name = "sign_consistency"

    def aggregate(self, client_evidence, ctx=None):
        signs = []
        for ce in client_evidence.values():
            ev = getattr(ce, "existence", None)
            r = getattr(ev, "pearson_r", None)
            if ev is None or r is None:
                continue
            if float(r) > 0:
                signs.append(1.0)
            elif float(r) < 0:
                signs.append(0.0)
        if not signs:
            return None
        pos = sum(signs)
        return round(max(pos, len(signs) - pos) / len(signs), 4)


@register_aggregator
class NmiSpread(Aggregator):
    """客户端 NMI 分布：中位数与 IQR（v4.32，客户端间强度一致性）。"""
    name = "nmi_spread"

    def aggregate(self, client_evidence, ctx=None):
        vals = []
        for ce in client_evidence.values():
            nmi = getattr(getattr(ce, "existence", None), "nmi", None)
            if nmi is not None:
                vals.append(float(nmi))
        if not vals:
            return None
        a = np.array(sorted(vals))
        med = float(np.median(a))
        iqr = float(np.percentile(a, 75) - np.percentile(a, 25))
        return round(med, 4), round(iqr, 4)


def build_aggregate_block(ctx, key: Tuple[str, str]) -> AggregateBlock:
    """对一条边：跑全部注册聚合器 → AggregateBlock。"""
    block = AggregateBlock()
    client_evidence = ctx.edge_records[key].client_evidence
    has_dir = any(getattr(ce, "direction", None) is not None
                  for ce in client_evidence.values())
    _DIR_ONLY = {"direction_split", "direction_hetero", "neutral_rate",
                 "stat_dir_global", "alg_dir_profile", "llm_dir_split"}
    for name in aggregator_registry.names():
        if name in _DIR_ONLY and not has_dir:
            continue                        # Phase1 无方向证据：方向字段保持默认（不伪造计数）
        value = aggregator_registry.get(name)().aggregate(client_evidence, ctx)
        if name == "direction_split" and isinstance(value, tuple):
            block.dir_split_uv, block.dir_split_vu = int(value[0]), int(value[1])
        elif name == "alg_dir_profile" and isinstance(value, tuple):
            block.alg_arrow_uv, block.alg_arrow_vu, block.alg_undirected, block.alg_absent = \
                (int(x) for x in value)
        elif name == "llm_dir_split" and isinstance(value, tuple):
            block.llm_uv, block.llm_vu = int(value[0]), int(value[1])
        elif name == "nmi_spread" and isinstance(value, tuple):
            block.nmi_median, block.nmi_iqr = float(value[0]), float(value[1])
        elif hasattr(block, name):
            setattr(block, name, value)
    return block


def rebuild_aggregate_blocks(ctx, keys=None) -> None:
    """对指定边（默认全部候选边）重建聚合块（补采回写后刷新，v4.21 修）。"""
    if keys is None:
        keys = list(ctx.edge_records)
    for key in keys:
        rec = ctx.edge_records.get(key)
        if rec is not None:
            rec.aggregate = build_aggregate_block(ctx, key)
