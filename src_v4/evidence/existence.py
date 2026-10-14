# -*- coding: utf-8 -*-
"""Edge-existence evidence provider.

组合 stat_registry 中 kind=existence 的统计量（MI/NMI/边际 p/条件 p/格密度）→ ExistenceEvidence 字段 dict。
options 由调用方（client）传入：neighbors（本地骨架邻居集）/ state_map / max_cond / alpha。
"""
from typing import Any, Dict, Tuple

from evidence.base import EvidenceProvider, EvidenceType, register_provider, stat_registry


FIELD_DOC: Dict[str, Dict[str, str]] = {
    # 关联强度
    "mi": {"term": "互信息", "dim": "关联强度", "meaning": "衡量两端变量关联强度（非线性关联也有效）"},
    "nmi": {"term": "归一化互信息", "dim": "关联强度", "meaning": "按两端熵归一化的互信息，可跨边比较"},
    "g2_p": {"term": "G² 边际独立性 p 值", "dim": "关联强度", "meaning": "离散口径边际独立性检验 p 值"},
    "pearson_r": {"term": "Pearson 相关系数", "dim": "关联强度", "meaning": "线性关联的方向与强度"},
    "pearson_p": {"term": "Pearson 显著性 p 值", "dim": "关联强度", "meaning": "线性边际独立性检验 p 值"},
    "spearman_r": {"term": "Spearman 秩相关", "dim": "关联强度", "meaning": "单调关联强度（含非线性单调），与 r 对比可观察非线性/离群影响"},
    "spearman_p": {"term": "Spearman 显著性 p 值", "dim": "关联强度", "meaning": "秩相关独立性检验 p 值"},
    "ratio_abs_sp_pearson": {"term": "秩相关/线性相关绝对值比", "dim": "关联强度", "meaning": "提示关联的线性程度"},
    "zero_cell_ratio": {"term": "零格占比", "dim": "关联强度", "meaning": "列联表空格比例，提示检验与 MI 估计的可信度"},
    # 直接性
    "cond": {"term": "条件化检验（条件 G²/偏相关）", "dim": "直接性", "meaning": "以候选节点为条件后关联是否仍存在；候选节点为两端在本地骨架上的邻居并集（含共同邻居与两跳中介）"},
    "delta": {"term": "条件化互信息变化 ΔMI", "dim": "直接性", "meaning": "条件化前后互信息的变化（减小=条件化后关联减弱；增大=条件化后关联增强）"},
    "cond_coverage": {"term": "条件化覆盖度", "dim": "直接性", "meaning": "实际条件化的候选数/可用候选数；0/0 表示没有可条件化节点"},
    "partial_all_r": {"term": "半偏相关", "dim": "直接性", "meaning": "控制其余全部变量后的偏相关，衡量关联在其他变量条件下的残留"},
    "partial_all_p": {"term": "半偏相关 p 值", "dim": "直接性", "meaning": "半偏相关显著性检验 p 值"},
    "explained_away": {"term": "被解释标记", "dim": "直接性", "meaning": "条件化后关联不显著（p>α 且 Δ<0）的标记"},
    # 确定性
    "h_u_given_v": {"term": "条件熵 H(U|V)", "dim": "确定性", "meaning": "给定 V 时 U 的剩余不确定性"},
    "h_v_given_u": {"term": "条件熵 H(V|U)", "dim": "确定性", "meaning": "给定 U 时 V 的剩余不确定性"},
    "ent_ratio": {"term": "熵比率", "dim": "确定性", "meaning": "条件熵/边际熵的最小比值，衡量一端对另一端的确定程度"},
    "max_cond_prob": {"term": "最大条件概率", "dim": "确定性", "meaning": "max P(V|U)，接近 1 表示一端取值几乎确定另一端"},
    # 结构上下文
    "skel_path_len": {"term": "骨架最短路径长度", "dim": "结构上下文", "meaning": "两端在本地骨架上的最短路径边数（1=相邻，>1=隔着中间节点）"},
    "mid_nodes": {"term": "路径中间节点", "dim": "结构上下文", "meaning": "骨架最短路径途经的中间节点名"},
    "n_common": {"term": "共同邻居数", "dim": "结构上下文", "meaning": "两端共同邻居的数量"},
    "n_twohop": {"term": "两跳邻居数", "dim": "结构上下文", "meaning": "距一端恰两跳的节点数"},
    # 检验可信度
    "n": {"term": "样本量", "dim": "检验可信度", "meaning": "本地数据行数"},
    "cell_min": {"term": "每格最小计数", "dim": "检验可信度", "meaning": "列联表最小格计数，过小提示检验不稳健"},
    "state_u": {"term": "U 状态数", "dim": "检验可信度", "meaning": "变量 U 的取值个数"},
    "state_v": {"term": "V 状态数", "dim": "检验可信度", "meaning": "变量 V 的取值个数"},
}


def field_doc_text() -> str:
    """存在性字段解释文本（prompt 渲染用：专业术语 / 判断维度 / 用处）。"""
    lines = []
    for name, doc in FIELD_DOC.items():
        lines.append("- %s【%s/维度=%s】：%s" % (name, doc["term"], doc["dim"], doc["meaning"]))
    return "\n".join(lines)


@register_provider
class ExistenceEvidenceProvider(EvidenceProvider):
    name = "existence"
    kind = EvidenceType.EXISTENCE

    def compute(self, data, columns, edges, discrete, options) -> Dict[Tuple[str, str], Dict[str, Any]]:
        stats = [stat_registry.get(n)() for n in stat_registry.names()
                 if (stat_registry.get(n) is not None
                     and getattr(stat_registry.get(n), "kind", "") == "existence")]
        out: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for edge in edges:
            fields: Dict[str, Any] = {}
            for stat in stats:
                try:
                    fields.update(stat.compute(data, columns, edge, discrete, options))
                except Exception as exc:
                    from util.log import get
                    get("evidence.existence").warning("统计量 %s 计算失败 %s: %s",
                                                      stat.name, "->".join(edge), exc)
            out[edge] = fields
        return out
