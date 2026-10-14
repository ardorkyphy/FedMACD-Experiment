# -*- coding: utf-8 -*-
"""Direction-evidence provider.

组合 stat_registry 中 kind=direction 的统计量（v 结构 ΔMI/ANM 残差）
+ 本地算法出图方向（alg_direction）→ DirectionEvidence 字段 dict。
stat_dir（0-1 统计方向强度，以 u->v 为基准）：连续=ANM 残差独立性；离散=0.5 无信息
（双变量方向无结构假设不可识别，v4.22 定稿）。

v4.32 方向性证据包：FIELD_DOC 字段解释（是什么/作用/适用场景）服务端定稿、
客户端本地算值、{dir_field_doc} 注入 prompt；结构上下文（骨架路径/邻居）在 provider
内计算；dir_identifiable 标记统计方向可辨识性（离散双变量不可辨识）。
"""
from typing import Any, Dict, Tuple

import numpy as np

from evidence.base import EvidenceProvider, EvidenceType, register_provider, stat_registry
from evidence.statistics import _shortest_path

FIELD_DOC: Dict[str, Dict[str, str]] = {
    # 结构线索
    "alg_direction": {"term": "本地算法输出方向", "dim": "结构线索",
                      "meaning": "该客户端因果发现算法对这条边的定向结论（u->v / v->u / 方向未定 / 未出此边）；适用=所有数据形态，算法假设与数据前提一致时最可靠"},
    "collider": {"term": "v 结构/对撞检验", "dim": "结构线索",
                 "meaning": "以候选节点为条件后关联增强（ΔMI>0 / 偏相关增强）→ 两端共同指向条件节点；适用=有共同邻居或两跳候选节点的边"},
    "cond_cand_n": {"term": "v 结构候选节点数", "dim": "结构线索",
                    "meaning": "可做对撞检验的候选条件节点数（骨架邻居并集）"},
    "cond_coverage": {"term": "v 结构检验覆盖度", "dim": "结构线索",
                      "meaning": "已检验的候选节点数/可用候选数；0/0=没有可做对撞检验的候选节点"},
    # 机制假设
    "stat_dir": {"term": "统计方向", "dim": "机制假设",
                 "meaning": "以 u->v 为基准的方向分数（0.5=无信息）；适用=连续数据（ANM 残差独立性折算）；离散双变量无结构假设下统计方向不可辨识"},
    "resid_u_to_v": {"term": "ANM 残差独立性（u->v）", "dim": "机制假设",
                     "meaning": "以 u→v 为机制方向的加性噪声回归后，残差与 u 的互信息；加性噪声假设成立时正确方向的残差更独立（数值小）；适用=连续数据"},
    "resid_v_to_u": {"term": "ANM 残差独立性（v->u）", "dim": "机制假设",
                     "meaning": "以 v→u 为机制方向回归后残差与 v 的互信息；适用=连续数据"},
    # 结构上下文
    "skel_path_len": {"term": "骨架最短路径长度", "dim": "结构上下文",
                      "meaning": "两端在本地骨架上的最短路径边数（1=相邻，>1=隔着中间节点）；适用=所有形态，提示该边可用的结构方向证据多少"},
    "mid_nodes": {"term": "路径中间节点", "dim": "结构上下文",
                  "meaning": "骨架最短路径途经的中间节点名"},
    "n_common": {"term": "共同邻居数", "dim": "结构上下文",
                 "meaning": "两端共同邻居数量（对撞检验候选节点的一部分）"},
    "n_twohop": {"term": "两跳邻居数", "dim": "结构上下文",
                 "meaning": "距一端恰两跳的节点数"},
}


def direction_support(fields: Dict[str, Any]) -> float:
    """统计方向强度（0-1，以 u->v 为基准；0.5=无信息）。

    v4.32 复刻 src/client/local_probs.py::direction_support（不跨 src import）：
    连续=ANM 残差独立性（正确方向的残差更独立 → 残差 MI 更小）；
    离散双变量无结构假设不可辨识（v4.22 定稿）→ 0.5，仅结构线索可辨识。
    """
    a = fields.get("resid_v_to_u")
    b = fields.get("resid_u_to_v")
    if a is None or b is None:
        return 0.5
    a, b = float(a or 0.0), float(b or 0.0)
    denom = a + b
    sig = (a - b) / denom if denom > 0 else 0.0
    return float(np.clip(0.5 + 0.5 * sig, 0.0, 1.0))


def field_doc_text() -> str:
    """方向性字段解释文本（prompt 渲染用：专业术语 / 判断维度 / 用处与适用场景）。"""
    lines = []
    for name, doc in FIELD_DOC.items():
        lines.append("- %s【%s/维度=%s】：%s" % (name, doc["term"], doc["dim"], doc["meaning"]))
    return "\n".join(lines)


@register_provider
class DirectionEvidenceProvider(EvidenceProvider):
    name = "direction"
    kind = EvidenceType.DIRECTION

    def compute(self, data, columns, edges, discrete, options) -> Dict[Tuple[str, str], Dict[str, Any]]:
        stats = [stat_registry.get(n)() for n in stat_registry.names()
                 if (stat_registry.get(n) is not None
                     and getattr(stat_registry.get(n), "kind", "") == "direction")]
        alg_dir = options.get("alg_direction") or {}        # edge -> "->"/"<-"/"none"
        neigh = options.get("neighbors") or {}
        out: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for edge in edges:
            fields: Dict[str, Any] = {"alg_direction": alg_dir.get(edge)}
            for stat in stats:
                try:
                    fields.update(stat.compute(data, columns, edge, discrete, options))
                except Exception as exc:
                    from util.log import get
                    get("evidence.direction").warning("统计量 %s 计算失败 %s: %s",
                                                      stat.name, "->".join(edge), exc)
            fields["stat_dir"] = direction_support(fields)
            fields["dir_identifiable"] = not bool(discrete)   # Bivariate statistics do not orient discrete pairs.
            # 结构上下文（v4.32，复用存在性 BFS）
            u, v = edge
            nu, nv = set(neigh.get(u, set())), set(neigh.get(v, set()))
            path = _shortest_path(neigh, u, v)
            fields["skel_path_len"] = (len(path) - 1) if path else -1
            mid = path[1:-1] if path else []
            if len(mid) > 2:
                mid = [mid[0], mid[-1]]
            fields["mid_nodes"] = mid
            fields["n_common"] = len(nu & nv)
            dist2 = set()
            for w in nu:
                dist2 |= set(neigh.get(w, set()))
            dist2 -= nu
            dist2.discard(u)
            fields["n_twohop"] = len(dist2)
            out[edge] = fields
        return out
