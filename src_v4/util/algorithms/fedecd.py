# -*- coding: utf-8 -*-
"""FedECD 本地方法选项（PC + bootstrap 重采样 + 方向多数投票，从 src/util/algorithms/fedecd.py 复刻）。

Only the local PC-plus-bootstrap method is exposed as an LLM-selectable option;
the original federated aggregation is not used:
- bootstrap：对本地数据有放回重采样 B 次（FedECD MATLAB 默认 B=15，boostrapping_correct_dag.m）
- 每次重采样跑一次 PC
- 投票：逐对边方向多数（dag_aggre.m：>= 平局取高索引为因）
- bootstrap 种子：每客户端独立派生（AlgorithmRunner.run 可传 seed=base+client_index），
  不再用 MATLAB 的全局 rng(666)
"""
from typing import Any, Dict, List, Optional, Set, Tuple

import math
import numpy as np

from util.algorithms.base import CausalDiscoveryAlgorithm
from util.algorithms.common import call_kwargs, resolve_cit
from util.algorithms.edge import Edge, graph_to_edges
from util.algorithms.registry import register


def aggregate_direction_votes(votes: Dict[Tuple[int, int], Dict[str, int]],
                              seen: Set[Tuple[int, int]], n: int,
                              min_votes: int = 1) -> List[Edge]:
    """bootstrap 投票聚合（对应 MATLAB dag_aggre.m + 可配边保留门槛）。

    逐对 (i, j)（i<j，索引序小端在前）：
    - 两方向票数合计为 0 且从未出现 → 无此边
    - 合计为 0 但出现过（全是 none）→ Edge(..., "none")
    - 合计 < min_votes（出现次数不足门槛，如非多数）→ 直接丢弃（bagging 语义：边不存在）
    - 否则方向多数；>= 平局取高索引（v）为因（与 MATLAB 一致）

    min_votes：边保留最低出现次数（1=任一折出现即保留=原并集行为；
    ceil(B/2)≈真多数投票 bagging，可抑制并集投票的假边爆炸）。
    """
    out: List[Edge] = []
    for i in range(n):
        for j in range(i + 1, n):
            key = (i, j)
            if key not in votes and key not in seen:
                continue
            v = votes.get(key)
            total = (v["->"] + v["<-"]) if v else 0
            if total == 0:
                out.append(Edge(i, j, "none"))
                continue
            if total < min_votes:
                continue
            if v["<-"] >= v["->"]:
                out.append(Edge(i, j, "<-"))   # 高索引为因
            else:
                out.append(Edge(i, j, "->"))
    return out


@register
class FedEcdLocalAlgorithm(CausalDiscoveryAlgorithm):
    """FedECD 本地方法：PC + bootstrap(B 次) + 方向多数投票（data_type=any）。"""
    name = "fedecd_local"
    family = "constraint"
    data_type = "discrete"   # 仅离散池：bootstrap 投票是离散稳健性手段，连续数据走评分/优化/PC
    cost = "slow"
    libs = ("causallearn",)
    requires_module = "causallearn.search.ConstraintBased.PC"
    param_schema = {
        "B": {"type": "integer", "default": 15,
              "description": "bootstrap 重采样次数（FedECD 默认 15；冒烟/调试可调小）"},
        "keep_ratio": {"type": "number", "default": 0.0,
                       "description": "边保留门槛：出现次数占成功折数比例（0=并集/任一折出现即保留（原行为，召回最高、假边最多）；0.5=多数投票 bagging（精度更高）；1.0=全部折都出现才保留）"},
        "indep_test": {"type": "string", "enum": ["auto", "gsq", "fisherz"], "default": "auto"},
    }

    description = "约束基：本地 PC + bootstrap 重采样 B 次（默认 15）投票聚合方向（FedECD 本地方法）；数据前提：离散。"

    def discover(self, data, discrete: bool, columns: Optional[List[str]]) -> List[Any]:
        from causallearn.search.ConstraintBased.PC import pc
        p = dict(self.params)
        b = int(p.pop("B", 15) or 15)
        if b < 1:
            b = 15
        keep_ratio = float(p.pop("keep_ratio", 0.0) or 0.0)
        cit = resolve_cit(p, discrete)
        arr = np.asarray(data)
        n = int(arr.shape[0])
        rng = np.random.default_rng(self.seed)
        votes: Dict[Tuple[int, int], Dict[str, int]] = {}
        seen: Set[Tuple[int, int]] = set()
        last_exc: Optional[Exception] = None
        ok = 0
        for _ in range(b):
            idx = rng.integers(0, n, size=n)
            try:
                cg = call_kwargs(pc, p, data=arr[idx], alpha=self.alpha,
                                 indep_test=cit, show_progress=False)
            except Exception as exc:                 # 单折失败跳过（bootstrap 多数决天然容忍）
                last_exc = exc
                continue
            ok += 1
            for e in graph_to_edges(cg.G.graph):
                key = (e.u, e.v)
                seen.add(key)
                if e.direction in ("->", "<-"):
                    votes.setdefault(key, {"->": 0, "<-": 0})[e.direction] += 1
        if ok == 0:
            raise RuntimeError(f"fedecd_local：{b} 折 bootstrap 全部失败（最后异常: {last_exc}）")
        min_votes = 1 if keep_ratio <= 0 else max(1, int(math.ceil(keep_ratio * ok)))
        return aggregate_direction_votes(votes, seen, n, min_votes=min_votes)
