# -*- coding: utf-8 -*-
"""算法层共享适配工具（从 src/util/algorithms/common.py 复刻）。

- default_cit / resolve_cit：indep_test "auto"/空 → 按数据特征选默认（离散 gsq / 连续 fisherz）。
- as_numpy：gcastle 输出（torch Tensor）统一转 numpy。
- pgmpy_frame / pgmpy_edges：numpy ↔ pgmpy DataFrame / DAG 适配。
"""
from typing import Dict, List, Optional

import numpy as np

from util.algorithms.edge import Edge


def default_cit(discrete: bool) -> str:
    """默认独立性检验：离散→gsq，连续→fisherz。"""
    return "gsq" if discrete else "fisherz"


def resolve_cit(params: Dict, discrete: bool) -> str:
    """indep_test 参数解析：auto/空 → 按数据特征选默认；否则原样返回。"""
    cit = params.pop("indep_test", "auto")
    if cit in (None, "", "auto"):
        return default_cit(discrete)
    return str(cit)


def call_kwargs(fn, params: Dict, **fixed):
    """把 params 按 fn 签名过滤后调用（v4.24 修：ALGORITHM_PARAMS 陈旧参数不再透传崩溃）。

    被过滤掉的参数打 warning 进实验日志（配置错误可见，不静默）；fixed 为已具名的固定参数。
    """
    import inspect
    from util.log import get
    allowed = set(inspect.signature(fn).parameters) - set(fixed)
    drop = [k for k in params if k not in allowed]
    if drop:
        get("algorithms").warning("参数被忽略（目标函数不接受）: %s", drop)
    return fn(**fixed, **{k: v for k, v in params.items() if k in allowed})


def as_numpy(x) -> np.ndarray:
    """gcastle 输出统一转 numpy（causal_matrix 可能是 torch Tensor）。"""
    if hasattr(x, "detach"):
        x = x.detach()
    return np.asarray(x, dtype=float)


def pgmpy_frame(data: np.ndarray, columns: Optional[List[str]], discrete: bool):
    """numpy → pgmpy DataFrame：离散列转 category。"""
    import pandas as pd
    cols = list(columns) if columns else [f"X{i}" for i in range(data.shape[1])]
    df = pd.DataFrame(data, columns=cols)
    return df.astype("category") if discrete else df.astype(float)


def pgmpy_edges(dag, columns: Optional[List[str]]) -> List[Edge]:
    """pgmpy DAG（edges 为列名元组）→ Edge 列表（索引序小端在前）。"""
    cols = list(columns) if columns else []
    name2idx = {c: i for i, c in enumerate(cols)}
    out: List[Edge] = []
    for s, t in dag.edges():
        i, j = name2idx.get(s, -1), name2idx.get(t, -1)
        if i < 0 or j < 0:
            continue
        if i < j:
            out.append(Edge(i, j, "->"))
        else:
            out.append(Edge(j, i, "<-"))
    return out
