# -*- coding: utf-8 -*-
"""数据层共享工具（从 src/util/data/common.py 复刻）。"""
import os
from typing import Any, Dict, List, Set, Tuple

import numpy as np
import pandas as pd


def total_n(data_cfg: Dict[str, Any]) -> int:
    """样本量：per_client_size*num_clients 优先；其次 sample_size；0=全量哨兵。"""
    num_clients = int(data_cfg.get("num_clients", 1) or 1)
    per_client = int(data_cfg.get("per_client_size", 0) or 0)
    sample_size = int(data_cfg.get("sample_size", 0) or 0)
    if per_client > 0:
        return per_client * num_clients
    if sample_size > 0:
        return sample_size
    return int(np.iinfo(np.int64).max)


def sampling_seed(data_cfg: Dict[str, Any]) -> int:
    """BIF 前向采样种子：sampling_seed<0 时跟随分割种子（每折独立采样）。"""
    seed = int(data_cfg.get("seed", 0) or 0)
    sampling_seed = int(data_cfg.get("sampling_seed", 0) or 0)
    return seed if sampling_seed < 0 else sampling_seed


def factorize_discrete(df: pd.DataFrame) -> pd.DataFrame:
    """把离散列编码为 0..k 整数。"""
    out = df.copy()
    for col in out.columns:
        out[col] = pd.factorize(out[col])[0]
    return out


def encode_by_states(df: pd.DataFrame, model) -> pd.DataFrame:
    """按 BIF states 定义序编码（同名义状态同整数，跨客户端/跨折稳定）。

    与 factorize_discrete（出现序）的区别：样本缺某状态时仍按定义序占位，
    保证全局数据与各分片、各折之间编码一致（het 审计 / MATLAB 族联合检验前提）。
    """
    out = df.copy()
    for col in out.columns:
        st = list(model.states[col])
        out[col] = [st.index(v) for v in out[col]]
    return out


def load_graph_edges(graph_path: str, columns: List[str]) -> Set[Tuple[int, int]]:
    """真图文件 → 索引边集：csv（source,target 两列）或 adjacency 矩阵 txt。"""
    ext = os.path.splitext(graph_path)[1].lower()
    name2idx = {name: i for i, name in enumerate(columns)}
    edges: Set[Tuple[int, int]] = set()
    if ext == ".csv":
        df = pd.read_csv(graph_path)
        for _, row in df.iterrows():
            src, tgt = row.iloc[0], row.iloc[1]
            si = name2idx.get(str(src), int(src))
            ti = name2idx.get(str(tgt), int(tgt))
            edges.add((si, ti))
    else:  # adjacency matrix txt
        adj = np.loadtxt(graph_path, delimiter=" ")
        edges = {(int(i), int(j)) for i, j in zip(*np.nonzero(adj))}
    return edges