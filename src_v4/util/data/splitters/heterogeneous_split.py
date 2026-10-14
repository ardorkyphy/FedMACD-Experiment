# -*- coding: utf-8 -*-
"""Heterogeneous splitting with type-specific distribution shifts.

The implementation preserves graph structure while shifting distributions:
- 离散 BN（BIF）：每客户端从软干预后的模型独立 forward_sample；独立生成不表示离散取值不会重复；
  软干预 P'(X_j|Pa_j) = (1-eps)*P + eps*Q_j，Q_j = 偏好状态 onehot，eps = heterogeneity_strength。
- 连续（无模型 CSV）：多变量偏好区加权无放回重采样；客户端内不重复，客户端间可共享源记录。
"""
import os
from typing import Any, Dict, List, Optional

import numpy as np

from util.data.base import Splitter


class HeterogeneousSplitter(Splitter):
    """异质场景：干预采样（BIF）/ 加权无放回重采样（CSV）。"""

    name = "heterogeneous"

    def split(self, data: np.ndarray, num_clients: int, seed: int,
              data_cfg: Optional[Dict[str, Any]] = None) -> List[np.ndarray]:
        cfg = data_cfg or {}
        fmt = str(cfg.get("format", "") or "").strip().lower()
        if not fmt:
            from util.data.datasets import resolve_path
            info = resolve_path(str(cfg.get("name", "") or ""))
            fmt = os.path.splitext(info["data"])[1].lstrip(".").lower()
        if fmt == "bif":
            return split_interventional(num_clients, seed, cfg)
        return split_resample(data, num_clients, seed, cfg)


# ---------- 离散 BN：软干预采样 ----------

def split_interventional(num_clients: int, seed: int,
                         cfg: Dict[str, Any]) -> List[np.ndarray]:
    """每客户端从软干预后的 BIF 模型独立采样（结构不变、分布漂移）。

    软干预 P'(X_j|Pa_j) = (1-eps)*P + eps*Q_j：随机选 2~3 个节点（
    cfg 显式给 intervention_nodes 时固定），偏好状态 s* 的 onehot 作 Q_j。
    干预集各客户端不同。
    """
    from pgmpy.readwrite import BIFReader
    from pgmpy.sampling import BayesianModelSampling
    from pgmpy.factors.discrete import TabularCPD

    from util.data.common import total_n
    from util.data.datasets import resolve_path

    path = resolve_path(str(cfg.get("name", "") or ""))["data"]
    eps = float(cfg.get("heterogeneity_strength", 0.7) or 0.7)
    cfg_iv = cfg.get("intervention_nodes")          # Explicit count or 2-3 randomly selected nodes per client.
    n_total = total_n(cfg)
    per = n_total // num_clients

    parts: List[np.ndarray] = []
    base = BIFReader(path).get_model()
    nodes = list(base.nodes())
    for c in range(num_clients):
        model = BIFReader(path).get_model()
        rng = np.random.default_rng(seed * 1000 + c)
        n_iv = min(len(nodes), max(int(cfg_iv) if cfg_iv else int(rng.integers(2, 4)), 1))
        iv_nodes = rng.choice(nodes, size=n_iv, replace=False)
        for node in iv_nodes:
            model.add_cpds(_soft_intervene_cpd(model, node, eps, rng))
        df = BayesianModelSampling(model).forward_sample(
            size=per, seed=seed * 100 + c, show_progress=False)
        # Encode by declared BIF states rather than observed appearance order,
        # 同名义状态跨客户端/跨折映射同一整数，避免 MATLAB 族跨客户端联合检验编码错位
        df = _encode_by_states(df, model)
        parts.append(df.to_numpy())
        print(f"  [het] 客户端{c} 干预节点={list(iv_nodes)} eps={eps:.2f} n={len(df)}")
    return parts


def _soft_intervene_cpd(model, node: str, eps: float, rng) -> Any:
    """软干预 CPD：P' = (1-eps)*P + eps*onehot(s*)，偏好状态 s* 由 rng 选。"""
    from pgmpy.factors.discrete import TabularCPD
    cpd = model.get_cpds(node)
    vals = np.asarray(cpd.values, dtype=float)
    card = vals.shape[0]
    s_star = int(rng.integers(card))
    new_vals = (1.0 - eps) * vals.copy()
    new_vals[s_star, ...] += eps
    new_vals = np.clip(new_vals, 1e-12, None)
    new_vals /= new_vals.sum(axis=0, keepdims=True)
    flat = np.ascontiguousarray(new_vals.reshape(card, -1))
    sn = {node: model.states[node]}
    if len(cpd.variables) > 1:
        evs = cpd.variables[1:]
        for ev in evs:
            sn[ev] = model.states[ev]
        return TabularCPD(node, card, flat, evidence=evs,
                          evidence_card=[vals.shape[i] for i in range(1, vals.ndim)],
                          state_names=sn)
    return TabularCPD(node, card, flat, state_names=sn)


def _encode_by_states(df, model):
    """按 BIF states 定义序编码（同名义状态同整数，跨客户端/跨折稳定）。"""
    from util.data.common import encode_by_states
    return encode_by_states(df, model)


# ---------- 连续（无模型）：多变量偏好区加权无放回 ----------

def split_resample(data: np.ndarray, num_clients: int, seed: int,
                   cfg: Dict[str, Any]) -> List[np.ndarray]:
    """Resample continuous rows using multivariable preference regions.

    每客户端随机选 intervention_nodes 个变量，各自随机取高分位区（>=P80）或低分位区（<=P20）
    作偏好区（preference_quantile 控制分位，默认 0.20）；偏好区 = 各干预变量偏好区并集（OR）。
    分段无放回抽取：先抽偏好区行至目标占比 rho = p + eps*(1-p)（p=偏好区自然占比），
    其余行从非偏好区补齐——eps 直接控制偏好区占比目标，容量不足（偏好区行数不够）自动降级。
    客户端内无重复；实际强度（边际 JS 散度）由 run.py het 审计落盘。
    """
    n, m = data.shape
    eps = float(cfg.get("heterogeneity_strength", 0.7) or 0.7)
    cfg_iv = cfg.get("intervention_nodes")          # Explicit count or 2-3 randomly selected variables per client.
    ql = float(cfg.get("preference_quantile", 0.20) or 0.20)
    n_c = [n // num_clients + (1 if c < n % num_clients else 0) for c in range(num_clients)]

    parts: List[np.ndarray] = []
    for c in range(num_clients):
        rng = np.random.default_rng(seed * 1000 + c)
        n_iv = min(m, max(int(cfg_iv) if cfg_iv else int(rng.integers(2, 4)), 1))
        iv_cols = rng.choice(m, size=n_iv, replace=False)
        pref = np.zeros(n, dtype=bool)
        desc = []
        for j in iv_cols:
            lo, hi = np.quantile(data[:, j], [ql, 1.0 - ql])
            high = bool(rng.integers(2))
            if high:
                zone = data[:, j] >= hi
                desc.append(f"col{j}>={int((1.0 - ql) * 100)}分位")
            else:
                zone = data[:, j] <= lo
                desc.append(f"col{j}<={int(ql * 100)}分位")
            pref |= zone
        p = float(pref.mean())
        rho = min(p + eps * (1.0 - p), 1.0)
        P_idx = np.flatnonzero(pref)
        N_idx = np.flatnonzero(~pref)
        k_p = min(int(round(n_c[c] * rho)), len(P_idx))
        rows = np.concatenate([
            rng.choice(P_idx, size=k_p, replace=False),
            rng.choice(N_idx, size=n_c[c] - k_p, replace=False),
        ])
        rng.shuffle(rows)
        parts.append(data[rows])
        print(f"  [het] 客户端{c} 偏好区={desc} 目标占比={rho:.2f} "
              f"实际偏好行={k_p}/{n_c[c]} ({k_p / n_c[c] * 100:.0f}%) 唯一行率=100%")
    return parts
