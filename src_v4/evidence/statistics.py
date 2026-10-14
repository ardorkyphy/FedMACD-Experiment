# -*- coding: utf-8 -*-
"""Statistical-evidence registry and implementations.

统计量实现从 src/util/analysis_stats.py 复刻（纯函数，数据不出本地）：
- 存在性：MI / NMI / 边际 p（G² 离散 / Pearson 连续）/ 条件独立 p（"被解释掉"检验）/ 格密度。
- 方向性：条件熵不对称（离散）/ v 结构 ΔMI / ANM 残差独立性双向（连续）。
每个统计量一个类（@register_stat），provider 组合成 ExistenceEvidence / DirectionEvidence 字段。
"""
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy.stats import chi2, pearsonr, spearmanr, t as tdist

from evidence.base import Statistic, register_stat

# ---------------- 复刻自 src/util/analysis_stats.py 的纯函数 ----------------

def _joint_counts(x, y, sx, sy) -> np.ndarray:
    flat = np.bincount(x.astype(np.int64) * sy + y.astype(np.int64),
                       minlength=int(sx * sy)).reshape(int(sx), int(sy)).astype(float)
    return flat


def _entropy(counts: np.ndarray, total: float) -> float:
    p = counts[counts > 0] / total
    return float(-(p * np.log(p)).sum()) if p.size else 0.0


def _mi_xy(x, y, sx, sy) -> float:
    c = _joint_counts(x, y, sx, sy)
    n = float(c.sum())
    if n <= 0:
        return 0.0
    hx = _entropy(c.sum(axis=1), n)
    hy = _entropy(c.sum(axis=0), n)
    hxy = _entropy(c.ravel(), n)
    return max(0.0, hx + hy - hxy)


def _g2_xy(x, y, sx, sy) -> Tuple[float, float]:
    c = _joint_counts(x, y, sx, sy)
    n = float(c.sum())
    px, py = c.sum(axis=1), c.sum(axis=0)
    g2 = 0.0
    exp = np.outer(px, py) / n
    nz = c > 0
    for i in range(int(sx)):
        for j in range(int(sy)):
            if nz[i, j] and exp[i, j] > 0:
                g2 += 2.0 * c[i, j] * np.log(c[i, j] / exp[i, j])
    dof = max(1, int((sx - 1) * (sy - 1)))
    return g2, float(chi2.sf(max(0.0, g2), dof))


def _cond_mi_g2(x, y, z, sx, sy, sz) -> Tuple[float, float, float]:
    n = int(len(x))
    c = np.bincount(x.astype(np.int64) * sy * sz + y.astype(np.int64) * sz + z.astype(np.int64),
                    minlength=int(sx * sy * sz)).reshape(int(sx), int(sy), int(sz)).astype(float)
    mi = 0.0
    g2 = 0.0
    for k in range(int(sz)):
        nk = float(c[:, :, k].sum())
        if nk <= 0:
            continue
        pz = nk / n
        hxz = _entropy(c[:, :, k].sum(axis=1), nk)
        hyz = _entropy(c[:, :, k].sum(axis=0), nk)
        hxyz = _entropy(c[:, :, k].ravel(), nk)
        mi += pz * (hxz + hyz - hxyz)
        pxz, pyz = c[:, :, k].sum(axis=1), c[:, :, k].sum(axis=0)
        for i in range(int(sx)):
            for j in range(int(sy)):
                o = c[i, j, k]
                if o <= 0:
                    continue
                e = pxz[i] * pyz[j] / nk if (pxz[i] > 0 and pyz[j] > 0) else 0.0
                if e > 0:
                    g2 += 2.0 * o * np.log(o / e)
    dof = max(1, int((sx - 1) * (sy - 1) * sz))
    return max(0.0, mi), g2, float(chi2.sf(max(0.0, g2), dof))


def _binned_xy(a: np.ndarray, b: np.ndarray, bins: int = 12):
    """分位分箱：返回 (分箱索引a, 分箱索引b, 状态数a, 状态数b)（连续数据 MI/NMI 共用，复刻 v3）。"""
    def _q(v: np.ndarray) -> np.ndarray:
        edges = np.quantile(v, np.linspace(0, 1, bins + 1)[1:-1])
        edges = np.unique(edges)
        return np.digitize(v, edges)
    qa, qb = _q(a), _q(b)
    return qa, qb, int(qa.max()) + 1, int(qb.max()) + 1


def _binned_mi(a: np.ndarray, b: np.ndarray, bins: int = 12) -> float:
    qa, qb, sa, sb = _binned_xy(a, b, bins)
    return _mi_xy(qa, qb, sa, sb)


def _partial_corr_p(x: np.ndarray, y: np.ndarray, zmat) -> Tuple[float, float]:
    """偏相关：x,y 对 z 回归后的残差相关系数 + t 检验 p 值（连续版"被解释"检验，复刻 v3 partial_r）。"""
    zmat = np.asarray(zmat, dtype=float)
    if zmat.ndim == 1:
        zmat = zmat.reshape(-1, 1)
    A = np.column_stack([np.ones(len(x))] + [zmat[:, k] for k in range(zmat.shape[1])])
    rx = x - A @ np.linalg.lstsq(A, x, rcond=None)[0]
    ry = y - A @ np.linalg.lstsq(A, y, rcond=None)[0]
    r = float(np.corrcoef(rx, ry)[0, 1])
    k = int(zmat.shape[1])
    dof = int(len(x) - 2 - k)
    if dof < 1 or abs(r) >= 1.0:
        return r, 1.0
    t = r * np.sqrt(dof / (1.0 - r * r))
    return r, float(2.0 * tdist.sf(abs(t), dof))


def _resid_dir(x: np.ndarray, y: np.ndarray) -> float:
    A = np.column_stack([np.ones(len(x)), x])
    resid = y - A @ np.linalg.lstsq(A, y, rcond=None)[0]
    return _binned_mi(resid, x)


# ---------------- 共享：边与邻居解析 ----------------

def _edge_ctx(data: np.ndarray, columns: List[str], edge: Tuple[str, str],
              discrete: bool, options: Dict[str, Any]):
    name2idx = {c: i for i, c in enumerate(columns)}
    u, v = edge
    i, j = name2idx[u], name2idx[v]
    xi, xj = data[:, i], data[:, j]
    state_map = options.get("state_map") or {}
    if discrete:
        su = int(state_map.get(u, data[:, i].max() + 1))
        sv = int(state_map.get(v, data[:, j].max() + 1))
    else:
        su = sv = 0
    return u, v, i, j, xi, xj, su, sv



def _cond_candidates(data: np.ndarray, columns: List[str], u: str, v: str,
                     xi: np.ndarray, xj: np.ndarray, su: int, sv: int,
                     discrete: bool, state_map: Dict[str, int], alpha: float,
                     neigh: Dict[str, set]) -> List[str]:
    """条件化候选集：骨架一跳邻居并集 ∪ 与 u/v 任一端边际关联显著（p<α）的变量。

    Under heterogeneous incomplete local skeletons, confounders or mediators
    may fall outside one-hop neighbours. Marginally associated variables are
    therefore included and ranked by summed association with both endpoints.
    """
    name2idx = {c: k for k, c in enumerate(columns)}
    base = set(neigh.get(u, set())) | set(neigh.get(v, set()))
    strength: Dict[str, float] = {}
    if discrete:
        for w in columns:
            if w in (u, v):
                continue
            k = name2idx[w]
            wk = data[:, k]
            sw = int(state_map.get(w, wk.max() + 1))
            s = _mi_xy(xi, wk, su, sw) + _mi_xy(xj, wk, sv, sw)
            if w in base or min(_g2_xy(xi, wk, su, sw)[1], _g2_xy(xj, wk, sv, sw)[1]) < alpha:
                strength[w] = s
    else:
        xi_f, xj_f = xi.astype(float), xj.astype(float)
        for w in columns:
            if w in (u, v):
                continue
            k = name2idx[w]
            wk = data[:, k].astype(float)
            s = abs(float(np.corrcoef(xi_f, wk)[0, 1])) + abs(float(np.corrcoef(xj_f, wk)[0, 1]))
            if w in base or min(pearsonr(xi_f, wk)[1], pearsonr(xj_f, wk)[1]) < alpha:
                strength[w] = s
    return sorted(strength, key=strength.get, reverse=True)


# ---------------- 存在性统计量 ----------------

@register_stat
class CellDensity(Statistic):
    """样本量 / 两端状态数 / 每格最小计数（检验可信度）。"""
    name = "cell_density"
    kind = "existence"

    def compute(self, data, columns, edge, discrete, options) -> Dict[str, Any]:
        u, v, i, j, xi, xj, su, sv = _edge_ctx(data, columns, edge, discrete, options)
        out: Dict[str, Any] = {"n": int(data.shape[0])}
        if discrete:
            c = _joint_counts(xi, xj, su, sv)
            out["state_u"], out["state_v"] = int(su), int(sv)
            out["cell_min"] = float(c.min()) if c.size else 0.0
        return out


@register_stat
class MarginalMI(Statistic):
    """边际互信息（离散）。"""
    name = "mi"
    kind = "existence"

    def compute(self, data, columns, edge, discrete, options) -> Dict[str, Any]:
        u, v, i, j, xi, xj, su, sv = _edge_ctx(data, columns, edge, discrete, options)
        if not discrete:
            return {"mi": round(_binned_mi(xi.astype(float), xj.astype(float)), 4)}
        return {"mi": round(_mi_xy(xi, xj, su, sv), 4)}


@register_stat
class MarginalNMI(Statistic):
    """归一化互信息（离散，跨客户端聚合用）。"""
    name = "nmi"
    kind = "existence"

    def compute(self, data, columns, edge, discrete, options) -> Dict[str, Any]:
        u, v, i, j, xi, xj, su, sv = _edge_ctx(data, columns, edge, discrete, options)
        if not discrete:
            qa, qb, sa, sb = _binned_xy(xi.astype(float), xj.astype(float))
            c = _joint_counts(qa, qb, sa, sb)
            n = float(c.sum())
            hx = _entropy(c.sum(axis=1), n)
            hy = _entropy(c.sum(axis=0), n)
            mi = _mi_xy(qa, qb, sa, sb)
            return {"nmi": round(mi / min(hx, hy), 4) if min(hx, hy) > 0 else 0.0}
        c = _joint_counts(xi, xj, su, sv)
        n = float(c.sum())
        hx = _entropy(c.sum(axis=1), n)
        hy = _entropy(c.sum(axis=0), n)
        mi = _mi_xy(xi, xj, su, sv)
        return {"nmi": round(mi / min(hx, hy), 4) if min(hx, hy) > 0 else 0.0}


@register_stat
class MarginalP(Statistic):
    """边际独立性 p 值（离散 G² / 连续 Pearson）。"""
    name = "marginal_p"
    kind = "existence"

    def compute(self, data, columns, edge, discrete, options) -> Dict[str, Any]:
        u, v, i, j, xi, xj, su, sv = _edge_ctx(data, columns, edge, discrete, options)
        if discrete:
            _, p0 = _g2_xy(xi, xj, su, sv)
            return {"g2_p": round(p0, 4)}
        r, p = pearsonr(xi.astype(float), xj.astype(float))
        return {"pearson_r": round(float(r), 4), "pearson_p": round(float(p), 4)}


@register_stat
class ConditionalP(Statistic):
    """Conditional-independence p-value using the strongest conditioning candidates.

    Candidates include the union of one-hop skeleton neighbours and variables
    marginally associated with either endpoint. ``cond_cand_n`` and
    ``cond_coverage`` report candidate count and tested-edge coverage.
    """
    name = "conditional_p"
    kind = "existence"

    def compute(self, data, columns, edge, discrete, options) -> Dict[str, Any]:
        u, v, i, j, xi, xj, su, sv = _edge_ctx(data, columns, edge, discrete, options)
        name2idx = {c: i for i, c in enumerate(columns)}
        state_map = options.get("state_map") or {}
        alpha = float(options.get("alpha", 0.05) or 0.05)
        if not discrete:
            # Continuous form: partial correlation with its t-test p-value.
            xi_f, xj_f = xi.astype(float), xj.astype(float)
            r0, _ = pearsonr(xi_f, xj_f)
            neigh = options.get("neighbors") or {}
            max_cond = int(options.get("max_cond", 2) or 2)
            cand_all = _cond_candidates(data, columns, u, v, xi, xj, su, sv,
                                       discrete, state_map, alpha, neigh)
            cand = cand_all[:max_cond]
            cond = []
            for w in cand:
                pr, pc = _partial_corr_p(xi_f, xj_f, data[:, name2idx[w]].astype(float))
                cond.append({"w": w, "partial_r": round(pr, 4), "p": round(pc, 4),
                             "delta": round(abs(pr) - abs(r0), 4)})
            explained = any(float(cd["p"]) > alpha and float(cd["delta"]) < 0 for cd in cond)
            return {"cond": cond, "explained_away": explained,
                    "cond_cand_n": len(cand_all),
                    "cond_checked_n": len(cond),
                    "cond_max_cond": max_cond,
                    "cond_coverage": round(len(cond) / min(len(cand_all), max_cond), 4) if cand_all else 0.0}
        mi = _mi_xy(xi, xj, su, sv)
        neigh = options.get("neighbors") or {}
        max_cond = int(options.get("max_cond", 2) or 2)
        cand_all = _cond_candidates(data, columns, u, v, xi, xj, su, sv,
                                   discrete, state_map, alpha, neigh)
        cand = cand_all[:max_cond]
        cond = []
        for w in cand:
            k = name2idx[w]
            mi_c, _, p_c = _cond_mi_g2(xi, xj, data[:, k], su, sv, int(state_map.get(w, data[:, k].max() + 1)))
            cond.append({"w": w, "mi_c": round(mi_c, 4), "p": round(p_c, 4),
                         "delta": round(mi_c - mi, 4)})
        explained = any(float(cd["p"]) > alpha and float(cd["delta"]) < 0 for cd in cond)
        return {"cond": cond, "explained_away": explained,
                "cond_cand_n": len(cand_all),
                "cond_checked_n": len(cond),
                "cond_max_cond": max_cond,
                "cond_coverage": round(len(cond) / min(len(cand_all), max_cond), 4) if cand_all else 0.0}


@register_stat
class SpearmanR(Statistic):
    """Spearman 秩相关 + p（连续）：单调非线性关联（Pearson 的线性口径会漏检）。"""
    name = "spearman"
    kind = "existence"

    def compute(self, data, columns, edge, discrete, options) -> Dict[str, Any]:
        if discrete:
            return {}
        u, v, i, j, xi, xj, su, sv = _edge_ctx(data, columns, edge, discrete, options)
        rho, p = spearmanr(xi.astype(float), xj.astype(float))
        out = {"spearman_r": round(float(rho), 4), "spearman_p": round(float(p), 4)}
        r0, _ = pearsonr(xi.astype(float), xj.astype(float))
        if abs(float(r0)) > 1e-9:
            out["ratio_abs_sp_pearson"] = round(abs(float(rho)) / abs(float(r0)), 4)
        return out


@register_stat
class PartialAll(Statistic):
    """半偏相关（连续）：控制其余全部变量后的偏相关（接近"直接关联"的检验）。"""
    name = "partial_all"
    kind = "existence"

    def compute(self, data, columns, edge, discrete, options) -> Dict[str, Any]:
        if discrete:
            return {}
        u, v, i, j, xi, xj, su, sv = _edge_ctx(data, columns, edge, discrete, options)
        others = [k for k, c in enumerate(columns) if c != u and c != v]
        if not others:
            return {}
        pr, pc = _partial_corr_p(xi.astype(float), xj.astype(float),
                                 data[:, others].astype(float))
        return {"partial_all_r": round(pr, 4), "partial_all_p": round(pc, 4)}


@register_stat
class Determinism(Statistic):
    """确定性/机械决定（离散）：条件熵 + 熵比率 + 最大条件概率。

    条件化后"天然低熵"与"关联被解释"是两回事（asia tub→either 先例），
    熵比率低/最大条件概率高提示两端近机械决定关系。
    """
    name = "determinism"
    kind = "existence"

    def compute(self, data, columns, edge, discrete, options) -> Dict[str, Any]:
        if not discrete:
            return {}
        u, v, i, j, xi, xj, su, sv = _edge_ctx(data, columns, edge, discrete, options)
        c = _joint_counts(xi, xj, su, sv)
        n = float(c.sum())
        if n <= 0:
            return {}
        hx = _entropy(c.sum(axis=1), n)
        hy = _entropy(c.sum(axis=0), n)
        hxy = _entropy(c.ravel(), n)
        h_v_given_u = hxy - hx   # H(V|U)
        h_u_given_v = hxy - hy   # H(U|V)
        ratios = []
        if hy > 0:
            ratios.append(h_v_given_u / hy)
        if hx > 0:
            ratios.append(h_u_given_v / hx)
        ent_ratio = round(min(ratios), 4) if ratios else None
        px = c.sum(axis=1)   # 行计数（归一化得 P(U)）
        max_cond_prob = 0.0
        for ii in range(int(su)):
            if px[ii] > 0:
                max_cond_prob = max(max_cond_prob, float((c[ii, :] / px[ii]).max()))
        return {"h_u_given_v": round(h_u_given_v, 4), "h_v_given_u": round(h_v_given_u, 4),
                "ent_ratio": ent_ratio, "max_cond_prob": round(max_cond_prob, 4)}


@register_stat
class ZeroCell(Statistic):
    """零格占比（离散）：列联表空格比例，格子稀疏时检验与 MI 估计不稳健。"""
    name = "zero_cell"
    kind = "existence"

    def compute(self, data, columns, edge, discrete, options) -> Dict[str, Any]:
        if not discrete:
            return {}
        u, v, i, j, xi, xj, su, sv = _edge_ctx(data, columns, edge, discrete, options)
        c = _joint_counts(xi, xj, su, sv)
        return {"zero_cell_ratio": round(float((c == 0).mean()), 4)}


def _shortest_path(neigh, start, end):
    """骨架图 BFS 最短路径（返回节点列表；不连通 → []）。"""
    if start == end:
        return [start]
    from collections import deque
    prev = {start: None}
    q = deque([start])
    while q:
        cur = q.popleft()
        if cur == end:
            path = []
            while cur is not None:
                path.append(cur)
                cur = prev[cur]
            return path[::-1]
        for w in neigh.get(cur, set()):
            if w not in prev:
                prev[w] = cur
                q.append(w)
    return []


@register_stat
class StructContext(Statistic):
    """结构上下文（本地骨架图）：最短路径/中间节点/共同与两跳邻居数。

    边际强关联既可能来自直接边，也可能来自隔着中间节点的间接路径；
    路径长度与中间节点名供 LLM 结合条件化检验综合判断（证据，不预设结论）。
    """
    name = "struct_context"
    kind = "existence"

    def compute(self, data, columns, edge, discrete, options) -> Dict[str, Any]:
        neigh = options.get("neighbors") or {}
        u, v = edge
        nu, nv = set(neigh.get(u, set())), set(neigh.get(v, set()))
        n_common = len(nu & nv)
        path = _shortest_path(neigh, u, v)
        skel_path_len = (len(path) - 1) if path else -1
        mid = path[1:-1] if path else []
        if len(mid) > 2:
            mid = [mid[0], mid[-1]]
        dist2 = set()
        for w in nu:
            dist2 |= set(neigh.get(w, set()))
        dist2 -= nu
        dist2.discard(u)
        return {"skel_path_len": skel_path_len, "mid_nodes": mid,
                "n_common": n_common, "n_twohop": len(dist2)}


# ---------------- 方向性统计量 ----------------

class EntropyAsym(Statistic):
    """Deprecated and no longer registered in the evidence matrix.

    条件熵差值 H(u|v)-H(v|u) 恒等于 H(u)-H(v)（联合熵相消），符号只由边际熵差决定、
    与因果方向无关——双变量方向在无结构假设下统计不可识别，此类判据从配方移除。
    """
    name = "entropy_asym"
    kind = "direction"

    def compute(self, data, columns, edge, discrete, options) -> Dict[str, Any]:
        return {}


@register_stat
class ColliderDelta(Statistic):
    """v 结构/collider 候选（条件化后关联增强 Δ>0，方向线索）。

    Uses the same conditioning candidates as ``conditional_p`` and supports
    continuous data through partial-correlation enhancement.
    """
    name = "collider_delta"
    kind = "direction"

    def compute(self, data, columns, edge, discrete, options) -> Dict[str, Any]:
        u, v, i, j, xi, xj, su, sv = _edge_ctx(data, columns, edge, discrete, options)
        name2idx = {c: i for i, c in enumerate(columns)}
        state_map = options.get("state_map") or {}
        if not discrete:
            xi_f, xj_f = xi.astype(float), xj.astype(float)
            r0, _ = pearsonr(xi_f, xj_f)
            _thr = 0.01
            _delta = lambda pr: abs(float(pr)) - abs(float(r0))
            neigh = options.get("neighbors") or {}
            alpha = float(options.get("alpha", 0.05) or 0.05)
            max_cond = int(options.get("max_cond", 2) or 2)
            cand_all = _cond_candidates(data, columns, u, v, xi, xj, su, sv,
                                       discrete, state_map, alpha, neigh)
            cond = []
            for w in cand_all[:max_cond]:
                pr, pc = _partial_corr_p(xi_f, xj_f, data[:, name2idx[w]].astype(float))
                cond.append({"w": w, "partial_r": round(pr, 4), "p": round(pc, 4),
                             "delta": round(_delta(pr), 4)})
            collider = [cd for cd in cond if float(cd["delta"]) > _thr]
            return {"collider": collider, "cond_cand_n": len(cand_all),
                    "cond_checked_n": len(cond),
                    "cond_max_cond": max_cond,
                    "cond_coverage": round(len(cond) / min(len(cand_all), max_cond), 4) if cand_all else 0.0}
        mi = _mi_xy(xi, xj, su, sv)
        neigh = options.get("neighbors") or {}
        alpha = float(options.get("alpha", 0.05) or 0.05)
        max_cond = int(options.get("max_cond", 2) or 2)
        cand_all = _cond_candidates(data, columns, u, v, xi, xj, su, sv,
                                   discrete, state_map, alpha, neigh)
        cond = []
        for w in cand_all[:max_cond]:
            k = name2idx[w]
            mi_c, _, _ = _cond_mi_g2(xi, xj, data[:, k], su, sv, int(state_map.get(w, data[:, k].max() + 1)))
            cond.append({"w": w, "mi_c": round(mi_c, 4), "p": None,
                         "delta": round(mi_c - mi, 4)})
        collider = [cd for cd in cond if float(cd["delta"]) > 0.005]
        return {"collider": collider, "cond_cand_n": len(cand_all),
                "cond_checked_n": len(cond),
                "cond_max_cond": max_cond,
                "cond_coverage": round(len(cond) / min(len(cand_all), max_cond), 4) if cand_all else 0.0}


@register_stat
class ResidualANM(Statistic):
    """ANM 残差独立性双向（连续方向主证据，v4.5）。"""
    name = "residual_anm"
    kind = "direction"

    def compute(self, data, columns, edge, discrete, options) -> Dict[str, Any]:
        if discrete:
            return {}
        u, v, i, j, xi, xj, su, sv = _edge_ctx(data, columns, edge, discrete, options)
        xi_f, xj_f = xi.astype(float), xj.astype(float)
        return {"resid_u_to_v": round(_resid_dir(xi_f, xj_f), 4),
                "resid_v_to_u": round(_resid_dir(xj_f, xi_f), 4)}
