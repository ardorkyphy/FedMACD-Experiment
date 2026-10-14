# -*- coding: utf-8 -*-
"""算法选择证据包（src_v4/evidence/select.py）。

This evidence package is used only for local LLM algorithm selection:
- 只用于 LLM 本地选算法；不做全局化，各客户端在本地数据上算自己的。
- Statistics are generic and bounded by O(n*d^2), rather than tailored to a specific graph.
- The server defines the FIELD_DOC template; clients compute and insert local values,
  证据包直接给本地 LLM 分析（无服务端预计算诊断标签）。
- prompt 公正性：证据包只描述数据特征观测，不隐含对任何算法的推荐或否定；不写"若 X 则选 Y"规则。

维度（四组）：
- form：形态（样本/变量数/离散列/唯一值占比/缺失率）
- relation：关联结构（Pearson/Spearman 强度 + 对比对 + 关联密度）
- distribution：分布（标准偏度/偏度显著性/超额峰度）
- scale：规模（n/d^2、离散格厚、状态数）
"""
from typing import Any, Dict, List

import numpy as np
from scipy.stats import kurtosis, rankdata, skew

# ---------------- 纯函数统计量 ----------------

def _pearson_abs_mean(x: np.ndarray) -> float:
    """Pearson 相关绝对值均值（O(n*d^2)，排除对角）。"""
    d = x.shape[1]
    if d < 2:
        return 0.0
    corr = np.corrcoef(x, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0)
    return float(np.mean(np.abs(corr[np.triu_indices(d, 1)])))


def _spearman_abs_mean(x: np.ndarray) -> float:
    """Spearman 秩相关绝对值均值（O(n*d^2)，与 Pearson 同量级）。"""
    d = x.shape[1]
    if d < 2:
        return 0.0
    ranks = np.apply_along_axis(rankdata, 0, x)
    corr = np.corrcoef(ranks, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0)
    return float(np.mean(np.abs(corr[np.triu_indices(d, 1)])))


def _assoc_density(x: np.ndarray, threshold: float = 0.3) -> float:
    """关联密度：|r|>threshold 的成对占比（稀疏/稠密代理，由相关矩阵派生，0 成本）。"""
    d = x.shape[1]
    if d < 2:
        return 0.0
    corr = np.corrcoef(x, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0)
    off = np.abs(corr[np.triu_indices(d, 1)])
    if off.size == 0:
        return 0.0
    return float(np.mean(off > threshold))


def _skew_stats(x: np.ndarray) -> Dict[str, float]:
    """标准偏度均值（带符号）+ 偏度显著性 z（|skew|/sqrt(6/n) 均值）。

    v4.32 修：旧实现 skew_abs_mean=mean(|z^3|) 是绝对三阶矩而非偏度（恒为正、无法区分正负偏、
    且无显著性参照）；此处为标准 Fisher-Pearson 偏度，并给 z 值量化"小样本下是否显著偏态"。
    """
    d = x.shape[1]
    if d == 0:
        return {"skew_mean": 0.0, "skew_z_abs_mean": 0.0}
    sk = skew(x, axis=0, bias=True, nan_policy="omit")
    sk = np.nan_to_num(sk, nan=0.0)
    n = float(x.shape[0])
    se = float(np.sqrt(6.0 / n)) if n > 0 else 1.0
    z_abs = np.abs(sk) / se if se > 0 else np.zeros_like(sk)
    return {"skew_mean": round(float(np.mean(sk)), 4),
            "skew_z_abs_mean": round(float(np.mean(z_abs)), 4)}


def _kurtosis_mean(x: np.ndarray) -> float:
    """超额峰度均值（E[z^4]-3，高斯=0；v4.32 由 E[z^4] 修正为超额口径，基准可读）。"""
    d = x.shape[1]
    if d == 0:
        return 0.0
    k = kurtosis(x, axis=0, fisher=True, bias=True, nan_policy="omit")
    k = np.nan_to_num(k, nan=0.0)
    return round(float(np.mean(k)), 4)


def _missing_rate(x: np.ndarray) -> float:
    """缺失率（NaN 占比；O(n*d)，供 mv_fisherz/mvpc 类缺失场景提示）。"""
    try:
        f = x.astype(float)
    except (TypeError, ValueError):
        return 0.0
    return round(float(np.isnan(f).mean()), 4)


def _n_disc_like(x: np.ndarray, max_unique: int = 10) -> int:
    """近似离散列数：唯一值数 <= max_unique 的列数（连续口径下也可检出离散编码列）。"""
    n = 0
    for j in range(x.shape[1]):
        try:
            if len(np.unique(x[:, j])) <= max_unique:
                n += 1
        except (TypeError, ValueError):
            pass
    return n


# ---------------- 字段解释字典（prompt 渲染用） ----------------

FIELD_DOC: Dict[str, Dict[str, str]] = {
    # form
    "n": {"term": "样本量", "dim": "形态", "meaning": "本地数据行数，决定统计检验功效与算法运行成本"},
    "d": {"term": "变量数", "dim": "形态", "meaning": "本地数据列数，决定算法复杂度可行性（评分法组合搜索、LiNGAM 迭代成本随 d 增长）"},
    "discrete": {"term": "数据口径", "dim": "形态", "meaning": "实验配置指定的离散/连续口径（决定独立性检验族：G2 卡方 vs Fisher-Z）"},
    "n_disc_like": {"term": "近似离散列数", "dim": "形态", "meaning": "唯一值数不超过 10 的列数；连续口径下若 >0 说明数据含离散编码列（序数/类别），检验假设需谨慎"},
    "unique_ratio_mean": {"term": "唯一值占比均值", "dim": "形态", "meaning": "每列唯一值数/行数 的均值；接近 1 为连续取值，极低则接近离散或重复"},
    "missing_rate": {"term": "缺失率", "dim": "形态", "meaning": "NaN 占比；>0 时独立性检验需缺失值兼容版本（如 mv_fisherz）或先处理缺失"},
    # relation
    "pearson_abs_mean": {"term": "Pearson 相关绝对值均值", "dim": "关联结构", "meaning": "线性关联强度；对非线性关联会低估"},
    "spearman_abs_mean": {"term": "Spearman 秩相关绝对值均值", "dim": "关联结构", "meaning": "单调关联强度（含非线性单调）；与 Pearson 对比可识别单调非线性"},
    "spearman_div_pearson": {"term": "Spearman/Pearson 比值", "dim": "关联结构", "meaning": "对比对：≈1 线性关联为主；>1.3 提示单调非线性关联"},
    "assoc_density": {"term": "关联密度", "dim": "关联结构", "meaning": "|r|>0.3 的成对变量占比；稀疏/稠密结构的廉价代理"},
    # distribution
    "skew_mean": {"term": "标准偏度均值", "dim": "分布", "meaning": "分布不对称程度（带符号，正值右偏/负值左偏）；0 为对称"},
    "skew_z_abs_mean": {"term": "偏度显著性 z 均值", "dim": "分布", "meaning": "|偏度|/sqrt(6/n) 均值；z>=2 表示偏离对称在样本量下显著"},
    "exkurt_mean": {"term": "超额峰度均值", "dim": "分布", "meaning": "峰度-3 的均值；0=高斯、>1 重尾、<-0.5 轻尾（均匀/双峰类）"},
    # scale
    "n_div_d2": {"term": "样本量/变量数平方", "dim": "规模", "meaning": "样本相对维度充足度；<10 紧张（检验功效不足）、10~50 中等、>50 充裕"},
    "cell_est": {"term": "列联表每格估计计数", "dim": "规模", "meaning": "离散口径 n/平均状态数^3；<20 薄格（G2 检验 p 值不可靠）"},
    "state_mean": {"term": "平均状态数", "dim": "规模", "meaning": "离散列平均取值个数；反映离散粒度"},
    "state_max": {"term": "最大状态数", "dim": "规模", "meaning": "离散列最大取值个数；状态过多+样本不足会稀疏化列联表"},
}

def field_doc_text() -> str:
    """字段解释文本（prompt 渲染用：专业术语 / 判断维度 / 反映的数据特征）。"""
    lines = []
    for name, doc in FIELD_DOC.items():
        lines.append("- %s【%s/维度=%s】：%s" % (name, doc["term"], doc["dim"], doc["meaning"]))
    return "\n".join(lines)


# ---------------- 证据包构造 ----------------

class SelectEvidenceProvider:
    """Build the algorithm-selection package from four inexpensive local statistics."""

    name = "select"

    def compute(self, data: np.ndarray, columns: List[str] = None,
                discrete: bool = True) -> Dict[str, Any]:
        x = np.asarray(data)
        if x.ndim != 2 or x.shape[0] == 0 or x.shape[1] == 0:
            return {"evidence": {}, "error": "数据为空"}
        xf = np.nan_to_num(x.astype(float), nan=0.0, posinf=0.0, neginf=0.0)
        n, d = xf.shape
        raw: Dict[str, Any] = {
            "form": {
                "n": int(n), "d": int(d), "discrete": bool(discrete),
                "n_disc_like": int(_n_disc_like(x)),
                "unique_ratio_mean": round(float(np.mean([len(np.unique(x[:, j])) for j in range(d)]) / n), 4),
                "missing_rate": _missing_rate(x),
            },
            "relation": {
                "pearson_abs_mean": _pearson_abs_mean(xf),
                "spearman_abs_mean": _spearman_abs_mean(xf),
                "assoc_density": _assoc_density(xf),
            },
            "distribution": _skew_stats(xf),
            "scale": {
                "n_div_d2": round(float(n / (d * d)) if d else 0.0, 4),
            },
        }
        raw["distribution"]["exkurt_mean"] = _kurtosis_mean(xf)
        rel = raw["relation"]
        sp = rel["spearman_abs_mean"]
        rel["spearman_div_pearson"] = round(sp / rel["pearson_abs_mean"], 4) if rel["pearson_abs_mean"] > 0 else 1.0
        if discrete:
            n_state = np.array([len(np.unique(x[:, j])) for j in range(d)])
            mean_st = float(n_state.mean()) if d else 1.0
            raw["scale"]["cell_est"] = round(float(n / (mean_st ** 3)), 4)
            raw["scale"]["state_mean"] = round(mean_st, 4)
            raw["scale"]["state_max"] = int(n_state.max()) if d else 0
        return {"evidence": raw}
