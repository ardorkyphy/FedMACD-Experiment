# -*- coding: utf-8 -*-
"""FedMACD protocol data contracts.

原则：schema 由服务端定死，客户端只负责填充。所有 LLM 输出统一三件套
{answer, score, reason}（v4.9）。
"""
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class ConditionalTest:
    """条件独立性检验结果（存在性/方向性的共同邻居检验）。"""
    w: str            # 共同邻居变量名
    mi_c: float       # 条件互信息
    p: float          # 条件 G² p 值（离散）；偏相关 t 检验 p（连续）
    delta: float      # mi_c - mi（>0 → v 结构候选）
    partial_r: Optional[float] = None  # Partial correlation for continuous data.


@dataclass
class ExistenceEvidence:
    """存在性证据矩阵（每客户端 × 每候选边，v4.5）。"""
    skeleton_has: bool = False    # 该客户端骨架是否含此边（支持率统计基础）
    n: int = 0                    # 客户端样本量
    state_u: int = 0              # 状态数（离散）
    state_v: int = 0
    cell_min: float = 0.0         # 每格最小计数（检验可信度）
    mi: float = 0.0               # 边际互信息
    nmi: float = 0.0              # 归一化互信息（跨客户端聚合用）
    g2_p: float = 1.0             # 边际独立性 p 值（离散）
    pearson_r: Optional[float] = None   # Marginal correlation for continuous data.
    pearson_p: Optional[float] = None   # 边际独立性 p 值（连续）
    cond: List[ConditionalTest] = field(default_factory=list)   # 最强候选条件检验（骨架一跳邻居并集）
    explained_away: bool = False  # 条件化后被解释（p_cond>α）→ 全局"被解释率"
    # Existence evidence.
    spearman_r: Optional[float] = None    # Spearman 秩相关（连续，单调非线性关联）
    spearman_p: Optional[float] = None    # Spearman 显著性 p（连续）
    ratio_abs_sp_pearson: Optional[float] = None  # |ρ|/|r|（连续；>1 提示非线性/离群）
    cond_cand_n: int = 0                  # 候选条件集数（骨架一跳邻居并集大小）
    cond_coverage: float = 0.0            # 条件化覆盖度：做了检验的条件集数/候选数（0=无候选）
    partial_all_r: Optional[float] = None  # 半偏相关（连续，控制其余全部变量，接近"直接"）
    partial_all_p: Optional[float] = None  # 半偏相关 t 检验 p
    h_u_given_v: Optional[float] = None   # 条件熵 H(U|V)（离散）
    h_v_given_u: Optional[float] = None   # 条件熵 H(V|U)（离散）
    ent_ratio: Optional[float] = None     # min 熵比率 H(V|U)/H(V) 与 H(U|V)/H(U)（低=确定性/机械决定）
    max_cond_prob: Optional[float] = None # 最大条件概率 max P(V=v|U=u)（高=确定性）
    zero_cell_ratio: Optional[float] = None  # 零格占比（离散，G² 检验可信度）
    skel_path_len: int = -1               # 骨架最短路径边数（1=相邻；>1=间接路径；-1=不连通）
    mid_nodes: List[str] = field(default_factory=list)  # 骨架路径中间节点名（最多 2 个）
    n_common: int = 0                     # 共同邻居数
    n_twohop: int = 0                     # 两跳邻居数


@dataclass
class DirectionEvidence:
    """方向性证据矩阵（每客户端 × 每候选边，第二阶段，v4.5）。"""
    alg_direction: str = ""               # 本地因果算法输出（四态，全矩阵必填）：""=算法未出此边 / -> / <- / undirected=出边但方向未定
    stat_dir: float = 0.5                 # 统计方向证据 0-1（0.5=中性）
    h_u_given_v: Optional[float] = None   # 条件熵（离散）
    h_v_given_u: Optional[float] = None
    collider: List[ConditionalTest] = field(default_factory=list)  # v 结构候选
    resid_u_to_v: Optional[float] = None  # ANM 残差独立性（连续，小者支持 u->v）
    resid_v_to_u: Optional[float] = None
    # Direction evidence.
    cond_cand_n: int = 0                  # v 结构检验候选条件节点数（骨架邻居并集）
    cond_checked_n: int = 0               # v 结构检验已检验候选节点数
    cond_coverage: float = 0.0            # v 结构检验覆盖度：已检验候选数/候选数（0=无候选）
    skel_path_len: int = -1               # 骨架最短路径边数（1=相邻；>1=隔着中间节点；-1=不连通）
    mid_nodes: List[str] = field(default_factory=list)  # 路径中间节点名（最多 2 个）
    n_common: int = 0                     # 共同邻居数
    n_twohop: int = 0                     # 两跳邻居数
    dir_identifiable: bool = False        # 统计方向可辨识性（连续=True；离散双变量不可辨识）


@dataclass
class LLMJudge:
    """LLM decision with answer, score, and concise rationale."""
    answer: str = ""      # 存在性: exists/none/uncertain；方向: u->v/v->u/related
    score: float = 0.0    # 0-1 置信
    reason: str = ""


@dataclass
class ClientEvidence:
    """每客户端对该边的全部证据（v4.5 value 结构）。"""
    existence: Optional[ExistenceEvidence] = None
    direction: Optional[DirectionEvidence] = None
    llm: Optional[LLMJudge] = None            # 最近一次上传阶段的判断（Phase1=存在性 / Phase2=方向）
    llm_existence: Optional[LLMJudge] = None  # Phase1 存在性判断；Phase2 上传不覆盖（记录级审计保留，v4.25）


@dataclass
class AggregateBlock:
    """Global evidence computed after server aggregation."""
    support_rate: float = 0.0         # 骨架含该边客户端占比
    stat_support_rate: float = 0.0    # 边际 p<α 客户端占比
    explained_away_rate: float = 0.0  # 被解释率（p_cond>α 占比）
    fisher_p: float = 1.0             # 联邦 p 值合并（Fisher）
    stouffer_z: float = 0.0           # 联邦 p 值合并（Stouffer）
    dir_split_uv: int = 0             # 方向分裂：uv 票数
    dir_split_vu: int = 0             # 方向分裂：vu 票数
    dir_hetero: float = 0.0           # 跨客户端异质性（统计量方差）
    neutral_rate: float = 0.0         # 方向中立率：无方向信号客户端占比（v4.23）
    stat_dir_global: float = 0.5      # 统计方向聚合强度（非中性 stat_dir 均值；0.5=全局中性）
    alg_arrow_uv: int = 0             # 算法输出 -> 的客户端数（v4.23 算法方向画像）
    alg_arrow_vu: int = 0             # 算法输出 <- 的客户端数
    alg_undirected: int = 0           # 算法出边但方向未定的客户端数
    alg_absent: int = 0               # 算法未出此边的客户端数
    llm_uv: int = 0                   # LLM 方向票 u->v（v4.23，独立于统计方向分裂）
    llm_vu: int = 0                   # LLM 方向票 v->u
    # Aggregated existence evidence.
    cond_coverage_rate: float = 0.0       # 条件化覆盖率：做了条件化检验的客户端占比
    sign_consistency: Optional[float] = None  # 连续 r 符号多数占比（0.5=分裂，1=一致；离散=None）
    nmi_median: Optional[float] = None    # 客户端 NMI 分布中位数
    nmi_iqr: Optional[float] = None       # 客户端 NMI 分布四分位距
    v_struct_rate: float = 0.0            # v 结构支持率：检测到对撞信号（ΔMI>0）的客户端占比（v4.32）


@dataclass
class DebateRound:
    """辩论单轮记录（v4.29 审计补全：逐轮交锋过程可复现）。"""
    round: int = 0
    a_answer: str = ""
    a_conf: float = 0.0
    a_reason: str = ""
    b_answer: str = ""
    b_conf: float = 0.0
    b_reason: str = ""


@dataclass
class DebateOutcome:
    """一场辩论的结果（服务端两辩手）。"""
    stage: str = ""                   # existence / direction
    answer_a: str = ""
    answer_b: str = ""
    conf_a: float = 0.0
    conf_b: float = 0.0
    reason_a: str = ""
    reason_b: str = ""
    agreed: bool = False
    rounds: int = 0
    trace: List[DebateRound] = field(default_factory=list)   # 逐轮过程（v4.29）


@dataclass
class Verdict:
    """裁判终裁（两阶段各一次）。"""
    answer: str = ""                  # 4 类/2 类答案
    confidence: float = 0.0           # 0-1（与评分器同刻度）
    reason: str = ""
    applied: bool = False


@dataclass
class FinalGraph:
    """Final graph with directed and unresolved undirected edges."""
    nodes: List[str] = field(default_factory=list)
    edges: List[Tuple[str, str]] = field(default_factory=list)        # 有向定边
    undirected: List[Tuple[str, str]] = field(default_factory=list)   # 无向保留
