# -*- coding: utf-8 -*-
"""Client resources and local causal-discovery capabilities.

资源：config / data（本地分片）/ columns / discrete / llm（本地模型）/ prompts。
能力方法（被 protocol/stages 编排调用）：
    select_algorithm            本地 LLM 选算法（AlgorithmSelector）→ LocalDiscoveryStage
    do_causal_algorithm         本地出图 → 无向骨架 + alg_direction 留档 → LocalDiscoveryStage
    compute_existence_evidence  本地骨架边统计量（evidence.existence provider）→ ExistenceEvidenceStage
    analyze(kind)               本地 LLM 逐边分析（三件套 {answer, score, reason}，v4.9）
    supplement_existence_evidence 候选集缺边定向补算独立性证据（方案 B，只统计不调 LLM）
    compute_direction_evidence  全部候选边统计方向证据（evidence.direction provider）→ DirectionEvidenceStage
    has_skeleton_edge / upload_evidence / save
隐私：只上传证据（统计量/LLM 判断），不传原始数据；所有 LLM 调用注入 privacy_guard。
"""
import json
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

# 客户端 LLM 分析批大小（每条边一行，一次调用多条；默认值内聚代码，不进 config）
ANALYSIS_BATCH_EDGES: int = 20


class Client:
    """客户端资源容器 + 本地能力方法（流程调用方见各方法 docstring）。"""

    def __init__(self, client_id: str, config: Dict[str, Any] = None,
                 data_part: Optional[np.ndarray] = None,
                 columns: Optional[List[str]] = None, discrete: bool = True,
                 llm=None) -> None:
        self.client_id: str = client_id
        self.config: Dict[str, Any] = config or {}
        self.data: Optional[np.ndarray] = np.asarray(data_part) if data_part is not None else None
        self.columns: Optional[List[str]] = list(columns) if columns else None
        self.discrete: bool = bool(discrete)
        self.prompts: Dict[str, str] = self._load_prompts()

        # ---- 本地产物（上传/审计） ----
        self.algorithm: str = ""                 # 本次本地出图所用算法名
        self.select_reason: str = ""
        self.select_source: str = ""             # llm / manual / rule
        self.select_assessment: Dict = {}
        self.local_edges: List[Tuple[str, str, str]] = []      # (u, v, direction)，u<v
        self.skeleton_edges: Set[Tuple[str, str]] = set()      # 无向骨架（Phase1 抹方向）
        self.edge_directions: Dict[Tuple[str, str], str] = {}  # (u,v) -> "->"/"<-"/"none"（Phase2 用）
        self.existence_evidence: Dict[Tuple[str, str], Any] = {}   # edge -> ExistenceEvidence
        self.direction_evidence: Dict[Tuple[str, str], Any] = {}   # edge -> DirectionEvidence
        self.llm_existence: Dict[Tuple[str, str], Any] = {}        # edge -> LLMJudge
        self.llm_direction: Dict[Tuple[str, str], Any] = {}        # edge -> LLMJudge
        self.candidate_set: List[Tuple[str, str]] = []             # 服务端下发（Phase2）
        self.evidence_pack: Dict[str, Any] = {}                  # Server-aligned semantics and field documentation.
        self.field_docs: Dict[str, str] = {}                     # 下发的字段解释文本（select/existence/direction）

        self._llm: Any = llm
        self._log = None

    # ---------------- 资源 ----------------

    @staticmethod
    def _load_prompts() -> Dict[str, str]:
        try:
            from config.prompt import PROMPTS
            return dict(PROMPTS)
        except Exception:
            return {}

    @property
    def llm(self):
        """本地 LLM（util/LLM.py，按角色取配置：select / client_existence / client_direction）。"""
        if self._llm is None:
            from util.LLM import LLM
            self._llm = LLM(self.config)
        return self._llm

    def log(self):
        if self._log is None:
            from util.log import get
            self._log = get(f"client.{self.client_id}")
        return self._log

    def _seed(self) -> int:
        base = int(self.config.get("SEED", 0) or 0)
        try:
            idx = int(self.client_id.rsplit("_", 1)[-1])
        except (ValueError, IndexError):
            idx = 0
        return base + idx

    def _semantics(self) -> Dict:
        """Variable semantics, preferring the server-aligned evidence package;
        未下发时兜底 config.DATA.semantics；不传原始数据）。"""
        pack_sem = (self.evidence_pack or {}).get("semantics")
        if isinstance(pack_sem, dict) and pack_sem:
            return dict(pack_sem)
        return dict(self.config.get("DATA", {}).get("semantics", {}) or {})

    def receive_evidence_pack(self, pack: Dict[str, Any]) -> None:
        """服务端变量对齐阶段统一下发的证据包（v4.33）。

        pack = {"semantics": {...}, "field_docs": {select/existence/direction/aggregate}};
        客户端本地只算统计值填入，prompt 渲染一律用这份下发文本。
        """
        self.evidence_pack = dict(pack or {})
        self.field_docs = dict((pack or {}).get("field_docs", {}) or {})

    def _select_evidence(self) -> Dict[str, Any]:
        """算法选择证据包（替代旧 _data_summary，v4.32）。

        见 evidence/select.py：四维（形态/关联结构/分布/规模）廉价通用量，
        字段模板由服务端定稿，客户端本地计算填值，直接给本地 LLM 分析。
        """
        from evidence.select import SelectEvidenceProvider
        if self.data is None:
            return {"evidence": {}}
        return SelectEvidenceProvider().compute(self.data, self.columns, bool(self.discrete))

    # ---------------- 本地能力方法（protocol stages 调用） ----------------

    def select_algorithm(self) -> str:
        """本地 LLM 选算法（AlgorithmSelector；LLM 只选算法不调参，参数走 ALGORITHM_PARAMS）。"""
        from util.algorithms import cl  # noqa: F401  触发注册
        from util.algorithms.registry import registry
        from util.algorithms.selector import AlgorithmSelector
        selector = AlgorithmSelector(self.config, reg=registry, llm=self.llm,
                                    field_doc=self.field_docs.get("select"))
        choice = selector.select(self._select_evidence(), self._semantics(), self.discrete)
        self.algorithm = choice.name
        self.select_reason = choice.reason
        self.select_source = choice.source
        self.select_assessment = dict(choice.assessment or {})
        return choice.name

    def do_causal_algorithm(self) -> List[Tuple[str, str, str]]:
        """本地出图（AlgorithmRunner.run → 无向骨架 + 方向留档）。方向抹掉只做存在性。"""
        from util.algorithms import cl  # noqa: F401  触发注册
        from util.algorithms.edge import edges_to_evidence
        from util.algorithms.factory import AlgorithmRunner
        from util.algorithms.registry import registry
        if self.data is None:
            raise ValueError("客户端未绑定数据（bind_data）")
        if not self.algorithm:
            self.select_algorithm()
        runner = AlgorithmRunner(self.config, reg=registry)
        edges = runner.run(self.algorithm, self.data, self.discrete, self.columns,
                           seed=self._seed())
        self.local_edges = edges_to_evidence(edges, self.columns or [])
        self.skeleton_edges = set()
        self.edge_directions = {}
        idx = {c: i for i, c in enumerate(self.columns or [])}
        for u, v, direction in self.local_edges:
            # key 按本地列索引序小端在前（v3 同约定），方向以索引小端 u→v 为基准（v4.23 修：
            # 之前用字符串序定 key，列名非字典序时算法输出的 ->/<- 会被整体取反）
            if idx.get(u, 0) <= idx.get(v, 0):
                key, d = (u, v), direction
            else:
                key = (v, u)
                d = {"->": "<-", "<-": "->"}.get(direction, direction)
            self.skeleton_edges.add(key)
            # 算法方向四态：-> / <- / undirected（出边但未定向，等价类/双向）/ 缺省=未出边
            self.edge_directions[key] = "undirected" if d == "none" else d
        self.log().info("本地出图：算法=%s 边=%d", self.algorithm, len(self.local_edges))
        return self.local_edges

    def _evidence_options(self) -> Dict[str, Any]:
        """统计量 options：邻居集（本地骨架）+ 状态数 + 检验参数（数据不出本地）。"""
        data = self.data
        columns = self.columns or []
        neigh: Dict[str, set] = {c: set() for c in columns}
        for u, v in self.skeleton_edges:
            neigh.setdefault(u, set()).add(v)
            neigh.setdefault(v, set()).add(u)
        state_map: Dict[str, int] = {}
        if self.discrete and data is not None:
            name2idx = {c: i for i, c in enumerate(columns)}
            for c in columns:
                state_map[c] = int(data[:, name2idx[c]].max()) + 1
        return {
            "neighbors": neigh,
            "state_map": state_map,
            "max_cond": int(self.config.get("MAX_COND", 2) or 2),
            "alpha": float(self.config.get("ALPHA", 0.05) or 0.05),
        }

    def compute_existence_evidence(self) -> None:
        """对本地骨架上的边：统计量（evidence.existence provider）→ ExistenceEvidence。"""
        if self.data is None:
            raise ValueError("客户端未绑定数据")
        from evidence.base import provider_registry
        from evidence.existence import ExistenceEvidenceProvider  # noqa: F401  触发注册
        provider = provider_registry.get("existence")()
        edges = sorted(self.skeleton_edges)
        options = self._evidence_options()
        fields = provider.compute(self.data, self.columns or [], edges,
                                  self.discrete, options)
        self.existence_evidence = self._to_dataclass(fields, "ExistenceEvidence")
        for ev in self.existence_evidence.values():
            ev.skeleton_has = True                       # 骨架边支持率标记（v4.21 修）
        self.log().info("存在性证据：%d 条边", len(self.existence_evidence))

    def compute_direction_evidence(self) -> None:
        """对候选集全部边：统计方向证据（evidence.direction provider）+ alg_direction。"""
        if self.data is None:
            raise ValueError("客户端未绑定数据")
        if not self.candidate_set:
            self.direction_evidence = {}          # Phase1 全裁 → 空候选集正常终止（v4.25 修：不再回退骨架白算）
            return
        from evidence.base import provider_registry
        from evidence.direction import DirectionEvidenceProvider  # noqa: F401  触发注册
        provider = provider_registry.get("direction")()
        edges = list(self.candidate_set)
        options = self._evidence_options()
        options["alg_direction"] = dict(self.edge_directions)
        fields = provider.compute(self.data, self.columns or [], edges,
                                  self.discrete, options)
        for fields_ in fields.values():
            fields_["alg_direction"] = fields_.get("alg_direction") or ""   # 全矩阵四态，不缺失（v4.22 修）
        self.direction_evidence = self._to_dataclass(fields, "DirectionEvidence")
        self.log().info("方向性证据：%d 条边", len(self.direction_evidence))

    def _to_dataclass(self, fields: Dict[Tuple[str, str], Dict[str, Any]],
                      schema_name: str):
        from dataclasses import fields as _dc_fields
        from config import schema
        cls = getattr(schema, schema_name)
        valid = {f.name for f in _dc_fields(cls)}
        out = {}
        for edge, d in fields.items():
            out[edge] = cls(**{k: v for k, v in d.items() if k in valid})
        return out

    def supplement_existence_evidence(self, missing: List[Tuple[str, str]]) -> None:
        """存在性证据补采（方案 B，v4.3）：候选集缺边只补统计量，不调 LLM。"""
        if not missing or self.data is None:
            return
        from evidence.base import provider_registry
        from evidence.existence import ExistenceEvidenceProvider  # noqa: F401
        provider = provider_registry.get("existence")()
        fields = provider.compute(self.data, self.columns or [], missing,
                                  self.discrete, self._evidence_options())
        for edge, d in fields.items():
            ev = self._to_dataclass({edge: d}, "ExistenceEvidence")[edge]
            ev.skeleton_has = False                      # 补采边非骨架（支持率标记，v4.21 修）
            self.existence_evidence[edge] = ev
        self.log().info("补采 %d 条缺失边证据", len(missing))

    def analyze(self, kind: str, edges: Optional[List[Tuple[str, str]]] = None) -> None:
        """本地 LLM 逐边分析（三件套，v4.9）：kind=existence（exists/none/uncertain）/
        direction（u->v/v->u/related）。模板缺失 → 跳过（调用方按无 LLM 判断处理）。
        edges：显式指定分析边集（absence_supplement 补采用，v4.21 修）；None=默认池。
        Parse failures retry only the uncovered subset by recursive bisection.
        The component ablation skips this step when CLIENT_ANALYZE.enabled=False."""
        if not (self.config.get("CLIENT_ANALYZE") or {}).get("enabled", True):
            tpl_key = "client_existence" if kind == "existence" else "client_direction"
            self.log().info("%s 本地 LLM 分析已关闭（CLIENT_ANALYZE.enabled=False，消融）", tpl_key)
            return
        tpl_key = "client_existence" if kind == "existence" else "client_direction"
        target = self.llm_existence if kind == "existence" else self.llm_direction
        if edges is not None:
            edge_pool = sorted(edges)
        elif kind == "existence":
            edge_pool = sorted(self.skeleton_edges)
        else:
            edge_pool = list(self.candidate_set)
        if not edge_pool:
            return
        tpl = self.prompts.get(tpl_key) or ""
        if not tpl:
            self.log().info("%s 模板缺失，跳过本地 LLM 分析", tpl_key)
            return
        sem = json.dumps(self._semantics(), ensure_ascii=False) or "（无）"
        system = self.prompts.get("privacy_guard") or None
        for start in range(0, len(edge_pool), ANALYSIS_BATCH_EDGES):
            batch = edge_pool[start:start + ANALYSIS_BATCH_EDGES]
            self._analyze_recursive(kind, batch, tpl_key, tpl, sem, system, target)
        self.log().info("%s LLM 分析完成：%d 条", kind, len(target))

    def _analyze_recursive(self, kind: str, chunk: List[Tuple[str, str]], tpl_key: str,
                           tpl: str, sem: str, system: Optional[str],
                           target: Dict[Tuple[str, str], Any]) -> None:
        """单批本地分析 + 拆半兜底（v4.29）：整批失败或个别漏边 → 仅对未覆盖子集
        拆半递归重试；单条仍失败才放弃（该边无本地 LLM 判断，与裁判 v3 兜底一致）。"""
        if not chunk:
            return
        _se = self.config.get("STATISTICAL_EVIDENCE") or {}
        _stats_off = not bool(_se.get("enabled", True))
        stats_block = self._stats_block(chunk, kind, include_stats=not _stats_off)   # 只渲染批内边（v4.19 修）
        # 存在性阶段无向渲染（v4.27 修：箭头会把 LLM 往方向判断带，Phase1 只看存在性）
        _join = " - " if kind == "existence" else "->"
        edges_txt = "\n".join(_join.join(k) for k in chunk)
        if kind == "existence":
            from evidence.existence import field_doc_text
            _fd = self.field_docs.get("existence") or field_doc_text()
            _dir_fd = ""
        else:
            from evidence.direction import field_doc_text as _dir_field_doc_text
            _fd = ""
            _dir_fd = self.field_docs.get("direction") or _dir_field_doc_text()
        if _stats_off:
            # When statistical evidence is disabled, omit its field descriptions too.
            _fd = ""
            _dir_fd = ""
        prompt = (tpl.replace("{edges}", edges_txt)
                  .replace("{stats}", stats_block)
                  .replace("{field_doc}", _fd)
                  .replace("{dir_field_doc}", _dir_fd)
                  .replace("{semantics}", sem))
        try:
            out = self.llm.chat_json(prompt, role=tpl_key, system_prompt=system) or {}
        except Exception as exc:
            self.log().warning("%s 批调用失败（%d 条），拆半重试: %s", tpl_key, len(chunk), exc)
            out = None
        from debate.base import norm_edge_key, norm_undirected_key
        _norm = norm_undirected_key if kind == "existence" else norm_edge_key
        parsed_here: Dict[Tuple[str, str], Any] = {}
        if out:
            by_edge: Dict[str, Dict] = {}
            for it in (out.get("edges", []) or []):
                if isinstance(it, dict) and str(it.get("edge", "") or "").strip():
                    by_edge[_norm(it.get("edge"))] = it
            for edge in chunk:
                # 反序命中（方向阶段）：LLM 以反序真名写边（列表给 "A->B"，LLM 回 "B->A"）
                # 时，其 answer 的 u->v/v->u 相对反序基准，解析前翻基准、解析后翻答案
                # Keep orientation parsing consistent with server-side batch parsing.
                fwd = _norm("->".join(edge))
                rev = _norm("->".join(reversed(edge)))
                rec = by_edge.get(fwd)
                base = edge
                if rec is None and rev != fwd:
                    rec = by_edge.get(rev)
                    base = (edge[1], edge[0])
                parsed = self._parse_triple(rec, kind, base)
                if parsed is not None and kind == "direction" and base != edge:
                    parsed.answer = {"u->v": "v->u", "v->u": "u->v"}.get(parsed.answer, parsed.answer)
                if parsed is not None:
                    parsed_here[edge] = parsed
        for edge, parsed in parsed_here.items():
            target[edge] = parsed
        missed = [e for e in chunk if e not in parsed_here]
        if not missed:
            return
        if len(missed) == 1:
            self.log().warning("%s 单条分析失败（%s），该边无本地 LLM 判断",
                               tpl_key, _join.join(missed[0]))
            return
        half = len(missed) // 2
        self._analyze_recursive(kind, missed[:half], tpl_key, tpl, sem, system, target)
        self._analyze_recursive(kind, missed[half:], tpl_key, tpl, sem, system, target)

    def _stats_block(self, edges: List[Tuple[str, str]], kind: str, include_stats: bool = True) -> str:
        """把本地证据渲染成 prompt 统计块（聚合数值，不传原始数据）。"""
        from dataclasses import asdict
        from evidence.render import fmt_direction_stats, fmt_stats
        pool = self.existence_evidence if kind == "existence" else self.direction_evidence
        lines = []
        for edge in edges:
            d = pool.get(edge)
            if d is None:
                continue
            if kind == "direction":
                lines.append("%s | %s" % ("->".join(edge),
                                          fmt_direction_stats(asdict(d), stats=include_stats)))
            else:
                lines.append("%s | %s" % (" - ".join(edge), fmt_stats(asdict(d), stats=include_stats)))
        return "\n".join(lines) or "（无统计量）"

    def _parse_triple(self, rec: Optional[Dict], kind: str, edge: Tuple[str, str]) -> Optional[Any]:
        """解析三件套 {answer, score, reason}（v4.9）；非法返回 None（调用方降级跳过）。

        方向答案容忍 LLM 直接写真实变量名（相对基准 edge 归一化为 u->v/v->u）。
        """
        if not isinstance(rec, dict):
            return None
        from config.schema import LLMJudge
        from debate.base import normalize_direction
        if kind == "existence":
            ans = str(rec.get("answer", "") or "").strip().lower().replace(" ", "")
            if ans not in ("exists", "none", "uncertain"):
                return None
        else:
            ans = normalize_direction(rec.get("answer"), edge)
            if not ans:
                return None
        try:
            score = float(min(max(float(rec.get("score", 0.0) or 0.0), 0.0), 1.0))
        except (TypeError, ValueError):
            score = 0.0
        return LLMJudge(answer=ans, score=score,
                        reason=str(rec.get("reason", "") or ""))

    def has_skeleton_edge(self, key: Tuple[str, str]) -> bool:
        return key in self.skeleton_edges

    def upload_evidence(self, ctx, only: Optional[Set[Tuple[str, str]]] = None,
                        slot: str = "existence") -> None:
        """上传证据到 ctx.edge_records（每条候选边挂 ClientEvidence；不传原始数据）。

        only：限定上传边集合（Phase2 传 ctx.candidate_set，避免把 Phase1 被裁掉的边
        重新写回 edge_records 造成"复活"；None=全量 union，Phase1 用）。
        slot：existence / direction——决定 ce.llm 取哪个槽；方向阶段绝不回退存在性判断
        （v4.19 修：direction 缺失即置空，防止 exists/none 污染方向投票与辩论包）。
        """
        from config.schema import ClientEvidence
        from protocol.state import EdgeRecord
        keys = sorted(set(self.skeleton_edges) | set(self.existence_evidence)
                      | set(self.direction_evidence) | set(self.llm_existence)
                      | set(self.llm_direction))
        if only is not None:
            keys = [k for k in keys if k in only]
        for key in keys:
            rec = ctx.edge_records.setdefault(key, EdgeRecord(key))
            llm = (self.llm_direction.get(key) if slot == "direction"
                   else self.llm_existence.get(key))
            ce = rec.client_evidence.get(self.client_id)
            if ce is None:
                ce = ClientEvidence(
                    existence=self.existence_evidence.get(key),
                    direction=self.direction_evidence.get(key),
                    llm=llm)
                rec.client_evidence[self.client_id] = ce
            else:
                # 合并更新（v4.25）：Phase2 上传只更新 direction/llm，不整体替换，
                # 保留 Phase1 存在性判断与 llm_existence 审计；新证据非 None 才覆盖
                if self.existence_evidence.get(key) is not None:
                    ce.existence = self.existence_evidence.get(key)
                if self.direction_evidence.get(key) is not None:
                    ce.direction = self.direction_evidence.get(key)
                ce.llm = llm
            if slot == "existence" and self.llm_existence.get(key) is not None:
                ce.llm_existence = self.llm_existence.get(key)

    def save(self, output_root: Optional[str] = None) -> Dict[str, str]:
        """保存本地审计产物（骨架/证据/LLM 判断/选算法），路径由调用方决定。"""
        import os
        if not output_root:
            return {}
        os.makedirs(output_root, exist_ok=True)
        paths: Dict[str, str] = {}
        from dataclasses import asdict
        try:
            p = os.path.join(output_root, f"{self.client_id}_edges.json")
            with open(p, "w", encoding="utf-8") as f:
                json.dump([{"u": u, "v": v, "direction": d} for u, v, d in self.local_edges],
                          f, ensure_ascii=False, indent=2)
            paths["edges"] = p
            for name, pool in (("existence", self.existence_evidence),
                               ("direction", self.direction_evidence),
                               ("llm_existence", self.llm_existence),
                               ("llm_direction", self.llm_direction)):
                p = os.path.join(output_root, f"{self.client_id}_{name}.json")
                with open(p, "w", encoding="utf-8") as f:
                    json.dump({f"{u}->{v}": asdict(val) if hasattr(val, "__dataclass_fields__") else val
                               for (u, v), val in pool.items()},
                              f, ensure_ascii=False, indent=2)
                paths[name] = p
            p = os.path.join(output_root, f"{self.client_id}_select.json")
            with open(p, "w", encoding="utf-8") as f:
                json.dump({"client_id": self.client_id, "algorithm": self.algorithm,
                           "source": self.select_source, "reason": self.select_reason,
                           "assessment": self.select_assessment},
                          f, ensure_ascii=False, indent=2)
            paths["select"] = p
        except OSError as exc:
            self.log().warning("客户端审计保存失败: %s", exc)
        return paths
