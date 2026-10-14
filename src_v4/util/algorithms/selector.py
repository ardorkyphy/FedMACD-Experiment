# -*- coding: utf-8 -*-
"""算法选择器（从 src/util/algorithms/selector.py 复刻）。

定位：本地 LLM 的"工具调用"层——候选算法池描述成工具（schema），LLM 一次推理
只选算法 {algorithm, reason}；manual 强制覆盖；失败回落规则。
参数一律由 config.ALGORITHM_PARAMS 手动控制，LLM 输出带 params 也忽略。
"""
import json
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from util.algorithms.registry import AlgorithmRegistry, registry


@dataclass
class AlgorithmChoice:
    """选算法结果。"""
    name: str
    params: Dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    source: str = "llm"        # llm / manual / rule
    assessment: Dict[str, Any] = field(default_factory=dict)


class AlgorithmSelector:
    """算法选择策略：llm（默认）/ manual / rule 兜底。"""

    def __init__(self, config: Dict[str, Any], reg: Optional[AlgorithmRegistry] = None,
                 llm=None, field_doc: str = "") -> None:
        self.config = config or {}
        self.registry = reg or registry
        self.llm = llm
        self.field_doc = field_doc or ""   # 服务端对齐阶段下发的算法选择字段解释（v4.33）

    def select(self, evidence: Dict[str, Any], semantics: Dict[str, Any],
               discrete: bool) -> AlgorithmChoice:
        override = str(self.config.get("ALGORITHM_OVERRIDE", "") or "").strip().lower()
        if override and self.registry.get(override) is not None:
            return AlgorithmChoice(name=override, reason="ALGORITHM_OVERRIDE 强制覆盖",
                                   source="manual")
        mode = str(self.config.get("ALGORITHM_SELECTION", "llm") or "llm").strip().lower()
        try:
            if mode == "manual":
                return self._manual(discrete)
            return self._select_via_llm(evidence, semantics, discrete)
        except Exception as exc:
            print(f"[selector] LLM 选算法失败（{exc}），回落规则选择")
            return self._select_rule(discrete)

    def _select_via_llm(self, evidence: Dict[str, Any],
                        semantics: Dict[str, Any], discrete: bool) -> AlgorithmChoice:
        tpl = self._prompt("select_algorithm")
        if not tpl:
            raise RuntimeError("select_algorithm prompt 缺失")
        text = self._build_prompt(tpl, evidence, semantics, discrete)
        if self.llm is None:
            raise RuntimeError("AlgorithmSelector.llm 未注入")
        out = self.llm.chat_json(text, role="select",
                                 system_prompt=self._prompt("privacy_guard") or None)
        name = str(out.get("algorithm", "")).strip().lower()
        cands = self.registry.candidates(discrete)
        if name not in cands:
            raise ValueError(f"LLM 选择 {name!r} 不在候选池（{cands}）")
        assessment = out.get("assessment")
        if not isinstance(assessment, dict):
            assessment = {}
        return AlgorithmChoice(name=name, params={},
                               reason=str(out.get("reason", "")), source="llm",
                               assessment=assessment)

    def _build_prompt(self, tpl: str, evidence: Dict[str, Any],
                      semantics: Dict[str, Any], discrete: bool) -> str:
        """Build the evidence, field documentation, semantics, and algorithm pool prompt."""
        # The template consumes the textual pool and returns JSON.
        from evidence.select import field_doc_text
        fd = self.field_doc or field_doc_text()
        ev_text = json.dumps(evidence or {}, ensure_ascii=False, indent=1)
        sem = json.dumps(semantics or {}, ensure_ascii=False) or "（无）"
        try:
            text = tpl.format(evidence=ev_text, field_doc=fd,
                              semantics=sem, pool=self.registry.pool_text(discrete))
        except (KeyError, ValueError):
            text = (tpl.replace("{evidence}", ev_text).replace("{field_doc}", fd)
                    .replace("{semantics}", sem).replace("{pool}", self.registry.pool_text(discrete)))
        return text

    def _prompt(self, key: str) -> str:
        try:
            from config.prompt import PROMPTS
            return str(PROMPTS.get(key, "") or "")
        except Exception:
            return ""

    def _select_rule(self, discrete: bool) -> AlgorithmChoice:
        return AlgorithmChoice(name=self.registry.default_for(discrete),
                               reason="rule fallback", source="rule")

    def _manual(self, discrete: bool) -> AlgorithmChoice:
        name = str(self.config.get("ALGORITHM_OVERRIDE", "") or "").strip().lower()
        if not name or self.registry.get(name) is None:
            name = self.registry.default_for(discrete)
        return AlgorithmChoice(name=name, reason="manual override", source="manual")
