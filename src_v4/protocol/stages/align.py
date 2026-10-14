# -*- coding: utf-8 -*-
"""Server-side variable alignment.

identity：全局变量=数据列名（DataService 已统一），var_mapping 恒等映射；
ALIGN_MODE=llm（近义词映射）预留未实现（真实场景不同客户端 LLM/算法口径差异在此设计）。
"""
from protocol.base import Stage
from protocol.context import ProtocolContext
from protocol.pipeline import register_stage


@register_stage
class AlignStage(Stage):
    name = "align"

    def run(self, ctx: ProtocolContext) -> None:
        mode = str(ctx.config.get("ALIGN_MODE", "identity") or "identity").strip().lower()
        if mode == "identity":
            self._align_identity(ctx)
        else:
            raise NotImplementedError(f"ALIGN_MODE={mode!r} 预留未实现（当前仅 identity）")
        self._issue_evidence_pack(ctx)

    def _align_identity(self, ctx: ProtocolContext) -> None:
        columns = list(ctx.loaded.columns or [])
        ctx.global_vars = columns
        ctx.var_mapping = {c.client_id: {name: name for name in columns}
                           for c in ctx.clients}
        self.log(ctx).info("变量对齐完成：%d 变量（identity）", len(columns))

    def _issue_evidence_pack(self, ctx: ProtocolContext) -> None:
        """Distribute aligned semantics and evidence-field documentation.

        语义（config.DATA.semantics，数据预处理阶段定稿）+ 算法选择/存在性/方向性
        证据包字段解释（FIELD_DOC 服务端定稿）在对齐阶段生成一份，写 ctx 并下发各客户端；
        客户端本地只算统计值填入，prompt 渲染一律用这份下发文本（不再各自 import 生成）。
        """
        from evidence.select import field_doc_text as _sel_fd
        from evidence.existence import field_doc_text as _ex_fd
        from evidence.direction import field_doc_text as _dir_fd
        from evidence.aggregate import agg_field_doc_text as _agg_fd
        pack = {
            "semantics": dict((ctx.config.get("DATA") or {}).get("semantics", {}) or {}),
            "field_docs": {
                "select": _sel_fd(),
                "existence": _ex_fd(),
                "direction": _dir_fd(),
                "aggregate": _agg_fd(),
            },
        }
        ctx.evidence_pack = pack
        for c in ctx.clients:
            if hasattr(c, "receive_evidence_pack"):
                c.receive_evidence_pack(pack)
        self.log(ctx).info("证据包统一发放：语义 %d 条，字段解释 %d 套",
                           len(pack["semantics"]), len(pack["field_docs"]))
