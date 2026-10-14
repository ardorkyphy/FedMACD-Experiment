# -*- coding: utf-8 -*-
"""Server-side resource container.

职责：只管理服务端资源（config / prompts / 裁判 LLM），不含流程逻辑。
流程由 protocol/stages 编排，阶段经 ctx.server.llm 调用裁判/辩手模型（角色配置在 config.LLM.roles）。
流程状态统一放 ProtocolContext。
"""
from typing import Any, Dict, List, Optional

from config.schema import FinalGraph  # noqa: F401（schema 契约在此层可见）


class Server:
    """服务端资源容器：config / prompts / llm（辩手与裁判共用，角色内部分流）。"""

    def __init__(self, config: Dict[str, Any] = None, clients: Optional[List] = None,
                 llm: Any = None) -> None:
        self.config = config or {}
        self.clients: List = list(clients or [])
        self.prompts: Dict[str, str] = self._load_prompts()
        self._llm: Any = llm
        self._log = None

    @staticmethod
    def _load_prompts() -> Dict[str, str]:
        try:
            from config.prompt import PROMPTS
            return dict(PROMPTS)
        except Exception:
            return {}

    @property
    def llm(self):
        """服务端 LLM（util/LLM.py；辩手 debater_a/debater_b、裁判 judge 按角色分流）。"""
        if self._llm is None:
            from util.LLM import LLM
            self._llm = LLM(self.config)
        return self._llm

    def log(self):
        """日志（util/log.py，命名 server）。"""
        if self._log is None:
            from util.log import get
            self._log = get("server")
        return self._log
