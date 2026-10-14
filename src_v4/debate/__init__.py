# -*- coding: utf-8 -*-
"""Server-side debate and adjudication layer.

- packets：证据包构造（存在包/不存在包、uv 包/vu 包）——辩论的"不同视角"输入。
- base：DebateProtocol / Judge 抽象 + 注册表（debate_protocols / judges）。
- 具体协议/裁判：import 本包即触发 @register_debate / @register_judge 自注册。
"""
from debate import base, packets, existence_debate, direction_debate, judge  # noqa: F401
from debate.base import (  # noqa: F401
    DebateProtocol, Judge, debate_protocols, judges,
    register_debate, register_judge,
)

__all__ = ["DebateProtocol", "Judge", "debate_protocols", "judges",
           "register_debate", "register_judge"]
