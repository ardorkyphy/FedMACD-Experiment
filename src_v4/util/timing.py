# -*- coding: utf-8 -*-
"""计时基础设施（从 src/util/timing.py 复刻，+timer() 便捷方法）。

记录协议各阶段 / LLM 调用等耗时，供"运行时间 / 通信效率"指标使用。
"""
import time
from typing import Any, Dict, List


class PhaseTimings:
    """阶段耗时收集器：phase -> 每次耗时列表。"""

    def __init__(self) -> None:
        self._records: Dict[str, List[float]] = {}

    def add(self, phase: str, seconds: float) -> None:
        self._records.setdefault(phase, []).append(seconds)

    def summary(self) -> Dict[str, Dict[str, float]]:
        out = {}
        for phase, vals in self._records.items():
            out[phase] = {"count": len(vals), "total": sum(vals),
                          "mean": sum(vals) / len(vals)}
        return out

    def timer(self, phase: str) -> "Timer":
        """便捷：with timings.timer(phase): ..."""
        return Timer(phase, self)


class Timer:
    """上下文管理器：自动把耗时记入 collector。"""

    def __init__(self, phase: str, collector: PhaseTimings) -> None:
        self.phase = phase
        self.collector = collector

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.collector.add(self.phase, time.perf_counter() - self._start)