# -*- coding: utf-8 -*-
"""统一线程池执行器（从 src/util/execution.py 复刻）。

并行策略：整个代码统一一套线程池，客户端级任务并行，最多 max_workers（默认 4）；
EXECUTION.parallel=False 时完全串行（复现/调试）。
"""
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable, Iterable, List, Optional


class TaskExecutor:
    """线程池门面：submit/run_all 两个入口，run_all 保持输入顺序返回结果。"""

    def __init__(self, max_workers: int = 4, enabled: bool = True) -> None:
        self.max_workers = max(1, int(max_workers))
        self.enabled = bool(enabled)
        self._pool: Optional[ThreadPoolExecutor] = None

    def _pool_or_none(self) -> Optional[ThreadPoolExecutor]:
        if not self.enabled or self.max_workers == 1:
            return None
        if self._pool is None:
            self._pool = ThreadPoolExecutor(max_workers=self.max_workers)
        return self._pool

    def submit(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Future:
        pool = self._pool_or_none()
        if pool is None:
            fut: Future = Future()
            try:
                fut.set_result(fn(*args, **kwargs))
            except BaseException as exc:
                fut.set_exception(exc)
            return fut
        return pool.submit(fn, *args, **kwargs)

    def run_all(self, tasks: Iterable[Callable[[], Any]]) -> List[Any]:
        futs = [self.submit(t) for t in tasks]
        return [f.result() for f in futs]

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None

    def __enter__(self) -> "TaskExecutor":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()