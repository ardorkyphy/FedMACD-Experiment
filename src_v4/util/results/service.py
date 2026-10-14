# -*- coding: utf-8 -*-
"""结果服务门面（从 src/util/results/service.py 复刻）。"""
import os
from typing import Dict, Iterable, Optional, Tuple

import pandas as pd

from util.results.base import ResultContext
from util.results.registry import MetricRegistry, metric_registry


class ResultService:
    """实验评估门面（run.py 用它计算结构指标 + 动态效率键）。"""

    def __init__(self, registry: Optional[MetricRegistry] = None) -> None:
        self.registry = registry or metric_registry

    def compute(self, est_edges: Iterable[Tuple[int, int]],
                true_edges: Iterable[Tuple[int, int]], n_nodes: int,
                timings: Optional[Dict] = None,
                llm_calls: Optional[Dict] = None,
                metrics: Optional[Dict[str, bool]] = None) -> Dict[str, float]:
        ctx = ResultContext(est_edges=set(est_edges), true_edges=set(true_edges),
                            n_nodes=n_nodes, timings=timings, llm_calls=llm_calls)
        out: Dict[str, float] = self.registry.compute_all(ctx, enabled=metrics)
        if timings:
            for phase, stats in timings.items():
                out[f"time_{phase}"] = float(stats.get("total", 0.0))
        if llm_calls:
            for role, n in llm_calls.items():
                out[f"llm_{role}"] = float(n)
        if metrics:
            out = {k: v for k, v in out.items() if metrics.get(k, True)}
        return out

    def load(self, path: str) -> pd.DataFrame:
        if path.endswith(".json"):
            return pd.read_json(path)
        return pd.read_csv(path)

    def draw(self, results: Dict[str, Dict[str, float]], output_path: str,
             indicators: Tuple[str, ...] = ("shd", "f1", "nshd")) -> None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        names = list(results.keys())
        fig, axes = plt.subplots(1, len(indicators), figsize=(4 * len(indicators), 4), squeeze=False)
        for ax, ind in zip(axes[0], indicators):
            vals = [results[n].get(ind, 0.0) for n in names]
            ax.bar(range(len(names)), vals)
            ax.set_title(ind)
            ax.set_xticks(range(len(names)))
            ax.set_xticklabels(names, rotation=30, ha="right")
        fig.tight_layout()
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        fig.savefig(output_path, dpi=150)
        plt.close(fig)