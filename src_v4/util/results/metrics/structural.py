# -*- coding: utf-8 -*-
"""结构指标（从 src/util/results/metrics/structural.py 复刻）。"""
from typing import Tuple

from util.results.base import Metric, ResultContext
from util.results.registry import register_metric


def _confusion(ctx: ResultContext) -> Tuple[int, int, int]:
    est, true = ctx.est_edges, ctx.true_edges
    tp = len(est & true)
    fp = len(est - true)
    fn = len(true - est)
    return tp, fp, fn


@register_metric
class TpMetric(Metric):
    name = "tp"

    def compute(self, ctx: ResultContext) -> float:
        return float(_confusion(ctx)[0])


@register_metric
class FpMetric(Metric):
    name = "fp"

    def compute(self, ctx: ResultContext) -> float:
        return float(_confusion(ctx)[1])


@register_metric
class FnMetric(Metric):
    name = "fn"

    def compute(self, ctx: ResultContext) -> float:
        return float(_confusion(ctx)[2])


@register_metric
class ShdMetric(Metric):
    name = "shd"

    def compute(self, ctx: ResultContext) -> float:
        tp, fp, fn = _confusion(ctx)
        return float(fp + fn)


@register_metric
class PrecisionMetric(Metric):
    name = "precision"

    def compute(self, ctx: ResultContext) -> float:
        tp, fp, _ = _confusion(ctx)
        return float(tp / (tp + fp)) if tp + fp else 0.0


@register_metric
class RecallMetric(Metric):
    name = "recall"

    def compute(self, ctx: ResultContext) -> float:
        tp, _, fn = _confusion(ctx)
        return float(tp / (tp + fn)) if tp + fn else 0.0


@register_metric
class F1Metric(Metric):
    name = "f1"

    def compute(self, ctx: ResultContext) -> float:
        p = PrecisionMetric().compute(ctx)
        r = RecallMetric().compute(ctx)
        return float(2 * p * r / (p + r)) if p + r else 0.0


@register_metric
class NShdMetric(Metric):
    name = "nshd"

    def compute(self, ctx: ResultContext) -> float:
        shd = ShdMetric().compute(ctx)
        return float(shd / max(ctx.n_nodes * (ctx.n_nodes - 1) / 2, 1))