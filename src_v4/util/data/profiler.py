# -*- coding: utf-8 -*-
"""Deterministic local-data summaries used by algorithm selection."""
from typing import Any, Dict

import numpy as np


class DataProfiler:
    """Summarize local data without exposing raw observations."""

    @staticmethod
    def summarize(data: np.ndarray, discrete: bool,
                  semantics: Dict[str, Any] = None) -> Dict[str, Any]:
        data = np.asarray(data)
        out: Dict[str, Any] = {
            "n": int(data.shape[0]),
            "d": int(data.shape[1]),
            "discrete": bool(discrete),
        }
        if data.shape[0] == 0:
            return out

        arr = data.astype(float)
        out["missing_rate"] = float(np.isnan(arr).mean())
        with np.errstate(all="ignore"):
            means = np.nanmean(arr, axis=0)
            stds = np.nanstd(arr, axis=0)
        out["mean_range"] = [
            round(float(np.nanmin(means)), 3),
            round(float(np.nanmax(means)), 3),
        ]
        out["std_range"] = [
            round(float(np.nanmin(stds)), 3),
            round(float(np.nanmax(stds)), 3),
        ]
        if discrete:
            uniq = [len(np.unique(data[:, k])) for k in range(data.shape[1])]
            out["cardinality"] = {"min": int(min(uniq)), "max": int(max(uniq))}
        else:
            clean = arr[~np.isnan(arr).any(axis=1)] if np.isnan(arr).any() else arr
            if clean.shape[0] >= 2:
                tri = np.abs(np.tril(np.corrcoef(clean, rowvar=False), -1))
                out["abs_corr_max"] = round(
                    float(np.nanmax(tri)) if tri.size else 0.0, 3
                )
            else:
                out["abs_corr_max"] = 0.0
        return out
