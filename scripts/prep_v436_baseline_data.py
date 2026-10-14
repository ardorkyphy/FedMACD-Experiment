# -*- coding: utf-8 -*-
"""Prepare the v4.36 baseline inputs for all paper datasets.

Outputs paper-aligned baseline inputs under ``generated/baseline_inputs`` by default.
  10 数据集（asia/child/alarm/survey/earthquake/Auto_MPG/DWD_climate/Sachs/10nodes/ER20）× hom/het × seed 0/1/2，4 客户端。
Construction uses the current ``src_v4`` DataService, including independent
soft-intervention sampling for BIF data and preference-region resampling for CSV data.
For historical fidelity, homogeneous baseline folds split one seed-0 sample;
heterogeneous BIF folds use their own intervention and sampling seed.
文件格式对齐 .m runner：<ds>_<n>.txt（全量）、<ds>_graph.txt（row->col 0/1 邻接）、client_1..4.txt。
用法: python scripts/prep_v436_baseline_data.py [--smoke]
"""
import argparse
import os
import sys

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_V4_ROOT = os.path.join(PROJECT_ROOT, "src_v4")
for _p in (PROJECT_ROOT, SRC_V4_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from util.data.service import DataService  # noqa: E402
from util.data.splitters.random_split import RandomSplitter  # noqa: E402
from util.data.splitters.heterogeneous_split import (  # noqa: E402
    split_interventional, split_resample,
)

OUT_ROOT = os.environ.get(
    "FEDMACD_BASELINE_OUTPUT_DIR",
    os.path.join(PROJECT_ROOT, "generated", "baseline_inputs"),
)
SEEDS = [0, 1, 2]
CLIENTS = 4
STRENGTH = 0.7

# (name, sample_size, discrete)
DATASETS = [
    ("asia", 10000, True), ("child", 10000, True), ("alarm", 10000, True),
    ("survey", 10000, True), ("earthquake", 10000, True),
    ("Auto_MPG", 392, False), ("DWD_climate", 349, False), ("Sachs", 7466, False),
    ("10nodes", 5000, False), ("ER20", 256, False),
]


def _cfg(name, sample_size, seed, scenario, discrete):
    return {
        "name": name, "sample_size": sample_size, "num_clients": CLIENTS,
        "seed": seed, "split": scenario, "sampling_seed": -1, "discrete": discrete,
        "per_client_size": 0, "variables": [], "format": "",
        "heterogeneity_strength": STRENGTH,
    }


def _save(ds, scen, seed, full, parts, edges, discrete):
    d = os.path.join(OUT_ROOT, f"rep5_{scen}", ds, f"s{seed}")
    os.makedirs(d, exist_ok=True)
    n = full.shape[0]
    np.savetxt(os.path.join(d, f"{ds}_{n}.txt"), full, fmt="%d" if discrete else "%.6f")
    adj = np.zeros((full.shape[1], full.shape[1]), dtype=int)
    for i, j in sorted(edges):
        adj[i, j] = 1
    np.savetxt(os.path.join(d, f"{ds}_graph.txt"), adj, fmt="%d")
    for c, part in enumerate(parts, start=1):
        np.savetxt(os.path.join(d, f"client_{c}.txt"), np.asarray(part),
                   fmt="%d" if discrete else "%.6f")
    uniq = [len(np.unique(np.asarray(p), axis=0)) / len(p) for p in parts]
    print(f"  {ds} {scen} s{seed}: n={n} vars={full.shape[1]} edges={len(edges)} "
          f"唯一行率={['%.2f' % u for u in uniq]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="只跑 asia+survey hom/het s0")
    args = ap.parse_args()
    svc = DataService()
    items = DATASETS[:1] + DATASETS[3:4] if args.smoke else DATASETS
    for name, sample_size, discrete in items:
        loaded = svc.load(_cfg(name, sample_size, 0, "homogeneous", discrete))
        full = np.asarray(loaded.data)
        edges = set(loaded.true_edges)
        for k in ([0] if args.smoke else SEEDS):
            parts_hom = RandomSplitter().split(full, CLIENTS, k)
            _save(name, "homogeneous", k, full, parts_hom, edges, discrete)
            cfg = _cfg(name, sample_size, k, "heterogeneous", discrete)
            parts_het = (split_interventional(CLIENTS, k, cfg) if discrete
                         else split_resample(full, CLIENTS, k, cfg))
            _save(name, "heterogeneous", k, full, parts_het, edges, discrete)
    print("prep_v436_baseline_data done ->", OUT_ROOT)


if __name__ == "__main__":
    main()
