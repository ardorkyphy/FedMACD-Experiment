# -*- coding: utf-8 -*-
"""论文十个数据集的可迁移路径注册。"""
import os
from typing import Dict, Optional

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
PUBLIC_DATA_ROOT = os.environ.get("FEDMACD_DATA_DIR", "").strip()

PUBLIC_DATA_PATHS: Dict[str, Dict[str, Optional[str]]] = {
    name: {"data": os.path.join("bayesian_network_structures", f"{name}.bif"), "graph": None}
    for name in ("asia", "child", "alarm", "survey", "earthquake")
}
PUBLIC_DATA_PATHS.update({
    "ER20": {"data": os.path.join("ER20", "data.csv"),
             "graph": os.path.join("ER20", "graph.txt")},
    "10nodes": {"data": os.path.join("10nodes", "data.csv"),
                "graph": os.path.join("10nodes", "graph.txt")},
    "Auto_MPG": {"data": os.path.join("Auto_MPG", "data.csv"),
                 "graph": os.path.join("Auto_MPG", "graph.txt")},
    "DWD_climate": {"data": os.path.join("DWD_climate", "data.csv"),
                    "graph": os.path.join("DWD_climate", "graph.txt")},
    "Sachs": {"data": os.path.join("Sachs", "data.csv"),
              "graph": os.path.join("Sachs", "graph.txt")},
})

# Tests and callers may register an explicit local fixture or reconstructed table.
DATASET_PATHS: Dict[str, Dict[str, Optional[str]]] = {}


def resolve_path(name: str) -> Dict[str, Optional[str]]:
    """数据集名 → 绝对路径（data/graph）；未知数据集抛 KeyError。"""
    if name in DATASET_PATHS:
        custom = DATASET_PATHS[name]
        return {
            key: (os.path.abspath(value) if os.path.isabs(value) else os.path.join(PROJECT_ROOT, value))
            if value else None for key, value in custom.items()
        }
    info = PUBLIC_DATA_PATHS.get(name)
    if info is None:
        raise KeyError(f"Unknown dataset: {name}")
    root = PUBLIC_DATA_ROOT or os.path.join(os.path.dirname(PROJECT_ROOT), "data")
    return {"data": os.path.join(root, info["data"]),
            "graph": os.path.join(root, info["graph"]) if info.get("graph") else None}
