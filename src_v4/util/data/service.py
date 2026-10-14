# -*- coding: utf-8 -*-
"""数据服务门面（从 src/util/data/service.py 复刻）。

注册表选 loader/splitter，统一编排：load → sample → split → true_edges。
"""
import os
from typing import Any, Dict, List, Optional

from util.data.base import DataLoader, LoadedData, Splitter
from util.data.common import load_graph_edges
from util.data.datasets import PROJECT_ROOT, resolve_path


class DataService:
    """数据加载门面（唯一入口）。"""

    def __init__(self, loaders: Optional[List[DataLoader]] = None,
                 splitters: Optional[List[Splitter]] = None) -> None:
        self._loaders: Dict[str, DataLoader] = {}
        self._splitters: Dict[str, Splitter] = {}
        for loader in (loaders or []):
            self.register_loader(loader)
        for splitter in (splitters or []):
            self.register_splitter(splitter)

    def register_loader(self, loader: DataLoader) -> None:
        self._loaders[loader.name] = loader

    def register_splitter(self, splitter: Splitter) -> None:
        self._splitters[splitter.name] = splitter

    def _ensure_defaults(self) -> None:
        if not self._loaders:
            from util.data.loaders.bif_loader import BifLoader
            from util.data.loaders.csv_loader import CsvLoader
            from util.data.loaders.npy_loader import NpyLoader
            for cls in (BifLoader, CsvLoader, NpyLoader):
                self.register_loader(cls())
        if not self._splitters:
            from util.data.splitters.random_split import RandomSplitter
            from util.data.splitters.heterogeneous_split import HeterogeneousSplitter
            for cls in (RandomSplitter, HeterogeneousSplitter):
                self.register_splitter(cls())

    def load(self, data_cfg: Dict[str, Any]) -> LoadedData:
        """门面：按 DATA 配置加载数据集并完成大小控制、真图、变量子集与分割。"""
        self._ensure_defaults()
        info = resolve_path(str(data_cfg.get("name", "") or ""))
        fmt = str(data_cfg.get("format", "") or "").strip().lower()
        if not fmt:
            fmt = os.path.splitext(info["data"])[1].lstrip(".").lower()
        loader = self._loaders.get(fmt)
        if loader is None:
            raise KeyError(f"未注册数据格式 {fmt!r}（已注册: {sorted(self._loaders)}）")
        data = loader.load(data_cfg)
        columns = loader.columns(data_cfg)

        true_edges = set()
        graph_path = str(data_cfg.get("graph_path") or info.get("graph") or "") or None
        if graph_path:
            if not os.path.isabs(graph_path):
                graph_path = os.path.join(PROJECT_ROOT, graph_path)
            true_edges = load_graph_edges(graph_path, columns)
        if not true_edges:
            true_edges = set(getattr(loader, "bif_edges", set()))

        variables = data_cfg.get("variables") or []
        if variables:
            sel = [columns.index(v) for v in variables]
            data = data[:, sel]
            columns = [columns[i] for i in sel]
            fidx = {old: new for new, old in enumerate(sel)}
            true_edges = {(fidx[i], fidx[j]) for (i, j) in true_edges if i in fidx and j in fidx}

        num_clients = int(data_cfg.get("num_clients", 1) or 1)
        if num_clients > data.shape[0]:
            raise ValueError(
                f"num_clients={num_clients} > 样本量 {data.shape[0]}：会产生空数据分片，"
                "因果发现算法无法运行。请增大 DATA.sample_size 或减小 DATA.num_clients。")
        seed = int(data_cfg.get("seed", 0) or 0)
        split = str(data_cfg.get("split", "") or "").strip().lower()
        split = {"": "random", "homogeneous": "random",
                 "heterogeneous": "heterogeneous"}.get(split, split)
        splitter = self._splitters.get(split)
        if splitter is None:
            raise KeyError(f"未注册分割策略 {split!r}（已注册: {sorted(self._splitters)}）")
        parts = splitter.split(data, num_clients, seed, data_cfg)
        return LoadedData(data=data, parts=parts, columns=columns,
                          discrete=bool(data_cfg.get("discrete", True)),
                          true_edges=true_edges)
