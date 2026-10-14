# -*- coding: utf-8 -*-
"""数据服务测试：BIF 加载/真图/分割（同质+异质）/变量子集/未知格式报错。"""
import os
import sys
import unittest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (PROJECT_ROOT, os.path.join(PROJECT_ROOT, "src_v4")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402

from util.data import DataService, DataProfiler  # noqa: E402
from util.data.datasets import DATASET_PATHS  # noqa: E402


def _child_cfg(**over):
    cfg = {"name": "_test_bif", "discrete": True, "sample_size": 800,
           "num_clients": 4, "split": "homogeneous", "seed": 0}
    cfg.update(over)
    return cfg


class TestDataService(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        DATASET_PATHS["_test_bif"] = {
            "data": os.path.join("tests", "fixtures", "synthetic_chain.bif"),
            "graph": None,
        }
        cls.svc = DataService()

    @classmethod
    def tearDownClass(cls):
        DATASET_PATHS.pop("_test_bif", None)

    def test_bif_load_and_true_edges(self):
        ld = self.svc.load(_child_cfg())
        self.assertEqual(ld.data.shape, (800, 3))
        self.assertEqual(len(ld.columns), 3)
        self.assertGreater(len(ld.true_edges), 0)      # BIF 自带真图
        self.assertEqual(ld.discrete, True)

    def test_homogeneous_split(self):
        ld = self.svc.load(_child_cfg())
        self.assertEqual([p.shape for p in ld.parts], [(200, 3)] * 4)
        # 分片并集与全量一致（随机切块不重复不遗漏）
        cat = np.concatenate(ld.parts)
        self.assertEqual(sorted(map(tuple, cat.tolist())), sorted(map(tuple, ld.data.tolist())))

    def test_heterogeneous_split(self):
        ld = self.svc.load(_child_cfg(split="heterogeneous"))
        self.assertEqual(len(ld.parts), 4)
        self.assertEqual(sum(p.shape[0] for p in ld.parts), 800)

    def test_variables_subset(self):
        ld = self.svc.load(_child_cfg(variables=["BirthAsphyxia", "HypDistrib", "HypoxiaInO2"]))
        self.assertEqual(ld.data.shape, (800, 3))
        self.assertEqual(ld.columns, ["BirthAsphyxia", "HypDistrib", "HypoxiaInO2"])
        for i, j in ld.true_edges:
            self.assertLess(i, 3)
            self.assertLess(j, 3)

    def test_csv_continuous_with_header_is_numeric(self):
        """连续 CSV 带表头：read_csv(header=None) 的表头文本行不得污染 dtype（回归：Auto_MPG 等 object dtype 崩溃）。"""
        import tempfile
        tmp = tempfile.mkdtemp()
        p = os.path.join(tmp, "d.csv")
        with open(p, "w", encoding="utf-8") as f:
            f.write("A,B,C\n1.0,2.0,3.0\n4.0,5.0,6.0\n")
        DATASET_PATHS["_tmp_csv"] = {"data": p, "graph": None}
        try:
            ld = self.svc.load({"name": "_tmp_csv", "format": "csv", "discrete": False,
                                "sample_size": 2, "num_clients": 1, "split": "", "seed": 0})
            self.assertEqual(ld.data.dtype.kind, "f")          # 连续数据必须是数值 dtype
            self.assertEqual(ld.columns, ["A", "B", "C"])
            self.assertEqual(ld.data.shape, (2, 3))
        finally:
            DATASET_PATHS.pop("_tmp_csv", None)


    def test_unknown_format_raises(self):
        with self.assertRaises(KeyError):
            self.svc.load(_child_cfg(format="parquet"))

    def test_unknown_dataset_raises(self):
        with self.assertRaises(KeyError):
            self.svc.load({"name": "no_such_dataset", "sample_size": 10})


class TestDataProfiler(unittest.TestCase):
    def test_summarize_discrete(self):
        data = np.array([[0, 1], [1, 0], [0, 2], [2, 0]], dtype=float)
        out = DataProfiler.summarize(data, discrete=True)
        self.assertEqual(out["n"], 4)
        self.assertEqual(out["d"], 2)
        self.assertTrue(out["discrete"])
        self.assertEqual(out["cardinality"], {"min": 3, "max": 3})

    def test_summarize_continuous(self):
        data = np.array([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]])
        out = DataProfiler.summarize(data, discrete=False)
        self.assertIn("abs_corr_max", out)
        self.assertAlmostEqual(out["abs_corr_max"], 1.0, places=6)

    def test_summarize_empty(self):
        out = DataProfiler.summarize(np.empty((0, 2)), True)
        self.assertEqual(out["n"], 0)


if __name__ == "__main__":
    unittest.main()
