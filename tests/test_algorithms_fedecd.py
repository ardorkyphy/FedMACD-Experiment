# -*- coding: utf-8 -*-
"""fedecd_local 测试：bootstrap 投票聚合 + 完整 discover（PC+B 折+多数方向）。"""
import os
import sys
import unittest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (PROJECT_ROOT, os.path.join(PROJECT_ROOT, "src_v4")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from util.algorithms.edge import Edge  # noqa: E402
from util.algorithms.fedecd import aggregate_direction_votes  # noqa: E402
from tests._shared import base_config, gen_discrete_chain  # noqa: E402
import util.algorithms as A  # noqa: E402


class TestAggregateVotes(unittest.TestCase):
    def test_majority_direction(self):
        votes = {(0, 1): {"->": 10, "<-": 2}, (1, 2): {"->": 3, "<-": 9}}
        seen = {(0, 1), (1, 2), (0, 2)}
        out = aggregate_direction_votes(votes, seen, 3)
        self.assertIn(Edge(0, 1, "->"), out)
        self.assertIn(Edge(1, 2, "<-"), out)

    def test_tie_prefers_high_index_as_cause(self):
        """dag_aggre.m：>= 平局取高索引为因（(0,1) 平局 → 1→0 = "<-"）。"""
        votes = {(0, 1): {"->": 4, "<-": 4}}
        out = aggregate_direction_votes(votes, {(0, 1)}, 2)
        self.assertEqual(out, [Edge(0, 1, "<-")])

    def test_only_none_edges(self):
        votes = {}
        out = aggregate_direction_votes(votes, {(0, 1)}, 2)
        self.assertEqual(out, [Edge(0, 1, "none")])

    def test_absent_pair_skipped(self):
        out = aggregate_direction_votes({}, set(), 3)
        self.assertEqual(out, [])

    def test_min_votes_threshold(self):
        """出现次数不足 min_votes 的边直接丢弃（bagging 门槛）。"""
        votes = {(0, 1): {"->": 6, "<-": 4}, (1, 2): {"->": 1, "<-": 1}}
        out = aggregate_direction_votes(votes, {(0, 1), (1, 2)}, 3, min_votes=6)
        self.assertIn(Edge(0, 1, "->"), out)
        self.assertNotIn(Edge(1, 2, "->"), out)
        self.assertNotIn(Edge(1, 2, "<-"), out)


class TestFedEcdDiscover(unittest.TestCase):
    def test_runs_with_votes(self):
        data = gen_discrete_chain(200)
        alg = A.registry.create("fedecd_local", base_config({"B": 5}))
        edges = alg.discover(data, discrete=True, columns=None)
        self.assertIsInstance(edges, list)
        for e in edges:
            self.assertIn(e.direction, ("->", "<-", "none"))
            self.assertLess(e.u, e.v)

    def test_all_folds_fail_raises(self):
        """全部折失败（如空数据）应明确报错而非静默返回空。"""
        import numpy as np
        alg = A.registry.create("fedecd_local", base_config({"B": 3}))
        with self.assertRaises(RuntimeError):
            alg.discover(np.zeros((0, 3)), discrete=True, columns=None)


if __name__ == "__main__":
    unittest.main()
