# -*- coding: utf-8 -*-
"""算法层输出适配器测试（开发规则 10.2：模块代码必须有针对每个功能的测试）。

覆盖 graph_to_edges / fcm_to_edges / edges_to_evidence 的三种方向与边界。
"""
import os
import sys
import unittest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (PROJECT_ROOT, os.path.join(PROJECT_ROOT, "src_v4")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402

from util.algorithms.edge import Edge, edges_to_evidence, fcm_to_edges, graph_to_edges  # noqa: E402


class TestGraphToEdges(unittest.TestCase):
    """causal-learn 邻接矩阵 → Edge 列表。"""

    def test_directed_both_ways(self):
        # g[i][j]：边 (i,j) 在 i 端端点（TAIL=-1/ARROW=1/NULL=0）。0→1、2→1
        g = np.zeros((3, 3), dtype=int)
        g[1][0] = 1; g[0][1] = -1     # 0→1
        g[1][2] = 1; g[2][1] = -1     # 2→1 ⇒ 相对 (1,2) 为 "<-"
        edges = graph_to_edges(g)
        self.assertIn(Edge(0, 1, "->"), edges)
        self.assertIn(Edge(1, 2, "<-"), edges)
        self.assertEqual(len(edges), 2)

    def test_none_and_no_edge(self):
        g = np.zeros((3, 3), dtype=int)
        g[0][1] = 1; g[1][0] = 1      # 双向箭头 → none
        edges = graph_to_edges(g)
        self.assertIn(Edge(0, 1, "none"), edges)
        self.assertEqual(len(edges), 1)   # (1,2)/(0,2) 无边


class TestFcmToEdges(unittest.TestCase):
    """FCM/邻接矩阵（i→j 约定）→ Edge 列表。"""

    def test_directed_and_none(self):
        adj = np.zeros((3, 3), dtype=float)
        adj[0][1] = 0.8               # 0→1
        adj[1][2] = -0.5; adj[2][1] = 0.3   # 双向 → none
        edges = fcm_to_edges(adj)
        self.assertIn(Edge(0, 1, "->"), edges)
        self.assertIn(Edge(1, 2, "none"), edges)
        self.assertEqual(len(edges), 2)


class TestEdgesToEvidence(unittest.TestCase):
    """Edge → variable-name evidence tuples."""

    def test_mapping(self):
        edges = [Edge(0, 1, "->"), Edge(1, 2, "<-")]
        cols = ["A", "B", "C"]
        self.assertEqual(
            edges_to_evidence(edges, cols),
            [("A", "B", "->"), ("B", "C", "<-")],
        )

    def test_empty(self):
        self.assertEqual(edges_to_evidence([], ["A"]), [])


if __name__ == "__main__":
    unittest.main()
