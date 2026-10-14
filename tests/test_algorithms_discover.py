# -*- coding: utf-8 -*-
"""算法 discover 冒烟测试：注册表内每个算法在合适数据上可跑、输出合法 Edge 列表。

- 离散池算法用离散链数据、连续池算法用连续链数据（小样本控时）
- GOLEM 训练 >90s，默认跳过（RUN_SLOW_ALGORITHMS=1 开启；另由 scripts 冒烟覆盖）
- 真库方向抽查：direct/ica/multigroup/notears 在连续链 0→1→2 上应复现方向
  （highdim 小样本（n=800）Lasso 剪枝边检出不稳 5/10，但检出边的方向符号恒正确，
  方向约定由同一转置适配器 + 前述矩阵约定验证，不单独断言）
"""
import os
import sys
import unittest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (PROJECT_ROOT, os.path.join(PROJECT_ROOT, "src_v4")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import util.algorithms as A  # noqa: E402
from tests._shared import base_config, gen_cont_chain, gen_discrete_chain  # noqa: E402

RUN_SLOW = os.environ.get("RUN_SLOW_ALGORITHMS") == "1"
VALID_DIRS = ("->", "<-", "none")


def _run_all():
    """跑注册表全部算法（golem 仅 RUN_SLOW 时实跑），返回 (name, discrete, edges)。"""
    results = []
    disc_data = gen_discrete_chain(250)
    cont_data = gen_cont_chain(250)
    for name in A.registry.names():
        info = A.registry.get(name)
        if not info(params={}, alpha=0.05, seed=0).available():
            continue          # 依赖/子模块缺失的算法不进候选池，也不冒烟（如 castle 缺失的 gcastle 族）
        if name == "golem" and not RUN_SLOW:
            continue
        discrete = info.data_type != "continuous"
        data = disc_data if discrete else cont_data
        extra = {"fedecd_local": {"B": 5}} if name == "fedecd_local" else None
        alg = A.registry.create(name, base_config(extra))
        edges = alg.discover(data, discrete=discrete, columns=None)
        results.append((name, discrete, edges))
    return results


class TestDiscoverSmoke(unittest.TestCase):
    """每个已注册算法在对应数据类型上可跑通且输出合法 Edge 列表。"""

    def test_all_algorithms_run_and_valid_edges(self):
        for name, _, edges in _run_all():
            self.assertIsInstance(edges, list, name)
            for e in edges:
                self.assertIsInstance(e, A.Edge, f"{name} 输出非 Edge")
                self.assertLess(e.u, e.v, f"{name} Edge 未满足 u<v")
                self.assertIn(e.direction, VALID_DIRS, f"{name} 非法方向")

    def test_lingam_family_recovers_chain_direction(self):
        """连续链 0→1→2：LiNGAM 系/notears 应复现方向（防方向约定回归）。"""
        data = gen_cont_chain(800, seed=1)
        for name in ("direct_lingam", "ica_lingam", "multigroup_lingam", "notears"):
            if not A.registry.get(name)(params={}, alpha=0.05, seed=0).available():
                continue      # 缺依赖（如 castle）则跳过
            alg = A.registry.create(name, base_config())
            edges = {(e.u, e.v): e.direction for e in alg.discover(data, False, None)}
            self.assertEqual(edges.get((0, 1)), "->", f"{name} 0→1")
            self.assertEqual(edges.get((1, 2)), "->", f"{name} 1→2")

if __name__ == "__main__":
    unittest.main()
