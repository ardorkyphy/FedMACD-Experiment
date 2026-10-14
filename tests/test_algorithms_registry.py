# -*- coding: utf-8 -*-
"""算法注册表测试：注册/候选池过滤/实例化参数合并/种子覆盖。"""
import os
import sys
import unittest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (PROJECT_ROOT, os.path.join(PROJECT_ROOT, "src_v4")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import util.algorithms as A  # noqa: E402


class TestRegistry(unittest.TestCase):
    def test_all_registered(self):
        expected = {
            "anm", "cdnod", "direct_lingam", "fci", "fedecd_local",
            "ges", "golem", "grasp", "hc", "hc_k2",
            "highdim_lingam", "ica_lingam", "multigroup_lingam",
            "notears", "pc", "pgmpy_pc", "rcd",
        }
        self.assertEqual(set(A.registry.names()), expected)

    def test_candidates_discrete(self):
        cands = A.registry.candidates(True)
        self.assertIn("pc", cands)
        self.assertNotIn("direct_lingam", cands)   # 连续算法不进离散池
        self.assertIn("hc_k2", cands)               # 仅离散算法进离散池
        for n in cands:
            dt = A.registry.get(n).data_type
            self.assertTrue(dt in ("any", "discrete"))

    def test_candidates_continuous(self):
        cands = A.registry.candidates(False)
        self.assertIn("direct_lingam", cands)
        self.assertNotIn("hc_k2", cands)           # 仅离散算法不进连续池
        self.assertIn("grasp", cands)              # 修好 score_func 后保留在池内

    def test_default_falls_to_pc(self):
        self.assertEqual(A.registry.default_for(True), "pc")
        self.assertEqual(A.registry.default_for(False), "pc")

    def test_create_merges_params_and_seed(self):
        cfg = {"ALGORITHM_PARAMS": {"pc": {"max_k": 2}}, "ALPHA": 0.01, "SEED": 7}
        alg = A.registry.create("pc", cfg, extra_params={"max_k": 5}, seed=99)
        self.assertEqual(alg.params["max_k"], 2)          # ALGORITHM_PARAMS 强制覆盖 extra
        self.assertEqual(alg.params["indep_test"], "auto")
        self.assertTrue(alg.params["stable"])
        self.assertAlmostEqual(alg.alpha, 0.01)
        self.assertEqual(alg.seed, 99)

    def test_create_extra_params_when_no_override(self):
        cfg = {"ALGORITHM_PARAMS": {}, "ALPHA": 0.05, "SEED": 0}
        alg = A.registry.create("pc", cfg, extra_params={"max_k": 3})
        self.assertEqual(alg.params["max_k"], 3)
        self.assertEqual(alg.params["indep_test"], "auto")
        self.assertTrue(alg.params["stable"])

    def test_create_unknown_raises(self):
        with self.assertRaises(ValueError):
            A.registry.create("not_a_real_algo", {})

    def test_pool_text_contains_description(self):
        txt = A.registry.pool_text(True)
        self.assertIn("hc_k2", txt)
        self.assertIn("约束基", txt)       # 家族/场景信息注入
        self.assertIn("K2 评分", txt)      # hc_k2 的当前工具描述注入


if __name__ == "__main__":
    unittest.main()

