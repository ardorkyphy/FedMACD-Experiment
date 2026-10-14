# -*- coding: utf-8 -*-
"""算法选择器测试：override 强制 / LLM 选择与参数 / 非法选择回落 / LLM 异常回落 / manual。"""
import os
import sys
import unittest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (PROJECT_ROOT, os.path.join(PROJECT_ROOT, "src_v4")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from util.algorithms.selector import AlgorithmSelector  # noqa: E402


class _FakeLlm:
    """可控 LLM 桩：按调用次数返回预设 JSON。"""

    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []

    def chat_json(self, prompt, system_prompt=None, role=None, max_parse_tries=4):
        self.calls.append((prompt, system_prompt, role))
        out = self.outputs.pop(0) if self.outputs else {}
        if isinstance(out, Exception):
            raise out
        return out


def _cfg(selection="llm", override=""):
    return {"ALGORITHM_SELECTION": selection, "ALGORITHM_OVERRIDE": override,
            "ALGORITHM_PARAMS": {}, "ALPHA": 0.05, "SEED": 0}


class TestAlgorithmSelector(unittest.TestCase):
    def test_llm_choice_ignores_params(self):
        """The LLM selects an algorithm only; any returned parameters are ignored."""
        llm = _FakeLlm([{"algorithm": "hc", "params": {"max_indegree": 2}, "reason": "小数据"}])
        sel = AlgorithmSelector(_cfg(), llm=llm)
        choice = sel.select({"n": 100}, {"sem": "x"}, discrete=True)
        self.assertEqual(choice.name, "hc")
        self.assertEqual(choice.params, {})           # LLM 参数建议被忽略
        self.assertEqual(choice.source, "llm")
        self.assertEqual(len(llm.calls), 1)
        self.assertIn("hc", llm.calls[0][0])          # 工具 schema 注入 prompt
        self.assertIn("<Available Tools>", llm.calls[0][0])

    def test_llm_illegal_choice_falls_back_to_rule(self):
        llm = _FakeLlm([{"algorithm": "direct_lingam", "reason": "连续"}] )  # 离散池不可用
        sel = AlgorithmSelector(_cfg(), llm=llm)
        choice = sel.select({}, {}, discrete=True)
        self.assertEqual(choice.source, "rule")
        self.assertEqual(choice.name, "pc")

    def test_llm_exception_falls_back(self):
        llm = _FakeLlm([RuntimeError("boom")])
        sel = AlgorithmSelector(_cfg(), llm=llm)
        choice = sel.select({}, {}, discrete=False)
        self.assertEqual(choice.source, "rule")

    def test_no_llm_injected_falls_back(self):
        sel = AlgorithmSelector(_cfg(), llm=None)
        choice = sel.select({}, {}, discrete=True)
        self.assertEqual(choice.source, "rule")

    def test_override_forces_everywhere(self):
        cfg = _cfg(selection="llm", override="ges")
        llm = _FakeLlm([{"algorithm": "hc", "reason": "whatever"}])
        sel = AlgorithmSelector(cfg, llm=llm)
        choice = sel.select({}, {}, discrete=True)
        self.assertEqual(choice.name, "ges")
        self.assertEqual(choice.source, "manual")
        self.assertEqual(len(llm.calls), 0)           # override 时不调用 LLM

    def test_manual_mode_with_override(self):
        sel = AlgorithmSelector(_cfg(selection="manual", override="hc"), llm=None)
        choice = sel.select({}, {}, discrete=True)
        self.assertEqual(choice.name, "hc")
        self.assertEqual(choice.source, "manual")

    def test_llm_assessment_parsed(self):
        """LLM 显式档位映射（assessment）透传到结果（审计用）。"""
        llm = _FakeLlm([{"assessment": {"sample_sufficiency": "紧张", "cell_thickness": "薄",
                                        "distribution": "偏态"},
                         "algorithm": "fedecd_local", "reason": "多状态小样本"}])
        sel = AlgorithmSelector(_cfg(), llm=llm)
        choice = sel.select({"discrete": True, "state_max": 6}, {}, discrete=True)
        self.assertEqual(choice.name, "fedecd_local")
        self.assertEqual(choice.assessment.get("sample_sufficiency"), "紧张")
        self.assertEqual(choice.assessment.get("cell_thickness"), "薄")

    def test_llm_assessment_missing_tolerated(self):
        """旧格式输出（无 assessment）不报错，assessment 默认空。"""
        llm = _FakeLlm([{"algorithm": "pc", "reason": "x"}])
        sel = AlgorithmSelector(_cfg(), llm=llm)
        choice = sel.select({}, {}, discrete=True)
        self.assertEqual(choice.name, "pc")
        self.assertEqual(choice.assessment, {})

if __name__ == "__main__":
    unittest.main()

