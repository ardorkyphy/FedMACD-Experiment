# -*- coding: utf-8 -*-
"""算法注册表（从 src/util/algorithms/registry.py 复刻）。

注册表 = 单例，@register 装饰器自动收集；候选池按 数据兼容+依赖可用 过滤。
"""
import re

from typing import Any, Dict, List, Optional, Type

from util.algorithms.base import CausalDiscoveryAlgorithm



def _strip_paper_citations(desc: str) -> str:
    """剥离 description 中的论文出处/验证从句（论文场景/论文验证/经典验证/验证于），
    The algorithm-pool prompt omits publication metadata to avoid selection bias.
    从句边界为中文句号/分号/结尾；实现细节句（如"本地方法=..."）保留。"""
    s = re.sub(r"(论文场景|论文验证|经典验证|验证于)[^。；]*(?=[。；]|$)", "", desc)
    s = re.sub(r"；{2,}", "；", s)
    s = s.replace("。；", "。")
    s = s.replace("。。", "。")
    return s.strip("；。 ")

class AlgorithmRegistry:
    """名字 → 算法类；负责注册、过滤、实例化（合并 config.ALGORITHM_PARAMS）。"""

    def __init__(self) -> None:
        self._classes: Dict[str, Type[CausalDiscoveryAlgorithm]] = {}

    def register(self, cls: Type[CausalDiscoveryAlgorithm]) -> Type[CausalDiscoveryAlgorithm]:
        if cls.name in self._classes:
            print(f"[algorithms] 重复注册覆盖: {cls.name}")
        self._classes[cls.name] = cls
        return cls

    def get(self, name: str) -> Optional[Type[CausalDiscoveryAlgorithm]]:
        return self._classes.get(name)

    def names(self) -> List[str]:
        return list(self._classes.keys())

    def candidates(self, discrete: bool) -> List[str]:
        out = []
        for name, cls in self._classes.items():
            dt = cls.data_type
            if dt not in ("any", "discrete" if discrete else "continuous"):
                continue
            if not cls().available():
                continue
            out.append(name)
        return out

    def default_for(self, discrete: bool) -> str:
        cands = self.candidates(discrete)
        return cands[0] if cands else "pc"

    def create(self, name: str, config: Dict[str, Any],
               extra_params: Optional[Dict[str, Any]] = None,
               seed: Optional[int] = None) -> CausalDiscoveryAlgorithm:
        cls = self.get(name)
        if cls is None:
            raise ValueError(f"算法 {name!r} 未注册（注册表: {self.names()}）")
        forced = dict(config.get("ALGORITHM_PARAMS", {}).get(name, {}) or {})
        # Apply schema defaults first so implementations need no duplicate fallback.
        # 再被 extra_params / ALGORITHM_PARAMS 覆盖；default=None 不填（由 discover 决定）
        params = {}
        for _k, _v in (getattr(cls, "param_schema", None) or {}).items():
            if isinstance(_v, dict) and _v.get("default") is not None:
                params[_k] = _v["default"]
        params.update(extra_params or {})
        params.update(forced)
        alpha = float(config.get("ALPHA", 0.05))
        seed = int(seed if seed is not None else config.get("SEED", 0) or 0)
        return cls(params=params, alpha=alpha, seed=seed)

    def pool_text(self, discrete: bool) -> str:
        lines = []
        for name in self.candidates(discrete):
            info = self._classes[name]
            desc = _strip_paper_citations(getattr(info, "description", "") or "")
            if desc:
                lines.append(f"- {name}：{desc}（家族 {info.family}，数据 {info.data_type}，成本 {info.cost}）")
            else:
                lines.append(f"- {name}（{info.family}，数据 {info.data_type}，成本 {info.cost}）")
        return "\n".join(lines) or "- 无（回落 PC）"

registry = AlgorithmRegistry()


def register(cls: Type[CausalDiscoveryAlgorithm]) -> Type[CausalDiscoveryAlgorithm]:
    """类装饰器：注册并返回原类。"""
    registry.register(cls)
    return cls
