# -*- coding: utf-8 -*-
"""Two-stage protocol pipeline orchestration.

- StageRegistry：阶段类注册表（@register_stage 自动收集），新增阶段=注册+config 加名字。
- StageFactory：按名字实例化阶段。
- PhaseFactory：按 config.PHASE1_STAGES / PHASE2_STAGES 组装两阶段（kind 决定 debate/judge 分发）。
- PipelineV4：顺序跑 Phase；返回 final_graph。
"""
from typing import Any, Dict, List, Type

from protocol.base import Phase, Stage
from protocol.context import ProtocolContext

_STAGE_REGISTRY: Dict[str, Type[Stage]] = {}


def register_stage(cls: Type[Stage]) -> Type[Stage]:
    """类装饰器：注册 stage（key=cls.name）。"""
    if cls.name in _STAGE_REGISTRY:
        print(f"[pipeline] 重复注册 stage 覆盖: {cls.name}")
    _STAGE_REGISTRY[cls.name] = cls
    return cls


class StageFactory:
    """按名字实例化 stage（config 可选，stage 自行从 ctx 读配置）。"""

    def create(self, name: str) -> Stage:
        cls = _STAGE_REGISTRY.get(name)
        if cls is None:
            raise KeyError(f"stage {name!r} 未注册（已注册: {sorted(_STAGE_REGISTRY)}）")
        return cls()


class PhaseFactory:
    """按 config 的阶段清单组装 Phase 列表（默认两阶段，可配置增删）。"""

    def __init__(self, factory: StageFactory) -> None:
        self.factory = factory

    def build(self, config: Dict[str, Any]) -> List[Phase]:
        phases = []
        for phase_name, key, kind in (("phase1_existence", "PHASE1_STAGES", "existence"),
                                      ("phase2_direction", "PHASE2_STAGES", "direction")):
            stage_names: List[str] = list(config.get(key, []) or [])
            if not stage_names:
                continue
            stages = [self.factory.create(n) for n in stage_names]
            phases.append(Phase(phase_name, stages, kind=kind))
        return phases


class PipelineV4:
    """v4 流水线：顺序跑两阶段（模板方法）。"""

    def __init__(self, ctx: ProtocolContext, phases: List[Phase]) -> None:
        self.ctx = ctx
        self.phases = phases

    def run(self) -> Any:
        from util.log import get
        log = get("protocol.pipeline")
        for phase in self.phases:
            log.info("==== 进入阶段 %s（%d 个 stage，kind=%s）====",
                     phase.name, len(phase.stages), phase.kind or "?")
            with self.ctx.timings.timer(phase.name):
                phase.run(self.ctx)
            log.info("==== 阶段 %s 完成 ====", phase.name)
        return self.ctx.final_graph
