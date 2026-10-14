# -*- coding: utf-8 -*-
"""FedMACD two-stage server-side debate entry point.

读配置 → DataService 加载 → 构造资源（Client/Server/executor/ctx）→ PipelineV4 编排
（两阶段：存在性→候选集；方向→最终图）→ 双口径评估（骨架 F1 + 有向 F1/SHD）→ 落盘审计。
"""
import json
import os
import sys
import time
from typing import Dict, List, Optional

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_V4_ROOT = os.path.join(PROJECT_ROOT, "src_v4")

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
except ImportError:
    pass
for _p in (PROJECT_ROOT, SRC_V4_ROOT):   # common/（项目根）与 src_v4/
    if _p not in sys.path:
        sys.path.insert(0, _p)

from config import config as G          # noqa: E402  全局配置（唯一配置源）
from util.log import get as get_logger  # noqa: E402
from util.log import setup as setup_logging  # noqa: E402
from util.execution import TaskExecutor  # noqa: E402
from util.data.service import DataService  # noqa: E402
from util.results.service import ResultService  # noqa: E402
from client.client import Client  # noqa: E402
from server.server import Server  # noqa: E402
from protocol.context import ProtocolContext  # noqa: E402
from protocol.pipeline import PipelineV4, PhaseFactory, StageFactory  # noqa: E402
import protocol.stages  # noqa: F401,E402  导入即触发 @register_stage 自注册

_TYPING_NAMES = {"Dict", "List"}


def build_config() -> Dict:
    """收集 src_v4/config/config.py 全部配置项；嵌套 dict 浅拷贝。"""
    cfg = {k: v for k, v in vars(G).items()
           if not k.startswith("_") and k not in _TYPING_NAMES}
    for key in ("LLM", "LLM_CALL", "DATA", "ALGORITHM_PARAMS", "EXECUTION"):
        if isinstance(cfg.get(key), dict):
            cfg[key] = dict(cfg[key])
    return cfg


def main(cfg: Optional[Dict] = None) -> Dict:
    """v4 入口：数据 → 资源 → 上下文 → 两阶段流水线 → 双口径评估 → 落盘。"""
    cfg = cfg if cfg is not None else build_config()
    data_cfg = cfg["DATA"]

    task = f"v4_{data_cfg.get('name', '') or 'data'}_{data_cfg.get('split', '') or 'homogeneous'}"
    setup_logging(task, log_dir=cfg.get("LOGS_ROOT") or None)
    log = get_logger("experiment.run")
    log.info("==== v4 实验开始：数据集=%s 场景=%s 客户端=%d 离散=%s ====",
             data_cfg.get("name", ""), data_cfg.get("split", "") or "homogeneous",
             data_cfg.get("num_clients", 0), bool(data_cfg.get("discrete", True)))

    # 1. 数据层
    loaded = DataService().load(data_cfg)
    log.info("数据加载完成：样本=%d×%d 客户端=%d 真边=%d",
             loaded.data.shape[0], loaded.data.shape[1], len(loaded.parts), len(loaded.true_edges))

    # 2. 资源层：客户端（数据分片 + 本地 LLM）/ 服务端（辩手/裁判 LLM）
    clients: List[Client] = []
    for i, part in enumerate(loaded.parts):
        clients.append(Client(client_id=f"client_{i}", config=cfg, data_part=part,
                              columns=loaded.columns, discrete=loaded.discrete))
    server = Server(config=cfg, clients=clients)

    # 3. 并行基建
    exec_cfg = cfg.get("EXECUTION", {}) or {}
    executor = TaskExecutor(max_workers=int(exec_cfg.get("max_workers", 4) or 4),
                            enabled=bool(exec_cfg.get("parallel", True)))

    # 4. 协议层：上下文 + 两阶段流水线
    with executor:
        ctx = ProtocolContext(config=cfg, clients=clients, server=server,
                              executor=executor, loaded=loaded)
        pipeline = PipelineV4(ctx, phases=PhaseFactory(StageFactory()).build(cfg))
        final_graph = pipeline.run()

    # 5. 双口径评估（v4.13）：骨架（Phase1 候选集）+ 有向（Phase2 最终图）
    name2idx = {n: i for i, n in enumerate(final_graph.nodes)} if final_graph else {}
    est_directed: set = set()
    for u, v in (final_graph.edges if final_graph else []):
        if u in name2idx and v in name2idx:
            est_directed.add((name2idx[u], name2idx[v]))
    # 无向边 = 存在但方向未定：评估按双向计入（同时贡献 TP+1、FP+1），
    # 与定向边口径不同；骨架口径（skel_*）另按 Phase1 候选集无向计算（v4.19 确认保留）。
    for s, t in (final_graph.undirected if final_graph else []):
        if s in name2idx and t in name2idx:
            est_directed.add((name2idx[s], name2idx[t]))
            est_directed.add((name2idx[t], name2idx[s]))

    result_service = ResultService()
    llm_calls = _count_llm_calls(clients, server)   # 只统计已实例化的 LLM（不触发懒加载）
    result = result_service.compute(est_directed, loaded.true_edges, len(loaded.columns),
                                    timings=ctx.timings.summary())
    result["llm_calls"] = float(llm_calls)
    pt, ct = _count_llm_tokens(clients, server)   # resp.usage 累计（实际调用）
    result["llm_prompt_tokens"] = float(pt)
    result["llm_completion_tokens"] = float(ct)
    result["llm_total_tokens"] = float(pt + ct)

    # Skeleton metrics use the undirected Phase 1 candidate set.
    # 旧实现把每条候选边双向展开再喂有向口径，完美骨架 F1 上限仅 0.667（每条真边配反向 FP）。
    # 现按无向归一化 (min,max) 计算：skel_* 指标 = 骨架层标准 F1/P/R/SHD。
    def _undirected(edges) -> set:
        return {(min(u, v), max(u, v)) for u, v in edges}
    skel: set = set()
    for u, v in ctx.candidate_set:
        if u in name2idx and v in name2idx:
            skel.add((name2idx[u], name2idx[v]))
    skel_res = result_service.compute(_undirected(skel), _undirected(loaded.true_edges),
                                      len(loaded.columns))
    for k, v in skel_res.items():
        result[f"skel_{k}"] = v
    result["candidate_edges"] = float(len(ctx.candidate_set))
    result["dropped_edges"] = float(len(ctx.dropped_edges))
    result["settled"] = float(sum(1 for r in ctx.edge_records.values() if getattr(r, "direction", None)))
    result["doubtful"] = float(sum(1 for r in ctx.edge_records.values() if getattr(r, "doubtful", False)))

    log.info("评估结果：SHD=%g F1=%.4f P=%.4f R=%.4f nSHD=%.4f | 骨架F1=%.4f LLM调用=%d",
             result.get("shd", 0.0), result.get("f1", 0.0), result.get("precision", 0.0),
             result.get("recall", 0.0), result.get("nshd", 0.0),
             result.get("skel_f1", 0.0), result.get("llm_calls", 0))

    # 6. 落盘：结果行（RESULT_FORMAT 约定）
    output_root = cfg.get("OUTPUT_ROOT") or os.path.join(PROJECT_ROOT, "results")
    os.makedirs(output_root, exist_ok=True)
    row = {
        "dataset": data_cfg.get("name", ""),
        "scenario": data_cfg.get("split", "") or "homogeneous",
        "num_clients": data_cfg.get("num_clients", len(clients)),
        "sample_size": data_cfg.get("sample_size", 0) or 0,
        "seed": data_cfg.get("seed", 0),
        "algorithm_selection": cfg.get("ALGORITHM_SELECTION", ""),
        "algorithm": cfg.get("ALGORITHM_OVERRIDE", "") or "",
    }
    merged = {**row, **result}
    fmt = str(cfg.get("RESULT_FORMAT", "csv")).lower()
    if fmt == "json":
        path = os.path.join(output_root, "results_v4.jsonl")
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(merged, ensure_ascii=False) + "\n")
    else:
        import pandas as pd
        path = os.path.join(output_root, "results_v4.csv")
        header = not os.path.exists(path) or os.path.getsize(path) == 0
        pd.DataFrame([merged]).to_csv(path, mode="a", header=header, index=False, encoding="utf-8")
    print(f"[run_v4] 结果已追加: {path}")

    # 7. 审计产物
    if cfg.get("SAVE_ARTIFACTS", True):
        audit_root = os.path.join(output_root, "audit", time.strftime("%Y%m%d_%H%M%S"))
        os.makedirs(audit_root, exist_ok=True)
        if final_graph is not None:    # A malformed stage list can leave final_graph unset.
            fg = {"nodes": final_graph.nodes, "edges": final_graph.edges,
                  "undirected": final_graph.undirected}
        else:
            fg = {"nodes": [], "edges": [], "undirected": []}
        with open(os.path.join(audit_root, "final_graph.json"), "w", encoding="utf-8") as f:
            json.dump(fg, f, ensure_ascii=False, indent=2)
        with open(os.path.join(audit_root, "candidate_set.json"), "w", encoding="utf-8") as f:
            json.dump({"candidate": [list(k) for k in ctx.candidate_set],
                       "dropped": [list(k) for k in ctx.dropped_edges]},
                      f, ensure_ascii=False, indent=2)
        with open(os.path.join(audit_root, "edge_records.json"), "w", encoding="utf-8") as f:
            json.dump(_dump_records(ctx), f, ensure_ascii=False, indent=2)
        if ctx.dropped_records:      # Persist rejected edges with their full decision evidence.
            with open(os.path.join(audit_root, "dropped_records.json"), "w", encoding="utf-8") as f:
                json.dump(_serialize_records(ctx.dropped_records), f, ensure_ascii=False, indent=2)
        with open(os.path.join(audit_root, "timings.json"), "w", encoding="utf-8") as f:
            json.dump(ctx.timings.summary(), f, ensure_ascii=False, indent=2)
        if (data_cfg.get("split", "") or "").strip().lower() == "heterogeneous":
            try:
                het_rep = _het_report(loaded.parts, loaded.data,
                                      bool(data_cfg.get("discrete", True)), loaded.columns)
                het_rep["strength"] = data_cfg.get("heterogeneity_strength", 0.7)
                _dfmt = str(data_cfg.get("format", "") or "").strip().lower()
                if not _dfmt:
                    import os as _os
                    from util.data.datasets import resolve_path as _rp
                    _dfmt = _os.path.splitext(_rp(str(data_cfg.get("name", "")))["data"])[1].lstrip(".").lower()
                het_rep["method"] = ("intervention_sampling" if _dfmt == "bif" else "weighted_resample")
                with open(os.path.join(audit_root, "het_report.json"), "w", encoding="utf-8") as f:
                    json.dump(het_rep, f, ensure_ascii=False, indent=2)
            except Exception as exc:
                log.warning("het_report 生成失败（不影响结果）：%s", exc)
        for c in clients:
            c.save(audit_root)
        print(f"[run_v4] 审计产物: {audit_root}")

    # 8. 指标图
    try:
        plot_path = os.path.join(output_root, "metrics_v4.png")
        result_service.draw({f"{data_cfg.get('name', 'data')}_{data_cfg.get('split', '') or 'homogeneous'}": result},
                            plot_path)
        log.info("指标图已保存：%s", plot_path)
    except Exception as exc:
        log.warning("指标图保存失败（不影响结果）：%s", exc)

    return merged


def _het_report(parts, data, discrete: bool, columns: list) -> dict:
    """Audit heterogeneous construction using per-client marginal JS divergence,
    客户端两两 JS、唯一行率。离散=边际频率；连续=全局 10 分位箱频率。"""
    import numpy as np

    def _jsd(p, q):
        p = np.asarray(p, float); q = np.asarray(q, float)
        p = p / p.sum(); q = q / q.sum()
        m = 0.5 * (p + q)
        mask = m > 0
        return float(0.5 * ((p[mask] * np.log(np.where(p[mask] > 0, p[mask] / m[mask], 1.0))).sum())
                     + 0.5 * ((q[mask] * np.log(np.where(q[mask] > 0, q[mask] / m[mask], 1.0))).sum()))

    # Use the global/client union for the discrete state space because a preferred
    # 全局样本可能缺该态 → 两端 bincount 长度不一致会 ValueError 崩掉整个审计）
    _minlens = None
    if discrete:
        _minlens = [int(data[:, j].max()) + 1 for j in range(data.shape[1])]
        for part in parts:
            for j in range(part.shape[1]):
                _minlens[j] = max(_minlens[j], int(part[:, j].max()) + 1)

    def _marginals(X, edges=None, minlens=None):
        out = []
        if edges is None:                       # 离散：边际频率
            for j in range(X.shape[1]):
                ml = int(minlens[j]) if minlens is not None else int(X[:, j].max()) + 1
                out.append(np.bincount(X[:, j].astype(int),
                                       minlength=ml).astype(float) / len(X))
            return out
        for j in range(X.shape[1]):             # 连续：全局 10 分位箱
            e = edges[j].copy(); e[-1] += 1e-9
            h, _ = np.histogram(X[:, j], bins=e)
            out.append(h.astype(float) / len(X))
        return out

    g_edges = None if discrete else [
        np.quantile(data[:, j], np.linspace(0, 1, 11)) for j in range(data.shape[1])]
    g = _marginals(data, g_edges, _minlens)
    rep = {"per_client": [], "pairwise_js": []}
    for c, part in enumerate(parts):
        m = _marginals(part, g_edges, _minlens)
        js = [_jsd(a, b) for a, b in zip(m, g)]
        uniq = len({tuple(row) for row in part.tolist()})
        rep["per_client"].append({
            "client": c, "n": int(len(part)),
            "unique_rows": int(uniq), "unique_rate": round(uniq / len(part), 4),
            "js_mean": round(float(np.mean(js)), 4),
            "js_max": round(float(np.max(js)), 4),
            "js_argmax_col": int(np.argmax(js)),
            "js_per_var": {str(columns[j] if j < len(columns) else j): round(v, 4)
                           for j, v in enumerate(js)},
        })
    for i in range(len(parts)):
        row = []
        for j in range(len(parts)):
            if i == j:
                row.append(0.0)
            else:
                mi, mj = _marginals(parts[i], g_edges, _minlens), _marginals(parts[j], g_edges, _minlens)
                row.append(round(float(np.mean([_jsd(a, b) for a, b in zip(mi, mj)])), 4))
        rep["pairwise_js"].append(row)
    return rep


def _count_llm_calls(clients, server) -> int:
    """累计已实例化 LLM 的调用数（不触发懒加载；_llm 未建实例=无调用）。"""
    total = 0
    for c in clients:
        inst = getattr(c, "_llm", None)
        if inst is not None:
            total += int(getattr(inst, "call_count", 0))
    inst = getattr(server, "_llm", None)
    if inst is not None:
        total += int(getattr(inst, "call_count", 0))
    return total


def _count_llm_tokens(clients, server):
    """累计已实例化 LLM 的 token 消耗（resp.usage；不触发懒加载）。"""
    pt = ct = 0
    for obj in list(clients) + [server]:
        inst = getattr(obj, "_llm", None)
        if inst is not None:
            tc = getattr(inst, "token_counts", None) or {}
            pt += int(tc.get("prompt_tokens", 0) or 0)
            ct += int(tc.get("completion_tokens", 0) or 0)
    return pt, ct


def _dump_records(ctx) -> Dict:
    """EdgeRecord 审计摘要（dataclass 序列化兼容）。"""
    return _serialize_records(ctx.edge_records)


def _serialize_records(records: Dict) -> Dict:
    """EdgeRecord 集合序列化（存活边 / 被裁边共用，v4.29 审计补全）。"""
    from dataclasses import asdict
    out = {}
    for (u, v), rec in records.items():
        def _d(obj):
            if obj is None:
                return None
            if hasattr(obj, "__dataclass_fields__"):
                return asdict(obj)
            return vars(obj) if hasattr(obj, "__dict__") else str(obj)
        out[f"{u}-{v}"] = {
            "exist_decision": rec.exist_decision,
            "direction": rec.direction,
            "doubtful": rec.doubtful,
            "decision": rec.decision,
            "refine_action": rec.refine_action,
            "aggregate": _d(rec.aggregate),
            "debate": _d(rec.debate),
            "verdict": _d(rec.verdict),
            "client_evidence": {cid: {k: _d(v) for k, v in vars(ce).items()}
                                for cid, ce in rec.client_evidence.items()},
        }
    return out
