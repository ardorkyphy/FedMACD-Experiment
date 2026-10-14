# -*- coding: utf-8 -*-
"""Component-ablation experiment entry point.

用法:
  python src_v4/experiment/run_ablation.py                                  # 全部组合 × asia/hom/seed0
  python src_v4/experiment/run_ablation.py --ablation no_client_analyze     # 只跑单组合
  python src_v4/experiment/run_ablation.py --dataset child --sample 10000 --clients 4 --seeds 0 1 2

论文使用的三种消融：
  - no_client_analyze：关闭客户端本地 LLM 逐边分析（CLIENT_ANALYZE.enabled=False）
  - no_debate       ：两阶段移除 debate（聚合后直接进裁判）
  - no_client_analyze_no_debate：同时关闭本地分析和辩论

每次运行独立 build_config 并做防御性拷贝（CLIENT_ANALYZE / PHASE1/2_STAGES 等
build_config 只给全局引用，直接改会污染 config 模块，多折循环会串味）。
产物按组合隔离到 results/ablation/{组合}/results_v4.csv + audit/。
"""
import argparse
import copy
import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC_V4_ROOT = os.path.join(PROJECT_ROOT, "src_v4")
for _p in (PROJECT_ROOT, SRC_V4_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from run import build_config, main  # noqa: E402
from experiment.run_rep5 import EXTERNAL_DATA, DATASET_SEMANTICS  # noqa: E402
RESULTS_ROOT = os.path.join(SRC_V4_ROOT, "experiment", "results", "ablation")


def _fresh_config(dataset: str, sample: int, clients: int, seed: int,
                  scenario: str) -> dict:
    """独立快照 + 防御性拷贝：build_config 对 CLIENT_ANALYZE / PHASE 列表只给全局引用。"""
    cfg = build_config()
    for k in ("LLM", "LLM_CALL", "DATA", "ALGORITHM_PARAMS", "EXECUTION",
              "CLIENT_ANALYZE"):
        if isinstance(cfg.get(k), dict):
            cfg[k] = copy.deepcopy(cfg[k])
    for k in ("PHASE1_STAGES", "PHASE2_STAGES"):
        if isinstance(cfg.get(k), list):
            cfg[k] = list(cfg[k])
    sem = dict(DATASET_SEMANTICS.get(dataset, {}) or {})
    data = dict(cfg["DATA"])
    data.update(name=dataset, sample_size=sample, num_clients=clients,
                seed=seed, split=scenario, semantics=sem)
    ext = EXTERNAL_DATA.get(dataset)
    if ext:
        data.update(ext)
    cfg["DATA"] = data
    return cfg


def _rm_stage(cfg: dict, stage: str) -> None:
    for key in ("PHASE1_STAGES", "PHASE2_STAGES"):
        if stage in cfg[key]:
            cfg[key].remove(stage)


ABLATIONS = {
    "no_client_analyze": lambda c: c["CLIENT_ANALYZE"].update(enabled=False),
    "no_debate": lambda c: _rm_stage(c, "debate"),
    "no_client_analyze_no_debate": lambda c: (
        c["CLIENT_ANALYZE"].update(enabled=False), _rm_stage(c, "debate")),
}


def run_one(ablation: str, dataset: str, sample: int, clients: int,
            seed: int, scenario: str) -> dict:
    cfg = _fresh_config(dataset, sample, clients, seed, scenario)
    mut = ABLATIONS[ablation]
    mut(cfg)
    cfg["OUTPUT_ROOT"] = os.path.join(RESULTS_ROOT, ablation)
    cfg["LOGS_ROOT"] = os.path.join(RESULTS_ROOT, "logs")
    cfg["SAVE_ARTIFACTS"] = True
    cfg["EXECUTION"] = {"parallel": True, "max_workers": min(int(clients), 8)}
    t0 = time.time()
    res = main(cfg)
    res["ablation"] = ablation
    res["seconds"] = round(time.time() - t0, 1)
    return res


def main_cli(args) -> None:
    ablations = list(ABLATIONS) if args.ablation == "all" else [args.ablation]
    unknown = [a for a in ablations if a not in ABLATIONS]
    if unknown:
        raise SystemExit(f"未知消融组合: {unknown}（可选 {sorted(ABLATIONS)}）")
    for ablation in ablations:
        for seed in args.seeds:
            label = f"{ablation}/{args.dataset}/{args.scenario}/seed{seed}"
            print(f"[ablation] 开始 {label}", flush=True)
            try:
                res = run_one(ablation, args.dataset, args.sample,
                              args.clients, seed, args.scenario)
                print("[ablation] 完成 %-18s F1=%.4f skelF1=%.4f SHD=%g LLM=%d %.1fs"
                      % (label, res.get("f1", 0.0), res.get("skel_f1", 0.0),
                         res.get("shd", 0.0), res.get("llm_calls", 0),
                         res.get("seconds", 0.0)), flush=True)
            except Exception as exc:
                print(f"[ablation] 失败 {label}: {exc}", flush=True)
                raise


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="v4 组件消融（实验 C）")
    ap.add_argument("--ablation", default="all",
                    choices=sorted(ABLATIONS) + ["all"],
                    help="消融组合（all=全部）")
    ap.add_argument("--dataset", default="asia")
    ap.add_argument("--sample", type=int, default=1000)
    ap.add_argument("--clients", type=int, default=4)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--scenario", default="homogeneous",
                    choices=["homogeneous", "heterogeneous"])
    main_cli(ap.parse_args())
