# -*- coding: utf-8 -*-
"""v4.36 component-ablation single-fold driver.

The three reported variants use the same main-experiment parameters:
  变体 1 no_client_analyze = 本地分析组件关（CLIENT_ANALYZE.enabled=False，保留统计证据，LLM 逐边分析跳过）
  变体 2 no_debate         = 辩论组件关（两阶段移除 debate 阶段；裁判终裁保留，与 v4_35 消融同口径）
  变体 3 no_client_analyze_no_debate = 双关（本地分析 + 辩论都关，只留统计聚合 + 裁判；
        completes the 2x2 component design)
参数与主算法一致：NUM_CLIENTS=4、STRENGTH=0.7（het）、sampling_seed=-1、SEED=seed、
  DEBATE_MAX_ROUNDS=2 / DEBATE_BATCH_EDGES=20 / JUDGE_BATCH_EDGES=20（config 既有值）、
  EXECUTION.max_workers=1（deepseek 一次一请求，进程间 3 路并行）。
指标俱全：F1/P/R/SHD/nSHD + skel_* + 候选边/删边/settled/doubtful + LLM 调用数，
  并含 token 消耗量：llm_prompt_tokens / llm_completion_tokens / llm_total_tokens（resp.usage 实际调用累计）。

用法: python scripts/run_v436_ablation.py <variant> <dataset> <scenario> <seed>
  variant: no_client_analyze | no_debate | no_client_analyze_no_debate
  scenario: homogeneous | heterogeneous
产物（唯一落盘处 v4_36_final/ablation/）:
  v4_36_ablation_<variant>_<ds>_<scenario>_seed<N>/  (audit/ prompts/ results_v4.csv metrics_v4.png logs/)
  v4_36_ablation_runs.jsonl  （3 变体：10 数据集 × 同质/异质 × seed0-2 = 180 行；重复折自动跳过，--force 可重跑）
"""
import argparse
import json
import os
import sys
import time

os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_V4_ROOT = os.path.join(PROJECT_ROOT, "src_v4")
for _p in (PROJECT_ROOT, SRC_V4_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
except ImportError:
    pass

from run import main  # noqa: E402
from config.config import VERSION, LLM as LLM_CFG  # noqa: E402
from experiment.run_rep5 import DATASET_SEMANTICS  # noqa: E402
from experiment.run_ablation import _fresh_config, ABLATIONS  # noqa: E402

_ACTIVE_PROVIDER = (LLM_CFG.get("active_provider") or "deepseek").strip()
_PROVIDERS = LLM_CFG.get("providers", {})
if _ACTIVE_PROVIDER not in _PROVIDERS:
    raise SystemExit(f"Unknown FEDMACD_PROVIDER: {_ACTIVE_PROVIDER}")
_MODEL = _PROVIDERS[_ACTIVE_PROVIDER]["model"]
_PAPER_MODELS = {"deepseek": "deepseek-v4-flash", "qwen": "qwen3.8-flash", "zhipu": "glm-5.3-flash"}

_LLM_TAG = "" if _ACTIVE_PROVIDER == "deepseek" else f"_{_ACTIVE_PROVIDER}"
if _MODEL != _PAPER_MODELS[_ACTIVE_PROVIDER]:
    import re
    _LLM_TAG += "_" + re.sub(r"[^A-Za-z0-9_.-]", "_", _MODEL)

RESULT_ROOT = os.path.join(SRC_V4_ROOT, "experiment", "results", f"{VERSION}_final{_LLM_TAG}", "ablation")
RUNS = os.path.join(RESULT_ROOT, f"{VERSION}_ablation_runs.jsonl")
NUM_CLIENTS = 4
STRENGTH = 0.7
OFFICIAL_DATASETS = {
    "asia": 10000, "child": 10000, "alarm": 10000,
    "survey": 10000, "earthquake": 10000,
    "Auto_MPG": 392, "DWD_climate": 349, "Sachs": 7466,
    "10nodes": 5000, "ER20": 256,
}
VARIANTS = ("no_client_analyze", "no_debate", "no_client_analyze_no_debate")


def run_one(variant: str, name: str, split: str, seed: int, force: bool = False) -> None:
    if variant not in VARIANTS:
        raise SystemExit(f"未知变体 {variant!r}（可选 {VARIANTS}）")
    if name not in OFFICIAL_DATASETS:
        raise SystemExit(f"未知数据集 {name!r}（可选 {sorted(OFFICIAL_DATASETS)}）")
    sample_size = OFFICIAL_DATASETS[name]
    fold = f"{VERSION}_ablation_{variant}_{name}_{split or 'homogeneous'}_seed{seed}"
    fold_root = os.path.join(RESULT_ROOT, fold)

    cfg = _fresh_config(name, sample_size, NUM_CLIENTS, seed, split)   # 防御性拷贝，不污染 config 模块
    ABLATIONS[variant](cfg)                                            # 复用既有消融改法（run_ablation.py）
    cfg["DATA"]["sampling_seed"] = -1                                  # 每折独立采样（主口径）
    cfg["DATA"]["semantics"] = dict(DATASET_SEMANTICS.get(name, {}))   # 6 套数据集语义全覆盖
    cfg["SEED"] = seed
    if split == "heterogeneous":
        cfg["DATA"]["heterogeneity_strength"] = STRENGTH
        cfg["HETEROGENEITY_STRENGTH"] = STRENGTH
    cfg["EXECUTION"] = {"parallel": True, "max_workers": 1}            # LLM 串行（deepseek 一次一请求）
    cfg["OUTPUT_ROOT"] = fold_root
    cfg["LOGS_ROOT"] = os.path.join(fold_root, "logs")
    cc = dict(cfg.get("LLM_CALL") or {})
    cc["prompt_log_dir"] = os.path.join(fold_root, "prompts")
    cfg["LLM_CALL"] = cc

    # 已完成的折跳过（可断点续跑）；--force 重跑
    if not force and os.path.exists(RUNS):
        try:
            done = {(j.get("variant"), j.get("dataset"), j.get("scenario"), int(j.get("seed", -1)))
                    for ln in open(RUNS, encoding="utf-8") if ln.strip()
                    for j in [json.loads(ln)]}
        except (ValueError, TypeError, json.JSONDecodeError):
            done = set()
        if (variant, name, split or "homogeneous", seed) in done:
            print(f"SKIP_DONE {fold}", flush=True)
            return

    print(f"===== {VERSION} 消融单折: {variant} {name} {split or 'homogeneous'} seed={seed} "
          f"n={sample_size} clients={NUM_CLIENTS} =====", flush=True)
    t0 = time.time()
    res = main(cfg)
    res["variant"] = variant
    res["dataset"] = name
    res["scenario"] = split or "homogeneous"
    res["seed"] = seed
    res["seconds"] = round(time.time() - t0, 1)
    os.makedirs(RESULT_ROOT, exist_ok=True)

    key = (variant, name, split or "homogeneous", seed)
    if os.path.exists(RUNS):
        try:
            keep = []
            for ln in open(RUNS, encoding="utf-8"):
                if not ln.strip():
                    continue
                j = json.loads(ln)
                if not (j.get("variant") == key[0] and j.get("dataset") == key[1]
                        and j.get("scenario") == key[2] and int(j.get("seed", -1)) == key[3]):
                    keep.append(ln)
        except (ValueError, TypeError, json.JSONDecodeError):
            keep = None
        if keep is not None:
            with open(RUNS, "w", encoding="utf-8") as f:
                f.writelines(keep)
    with open(RUNS, "a", encoding="utf-8") as f:
        f.write(json.dumps(res, ensure_ascii=False, default=str) + "\n")
    print("RUN_DONE variant=%s dataset=%s scenario=%s seed=%d f1=%.4f skel_f1=%.4f shd=%s "
          "llm=%s ptok=%s ctok=%s tok=%s cand=%s dropped=%s secs=%.1f" % (
              variant, name, split or "homogeneous", seed,
              float(res.get("f1", 0) or 0), float(res.get("skel_f1", 0) or 0),
              res.get("shd", "?"), res.get("llm_calls", "?"),
              res.get("llm_prompt_tokens", "?"), res.get("llm_completion_tokens", "?"),
              res.get("llm_total_tokens", "?"), res.get("candidate_edges", "?"),
              res.get("dropped_edges", "?"), res["seconds"]), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Run one paper fold; paid API calls are possible.")
    ap.add_argument("variant", choices=VARIANTS)
    ap.add_argument("dataset", choices=tuple(OFFICIAL_DATASETS))
    ap.add_argument("scenario", choices=("homogeneous", "heterogeneous"))
    ap.add_argument("seed", type=int, choices=(0, 1, 2))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    run_one(args.variant, args.dataset, args.scenario, args.seed, force=args.force)
