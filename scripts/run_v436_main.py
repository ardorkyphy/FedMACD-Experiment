# -*- coding: utf-8 -*-
"""FedMACD 论文主实验的单折可复现入口。

Paper configuration (src_v4/config/config.py VERSION=v4_36):
  - 60 折 = 10 数据集 × homogeneous/heterogeneous × seed 0/1/2
  - 数据集：离散 asia(8v)/child(20v)/alarm(37v)/survey(6v)/earthquake(5v) n=10000；连续 Auto_MPG(5v)/DWD_climate(6v)/Sachs(11v)/10nodes(10v)/ER20(20v) 官方量
  - NUM_CLIENTS=4；STRENGTH=0.7；sampling_seed=-1；SEED=seed
  - 无盲筛（v4_36 代码已无 BLINDSPOT/CANDIDATE_NOMINATE 配置键）；辩论两阶段不分包；
    DEBATE_MAX_ROUNDS=2 / DEBATE_BATCH_EDGES=20 / JUDGE_BATCH_EDGES=20（config 既有值）
  - LLM 线程参数=1（EXECUTION.max_workers=1，deepseek 一次一个请求）
  - 产物：v4_36_final/algorithm/v4_36_<ds>_<scen>_seed<N>/（audit/ prompts/ results_v4.csv metrics_v4.png）
    + v4_36_final/algorithm/v4_36_runs.jsonl（60 行）
用法: python scripts/run_v436_main.py <dataset> <scenario> <seed>
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

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_V4_ROOT = os.path.join(PROJECT_ROOT, "src_v4")
for _p in (PROJECT_ROOT, SRC_V4_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
except ImportError:
    pass

from run import build_config, main  # noqa: E402
from config.config import VERSION, LLM as LLM_CFG  # noqa: E402
from experiment.run_rep5 import EXTERNAL_DATA, DATASET_SEMANTICS  # noqa: E402

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

RESULT_ROOT = os.path.join(SRC_V4_ROOT, "experiment", "results", f"{VERSION}_final{_LLM_TAG}", "algorithm")
RUNS = os.path.join(RESULT_ROOT, f"{VERSION}_runs.jsonl")
NUM_CLIENTS = 4
STRENGTH = 0.7
OFFICIAL_DATASETS = {
    "asia": 10000, "child": 10000, "alarm": 10000,
    "survey": 10000, "earthquake": 10000,
    "Auto_MPG": 392, "DWD_climate": 349, "Sachs": 7466,
    "10nodes": 5000, "ER20": 256,
}


def run_one(name: str, split: str, seed: int) -> None:
    sample_size = OFFICIAL_DATASETS[name]
    fold = f"{VERSION}_{name}_{split or 'homogeneous'}_seed{seed}"
    fold_root = os.path.join(RESULT_ROOT, fold)
    # Skip completed folds to support resumable runs.
    if os.path.exists(RUNS):
        try:
            done = {(j.get("dataset"), j.get("scenario"), int(j.get("seed", -1)))
                    for ln in open(RUNS, encoding="utf-8") if ln.strip()
                    for j in [json.loads(ln)]}
        except (ValueError, TypeError, json.JSONDecodeError):
            done = set()
        if (name, split or "homogeneous", seed) in done:
            print(f"SKIP_DONE {fold}", flush=True)
            return

    cfg = build_config()
    cfg["DATA"].update(name=name, sample_size=sample_size, num_clients=NUM_CLIENTS,
                       seed=seed, split=split, sampling_seed=-1,
                       semantics=DATASET_SEMANTICS.get(name, {}))
    cfg["SEED"] = seed
    if name in EXTERNAL_DATA:
        cfg["DATA"].update(EXTERNAL_DATA[name])
    if split == "heterogeneous":
        cfg["DATA"]["heterogeneity_strength"] = STRENGTH
        cfg["HETEROGENEITY_STRENGTH"] = STRENGTH
    cfg["EXECUTION"] = {"parallel": True, "max_workers": 1}   # LLM 串行（deepseek 一次一请求）
    cfg["OUTPUT_ROOT"] = fold_root
    cfg["LOGS_ROOT"] = os.path.join(fold_root, "logs")
    cc = dict(cfg.get("LLM_CALL") or {})
    cc["prompt_log_dir"] = os.path.join(fold_root, "prompts")
    cfg["LLM_CALL"] = cc
    print(f"===== {VERSION} 单折: {name} {split or 'homogeneous'} seed={seed} "
          f"n={sample_size} clients={NUM_CLIENTS} =====", flush=True)
    t0 = time.time()
    res = main(cfg)
    res["dataset"] = name
    res["scenario"] = split or "homogeneous"
    res["seed"] = seed
    res["seconds"] = round(time.time() - t0, 1)
    os.makedirs(RESULT_ROOT, exist_ok=True)
    key = (name, split or "homogeneous", seed)
    if os.path.exists(RUNS):
        try:
            keep = []
            for ln in open(RUNS, encoding="utf-8"):
                if not ln.strip():
                    continue
                j = json.loads(ln)
                if not (j.get("dataset") == key[0] and j.get("scenario") == key[1]
                        and int(j.get("seed", -1)) == key[2]):
                    keep.append(ln)
        except (ValueError, TypeError, json.JSONDecodeError):
            keep = None
        if keep is not None:
            with open(RUNS, "w", encoding="utf-8") as f:
                f.writelines(keep)
    with open(RUNS, "a", encoding="utf-8") as f:
        f.write(json.dumps(res, ensure_ascii=False, default=str) + "\n")
    print("RUN_DONE dataset=%s scenario=%s seed=%d f1=%.4f skel_f1=%.4f shd=%s "
          "llm=%s cand=%s dropped=%s secs=%.1f" % (
              name, split or "homogeneous", seed,
              float(res.get("f1", 0) or 0), float(res.get("skel_f1", 0) or 0),
              res.get("shd", "?"), res.get("llm_calls", "?"),
              res.get("candidate_edges", "?"), res.get("dropped_edges", "?"),
              res["seconds"]), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Run one paper fold; paid API calls are possible.")
    ap.add_argument("dataset", choices=tuple(OFFICIAL_DATASETS))
    ap.add_argument("scenario", choices=("homogeneous", "heterogeneous"))
    ap.add_argument("seed", type=int, choices=(0, 1, 2))
    args = ap.parse_args()
    run_one(args.dataset, args.scenario, args.seed)
