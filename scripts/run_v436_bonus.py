# -*- coding: utf-8 -*-
"""Run one v4.36 additional-analysis fold with the DeepSeek provider.

The requested analysis parameters are overridden in-process without changing config.py:
  exp=nc       客户端数量：child/alarm × homogeneous × NC 7/12/17 × seed0-2（每客户端样本=10000/k）
  exp=rounds1  辩论轮数中间点：survey/DWD_climate × hom+het × seed0-2，DEBATE_MAX_ROUNDS=1（0 轮=no_debate 消融、2 轮=主算法，端点已存在）
  exp=strength 异质强度阶梯：survey/DWD_climate × heterogeneous × strength 0.3/0.5/0.9 × seed0-2（0.7 端点=主算法 het 已有；NC=4）

产物（独立于三个 v4_36_final* 保护区）：
  v4_36_bonus_deepseek/algorithm_nc/       折目录 v4_36_<ds>_homogeneous_nc<nc>_seed<s>/ + v4_36_runs_nc.jsonl
  v4_36_bonus_deepseek/ablation_rounds/    折目录 v4_36_rounds1_<ds>_<scen>_seed<s>/ + v4_36_runs_rounds1.jsonl
  v4_36_bonus_deepseek/ablation_strength/  折目录 v4_36_strength<s>_<ds>_heterogeneous_seed<s>/ + v4_36_runs_strength.jsonl

用法: python scripts/run_v436_bonus.py <exp> <dataset> <scenario> <seed> [param]
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

from run import build_config, main  # noqa: E402
from config.config import VERSION, LLM as LLM_CFG  # noqa: E402
from experiment.run_rep5 import EXTERNAL_DATA, DATASET_SEMANTICS  # noqa: E402

_MODEL = LLM_CFG["providers"]["deepseek"]["model"]
_MODEL_TAG = "" if _MODEL == "deepseek-v4-flash" else "_" + _MODEL.replace("/", "_").replace("\\", "_")
BONUS_ROOT = os.path.join(SRC_V4_ROOT, "experiment", "results", f"{VERSION}_bonus_deepseek{_MODEL_TAG}")
NC_DATASETS = ("child", "alarm")
NC_VALUES = (7, 12, 17)
SR_DATASETS = ("survey", "DWD_climate")
STRENGTH_VALUES = (0.3, 0.5, 0.9)
OFFICIAL_SAMPLE = {"child": 10000, "alarm": 10000, "survey": 10000, "DWD_climate": 349}
NUM_CLIENTS_BASE = 4
STRENGTH_BASE = 0.7
BONUS_PROVIDER = "deepseek"

EXP_SPEC = {
    "nc": {"scen": ("homogeneous",), "datasets": NC_DATASETS, "param": NC_VALUES,
           "out": "algorithm_nc", "runs": f"{VERSION}_runs_nc.jsonl"},
    "rounds1": {"scen": ("homogeneous", "heterogeneous"), "datasets": SR_DATASETS, "param": None,
                "out": "ablation_rounds", "runs": f"{VERSION}_runs_rounds1.jsonl"},
    "strength": {"scen": ("heterogeneous",), "datasets": SR_DATASETS, "param": STRENGTH_VALUES,
                 "out": "ablation_strength", "runs": f"{VERSION}_runs_strength.jsonl"},
}


def _validate(exp: str, name: str, split: str, seed: int, param) -> None:
    spec = EXP_SPEC.get(exp)
    if spec is None:
        raise SystemExit(f"未知 exp={exp}（可用: {sorted(EXP_SPEC)}）")
    if name not in spec["datasets"]:
        raise SystemExit(f"exp={exp} 数据集必须是 {spec['datasets']}，收到 {name}")
    if split not in spec["scen"]:
        raise SystemExit(f"exp={exp} 场景必须是 {spec['scen']}，收到 {split}")
    if spec["param"] is not None and param not in spec["param"]:
        raise SystemExit(f"exp={exp} 参数必须是 {spec['param']}，收到 {param}")
    if name not in OFFICIAL_SAMPLE:
        raise SystemExit(f"缺少官方样本量: {name}")
    if name not in DATASET_SEMANTICS:
        raise SystemExit(f"Missing variable semantics for dataset: {name}")


def run_one(exp: str, name: str, split: str, seed: int, param) -> None:
    _validate(exp, name, split, seed, param)
    spec = EXP_SPEC[exp]
    sample_size = OFFICIAL_SAMPLE[name]

    if exp == "nc":
        num_clients = int(param)
        strength = None
        tag = f"_nc{num_clients}"
        variant = f"nc{num_clients}"
    elif exp == "rounds1":
        num_clients = NUM_CLIENTS_BASE
        strength = STRENGTH_BASE if split == "heterogeneous" else None
        tag = "_rounds1"
        variant = "rounds1"
    else:  # strength
        num_clients = NUM_CLIENTS_BASE
        strength = float(param)
        tag = f"_strength{strength}"
        variant = f"strength{strength}"

    fold = f"{VERSION}_{name}_{split or 'homogeneous'}{tag}_seed{seed}"
    out_dir = os.path.join(BONUS_ROOT, spec["out"])
    fold_root = os.path.join(out_dir, fold)
    runs_path = os.path.join(out_dir, spec["runs"])

    cfg = build_config()
    if cfg["LLM"].get("active_provider") != BONUS_PROVIDER:
        cfg["LLM"]["active_provider"] = BONUS_PROVIDER      # 仅本进程生效，config.py 不动
    prov_cfg = (cfg["LLM"].get("providers") or {}).get(BONUS_PROVIDER) or {}
    if prov_cfg.get("api_key_env") and not os.environ.get(prov_cfg["api_key_env"]):
        raise SystemExit(f"环境变量 {prov_cfg['api_key_env']} 未设置，无法以 {BONUS_PROVIDER} 运行")

    cfg["DATA"].update(name=name, sample_size=sample_size, num_clients=num_clients,
                       seed=seed, split=split, sampling_seed=-1,
                       semantics=DATASET_SEMANTICS.get(name, {}))
    cfg["SEED"] = seed
    if name in EXTERNAL_DATA:
        cfg["DATA"].update(EXTERNAL_DATA[name])
    if strength is not None:
        cfg["DATA"]["heterogeneity_strength"] = strength
        cfg["HETEROGENEITY_STRENGTH"] = strength
    if exp == "rounds1":
        cfg["DEBATE_MAX_ROUNDS"] = 1                       # 中间点：0 轮=no_debate，2 轮=主算法
    cfg["EXECUTION"] = {"parallel": True, "max_workers": 1}  # LLM 串行（同主 driver 口径）
    cfg["OUTPUT_ROOT"] = fold_root
    cfg["LOGS_ROOT"] = os.path.join(fold_root, "logs")
    cc = dict(cfg.get("LLM_CALL") or {})
    cc["prompt_log_dir"] = os.path.join(fold_root, "prompts")
    cfg["LLM_CALL"] = cc

    resolved = dict(cfg["LLM"].get("default") or {})
    resolved.update(prov_cfg)
    print(f"===== {VERSION} additional fold: exp={exp} {name} {split or 'homogeneous'} seed={seed} "
          f"param={param} clients={num_clients} strength={strength} "
          f"provider={BONUS_PROVIDER} model={resolved.get('model')} =====", flush=True)
    t0 = time.time()
    res = main(cfg)
    res["dataset"] = name
    res["scenario"] = split or "homogeneous"
    res["seed"] = seed
    res["seconds"] = round(time.time() - t0, 1)
    res["exp"] = exp
    res["variant"] = variant
    res["num_clients"] = float(num_clients)
    if strength is not None:
        res["strength"] = float(strength)
    if exp == "rounds1":
        res["debate_max_rounds"] = 1
    res["provider"] = BONUS_PROVIDER
    res["model"] = resolved.get("model", "")

    os.makedirs(out_dir, exist_ok=True)
    idf = {"dataset": name, "scenario": split or "homogeneous", "seed": seed,
           "exp": exp, "num_clients": float(num_clients)}
    if exp == "strength":
        idf["strength"] = float(strength)
    if exp == "rounds1":
        idf["debate_max_rounds"] = 1
    if os.path.exists(runs_path):
        try:
            keep = []
            for ln in open(runs_path, encoding="utf-8"):
                if not ln.strip():
                    continue
                j = json.loads(ln)
                drop = all(j.get(k) == idf[k] for k in idf) and j.get("seed") == seed
                if not drop:
                    keep.append(ln)
        except (ValueError, TypeError, json.JSONDecodeError):
            keep = None
        if keep is not None:
            with open(runs_path, "w", encoding="utf-8") as f:
                f.writelines(keep)
    with open(runs_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(res, ensure_ascii=False, default=str) + "\n")
    print("RUN_DONE exp=%s dataset=%s scenario=%s seed=%d param=%s clients=%d "
          "f1=%.4f skel_f1=%.4f shd=%s llm=%s cand=%s dropped=%s "
          "pt=%s ct=%s tot=%s secs=%.1f" % (
              exp, name, split or "homogeneous", seed, param, num_clients,
              float(res.get("f1", 0) or 0), float(res.get("skel_f1", 0) or 0),
              res.get("shd", "?"), res.get("llm_calls", "?"),
              res.get("candidate_edges", "?"), res.get("dropped_edges", "?"),
              res.get("llm_prompt_tokens", "?"), res.get("llm_completion_tokens", "?"),
              res.get("llm_total_tokens", "?"), res["seconds"]), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Run one additional-analysis fold; paid API calls are possible.")
    ap.add_argument("exp", choices=tuple(EXP_SPEC))
    ap.add_argument("dataset", choices=tuple(sorted(set(NC_DATASETS + SR_DATASETS))))
    ap.add_argument("scenario", choices=("homogeneous", "heterogeneous"))
    ap.add_argument("seed", type=int, choices=(0, 1, 2))
    ap.add_argument("param", nargs="?", type=float)
    args = ap.parse_args()
    run_one(args.exp, args.dataset, args.scenario, args.seed, args.param)
