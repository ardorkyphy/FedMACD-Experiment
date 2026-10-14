# -*- coding: utf-8 -*-
"""v4 单折真实 LLM 冒烟（docs/代码框架设计.md v4 章节，对齐 v3 scripts/run_v2_smoke.ps1）。

用法:
  python src_v4/experiment/run_smoke.py                          # 默认 asia/600/3/manual+pc
  python src_v4/experiment/run_smoke.py --dataset child --sample 300 --clients 2
  python src_v4/experiment/run_smoke.py --algorithm pc --selection llm   # 走真实选算法

验证点：选算法（manual 跳过 / llm 真实推理）、client 三件套解析、辩论两包、
裁判全员分批终裁、llm_calls 计数、结果落盘与审计产物。
"""
import argparse
import json
import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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


def main_smoke(args) -> None:
    cfg = build_config()
    sem = (cfg["DATA"].get("semantics") or {})
    if not sem and args.semantics:
        from experiment.run_rep5 import DATASET_SEMANTICS
        sem = DATASET_SEMANTICS.get(args.dataset, {})
    cfg["DATA"].update(name=args.dataset, sample_size=args.sample,
                       num_clients=args.clients, seed=args.seed, split="",
                       semantics=sem)
    cfg["ALGORITHM_SELECTION"] = args.selection
    cfg["ALGORITHM_OVERRIDE"] = args.algorithm if args.selection == "manual" else ""
    cfg["EXECUTION"] = {"parallel": True, "max_workers": int(args.clients)}
    t0 = time.time()
    res = main(cfg)
    print("SMOKE_DONE seconds=%.1f" % (time.time() - t0))
    print(json.dumps({k: v for k, v in res.items()
                      if not isinstance(v, (dict, list))},
                     ensure_ascii=False, default=str))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="asia")
    ap.add_argument("--sample", type=int, default=600)
    ap.add_argument("--clients", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--selection", default="manual", choices=["manual", "llm"])
    ap.add_argument("--algorithm", default="pc")
    ap.add_argument("--semantics", action="store_true",
                    help="数据集无默认语义时从 run_rep5.DATASET_SEMANTICS 取")
    args = ap.parse_args()
    main_smoke(args)