"""Reconstruct Bayesian-network client datasets under the paper configuration.

Each fold contains four client CSV files whose row counts sum to 10,000.
Homogeneous folds use one forward-sampled instance followed by a random split.
Heterogeneous folds independently sample each client after its soft interventions.
These are regenerated samples, not verified byte-identical historical run inputs.
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src_v4"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from util.data import DataService  # noqa: E402


DATASETS = ("asia", "child", "alarm", "survey", "earthquake")
SCENARIOS = ("homogeneous", "heterogeneous")
SEEDS = (0, 1, 2)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def generate(output_root: Path) -> list[dict]:
    output_root.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    service = DataService()
    for dataset in DATASETS:
        for scenario in SCENARIOS:
            for seed in SEEDS:
                cfg = {
                    "name": dataset,
                    "format": "bif",
                    "discrete": True,
                    "sample_size": 10_000,
                    "num_clients": 4,
                    "split": scenario,
                    "seed": seed,
                    "sampling_seed": -1,
                    "heterogeneity_strength": 0.7,
                }
                loaded = service.load(cfg)
                fold = output_root / dataset / scenario / f"seed_{seed}"
                fold.mkdir(parents=True, exist_ok=True)
                files = []
                for client_id, part in enumerate(loaded.parts):
                    path = fold / f"client_{client_id}.csv"
                    with path.open("w", encoding="utf-8", newline="") as fh:
                        writer = csv.writer(fh)
                        writer.writerow(loaded.columns)
                        writer.writerows(part.tolist())
                    files.append({
                        "client": client_id,
                        "rows": int(part.shape[0]),
                        "columns": int(part.shape[1]),
                        "path": path.relative_to(output_root).as_posix(),
                        "sha256": _sha256(path),
                    })
                record = {
                    "artifact_status": "regenerated_under_paper_configuration",
                    "historical_input_identity_verified": False,
                    "dataset": dataset,
                    "scenario": scenario,
                    "seed": seed,
                    "sample_size_total": sum(item["rows"] for item in files),
                    "num_clients": 4,
                    "sampling_seed": -1,
                    "heterogeneity_strength": 0.7 if scenario == "heterogeneous" else None,
                    "construction": (
                        "forward sampling followed by a disjoint random split"
                        if scenario == "homogeneous"
                        else "independent client forward sampling after client-specific soft interventions"
                    ),
                    "files": files,
                }
                (fold / "config.json").write_text(
                    json.dumps(record, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                records.append(record)
    (output_root / "MANIFEST.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return records


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "release" / "bn_client_data"
    rows = generate(target)
    print(f"Exported {len(rows)} Bayesian-network folds to {target}")
