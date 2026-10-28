"""Reproduce the ten-dataset component-ablation heatmap (Fig. 5).

The input directory must contain ``main/glm-5.3-flash/v4_36_runs.jsonl`` and
``ablation/component_ablation.jsonl``.  Private experiment directories with the
corresponding v4_36 filenames are also accepted.  No experiment is executed.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
from pathlib import Path

DATASETS = ["asia", "child", "alarm", "survey", "earthquake", "Auto_MPG", "Sachs", "DWD_climate", "10nodes", "ER20"]
VARIANTS = [("Debate", "no_client_analyze"), ("Local", "no_debate"), ("Off", "no_client_analyze_no_debate")]
SPECS = [("f1", "Directed F1", 1, 0.3), ("precision", "Precision", 1, 0.4), ("recall", "Recall", 1, 0.3),
         ("skel_f1", "Skeleton F1", 1, 0.3), ("shd", "SHD", -1, 20), ("nshd", "nSHD", -1, 0.4)]


def _read(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_records(results_dir: Path):
    candidates = [
        (results_dir / "main/glm-5.3-flash/v4_36_runs.jsonl", results_dir / "ablation/component_ablation.jsonl"),
        (results_dir / "v4_36_final_zhipu/algorithm/v4_36_runs.jsonl", results_dir / "v4_36_final_zhipu/ablation/v4_36_ablation_runs.jsonl"),
    ]
    for full_path, ablation_path in candidates:
        if full_path.exists() and ablation_path.exists():
            full, ablated = _read(full_path), _read(ablation_path)
            break
    else:
        raise FileNotFoundError("Could not locate the full and ablation JSONL ledgers")
    if len(full) != 60 or len(ablated) != 180:
        raise ValueError(f"Expected 60 full and 180 ablation records, got {len(full)} and {len(ablated)}")
    return full, ablated


def summarize(full, ablated):
    rows = []
    for scene in ("homogeneous", "heterogeneous"):
        for dataset in DATASETS:
            f = [r for r in full if r["scenario"] == scene and r["dataset"] == dataset]
            if len(f) != 3 or {r["seed"] for r in f} != {0, 1, 2}:
                raise ValueError(f"full records are incomplete for {dataset}/{scene}")
            for label, variant in VARIANTS:
                a = [r for r in ablated if r["scenario"] == scene and r["dataset"] == dataset and r["variant"] == variant]
                if len(a) != 3 or {r["seed"] for r in a} != {0, 1, 2}:
                    raise ValueError(f"ablation records are incomplete for {dataset}/{scene}/{variant}")
                for metric, _, sign, _ in SPECS:
                    fm = statistics.fmean(r[metric] for r in f)
                    am = statistics.fmean(r[metric] for r in a)
                    rows.append({"scenario": scene, "dataset": dataset, "variant": label, "metric": metric,
                                 "full_mean": fm, "variant_mean": am, "delta": sign * (fm - am)})
    assert len(rows) == 360
    return rows


def render(rows, output_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    from matplotlib.patches import Rectangle

    plt.rcParams.update({"font.family": "Arial", "font.size": 8, "pdf.fonttype": 42,
                         "ps.fonttype": 42, "svg.fonttype": "none", "axes.unicode_minus": False})
    W, H = 150, 215
    fig = plt.figure(figsize=(W / 25.4, H / 25.4), facecolor="white")
    cmap = plt.get_cmap("RdBu_r")
    lefts = [30, 70, 110]
    for i, (metric, title, sign, limit) in enumerate(SPECS):
        rr, cc = divmod(i, 3)
        left, bottom = lefts[cc], 125 if rr == 0 else 20
        vals = [[next(r["delta"] for r in rows if r["scenario"] == scene and r["dataset"] == ds
                      and r["variant"] == label and r["metric"] == metric)
                 for label, _ in VARIANTS] for scene in ("homogeneous", "heterogeneous") for ds in DATASETS]
        if max(abs(v) for line in vals for v in line) > limit:
            raise ValueError(f"{metric} exceeds the fixed color scale")
        norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
        ax = fig.add_axes([left / W, bottom / H, 36 / W, 72 / H])
        ax.set_xlim(-.5, 2.5); ax.set_ylim(19.5, -.5)
        for y, line in enumerate(vals):
            for x, value in enumerate(line):
                rgb = cmap(norm(value))[:3]
                lum = sum(w * (c / 12.92 if c <= .04045 else ((c + .055) / 1.055) ** 2.4)
                          for w, c in zip((.2126, .7152, .0722), rgb))
                ax.add_patch(Rectangle((x - .5, y - .5), 1, 1, facecolor=rgb, edgecolor="none"))
                threshold = .05 if metric == "shd" else .0005
                label = ("0.0" if metric == "shd" else "0.000") if abs(value) < threshold else (f"{value:+.1f}" if metric == "shd" else f"{value:+.3f}")
                ax.text(x, y, label, ha="center", va="center", fontsize=8,
                        color="black" if (lum + .05) / .05 >= 4.5 else "white")
        ax.axhline(9.5, color="#555555", linewidth=.8)
        ax.set_xticks(range(3), [v[0] for v in VARIANTS], fontsize=8); ax.xaxis.tick_top()
        ax.set_yticks(range(20), DATASETS * 2 if cc == 0 else [""] * 20, fontsize=8)
        ax.tick_params(length=0, pad=2)
        for spine in ax.spines.values(): spine.set_visible(False)
        fig.text(left / W, (bottom + 79) / H, f"({chr(97 + i)}) {title}", fontsize=8, va="center")
        cb = fig.add_axes([(left + 6) / W, (bottom - 11) / H, 24 / W, 1.3 / H]); cb.set_xlim(-limit, limit); cb.set_ylim(0, 1)
        for j in range(64):
            step = 2 * limit / 64; v = -limit + (j + .5) * step
            cb.add_patch(Rectangle((-limit + j * step, 0), step, 1, facecolor=cmap(norm(v)), edgecolor="none"))
        fmt = (lambda v: str(int(v))) if metric == "shd" else (lambda v: f"{v:g}")
        cb.set_xticks([-limit, 0, limit], [fmt(-limit), "0", "+" + fmt(limit)], fontsize=8)
        cb.set_yticks([]); cb.tick_params(width=.55, length=2, pad=1)
        for spine in cb.spines.values(): spine.set_linewidth(.55)
        if cc == 0:
            fig.text(5 / W, (bottom + 54) / H, "Homogeneous", fontsize=8, rotation=90, ha="center", va="center")
            fig.text(5 / W, (bottom + 18) / H, "Heterogeneous", fontsize=8, rotation=90, ha="center", va="center")
    output_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "eps"):
        fig.savefig(output_dir / f"fig5_ablation.{ext}", dpi=450, facecolor="white")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path(os.environ.get("FEDMACD_RESULTS_DIR", "results")))
    parser.add_argument("--output-dir", type=Path, default=Path(os.environ.get("FEDMACD_OUTPUT_DIR", "fig5_ablation")))
    args = parser.parse_args()
    rows = summarize(*load_records(args.results_dir))
    render(rows, args.output_dir)
    print(f"wrote 360 cells to {args.output_dir}")


if __name__ == "__main__":
    main()
