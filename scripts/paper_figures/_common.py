# -*- coding: utf-8 -*-
"""Shared readers for the paper figures and tables.

The readers use the curated Zenodo layout. Set ``FEDMACD_RESULTS_DIR`` to the
archive's results directory. ``FEDMACD_OUTPUT_DIR`` selects output.

* main per-fold results     ``v4_36_final[_qwen|_zhipu]/algorithm/v4_36_runs.jsonl``
* ablation per-fold results ``v4_36_final[_qwen|_zhipu]/ablation/v4_36_ablation_runs.jsonl``
* federated baselines       ``v4_36_final/baseline/<method>/results/*.csv``
* additional analyses       ``v4_36_bonus/{ablation_strength,ablation_rounds,algorithm_nc}/*.jsonl``
* orientation ledger        ``v4_36_final_zhipu/analysis/v4_36_edge_contribution_zhipu.csv``

Nothing here re-runs an experiment and nothing here hard-codes a measured
value: every number is computed from the per-fold records above.
"""
from __future__ import annotations

import csv
import json
import os
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, '..', '..'))
RES = os.environ.get('FEDMACD_RESULTS_DIR', os.path.join(ROOT, 'results'))
if not os.path.isdir(os.path.join(RES, 'main')):
    raise RuntimeError('Set FEDMACD_RESULTS_DIR to the extracted Zenodo results directory')
ARCHIVE_MODE = True
ASSETS = os.environ.get('FEDMACD_OUTPUT_DIR', os.path.join(ROOT, 'reproduced_outputs'))
TABLE_DIR = os.path.join(ASSETS, 'tables')
PANEL_DIR = os.path.join(ASSETS, 'fig1_panels')
os.makedirs(TABLE_DIR, exist_ok=True)
os.makedirs(PANEL_DIR, exist_ok=True)

DISCRETE = ['asia', 'child', 'alarm', 'survey', 'earthquake']
CONTINUOUS = ['Auto_MPG', 'DWD_climate', 'Sachs', '10nodes', 'ER20']
DATASETS = DISCRETE + CONTINUOUS
DOMAIN_OF = dict([(d, 'discrete') for d in DISCRETE] + [(d, 'continuous') for d in CONTINUOUS])

# display name -> result sub-directory (paper main backbone = glm-5.3-flash)
BACKBONES = {
    'qwen3.8-flash': 'qwen3.8-flash' if ARCHIVE_MODE else 'v4_36_final_qwen',
    'deepseek-v4-flash': 'deepseek-v4-flash' if ARCHIVE_MODE else 'v4_36_final',
    'glm-5.3-flash': 'glm-5.3-flash' if ARCHIVE_MODE else 'v4_36_final_zhipu',
}
BACKBONE_ORDER = ['qwen3.8-flash', 'deepseek-v4-flash', 'glm-5.3-flash']
MAIN_BACKBONE = 'glm-5.3-flash'

# ablation / additional analyses exist for glm-5.3-flash only (plan D15)
ABLATION_DATASETS = ['child', 'alarm', 'Sachs']
ABL_SETTINGS = ['Ours', 'no_client_analyze', 'no_debate', 'no_client_analyze_no_debate']
ABL_LABEL = {
    'Ours': 'Ours',
    'no_client_analyze': 'w/o client',
    'no_debate': 'w/o debate',
    'no_client_analyze_no_debate': 'w/o both',
}

if ARCHIVE_MODE:
    BONUS = {
        'strength': os.path.join(RES, 'additional', 'heterogeneity_strength.jsonl'),
        'rounds1': os.path.join(RES, 'additional', 'debate_rounds_1.jsonl'),
        'rounds3': os.path.join(RES, 'additional', 'debate_rounds_3.jsonl'),
        'nc': os.path.join(RES, 'additional', 'client_count.jsonl'),
    }
    EDGE_CSV = os.path.join(RES, 'additional', 'direction_contribution.csv')
else:
    BONUS = {
        'strength': os.path.join(RES, 'v4_36_bonus', 'ablation_strength', 'v4_36_runs_strength.jsonl'),
        'rounds1': os.path.join(RES, 'v4_36_bonus', 'ablation_rounds', 'v4_36_runs_rounds1.jsonl'),
        'rounds3': os.path.join(RES, 'v4_36_bonus', 'ablation_rounds', 'v4_36_runs_rounds3.jsonl'),
        'nc': os.path.join(RES, 'v4_36_bonus', 'algorithm_nc', 'v4_36_runs_nc.jsonl'),
    }
    EDGE_CSV = os.path.join(RES, 'v4_36_final_zhipu', 'analysis',
                            'v4_36_edge_contribution_zhipu.csv')

# name -> (csv path, algorithm filter, treat undefined F1 as 0)
_B = os.path.join(RES, 'baselines') if ARCHIVE_MODE else os.path.join(RES, 'v4_36_final', 'baseline')
_MID = '' if ARCHIVE_MODE else 'results'
def _baseline(method, filename):
    return os.path.join(_B, method, _MID, filename) if _MID else os.path.join(_B, method, filename)


BASELINES = {
    'FedPC': (_baseline('fedpc', 'v4_36_baseline_fedpc_results_n10000_4clients.csv'), None, False),
    'FedCSL': (_baseline('fedcsl', 'v4_36_baseline_fedcsl_results_n10000_4clients.csv'), None, True),
    'FedECD': (_baseline('fedecd', 'v4_36_baseline_fedecd_results_n10000_4clients.csv'), None, True),
    'FedRCD-def': (_baseline('fedrcd', 'v4_36_baseline_fedlingam_results_official.csv'), 'fedrcd_def', False),
    'FedRCD-ni': (_baseline('fedrcd', 'v4_36_baseline_fedlingam_results_official.csv'), 'fedrcd_ni', False),
    'NOTEARS-ADMM': (_baseline('notears_admm', 'v4_36_baseline_notears_admm_results_official.csv'), None, False),
    'Centralized-PC': (_baseline('centralized_pc_fci', 'v4_36_baseline_centralized_results_official.csv'), 'pc', False),
    'Centralized-FCI': (_baseline('centralized_pc_fci', 'v4_36_baseline_centralized_results_official.csv'), 'fci', False),
    'GES': (_baseline('ges_lingam', 'v4_36_baseline_ges_lingam_results.csv'), 'ges', False),
    'LiNGAM': (_baseline('ges_lingam', 'v4_36_baseline_ges_lingam_results.csv'), 'lingam', False),
}


def main_result_path(backbone=MAIN_BACKBONE):
    if ARCHIVE_MODE:
        return os.path.join(RES, 'main', BACKBONES[backbone], 'v4_36_runs.jsonl')
    return os.path.join(RES, BACKBONES[backbone], 'algorithm', 'v4_36_runs.jsonl')


# --------------------------------------------------------------------------- io
def load_jsonl(path):
    with open(path, encoding='utf-8') as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_csv(path, algorithm=None):
    out = []
    with open(path, encoding='utf-8') as fh:
        for row in csv.DictReader(fh):
            if algorithm is not None:
                alg = row.get('algorithm', row.get('algorithm_selection'))
                if alg != algorithm:
                    continue
            out.append(row)
    return out


def load_csv_numeric(path, algorithm=None):
    out = []
    for row in load_csv(path, algorithm):
        out.append({k: (v if k in ('dataset', 'scenario') else fnum(v)) for k, v in row.items()})
    return out


def fnum(x):
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return None if x != x else float(x)
    s = str(x).strip()
    return None if s.lower() in ('', 'nan', 'none', 'null') else float(s)


# ------------------------------------------------------------------ selection
def pick(rows, **conds):
    out = rows
    for key, want in conds.items():
        if want is None:
            continue
        if isinstance(want, (list, tuple, set)):
            allowed = {str(w) for w in want}
            out = [r for r in out if str(r.get(key)) in allowed]
        else:
            out = [r for r in out if str(r.get(key)) == str(want)]
    return out


def seeds_of(rows):
    return sorted({int(r['seed']) for r in rows})


# -------------------------------------------------------------------- metrics
def micro_f1(rows, missing_to_zero=False):
    """Directed F1 rebuilt from the fold-level confusion counts."""
    tp = fp = fn = 0.0
    for r in rows:
        t, p, n = fnum(r.get('tp')), fnum(r.get('fp')), fnum(r.get('fn'))
        if t is None or p is None or n is None:
            if not missing_to_zero:
                continue
            t, p, n = 0.0, 0.0, 0.0
        tp, fp, fn = tp + t, fp + p, fn + n
    den = 2 * tp + fp + fn
    return 0.0 if den == 0 else 2 * tp / den


def mean_of(rows, key):
    vals = [v for v in (fnum(r.get(key)) for r in rows) if v is not None]
    return st.fmean(vals) if vals else float('nan')


def sd_of(rows, key):
    vals = [v for v in (fnum(r.get(key)) for r in rows) if v is not None]
    return st.stdev(vals) if len(vals) > 1 else 0.0


def d_acc(rows):
    """Orientation-stage contribution rate over the pooled edge ledger."""
    corr = sum(r['D_correct'] for r in rows)
    den = sum(r['D_den'] for r in rows)
    return corr / den if den else float('nan')


# ------------------------------------------------------------------ main data
_CACHE = {}


def main_rows(backbone=MAIN_BACKBONE):
    key = ('main', backbone)
    if key not in _CACHE:
        _CACHE[key] = load_jsonl(main_result_path(backbone))
    return _CACHE[key]


def ablation_rows(backbone=MAIN_BACKBONE):
    key = ('abl', backbone)
    if key not in _CACHE:
        path = (os.path.join(RES, 'ablation', 'component_ablation.jsonl') if ARCHIVE_MODE
                else os.path.join(RES, BACKBONES[backbone], 'ablation',
                                  'v4_36_ablation_runs.jsonl'))
        _CACHE[key] = load_jsonl(path)
    return _CACHE[key]


def bonus_rows(name):
    key = ('bonus', name)
    if key not in _CACHE:
        _CACHE[key] = load_jsonl(BONUS[name])
    return _CACHE[key]


def edge_rows():
    if 'edge' not in _CACHE:
        rows = []
        with open(EDGE_CSV, encoding='utf-8') as fh:
            for row in csv.DictReader(fh):
                rows.append({k: (v if k in ('variant', 'scenario', 'dataset', 'audit')
                                 else (int(float(v)) if k == 'seed' else fnum(v)))
                             for k, v in row.items()})
        _CACHE['edge'] = rows
    return _CACHE['edge']


def baseline_rows(name):
    key = ('base', name)
    if key not in _CACHE:
        path, alg, _ = BASELINES[name]
        _CACHE[key] = load_csv_numeric(path, alg)
    return _CACHE[key]


def main_cell(backbone, dataset, scenario):
    return pick(main_rows(backbone), dataset=dataset, scenario=scenario)


# ------------------------------------------------------ pooled ablated metrics
def ablation_setting_rows(setting, scenario=None, backbone=MAIN_BACKBONE):
    """Per-fold rows of one ablation setting over child / alarm / Sachs."""
    if setting == 'Ours':
        return pick(main_rows(backbone), dataset=ABLATION_DATASETS, scenario=scenario)
    return pick(ablation_rows(backbone), variant=setting, dataset=ABLATION_DATASETS,
                scenario=scenario)


def ablation_pooled(setting, metric, scenario=None, backbone=MAIN_BACKBONE):
    """(pooled value, seed sd) of one ablation cell.

    ``scenario=None`` pools homogeneous + heterogeneous together (the "both"
    row of the summary table).  The point value is the quantity pooled over
    *every* fold of the cell (9 folds, or 18 for the "both" row); the reported
    sd is the seed-to-seed spread of the same pooled quantity restricted to a
    single seed (n = 3).
    """
    rows = ablation_setting_rows(setting, scenario, backbone)
    led = pick(edge_rows(), variant=setting, dataset=ABLATION_DATASETS, scenario=scenario)

    def one(sub, lsub):
        if metric == 'D_acc':
            return d_acc(lsub)
        if metric == 'F1':
            return micro_f1(sub)
        if metric == 'SHD':
            return st.fmean([fnum(r['shd']) for r in sub])
        if metric == 'nSHD':
            return st.fmean([fnum(r['nshd']) for r in sub])
        raise KeyError(metric)

    per_seed = [one(pick(rows, seed=s), pick(led, seed=s)) for s in seeds_of(rows)]
    value = one(rows, led)
    return value, (st.stdev(per_seed) if len(per_seed) > 1 else 0.0)


# ----------------------------------------------------------------- formatting
def fmt_pm(value, sd, nd=3):
    return '%.*f $\\pm$ %.*f' % (nd, value, nd, sd)


def fmt_delta(value, nd=3):
    return '%+.*f' % (nd, value)
