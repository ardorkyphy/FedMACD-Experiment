# -*- coding: utf-8 -*-
"""Fig. 1 的主结果面板（单坐标轴）：横条形，y = 数据集（每套 6 根条），x = 指标值。

条长 = 该数据集 3 折均值，误差线 = 3 折样本标准差（matplotlib 以条端为中心对称画，
故左半落在条内）。每套数据集的 6 根条 = 本方案 3 个 LLM 底座（qwen3.8-flash /
deepseek-v4-flash / glm-5.3-flash）+ 该域的 3 条联邦基线（离散 FedPC/FedCSL/FedECD，
连续 FedPC/FedRCD-def/NOTEARS-ADMM）。

数值横排写在条末：每个数值独占自己那一行（行距 = 坐标区高 / 总行数），行距远大于数字
墨迹高度，所以数值之间不会压叠、也不需要任何连接线。数据集名横排写在本组 6 根条上方
的空档里（空档高 LAB_ROWH 行，比字高有余），既不压条也不越轴。

本模块只画「一个坐标轴」；2 行 x 4 列的拼版、场景前缀标题与共享图例在
fig1_composite.py。版面账本（印刷英寸/磅）也放在本模块，拼版脚本照抄，两处不各写一份。
"""
import csv
import json
import os
import statistics
import sys
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['pdf.fonttype'] = 42          # embed TrueType outlines, not Type 3
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['savefig.facecolor'] = 'white'
plt.rcParams['savefig.dpi'] = 300

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import _common as C

ASSETS = C.ASSETS
OUT = ASSETS
QWEN = C.main_result_path('qwen3.8-flash')
DS = C.main_result_path('deepseek-v4-flash')
GLM = C.main_result_path('glm-5.3-flash')

DISCRETE = ['asia', 'child', 'alarm', 'survey', 'earthquake']
CONTINUOUS = ['Auto_MPG', 'DWD_climate', 'Sachs', '10nodes', 'ER20']

COLOR = {
    'qwen3.8-flash': '#1f77b4', 'deepseek-v4-flash': '#2ca02c', 'glm-5.3-flash': '#ff8c1a',
    'FedPC': '#c0392b', 'FedCSL': '#9467bd', 'FedECD': '#17becf',
    'FedRCD-def': '#8c564b', 'NOTEARS-ADMM': '#7f7f7f',
}
EDGE = {
    'qwen3.8-flash': '#123a63', 'deepseek-v4-flash': '#155b35', 'glm-5.3-flash': '#8a4d00',
    'FedPC': '#6b1209', 'FedCSL': '#4d3270', 'FedECD': '#0e6a75',
    'FedRCD-def': '#4d2f26', 'NOTEARS-ADMM': '#3d3d3d',
}
LABEL = {
    'qwen3.8-flash': 'Ours (qwen3.8-flash)', 'deepseek-v4-flash': 'Ours (deepseek-v4-flash)',
    'glm-5.3-flash': 'Ours (glm-5.3-flash)', 'FedPC': 'FedPC', 'FedCSL': 'FedCSL',
    'FedECD': 'FedECD', 'FedRCD-def': 'FedRCD-def', 'NOTEARS-ADMM': 'NOTEARS-ADMM',
}
LEGEND_ORDER = ['qwen3.8-flash', 'deepseek-v4-flash', 'glm-5.3-flash', 'FedPC',
                'FedCSL', 'FedECD', 'FedRCD-def', 'NOTEARS-ADMM']
DOMAIN_METHODS = {
    'disc': ['qwen3.8-flash', 'deepseek-v4-flash', 'glm-5.3-flash', 'FedPC', 'FedCSL', 'FedECD'],
    'cont': ['qwen3.8-flash', 'deepseek-v4-flash', 'glm-5.3-flash', 'FedPC', 'FedRCD-def', 'NOTEARS-ADMM'],
}


def _num(x):
    if x is None:
        return None
    s = str(x).strip()
    return None if (not s or s.lower() in ('nan', 'none')) else float(x)


def jsonl_runs(path):
    d = defaultdict(list)
    for line in open(path, encoding='utf-8'):
        r = json.loads(line)
        d[(r['dataset'], r['scenario'])].append(r)
    return d


def csv_runs(path, alg=None):
    d = defaultdict(list)
    for r in csv.DictReader(open(path, encoding='utf-8')):
        if alg and r.get('algorithm', r.get('algorithm_selection')) != alg:
            continue
        d[(r['dataset'], r['scenario'])].append(r)
    return d


def cell_stats(runs, key, missing_to_zero=False):
    out = {}
    for k, rows in runs.items():
        vals = []
        for r in rows:
            v = _num(r[key])
            if v is None:
                v = 0.0 if missing_to_zero else None
            if v is not None:
                vals.append(v)
        out[k] = (statistics.fmean(vals), statistics.stdev(vals) if len(vals) > 1 else 0.0)
    return out


def build_cells():
    q, d, g = jsonl_runs(QWEN), jsonl_runs(DS), jsonl_runs(GLM)
    bl = {
        'FedPC': csv_runs(C.BASELINES['FedPC'][0]),
        'FedCSL': csv_runs(C.BASELINES['FedCSL'][0]),
        'FedECD': csv_runs(C.BASELINES['FedECD'][0]),
        'FedRCD-def': csv_runs(C.BASELINES['FedRCD-def'][0], 'fedrcd_def'),
        'NOTEARS-ADMM': csv_runs(C.BASELINES['NOTEARS-ADMM'][0]),
    }
    data = {}
    for name, runs in {'qwen3.8-flash': q, 'deepseek-v4-flash': d, 'glm-5.3-flash': g}.items():
        f1, shd = cell_stats(runs, 'f1'), cell_stats(runs, 'shd')
        data[name] = {k: (f1[k], shd[k]) for k in f1}
    for name, runs in bl.items():
        f1 = cell_stats(runs, 'f1', missing_to_zero=(name in ('FedCSL', 'FedECD')))
        shd = cell_stats(runs, 'shd')
        data[name] = {k: (f1[k], shd[k]) for k in f1}
    return data, q


def total_mean(runs, key):
    vals = [_num(r[key]) for ds in DISCRETE + CONTINUOUS
            for r in runs[(ds, 'homogeneous')] + runs[(ds, 'heterogeneous')]]
    return statistics.fmean([v for v in vals if v is not None])


def sanity(data, q):
    # Reference values from the manuscript
    assert abs(total_mean(q, 'f1') - 0.7433) < 1e-4
    assert abs(total_mean(q, 'shd') - 8.67) < 1e-2
    assert abs(data['FedPC'][('DWD_climate', 'homogeneous')][0][0] - 0.4396) < 1e-3
    print('sanity ok')


_SCAN = {}


def scan_limits(data):
    """按（域, 指标）在两个场景上取同一个 x 轴上限，两行才可逐列对读。"""
    for dom, ds_list in (('disc', DISCRETE), ('cont', CONTINUOUS)):
        for metric, idx in (('F1', 0), ('SHD', 1)):
            vmax = 0.0
            for scen in ('homogeneous', 'heterogeneous'):
                for mname in DOMAIN_METHODS[dom]:
                    for ds in ds_list:
                        m, s = data[mname][(ds, scen)][idx]
                        vmax = max(vmax, m + s)
            _SCAN[(dom, metric)] = vmax


# ---------------------------------------------------------------------------
# 版面账本（印刷单位）。fig1_composite.py 只许照抄，不许另写一套数字。
# ---------------------------------------------------------------------------
# 2026-10-02：原来 4x2（两场景叠在一张画布）在正文栏宽下只有 4.3 pt，即使插满
# \textwidth 也只有 5.8 pt：8 个面板 x 5 套数据集 x 6 条 = 73 行文字放进一页的
# 高度里，任何字号都排不下。改成一场景一张画布（4x1），一页放得下，字号 7.8 pt，
# 插满 \textwidth 后仍是 7.5 pt（>= 7 pt 的印刷下限）。
CANVAS_W_IN, CANVAS_H_IN = 6.7717, 6.20      # 正文栏宽 1:1 / 单场景一行四列
NCOL, NROW = 4, 1                            # 列 = 域 x 指标；每张画布一个场景
LEG_H_IN = 0.66                              # 底部共享图例条高
ROW_GAP_IN = 0.0                             # 单行画布没有行间空白
TILE_W_IN = CANVAS_W_IN / NCOL
TILE_H_IN = (CANVAS_H_IN - LEG_H_IN - ROW_GAP_IN) / NROW
PAD_L_IN, PAD_R_IN, PAD_T_IN, PAD_B_IN = 0.09, 0.05, 0.32, 0.42
AX_W_IN = TILE_W_IN - PAD_L_IN - PAD_R_IN
AX_H_IN = TILE_H_IN - PAD_T_IN - PAD_B_IN

FS_VALUE, FS_TICK, FS_AX, FS_TITLE, FS_LEGEND = 7.8, 7.8, 9.0, 8.2, 9.0  # 印刷磅
TITLE_X_IN = 0.03    # 面板标题相对坐标区左边界再往左挪一点（瓦片左边缘之内），
                     # 否则最右一列的标题会顶出画布（实测 6.6 pt 时超 1.7 pt）
BARR = 0.69          # 条厚 / 行距（专利图观感：条宽 : 条缝 约 2.2 : 1）
GGAP = 1.2           # 相邻两套数据集之间空出的行数
LAB_ROWH = 1.05      # 数据集名占用的行高（7.8 pt 字约 1.0 行）
Y_TOP_ROW = -(GGAP / 2.0 + LAB_ROWH + 0.05)     # 轴顶（行坐标）
NAME_DY = GGAP / 2.0 + LAB_ROWH / 2.0           # 数据集名中心相对本组第一根条的行偏移
VALUE_GAP = 0.012    # 数值离条端的空隙（取 x 上限的比例）
CAPSIZE = 1.2        # 误差线端帽长（磅）
EDGE_LW, ERR_LW, GRID_LW, SPINE_LW, TICK_LW = 0.5, 0.6, 0.35, 0.5, 0.5

TICK_STEP = {'F1': 0.2, 'SHD': 5.0}          # x 轴刻度间隔
AX_LABEL = {'F1': 'Directed F1', 'SHD': 'SHD'}
FMT = {'F1': '{:.2f}', 'SHD': '{:.1f}'}
DOM_EN = {'disc': 'Discrete', 'cont': 'Continuous'}
SCEN_EN = {'homogeneous': 'Homogeneous', 'heterogeneous': 'Heterogeneous'}
SCENARIOS = ['homogeneous', 'heterogeneous']
PANELS = [('disc', 'F1'), ('cont', 'F1'), ('disc', 'SHD'), ('cont', 'SHD')]
PANEL_LETTERS = 'abcdefgh'


def _measure_pt(fig, s, fs, weight='normal'):
    """一个字符串印出来多宽（磅）。不落盘、不改变画布上已有内容。"""
    t = fig.text(0.0, 0.0, s, fontsize=fs, fontweight=weight)
    w = t.get_window_extent(renderer=fig.canvas.get_renderer()).width / fig.dpi * 72.0
    t.remove()
    return w


_W_CACHE = {}


def value_foot(fig, dom, metric, data):
    """条末数值需要的空地（磅）：最宽的那个数值串 + 一点空隙。"""
    key = (dom, metric)
    if key not in _W_CACHE:
        ds_list = DISCRETE if dom == 'disc' else CONTINUOUS
        idx = 0 if metric == 'F1' else 1
        widest = 0.0
        for scen in SCENARIOS:
            for mname in DOMAIN_METHODS[dom]:
                for ds in ds_list:
                    mu = data[mname][(ds, scen)][idx][0]
                    widest = max(widest, _measure_pt(fig, FMT[metric].format(mu), FS_VALUE))
        _W_CACHE[key] = widest + 2.5
    return _W_CACHE[key]


def x_limit(dom, metric, foot_pt):
    """x 轴上限：刻度顶再往右留出 foot_pt 磅的空地，条末数值才不会出轴。"""
    step = TICK_STEP[metric]
    tick_top = float(np.ceil(_SCAN[(dom, metric)] / step) * step)
    plot_pt = AX_W_IN * 72.0
    return tick_top / (1.0 - foot_pt / plot_pt), tick_top, step


def panel(ax, data, scenario, dom, metric, letter):
    """在 ax 上画一个面板，返回自检需要的图元（条、数值、数据集名）。"""
    methods = DOMAIN_METHODS[dom]
    ds_list = DISCRETE if dom == 'disc' else CONTINUOUS
    idx = 0 if metric == 'F1' else 1
    fmt = FMT[metric]
    nm, npd = len(methods), len(ds_list)

    xmax, tick_top, step = x_limit(dom, metric, value_foot(ax.figure, dom, metric, data))
    row_bottom = (npd - 1) * (nm + GGAP) + (nm - 1)

    bars, values, names = [], [], []
    for i, ds in enumerate(ds_list):
        base = i * (nm + GGAP)
        for j, mname in enumerate(methods):
            mu, sd = data[mname][(ds, scenario)][idx]
            y = base + j
            cont = ax.barh(y, mu, height=BARR, xerr=sd, capsize=CAPSIZE,
                                color=COLOR[mname], edgecolor=EDGE[mname],
                                linewidth=EDGE_LW, zorder=3,
                                error_kw=dict(ecolor='0.25', elinewidth=ERR_LW,
                                              capthick=ERR_LW))
            bars.extend(cont.patches)
            values.append(ax.text(mu + VALUE_GAP * xmax, y, fmt.format(mu),
                                  va='center', ha='left', fontsize=FS_VALUE,
                                  color='#1a1a1a', zorder=5,
                                  bbox=dict(boxstyle='square,pad=0.05',
                                            facecolor='white', edgecolor='none')))
        names.append(ax.text(VALUE_GAP * xmax, base - NAME_DY, ds, va='center', ha='left',
                             fontsize=FS_TICK, color='0.15', zorder=5))

    ax.set_ylim(row_bottom + GGAP / 2.0, Y_TOP_ROW)      # y 反号：第一套数据集在最上面
    ax.set_yticks([])
    ax.set_xlim(0.0, xmax)
    ax.set_xticks(np.arange(0.0, tick_top + 1e-9, step))
    ax.tick_params(axis='x', labelsize=FS_TICK, length=1.6, width=TICK_LW, pad=1.5)
    ax.set_xlabel(AX_LABEL[metric], fontsize=FS_AX, labelpad=1.5)
    ax.grid(axis='x', ls='--', lw=GRID_LW, color='0.86', zorder=0)
    ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_linewidth(SPINE_LW)
    # 场景已经由「一张画布一个场景」表达（图注里写明），标题只留域，标题才放得下。
    ax.set_title('(%s) %s' % (letter, DOM_EN[dom]),
                 loc='left', x=-TITLE_X_IN / AX_W_IN, fontsize=FS_TITLE,
                 fontweight='bold', pad=2.5)
    return {'bars': bars, 'values': values, 'names': names,
            'xmax': xmax, 'tick_top': tick_top}


def row_pitch_pt(dom='disc'):
    """行距（磅）：坐标区高 / 总行数。八个面板都是 5 套数据集 x 6 条，故只有一个数。"""
    ds_list = DISCRETE if dom == 'disc' else CONTINUOUS
    nm = len(DOMAIN_METHODS[dom])
    rows = (len(ds_list) - 1) * (nm + GGAP) + (nm - 1) + GGAP / 2.0 - Y_TOP_ROW
    return AX_H_IN * 72.0 / rows


def legend_handles():
    """8 条图例项，顺序固定，供合成图统一画。"""
    return [Patch(facecolor=COLOR[m], edgecolor=EDGE[m], label=LABEL[m]) for m in LEGEND_ORDER]


if __name__ == '__main__':
    data, q = build_cells()
    sanity(data, q)
    scan_limits(data)
    print('geometry: canvas %.4f x %.2f in, tile %.4f x %.4f in, axes %.4f x %.4f in'
          % (CANVAS_W_IN, CANVAS_H_IN, TILE_W_IN, TILE_H_IN, AX_W_IN, AX_H_IN))
    fig = plt.figure(figsize=(CANVAS_W_IN, CANVAS_H_IN))
    for dom, metric in PANELS:
        xmax, tick_top, step = x_limit(dom, metric, value_foot(fig, dom, metric, data))
        pitch = row_pitch_pt(dom)
        foot = value_foot(fig, dom, metric, data)
        print('  %-4s %-3s  scan %.3f -> tick %.1f, xmax %.3f | axes %.1f pt, '
              'foot %.1f pt (%.0f%%) | row pitch %.2f pt, bar %.2f pt, gap %.2f pt'
              % (dom, metric, _SCAN[(dom, metric)], tick_top, xmax, AX_W_IN * 72.0,
                 foot, 100.0 * foot / (AX_W_IN * 72.0),
                 pitch, BARR * pitch, (1.0 - BARR) * pitch))
    plt.close(fig)
