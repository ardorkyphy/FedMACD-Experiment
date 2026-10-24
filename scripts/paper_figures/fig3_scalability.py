# -*- coding: utf-8 -*-
"""Fig. 3 (panel b of the additional-analysis figure) -- accuracy scalability
with the number of data holders.

x = number of clients {4, 7, 12, 17} on a true linear axis (4 comes from the
main experiment, the rest from ``v4_36_bonus/algorithm_nc``).  Two discrete
datasets (alarm, 37 variables; child, 20 variables), homogeneous split, three
seeds each, glm-5.3-flash only.  Top panel = directed F1, bottom panel = SHD;
points are one marker per dataset, whiskers are +/- one seed sd.
"""
import os
import statistics as st

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as C

NCS = [4, 7, 12, 17]
DATASETS = [('alarm', 's'), ('child', 'o')]
STYLE = {'alarm': '#1f4e79', 'child': '#c0504d'}
# reference values from the manuscript
ANCHOR_F1 = {'alarm': [0.914, 0.919, 0.880, 0.922], 'child': [0.872, 0.884, 0.858, 0.855]}
FS = dict(label=11.0, tick=10.0, legend=10.0, letter=13.0)


def series(dataset, key):
    sweep = C.pick(C.bonus_rows('nc'), dataset=dataset)
    out = []
    for nc in NCS:
        rows = (C.pick(C.main_rows(), dataset=dataset, scenario='homogeneous')
                if nc == 4 else C.pick(sweep, variant='nc%d' % nc))
        vals = [C.fnum(r[key]) for r in rows]
        assert len(vals) == 3, (dataset, nc, len(vals))
        out.append((st.fmean(vals), st.stdev(vals)))
    return out


def draw_metric(ax, metric, fs=None, lw=1.6, ms=6.0, elw=1.0, capsize=3.0,
                grid_lw=0.6, legend=True, legend_loc='lower left', xlabel=True):
    """Draw one metric of the client sweep ('f1' or 'shd') into a single axis."""
    fs = dict(FS, **(fs or {}))
    f1 = {ds: series(ds, 'f1') for ds, _ in DATASETS}
    for ds, _ in DATASETS:
        for i, a in enumerate(ANCHOR_F1[ds]):
            assert abs(f1[ds][i][0] - a) < 5e-4, (ds, NCS[i], f1[ds][i][0], a)
    for ds, marker in DATASETS:
        data = f1[ds] if metric == 'f1' else series(ds, 'shd')
        mean = [m for m, _ in data]
        sd = [s for _, s in data]
        ax.errorbar(NCS, mean, yerr=sd, lw=lw, ms=ms, marker=marker, capsize=capsize,
                    elinewidth=elw, ecolor='0.35', color=STYLE[ds], label=ds, zorder=3)
    ax.set_ylabel('Directed F1' if metric == 'f1' else 'SHD (lower is better)',
                  fontsize=fs['label'])
    ax.set_xticks(NCS)
    if xlabel:
        ax.set_xlabel('Number of clients', fontsize=fs['label'])
    ax.grid(axis='y', ls='--', lw=grid_lw, color='0.85', zorder=0)
    ax.set_axisbelow(True)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    ax.tick_params(axis='y', labelsize=fs['tick'])
    ax.tick_params(axis='x', labelsize=fs['tick'])
    if legend:
        ax.legend(frameon=False, fontsize=fs['legend'], loc=legend_loc, ncol=1,
                  handletextpad=0.5, labelspacing=0.3)


def draw(axes, fs=None, letter=None, lw=1.6, ms=6.0, elw=1.0, capsize=3.0, grid_lw=0.6):
    """Standalone stacked version: F1 on top of SHD, one shared x axis."""
    fs = dict(FS, **(fs or {}))
    draw_metric(axes[0], 'f1', fs=fs, lw=lw, ms=ms, elw=elw, capsize=capsize,
                grid_lw=grid_lw, legend=True, xlabel=False)
    draw_metric(axes[1], 'shd', fs=fs, lw=lw, ms=ms, elw=elw, capsize=capsize,
                grid_lw=grid_lw, legend=False)
    axes[0].tick_params(axis='x', labelbottom=False)
    if letter is not None:
        axes[0].text(0.025, 0.965, '(%s)' % letter, transform=axes[0].transAxes, ha='left',
                     va='top', fontsize=fs['letter'], fontweight='bold', zorder=8)


def main():
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'pdf.fonttype': 42, 'ps.fonttype': 42,
                         'axes.unicode_minus': False, 'savefig.facecolor': 'white'})
    fig, axes = plt.subplots(2, 1, figsize=(7.0, 6.4), sharex=True,
                             gridspec_kw=dict(height_ratios=[1.45, 1.0]))
    draw(axes)
    fig.tight_layout()
    for ext in ('.pdf', '.png'):
        fig.savefig(os.path.join(C.ASSETS, 'fig3_scalability' + ext), dpi=300)
    plt.close(fig)
    print('saved paper_assets/fig3_scalability.pdf + .png')


if __name__ == '__main__':
    main()
