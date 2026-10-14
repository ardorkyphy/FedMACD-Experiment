# -*- coding: utf-8 -*-
"""Fig. 2 (panel a of the additional-analysis figure) -- sensitivity to the
degree of data heterogeneity.

x = stratification strength {0.3, 0.5, 0.7, 0.9}; the default operating point
0.7 comes straight from the main experiment, the other three from the strength
sweep under ``v4_36_bonus/ablation_strength``.  Two datasets (DWD_climate,
survey), heterogeneous split, three seeds each, glm-5.3-flash only.
Lines are mean directed F1; the shaded band is +/- one seed sd.
"""
import os
import statistics as st

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as C

STRENGTHS = [0.3, 0.5, 0.7, 0.9]
DATASETS = ['DWD_climate', 'survey']
STYLE = {
    'DWD_climate': dict(color='#1f4e79', marker='s', label='DWD_climate'),
    'survey': dict(color='#c0504d', marker='o', label='survey'),
}
# anchors, plan of record section 11.4
ANCHOR = {
    'DWD_climate': [0.573, 0.653, 0.667, 0.621],
    'survey': [0.711, 0.665, 0.711, 0.656],
}
# SHD of the very same folds (recomputed from the jsonl and logged to
# logs/paper_figures/strength_shd_*.log before being pinned here)
ANCHOR_SHD = {
    'DWD_climate': [6.3333, 5.3333, 4.6667, 5.6667],
    'survey': [2.6667, 3.6667, 2.6667, 3.0000],
}
METRICS = {
    'f1': dict(key='f1', ylab='Directed F1', ylim=(0.35, 0.90),
               yticks=[0.4, 0.5, 0.6, 0.7, 0.8, 0.9], anchor=ANCHOR,
               legend_loc='lower right'),
    'shd': dict(key='shd', ylab='SHD (lower is better)', ylim=(0, 9.5),
                yticks=[0, 2, 4, 6, 8], anchor=ANCHOR_SHD,
                legend_loc='upper right'),
}
FS = dict(label=11.0, tick=10.0, legend=10.0, letter=13.0)


def series(dataset, metric='f1'):
    key = METRICS[metric]['key']
    sweep = C.pick(C.bonus_rows('strength'), dataset=dataset)
    out = []
    for s in STRENGTHS:
        rows = (C.pick(C.main_rows(), dataset=dataset, scenario='heterogeneous')
                if s == 0.7 else C.pick(sweep, strength=s))
        vals = [C.fnum(r[key]) for r in rows]
        assert len(vals) == 3, (dataset, s, len(vals))
        out.append((st.fmean(vals), st.stdev(vals)))
    return out


def draw(ax, fs=None, letter=None, metric='f1', lw=2.0, ms=6.5, grid_lw=0.6):
    """Draw the strength panel into ``ax`` (shared by the standalone and merged file).

    ``lw`` / ``ms`` / ``grid_lw`` default to the standalone sizes; the merged
    figure passes smaller ones so that the strokes match Fig. 3 and Fig. 5 on the
    same 6.77 in text-width canvas.
    """
    fs = dict(FS, **(fs or {}))
    M = METRICS[metric]
    for ds in DATASETS:
        data = series(ds, metric)
        mean = [m for m, _ in data]
        sd = [s for _, s in data]
        for i, a in enumerate(M['anchor'][ds]):
            assert abs(mean[i] - a) < 5e-4, (ds, STRENGTHS[i], mean[i], a)
        ax.plot(STRENGTHS, mean, lw=lw, ms=ms, zorder=3, **STYLE[ds])
        ax.fill_between(STRENGTHS, [m - s for m, s in data], [m + s for m, s in data],
                        color=STYLE[ds]['color'], alpha=0.16, lw=0, zorder=2)
    ax.set_xlabel('Heterogeneity strength', fontsize=fs['label'])
    ax.set_ylabel(M['ylab'], fontsize=fs['label'])
    ax.set_xticks(STRENGTHS)
    ax.set_xticklabels(['%.1f' % s for s in STRENGTHS], fontsize=fs['tick'])
    ax.tick_params(axis='y', labelsize=fs['tick'])
    ax.set_ylim(*M['ylim'])
    ax.set_yticks(M['yticks'])
    ax.grid(axis='y', ls='--', lw=grid_lw, color='0.85', zorder=0)
    ax.set_axisbelow(True)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    ax.legend(frameon=False, fontsize=fs['legend'], loc=M['legend_loc'], ncol=1,
              handletextpad=0.5, labelspacing=0.3)
    if letter is not None:
        ax.text(0.025, 0.965, '(%s)' % letter, transform=ax.transAxes, ha='left',
                va='top', fontsize=fs['letter'], fontweight='bold', zorder=8)


def main():
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'pdf.fonttype': 42, 'ps.fonttype': 42,
                         'axes.unicode_minus': False, 'savefig.facecolor': 'white'})
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    draw(ax)
    fig.tight_layout()
    for ext in ('.pdf', '.png'):
        fig.savefig(os.path.join(C.ASSETS, 'fig2_ladder' + ext), dpi=300)
    plt.close(fig)
    print('saved paper_assets/fig2_ladder.pdf + .png')


if __name__ == '__main__':
    main()