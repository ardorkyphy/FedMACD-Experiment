# -*- coding: utf-8 -*-
"""Fig. 4 -- the additional analyses on one canvas, 2 x 2.

Rows are the two analyses, columns the two metrics:

    (a) heterogeneity strength, directed F1      (b) heterogeneity strength, SHD
    (c) scalability vs. clients, directed F1     (d) scalability vs. clients, SHD

Two panels were removed on 2026-09-26 (user instruction):

* the debate-budget panel is not drawn at all -- the same four dataset--scenario
  cells already appear as Table 2, so a figure would only re-draw numbers the
  table carries;
* the accuracy--cost scatter is gone as well, because cost moved to its own
  figure, Fig. 5 (grouped bars over the ten datasets).

Panels (a)(b) are the heterogeneous-split strength sweep (DWD_climate, survey)
and (c)(d) the client sweep (alarm, child, homogeneous split); both carry three
seeds per point.  SHD is plotted with "lower is better", which the axis label
says.  Drawing code, data source and assertion anchors still live in the
single-panel modules (``fig2_heterogeneity_strength`` and ``fig3_scalability``);
this file only arranges them and never recomputes anything.  Typography, grid
and canvas width come from ``fig1_performance`` (the script behind Fig. 3), as in
Fig. 5, so the three figures keep one visual language; on 2026-10-02 the panel
legend was raised to the panel-title size, the strokes thickened and the canvas
grown to 5.20 in so that no label sets below 7.8 pt at \textwidth.
"""
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import _common as C
import fig1_performance as F3          # Fig. 3 / Fig. 5: typography + canvas width
import fig2_heterogeneity_strength as A
import fig3_scalability as B

FIG_W, FIG_H = F3.CANVAS_W_IN, 5.20
FS = dict(label=F3.FS_AX, tick=F3.FS_TICK, legend=F3.FS_TITLE)
LW, MS, ELW, CAP = 1.3, 4.2, 0.8, 2.2
GRID_LW = F3.GRID_LW
TITLES = ('(a) Heterogeneity strength \u00b7 F1', '(b) Heterogeneity strength \u00b7 SHD',
          '(c) Scalability \u00b7 F1', '(d) Scalability \u00b7 SHD')
NAME = 'fig3_additional_analyses'


def main():
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'pdf.fonttype': 42,
                         'ps.fonttype': 42, 'axes.unicode_minus': False,
                         'figure.facecolor': 'white', 'savefig.facecolor': 'white'})
    fig, axes = plt.subplots(2, 2, figsize=(FIG_W, FIG_H), layout='constrained')
    A.draw(axes[0][0], fs=FS, metric='f1', lw=LW, ms=MS, grid_lw=GRID_LW)
    A.draw(axes[0][1], fs=FS, metric='shd', lw=LW, ms=MS, grid_lw=GRID_LW)
    B.draw_metric(axes[1][0], 'f1', fs=FS, lw=LW, ms=MS, elw=ELW, capsize=CAP,
                  grid_lw=GRID_LW, legend=True, legend_loc='lower left')
    B.draw_metric(axes[1][1], 'shd', fs=FS, lw=LW, ms=MS, elw=ELW, capsize=CAP,
                  grid_lw=GRID_LW, legend=True, legend_loc='upper left')
    for ax, title in zip(axes.ravel(), TITLES):
        ax.set_title(title, fontsize=F3.FS_TITLE, fontweight='bold', loc='left',
                     pad=3.0)
    for ext in ('.pdf', '.png'):
        fig.savefig(os.path.join(C.ASSETS, NAME + ext), dpi=300)
    plt.close(fig)
    print('saved paper_assets/%s.pdf + .png (%.4f x %.2f in, 2 x 2: two analyses '
          'x {F1, SHD})' % (NAME, FIG_W, FIG_H))


if __name__ == '__main__':
    main()
