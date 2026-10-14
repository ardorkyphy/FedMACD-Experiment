# -*- coding: utf-8 -*-
"""Fig. 1：主结果图。一张画布，2 行 x 4 列共 8 个横条形面板。

行 = 场景（上行 homogeneous、下行 heterogeneous），列 = 域 x 指标（离散 F1 / 连续 F1 /
离散 SHD / 连续 SHD），字母 (a)-(h) 按行优先。每个面板的标题自带场景前缀，如
`(a) Homogeneous · Discrete`；指标名写在各自的 x 轴标题上（Directed F1 / SHD）。
底部一条共享图例，8 项。

画布 = 正文栏宽 1:1 的 6.7717 in x 9.53 in（A4 版面上限，扣掉图注），一列 = 6.7717/4 =
1.6929 in。版面数字全部取自 fig1_performance（同一本账，不在这里另写一套）。同一列的
上下两个面板共用 x 轴范围（scan_limits 在两个场景上取同一上限），故两行可以逐列直接对读。

自检（图形几何，不靠肉眼、也不靠代理读图）：x 刻度互不压；数值全在轴内且互不重叠；
数据集名全在轴内且不压任何一根条；面板标题在画布内且不压任何别的坐标轴；图例不超
画布、不超图例条高；同一列上下两个面板的刻度完全一致。
"""
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fig1_performance as F
from _common import ASSETS

# One canvas per splitting scenario (2026-10-02).  The former single canvas held
# both scenarios in two rows, which cannot be read at \textwidth without going
# under 7 pt; see the note in fig1_performance.py.
STEM = {'homogeneous': 'fig_perf_homogeneous',
        'heterogeneous': 'fig_perf_heterogeneous'}


def tile_rects():
    """8 个面板的 axes 矩形（画布分数坐标），行优先：行 0 = homogeneous。"""
    out = []
    for r in range(F.NROW):
        for c in range(F.NCOL):
            x0 = c * F.TILE_W_IN + F.PAD_L_IN
            y0 = F.LEG_H_IN + (F.NROW - 1 - r) * (F.TILE_H_IN + F.ROW_GAP_IN) + F.PAD_B_IN
            out.append([x0 / F.CANVAS_W_IN, y0 / F.CANVAS_H_IN,
                        F.AX_W_IN / F.CANVAS_W_IN, F.AX_H_IN / F.CANVAS_H_IN])
    return out


def _inter(a, b):
    """两个包围盒的交面积（pt^2）。"""
    w = min(a.x1, b.x1) - max(a.x0, b.x0)
    h = min(a.y1, b.y1) - max(a.y0, b.y0)
    return max(w, 0.0) * max(h, 0.0)


def check_xticks(axes):
    """x 刻度文字两两不压。"""
    gaps = 0
    for ax in axes:
        fig = ax.figure
        fig.canvas.draw()
        ren = fig.canvas.get_renderer()
        boxes = [t.get_window_extent(renderer=ren) for t in ax.get_xticklabels()]
        for i in range(len(boxes) - 1):
            assert boxes[i].x1 <= boxes[i + 1].x0 + 0.01, 'x tick labels overlap'
            gaps += 1
    print('   x ticks: %d adjacent pairs, none overlapping' % gaps)


def check_marks(axes, info):
    """数值全在轴内、互不重叠；数据集名全在轴内、不压条、不互相压。"""
    nval = nname = 0
    for ax, it in zip(axes, info):
        fig = ax.figure
        fig.canvas.draw()
        ren = fig.canvas.get_renderer()
        ab = ax.get_window_extent()
        vb = [t.get_window_extent(renderer=ren) for t in it['values']]
        for t, b in zip(it['values'], vb):
            assert b.x0 >= ab.x0 - 0.01 and b.x1 <= ab.x1 + 0.01, \
                'value label runs out of the axes: %r' % t.get_text()
            assert b.y0 >= ab.y0 - 0.01 and b.y1 <= ab.y1 + 0.01, \
                'value label above/below the axes: %r' % t.get_text()
        for i in range(len(vb)):
            for j in range(i + 1, len(vb)):
                assert _inter(vb[i], vb[j]) <= 0.01, 'two value labels overlap'

        nb = [t.get_window_extent(renderer=ren) for t in it['names']]
        for t, b in zip(it['names'], nb):
            assert b.x0 >= ab.x0 - 0.01 and b.x1 <= ab.x1 + 0.01, \
                'dataset name runs out of the axes: %r' % t.get_text()
            assert b.y0 >= ab.y0 - 0.01 and b.y1 <= ab.y1 + 0.01, \
                'dataset name above/below the axes: %r' % t.get_text()
        for bar in it['bars']:
            bb = bar.get_window_extent(renderer=ren)
            for t, b in zip(it['names'], nb):
                assert _inter(b, bb) <= 0.01, 'dataset name %r sits on a bar' % t.get_text()
        for i in range(len(nb)):
            for j in range(i + 1, len(nb)):
                assert _inter(nb[i], nb[j]) <= 0.01, 'two dataset names overlap'
        nval += len(vb)
        nname += len(nb)
    print('   %d value labels inside their axes, none overlapping; %d dataset names '
          'inside their axes, none touching a bar' % (nval, nname))


def check_titles(axes):
    """面板标题在画布内，且不压任何别的坐标轴。"""
    fig = axes[0].figure
    fig.canvas.draw()
    ren = fig.canvas.get_renderer()
    W, H = fig.get_figwidth() * fig.dpi, fig.get_figheight() * fig.dpi
    boxes = [ax.get_window_extent() for ax in axes]
    widest = 0.0
    for ax, own in zip(axes, boxes):
        art = ax._left_title
        assert art.get_text(), 'panel has no left-aligned title'
        bb = art.get_window_extent(renderer=ren)
        assert bb.x0 >= 1.0 and bb.x1 <= W - 1.0, 'panel title outside the canvas (x)'
        assert bb.y1 <= H - 1.0, 'panel title above the canvas'
        for other, ob in zip(axes, boxes):
            if other is ax:
                continue
            assert _inter(bb, ob) <= 0.01, \
                'panel title overlaps a neighbouring axes: %r' % art.get_text()
        widest = max(widest, bb.width / fig.dpi * 72.0)
    print('   panel titles: %d written, widest %.1f pt (tile %.1f pt, axes %.1f pt)'
          % (len(axes), widest, F.TILE_W_IN * 72.0, F.AX_W_IN * 72.0))
    return widest


def check_rows(info):
    """同一列上下两个面板的 x 范围与刻度必须逐点相同，否则不能对读。"""
    if F.NROW < 2:
        print('   columns: single-row canvas, no cross-row comparison to check')
        return
    for c in range(F.NCOL):
        top, bot = info[c], info[F.NCOL + c]
        assert abs(top['xmax'] - bot['xmax']) < 1e-9, 'column %d: x limits differ' % c
        assert abs(top['tick_top'] - bot['tick_top']) < 1e-9, 'column %d: tick tops differ' % c
    print('   columns: both rows share one x range in all %d columns' % F.NCOL)


def render_scenario(data, scenario):
    W, H = F.CANVAS_W_IN, F.CANVAS_H_IN
    fig = plt.figure(figsize=(W, H))
    axes, info = [], []
    for c, (dom, metric) in enumerate(F.PANELS):
        ax = fig.add_axes(tile_rects()[c])
        info.append(F.panel(ax, data, scenario, dom, metric, F.PANEL_LETTERS[c]))
        axes.append(ax)
    pitch = F.row_pitch_pt()
    print('composite %s: %d rows x %d cols, canvas %.4f x %.2f in, tile %.4f x %.4f in, '
          'axes %.4f x %.4f in' % (scenario, F.NROW, F.NCOL, W, H, F.TILE_W_IN, F.TILE_H_IN,
                                   F.AX_W_IN, F.AX_H_IN))
    print('   row pitch %.2f pt, bar %.2f pt, bar gap %.2f pt (bar/pitch %.2f), '
          'font: value %.1f pt, tick %.1f pt, axis %.1f pt, title %.1f pt, legend %.1f pt'
          % (pitch, F.BARR * pitch, (1.0 - F.BARR) * pitch, F.BARR,
             F.FS_VALUE, F.FS_TICK, F.FS_AX, F.FS_TITLE, F.FS_LEGEND))

    fig.canvas.draw()
    check_xticks(axes)
    check_marks(axes, info)
    check_titles(axes)
    check_rows(info)

    leg = fig.legend(handles=F.legend_handles(), loc='lower center', ncol=4, frameon=False,
                     fontsize=F.FS_LEGEND, columnspacing=1.6, handlelength=1.6,
                     handleheight=1.3, labelspacing=0.7, bbox_to_anchor=(0.5, 0.008))
    fig.canvas.draw()
    bb = leg.get_window_extent()
    lw_in, lh_in = bb.width / fig.dpi, bb.height / fig.dpi
    print('   legend %.2f x %.2f in (canvas %.2f x %.2f in, legend strip %.2f in)'
          % (lw_in, lh_in, W, H, F.LEG_H_IN))
    assert lw_in <= W - 0.10, 'legend wider than the canvas'
    assert lh_in <= F.LEG_H_IN, 'legend taller than its strip'

    for ext in ('.pdf', '.png'):
        fig.savefig(os.path.join(ASSETS, STEM[scenario] + ext), dpi=300)
    plt.close(fig)
    print('saved paper_assets/%s.pdf + .png' % STEM[scenario])


def main():
    data, q = F.build_cells()
    F.sanity(data, q)
    F.scan_limits(data)
    for scenario in F.SCENARIOS:
        render_scenario(data, scenario)


if __name__ == '__main__':
    main()
