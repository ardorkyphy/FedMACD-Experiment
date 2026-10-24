# -*- coding: utf-8 -*-
"""Fig. 5 -- per-dataset cost and accuracy of the three reasoning backbones (bars).

Three stacked panels sharing one categorical x axis (the ten datasets, ordered
by glm's token cost, cheapest first); inside a dataset the three bars are the
three backbones:

    (a) average LLM tokens per fold [k tokens]   (linear y)
    (b) average wall-clock time per fold [s]     (linear y)
    (c) mean directed F1                         (linear y)

The figure shares typography and layout settings with the other main-results figures. Bar colors distinguish the three backbones. Values are computed from the archived fold records and checked against the manuscript summary. Cost measurements were collected in separate runs; token counts are less sensitive to machine load than elapsed time."""
from __future__ import annotations

import os
import statistics as st
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as C
import fig1_performance as F3          # Fig. 3: palette, typography, canvas width

DATASET_ORDER = C.DISCRETE + C.CONTINUOUS

# The red/teal/sand palette distinguishes the three backbones. Each fill gets a
# darker edge of the same hue for contrast.
# 'triad' is the default palette; 'triad_base' keeps its original RGB values.
# 'pastel' is an earlier light blue / light green /
# light purple triad; 'okabe' is the Okabe-Ito set listed as
# .palette_colors_hex[["Okabe-Ito"]] in R grDevices (R-4.5.1,
# src/library/grDevices/R/colorstuff.R).
EDGE_FACTOR = 0.62
PALETTES = {
    # Okabe-Ito / Color Universal Design set (Masataka Okabe & Kei Ito,
    # http://jfly.iam.u-tokyo.ac.jp/color/) -- the colour-blind-safe qualitative
    # palette recommended for scientific figures (B. Wong, "Points of view: Color
    # blindness", Nature Methods 8:441, 2011) and shipped as the reference palette
    # of R >= 4.0 (grDevices::palette.colors("Okabe-Ito")).
    'okabe': {'qwen3.8-flash': '#0072B2', 'deepseek-v4-flash': '#009E73',
              'glm-5.3-flash': '#D55E00'},
    # Red/teal/sand palette, with slightly increased lightness.
    'triad': {'qwen3.8-flash': '#DC7B7C', 'deepseek-v4-flash': '#59BDC5',
              'glm-5.3-flash': '#EFCD98'},
    'triad_base': {'qwen3.8-flash': '#D15354', 'deepseek-v4-flash': '#3DA6AE',
                   'glm-5.3-flash': '#E8B86C'},
    # Alternative pastel variant: light blue
    # rgb(148,179,223), light green rgb(179,212,155), light purple rgb(205,163,203).
    'pastel': {'qwen3.8-flash': '#94B3DF', 'deepseek-v4-flash': '#B3D49B',
               'glm-5.3-flash': '#CDA3CB'},
    'npg': {'qwen3.8-flash': '#E64B35', 'deepseek-v4-flash': '#00A087',
            'glm-5.3-flash': '#3C5488'},
    'vega': {'qwen3.8-flash': '#F58518', 'deepseek-v4-flash': '#54A24B',
             'glm-5.3-flash': '#4C78A8'},
    'vivid': {'qwen3.8-flash': '#E7298A', 'deepseek-v4-flash': '#1B9E77',
              'glm-5.3-flash': '#7570B3'},
}


def _darken(hex_color, factor=EDGE_FACTOR):
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
    return '#%02x%02x%02x' % (int(round(r * factor)), int(round(g * factor)),
                              int(round(b * factor)))


def _deeper(hex_color, lightness=0.50, saturation=0.40):
    """Same hue at a mid tone -- the outline that keeps a pastel fill crisp on
    white paper without shouting (lightness 0.50, saturation 0.40)."""
    import colorsys
    r, g, b = (int(hex_color[i:i + 2], 16) / 255.0 for i in (1, 3, 5))
    h, _l, _s = colorsys.rgb_to_hls(r, g, b)
    r2, g2, b2 = colorsys.hls_to_rgb(h, lightness, saturation)
    return '#%02x%02x%02x' % tuple(int(round(v * 255)) for v in (r2, g2, b2))


def _edges(palette, colors):
    # Mid-tone fills (okabe/npg/vega/vivid/triad) take a darkened edge of the same
    # hue; the very light 'pastel' fills need a mid tone instead of a dark one,
    # otherwise the outline goes grey.
    if palette == 'pastel':
        return {bb: _deeper(c) for bb, c in colors.items()}
    return {bb: _darken(c) for bb, c in colors.items()}


PALETTE = 'triad'
COLOR_OF = dict(PALETTES[PALETTE])
EDGE_OF = _edges(PALETTE, COLOR_OF)
LABEL_OF = {'qwen3.8-flash': 'Ours (qwen3.8-flash)',
            'deepseek-v4-flash': 'Ours (deepseek-v4-flash)',
            'glm-5.3-flash': 'Ours (glm-5.3-flash)'}
FS = dict(label=F3.FS_AX, tick=F3.FS_TICK, legend=F3.FS_LEGEND,
          title=F3.FS_TITLE, value=F3.FS_VALUE)
EDGE_LW = F3.EDGE_LW
GRID_LW = F3.GRID_LW
SPINE_LW = F3.SPINE_LW
TICK_LW = F3.TICK_LW
CANVAS_W = F3.CANVAS_W_IN
CANVAS_H = 6.30
BAR_W = 0.32
# the value labels are set upright (rotation=90) above the bars: at 7.5 pt the
# widest one ("11204", five digits) is about 23.5 pt long, while the horizontal
# pitch between two bars of a group is only about 13 pt, so upright labels would
# run into each other.  Rotated, they use the headroom each ylim now keeps.
FS_LABEL = 7.5

# Reference values from the manuscript: per-fold mean over the 60 main folds
ANCHOR_MEAN = {
    'glm-5.3-flash': dict(tokens_k=288, seconds=1328, f1=0.766),
    'qwen3.8-flash': dict(tokens_k=478, seconds=3705, f1=None),
    'deepseek-v4-flash': dict(tokens_k=581, seconds=2548, f1=None),
}
ANCHOR_CELL = {
    ('glm-5.3-flash', 'earthquake'): dict(tokens_k=87, seconds=247),
    ('glm-5.3-flash', 'alarm'): dict(tokens_k=603, seconds=2047),
    ('glm-5.3-flash', 'Sachs'): dict(tokens_k=566, seconds=3404),
}


def cell(backbone, dataset):
    """(mean tokens in thousands, mean wall-clock seconds, mean directed F1)."""
    rows = C.pick(C.main_rows(backbone), dataset=dataset)
    assert len(rows) == 6, (backbone, dataset, len(rows))
    tokens = st.fmean([C.fnum(r['llm_total_tokens']) for r in rows]) / 1000.0
    seconds = st.fmean([C.fnum(r['seconds']) for r in rows])
    f1 = st.fmean([C.fnum(r['f1']) for r in rows])
    return tokens, seconds, f1


def cell_sd(backbone, dataset):
    """Same three metrics, but the sd over the three seeds.

    Each seed first pools its two scenarios (discrete / continuous); the sd is then
    taken over the three seed-level values -- the pooled-value +- seed-sd convention
    of Table 1, not the sd over the six individual folds.
    """
    rows = C.pick(C.main_rows(backbone), dataset=dataset)
    seeds = C.seeds_of(rows)
    assert len(seeds) == 3, (backbone, dataset, seeds)
    out = []
    for key, scale in (('llm_total_tokens', 1000.0), ('seconds', 1.0), ('f1', 1.0)):
        per_seed = [st.fmean([C.fnum(r[key]) for r in rows if int(r['seed']) == sd])
                    / scale for sd in seeds]
        out.append(st.stdev(per_seed))
    return tuple(out)


def check_anchors(points):
    for bb in C.BACKBONE_ORDER:
        rows = C.main_rows(bb)
        assert len(rows) == 60, (bb, len(rows))
        tokens = st.fmean([C.fnum(r['llm_total_tokens']) for r in rows]) / 1000.0
        seconds = st.fmean([C.fnum(r['seconds']) for r in rows])
        f1 = st.fmean([C.fnum(r['f1']) for r in rows])
        a = ANCHOR_MEAN[bb]
        assert round(tokens) == a['tokens_k'], (bb, tokens, a['tokens_k'])
        assert round(seconds) == a['seconds'], (bb, seconds, a['seconds'])
        if a['f1'] is not None:
            assert abs(f1 - a['f1']) < 5e-4, (bb, f1, a['f1'])
        print('anchor  %-18s tokens %7.1fk (plan %dk)  wall %7.1fs (plan %ds)  F1 %.4f'
              % (bb, tokens, a['tokens_k'], seconds, a['seconds'], f1))
    for (bb, ds), a in ANCHOR_CELL.items():
        tokens, seconds, _ = points[(bb, ds)]
        assert round(tokens) == a['tokens_k'], (bb, ds, tokens, a['tokens_k'])
        assert round(seconds) == a['seconds'], (bb, ds, seconds, a['seconds'])
        print('anchor  %-18s %-12s tokens %7.1fk (plan %dk)  wall %7.1fs (plan %ds)'
              % (bb, ds, tokens, a['tokens_k'], seconds, a['seconds']))
    for ds in DATASET_ORDER:
        pt = points[('glm-5.3-flash', ds)]
        for bb in ('qwen3.8-flash', 'deepseek-v4-flash'):
            de = points[(bb, ds)]
            assert pt[0] <= de[0] and pt[1] <= de[1], (ds, bb, pt, de)
    print('anchor  glm is token- and time-cheaper than both other backbones on 10/10 datasets')


def draw(fig, axes, fs=None, value_labels=False, error_bars=False):
    fs = dict(FS, **(fs or {}))
    points = {(bb, ds): cell(bb, ds) for bb in C.BACKBONE_ORDER for ds in DATASET_ORDER}
    sds = {(bb, ds): cell_sd(bb, ds) for bb in C.BACKBONE_ORDER for ds in DATASET_ORDER}

    order = sorted(DATASET_ORDER, key=lambda d: points[('glm-5.3-flash', d)][0])
    print('order   datasets by glm token cost: %s' % ' < '.join(
        '%s(%.0fk)' % (d, points[('glm-5.3-flash', d)][0]) for d in order))

    # ylim leaves headroom above the tallest bar + its sd for the rotated value
    # label; the tick format is used on the axis, the label format on the bars.
    # Bar labels round (%.0f) rather than truncate (%d): a mean of 171.99 k reads as
    # 172, matching the rounded manuscript summary (288/478/581 k).
    # Draw bars without whiskers. The sd-over-
    # seeds numbers stay available through ``error_bars=True`` and are printed by
    # this module, and Table 1 carries them for the ablation grid.  Each ylim has
    # to leave one rotated-label length (about 24 pt, i.e. ~20 per cent of a panel)
    # above the tallest bar; the values below are the measured maxima (1242 k,
    # 11204 s, 1.00) plus that headroom, rounded up to the next tick.
    panels = [(axes[0], 0, 'Tokens per fold [k tokens]', '(a) Token cost',
               [0, 200, 400, 600, 800, 1000, 1200, 1400], '%dk', '%.0f', 1600, False),
              (axes[1], 1, 'Wall-clock time per fold [s]', '(b) Time cost',
               [0, 3000, 6000, 9000, 12000, 15000], '%ds', '%.0f', 15000, False),
              (axes[2], 2, 'Directed F1', '(c) Accuracy',
               [0.0, 0.2, 0.4, 0.6, 0.8, 1.0], '%.1f', '%.2f', 1.25, False)]

    xs = list(range(len(order)))
    for ax, idx, ylab, title, ticks, fmt, lfmt, ymax, want_err in panels:
        for k, bb in enumerate(C.BACKBONE_ORDER):
            off = (k - 1.0) * BAR_W
            vals = [points[(bb, ds)][idx] for ds in order]
            show_err = bool(error_bars and want_err)
            errs = [sds[(bb, ds)][idx] for ds in order] if show_err \
                else [0.0] * len(order)
            bars = ax.bar([x + off for x in xs], vals, width=BAR_W * 0.92,
                          yerr=errs if show_err else None, capsize=1.1,
                          error_kw=dict(ecolor=EDGE_OF[bb], elinewidth=0.45,
                                        capthick=0.45, zorder=4),
                          color=COLOR_OF[bb], edgecolor=EDGE_OF[bb],
                          linewidth=EDGE_LW, zorder=3)
            if value_labels:
                for b, v, e in zip(bars, vals, errs):
                    ax.text(b.get_x() + b.get_width() / 2.0, v + e,
                            lfmt % v, ha='center', va='bottom', rotation=90,
                            fontsize=FS_LABEL, color='0.15', zorder=5)
            print('plot    %-14s %-18s %s' % (title, bb,
                                              ' '.join('%.1f' % v for v in vals)))
            if show_err:
                print('sd      %-14s %-18s %s' % (title, bb,
                                                  ' '.join('%.1f' % v for v in errs)))
        ax.set_xticks(xs)
        ax.set_xticklabels(order, rotation=42, ha='right', fontsize=fs['tick'])
        ax.set_xlim(-0.65, len(order) - 0.35)
        ax.set_yticks(ticks)
        ax.set_yticklabels([fmt % t if t else '0' for t in ticks], fontsize=fs['tick'])
        ax.set_ylim(0, ymax)
        ax.grid(True, axis='y', ls='--', lw=GRID_LW, color='0.86', zorder=0)
        ax.set_axisbelow(True)
        ax.set_ylabel(ylab, fontsize=fs['label'])
        ax.set_title(title, fontsize=fs['title'], fontweight='bold', pad=3.0, loc='left')
        ax.tick_params(axis='x', length=1.6, width=TICK_LW, pad=1.5)
        ax.tick_params(axis='y', length=1.6, width=TICK_LW, pad=1.5)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
        for sp in ('left', 'bottom'):
            ax.spines[sp].set_linewidth(SPINE_LW)
        if ax is not axes[-1]:
            ax.tick_params(labelbottom=False)

    handles = [Patch(facecolor=COLOR_OF[bb], edgecolor=EDGE_OF[bb], linewidth=EDGE_LW,
                     label=LABEL_OF[bb]) for bb in C.BACKBONE_ORDER]
    fig.legend(handles=handles, frameon=False, fontsize=fs['legend'], ncol=3,
               loc='outside upper center', handletextpad=0.6, columnspacing=1.6,
               handleheight=1.2, handlelength=1.6)
    return points, order


def main():
    global PALETTE, COLOR_OF, EDGE_OF
    if len(sys.argv) > 1:
        if sys.argv[1] not in PALETTES:
            raise SystemExit('palette must be one of %s' % sorted(PALETTES))
        PALETTE = sys.argv[1]
        COLOR_OF = dict(PALETTES[PALETTE])
        EDGE_OF = _edges(PALETTE, COLOR_OF)
    out_dir = sys.argv[2] if len(sys.argv) > 2 else C.ASSETS
    name = 'fig5_cost_bars' if out_dir == C.ASSETS else 'fig5_cost_bars_%s' % PALETTE
    print('palette %s: %s' % (PALETTE, COLOR_OF))
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'pdf.fonttype': 42,
                         'ps.fonttype': 42, 'axes.unicode_minus': False,
                         'figure.facecolor': 'white', 'savefig.facecolor': 'white'})
    fig, axes = plt.subplots(3, 1, figsize=(CANVAS_W, CANVAS_H), sharex=True,
                             layout='constrained')
    points = {(bb, ds): cell(bb, ds) for bb in C.BACKBONE_ORDER for ds in DATASET_ORDER}
    check_anchors(points)
    draw(fig, axes, value_labels=True)
    for ext in ('.pdf', '.png'):
        fig.savefig(os.path.join(out_dir, name + ext), dpi=300)
    plt.close(fig)
    print('saved %s palette=%s -> %s' % (name, PALETTE, out_dir))

    print()
    print('%-12s %-18s %10s %10s %8s' % ('dataset', 'backbone', 'tokens_k', 'seconds', 'F1'))
    for ds in DATASET_ORDER:
        for bb in C.BACKBONE_ORDER:
            t, s, f = points[(bb, ds)]
            print('%-12s %-18s %10.1f %10.0f %8.4f' % (ds, bb, t, s, f))


if __name__ == '__main__':
    main()
