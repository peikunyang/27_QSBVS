#!/usr/bin/env python3
"""Publication-sized, independent SVG panels for the original QAE figure pipeline.

All statistics and FLAG-only resource data are taken from the existing computational
helpers. No cached previous figure TSVs are used as input. Each run uses current raw
analysis files below --data-root and refreshes source/summary TSVs.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from figure_utils import RECEPTORS, K_CHOICES, OCC_COUNTS, M0_PANELS
from preview_common import (OCC_PANELS, RESOURCE_PANELS, K_VALUES,
                            summarize_eight, load_flag_for_panels)
from supplement_common import (SUPP_OCC_PANELS, SUPP_M0_SHOT_COST,
                               _common_r_source, _optimized_flag_cost_rows)

FONT_TICK = 15
FIGSIZE = (5.7, 4.45)
ALPHA = 0.11  # Original main and supplementary median/IQR plotting opacity
PANEL_LETTERS = 'abcd'


def style():
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Tinos', 'Liberation Serif', 'DejaVu Serif'],
        'font.size': 14,
        'mathtext.fontset': 'stix',
        'axes.spines.top': True,
        'axes.spines.right': True,
        'svg.fonttype': 'none',
        'savefig.dpi': 240,
    })


def axes_style(ax, *, xscale='linear', yscale='linear', ylim=None):
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(True, which='major', linestyle='--', linewidth=.55, alpha=.33)
    ax.tick_params(axis='both', which='major', labelsize=FONT_TICK,
                   width=1.0, length=5.5)
    ax.tick_params(axis='both', which='minor', width=.8, length=3.0)
    for name in ('left', 'right', 'top', 'bottom'):
        ax.spines[name].set_visible(True)
        ax.spines[name].set_linewidth(1.1)
    ax.set_xlabel('')
    ax.set_ylabel('')


def plot_curves(ax, summary, xcol, *, draw_k=True, markers=False, alpha=ALPHA):
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for index, k in enumerate(K_VALUES if draw_k else (1,)):
        d = summary[summary['K_PLOT'] == k].sort_values(xcol)
        if d.empty:
            raise ValueError(f'No points for k={k}')
        if not np.all(np.isfinite(d[['MEDIAN', 'Q1', 'Q3']].to_numpy(float))):
            raise ValueError('Non-finite median/IQR values')
        x = d[xcol].to_numpy(float)
        med, lo, hi = [d[c].to_numpy(float) for c in ('MEDIAN', 'Q1', 'Q3')]
        if np.any(lo > med) or np.any(med > hi):
            raise ValueError('Invalid quartile ordering')
        kw = {'marker': 'o', 'ms': 3.2} if markers else {}
        ax.plot(x, med, lw=1.8, color=colors[index], **kw)
        ax.fill_between(x, lo, hi, color=colors[index], alpha=alpha)


def save(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format='svg', bbox_inches='tight', pad_inches=.10)
    plt.close(fig)
    print(f'SVG: {path}')


def remove_old_legend(output, figure_name):
    """Remove a legacy legend SVG left by an earlier version, if any."""
    legend = Path(output).parent / f'{figure_name}_legend.svg'
    if legend.is_file():
        legend.unlink()


def write_tables(output, source, summary):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    source.to_csv(output.with_name(output.stem + '_source.tsv'), sep='\t', index=False)
    summary.to_csv(output.with_name(output.stem + '_summary.tsv'), sep='\t', index=False)


def draw_main_batch(data_root, output, metric, is_log_y):
    style()
    df = pd.read_csv(Path(data_root) / 'batch_summary_pose.tsv', sep='\t')
    required = {'PDBID', 'OCC_COUNT', 'RESOLUTION_MULTIPLE_TARGET',
                'AMPLITUDE_THRESHOLD', metric}
    if required - set(df.columns):
        raise ValueError(f'Batch is missing {sorted(required - set(df.columns))}')
    source = df[df.OCC_COUNT.isin(OCC_PANELS) &
                df.RESOLUTION_MULTIPLE_TARGET.isin(K_VALUES)].copy()
    if not source[metric].between(0, 1).all():
        raise ValueError(f'Invalid probability in {metric}')
    summary = summarize_eight(source,
        ['OCC_COUNT', 'RESOLUTION_MULTIPLE_TARGET', 'AMPLITUDE_THRESHOLD'], metric)
    fig_name = 'Fig2' if metric == 'QAE_FLAG_PROB' else 'Fig3'
    out = Path(output)
    remove_old_legend(out, fig_name)
    for letter, occ in zip(PANEL_LETTERS, OCC_PANELS):
        fig, ax = plt.subplots(figsize=FIGSIZE)
        d = summary[summary.OCC_COUNT == occ].copy()
        d['K_PLOT'] = d.RESOLUTION_MULTIPLE_TARGET
        for k in K_VALUES:
            if len(d[d.K_PLOT == k]) != 13:
                raise ValueError(f'{fig_name}: expected 13 thresholds for N_OCC={occ}, k={k}')
        plot_curves(ax, d, 'AMPLITUDE_THRESHOLD', markers=True)
        axes_style(ax, xscale='log', yscale='log' if is_log_y else 'linear',
                   ylim=(1e-7, 1.35) if is_log_y else (0, 1.02))
        save(fig, out.with_name(f'{fig_name}_{letter}.svg'))
    write_tables(out, source, summary)


def _resource_source(data_root):
    """Identical 8-receptor common scan bound handling as original preview_common."""
    full = load_flag_for_panels(data_root)
    bounds = {}
    for occ, m0 in RESOURCE_PANELS:
        for k in K_VALUES:
            d = full[(full.OCC_COUNT == occ) & (full.BASE_M == m0) & (full.K == k)]
            if d.empty:
                raise ValueError(f'Missing N_OCC={occ}, m0={m0}, k={k}')
            ends = d.groupby('PDBID').GROVER_ITERATIONS.max()
            if len(ends) != 8 or set(ends.index) != set(RECEPTORS):
                raise ValueError(f'Incomplete 8 receptors at {occ}/{m0}/{k}')
            bound = int(ends.min())
            for _, receptor in d.groupby('PDBID'):
                if not set(range(bound + 1)).issubset(set(receptor.GROVER_ITERATIONS.astype(int))):
                    raise ValueError(f'Non-contiguous r at {occ}/{m0}/{k}')
            bounds[(occ, m0, k)] = bound
    source = pd.concat([
        full[(full.OCC_COUNT == occ) & (full.BASE_M == m0) &
             (full.K == k) & (full.GROVER_ITERATIONS <= bounds[(occ, m0, k)])]
        for occ, m0 in RESOURCE_PANELS for k in K_VALUES
    ], ignore_index=True)
    return source, bounds


def draw_main_resource(data_root, output, metric):
    style()
    out = Path(output)
    fig_name = 'Fig4' if metric == 'SHOTS95_TARGET' else 'Fig5'
    remove_old_legend(out, fig_name)
    source, bounds = _resource_source(data_root)
    summary = summarize_eight(source, ['OCC_COUNT', 'BASE_M', 'K', 'GROVER_ITERATIONS'], metric)
    for letter, (occ, m0) in zip(PANEL_LETTERS, RESOURCE_PANELS):
        fig, ax = plt.subplots(figsize=FIGSIZE)
        d = summary[(summary.OCC_COUNT == occ) & (summary.BASE_M == m0)].copy()
        d['K_PLOT'] = d.K
        for k in K_VALUES:
            if len(d[d.K == k]) != bounds[(occ, m0, k)] + 1:
                raise ValueError(f'{fig_name}: incomplete median curve at {occ}/{m0}/{k}')
        plot_curves(ax, d, 'GROVER_ITERATIONS')
        axes_style(ax, xscale='linear', yscale='log')
        rmax = max(bounds[(occ, m0, k)] for k in K_VALUES)
        ax.set_xlim(0, rmax)
        if rmax <= 8:
            ax.set_xticks(list(range(rmax + 1)))
        save(fig, out.with_name(f'{fig_name}_{letter}.svg'))
    write_tables(out, source, summary)


def draw_s1(data_root, output):
    style()
    file = Path(data_root) / 'batch_summary_pose.tsv'
    frame = pd.read_csv(file, sep='\t')
    required = {'PDBID', 'OCC_COUNT', 'RESOLUTION_MULTIPLE_TARGET',
                'AMPLITUDE_THRESHOLD', 'EXACT_GOOD_FRACTION'}
    if required - set(frame.columns):
        raise ValueError(f'Batch missing {sorted(required - set(frame.columns))}')
    source = frame[frame.OCC_COUNT.isin(SUPP_OCC_PANELS) &
                   (frame.RESOLUTION_MULTIPLE_TARGET == 1)].copy()
    if not source.EXACT_GOOD_FRACTION.between(0, 1).all():
        raise ValueError('Invalid q')
    summary = summarize_eight(source, ['OCC_COUNT', 'AMPLITUDE_THRESHOLD'], 'EXACT_GOOD_FRACTION')
    out = Path(output)
    remove_old_legend(out, 'FigS1')
    for letter, occ in zip(PANEL_LETTERS, SUPP_OCC_PANELS):
        fig, ax = plt.subplots(figsize=FIGSIZE)
        d = summary[summary.OCC_COUNT == occ].copy()
        if len(d) != 13:
            raise ValueError('S1 expected 13 thresholds')
        d['K_PLOT'] = 1
        plot_curves(ax, d, 'AMPLITUDE_THRESHOLD', draw_k=False, markers=True, alpha=.14)
        axes_style(ax, xscale='log', yscale='log', ylim=(1e-7, 1.35))
        save(fig, out.with_name(f'FigS1_{letter}.svg'))
    write_tables(out, source, summary)


def draw_s2_s3(data_root, output, kind):
    style()
    if kind not in ('shots', 'cost'):
        raise ValueError(kind)
    fig_name = 'FigS2' if kind == 'shots' else 'FigS3'
    out = Path(output)
    remove_old_legend(out, fig_name)
    source, bounds = _common_r_source(data_root, SUPP_OCC_PANELS, SUPP_M0_SHOT_COST)
    metric = 'SHOTS95_TARGET' if kind == 'shots' else 'TOTAL_Q_CALLS95_TARGET'
    summary = summarize_eight(source, ['OCC_COUNT', 'BASE_M', 'K', 'GROVER_ITERATIONS'], metric)
    for letter, occ in zip(PANEL_LETTERS, SUPP_OCC_PANELS):
        fig, ax = plt.subplots(figsize=FIGSIZE)
        d = summary[summary.OCC_COUNT == occ].copy()
        d['K_PLOT'] = d.K
        for k in K_VALUES:
            if len(d[d.K == k]) != bounds[(occ, k)] + 1:
                raise ValueError(f'{fig_name} incomplete r at {occ}/{k}')
        plot_curves(ax, d, 'GROVER_ITERATIONS')
        axes_style(ax, xscale='linear', yscale='log')
        rmax = max(bounds[(occ, k)] for k in K_VALUES)
        ax.set_xlim(0, rmax)
        if rmax <= 8:
            ax.set_xticks(list(range(rmax + 1)))
        save(fig, out.with_name(f'{fig_name}_{letter}.svg'))
    write_tables(out, source, summary)


def draw_s4(data_root, output):
    style()
    out = Path(output)
    remove_old_legend(out, 'FigS4')
    source = _optimized_flag_cost_rows(data_root)
    summary = summarize_eight(source, ['OCC_COUNT', 'BASE_M', 'K'], 'BEST_TOTAL_Q_CALLS95')
    for letter, m0 in zip(PANEL_LETTERS, M0_PANELS):
        fig, ax = plt.subplots(figsize=FIGSIZE)
        d = summary[summary.BASE_M == m0].copy()
        d['K_PLOT'] = d.K
        for k in K_VALUES:
            if len(d[d.K == k]) != len(OCC_COUNTS):
                raise ValueError(f'FigS4 missing library size m0={m0},k={k}')
        plot_curves(ax, d, 'OCC_COUNT', markers=True)
        axes_style(ax, xscale='log', yscale='log')
        ax.set_xscale('log', base=2)
        ax.set_xticks(OCC_COUNTS)
        ax.set_xticklabels([str(n) for n in OCC_COUNTS], rotation=45, ha='right', fontsize=13)
        save(fig, out.with_name(f'FigS4_{letter}.svg'))
    write_tables(out, source, summary)
