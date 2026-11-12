#!/usr/bin/env python3
"""Shared, provenance-conscious helpers for QAE pose-selection figure scripts.

Only files under a directory literally named 'pose' are considered.  Raw
per-iteration probabilities are mandatory for resource-cost reconstruction.
"""
from pathlib import Path
import math
import numpy as np
import pandas as pd

RECEPTORS = ('1ecq', '1h0r', '2aa9', '2q9y', '2ybk', '2ybs', '4gui', '6o4x')
OCC_COUNTS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)
M0_PANELS = (10, 14, 18, 22)
K_CHOICES = (1, 2, 4, 8)


def require_columns(frame, cols, location):
    missing = [c for c in cols if c not in frame.columns]
    if missing:
        raise ValueError(f'{location}: missing columns {missing}. Do not substitute FLAG probabilities for true-good data.')


def shots95(probabilities):
    """Integer independent shots for >=95% chance of >=1 success; p=0 -> inf."""
    p = np.asarray(probabilities, dtype=float)
    if np.any(~np.isfinite(p) | (p < 0) | (p > 1)):
        raise ValueError('A success probability is not finite or is outside [0, 1].')
    result = np.full(p.shape, np.inf, dtype=float)
    mid = (p > 0) & (p < 1)
    result[mid] = np.maximum(1., np.ceil(np.log(0.05) / np.log1p(-p[mid])))
    result[p == 1] = 1.
    return result


def discover_pose_runs(root, k=None, m0_values=None):
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f'Data root not found: {root}')
    files = sorted(root.rglob('grover_shots.tsv'))
    runs = []
    for path in files:
        if path.parent.name != 'pose':
            continue
        metadata_path = path.parent / 'scaling_summary.tsv'
        if not metadata_path.is_file():
            raise FileNotFoundError(f'Missing scaling_summary.tsv next to {path}')
        meta = pd.read_csv(metadata_path, sep='\t', nrows=1)
        require_columns(meta, ['RESOLUTION_MULTIPLE_TARGET', 'EVAL_QUBITS', 'AMPLITUDE_THRESHOLD'], metadata_path)
        if meta.empty:
            raise ValueError(f'Empty metadata: {metadata_path}')
        kk = int(meta['RESOLUTION_MULTIPLE_TARGET'].iloc[0])
        if kk not in K_CHOICES:
            continue
        mm = int(meta['EVAL_QUBITS'].iloc[0])
        m0 = mm - int(math.log2(kk))
        if k is not None and kk != k:
            continue
        if m0_values is not None and m0 not in m0_values:
            continue
        pdbid = path.parents[2].name.lower()
        if pdbid not in RECEPTORS:
            continue
        runs.append(dict(PDBID=pdbid, K=kk, BASE_M=m0, EVAL_QUBITS=mm,
                         AMPLITUDE_THRESHOLD=float(meta['AMPLITUDE_THRESHOLD'].iloc[0]),
                         GROVER_PATH=path))
    if not runs:
        raise FileNotFoundError(f'No matching pose/grover_shots.tsv files under {root}')
    keys = [(r['PDBID'], r['K'], r['BASE_M']) for r in runs]
    if len(keys) != len(set(keys)):
        raise ValueError('Duplicate pose runs for the same receptor, k and base m0. Remove duplicates or isolate your data root.')
    return runs


def load_curves(root, *, k=1, m0_values=None, occ_values=(1024,), target='true_good'):
    """Read original grover_shots.tsv, never inherited precomputed cost columns."""
    if target not in ('true_good', 'flag'):
        raise ValueError(f'Unknown target {target}')
    runs = discover_pose_runs(root, k=k, m0_values=m0_values)
    frames = []
    for run in runs:
        path = run['GROVER_PATH']
        raw = pd.read_csv(path, sep='\t')
        required = ['OCC_COUNT', 'GROVER_ITERATIONS', 'AMPLIFIED_FLAG_PROB',
                    'Q_CALLS_PER_SHOT']
        if target == 'true_good':
            required.append('TRUE_GOOD_PROB')
        require_columns(raw, required, path)
        raw = raw[raw['OCC_COUNT'].isin(occ_values)].copy()
        if raw.empty:
            continue
        for c in required:
            raw[c] = pd.to_numeric(raw[c], errors='coerce')
        if raw[required].isna().any().any():
            raise ValueError(f'Non-numeric/missing required values in {path}')
        if np.any((raw['GROVER_ITERATIONS'] < 0) | (raw['GROVER_ITERATIONS'] % 1 != 0)):
            raise ValueError(f'Invalid integer Grover iterations in {path}')
        if np.any(raw['Q_CALLS_PER_SHOT'] <= 0):
            raise ValueError(f'Invalid Q_CALLS_PER_SHOT in {path}')
        expected = (2 * raw['GROVER_ITERATIONS'].to_numpy() + 1) * (2**run['EVAL_QUBITS'] - 1)
        if not np.allclose(raw['Q_CALLS_PER_SHOT'], expected, rtol=0, atol=0.1):
            raise ValueError(f'Q_CALLS_PER_SHOT does not match (2r+1)(2^m-1): {path}')
        target_col = 'TRUE_GOOD_PROB' if target == 'true_good' else 'AMPLIFIED_FLAG_PROB'
        prob = raw[target_col].to_numpy(dtype=float)
        n = shots95(prob)
        raw['SUCCESS_PROB'] = prob
        raw['SHOTS95_TARGET'] = n
        raw['TOTAL_Q_CALLS95_TARGET'] = n * raw['Q_CALLS_PER_SHOT'].to_numpy(dtype=float)
        for name in ('PDBID','K','BASE_M','EVAL_QUBITS','AMPLITUDE_THRESHOLD'):
            raw[name] = run[name]
        raw['SUCCESS_EVENT'] = target
        for nocc, sub in raw.groupby('OCC_COUNT'):
            if sub['GROVER_ITERATIONS'].duplicated().any():
                raise ValueError(f'Duplicate iteration in {path}, N_OCC={nocc}')
            if 0 not in set(sub['GROVER_ITERATIONS']):
                raise ValueError(f'Missing r=0 baseline in {path}, N_OCC={nocc}')
        frames.append(raw)
    if not frames:
        raise ValueError(f'No pose curves at N_OCC={occ_values}, k={k}, m0={m0_values}')
    return pd.concat(frames, ignore_index=True)


def enforce_receptors(data, keys, expected=RECEPTORS):
    target = set(expected)
    for key, group in data.groupby(keys, dropna=False):
        names = list(group['PDBID'])
        if len(names) != len(target) or set(names) != target:
            raise ValueError(f'Incomplete receptor coverage for {key}: got {sorted(names)}, expected {sorted(target)}')


def median_iqr(data, keys, value, require_eight=True):
    if require_eight:
        enforce_receptors(data, keys)
    v = data.groupby(keys, as_index=False).agg(
        MEDIAN=(value,'median'), Q1=(value, lambda x: float(np.percentile(x,25))),
        Q3=(value, lambda x: float(np.percentile(x,75))),
        N_RECEPTORS=('PDBID','nunique'))
    return v


def optimize_cost(curves):
    """Best *scanned* r, separately for every receptor/base-m/Nocc/k."""
    keys = ['PDBID','K','BASE_M','EVAL_QUBITS','AMPLITUDE_THRESHOLD','OCC_COUNT']
    out = []
    for ident, sub in curves.groupby(keys):
        s0 = sub[sub['GROVER_ITERATIONS'] == 0]
        if len(s0) != 1:
            raise ValueError(f'Expected exactly one r=0 baseline: {ident}')
        finite = sub[np.isfinite(sub['TOTAL_Q_CALLS95_TARGET'])].copy()
        if finite.empty:
            raise ValueError(f'No finite true-good or FLAG success costs: {ident}')
        baseline = s0.iloc[0]
        if not np.isfinite(baseline['TOTAL_Q_CALLS95_TARGET']):
            raise ValueError(f'Undefined r=0 retrieval cost, cannot form baseline: {ident}')
        best = finite.sort_values(['TOTAL_Q_CALLS95_TARGET','GROVER_ITERATIONS']).iloc[0]
        out.append(dict(zip(keys,ident)) | dict(
            BASELINE_R=0, R0_SUCCESS_PROB=float(baseline['SUCCESS_PROB']),
            R0_SHOTS95=float(baseline['SHOTS95_TARGET']),
            R0_TOTAL_Q_CALLS95=float(baseline['TOTAL_Q_CALLS95_TARGET']),
            BEST_R=int(best['GROVER_ITERATIONS']),
            BEST_SUCCESS_PROB=float(best['SUCCESS_PROB']),
            BEST_SHOTS95=float(best['SHOTS95_TARGET']),
            BEST_TOTAL_Q_CALLS95=float(best['TOTAL_Q_CALLS95_TARGET']),
            COST_RATIO_OPT_TO_R0=float(best['TOTAL_Q_CALLS95_TARGET']/baseline['TOTAL_Q_CALLS95_TARGET']),
            SCANNED_R_MAX=int(sub['GROVER_ITERATIONS'].max()),
            SUCCESS_EVENT=str(best['SUCCESS_EVENT'])))
    if not out:
        raise ValueError('No curves were optimized.')
    return pd.DataFrame(out)


def save_tables(output, source, summary):
    output = Path(output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    a=output.with_name(output.stem+'_source.tsv')
    b=output.with_name(output.stem+'_summary.tsv')
    source.to_csv(a, sep='\t', index=False)
    summary.to_csv(b, sep='\t', index=False)
    print(f'Figure: {output}\nSource: {a}\nSummary: {b}')


def threshold_label(x):
    return f'{float(x):.3g}'


def plot_cost_pair(ax, data, title, ylabel, legend=False):
    import matplotlib.pyplot as plt
    xs=None
    for col, lab, ls, mark in (
        ('R0_TOTAL_Q_CALLS95', 'No QSEL (r=0)', '--', 'o'),
        ('BEST_TOTAL_Q_CALLS95','Optimized over scanned r','-','s')):
        d=data[data['METRIC']==col].sort_values('OCC_COUNT')
        if d.empty: raise ValueError(f'Missing {col} in {title}')
        x=d.OCC_COUNT.to_numpy(dtype=int)
        (line,) = ax.plot(x,d.MEDIAN.to_numpy(float), ls=ls,marker=mark,markersize=3.8,lw=1.6,label=lab)
        ax.fill_between(x,d.Q1.to_numpy(float),d.Q3.to_numpy(float),color=line.get_color(),alpha=.14)
    ax.set_xscale('log',base=2); ax.set_yscale('log')
    ax.set_xticks(OCC_COUNTS); ax.set_xticklabels([str(x) for x in OCC_COUNTS],rotation=45,ha='right')
    ax.set_xlabel(r'$N_{\mathrm{OCC}}$');ax.set_ylabel(ylabel);ax.set_title(title)
    ax.grid(True, which='both',ls='--',lw=.5,alpha=.3)
    if legend: ax.legend(fontsize=8)
