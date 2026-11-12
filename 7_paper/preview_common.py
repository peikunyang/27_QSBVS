from pathlib import Path
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from figure_utils import RECEPTORS, K_CHOICES, load_curves, shots95

OCC_PANELS = (1, 16, 128, 1024)
RESOURCE_PANELS = ((1024, 10), (1024, 14), (1024, 18), (1024, 22))
K_VALUES = (1, 2, 4, 8)


def style():
    plt.rcParams.update({
        'font.size': 10, 'axes.titlesize': 11, 'axes.labelsize': 11,
        'legend.fontsize': 9, 'mathtext.fontset': 'dejavusans',
        'axes.spines.top': False, 'axes.spines.right': False,
        'svg.fonttype': 'none', 'savefig.dpi': 240,
    })


def summarize_eight(df, group_cols, value):
    found = df.groupby(group_cols, dropna=False, sort=True)
    rows = []
    for key, grp in found:
        if len(grp) != 8 or set(grp.PDBID.str.lower()) != set(RECEPTORS):
            raise ValueError(f'Expected exactly 8 distinct receptors for {key}, got {list(grp.PDBID)}')
        v = grp[value].astype(float).to_numpy()
        if not np.all(np.isfinite(v)):
            raise ValueError(f'Invalid {value} for {key}')
        keys = key if isinstance(key, tuple) else (key,)
        rows.append(dict(zip(group_cols, keys)) | dict(
            MEDIAN=float(np.median(v)), Q1=float(np.percentile(v,25)),
            Q3=float(np.percentile(v,75)), N_RECEPTORS=8))
    return pd.DataFrame(rows)


def save_svg_and_tables(fig, output, source, summary):
    output = Path(output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() != '.svg':
        raise ValueError(f'Output must be SVG: {output}')
    fig.savefig(output, format='svg', bbox_inches='tight', pad_inches=.15)
    plt.close(fig)
    source.to_csv(output.with_name(output.stem+'_source.tsv'), sep='\t', index=False)
    summary.to_csv(output.with_name(output.stem+'_summary.tsv'), sep='\t', index=False)
    print(f'Created {output}')


def basic_plot(fig, axes, ylim=None, logy=False, scale_x='log'):
    for ax in axes.flat:
        ax.set_xscale(scale_x)
        if logy: ax.set_yscale('log')
        if ylim is not None: ax.set_ylim(*ylim)
        ax.grid(True, which='major', linestyle='--', linewidth=.55, alpha=.33)
        ax.tick_params(axis='both', labelsize=9)


def make_batch_figure(input_file, output, metric, title, ylabel, logy=False):
    style()
    df = pd.read_csv(input_file, sep='\t')
    required = ('PDBID','OCC_COUNT','RESOLUTION_MULTIPLE_TARGET','AMPLITUDE_THRESHOLD',metric)
    if set(required)-set(df.columns):
        raise ValueError(f'Missing columns: {sorted(set(required)-set(df.columns))}')
    source = df[(df.OCC_COUNT.isin(OCC_PANELS)) &
                (df.RESOLUTION_MULTIPLE_TARGET.isin(K_VALUES))].copy()
    if source.empty: raise ValueError('No batch data for selected N_OCC and k')
    if not np.isfinite(source[metric].astype(float)).all(): raise ValueError(f'{metric}: invalid value')
    if not source[metric].between(0, 1).all(): raise ValueError(f'{metric}: not a probability')
    summary = summarize_eight(source,
        ['OCC_COUNT','RESOLUTION_MULTIPLE_TARGET','AMPLITUDE_THRESHOLD'],metric)
    fig, axes = plt.subplots(2,2,figsize=(13,9.4),constrained_layout=True)
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for ax,letter,occ in zip(axes.flat,'abcd',OCC_PANELS):
        for ki,k in enumerate(K_VALUES):
            d = summary[(summary.OCC_COUNT==occ) &
                        (summary.RESOLUTION_MULTIPLE_TARGET==k)].sort_values('AMPLITUDE_THRESHOLD')
            if len(d)!=13: raise ValueError(f'N_OCC={occ},k={k}: expected 13 thresholds, got {len(d)}')
            x = d.AMPLITUDE_THRESHOLD.to_numpy(float)
            y = d.MEDIAN.to_numpy(float)
            ax.plot(x,y,label=f'k = {k}',marker='o',ms=3.2,lw=1.6,color=colors[ki])
            ax.fill_between(x,d.Q1.to_numpy(float),d.Q3.to_numpy(float),alpha=.11,color=colors[ki])
        ax.set_title(rf'({letter})  $N_{{\mathrm{{OCC}}}} = {occ}$')
        ax.set_xlabel(r'$A_{0,\mathrm{thr}}$')
        ax.set_ylabel(ylabel)
    basic_plot(fig,axes,ylim=(0,1.02) if not logy else None,logy=logy)
    if logy: 
        for ax in axes.flat: ax.set_ylim(1e-7,1.35)
    fig.legend(*axes.flat[0].get_legend_handles_labels(),loc='outside lower center',ncol=4,frameon=False)
    fig.suptitle(title,fontsize=14)
    fig.text(.99,.007,'Eight-receptor median; shaded IQR',ha='right',fontsize=9,color='.38')
    save_svg_and_tables(fig,output,source,summary)


def load_flag_for_panels(data_root):
    frames=[]
    for k in K_VALUES:
        frames.append(load_curves(data_root, k=k, m0_values=(10,14,18,22),
                                  occ_values=(1024,), target='flag'))
    data=pd.concat(frames,ignore_index=True)
    if set(data.PDBID)!=set(RECEPTORS): raise ValueError('Missing receptor')
    return data


def make_resource_figure(data_root, output, metric, ylabel, title):
    style()
    source_all=load_flag_for_panels(data_root)
    # Keep all scanned iterations shared by eight receptors separately for
    # each (m0, k). Do not truncate k=1 at the last iteration of k=8.
    bounds={}
    for occ,m0 in RESOURCE_PANELS:
        for k in K_VALUES:
            d=source_all[(source_all.OCC_COUNT==occ)&
                         (source_all.BASE_M==m0)&(source_all.K==k)]
            if len(d)==0: raise ValueError(f'No data at N_OCC={occ}, m0={m0}, k={k}')
            group=d.groupby('PDBID')['GROVER_ITERATIONS']
            maxes=group.max()
            if len(maxes)!=8 or set(maxes.index)!=set(RECEPTORS):
                raise ValueError(f'Incomplete receptor runs: N_OCC={occ}, m0={m0}, k={k}')
            common_max=int(maxes.min())
            for _,sub in d.groupby('PDBID'):
                rset=set(sub.GROVER_ITERATIONS.astype(int))
                if not set(range(common_max+1)).issubset(rset):
                    raise ValueError(f'Non-contiguous r: N_OCC={occ}, m0={m0}, k={k}')
            bounds[(occ,m0,k)]=common_max
    source=pd.concat([source_all[
        (source_all.OCC_COUNT==occ)&(source_all.BASE_M==m0)&
        (source_all.K==k)&(source_all.GROVER_ITERATIONS<=bounds[(occ,m0,k)])]
        for (occ,m0) in RESOURCE_PANELS for k in K_VALUES],ignore_index=True)
    summary=summarize_eight(source,['OCC_COUNT','BASE_M','K','GROVER_ITERATIONS'],metric)
    fig,axes=plt.subplots(2,2,figsize=(13,9.4),constrained_layout=True)
    colors=plt.rcParams['axes.prop_cycle'].by_key()['color']
    for ax,letter,(occ,m0) in zip(axes.flat,'abcd',RESOURCE_PANELS):
        rmax=max(bounds[(occ,m0,k)] for k in K_VALUES)
        for ki,k in enumerate(K_VALUES):
            kmax=bounds[(occ,m0,k)]
            d=summary[(summary.OCC_COUNT==occ)&(summary.BASE_M==m0)&
                      (summary.K==k)].sort_values('GROVER_ITERATIONS')
            if len(d)!=kmax+1:
                raise ValueError(f'Incomplete summary at N_OCC={occ},m0={m0},k={k}')
            x=d.GROVER_ITERATIONS.to_numpy(float)
            ax.plot(x,d.MEDIAN.to_numpy(float),label=f'k = {k}',lw=1.6,color=colors[ki])
            ax.fill_between(x,d.Q1.to_numpy(float),d.Q3.to_numpy(float),
                            color=colors[ki],alpha=.11)
        ax.set_title(rf'({letter})  $m_0 = {m0},\ N_{{\mathrm{{OCC}}}} = {occ}$')
        ax.set_xlabel(r'$r$')
        ax.set_ylabel(ylabel)
        ax.set_xlim(0,rmax)
        if rmax<=8: ax.set_xticks(list(range(0,rmax+1)))
    basic_plot(fig,axes,logy=True,scale_x='linear')
    fig.legend(*axes.flat[0].get_legend_handles_labels(),
               loc='outside lower center',ncol=4,frameon=False)
    fig.suptitle(title,fontsize=14)
    fig.text(.99,.007,'Eight-receptor median; shaded IQR',ha='right',fontsize=9,color='.38')
    save_svg_and_tables(fig,output,source,summary)
    print(f'Scanned r maxima (8 receptors per curve): {bounds}')
