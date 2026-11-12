#!/usr/bin/env python3
"""Supplementary figure generation from original QAE numerical outputs."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from figure_utils import (RECEPTORS, K_CHOICES, OCC_COUNTS, M0_PANELS,
                          load_curves, shots95, discover_pose_runs)
from preview_common import style, summarize_eight, save_svg_and_tables, basic_plot

SUPP_OCC_PANELS = (1, 16, 128, 1024)
SUPP_M0_SHOT_COST = 14


def draw_s1(data_root, output):
    """q = M/N, independent of QAE resolution k, against threshold."""
    style()
    file = Path(data_root) / 'batch_summary_pose.tsv'
    frame = pd.read_csv(file, sep='\t')
    needed = {'PDBID','OCC_COUNT','RESOLUTION_MULTIPLE_TARGET',
              'AMPLITUDE_THRESHOLD','EXACT_GOOD_FRACTION'}
    if not needed.issubset(frame.columns):
        raise ValueError(f'{file}: missing {sorted(needed-set(frame.columns))}')
    source=frame[(frame['OCC_COUNT'].isin(SUPP_OCC_PANELS)) &
                 (frame['RESOLUTION_MULTIPLE_TARGET']==1)].copy()
    if source.empty or not source.EXACT_GOOD_FRACTION.between(0,1).all():
        raise ValueError('Invalid or absent q data')
    summary=summarize_eight(source,['OCC_COUNT','AMPLITUDE_THRESHOLD'],
                            'EXACT_GOOD_FRACTION')
    fig,axes=plt.subplots(2,2,figsize=(13,9.4),constrained_layout=True)
    for ax,letter,occ in zip(axes.flat,'abcd',SUPP_OCC_PANELS):
        sub=summary[summary.OCC_COUNT==occ].sort_values('AMPLITUDE_THRESHOLD')
        if len(sub)!=13: raise ValueError(f'Expected 13 thresholds at N_OCC={occ}')
        x=sub.AMPLITUDE_THRESHOLD.to_numpy(float)
        y=sub.MEDIAN.to_numpy(float)
        ax.plot(x,y,color='C0',marker='o',ms=3.2,lw=1.6,label=r'$q$')
        ax.fill_between(x,sub.Q1.to_numpy(float),sub.Q3.to_numpy(float),
                        alpha=.14,color='C0')
        ax.set_xlabel(r'$A_{0,\mathrm{thr}}$')
        ax.set_ylabel(r'$q$')
        ax.set_title(rf'({letter})  $N_{{\mathrm{{OCC}}}} = {occ}$')
    basic_plot(fig,axes,logy=True)
    for ax in axes.flat: ax.set_ylim(1e-7,1.35)
    fig.suptitle('Figure S1. Threshold-qualified fraction in benchmark libraries',fontsize=14)
    fig.text(.99,.007,'Eight-receptor median; shaded IQR',ha='right',fontsize=9,color='.38')
    save_svg_and_tables(fig,output,source,summary)


def _common_r_source(data_root, occ_panels, m0):
    chunks=[load_curves(data_root,k=k,m0_values=(m0,),
                        occ_values=occ_panels,target='flag') for k in K_CHOICES]
    full=pd.concat(chunks,ignore_index=True)
    bounds={}
    for occ in occ_panels:
        for k in K_CHOICES:
            df=full[(full.OCC_COUNT==occ)&(full.BASE_M==m0)&(full.K==k)]
            if df.empty: raise ValueError(f'Missing N_OCC={occ},m0={m0},k={k}')
            last=df.groupby('PDBID').GROVER_ITERATIONS.max()
            if len(last)!=8 or set(last.index)!=set(RECEPTORS):
                raise ValueError(f'Incomplete eight receptors at {occ}/{m0}/{k}')
            endpoint=int(last.min())
            for name,g in df.groupby('PDBID'):
                rs=set(g.GROVER_ITERATIONS.astype(int))
                if not set(range(endpoint+1)).issubset(rs):
                    raise ValueError(f'Noncontiguous r for {name}/{occ}/{m0}/{k}')
            bounds[(occ,k)]=endpoint
    source=pd.concat([full[(full.OCC_COUNT==occ)&(full.K==k)&
                           (full.GROVER_ITERATIONS<=bounds[(occ,k)])]
                      for occ in occ_panels for k in K_CHOICES],ignore_index=True)
    return source,bounds


def draw_s2_s3(data_root, output, kind):
    """Equal settings for shots and cost, 4 library sizes, 4 k curves."""
    if kind not in ('shots','cost'): raise ValueError(kind)
    style()
    m0=SUPP_M0_SHOT_COST
    source,bounds=_common_r_source(data_root,SUPP_OCC_PANELS,m0)
    key='SHOTS95_TARGET' if kind=='shots' else 'TOTAL_Q_CALLS95_TARGET'
    ylabel=r'$S_{95}(r)$' if kind=='shots' else r'$C_{95}(r)$'
    summary=summarize_eight(source,['OCC_COUNT','BASE_M','K','GROVER_ITERATIONS'],key)
    fig,axes=plt.subplots(2,2,figsize=(13,9.4),constrained_layout=True)
    colors=plt.rcParams['axes.prop_cycle'].by_key()['color']
    for ax,letter,occ in zip(axes.flat,'abcd',SUPP_OCC_PANELS):
        rmax=max(bounds[(occ,k)] for k in K_CHOICES)
        for ki,k in enumerate(K_CHOICES):
            d=summary[(summary.OCC_COUNT==occ)&(summary.K==k)].sort_values('GROVER_ITERATIONS')
            if len(d)!=bounds[(occ,k)]+1:
                raise ValueError(f'Incomplete r summary at {occ}/{k}')
            x=d.GROVER_ITERATIONS.to_numpy(float)
            ax.plot(x,d.MEDIAN.to_numpy(float),label=f'k = {k}',lw=1.6,color=colors[ki])
            ax.fill_between(x,d.Q1.to_numpy(float),d.Q3.to_numpy(float),
                            alpha=.11,color=colors[ki])
        ax.set_title(rf'({letter})  $N_{{\mathrm{{OCC}}}} = {occ},\ m_0={m0}$')
        ax.set_xlabel(r'$r$')
        ax.set_ylabel(ylabel)
        ax.set_xlim(0,rmax)
        if rmax<=8: ax.set_xticks(list(range(0,rmax+1)))
    basic_plot(fig,axes,logy=True,scale_x='linear')
    fig.legend(*axes.flat[0].get_legend_handles_labels(),loc='outside lower center',
               ncol=4,frameon=False)
    title=('Figure S2. FLAG-detection shots across library sizes'
           if kind=='shots' else 'Figure S3. FLAG-detection Q-operator calls across library sizes')
    fig.suptitle(title,fontsize=14)
    fig.text(.99,.007,'Eight-receptor median; shaded IQR',ha='right',fontsize=9,color='.38')
    save_svg_and_tables(fig,output,source,summary)
    print(f'Scanned r limits for m0={m0}: {bounds}')


def _optimized_flag_cost_rows(data_root):
    """Load one run at a time. Compute FLAG-only costs from observed probabilities."""
    out=[]
    required=['OCC_COUNT','GROVER_ITERATIONS','AMPLIFIED_FLAG_PROB','Q_CALLS_PER_SHOT']
    for run in discover_pose_runs(data_root,m0_values=M0_PANELS):
        path=run['GROVER_PATH']
        raw=pd.read_csv(path,sep='\t',usecols=required)
        for col in required:
            raw[col]=pd.to_numeric(raw[col],errors='raise')
        if not np.all(np.isfinite(raw[required])):
            raise ValueError(f'Nonfinite input {path}')
        for occ,sub in raw.groupby('OCC_COUNT'):
            if occ not in OCC_COUNTS: continue
            if sub.GROVER_ITERATIONS.duplicated().any():
                raise ValueError(f'Duplicate r for {path} N_OCC={occ}')
            if 0 not in set(sub.GROVER_ITERATIONS):
                raise ValueError(f'Missing r=0 for {path},N_OCC={occ}')
            r=sub.GROVER_ITERATIONS.to_numpy(dtype=float)
            if np.any(r<0) or np.any(r!=np.floor(r)):
                raise ValueError(f'Invalid r in {path}')
            qc=sub.Q_CALLS_PER_SHOT.to_numpy(dtype=float)
            predicted=(2*r+1)*(2**run['EVAL_QUBITS']-1)
            if not np.allclose(qc,predicted,rtol=0,atol=0.1):
                raise ValueError(f'Q calls incorrect in {path}')
            p=sub.AMPLIFIED_FLAG_PROB.to_numpy(dtype=float)
            shots=shots95(p)
            costs=shots*qc
            valid=np.where(np.isfinite(costs))[0]
            if not len(valid):
                raise ValueError(f'No finite FLAG detection cost in {path},N_OCC={occ}')
            idx=valid[np.argmin(costs[valid])]
            zero=np.where(r==0)[0]
            if len(zero)!=1: raise ValueError(f'No unique r=0 for {path}')
            out.append(dict(PDBID=run['PDBID'],BASE_M=run['BASE_M'],K=run['K'],
                            OCC_COUNT=int(occ),AMPLITUDE_THRESHOLD=run['AMPLITUDE_THRESHOLD'],
                            EVAL_QUBITS=run['EVAL_QUBITS'],BEST_R=int(r[idx]),
                            BEST_TOTAL_Q_CALLS95=float(costs[idx]),
                            R0_TOTAL_Q_CALLS95=float(costs[zero[0]]),
                            SCANNED_R_MAX=int(np.max(r)),SUCCESS_EVENT='flag'))
    if not out: raise ValueError('No S4 data')
    return pd.DataFrame(out)

def draw_s4(data_root,output):
    """For each receptor, minimize over measured r first; then show median/IQR."""
    style()
    source=_optimized_flag_cost_rows(data_root)
    summary=summarize_eight(source,['OCC_COUNT','BASE_M','K'],
                            'BEST_TOTAL_Q_CALLS95')
    fig,axes=plt.subplots(2,2,figsize=(13,9.4),constrained_layout=True)
    colors=plt.rcParams['axes.prop_cycle'].by_key()['color']
    for ax,letter,m0 in zip(axes.flat,'abcd',M0_PANELS):
        for ki,k in enumerate(K_CHOICES):
            d=summary[(summary.BASE_M==m0)&(summary.K==k)].sort_values('OCC_COUNT')
            if len(d)!=len(OCC_COUNTS):
                raise ValueError(f'Missing library sizes at m0={m0},k={k}')
            x=d.OCC_COUNT.to_numpy(float)
            ax.plot(x,d.MEDIAN.to_numpy(float),marker='o',ms=3.2,
                    lw=1.6,color=colors[ki],label=f'k = {k}')
            ax.fill_between(x,d.Q1.to_numpy(float),d.Q3.to_numpy(float),
                            color=colors[ki],alpha=.11)
        ax.set_title(rf'({letter})  $m_0 = {m0}$')
        ax.set_xscale('log',base=2)
        ax.set_yscale('log')
        ax.set_xticks(OCC_COUNTS)
        ax.set_xticklabels([str(n) for n in OCC_COUNTS],rotation=45,ha='right')
        ax.set_xlabel(r'$N_{\mathrm{OCC}}$')
        ax.set_ylabel(r'$\min_{r}\,C_{95}(r)$')
        ax.grid(True,which='major',linestyle='--',linewidth=.55,alpha=.33)
    fig.legend(*axes.flat[0].get_legend_handles_labels(),loc='outside lower center',
               ncol=4,frameon=False)
    fig.suptitle('Figure S4. Minimum FLAG-detection cost over sampled QSEL iterations',fontsize=14)
    fig.text(.99,.007,'Per-receptor minimum, then eight-receptor median; shaded IQR',
             ha='right',fontsize=9,color='.38')
    save_svg_and_tables(fig,output,source,summary)


def generate_s1_table(output):
    counts=[]
    for nocc in OCC_COUNTS:
        occ_qubits=int(np.log2(nocc))
        counts.append(dict(N_OCC=nocc,N_SEARCH=32768*nocc,
                           RECEPTOR_SELECTOR_QUBITS=1,OCC_SELECTOR_QUBITS=occ_qubits,
                           ROT_SELECTOR_QUBITS=5,TRANS_SELECTOR_QUBITS=9,
                           TOTAL_SELECTOR_QUBITS=15+occ_qubits))
    output=Path(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(counts).to_csv(output,sep='\t',index=False)
    print(f'Created {output}')


def generate_s2_table(output):
    rows=[]
    for m0 in range(10,23):
        row={'BASE_M0':m0,'A0_THR':float(np.sin(np.pi/(2**m0)))}
        for k in K_CHOICES:
            row[f'M_K{k}']=m0+int(np.log2(k))
        rows.append(row)
    output=Path(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(output,sep='\t',index=False,float_format='%.12g')
    print(f'Created {output}')
