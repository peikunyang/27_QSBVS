#!/usr/bin/env python3
"""Single receptor-state pre-QAE A0 comparison, using ORIGINAL normalized bins."""
from __future__ import annotations
import argparse
from pathlib import Path
import subprocess
import sys
import numpy as np

HERE=Path(__file__).resolve().parent


def choose(label,options):
    options=[str(x) for x in options]
    while True:
        val=input(label).strip()
        if val in options: return val
        print('Valid choices:',', '.join(options))


def main():
    p=argparse.ArgumentParser(description='Verify original 2_QAE pre-QPE A0 with independent non-FFT sparse summation.')
    p.add_argument('--input-root',required=True,type=Path,
                   help='Same 2_norm root used by original 2_QAE/energy.py, with PDBID/map/ and PDBID/occ/')
    p.add_argument('--qae-script',required=True,type=Path,help='Original unchanged 2_QAE/energy.py')
    p.add_argument('--pdbid',default=None)
    p.add_argument('--state',default=None,type=int,choices=(0,1))
    p.add_argument('--max-occ',default=1,type=int,choices=(1,2,4,8,16,32,64,128,256,512,1024))
    p.add_argument('--device',choices=('cpu','auto','cuda'),default='auto')
    p.add_argument('--batch-trans',type=int,default=64)
    p.add_argument('--out-dir',default='verify_out',type=Path)
    p.add_argument('--atol',type=float,default=1e-9)
    p.add_argument('--rtol',type=float,default=1e-9)
    args=p.parse_args()
    input_root=args.input_root.resolve()
    groups=sorted(d for d in input_root.iterdir() if d.is_dir() and (d/'map'/'len').is_file() and (d/'occ'/'len').is_file())
    if not groups: raise FileNotFoundError(f'No original 2_QAE normalized receptor groups under {input_root}')
    valid=[d.name for d in groups]
    if args.pdbid is None:
        print('Available receptor groups:',', '.join(valid))
        args.pdbid=choose('Choose PDBID: ',valid)
    if args.pdbid not in valid: raise ValueError(f'{args.pdbid} not in {valid}')
    if args.state is None: args.state=int(choose('Choose MAP conformation (0 or 1): ',['0','1']))
    receptor_dir=input_root/args.pdbid
    device=args.device
    if device=='auto':
        try:
            import torch
            device='cuda' if torch.cuda.is_available() else 'cpu'
        except ImportError:
            device='cpu'
    target=(args.out_dir/f'{args.pdbid}_map{args.state}_occ{args.max_occ}').resolve()
    target.mkdir(parents=True,exist_ok=True)
    base=['--receptor-dir',str(receptor_dir),'--state',str(args.state),'--out-dir',str(target)]
    print(f'Comparing pre-QPE A0: receptor {args.pdbid}, MAP {args.state}, OCCs {args.max_occ}, device {device}',flush=True)
    subprocess.run([sys.executable,str(HERE/'2_QAE_energy.py'),*base,
                    '--qae-script',str(args.qae_script),'--max-occ',str(args.max_occ),'--device',device],check=True)
    subprocess.run([sys.executable,str(HERE/'4_energy_direct.py'),*base,
                    '--device',device,'--batch-trans',str(args.batch_trans)],check=True)
    f=np.load(target/'A0_fft.npy',mmap_mode='r')
    d=np.load(target/'A0_direct.npy',mmap_mode='r')
    pf=np.load(target/'overlap_fft.npy',mmap_mode='r')
    pd=np.load(target/'overlap_direct.npy',mmap_mode='r')
    if f.shape!=d.shape or pf.shape!=pd.shape or f.shape!=pf.shape:
        raise AssertionError(f'Array shape mismatch FFT {f.shape}/{pf.shape}, DIRECT {d.shape}/{pd.shape}')
    err_max=-1.0; err_total=0.0; sum_sq=0.0; failures=0; checked=0;worst=None;examples=[]
    physical_max=0.0;physical_bad=0
    for i in range(f.shape[0]):
        a=np.asarray(f[i]);b=np.asarray(d[i]); pa=np.asarray(pf[i]);pb=np.asarray(pd[i])
        delta=np.abs(a-b)
        bad=~np.isclose(a,b,rtol=args.rtol,atol=args.atol)
        pdelta=np.abs(pa-pb)
        physical_max=max(physical_max,float(np.max(pdelta)))
        physical_bad+=int(np.count_nonzero(~np.isclose(pa,pb,atol=args.atol,rtol=args.rtol)))
        checked+=delta.size;err_total+=float(np.sum(delta));sum_sq+=float(np.sum(delta**2));failures+=int(np.count_nonzero(bad))
        idx=np.unravel_index(int(np.argmax(delta)),delta.shape)
        if float(delta[idx])>err_max:
            err_max=float(delta[idx]);worst=(i,*map(int,idx),float(a[idx]),float(b[idx]))
        if np.any(bad) and len(examples)<100:
            for r,t in np.argwhere(bad)[:100-len(examples)]:
                examples.append((i,int(r),int(t),float(a[r,t]),float(b[r,t])))
    report=(
        f'INPUT: original 2_QAE normalized 19-qubit .bin vectors\n'
        f'PDBID: {args.pdbid}\nMAP conformation: {args.state}\n'
        f'Number of occupancies: {args.max_occ}\nROT: 32, TRANS: 512\n'
        f'Configurations compared: {checked}\n'
        f'Quantity compared: full encoded A0 before QAE/QPE (no original-energy reconstruction)\n'
        f'FFT: original 2_QAE overlap path\nDIRECT: independent non-FFT sparse inner product\n'
        f'A0 max absolute error: {err_max:.16e}\n'
        f'A0 mean absolute error: {err_total/checked:.16e}\n'
        f'A0 RMSE: {(sum_sq/checked)**.5:.16e}\n'
        f'Physical overlap max absolute error: {physical_max:.16e}\n'
        f'A0 failed: {failures}\nPhysical overlap failed: {physical_bad}\n'
        f'Tolerance: atol={args.atol} rtol={args.rtol}\n'
        f'Worst case (OCC,ROT,TRANS,FFT_A0,DIRECT_A0): {worst}\n'
        f'Gate-level validation: NO; this is pre-QAE numerical consistency verification\n'
    )
    (target/'comparison.txt').write_text(report,encoding='utf-8')
    (target/'failed_examples.tsv').write_text('OCC\tROT\tTRANS\tFFT_A0\tDIRECT_A0\n'+''.join(f'{i}\t{r}\t{t}\t{a:.16e}\t{b:.16e}\n' for i,r,t,a,b in examples),encoding='utf-8')
    print('\n'+report)
    print('Results:',target)
    if failures or physical_bad: raise SystemExit(2)

if __name__=='__main__': main()
