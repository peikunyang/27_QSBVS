#!/usr/bin/env python3
"""INDEPENDENT, non-FFT sparse direct overlap on the SAME normalized 2_QAE bins.

No imports from 2_QAE or its geometry functions. Rotations are explicit
integer-coordinate transformations, shifts are coordinate additions modulo 32.
The full 19-qubit encoded A0 is physical sparse overlap + auxiliary overlap.
NO restoration of original E, and no QAE/QPE/QCOMP/QSEL.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np

N=32
PHYS=8*N**3
FULL=2*PHYS


def load_bin(path:Path):
    if not path.is_file(): raise FileNotFoundError(path)
    a=np.fromfile(path,dtype=np.float64)
    if len(a)!=FULL or not np.isfinite(a).all():
        raise ValueError(f'Expected 2^19 finite float64 entries at {path}, got {len(a)}')
    return a


def shifts():
    sel=np.arange(512,dtype=np.int64)
    def shift(low):
        b0=(sel>>low)&1
        b1=(sel>>(low+1))&1
        b2=(sel>>(low+2))&1
        return b0+2*b1-4*(1-b2)
    return shift(6),shift(3),shift(0)


def rotate_coords(z,y,x,rot):
    """Independent formula reproducing the 32 selector-defined grid permutations."""
    if rot&16: y,x=N-1-x,y
    if rot&8: y,x=N-1-y,N-1-x
    if rot&4: z,x=x,N-1-z
    if rot&2: z,x=N-1-z,N-1-x
    if rot&1: z,y=N-1-y,z
    return z,y,x


def sparse_occ(full):
    phys=full[:PHYS].reshape(8,N,N,N)
    c,z,y,x=np.nonzero(phys)
    weights=phys[c,z,y,x]
    return c.astype(np.int64),z.astype(np.int64),y.astype(np.int64),x.astype(np.int64),weights


def load_manifest(path):
    result=[]
    with path.open('r',encoding='utf-8') as f:
        header=next(f).strip()
        if header!='OCC\tOCC_NAME\tBIN_PATH':
            raise ValueError(f'Unrecognized OCC manifest header: {header}')
        for line in f:
            bits=line.rstrip('\n').split('\t')
            if len(bits)!=3: raise ValueError(f'Malformed OCC mapping: {line}')
            idx,name,filename=bits
            result.append((int(idx),name,Path(filename)))
    if [x[0] for x in result]!=list(range(len(result))):
        raise ValueError('OCC indices not sequential')
    return result


def direct_one_numpy(rec,c,z,y,x,w,shiftz,shifty,shiftx,batch):
    flat=rec[:PHYS]
    result=np.empty((32,512),dtype=np.float64)
    for r in range(32):
        rz,ry,rx=rotate_coords(z,y,x,r)
        for st in range(0,512,batch):
            en=min(st+batch,512)
            zs=(rz[None,:]+shiftz[st:en,None])%32
            ys=(ry[None,:]+shifty[st:en,None])%32
            xs=(rx[None,:]+shiftx[st:en,None])%32
            addresses=((c[None,:]*32+zs)*32+ys)*32+xs
            result[r,st:en]=np.sum(flat[addresses]*w[None,:],axis=1,dtype=np.float64)
    return result


def direct_one_cuda(receptor,c,z,y,x,w,shiftz,shifty,shiftx,batch,torch,dev):
    c = torch.as_tensor(c, device=dev, dtype=torch.long)
    z = torch.as_tensor(z, device=dev, dtype=torch.long)
    y = torch.as_tensor(y, device=dev, dtype=torch.long)
    x = torch.as_tensor(x, device=dev, dtype=torch.long)
    w = torch.as_tensor(w, device=dev, dtype=torch.float64)
    dz=torch.as_tensor(shiftz,device=dev,dtype=torch.long)
    dy=torch.as_tensor(shifty,device=dev,dtype=torch.long)
    dx=torch.as_tensor(shiftx,device=dev,dtype=torch.long)
    result=np.empty((32,512),dtype=np.float64)
    with torch.no_grad():
        for r in range(32):
            rz,ry,rx=rotate_coords(z,y,x,r)
            for st in range(0,512,batch):
                en=min(512,st+batch)
                zs=(rz[None,:]+dz[st:en,None])%32
                ys=(ry[None,:]+dy[st:en,None])%32
                xs=(rx[None,:]+dx[st:en,None])%32
                addresses=((c[None,:]*32+zs)*32+ys)*32+xs
                result[r,st:en]=torch.sum(receptor[addresses]*w[None,:],dim=1).cpu().numpy()
    return result


def main():
    parser=argparse.ArgumentParser(description='Independent sparse (non FFT) calculation from 2_QAE normalized .bin input.')
    parser.add_argument('--receptor-dir',required=True,type=Path)
    parser.add_argument('--out-dir',required=True,type=Path)
    parser.add_argument('--state',required=True,type=int,choices=(0,1))
    parser.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    parser.add_argument('--batch-trans',type=int,default=64)
    args=parser.parse_args()
    if not 1<=args.batch_trans<=512: raise ValueError('batch-trans must be 1..512')
    out=args.out_dir.resolve()
    manifest=load_manifest(out/'occ_mapping.tsv')
    recdir=args.receptor_dir.resolve()
    parts=(recdir/'map'/'len').read_text().split()
    if len(parts)!=3: raise ValueError('Expected one PDBID, Lm, shift line in map/len')
    pdbid=parts[0]
    rec=load_bin(recdir/'map'/f'{pdbid}_{args.state}.bin')
    if args.device=='cuda':
        import torch
        if not torch.cuda.is_available(): raise RuntimeError('CUDA not available')
        dev=torch.device('cuda')
        rec_gpu=torch.as_tensor(rec[:PHYS],device=dev,dtype=torch.float64)
    else:
        rec_gpu=None
    sz,sy,sx=shifts()
    result=np.lib.format.open_memmap(out/'A0_direct.npy',mode='w+',dtype=np.float64,shape=(len(manifest),32,512))
    overlaps=np.lib.format.open_memmap(out/'overlap_direct.npy',mode='w+',dtype=np.float64,shape=(len(manifest),32,512))
    print(f'[DIRECT] PDBID={pdbid} MAP={args.state} OCCs={len(manifest)} device={args.device}',flush=True)
    for i,(_,name,bin_path) in enumerate(manifest):
        occ=load_bin(bin_path)
        c,z,y,x,w=sparse_occ(occ)
        if args.device=='cuda':
            phys=direct_one_cuda(rec_gpu,c,z,y,x,w,sz,sy,sx,args.batch_trans,torch,dev)
        else:
            phys=direct_one_numpy(rec,c,z,y,x,w,sz,sy,sx,args.batch_trans)
        aux=float(np.dot(rec[PHYS:],occ[PHYS:]))
        overlaps[i]=phys
        # A0 is exactly the 19-qubit FULL inner product, with the
        # auxiliary-pair contribution. Do NOT rescale to original E.
        result[i]=phys+aux
        if (i+1)%16==0 or i+1==len(manifest):
            overlaps.flush();result.flush()
            print(f'[DIRECT] {i+1}/{len(manifest)} ({name})',flush=True)
    print('[DIRECT] saved overlap_direct.npy and A0_direct.npy',flush=True)

if __name__=='__main__': main()
