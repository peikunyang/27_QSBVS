#!/usr/bin/env python3
"""Extract the PRE-QAE A0 for one receptor state, using original 2_QAE/energy.py.

This deliberately IMPORTS the exact original 2_QAE loader, selector logic,
rotations, and FFT correlation rather than copying them. It does not execute
QAE/QPE, QCOMP or QSEL, and it never reconstructs unnormalized energy E.
"""
from __future__ import annotations
import argparse
import importlib.util
import sys
from pathlib import Path
import numpy as np


def load_original(path: Path):
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f'Original 2_QAE energy.py not found: {path}')
    spec = importlib.util.spec_from_file_location('original_2_qae_energy', path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Cannot import {path}')
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # required for dataclasses in original program
    spec.loader.exec_module(mod)
    return mod


def main():
    parser = argparse.ArgumentParser(description='Original 2_QAE normalized A0 before QAE/QPE, single MAP state.')
    parser.add_argument('--qae-script', required=True, type=Path, help='Path to unchanged 2_QAE/energy.py')
    parser.add_argument('--receptor-dir', required=True, type=Path, help='Same receptor directory supplied to original energy.py (has map/ and occ/)')
    parser.add_argument('--out-dir', required=True, type=Path)
    parser.add_argument('--state', required=True, type=int, choices=(0,1))
    parser.add_argument('--max-occ', default=1, type=int, choices=(1,2,4,8,16,32,64,128,256,512,1024))
    parser.add_argument('--device', choices=('cpu','cuda','auto'), default='auto')
    parser.add_argument('--norm-atol', type=float, default=1e-8)
    args=parser.parse_args()
    qae=load_original(args.qae_script)
    receptor_dir=args.receptor_dir.resolve()
    map_dir=receptor_dir/'map'
    occ_dir=receptor_dir/'occ'
    pdbid,Lm,shift=qae.read_map_len(map_dir/'len')
    occ_lengths=qae.read_occ_len(occ_dir/'len')
    entries=qae.collect_occ_entries(occ_dir,pdbid)
    Lo=qae.validate_occ_lengths(entries,occ_lengths)
    entries=entries[:args.max_occ]
    maps=qae.load_maps(map_dir,pdbid,Lm,shift,args.norm_atol)
    backend=qae.select_backend(args.device)
    cols=qae.build_pose_columns()
    spectra=qae.prepare_map_spectra(maps,backend)
    out=args.out_dir.resolve()
    out.mkdir(parents=True,exist_ok=True)

    # Format is [occupancy index, rotation selector, translation selector].
    overlaps=np.lib.format.open_memmap(out/'overlap_fft.npy',mode='w+',dtype=np.float64,shape=(len(entries),32,512))
    amplitudes=np.lib.format.open_memmap(out/'A0_fft.npy',mode='w+',dtype=np.float64,shape=(len(entries),32,512))
    with (out/'occ_mapping.tsv').open('w',encoding='utf-8') as f:
        f.write('OCC\tOCC_NAME\tBIN_PATH\n')
        for item in entries:
            f.write(f'{item.selector}\t{item.name}\t{item.path.resolve()}\n')

    denominator=Lm*Lo
    print(f'[2_QAE FFT] PDBID={pdbid} MAP={args.state} OCCs={len(entries)} device={backend.name}',flush=True)
    for j,item in enumerate(entries):
        full=qae.read_full_vector(item.path,args.norm_atol)
        qae.validate_occ_auxiliary(full,item.path,Lo)
        phys=np.ascontiguousarray(full[:qae.PHYSICAL_DATA_SIZE])
        normalized_physical=qae.compute_occ_pose_overlaps(spectra,phys,cols,backend)[args.state]
        aux=float(np.dot(maps[args.state].amplitudes[qae.PHYSICAL_DATA_SIZE:], full[qae.PHYSICAL_DATA_SIZE:]))
        expected_aux=shift/denominator
        if not np.isclose(aux,expected_aux,atol=1e-10,rtol=1e-10):
            raise AssertionError(f'Auxiliary overlap and stored 2_QAE shift disagree: {aux}, {expected_aux}')
        # Exactly as in 2_QAE main loop, but without restoring original E.
        a0=np.clip((normalized_physical*denominator+shift)/denominator,0.0,1.0)
        overlaps[j]=normalized_physical.reshape(512,32).T
        amplitudes[j]=a0.reshape(512,32).T
        if (j+1)%16==0 or j+1==len(entries):
            overlaps.flush();amplitudes.flush()
            print(f'[2_QAE FFT] {j+1}/{len(entries)}',flush=True)
    (out/'source_info.txt').write_text(
        f'Original source: {args.qae_script.resolve()}\n'
        f'Input receptor directory: {receptor_dir}\n'
        f'PDBID: {pdbid}\nMAP state: {args.state}\n'
        f'Original FFT backend: {backend.name}\n'
        f'Output measure: normalized physical overlap and full encoded A0, NOT original E\n'
        f'OCC count: {len(entries)}\n',encoding='utf-8'
    )
    print('[2_QAE FFT] saved overlap_fft.npy and A0_fft.npy',flush=True)

if __name__=='__main__': main()
