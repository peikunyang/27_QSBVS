#!/bin/bash

src_root="/mnt/SSD/kun/job/22_QSBVS/2026_04_25/2_MD/1_charmm_gui/2_pro"
dst_root="./pdb"
common_public="./public"

mkdir -p "$dst_root"

for dir in "$src_root"/*; do
    [ -d "$dir" ] || continue

    name=$(basename "$dir")
    src="$dir/openmm"
    dst="$dst_root/$name"

    [ -d "$src" ] || continue
    mkdir -p "$dst"

    for f in step3_input.psf step3_input.crd sysinfo.dat toppar.str \
         step4_equilibration.inp step5_production.inp; do
        [ -e "$src/$f" ] && cp -a "$src/$f" "$dst/"
    done

    [ -e "$src/restraints/prot_pos.txt" ] && cp -a "$src/restraints/prot_pos.txt" "$dst/"

    echo "done: $name"
done

