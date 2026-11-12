#!/bin/bash
src_root="/mnt/SSD/kun/job/22_QSBVS/2026_04_25/2_MD/1_charmm_gui/1_comp"
dst_root="./pdb"

mkdir -p "$dst_root"

for dir in "$src_root"/*; do
    [ -d "$dir" ] || continue

    name=$(basename "$dir")
    src_openmm="$dir/openmm"
    dst="$dst_root/$name"

    [ -d "$src_openmm" ] || continue
    mkdir -p "$dst"

    # 固定需要的 openmm 檔案
    for f in step3_input.psf step3_input.crd sysinfo.dat toppar.str \
             step4_equilibration.inp step5_production.inp; do
        [ -e "$src_openmm/$f" ] && cp -a "$src_openmm/$f" "$dst/"
    done

    [ -e "$src_openmm/restraints/prot_pos.txt" ] && cp -a "$src_openmm/restraints/prot_pos.txt" "$dst/"

    found_lig=0

    # 把所有不是 openmm、不是 toppar 的子資料夾都 copy
    for sub in "$dir"/*; do
        [ -d "$sub" ] || continue
        base=$(basename "$sub")

        if [ "$base" != "openmm" ] && [ "$base" != "toppar" ]; then
            cp -a "$sub" "$dst/"
            echo "ligand folder copied for $name: $base"
            found_lig=1
        fi
    done

    if [ "$found_lig" -eq 0 ]; then
        echo "warning: no ligand folder found for $name"
    fi

    echo "done: $name"
done

