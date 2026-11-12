#!/usr/bin/env python3
import argparse
import re
import shutil
from pathlib import Path


def parse_atom_types(pdbqt_path):
    atom_types = []
    seen = set()

    with open(pdbqt_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith(("ATOM", "HETATM")):
                parts = line.split()
                if not parts:
                    continue
                atom_type = parts[-1].strip()
                if atom_type and atom_type not in seen:
                    seen.add(atom_type)
                    atom_types.append(atom_type)

    if not atom_types:
        raise ValueError(f"no atom types found: {pdbqt_path}")

    return atom_types


def read_selected_ligand_types(log_file):
    """
    Read the selected global ligand atom-type union from sel.py log.

    Expected line, for example:
        Atom-type union (7): A,C,HD,N,NA,OA,S
    """
    pattern = re.compile(
        r"^Atom-type union\s*\((\d+)\)\s*:\s*(.+?)\s*$"
    )

    with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            match = pattern.match(line.strip())
            if not match:
                continue

            expected_count = int(match.group(1))
            atom_types = [
                x.strip()
                for x in match.group(2).split(",")
                if x.strip()
            ]

            # Remove duplicates while preserving log order.
            atom_types = list(dict.fromkeys(atom_types))

            if len(atom_types) != expected_count:
                raise ValueError(
                    f"atom-type count mismatch in {log_file}: "
                    f"log says {expected_count}, parsed {len(atom_types)} "
                    f"({', '.join(atom_types)})"
                )

            return atom_types

    raise ValueError(
        f"'Atom-type union (...)' line not found in: {log_file}"
    )


def write_gpf(job_dir, job_name, receptor_name, ligand_name, receptor_types, ligand_types, param_name):
    gpf_path = job_dir / f"{job_name}.gpf"
    receptor_stem = Path(receptor_name).stem

    with open(gpf_path, "w", encoding="utf-8") as f:
        f.write("npts 33 33 33\n")
        f.write(f"parameter_file {param_name}\n")
        f.write(f"gridfld {receptor_stem}.maps.fld\n")
        f.write("spacing 0.375\n")
        f.write("gridcenter 0.0 0.0 0.0\n")
        f.write("smooth 0.500\n")
        f.write(f"receptor_types {' '.join(receptor_types)}\n")
        f.write(f"ligand_types {' '.join(ligand_types)}\n")
        f.write(f"receptor {receptor_name}\n")

        for at in ligand_types:
            f.write(f"map {receptor_stem}.{at}.map\n")

        f.write(f"elecmap {receptor_stem}.e.map\n")
        f.write(f"dsolvmap {receptor_stem}.d.map\n")
        f.write("dielectric -0.1465\n")


def write_run_script(job_dir, job_name):
    sh_path = job_dir / "run_autogrid.sh"
    with open(sh_path, "w", encoding="utf-8") as f:
        f.write("#!/bin/bash\n")
        f.write(f'autogrid4 -p "{job_name}.gpf" -l "{job_name}.glg"\n')
    sh_path.chmod(0o755)


def prepare_job(src_dir, out_root, pdbid, idx, param_file, ligand_types):
    lig_src = src_dir / f"{pdbid}_lig.pdbqt"
    pro_src = src_dir / f"{pdbid}_pro_{idx}.pdbqt"

    if not lig_src.exists():
        raise FileNotFoundError(str(lig_src))
    if not pro_src.exists():
        raise FileNotFoundError(str(pro_src))

    job_name = f"{pdbid}_{idx}"
    job_dir = out_root / job_name
    job_dir.mkdir(parents=True, exist_ok=True)

    lig_dst = job_dir / lig_src.name
    pro_dst = job_dir / pro_src.name

    shutil.copy2(lig_src, lig_dst)
    shutil.copy2(pro_src, pro_dst)

    receptor_types = parse_atom_types(pro_src)

    param_name = "AD4_parameters.dat"
    if param_file is not None:
        shutil.copy2(param_file, job_dir / param_name)

    write_gpf(
        job_dir,
        job_name,
        pro_src.name,
        lig_src.name,
        receptor_types,
        ligand_types,
        param_name,
    )

    write_run_script(job_dir, job_name)

    return job_dir


def write_run_all(out_root, job_dirs):
    sh_path = out_root / "run_all_autogrid.sh"

    with open(sh_path, "w", encoding="utf-8") as f:
        f.write("#!/bin/bash\n\n")
        for job_dir in job_dirs:
            f.write(f'echo "running {job_dir.name}"\n')
            f.write(f'cd "{job_dir.resolve()}" || exit 1\n')
            f.write("bash run_autogrid.sh\n")
            f.write(f'cd "{out_root.resolve()}" || exit 1\n\n')
        f.write('echo "all done"\n')

    sh_path.chmod(0o755)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("--param", default=None)
    args = parser.parse_args()

    input_root = Path(args.input)
    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)

    param_file = Path(args.param) if args.param else None

    script_dir = Path(__file__).resolve().parent
    log_file = (
        script_dir
        / "../../1_data/4_lig_pdbqt/3_sel/2_decoy/log"
    ).resolve()

    if not log_file.is_file():
        print(f"[error] selection log not found: {log_file}")
        return

    ligand_types = read_selected_ligand_types(log_file)
    print(
        f"selected global ligand types ({len(ligand_types)}): "
        + " ".join(ligand_types)
    )

    job_dirs = []

    for src_dir in sorted(input_root.iterdir()):
        if not src_dir.is_dir():
            continue

        pdbid = src_dir.name

        lig = src_dir / f"{pdbid}_lig.pdbqt"
        pro0 = src_dir / f"{pdbid}_pro_0.pdbqt"
        pro1 = src_dir / f"{pdbid}_pro_1.pdbqt"

        if not lig.exists():
            print(f"[skip] {pdbid}: missing ligand")
            continue
        if not pro0.exists():
            print(f"[skip] {pdbid}: missing pro_0")
            continue
        if not pro1.exists():
            print(f"[skip] {pdbid}: missing pro_1")
            continue

        try:
            d0 = prepare_job(src_dir, output_root, pdbid, 0, param_file, ligand_types)
            d1 = prepare_job(src_dir, output_root, pdbid, 1, param_file, ligand_types)

            job_dirs += [d0, d1]

            print(f"[ok] {pdbid}")

        except Exception as e:
            print(f"[fail] {pdbid}: {e}")

    write_run_all(output_root, job_dirs)

    print(f"prepared {len(job_dirs)} jobs")


if __name__ == "__main__":
    main()

