import sys
import shutil
from pathlib import Path

import numpy as np


def read_pdb_atoms(pdb_path):
    lines = []
    atom_indices = []
    coords = []

    with open(pdb_path, "r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            lines.append(line)
            if line.startswith("ATOM") or line.startswith("HETATM"):
                try:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                except ValueError:
                    continue
                atom_indices.append(i)
                coords.append([x, y, z])

    coords = np.array(coords, dtype=float)

    if len(atom_indices) == 0:
        raise ValueError(f"No ATOM/HETATM coordinates found in {pdb_path}")

    return lines, atom_indices, coords


def write_pdb_with_new_coords(lines, atom_indices, new_coords, out_path):
    new_lines = list(lines)

    for idx, (x, y, z) in zip(atom_indices, new_coords):
        raw = lines[idx].rstrip("\n")
        if len(raw) < 54:
            raw = raw.ljust(54)
        new_line = f"{raw[:30]}{x:8.3f}{y:8.3f}{z:8.3f}{raw[54:]}\n"
        new_lines[idx] = new_line

    with open(out_path, "w", encoding="utf-8") as f:
        f.writelines(new_lines)


def kabsch_align(ref_coords, mob_coords):
    if ref_coords.shape != mob_coords.shape:
        raise ValueError(
            f"Atom count mismatch: ref={ref_coords.shape}, mob={mob_coords.shape}"
        )

    ref_centroid = ref_coords.mean(axis=0)
    mob_centroid = mob_coords.mean(axis=0)

    ref_centered = ref_coords - ref_centroid
    mob_centered = mob_coords - mob_centroid

    h = mob_centered.T @ ref_centered
    u, s, vt = np.linalg.svd(h)
    r = u @ vt

    if np.linalg.det(r) < 0:
        vt[-1, :] *= -1
        r = u @ vt

    aligned = mob_centered @ r + ref_centroid
    return aligned


def calc_rmsd(a, b):
    diff = a - b
    return np.sqrt(np.mean(np.sum(diff * diff, axis=1)))


def find_one(folder, pattern):
    files = sorted(folder.glob(pattern))
    if len(files) == 0:
        raise FileNotFoundError(f"Missing file: {folder}/{pattern}")
    if len(files) > 1:
        raise RuntimeError(f"Multiple files found for {folder}/{pattern}: {files}")
    return files[0]


def superimpose_pdb(ref_pdb, mob_pdb, out_pdb):
    ref_lines, ref_idx, ref_coords = read_pdb_atoms(ref_pdb)
    mob_lines, mob_idx, mob_coords = read_pdb_atoms(mob_pdb)

    if len(ref_idx) != len(mob_idx):
        raise ValueError(
            f"Atom number mismatch:\n  {ref_pdb}: {len(ref_idx)}\n  {mob_pdb}: {len(mob_idx)}"
        )

    aligned_coords = kabsch_align(ref_coords, mob_coords)
    rmsd = calc_rmsd(ref_coords, aligned_coords)

    write_pdb_with_new_coords(mob_lines, mob_idx, aligned_coords, out_pdb)
    return rmsd


def main():
    if len(sys.argv) != 3:
        print(f"Usage: python {Path(sys.argv[0]).name} INPUT_DIR OUTPUT_DIR")
        sys.exit(1)

    input_root = Path(sys.argv[1])
    output_root = Path(sys.argv[2])

    if not input_root.is_dir():
        print(f"[error] input dir not found: {input_root}")
        sys.exit(1)

    output_root.mkdir(parents=True, exist_ok=True)

    subdirs = sorted([x for x in input_root.iterdir() if x.is_dir()])

    for src_dir in subdirs:
        try:
            pro0 = find_one(src_dir, "*_pro_0.pdb")
            pro1 = find_one(src_dir, "*_pro_1.pdb")
            lig = find_one(src_dir, "*_lig.pdb")
            sdf_files = sorted(src_dir.glob("*.sdf"))

            dst_dir = output_root / src_dir.name
            dst_dir.mkdir(parents=True, exist_ok=True)

            # copy reference protein & ligand
            shutil.copy2(pro0, dst_dir / pro0.name)
            shutil.copy2(lig, dst_dir / lig.name)

            # copy sdf
            for sdf in sdf_files:
                shutil.copy2(sdf, dst_dir / sdf.name)

            # align protein
            pro_rmsd = superimpose_pdb(pro0, pro1, dst_dir / pro1.name)

            print(
                f"[ok] {src_dir.name}  "
                f"pro_RMSD={pro_rmsd:.6f}  "
                f"sdf={len(sdf_files)}"
            )

        except Exception as e:
            print(f"[skip] {src_dir.name}: {e}")

    print("Done")


if __name__ == "__main__":
    main()

