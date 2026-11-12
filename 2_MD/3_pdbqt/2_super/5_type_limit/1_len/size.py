import sys
import math
from pathlib import Path

import numpy as np

GRID_SPACING = 0.375


def read_pdbqt_coords(file_path):
    coords = []

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                try:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                except ValueError:
                    tail = line[30:].split()
                    if len(tail) < 3:
                        continue
                    x, y, z = map(float, tail[:3])

                coords.append([x, y, z])

    if not coords:
        raise ValueError(f"No ATOM/HETATM coordinates found in {file_path}")

    return np.array(coords, dtype=float)


def count_atom_types(file_path):
    atom_types = set()

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                parts = line.split()
                if parts:
                    atom_types.add(parts[-1])

    return len(atom_types)


def grid_index(value):
    if value >= 0:
        return math.ceil(value / GRID_SPACING)
    return math.floor(value / GRID_SPACING)


def calc_xyz_grid_indices(file_path):
    coords = read_pdbqt_coords(file_path)

    xmin, ymin, zmin = coords.min(axis=0)
    xmax, ymax, zmax = coords.max(axis=0)

    ix_min = grid_index(xmin)
    ix_max = grid_index(xmax)

    iy_min = grid_index(ymin)
    iy_max = grid_index(ymax)

    iz_min = grid_index(zmin)
    iz_max = grid_index(zmax)

    return ix_min, ix_max, iy_min, iy_max, iz_min, iz_max


def process_pdbid_folders(input_root, output_file):
    root = Path(input_root)

    if not root.is_dir():
        raise FileNotFoundError(f"Input folder not found: {input_root}")

    pdbid_dirs = sorted([p for p in root.iterdir() if p.is_dir()])
    lines = []

    for pdbid_dir in pdbid_dirs:
        pdbid = pdbid_dir.name
        lig_file = pdbid_dir / f"{pdbid}_lig.pdbqt"

        if not lig_file.is_file():
            print(f"[SKIP] missing ligand: {lig_file}")
            continue

        try:
            ix_min, ix_max, iy_min, iy_max, iz_min, iz_max = calc_xyz_grid_indices(lig_file)
            num_atom_types = count_atom_types(lig_file)
        except Exception as e:
            print(f"[SKIP] {lig_file}: {e}")
            continue

        lines.append(
            f"{pdbid}\t"
            f"{ix_min:4d}\t{ix_max:4d}\t"
            f"{iy_min:4d}\t{iy_max:4d}\t"
            f"{iz_min:4d}\t{iz_max:4d}\t"
            f"{num_atom_types:d}"
        )

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if len(sys.argv) != 3:
    print("Usage: python size.py <input_pdbqt_root_folder> <output_file>")
    sys.exit(1)

input_root = sys.argv[1]
output_file = sys.argv[2]

process_pdbid_folders(input_root, output_file)

print(f"Done. Output written to {output_file}")

