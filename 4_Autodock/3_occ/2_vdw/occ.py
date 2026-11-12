import argparse
import math
from collections import defaultdict
from pathlib import Path

GRID = 0.375
GRID_MIN = -16.0 * GRID
N = 32


def parse_pdbqt_atoms_by_type(pdbqt_file):
    atoms_by_type = defaultdict(list)

    with open(pdbqt_file, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])

                parts = line.split()
                atom_type = parts[-1].strip()

                atoms_by_type[atom_type].append((x, y, z))

    if not atoms_by_type:
        raise ValueError(f"no atom types found: {pdbqt_file}")

    return atoms_by_type


def add_if_valid(grid, ix, iy, iz, value):
    if 0 <= ix < N and 0 <= iy < N and 0 <= iz < N:
        grid[iz][iy][ix] += value
        return True
    return False


def build_occ_grid(coords):
    grid = [[[0.0 for _ in range(N)] for _ in range(N)] for _ in range(N)]

    kept = 0
    skipped = 0

    for x, y, z in coords:
        tx = (x - GRID_MIN) / GRID
        ty = (y - GRID_MIN) / GRID
        tz = (z - GRID_MIN) / GRID

        ix0 = math.floor(tx)
        iy0 = math.floor(ty)
        iz0 = math.floor(tz)

        fx = tx - ix0
        fy = ty - iy0
        fz = tz - iz0

        # With floor(), each interpolation fraction must be in [0, 1).
        # This guarantees all trilinear occupancy weights are nonnegative.
        if not (0.0 <= fx < 1.0 and 0.0 <= fy < 1.0 and 0.0 <= fz < 1.0):
            raise ArithmeticError(
                f"invalid interpolation fraction: fx={fx}, fy={fy}, fz={fz}"
            )

        ix1 = ix0 + 1
        iy1 = iy0 + 1
        iz1 = iz0 + 1

        weights = [
            (ix0, iy0, iz0, (1.0 - fx) * (1.0 - fy) * (1.0 - fz)),
            (ix1, iy0, iz0, fx * (1.0 - fy) * (1.0 - fz)),
            (ix0, iy1, iz0, (1.0 - fx) * fy * (1.0 - fz)),
            (ix1, iy1, iz0, fx * fy * (1.0 - fz)),
            (ix0, iy0, iz1, (1.0 - fx) * (1.0 - fy) * fz),
            (ix1, iy0, iz1, fx * (1.0 - fy) * fz),
            (ix0, iy1, iz1, (1.0 - fx) * fy * fz),
            (ix1, iy1, iz1, fx * fy * fz),
        ]

        touched = False

        for ix, iy, iz, w in weights:
            touched |= add_if_valid(grid, ix, iy, iz, w)

        if touched:
            kept += 1
        else:
            skipped += 1

    return grid, kept, skipped


def flatten_grid(grid):
    values = []

    for z in range(N):
        for y in range(N):
            for x in range(N):
                values.append(grid[z][y][x])

    return values


def get_pdbid_from_lig_name(stem):
    # Support:
    #   xxxx_lig.pdbqt        -> xxxx
    #   xxxx_ligand.pdbqt     -> xxxx
    #   000_xxxx.pdbqt        -> 000_xxxx      (old ranked format)
    #   0001_xxxx.pdbqt       -> 0001_xxxx     (current ranked format)
    if stem.endswith("_ligand"):
        return stem[:-len("_ligand")]

    if stem.endswith("_lig"):
        return stem[:-len("_lig")]

    parts = stem.split("_", 1)
    if (
        len(parts) == 2
        and len(parts[0]) in (3, 4)
        and parts[0].isdigit()
        and parts[1]
    ):
        return stem

    raise ValueError(f"unexpected filename: {stem}")


def process_one_file(infile, input_root, output_root):
    atoms_by_type = parse_pdbqt_atoms_by_type(infile)
    pdbid = get_pdbid_from_lig_name(infile.stem)

    # Keep the directory structure below input_root, but avoid duplicating
    # the PDBID when the input is already inside a folder with that name.
    # Examples:
    #   input_root/1ecq/1ecq_lig.pdbqt
    #       -> output_root/1ecq/<atom_type>
    #   input_root/3_0007/1gu3_ligand.pdbqt
    #       -> output_root/3_0007/1gu3/<atom_type>
    #   input_root/000_4oru.pdbqt
    #       -> output_root/000_4oru/<atom_type>
    rel_parent = infile.relative_to(input_root).parent

    if rel_parent.name == pdbid:
        outdir = output_root / rel_parent
    else:
        outdir = output_root / rel_parent / pdbid

    outdir.mkdir(parents=True, exist_ok=True)

    summary = []

    for atom_type, coords in sorted(atoms_by_type.items()):
        grid, kept, skipped = build_occ_grid(coords)
        values = flatten_grid(grid)

        outfile = outdir / atom_type

        with open(outfile, "w", encoding="utf-8") as f:
            for i in range(0, len(values), 32):
                chunk = values[i:i + 32]
                f.write(" ".join(f"{v:.6f}" for v in chunk) + "\n")

        summary.append((atom_type, kept, skipped, outfile))

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_root", required=True, help="input root directory")
    parser.add_argument("-o", "--output_root", required=True, help="output root directory")
    args = parser.parse_args()

    input_root = Path(args.input_root).resolve()
    output_root = Path(args.output_root).resolve()

    output_root.mkdir(parents=True, exist_ok=True)

    # Support old ligand names plus both ranked formats:
    #   xxxx_lig.pdbqt
    #   xxxx_ligand.pdbqt
    #   000_xxxx.pdbqt
    #   0001_xxxx.pdbqt ... 1023_xxxx.pdbqt
    files = sorted(
        set(input_root.rglob("*_lig.pdbqt")) |
        set(input_root.rglob("*_ligand.pdbqt")) |
        set(input_root.rglob("[0-9][0-9][0-9]_*.pdbqt")) |
        set(input_root.rglob("[0-9][0-9][0-9][0-9]_*.pdbqt"))
    )

    if not files:
        print(
            "No matching *_lig.pdbqt, *_ligand.pdbqt, "
            "NNN_*.pdbqt, or NNNN_*.pdbqt files found."
        )
        return

    for infile in files:
        try:
            summary = process_one_file(infile, input_root, output_root)
            rel = infile.relative_to(input_root)

            print(f"[OK] {rel}")

            for atom_type, kept, skipped, outfile in summary:
                print(
                    f"     {atom_type:>4s} -> {outfile}  "
                    f"kept={kept} skipped={skipped}"
                )

        except Exception as e:
            print(f"[FAIL] {infile}  {e}")

    print("Done")


if __name__ == "__main__":
    main()

