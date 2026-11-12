import argparse
import math
from pathlib import Path

GRID = 0.375
GRID_MIN = -16.0 * GRID
N = 32


def parse_pdbqt_atoms(pdbqt_file):
    atoms = []

    with open(pdbqt_file, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                q = float(line[70:76])
                atoms.append((x, y, z, q))

    if not atoms:
        raise ValueError(f"no ATOM/HETATM found: {pdbqt_file}")

    return atoms


def add_if_valid(grid, ix, iy, iz, value):
    if 0 <= ix < N and 0 <= iy < N and 0 <= iz < N:
        grid[iz][iy][ix] += value
        return True
    return False


def build_charge_grid(atoms):
    grid = [[[0.0 for _ in range(N)] for _ in range(N)] for _ in range(N)]

    kept = 0
    skipped = 0

    for x, y, z, q in atoms:
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
        # The charge q itself may be positive or negative, but interpolation
        # weights must never become negative because of index calculation.
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
            touched |= add_if_valid(grid, ix, iy, iz, q * w)

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


def get_output_name(stem):
    # Support:
    #   xxxx_lig.pdbqt       -> xxxx
    #   xxxx_ligand.pdbqt    -> xxxx
    #   000_xxxx.pdbqt       -> 000_xxxx      (old ranked format)
    #   0001_xxxx.pdbqt      -> 0001_xxxx     (current ranked format)
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
    atoms = parse_pdbqt_atoms(infile)
    grid, kept, skipped = build_charge_grid(atoms)
    values = flatten_grid(grid)

    pdbid = get_output_name(infile.stem)

    # Keep the directory structure below input_root, but avoid duplicating
    # the PDBID when the input is already inside a folder with that name.
    # Examples:
    #   input_root/1ecq/1ecq_lig.pdbqt
    #       -> output_root/1ecq/e
    #   input_root/3_0007/1gu3_ligand.pdbqt
    #       -> output_root/3_0007/1gu3/e
    rel_parent = infile.relative_to(input_root).parent

    if rel_parent.name == pdbid:
        outdir = output_root / rel_parent
    else:
        outdir = output_root / rel_parent / pdbid

    outdir.mkdir(parents=True, exist_ok=True)

    outfile = outdir / "e"

    with open(outfile, "w", encoding="utf-8") as f:
        for i in range(0, len(values), 32):
            chunk = values[i:i + 32]
            f.write(" ".join(f"{v:9.6f}" for v in chunk) + "\n")

    return kept, skipped, outfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_root", required=True, help="input root directory")
    parser.add_argument("-o", "--output_root", required=True, help="output root directory")
    args = parser.parse_args()

    input_root = Path(args.input_root).resolve()
    output_root = Path(args.output_root).resolve()

    output_root.mkdir(parents=True, exist_ok=True)

    # Support old ligand names plus both ranked formats:
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
            "000_xxxx.pdbqt, or 0001_xxxx.pdbqt files found."
        )
        return

    for infile in files:
        try:
            kept, skipped, outfile = process_one_file(infile, input_root, output_root)
            rel = infile.relative_to(input_root)
            print(f"[OK] {rel} -> {outfile}  kept={kept} skipped={skipped}")

        except Exception as e:
            print(f"[FAIL] {infile}  {e}")

    print("Done")


if __name__ == "__main__":
    main()
