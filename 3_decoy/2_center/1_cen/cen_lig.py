import argparse
from pathlib import Path


INPUT_DIR = Path("../../1_type_less_7/lig")


def read_coords_and_lines(pdbqt_path):
    lines = []
    coords = []

    with open(pdbqt_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            lines.append(line)

            if line.startswith(("ATOM", "HETATM")):
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                coords.append((x, y, z))

    if not coords:
        raise ValueError(f"No ATOM/HETATM found in {pdbqt_path}")

    return lines, coords


def box_center(coords):
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    zs = [c[2] for c in coords]

    cx = (min(xs) + max(xs)) / 2.0
    cy = (min(ys) + max(ys)) / 2.0
    cz = (min(zs) + max(zs)) / 2.0

    return cx, cy, cz


def shift_pdbqt_lines(lines, dx, dy, dz):
    new_lines = []

    for line in lines:
        if line.startswith(("ATOM", "HETATM")):
            x = float(line[30:38]) + dx
            y = float(line[38:46]) + dy
            z = float(line[46:54]) + dz

            new_line = f"{line[:30]}{x:8.3f}{y:8.3f}{z:8.3f}{line[54:]}"
            new_lines.append(new_line)
        else:
            new_lines.append(line)

    return new_lines


def process_one_ligand(src, output_dir):
    try:
        lines, coords = read_coords_and_lines(src)

        cx, cy, cz = box_center(coords)

        dx = -cx
        dy = -cy
        dz = -cz

        shifted = shift_pdbqt_lines(lines, dx, dy, dz)

        dst = output_dir / src.name

        with open(dst, "w", encoding="utf-8") as f:
            f.writelines(shifted)

        print(
            f"[done] {src.name}  "
            f"box_center = ({cx:.3f}, {cy:.3f}, {cz:.3f})  "
            f"shift = ({dx:.3f}, {dy:.3f}, {dz:.3f})"
        )

        return True

    except Exception as e:
        print(f"[skip] {src.name}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Shift each ligand PDBQT independently so its bounding-box "
            "center becomes (0, 0, 0)."
        )
    )
    parser.add_argument("output_dir", help="output folder")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    if not INPUT_DIR.is_dir():
        print(f"[error] input folder not found: {INPUT_DIR}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # New ranked filenames:
    # 0001_3isw.pdbqt ... 1023_xxxx.pdbqt
    ligands = sorted(INPUT_DIR.glob("*.pdbqt"))

    if not ligands:
        print(f"[error] no *.pdbqt files found in {INPUT_DIR}")
        return

    done = 0
    skipped = 0

    for src in ligands:
        if process_one_ligand(src, output_dir):
            done += 1
        else:
            skipped += 1

    print()
    print("Finished.")
    print(f"Processed: {done}")
    print(f"Skipped  : {skipped}")
    print(f"Output   : {output_dir}")


if __name__ == "__main__":
    main()
