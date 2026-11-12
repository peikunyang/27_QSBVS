import argparse
from pathlib import Path


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


def write_lines(out_path, lines):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        f.writelines(lines)


def find_one(folder, pattern):
    files = sorted(folder.glob(pattern))

    if len(files) == 0:
        raise FileNotFoundError(f"missing: {folder}/{pattern}")

    if len(files) > 1:
        raise RuntimeError(f"multiple files: {folder}/{pattern}")

    return files[0]


def process_one_folder(in_dir, out_root):
    try:
        lig_path = find_one(in_dir, "*_lig.pdbqt")
        prefix = lig_path.name[:-len("_lig.pdbqt")]

        targets = [
            f"{prefix}_lig.pdbqt",
            f"{prefix}_pro_0.pdbqt",
            f"{prefix}_pro_1.pdbqt",
        ]

        missing = [name for name in targets if not (in_dir / name).exists()]
        if missing:
            print(f"[skip] {in_dir.name}: missing files -> {', '.join(missing)}")
            return

        lig_lines, lig_coords = read_coords_and_lines(lig_path)
        cx, cy, cz = box_center(lig_coords)

        dx = -cx
        dy = -cy
        dz = -cz

        for name in targets:
            src = in_dir / name
            dst = out_root / in_dir.name / name

            lines, _ = read_coords_and_lines(src)
            shifted = shift_pdbqt_lines(lines, dx, dy, dz)
            write_lines(dst, shifted)

        print(f"[done] {in_dir.name}  box_center = ({cx:.3f}, {cy:.3f}, {cz:.3f})  shift = ({dx:.3f}, {dy:.3f}, {dz:.3f})")

    except Exception as e:
        print(f"[skip] {in_dir.name}: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_root", help="input root folder")
    parser.add_argument("output_root", help="output root folder")
    args = parser.parse_args()

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)

    if not input_root.is_dir():
        print(f"[error] input root not found: {input_root}")
        return

    output_root.mkdir(parents=True, exist_ok=True)

    for sub in sorted(input_root.iterdir()):
        if sub.is_dir():
            process_one_folder(sub, output_root)

    print("Finished.")


if __name__ == "__main__":
    main()

