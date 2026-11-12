import argparse
from pathlib import Path

SHIFT = 0.375 / 2.0


def parse_xyz(line):
    x = float(line[30:38])
    y = float(line[38:46])
    z = float(line[46:54])
    return x, y, z


def format_xyz(line, x, y, z):
    return f"{line[:30]}{x:8.3f}{y:8.3f}{z:8.3f}{line[54:]}"


def is_atom_line(line):
    return line.startswith("ATOM") or line.startswith("HETATM")


def process_one_file(infile, outfile):
    with open(infile, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    new_lines = []

    for line in lines:
        if is_atom_line(line):
            x, y, z = parse_xyz(line)
            x -= SHIFT
            y -= SHIFT
            z -= SHIFT
            new_lines.append(format_xyz(line, x, y, z))
        else:
            new_lines.append(line)

    outfile.parent.mkdir(parents=True, exist_ok=True)
    with open(outfile, "w", encoding="utf-8") as f:
        f.writelines(new_lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_root", required=True, help="input root directory")
    parser.add_argument("-o", "--output_root", required=True, help="output root directory")
    args = parser.parse_args()

    input_root = Path(args.input_root).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    files = sorted(input_root.rglob("*.pdbqt"))

    if not files:
        print("No .pdbqt files found.")
        return

    for infile in files:
        rel = infile.relative_to(input_root)
        outfile = output_root / rel
        try:
            process_one_file(infile, outfile)
            print(f"[OK] {rel}")
        except Exception as e:
            print(f"[FAIL] {rel}  {e}")


if __name__ == "__main__":
    main()

