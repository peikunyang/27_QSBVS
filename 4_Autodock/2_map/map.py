import argparse
from pathlib import Path


def read_map_values(map_file):
    values = []

    with open(map_file, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue

            parts = s.split()
            if len(parts) != 1:
                continue

            try:
                values.append(float(parts[0]))
            except ValueError:
                continue

    return values


def reshape_33x33x33(values):
    cube = [[[0.0 for _ in range(33)] for _ in range(33)] for _ in range(33)]

    idx = 0
    for z in range(33):
        for y in range(33):
            for x in range(33):
                cube[z][y][x] = values[idx]
                idx += 1

    return cube


def crop_to_32x32x32(cube):
    cropped = []

    for z in range(32):
        for y in range(32):
            row = []
            for x in range(32):
                row.append(cube[z][y][x])
            cropped.append(row)

    return cropped


def get_output_name(map_file):
    parts = map_file.name.split(".")
    if len(parts) >= 3:
        return parts[-2]
    return map_file.stem


def clamp_value(v, limit):
    if limit is None:
        return v
    if v < -limit:
        return -limit
    if v > limit:
        return limit
    return v


def build_clamped_data(cropped_rows, limit):
    out_rows = []

    for row in cropped_rows:
        new_row = []
        for v in row:
            new_row.append(clamp_value(v, limit))
        out_rows.append(new_row)

    return out_rows


def write_rows(outfile, rows):
    with open(outfile, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(" ".join(f"{v:10.6f}" for v in row) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_root", required=True, help="input root directory")
    parser.add_argument("-o", "--output_root", required=True, help="output root directory")
    args = parser.parse_args()

    input_root = Path(args.input_root).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    bucket_limits = {
        "map": 1000,
    }

    map_files = []
    for p in input_root.rglob("*.map"):
        if p.name.endswith(".d.map"):
            continue
        map_files.append(p)

    map_files = sorted(map_files)

    if not map_files:
        print("No .map files found.")
        return

    for map_file in map_files:
        rel = map_file.relative_to(input_root)
        values = read_map_values(map_file)

        if len(values) != 33 * 33 * 33:
            print(f"[FAIL] {rel} value count = {len(values)}, expected 35937")
            continue

        cube = reshape_33x33x33(values)
        cropped_rows = crop_to_32x32x32(cube)
        outname = get_output_name(map_file)

        for bucket_name, limit in bucket_limits.items():
            outdir = output_root / bucket_name / rel.parent
            outdir.mkdir(parents=True, exist_ok=True)

            outfile = outdir / outname
            clamped_rows = build_clamped_data(cropped_rows, limit)
            write_rows(outfile, clamped_rows)

        print(f"[OK] {rel} -> 7 files written")


if __name__ == "__main__":
    main()

