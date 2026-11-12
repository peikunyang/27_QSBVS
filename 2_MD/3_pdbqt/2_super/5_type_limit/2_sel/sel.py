import argparse
import shutil
from pathlib import Path


LOWER = -12
UPPER = 11
MAX_TYPE = 3


def read_dim_file(dim_file):
    selected = []

    with open(dim_file, "r", encoding="utf-8", errors="ignore") as f:
        for line_no, line in enumerate(f, start=1):
            parts = line.split()

            if not parts:
                continue

            if len(parts) < 8:
                print(f"[skip] line {line_no}: not enough columns")
                continue

            pdbid = parts[0]

            try:
                values = [int(x) for x in parts[1:7]]
                num_type = int(parts[7])
            except ValueError:
                print(f"[skip] line {line_no}: invalid number")
                continue

            ok_range = all(LOWER <= v <= UPPER for v in values)
            ok_type = num_type <= MAX_TYPE

            if ok_range and ok_type:
                selected.append(pdbid)

    return selected


def copy_selected(selected, src_root, dst_root):
    src_root = Path(src_root)
    dst_root = Path(dst_root)

    dst_root.mkdir(parents=True, exist_ok=True)

    copied = 0
    skipped = 0

    for pdbid in selected:
        src_path = src_root / pdbid
        dst_path = dst_root / pdbid

        if not src_path.exists():
            print(f"[skip] missing source: {src_path}")
            skipped += 1
            continue

        if dst_path.exists():
            print(f"[skip] exists: {dst_path}")
            skipped += 1
            continue

        if src_path.is_dir():
            shutil.copytree(src_path, dst_path)
        else:
            shutil.copy2(src_path, dst_path)

        print(f"[copy] {pdbid}")
        copied += 1

    print(f"Selected: {len(selected)}")
    print(f"Copied: {copied}")
    print(f"Skipped: {skipped}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dim_file")
    parser.add_argument("src_root")
    parser.add_argument("dst_root")
    args = parser.parse_args()

    dim_file = Path(args.dim_file)
    src_root = Path(args.src_root)
    dst_root = Path(args.dst_root)

    if not dim_file.is_file():
        print(f"[error] dim file not found: {dim_file}")
        return

    if not src_root.exists():
        print(f"[error] source root not found: {src_root}")
        return

    selected = read_dim_file(dim_file)
    copy_selected(selected, src_root, dst_root)


if __name__ == "__main__":
    main()

