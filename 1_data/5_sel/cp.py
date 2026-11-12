import argparse
import subprocess
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("list_file")
parser.add_argument("src_dir")
parser.add_argument("dst_dir")
args = parser.parse_args()

list_file = Path(args.list_file)
src_dir = Path(args.src_dir)
dst_dir = Path(args.dst_dir)

dst_dir.mkdir(parents=True, exist_ok=True)

pdbids = []
seen = set()

with open(list_file, "r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        parts = line.split()
        if not parts:
            continue
        pdbid = parts[0]
        if pdbid not in seen:
            seen.add(pdbid)
            pdbids.append(pdbid)

for pdbid in pdbids:
    src_path = src_dir / pdbid
    dst_path = dst_dir / pdbid

    if not src_path.is_dir():
        print(f"not found: {src_path}")
        continue

    if dst_path.exists():
        print(f"skip existing: {dst_path}")
        continue

    subprocess.run(["cp", "-r", str(src_path), str(dst_dir)], check=True)
    print(f"copied: {pdbid}")

print("Done")

