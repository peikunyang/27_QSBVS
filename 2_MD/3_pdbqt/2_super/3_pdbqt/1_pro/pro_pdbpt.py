import sys
import subprocess
from pathlib import Path

PYTHONSH = "/home/kun/package/MGLTools/mgltools_x86_64Linux2_1.5.7/bin/pythonsh"
PREPARE_RECEPTOR = "/home/kun/package/MGLTools/mgltools_x86_64Linux2_1.5.7/MGLToolsPckgs/AutoDockTools/Utilities24/prepare_receptor4.py"


def convert_one(in_pdb, out_pdbqt):
    cmd = [
        PYTHONSH,
        PREPARE_RECEPTOR,
        "-r", str(in_pdb),
        "-o", str(out_pdbqt),
        "-A", "None",
    ]
    subprocess.run(cmd, check=True)


def main():
    if len(sys.argv) != 3:
        print(f"Usage: python {Path(sys.argv[0]).name} INPUT_DIR OUTPUT_DIR")
        sys.exit(1)

    input_root = Path(sys.argv[1])
    output_root = Path(sys.argv[2])

    if not input_root.is_dir():
        print(f"[error] input dir not found: {input_root}")
        sys.exit(1)

    if not Path(PYTHONSH).is_file():
        print(f"[error] pythonsh not found: {PYTHONSH}")
        sys.exit(1)

    if not Path(PREPARE_RECEPTOR).is_file():
        print(f"[error] prepare_receptor4.py not found: {PREPARE_RECEPTOR}")
        sys.exit(1)

    output_root.mkdir(parents=True, exist_ok=True)

    subdirs = sorted([x for x in input_root.iterdir() if x.is_dir()])

    for src_dir in subdirs:
        dst_dir = output_root / src_dir.name
        dst_dir.mkdir(parents=True, exist_ok=True)

        pdb_files = sorted(src_dir.glob("*_pro_0.pdb")) + sorted(src_dir.glob("*_pro_1.pdb"))

        if not pdb_files:
            print(f"[skip] {src_dir.name}: no *_pro_0.pdb or *_pro_1.pdb")
            continue

        for in_pdb in pdb_files:
            out_pdbqt = dst_dir / in_pdb.with_suffix(".pdbqt").name
            try:
                convert_one(in_pdb, out_pdbqt)
                print(f"[ok] {in_pdb} -> {out_pdbqt}")
            except subprocess.CalledProcessError as e:
                print(f"[fail] {in_pdb}: {e}")

    print("Done")


if __name__ == "__main__":
    main()

