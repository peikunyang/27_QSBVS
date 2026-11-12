import sys
import subprocess
from pathlib import Path

MK_PREPARE = "/home/kun/anaconda3/envs/molgen/bin/mk_prepare_ligand.py"


def force_rigid_pdbqt(pdbqt_path):
    out_lines = []

    with open(pdbqt_path, "r", encoding="utf-8", errors="ignore") as fr:
        for line in fr:
            s = line.lstrip()

            if s.startswith("BRANCH") or s.startswith("ENDBRANCH"):
                continue
            if s.startswith("TORSDOF"):
                continue
            if s.startswith("REMARK") and "torsion" in s.lower():
                continue

            out_lines.append(line)

    out_lines.append("TORSDOF 0\n")

    tmp_path = str(pdbqt_path) + ".tmp"

    with open(tmp_path, "w", encoding="utf-8") as fw:
        fw.writelines(out_lines)

    Path(tmp_path).replace(pdbqt_path)


def convert_one(in_sdf, out_pdbqt):
    cmd = [
        MK_PREPARE,
        "-i", str(in_sdf),
        "-o", str(out_pdbqt),
        "--rigid_macrocycles",
    ]

    subprocess.run(cmd, check=True)
    force_rigid_pdbqt(out_pdbqt)


def main():
    if len(sys.argv) != 3:
        print(f"Usage: python {Path(sys.argv[0]).name} INPUT_DIR OUTPUT_DIR")
        sys.exit(1)

    input_root = Path(sys.argv[1])
    output_root = Path(sys.argv[2])

    if not input_root.is_dir():
        print(f"[error] input dir not found: {input_root}")
        sys.exit(1)

    if not Path(MK_PREPARE).is_file():
        print(f"[error] mk_prepare_ligand.py not found: {MK_PREPARE}")
        sys.exit(1)

    output_root.mkdir(parents=True, exist_ok=True)

    subdirs = sorted([x for x in input_root.iterdir() if x.is_dir()])

    for src_dir in subdirs:
        dst_dir = output_root / src_dir.name
        dst_dir.mkdir(parents=True, exist_ok=True)

        sdf_files = sorted(src_dir.glob("*_lig.sdf"))

        if not sdf_files:
            print(f"[skip] {src_dir.name}: no *_lig.sdf")
            continue

        for in_sdf in sdf_files:
            out_pdbqt = dst_dir / in_sdf.with_suffix(".pdbqt").name

            try:
                convert_one(in_sdf, out_pdbqt)
                print(f"[ok] {in_sdf} -> {out_pdbqt}")

            except Exception as e:
                print(f"[fail] {in_sdf}: {e}")

    print("Done")


if __name__ == "__main__":
    main()

