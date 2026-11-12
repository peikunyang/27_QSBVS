import sys
import shutil
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
    with open(tmp_path, "w", encoding="utf-8", newline="\n") as fw:
        fw.writelines(out_lines)

    Path(tmp_path).replace(pdbqt_path)


def is_rigid_pdbqt(pdbqt_path):
    tors_ok = False
    has_branch = False
    has_endbranch = False

    with open(pdbqt_path, "r", encoding="utf-8", errors="ignore") as fr:
        for line in fr:
            s = line.strip()

            if s.startswith("TORSDOF"):
                parts = s.split()
                if len(parts) >= 2 and parts[1] == "0":
                    tors_ok = True

            if s.startswith("BRANCH"):
                has_branch = True

            if s.startswith("ENDBRANCH"):
                has_endbranch = True

    return tors_ok and (not has_branch) and (not has_endbranch)


def find_ligand_file(pdb_dir):
    pdbid = pdb_dir.name

    sdf_file = pdb_dir / f"{pdbid}_ligand.sdf"
    mol2_file = pdb_dir / f"{pdbid}_ligand.mol2"

    if sdf_file.is_file():
        return sdf_file

    if mol2_file.is_file():
        return mol2_file

    return None


def convert_one_ligand(ligand_file, out_file):
    cmd = [
        MK_PREPARE,
        "-i", str(ligand_file),
        "-o", str(out_file),
    ]

    res = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    return res


def process_one_folder(pdb_dir, output_root):
    pdbid = pdb_dir.name
    ligand_file = find_ligand_file(pdb_dir)

    if ligand_file is None:
        print(f"SKIP folder: {pdbid} (no {pdbid}_ligand.sdf or {pdbid}_ligand.mol2)")
        return "skip", 0

    final_file = output_root / f"{pdbid}_ligand.pdbqt"
    tmp_file = output_root / f".{pdbid}_ligand.pdbqt.tmp"

    if tmp_file.exists():
        tmp_file.unlink()

    try:
        res = convert_one_ligand(ligand_file, tmp_file)

        if res.returncode != 0:
            print(f"FAIL folder: {pdbid}")
            print(f"  file: {ligand_file.name}")

            if res.stdout.strip():
                print("  STDOUT:")
                print(res.stdout)

            if res.stderr.strip():
                print("  STDERR:")
                print(res.stderr)

            raise RuntimeError("mk_prepare_ligand failed")

        if not tmp_file.is_file():
            print(f"FAIL folder: {pdbid}")
            print(f"  file: {ligand_file.name}")
            print("  ERROR: no output pdbqt")
            raise RuntimeError("no output pdbqt")

        force_rigid_pdbqt(tmp_file)

        if not is_rigid_pdbqt(tmp_file):
            print(f"FAIL folder: {pdbid}")
            print(f"  file: {ligand_file.name}")
            print("  ERROR: not rigid after force")
            raise RuntimeError("not rigid after force")

        if final_file.exists():
            final_file.unlink()

        tmp_file.rename(final_file)

        print(f"OK folder: {pdbid}")
        print(f"  input : {ligand_file.name}")
        print(f"  output: {final_file.name}")

        return "ok", 1

    except Exception as e:
        if tmp_file.exists():
            tmp_file.unlink()

        print(f"DROP folder: {pdbid}")
        print(f"ERROR: {e}")

        return "fail", 0


def main():
    if len(sys.argv) != 3:
        print("Usage: python pdbbind_ligand_to_pdbqt_meeko.py <pdbbind_root> <output_root>")
        print("Example:")
        print("  python pdbbind_ligand_to_pdbqt_meeko.py ../../2_PDBbind ./pdbqt_out")
        sys.exit(1)

    input_root = Path(sys.argv[1])
    output_root = Path(sys.argv[2])

    if not input_root.is_dir():
        print("Error: input_root is not a directory")
        sys.exit(1)

    if not Path(MK_PREPARE).is_file():
        print("Error: mk_prepare_ligand.py not found")
        print(MK_PREPARE)
        sys.exit(1)

    output_root.mkdir(parents=True, exist_ok=True)

    ok_folders = 0
    skip_folders = 0
    fail_folders = 0
    ok_files = 0

    for pdb_dir in sorted(input_root.iterdir()):
        if not pdb_dir.is_dir():
            continue

        status, n_ok = process_one_folder(pdb_dir, output_root)

        if status == "ok":
            ok_folders += 1
            ok_files += n_ok
        elif status == "skip":
            skip_folders += 1
        else:
            fail_folders += 1

    print("SUMMARY")
    print("OK folders  :", ok_folders)
    print("SKIP folders:", skip_folders)
    print("FAIL folders:", fail_folders)
    print("OK files    :", ok_files)


if __name__ == "__main__":
    main()

