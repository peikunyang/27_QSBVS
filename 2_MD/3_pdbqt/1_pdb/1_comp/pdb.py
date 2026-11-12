import argparse
import shutil
from pathlib import Path

WATER_ION_RESNAMES = {
    "HOH", "WAT", "TIP3", "TIP3P", "SOL",
    "NA", "SOD", "K", "POT", "CL", "CLA", "CAL", "CA", "MG"
}

def get_resname(pdb_line):
    return pdb_line[17:20].strip().upper()

def find_ligand_dir(complex_dir):
    candidates = []
    for x in complex_dir.iterdir():
        if x.is_dir():
            sdf_files = list(x.glob("*.sdf"))
            if sdf_files:
                candidates.append(x)
    return candidates

def split_pdb(pdb_path, ligand_resname, protein_out, ligand_out):
    n_protein = 0
    n_ligand = 0

    with open(pdb_path, "r", encoding="utf-8", errors="ignore") as fin, \
         open(protein_out, "w", encoding="utf-8") as fpro, \
         open(ligand_out, "w", encoding="utf-8") as फ्लig:

        for line in fin:
            if not line.startswith(("ATOM  ", "HETATM")):
                continue

            resname = get_resname(line)

            if resname in WATER_ION_RESNAMES:
                continue

            if resname == ligand_resname:
                फ्लig.write(line)
                n_ligand += 1
            else:
                fpro.write(line)
                n_protein += 1

        fpro.write("END\n")
        फ्लig.write("END\n")

    return n_protein, n_ligand

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir")
    parser.add_argument("output_dir")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for complex_dir in sorted(input_dir.iterdir()):
        if not complex_dir.is_dir():
            continue

        pdb_path = complex_dir / "step6_min.pdb"
        if not pdb_path.is_file():
            print(f"[skip] {complex_dir.name}: no step6_min.pdb")
            continue

        ligand_dirs = find_ligand_dir(complex_dir)

        if len(ligand_dirs) != 1:
            print(f"[skip] {complex_dir.name}: ligand dir count = {len(ligand_dirs)}")
            continue

        ligand_dir = ligand_dirs[0]
        ligand_resname = ligand_dir.name.upper()

        protein_out = output_dir / f"{complex_dir.name}_pro.pdb"
        ligand_out = output_dir / f"{complex_dir.name}_lig.pdb"

        n_protein, n_ligand = split_pdb(
            pdb_path=pdb_path,
            ligand_resname=ligand_resname,
            protein_out=protein_out,
            ligand_out=ligand_out
        )

        sdf_files = sorted(ligand_dir.glob("*.sdf"))
        for sdf in sdf_files:
            sdf_out = output_dir / f"{complex_dir.name}_{sdf.name}"
            shutil.copy2(sdf, sdf_out)

        print(
            f"[ok] {complex_dir.name}: ligand={ligand_resname}, "
            f"protein_atoms={n_protein}, ligand_atoms={n_ligand}, sdf_copied={len(sdf_files)}"
        )

if __name__ == "__main__":
    main()

