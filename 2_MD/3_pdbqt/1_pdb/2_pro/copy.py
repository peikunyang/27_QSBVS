import argparse
from pathlib import Path

WATER_ION_RESNAMES = {
    "HOH", "WAT", "TIP3", "TIP3P", "SOL",
    "NA", "SOD", "K", "POT", "CL", "CLA", "CAL", "CA", "MG"
}

def get_resname(pdb_line):
    return pdb_line[17:20].strip().upper()

def find_ligand_dirs(complex_dir):
    candidates = []
    for x in complex_dir.iterdir():
        if x.is_dir():
            sdf_files = list(x.glob("*.sdf"))
            if sdf_files:
                candidates.append(x)
    return candidates

def extract_protein(pdb_path, protein_out, ligand_resname=None):
    n_protein = 0

    with open(pdb_path, "r", encoding="utf-8", errors="ignore") as fin, \
         open(protein_out, "w", encoding="utf-8") as fpro:

        for line in fin:
            if not line.startswith(("ATOM  ", "HETATM")):
                continue

            resname = get_resname(line)

            if resname in WATER_ION_RESNAMES:
                continue

            if ligand_resname is not None and resname == ligand_resname:
                continue

            fpro.write(line)
            n_protein += 1

        fpro.write("END\n")

    return n_protein

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

        pdb_path = complex_dir / "step5.pdb"
        if not pdb_path.is_file():
            print(f"[skip] {complex_dir.name}: no step6_min.pdb")
            continue

        ligand_dirs = find_ligand_dirs(complex_dir)

        ligand_resname = None
        if len(ligand_dirs) == 1:
            ligand_resname = ligand_dirs[0].name.upper()
        elif len(ligand_dirs) > 1:
            print(f"[skip] {complex_dir.name}: ligand dir count = {len(ligand_dirs)}")
            continue

        protein_out = output_dir / f"{complex_dir.name}_pro.pdb"

        n_protein = extract_protein(
            pdb_path=pdb_path,
            protein_out=protein_out,
            ligand_resname=ligand_resname
        )

        if ligand_resname is None:
            print(f"[ok] {complex_dir.name}: no ligand, protein_atoms={n_protein}")
        else:
            print(f"[ok] {complex_dir.name}: ligand={ligand_resname}, protein_atoms={n_protein}")

if __name__ == "__main__":
    main()

