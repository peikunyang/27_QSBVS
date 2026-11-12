import re
import sys
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import rdMolTransforms
from rdkit.Geometry import Point3D

pt = Chem.GetPeriodicTable()


def find_one(folder, pattern):
    files = sorted(folder.glob(pattern))
    if len(files) == 0:
        raise FileNotFoundError(f"missing: {folder}/{pattern}")
    if len(files) > 1:
        raise RuntimeError(f"multiple files: {folder}/{pattern}")
    return files[0]


def load_one_sdf(sdf_path):
    suppl = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
    mol = None

    for x in suppl:
        if x is not None:
            mol = x
            break

    if mol is None:
        raise ValueError(f"cannot read sdf: {sdf_path}")

    if mol.GetNumConformers() == 0:
        raise ValueError(f"sdf has no conformer: {sdf_path}")

    return mol


def infer_element(atom_name, element_field):
    element_field = element_field.strip()

    if element_field:
        sym = element_field.title()
        if pt.GetAtomicNumber(sym) > 0:
            return sym

    s = re.sub(r"[^A-Za-z]", "", atom_name)

    if not s:
        return None

    if len(s) >= 2:
        sym2 = s[:2].title()
        if pt.GetAtomicNumber(sym2) > 0:
            return sym2

    sym1 = s[:1].title()
    if pt.GetAtomicNumber(sym1) > 0:
        return sym1

    return None


def is_pseudo_atom(atom_name, symbol):
    a = atom_name.strip().upper()
    s = (symbol or "").strip().upper()

    if a.startswith("LP") or a.startswith("EP"):
        return True

    if s in {"LP", "EP", "DU", "DUM"}:
        return True

    return False


def read_pdb_atoms_skip_lp(pdb_path):
    atoms = []

    with open(pdb_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue

            atom_name = line[12:16].strip()
            element_field = line[76:78] if len(line) >= 78 else ""
            symbol = infer_element(atom_name, element_field)

            if is_pseudo_atom(atom_name, symbol):
                continue

            if symbol is None:
                raise ValueError(f"cannot infer element in {pdb_path}: {line.rstrip()}")

            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except ValueError:
                raise ValueError(f"bad coordinate in {pdb_path}: {line.rstrip()}")

            atoms.append({
                "symbol": symbol,
                "atom_name": atom_name,
                "x": x,
                "y": y,
                "z": z,
            })

    if not atoms:
        raise ValueError(f"no atoms found in {pdb_path}")

    return atoms


def build_mol_with_pdb_coords(template_mol, pdb_atoms):
    tpl_symbols = [a.GetSymbol() for a in template_mol.GetAtoms()]
    pdb_symbols = [a["symbol"] for a in pdb_atoms]

    if len(tpl_symbols) != len(pdb_symbols):
        raise ValueError(
            f"atom count mismatch: template={len(tpl_symbols)} pdb={len(pdb_symbols)}"
        )

    for i, (s1, s2) in enumerate(zip(tpl_symbols, pdb_symbols), start=1):
        if s1 != s2:
            raise ValueError(
                f"element order mismatch at atom {i}: template={s1} pdb={s2}"
            )

    mol = Chem.Mol(template_mol)
    mol.RemoveAllConformers()

    conf = Chem.Conformer(mol.GetNumAtoms())

    for i, a in enumerate(pdb_atoms):
        conf.SetAtomPosition(i, Point3D(a["x"], a["y"], a["z"]))

    mol.AddConformer(conf, assignId=True)
    return mol


def bond_length_limits(atom1, atom2, bond):
    z1 = atom1.GetAtomicNum()
    z2 = atom2.GetAtomicNum()

    r1 = pt.GetRcovalent(z1)
    r2 = pt.GetRcovalent(z2)

    base = r1 + r2

    min_len = max(0.55, 0.65 * base)
    max_len = 1.30 * base + 0.20

    if bond.GetIsAromatic():
        max_len -= 0.05
    else:
        order = bond.GetBondTypeAsDouble()
        if order >= 3:
            max_len -= 0.15
        elif order >= 2:
            max_len -= 0.10

    if atom1.GetSymbol() == "S" and atom2.GetSymbol() == "S":
        max_len = max(max_len, 2.25)

    return min_len, max_len


def check_bond_lengths(mol):
    conf = mol.GetConformer()

    rows = []
    bad_rows = []

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        ai = bond.GetBeginAtom()
        aj = bond.GetEndAtom()

        d = rdMolTransforms.GetBondLength(conf, i, j)
        min_len, max_len = bond_length_limits(ai, aj, bond)

        row = {
            "i": i + 1,
            "j": j + 1,
            "sym_i": ai.GetSymbol(),
            "sym_j": aj.GetSymbol(),
            "bond_type": str(bond.GetBondType()),
            "length": d,
            "min_len": min_len,
            "max_len": max_len,
            "ok": min_len <= d <= max_len,
        }

        rows.append(row)

        if not row["ok"]:
            bad_rows.append(row)

    return rows, bad_rows


def write_check_report(report_path, rows):
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("atom_i\tatom_j\tsym_i\tsym_j\tbond_type\tlength\tmin_len\tmax_len\tok\n")

        for r in rows:
            f.write(
                f"{r['i']}\t{r['j']}\t"
                f"{r['sym_i']}\t{r['sym_j']}\t"
                f"{r['bond_type']}\t"
                f"{r['length']:.4f}\t"
                f"{r['min_len']:.4f}\t"
                f"{r['max_len']:.4f}\t"
                f"{int(r['ok'])}\n"
            )


def convert_one(template_sdf, lig_pdb, out_sdf, report_path):
    template_mol = load_one_sdf(template_sdf)
    pdb_atoms = read_pdb_atoms_skip_lp(lig_pdb)

    new_mol = build_mol_with_pdb_coords(template_mol, pdb_atoms)

    rows, bad_rows = check_bond_lengths(new_mol)
    write_check_report(report_path, rows)

    if bad_rows:
        worst = max(
            bad_rows,
            key=lambda r: max(
                r["length"] - r["max_len"],
                r["min_len"] - r["length"]
            )
        )

        raise ValueError(
            f"bond length check failed, worst: "
            f"{worst['i']}-{worst['j']} "
            f"{worst['sym_i']}-{worst['sym_j']} "
            f"length={worst['length']:.4f}, "
            f"allowed=[{worst['min_len']:.4f}, {worst['max_len']:.4f}]"
        )

    writer = Chem.SDWriter(str(out_sdf))
    writer.write(new_mol)
    writer.close()


def main():
    if len(sys.argv) != 3:
        print(f"Usage: python {Path(sys.argv[0]).name} INPUT_DIR OUTPUT_DIR")
        sys.exit(1)

    input_root = Path(sys.argv[1])
    output_root = Path(sys.argv[2])

    if not input_root.is_dir():
        print(f"[error] input dir not found: {input_root}")
        sys.exit(1)

    output_root.mkdir(parents=True, exist_ok=True)

    summary_path = output_root / "summary.tsv"

    subdirs = sorted([x for x in input_root.iterdir() if x.is_dir()])

    with open(summary_path, "w", encoding="utf-8") as summary:
        summary.write("folder\tligand\tstatus\tmessage\n")

        for d in subdirs:
            dst_dir = output_root / d.name
            dst_dir.mkdir(parents=True, exist_ok=True)

            try:
                template_sdf = find_one(d, "*.sdf")
                lig_pdb = d / f"{d.name}_lig.pdb"

                if not lig_pdb.exists():
                    raise FileNotFoundError(f"missing: {lig_pdb.name}")

                out_sdf = dst_dir / f"{d.name}_lig.sdf"
                report = dst_dir / f"{d.name}_lig_bond_check.tsv"

                convert_one(template_sdf, lig_pdb, out_sdf, report)

                print(f"[ok] {d.name} lig")
                summary.write(f"{d.name}\tlig\tOK\t-\n")

            except Exception as e:
                msg = str(e).replace("\t", " ")
                print(f"[fail] {d.name}: {e}")
                summary.write(f"{d.name}\tlig\tFAIL\t{msg}\n")

    print("Done")


if __name__ == "__main__":
    main()

