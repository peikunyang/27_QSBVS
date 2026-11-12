import argparse
from pathlib import Path

def extract_lig_fields(line):
    if not line.startswith("HETATM"):
        return None

    s = line.rstrip("\n").ljust(80)

    atom_name = s[12:16].strip()   # CL1
    res_name  = s[17:20].strip()   # DCE
    element   = s[76:78].strip()   # Cl

    return (atom_name, res_name, element)

def read_lig_fields(path):
    fields = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            x = extract_lig_fields(line)
            if x is not None:
                fields.append(x)
    return fields

def compare_lig_pdb(ref_path, tgt_path):
    ref_fields = read_lig_fields(ref_path)
    tgt_fields = read_lig_fields(tgt_path)

    n1 = len(ref_fields)
    n2 = len(tgt_fields)
    n = min(n1, n2)

    for i in range(n):
        if ref_fields[i] != tgt_fields[i]:
            return {
                "same": False,
                "reason": f"first difference at HETATM #{i+1}",
                "idx": i + 1,
                "ref": ref_fields[i],
                "tgt": tgt_fields[i],
            }

    if n1 != n2:
        return {
            "same": False,
            "reason": f"HETATM count differs: ref={n1}, target={n2}",
            "idx": None,
            "ref": None,
            "tgt": None,
        }

    return {
        "same": True,
        "reason": "all HETATM atom_name/res_name/element are identical",
        "idx": None,
        "ref": None,
        "tgt": None,
    }

parser = argparse.ArgumentParser()
parser.add_argument("--ref_dir", required=True)
parser.add_argument("--target_dir", required=True)
parser.add_argument("--pattern", default="*_lig.pdb")
parser.add_argument("--report", default="compare_lig_report.txt")
args = parser.parse_args()

ref_dir = Path(args.ref_dir)
target_dir = Path(args.target_dir)

ref_files = sorted(ref_dir.glob(args.pattern))
target_names = {p.name for p in target_dir.glob(args.pattern)}

same_count = 0
diff_count = 0
missing_count = 0

out = []
out.append(f"REF_DIR    : {ref_dir.resolve()}")
out.append(f"TARGET_DIR : {target_dir.resolve()}")
out.append(f"PATTERN    : {args.pattern}")
out.append("")

for ref_path in ref_files:
    tgt_path = target_dir / ref_path.name

    if not tgt_path.exists():
        missing_count += 1
        out.append(f"[MISSING] {ref_path.name}")
        continue

    result = compare_lig_pdb(ref_path, tgt_path)

    if result["same"]:
        same_count += 1
        out.append(f"[SAME] {ref_path.name}")
    else:
        diff_count += 1
        out.append(f"[DIFF] {ref_path.name} | {result['reason']}")
        if result["idx"] is not None:
            out.append(f"  REF : {result['ref']}")
            out.append(f"  TGT : {result['tgt']}")
            out.append("")

extra_in_target = sorted(target_names - {p.name for p in ref_files})
if extra_in_target:
    out.append("")
    out.append("[EXTRA FILES ONLY IN TARGET]")
    for name in extra_in_target:
        out.append(name)

out.append("")
out.append("[SUMMARY]")
out.append(f"total_ref_files : {len(ref_files)}")
out.append(f"same            : {same_count}")
out.append(f"diff            : {diff_count}")
out.append(f"missing         : {missing_count}")
out.append(f"extra_in_target : {len(extra_in_target)}")

report_path = Path(args.report)
with open(report_path, "w", encoding="utf-8") as f:
    f.write("\n".join(out) + "\n")

print("\n".join(out))
print(f"\nReport written to: {report_path.resolve()}")

