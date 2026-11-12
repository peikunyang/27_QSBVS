import argparse
from pathlib import Path

def normalize_pdb_line(line):
    s = line.rstrip("\n")
    if s.startswith(("ATOM  ", "HETATM")):
        s = s.ljust(80)
        s = s[:30] + (" " * 24) + s[54:]
    return s

def read_lines(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        raw = [line.rstrip("\n") for line in f]
    norm = [normalize_pdb_line(line) for line in raw]
    return raw, norm

def compare_pdb(ref_path, tgt_path):
    raw1, norm1 = read_lines(ref_path)
    raw2, norm2 = read_lines(tgt_path)

    n1 = len(norm1)
    n2 = len(norm2)
    n = min(n1, n2)

    for i in range(n):
        if norm1[i] != norm2[i]:
            return {
                "same": False,
                "reason": f"first difference at line {i+1}",
                "line_no": i + 1,
                "ref_raw": raw1[i],
                "tgt_raw": raw2[i],
                "ref_norm": norm1[i],
                "tgt_norm": norm2[i],
            }

    if n1 != n2:
        return {
            "same": False,
            "reason": f"line count differs: ref={n1}, target={n2}",
            "line_no": None,
            "ref_raw": None,
            "tgt_raw": None,
            "ref_norm": None,
            "tgt_norm": None,
        }

    return {
        "same": True,
        "reason": "all identical except coordinates",
        "line_no": None,
        "ref_raw": None,
        "tgt_raw": None,
        "ref_norm": None,
        "tgt_norm": None,
    }

parser = argparse.ArgumentParser()
parser.add_argument("--ref_dir", default="../../2_pro/pdb")
parser.add_argument("--target_dir", default="../../1_comp/pdb")
parser.add_argument("--pattern", default="*_pro.pdb")
parser.add_argument("--report", default="compare_pdb_report.txt")
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
out.append("")

for ref_path in ref_files:
    tgt_path = target_dir / ref_path.name

    if not tgt_path.exists():
        missing_count += 1
        out.append(f"[MISSING] {ref_path.name}")
        continue

    result = compare_pdb(ref_path, tgt_path)

    if result["same"]:
        same_count += 1
        out.append(f"[SAME] {ref_path.name}")
    else:
        diff_count += 1
        out.append(f"[DIFF] {ref_path.name} | {result['reason']}")
        if result["line_no"] is not None:
            out.append(f"  REF_RAW : {result['ref_raw']}")
            out.append(f"  TGT_RAW : {result['tgt_raw']}")
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

