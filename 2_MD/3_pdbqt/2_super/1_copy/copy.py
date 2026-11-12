import sys
import shutil
from pathlib import Path

if len(sys.argv) != 4:
    print(
        f"Usage: python {Path(sys.argv[0]).name} "
        "COMP_DIR PRO_DIR OUT_DIR"
    )
    sys.exit(1)

comp_dir = Path(sys.argv[1])
pro_dir = Path(sys.argv[2])
out_root = Path(sys.argv[3])

for d in [comp_dir, pro_dir]:
    if not d.is_dir():
        print(f"[error] not found: {d}")
        sys.exit(1)

out_root.mkdir(parents=True, exist_ok=True)

pdbids = sorted({
    f.name[:4]
    for f in pro_dir.glob("*.pdb")
    if len(f.name) >= 4
})

for pdbid in pdbids:
    dst_dir = out_root / pdbid
    dst_dir.mkdir(parents=True, exist_ok=True)

    src = comp_dir / f"{pdbid}_lig.pdb"
    dst = dst_dir / f"{pdbid}_lig.pdb"
    if src.exists():
        shutil.copy2(src, dst)
    else:
        print(f"[missing] {src}")

    src = comp_dir / f"{pdbid}_pro.pdb"
    dst = dst_dir / f"{pdbid}_pro_0.pdb"
    if src.exists():
        shutil.copy2(src, dst)
    else:
        print(f"[missing] {src}")

    src = pro_dir / f"{pdbid}_pro.pdb"
    dst = dst_dir / f"{pdbid}_pro_1.pdb"
    if src.exists():
        shutil.copy2(src, dst)
    else:
        print(f"[missing] {src}")

    sdf_files = list(comp_dir.glob(f"{pdbid}*.sdf"))
    if sdf_files:
        for sdf_file in sdf_files:
            shutil.copy2(sdf_file, dst_dir / sdf_file.name)
    else:
        print(f"[missing] no sdf for {pdbid} in {comp_dir}")

print("Done")

