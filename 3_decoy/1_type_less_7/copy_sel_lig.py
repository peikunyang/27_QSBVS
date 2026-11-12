import os
import sys
import shutil


# Ranked selection file.
SEL_FILE = "../../1_data/4_lig_pdbqt/3_sel/2_decoy/sel_union7"

# Folder containing the original ligand PDBQT files.
LIG_DIR = "../../1_data/4_lig_pdbqt/1_pdbqt/lig"

# Copy only the first 1023 ranked ligands.
MAX_LIGANDS = 1023


def main():
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} output_dir")
        sys.exit(1)

    output_dir = sys.argv[1]
    os.makedirs(output_dir, exist_ok=True)

    copied = 0
    missing = 0
    rank = 0

    with open(SEL_FILE, "r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()

            if not line:
                continue

            # Stop after the first 1023 ranked entries in sel_union7.
            if rank >= MAX_LIGANDS:
                break

            parts = line.split()
            if not parts:
                continue

            pdbid = parts[0]
            rank += 1

            src_name = f"{pdbid}_ligand.pdbqt"
            src = os.path.join(LIG_DIR, src_name)

            # Rank is zero-padded to four digits:
            #   0001_9icd.pdbqt
            #   ...
            #   1023_xxxx.pdbqt
            dst_name = f"{rank:04d}_{pdbid}.pdbqt"
            dst = os.path.join(output_dir, dst_name)

            if os.path.isfile(src):
                shutil.copy2(src, dst)
                copied += 1
            else:
                print(f"Missing rank {rank:04d}: {src}")
                missing += 1

    print()
    print("Done")
    print(f"Requested ranks : {rank}")
    print(f"Copied          : {copied}")
    print(f"Missing         : {missing}")
    print(f"Output          : {output_dir}")


if __name__ == "__main__":
    main()
