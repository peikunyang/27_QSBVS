import os
import sys
from itertools import combinations


def parse_atom_types(text):
    """Parse comma-separated atom types from the third column."""
    return frozenset(x for x in text.split(",") if x)


def main():
    if len(sys.argv) != 4:
        print(
            f"Usage: python {sys.argv[0]} "
            "input_file output_dir MAX_UNION_TYPES"
        )
        sys.exit(1)

    input_file = sys.argv[1]
    output_dir = sys.argv[2]
    max_union_types = int(sys.argv[3])

    if max_union_types < 1:
        raise ValueError("MAX_UNION_TYPES must be >= 1")

    output_file = os.path.join(
        output_dir,
        f"sel_union{max_union_types}",
    )

    os.makedirs(output_dir, exist_ok=True)

    # Input format:
    # PDBID  diameter  atom_type1,atom_type2,...
    candidates = []

    with open(input_file, "r", encoding="utf-8") as fin:
        for line_no, line in enumerate(fin, start=1):
            stripped = line.strip()
            if not stripped:
                continue

            parts = stripped.split()
            if len(parts) < 3:
                print(f"[SKIP] line {line_no}: expected 3 columns")
                continue

            pdbid = parts[0]

            try:
                diameter = float(parts[1])
            except ValueError:
                print(f"[SKIP] line {line_no}: invalid diameter")
                continue

            atom_types = parse_atom_types(parts[2])
            if not atom_types:
                print(f"[SKIP] line {line_no}: no atom types")
                continue

            # A ligand that alone already exceeds the allowed union
            # can never be part of a valid selected set.
            if len(atom_types) > max_union_types:
                continue

            candidates.append(
                {
                    "pdbid": pdbid,
                    "diameter": diameter,
                    "atom_types": atom_types,
                    "line": stripped,
                }
            )

    if not candidates:
        raise ValueError("No valid ligands found")

    all_types = sorted(
        set().union(*(item["atom_types"] for item in candidates))
    )

    best_indices = []
    best_union = frozenset()

    # Exact search:
    # enumerate every possible atom-type union U with |U| <= MAX_UNION_TYPES.
    # For a given U, the maximum ligand subset is all ligands whose
    # atom-type sets are subsets of U.
    max_r = min(max_union_types, len(all_types))

    # Also allow the rare case where no type combination selects anything.
    for r in range(1, max_r + 1):
        for combo in combinations(all_types, r):
            union_types = frozenset(combo)

            indices = [
                i
                for i, item in enumerate(candidates)
                if item["atom_types"].issubset(union_types)
            ]

            if len(indices) > len(best_indices):
                best_indices = indices
                best_union = union_types

            elif len(indices) == len(best_indices) and indices:
                # Deterministic tie-break:
                # 1) use fewer union atom types
                # 2) lexicographically smaller atom-type list
                current_key = (
                    len(union_types),
                    tuple(sorted(union_types)),
                )
                best_key = (
                    len(best_union),
                    tuple(sorted(best_union)),
                )

                if current_key < best_key:
                    best_indices = indices
                    best_union = union_types

    selected = [candidates[i] for i in best_indices]

    # Sort by the second column (diameter), from largest to smallest.
    selected.sort(key=lambda item: item["diameter"], reverse=True)

    with open(output_file, "w", encoding="utf-8") as fout:
        for item in selected:
            fout.write(item["line"] + "\n")

    print(f"Done: {output_file}")
    print(f"Selected ligands: {len(selected)} / {len(candidates)}")
    print(
        f"Atom-type union ({len(best_union)}): "
        + ",".join(sorted(best_union))
    )


if __name__ == "__main__":
    main()
