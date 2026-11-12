import sys
import random
from pathlib import Path
from itertools import combinations

import numpy as np

EPS = 1e-8


def read_pdbqt_coords(file_path):
    coords = []

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                try:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                except ValueError:
                    tail = line[30:].split()
                    if len(tail) < 3:
                        continue
                    x, y, z = map(float, tail[:3])

                coords.append([x, y, z])

    if not coords:
        raise ValueError(f"No ATOM/HETATM coordinates found in {file_path}")

    return np.array(coords, dtype=float)


def get_atom_types(file_path):
    atom_types = set()

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                parts = line.split()
                if not parts:
                    continue
                atom_type = parts[-1]
                atom_types.add(atom_type)

    return sorted(atom_types)


def point_in_sphere(p, center, radius):
    return np.linalg.norm(p - center) <= radius + EPS


def all_points_in_sphere(points, center, radius):
    d = np.linalg.norm(points - center, axis=1)
    return np.all(d <= radius + EPS)


def sphere_from_boundary(boundary):
    boundary = [np.array(p, dtype=float) for p in boundary]

    if len(boundary) == 0:
        return np.zeros(3, dtype=float), 0.0

    candidates = []

    for p in boundary:
        candidates.append((p.copy(), 0.0))

    for a, b in combinations(boundary, 2):
        center = (a + b) / 2.0
        radius = np.linalg.norm(a - center)
        candidates.append((center, radius))

    for a, b, c in combinations(boundary, 3):
        n = np.cross(b - a, c - a)
        if np.linalg.norm(n) < EPS:
            continue

        A = np.vstack([
            2.0 * (b - a),
            2.0 * (c - a),
            n
        ])

        B = np.array([
            np.dot(b, b) - np.dot(a, a),
            np.dot(c, c) - np.dot(a, a),
            np.dot(n, a)
        ], dtype=float)

        try:
            center = np.linalg.solve(A, B)
        except np.linalg.LinAlgError:
            continue

        radius = np.linalg.norm(a - center)
        candidates.append((center, radius))

    for a, b, c, d in combinations(boundary, 4):
        A = np.vstack([
            2.0 * (b - a),
            2.0 * (c - a),
            2.0 * (d - a)
        ])

        if abs(np.linalg.det(A)) < EPS:
            continue

        B = np.array([
            np.dot(b, b) - np.dot(a, a),
            np.dot(c, c) - np.dot(a, a),
            np.dot(d, d) - np.dot(a, a)
        ], dtype=float)

        try:
            center = np.linalg.solve(A, B)
        except np.linalg.LinAlgError:
            continue

        radius = np.linalg.norm(a - center)
        candidates.append((center, radius))

    valid = []
    pts = np.array(boundary, dtype=float)

    for center, radius in candidates:
        if all_points_in_sphere(pts, center, radius):
            valid.append((center, radius))

    if not valid:
        raise RuntimeError("Failed to build enclosing sphere from boundary points")

    return min(valid, key=lambda x: x[1])


def welzl(points, boundary, n):
    if n == 0 or len(boundary) == 4:
        return sphere_from_boundary(boundary)

    p = points[n - 1]
    center, radius = welzl(points, boundary, n - 1)

    if point_in_sphere(p, center, radius):
        return center, radius

    return welzl(points, boundary + [p], n - 1)


def minimum_enclosing_sphere(points):
    pts = [np.array(p, dtype=float) for p in points]
    random.shuffle(pts)
    sys.setrecursionlimit(max(10000, len(pts) + 100))
    return welzl(pts, [], len(pts))


def calc_diameter_from_pdbqt(file_path):
    coords = read_pdbqt_coords(file_path)

    if len(coords) <= 1:
        return None

    _, radius = minimum_enclosing_sphere(coords)
    return 2.0 * radius


def get_pdbid_from_filename(file_path):
    name = file_path.name

    if name.endswith("_ligand.pdbqt"):
        return name.replace("_ligand.pdbqt", "")

    return file_path.stem


def process_flat_folder(input_root, output_file):
    root = Path(input_root)

    if not root.exists():
        raise FileNotFoundError(f"Input folder not found: {input_root}")

    pdbqt_files = sorted(root.glob("*_ligand.pdbqt"))

    if not pdbqt_files:
        pdbqt_files = sorted(root.glob("*.pdbqt"))

    if not pdbqt_files:
        raise FileNotFoundError(f"No pdbqt files found in {input_root}")

    lines = []

    for file_path in pdbqt_files:
        pdbid = get_pdbid_from_filename(file_path)

        try:
            diameter = calc_diameter_from_pdbqt(file_path)
        except Exception as e:
            print(f"[SKIP] {file_path.name}: {e}")
            continue

        if diameter is None:
            print(f"[SKIP] {file_path.name}: only one atom")
            continue

        atom_types = get_atom_types(file_path)
        atom_types_text = ",".join(atom_types)

        lines.append(f"{pdbid}\t{diameter:6.3f}\t{atom_types_text}")

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if len(sys.argv) != 3:
    print("Usage: python size.py <input_pdbqt_folder> <output_file>")
    sys.exit(1)

input_root = sys.argv[1]
output_file = sys.argv[2]

process_flat_folder(input_root, output_file)

print(f"Done. Output written to {output_file}")
