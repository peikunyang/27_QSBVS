#!/usr/bin/env python3

import argparse
import math
from pathlib import Path

import numpy as np


DEFAULT_SEED = 20260812
TOL = 1e-10


def dimension_from_grid(grid_size):
    if grid_size not in (2, 4, 8):
        raise ValueError("--grid-size must be 2, 4, or 8")
    return 2 * grid_size**3


def make_occ_pair(rng, dimension):
    occs = rng.uniform(0.0, 1.0, size=(2, dimension))
    occs /= np.linalg.norm(occs, axis=1)[:, None]
    return occs


def make_target_map(target_occ, other_occ, target_value):
    c = float(np.dot(target_occ, other_occ))
    orth = other_occ - c * target_occ
    orth_norm = float(np.linalg.norm(orth))
    if orth_norm <= TOL:
        return None
    orth /= orth_norm
    beta = math.sqrt(max(0.0, 1.0 - target_value**2))
    vector = target_value * target_occ + beta * orth
    vector /= np.linalg.norm(vector)
    return vector


def make_other_map(occ1, occ2):
    vector = occ1 + occ2
    norm = float(np.linalg.norm(vector))
    if norm <= TOL:
        return None
    return vector / norm


def scores(maps, occs):
    return np.asarray(
        [[float(np.dot(maps[m], occs[o])) for o in range(2)] for m in range(2)],
        dtype=np.float64,
    )


def build_data(seed, grid_size, threshold, target_map, target_occ, target_value):
    if not 0.0 < threshold < 1.0:
        raise ValueError("--threshold must satisfy 0 < threshold < 1")

    if target_value is None:
        target_value = 0.5 * threshold
    if not 0.0 <= target_value < threshold:
        raise ValueError("--target-value must satisfy 0 <= target-value < threshold")

    dimension = dimension_from_grid(grid_size)
    rng = np.random.default_rng(seed)

    if target_map is None:
        target_map = int(rng.integers(1, 3))
    if target_occ is None:
        target_occ = int(rng.integers(1, 3))

    tm = target_map - 1
    to = target_occ - 1
    oo = 1 - to

    for _ in range(10000):
        occs = make_occ_pair(rng, dimension)

        target_vector = make_target_map(
            occs[to],
            occs[oo],
            target_value,
        )
        other_vector = make_other_map(occs[0], occs[1])
        if target_vector is None or other_vector is None:
            continue

        maps = np.empty((2, dimension), dtype=np.float64)
        maps[tm] = target_vector
        maps[1 - tm] = other_vector

        values = scores(maps, occs)
        below = values < threshold

        if int(np.sum(below)) != 1:
            continue
        if not bool(below[tm, to]):
            continue

        return {
            "grid_size": grid_size,
            "dimension": dimension,
            "threshold": threshold,
            "target_value": target_value,
            "target_map": target_map,
            "target_occ": target_occ,
            "occs": occs,
            "maps": maps,
            "scores": values,
        }

    raise RuntimeError(
        "could not generate exactly one MAP/OCC combination below threshold; "
        "try a smaller threshold or another seed"
    )


def write_input(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for name, vector in (
            ("O1", data["occs"][0]),
            ("O2", data["occs"][1]),
            ("M1", data["maps"][0]),
            ("M2", data["maps"][1]),
        ):
            handle.write(f"{name}\n")
            for start in range(0, len(vector), 8):
                block = vector[start:start + 8]
                handle.write(" ".join(f"{x:16.12f}" for x in block) + "\n")
            handle.write("\n")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--grid-size", type=int, choices=(2, 4, 8), default=2)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--target-map", type=int, choices=(1, 2), default=None)
    parser.add_argument("--target-occ", type=int, choices=(1, 2), default=None)
    parser.add_argument("--target-value", type=float, default=None)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def main():
    args = parse_args()
    data = build_data(
        seed=args.seed,
        grid_size=args.grid_size,
        threshold=args.threshold,
        target_map=args.target_map,
        target_occ=args.target_occ,
        target_value=args.target_value,
    )
    write_input(args.output, data)

    s = data["scores"]
    print(f"input      {args.output}")
    print(f"grid       {data['grid_size']}")
    print(f"values     {data['dimension']} per vector")
    print(f"threshold  {data['threshold']:.7f}")
    print(f"target     MAP{data['target_map']} OCC{data['target_occ']}")
    print(f"MAP1 OCC1  {s[0,0]:.10f}")
    print(f"MAP1 OCC2  {s[0,1]:.10f}")
    print(f"MAP2 OCC1  {s[1,0]:.10f}")
    print(f"MAP2 OCC2  {s[1,1]:.10f}")
    print(f"below thr  {int(np.sum(s < data['threshold']))}/4")


if __name__ == "__main__":
    main()
