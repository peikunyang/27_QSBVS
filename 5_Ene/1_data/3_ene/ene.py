#!/usr/bin/env python3
"""
Recompute energies from normalized MAP/OCC vectors only.

This program does NOT read a direct-energy reference file.

Typical use:
    python ene.py ../2_norm/2ybs ene
    python ene.py ../2_norm/4gui ene

For one normalized receptor group:
    input_dir/
        map/
            <PDBID>_0.bin
            <PDBID>_1.bin
            len
        occ/
            <PDBID>.bin
            0001_xxxx.bin
            ...
            1023_xxxx.bin
            len
        info

The program searches:
    1024 OCC
    x 2 MAP poses
    x 32 rotation selectors
    x 512 translation selectors

For every pose:
    INNER = dot(normalized MAP, transformed normalized OCC)
    ENERGY = INNER * Lm * Lo - shift

Only the physical OCC channels are rotated/translated.
Auxiliary amplitudes remain unchanged.

The globally lowest energies are written to the output file.
"""

from __future__ import annotations

import argparse
import heapq
import math
from pathlib import Path

import numpy as np


N = 32
GRID_SIZE = N ** 3
ROTATION_STATES = 32
TRANSLATION_STATES = 512
N_SLOTS = 8
PHYSICAL_VALUES = N_SLOTS * GRID_SIZE   # 8 * 32^3 = 262144 = 2^18
FULL_VALUES = 2 * PHYSICAL_VALUES       # 524288 = 2^19
EXPECTED_OCC = 1024                     # 1 native + 1023 decoys
DEFAULT_TOP = 100


def read_single_map_len(path: Path) -> tuple[str, float, float]:
    """Read one line: PDBID Lm shift."""
    if not path.is_file():
        raise FileNotFoundError(f"MAP len not found: {path}")

    rows = []

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) != 3:
                raise ValueError(
                    f"invalid MAP len line: {line}\n"
                    "expected: PDBID Lm shift"
                )

            pdbid = parts[0]
            Lm = float(parts[1])
            shift = float(parts[2])

            if not math.isfinite(Lm) or Lm <= 0.0:
                raise ValueError(f"invalid Lm: {Lm}")

            if not math.isfinite(shift) or shift < 0.0:
                raise ValueError(f"invalid shift: {shift}")

            rows.append((pdbid, Lm, shift))

    if len(rows) != 1:
        raise ValueError(
            f"{path} must contain exactly one receptor group; "
            f"found {len(rows)}"
        )

    return rows[0]


def read_occ_len(path: Path) -> dict[str, float]:
    """Read OCC len file: OCC_ID Lo."""
    if not path.is_file():
        raise FileNotFoundError(f"OCC len not found: {path}")

    result = {}

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            parts = line.split()

            if len(parts) != 2:
                raise ValueError(
                    f"invalid OCC len at {path}:{line_number}: {line}"
                )

            occ_id = parts[0]
            Lo = float(parts[1])

            if occ_id in result:
                raise ValueError(
                    f"duplicate OCC len entry: {occ_id}"
                )

            if not math.isfinite(Lo) or Lo <= 0.0:
                raise ValueError(
                    f"invalid Lo for {occ_id}: {Lo}"
                )

            result[occ_id] = Lo

    if not result:
        raise ValueError(f"no OCC len records found: {path}")

    return result


def read_vector(
    path: Path,
    expected_size: int,
) -> np.ndarray:
    """Read one normalized float64 binary vector."""
    if not path.is_file():
        raise FileNotFoundError(f"vector not found: {path}")

    vector = np.fromfile(
        path,
        dtype=np.float64,
    )

    if vector.size != expected_size:
        raise ValueError(
            f"wrong vector size: {path}: "
            f"got {vector.size}, expected {expected_size}"
        )

    if not np.all(np.isfinite(vector)):
        raise ValueError(
            f"non-finite value found in: {path}"
        )

    return vector


def collect_occ_files(
    occ_dir: Path,
    occ_lengths: dict[str, float],
) -> list[tuple[str, Path, float]]:
    """Collect 1 native + 1023 decoy normalized OCC vectors."""
    files = sorted(
        occ_dir.glob("*.bin"),
        key=lambda p: p.stem,
    )

    if len(files) != EXPECTED_OCC:
        raise ValueError(
            f"{occ_dir} has {len(files)} OCC .bin files; "
            f"expected {EXPECTED_OCC}"
        )

    result = []

    for path in files:
        occ_id = path.stem

        if occ_id not in occ_lengths:
            raise ValueError(
                f"OCC len missing for: {occ_id}"
            )

        result.append(
            (
                occ_id,
                path,
                occ_lengths[occ_id],
            )
        )

    return result


def build_translation_shifts():
    """Same translation selector mapping as comb.py."""
    selector = np.arange(
        TRANSLATION_STATES,
        dtype=np.int64,
    )

    x0 = (selector >> 0) & 1
    x1 = (selector >> 1) & 1
    x2 = (selector >> 2) & 1

    y0 = (selector >> 3) & 1
    y1 = (selector >> 4) & 1
    y2 = (selector >> 5) & 1

    z0 = (selector >> 6) & 1
    z1 = (selector >> 7) & 1
    z2 = (selector >> 8) & 1

    shift_x = (
        np.where(x2 == 1, 0, -4)
        + 2 * x1
        + x0
    )
    shift_y = (
        np.where(y2 == 1, 0, -4)
        + 2 * y1
        + y0
    )
    shift_z = (
        np.where(z2 == 1, 0, -4)
        + 2 * z1
        + z0
    )

    return (
        shift_z.astype(np.int64),
        shift_y.astype(np.int64),
        shift_x.astype(np.int64),
    )


def apply_rotation_numpy(
    vector: np.ndarray,
    rotation_selector: int,
) -> np.ndarray:
    """
    Same 5-bit OCC rotation as comb.py.

    vector shape:
        (n_slots, z, y, x)
    """
    output = vector

    if (rotation_selector >> 4) & 1:
        output = np.rot90(
            output,
            k=1,
            axes=(2, 3),
        )

    if (rotation_selector >> 3) & 1:
        output = np.rot90(
            output,
            k=2,
            axes=(2, 3),
        )

    if (rotation_selector >> 2) & 1:
        output = np.rot90(
            output,
            k=-1,
            axes=(1, 3),
        )

    if (rotation_selector >> 1) & 1:
        output = np.rot90(
            output,
            k=2,
            axes=(1, 3),
        )

    if rotation_selector & 1:
        output = np.rot90(
            output,
            k=1,
            axes=(1, 2),
        )

    return output


def apply_rotation_torch(
    vector,
    rotation_selector: int,
    torch,
):
    """CUDA equivalent of apply_rotation_numpy()."""
    output = vector

    if (rotation_selector >> 4) & 1:
        output = torch.rot90(
            output,
            k=1,
            dims=(2, 3),
        )

    if (rotation_selector >> 3) & 1:
        output = torch.rot90(
            output,
            k=2,
            dims=(2, 3),
        )

    if (rotation_selector >> 2) & 1:
        output = torch.rot90(
            output,
            k=-1,
            dims=(1, 3),
        )

    if (rotation_selector >> 1) & 1:
        output = torch.rot90(
            output,
            k=2,
            dims=(1, 3),
        )

    if rotation_selector & 1:
        output = torch.rot90(
            output,
            k=1,
            dims=(1, 2),
        )

    return output.contiguous()


def resolve_device(requested: str) -> str:
    if requested == "cpu":
        return "cpu"

    try:
        import torch
        cuda_ok = torch.cuda.is_available()
    except ImportError:
        cuda_ok = False

    if requested == "cuda":
        if not cuda_ok:
            raise RuntimeError(
                "--device cuda requested, "
                "but CUDA PyTorch is unavailable"
            )
        return "cuda"

    return "cuda" if cuda_ok else "cpu"


def keep_top(
    heap,
    counter: int,
    top_k: int,
    record: dict,
):
    """
    Maintain only global top_k lowest recovered energies.

    Heap stores the current worst item at heap[0].
    """
    energy = record["energy"]
    item = (
        -energy,
        counter,
        record,
    )

    if len(heap) < top_k:
        heapq.heappush(
            heap,
            item,
        )

    elif energy < -heap[0][0]:
        heapq.heapreplace(
            heap,
            item,
        )


def prepare_map_numpy(
    map_vectors: dict[int, np.ndarray],
    n_slots: int,
    physical_values: int,
):
    spectra = {}

    for pose in (0, 1):
        physical = (
            map_vectors[pose][:physical_values]
            .reshape(
                n_slots,
                N,
                N,
                N,
            )
        )

        spectra[pose] = np.fft.rfftn(
            physical,
            axes=(1, 2, 3),
        )

    return spectra


def score_occ_numpy(
    pdbid: str,
    occ_id: str,
    occ_vector: np.ndarray,
    Lo: float,
    map_vectors: dict[int, np.ndarray],
    map_spectra,
    Lm: float,
    shift: float,
    n_slots: int,
    physical_values: int,
    shift_z,
    shift_y,
    shift_x,
    heap,
    counter,
    top_k: int,
):
    physical = (
        occ_vector[:physical_values]
        .reshape(
            n_slots,
            N,
            N,
            N,
        )
    )

    # Auxiliary inner product is pose invariant.
    aux_inner = {
        pose: float(
            np.dot(
                map_vectors[pose][physical_values:],
                occ_vector[physical_values:],
            )
        )
        for pose in (0, 1)
    }

    source = (
        "native"
        if occ_id == pdbid
        else "decoy"
    )

    for rot in range(ROTATION_STATES):
        rotated = apply_rotation_numpy(
            physical,
            rot,
        )

        occ_fft = np.fft.rfftn(
            rotated,
            axes=(1, 2, 3),
        )

        for pose in (0, 1):
            correlation = np.fft.irfftn(
                np.sum(
                    map_spectra[pose]
                    * np.conj(occ_fft),
                    axis=0,
                ),
                s=(N, N, N),
                axes=(0, 1, 2),
            ).real

            physical_inner = correlation[
                shift_z,
                shift_y,
                shift_x,
            ]

            inner_values = (
                physical_inner
                + aux_inner[pose]
            )

            energy_values = (
                inner_values
                * Lm
                * Lo
                - shift
            )

            local_k = min(
                top_k,
                TRANSLATION_STATES,
            )

            idxs = np.argpartition(
                energy_values,
                local_k - 1,
            )[:local_k]

            for trans in idxs:
                trans = int(trans)

                record = {
                    "source": source,
                    "ligand": occ_id,
                    "pose": pose,
                    "rot": rot,
                    "trans": trans,
                    "dx": int(shift_x[trans]),
                    "dy": int(shift_y[trans]),
                    "dz": int(shift_z[trans]),
                    "inner": float(
                        inner_values[trans]
                    ),
                    "energy": float(
                        energy_values[trans]
                    ),
                }

                keep_top(
                    heap,
                    counter[0],
                    top_k,
                    record,
                )
                counter[0] += 1


def prepare_cuda():
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "CUDA mode requires PyTorch"
        ) from exc

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA requested, but unavailable"
        )

    return (
        torch,
        torch.device("cuda"),
    )


def prepare_map_cuda(
    map_vectors,
    n_slots,
    physical_values,
    torch,
    device,
):
    spectra = {}
    map_tensors = {}

    for pose in (0, 1):
        tensor = torch.from_numpy(
            map_vectors[pose]
        ).to(
            device=device,
            dtype=torch.float64,
        )

        map_tensors[pose] = tensor

        physical = (
            tensor[:physical_values]
            .reshape(
                n_slots,
                N,
                N,
                N,
            )
        )

        spectra[pose] = torch.fft.rfftn(
            physical,
            dim=(1, 2, 3),
        )

    return map_tensors, spectra


def score_occ_cuda(
    pdbid,
    occ_id,
    occ_vector,
    Lo,
    map_tensors,
    map_spectra,
    Lm,
    shift,
    n_slots,
    physical_values,
    shift_z_np,
    shift_y_np,
    shift_x_np,
    shift_z,
    shift_y,
    shift_x,
    heap,
    counter,
    top_k,
    torch,
    device,
):
    occ_tensor = torch.from_numpy(
        occ_vector
    ).to(
        device=device,
        dtype=torch.float64,
    )

    physical = (
        occ_tensor[:physical_values]
        .reshape(
            n_slots,
            N,
            N,
            N,
        )
    )

    aux_inner = {
        pose: torch.dot(
            map_tensors[pose][physical_values:],
            occ_tensor[physical_values:],
        )
        for pose in (0, 1)
    }

    source = (
        "native"
        if occ_id == pdbid
        else "decoy"
    )

    with torch.no_grad():
        for rot in range(
            ROTATION_STATES
        ):
            rotated = apply_rotation_torch(
                physical,
                rot,
                torch,
            )

            occ_fft = torch.fft.rfftn(
                rotated,
                dim=(1, 2, 3),
            )

            for pose in (0, 1):
                correlation = torch.fft.irfftn(
                    torch.sum(
                        map_spectra[pose]
                        * torch.conj(occ_fft),
                        dim=0,
                    ),
                    s=(N, N, N),
                    dim=(0, 1, 2),
                ).real

                physical_inner = correlation[
                    shift_z,
                    shift_y,
                    shift_x,
                ]

                inner_values = (
                    physical_inner
                    + aux_inner[pose]
                )

                energy_values = (
                    inner_values
                    * Lm
                    * Lo
                    - shift
                )

                local_k = min(
                    top_k,
                    TRANSLATION_STATES,
                )

                values, indices = torch.topk(
                    energy_values,
                    k=local_k,
                    largest=False,
                    sorted=False,
                )

                inner_selected = (
                    inner_values[indices]
                )

                values_np = (
                    values.detach()
                    .cpu()
                    .numpy()
                )
                indices_np = (
                    indices.detach()
                    .cpu()
                    .numpy()
                )
                inner_np = (
                    inner_selected.detach()
                    .cpu()
                    .numpy()
                )

                for j in range(local_k):
                    trans = int(
                        indices_np[j]
                    )

                    record = {
                        "source": source,
                        "ligand": occ_id,
                        "pose": pose,
                        "rot": rot,
                        "trans": trans,
                        "dx": int(
                            shift_x_np[trans]
                        ),
                        "dy": int(
                            shift_y_np[trans]
                        ),
                        "dz": int(
                            shift_z_np[trans]
                        ),
                        "inner": float(
                            inner_np[j]
                        ),
                        "energy": float(
                            values_np[j]
                        ),
                    }

                    keep_top(
                        heap,
                        counter[0],
                        top_k,
                        record,
                    )
                    counter[0] += 1


def write_output(
    path: Path,
    records: list[dict],
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
            f"{'rank':>5} "
            f"{'source':<7} "
            f"{'ligand':<16} "
            f"{'pose':>4} "
            f"{'rot':>4} "
            f"{'trans':>5} "
            f"{'dx':>3} "
            f"{'dy':>3} "
            f"{'dz':>3} "
            f"{'INNER':>18} "
            f"{'total':>14}\n"
        )

        for rank, row in enumerate(
            records,
            start=1,
        ):
            f.write(
                f"{rank:5d} "
                f"{row['source']:<7} "
                f"{row['ligand']:<16} "
                f"{row['pose']:4d} "
                f"{row['rot']:4d} "
                f"{row['trans']:5d} "
                f"{row['dx']:3d} "
                f"{row['dy']:3d} "
                f"{row['dz']:3d} "
                f"{row['inner']:18.12e} "
                f"{row['energy']:14.6f}\n"
            )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Recompute top energies from normalized "
            "MAP/OCC vectors without a reference file."
        )
    )

    parser.add_argument(
        "input_dir",
        type=Path,
        help=(
            "one normalized receptor group, "
            "e.g. ../2_norm/2ybs"
        ),
    )

    parser.add_argument(
        "output_file",
        type=Path,
        help="output text file",
    )

    parser.add_argument(
        "--top",
        type=int,
        default=DEFAULT_TOP,
        help=(
            f"number of lowest energies to save "
            f"(default {DEFAULT_TOP})"
        ),
    )

    parser.add_argument(
        "--device",
        choices=(
            "auto",
            "cpu",
            "cuda",
        ),
        default="auto",
        help=(
            "auto uses CUDA when available; "
            "otherwise NumPy CPU FFT"
        ),
    )

    args = parser.parse_args()

    if args.top < 1:
        raise ValueError(
            "--top must be >= 1"
        )

    input_dir = args.input_dir.resolve()
    map_dir = input_dir / "map"
    occ_dir = input_dir / "occ"

    if not input_dir.is_dir():
        raise FileNotFoundError(
            f"input directory not found: "
            f"{input_dir}"
        )

    pdbid, Lm, shift = (
        read_single_map_len(
            map_dir / "len"
        )
    )

    occ_lengths = read_occ_len(
        occ_dir / "len"
    )

    # Fixed current encoding:
    #   8 physical channels:
    #       e, A, C, HD, N, NA, OA, S
    #   physical = 8 * 32^3 = 2^18 values
    #   full     = 2 * physical = 2^19 values
    n_slots = N_SLOTS
    physical_values = PHYSICAL_VALUES
    full_values = FULL_VALUES

    if n_slots != 8:
        raise AssertionError(f"fixed n_slots must be 8, got {n_slots}")
    if physical_values != 2 ** 18:
        raise AssertionError(
            f"fixed physical_values must be 2^18, got {physical_values}"
        )
    if full_values != 2 ** 19:
        raise AssertionError(
            f"fixed full_values must be 2^19, got {full_values}"
        )

    occ_files = collect_occ_files(
        occ_dir,
        occ_lengths,
    )

    map_vectors = {
        pose: read_vector(
            map_dir / f"{pdbid}_{pose}.bin",
            full_values,
        )
        for pose in (0, 1)
    }

    (
        shift_z_np,
        shift_y_np,
        shift_x_np,
    ) = build_translation_shifts()

    device_mode = resolve_device(
        args.device
    )

    print(f"PDBID          {pdbid}")
    print(f"OCC vectors    {len(occ_files)}")
    print(f"channels       e A C HD N NA OA S")
    print(f"n_slots        {n_slots} (fixed)")
    print(f"physical_values {physical_values} (=2^18)")
    print(f"full_values    {full_values} (=2^19)")
    print(f"Lm             {Lm:.12e}")
    print(f"shift          {shift:.12f}")
    print(f"device         {device_mode}")
    print(f"top            {args.top}")
    print()

    heap = []
    counter = [0]

    if device_mode == "cuda":
        torch, device = prepare_cuda()

        print(
            f"CUDA GPU       "
            f"{torch.cuda.get_device_name(device)}"
        )

        (
            map_tensors,
            map_spectra,
        ) = prepare_map_cuda(
            map_vectors,
            n_slots,
            physical_values,
            torch,
            device,
        )

        shift_z = torch.as_tensor(
            shift_z_np,
            dtype=torch.long,
            device=device,
        )
        shift_y = torch.as_tensor(
            shift_y_np,
            dtype=torch.long,
            device=device,
        )
        shift_x = torch.as_tensor(
            shift_x_np,
            dtype=torch.long,
            device=device,
        )

        for i, (
            occ_id,
            occ_path,
            Lo,
        ) in enumerate(
            occ_files,
            start=1,
        ):
            print(
                f"[{pdbid}] OCC "
                f"{i}/{len(occ_files)} "
                f"{occ_id}"
            )

            occ_vector = read_vector(
                occ_path,
                full_values,
            )

            score_occ_cuda(
                pdbid,
                occ_id,
                occ_vector,
                Lo,
                map_tensors,
                map_spectra,
                Lm,
                shift,
                n_slots,
                physical_values,
                shift_z_np,
                shift_y_np,
                shift_x_np,
                shift_z,
                shift_y,
                shift_x,
                heap,
                counter,
                args.top,
                torch,
                device,
            )

    else:
        map_spectra = prepare_map_numpy(
            map_vectors,
            n_slots,
            physical_values,
        )

        for i, (
            occ_id,
            occ_path,
            Lo,
        ) in enumerate(
            occ_files,
            start=1,
        ):
            print(
                f"[{pdbid}] OCC "
                f"{i}/{len(occ_files)} "
                f"{occ_id}"
            )

            occ_vector = read_vector(
                occ_path,
                full_values,
            )

            score_occ_numpy(
                pdbid,
                occ_id,
                occ_vector,
                Lo,
                map_vectors,
                map_spectra,
                Lm,
                shift,
                n_slots,
                physical_values,
                shift_z_np,
                shift_y_np,
                shift_x_np,
                heap,
                counter,
                args.top,
            )

    records = [
        item[2]
        for item in heap
    ]

    records.sort(
        key=lambda row: (
            row["energy"],
            row["source"],
            row["ligand"],
            row["pose"],
            row["rot"],
            row["trans"],
        )
    )

    write_output(
        args.output_file,
        records,
    )

    print()
    print(
        f"pose scores    "
        f"{len(occ_files) * 2 * 32 * 512}"
    )
    print(
        f"saved          "
        f"{len(records)}"
    )
    print(
        f"output         "
        f"{args.output_file}"
    )


if __name__ == "__main__":
    main()
