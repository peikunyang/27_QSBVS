#!/usr/bin/env python3
"""
Prepare independent receptor-PDBID datasets for the downstream quantum circuit.

Input layout
------------
map_root/
    <PDBID>_0/
    <PDBID>_1/
    ...

occ_root/
    occ/
        <PDBID>/
        ...
    occ_decoy/
        00002_xxxx/
        0002_xxxx/
        ...
        1023_xxxx/

For each receptor PDBID independently:
    - MAP pair:
        <PDBID>_0
        <PDBID>_1
    - OCC set:
        native OCC with the same PDBID
        + all 1023 decoy OCC folders
      = 1024 OCC vectors

Each group gets its own:
    shift
    Lm  (shared by MAP _0/_1)
    Lo  (shared by that group's 1024 OCC vectors)

The nonnegative shift is determined over the complete pose set:
    2 MAP poses
    x 1024 OCC ligands
    x 32 rotation selectors
    x 512 translation selectors

Translations use the same cyclic modulo-32 convention as the original code.

Output layout
-------------
output_root/
    <PDBID>/
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

Important encoding rule
-----------------------
All MAP and OCC vectors use exactly eight physical channels in the same order:

    e, followed by the seven atom types listed in
    ../../../1_data/4_lig_pdbqt/3_sel/2_decoy/log

For the current selection log this is:

    e, A, C, HD, N, NA, OA, S

If a MAP or OCC folder does not contain one of these channels, that channel is
filled with an all-zero 32 x 32 x 32 grid.

Physical vector:
    8 * 32^3 = 262144 = 2^18

One auxiliary qubit is then added for shift / norm equalization:

    full vector = 2 * 2^18 = 2^19
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
import re
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N = 32
GRID_SIZE = N ** 3
TOL = 1e-10

ROTATION_STATES = 32
TRANSLATION_STATES = 512
EXPECTED_DECOYS = 1023

# The seven selected ligand atom types are read from this log.
# "e" is the electrostatic channel and is always placed first.
TYPES_LOG = (
    Path(__file__).resolve().parent
    / "../../../1_data/4_lig_pdbqt/3_sel/2_decoy/log"
).resolve()

N_SLOTS = 8

# Initialized in main() from TYPES_LOG.
ATOM_TYPE_ORDER: tuple[str, ...] = ()
ATOM_TYPE_SET: set[str] = set()

MAP_RE = re.compile(r"^(?P<pdbid>.+)_(?P<index>[01])$")


def load_fixed_channel_order(log_file: Path) -> tuple[str, ...]:
    """
    Read:
        Atom-type union (7): A,C,HD,N,NA,OA,S

    and return:
        ("e", "A", "C", "HD", "N", "NA", "OA", "S")
    """
    if not log_file.is_file():
        raise FileNotFoundError(
            f"selection log not found: {log_file}"
        )

    pattern = re.compile(
        r"^Atom-type union\s*\((\d+)\)\s*:\s*(.+?)\s*$"
    )

    with log_file.open(
        "r",
        encoding="utf-8",
        errors="ignore",
    ) as f:
        for line in f:
            match = pattern.match(line.strip())
            if match is None:
                continue

            expected_count = int(match.group(1))
            selected_types = [
                x.strip()
                for x in match.group(2).split(",")
                if x.strip()
            ]
            selected_types = list(dict.fromkeys(selected_types))

            if expected_count != 7:
                raise ValueError(
                    f"{log_file}: expected a 7-type union, "
                    f"but log reports {expected_count}"
                )

            if len(selected_types) != 7:
                raise ValueError(
                    f"{log_file}: expected 7 unique atom types, "
                    f"parsed {len(selected_types)}: "
                    + ",".join(selected_types)
                )

            if "e" in selected_types:
                raise ValueError(
                    f"{log_file}: electrostatic channel 'e' must not "
                    "be included in the 7 ligand atom types"
                )

            return ("e", *selected_types)

    raise ValueError(
        f"Atom-type union (...) line not found in: {log_file}"
    )


@dataclass(frozen=True)
class Backend:
    name: str
    torch_module: object | None = None
    torch_device: object | None = None


@dataclass(frozen=True)
class Layout:
    group_types: tuple[str, ...]
    n_slots: int
    physical_size: int
    full_size: int
    aux0: int
    physical_qubits: int
    full_qubits: int


# ---------------------------------------------------------------------------
# Basic readers / validation
# ---------------------------------------------------------------------------

def read_grid(path: Path) -> np.ndarray:
    """Read exactly 32^3 float64 values."""
    values = np.fromstring(
        path.read_text(encoding="utf-8", errors="ignore"),
        sep=" ",
        dtype=np.float64,
    )

    if values.size != GRID_SIZE:
        raise ValueError(
            f"wrong number of values: {path}: "
            f"got {values.size}, expected {GRID_SIZE}"
        )

    if not np.all(np.isfinite(values)):
        raise ValueError(f"non-finite value found in: {path}")

    return values


def child_dirs(root: Path) -> list[Path]:
    dirs = sorted(
        (p for p in root.iterdir() if p.is_dir()),
        key=lambda p: p.name,
    )

    if not dirs:
        raise ValueError(f"no subfolders found in: {root}")

    return dirs


def folder_atom_types(folder: Path) -> tuple[str, ...]:
    """
    Return the fixed-order channels that are present in one OCC folder.

    Every OCC may contain any subset of the fixed eight channels.
    Missing channels are zero-filled later.
    """
    entries = list(folder.iterdir())

    bad_dirs = sorted(p.name for p in entries if p.is_dir())
    if bad_dirs:
        raise ValueError(
            f"unexpected subfolder(s) inside OCC folder {folder}: "
            + ", ".join(bad_dirs)
        )

    names = {p.name for p in entries if p.is_file()}

    unknown = sorted(names - ATOM_TYPE_SET)
    if unknown:
        raise ValueError(
            f"unexpected OCC atom type(s) in {folder}: "
            + ", ".join(unknown)
            + "\nfixed channel order: "
            + " ".join(ATOM_TYPE_ORDER)
        )

    ordered = tuple(t for t in ATOM_TYPE_ORDER if t in names)

    if not ordered:
        raise ValueError(
            f"{folder} has no recognized atom-type/electrostatic files"
        )

    return ordered


def collect_maps(root: Path) -> dict[str, dict[int, Path]]:
    """Collect <PDBID>_0 and <PDBID>_1 MAP folders."""
    out: dict[str, dict[int, Path]] = {}

    for folder in child_dirs(root):
        match = MAP_RE.match(folder.name)

        if match is None:
            raise ValueError(
                f"invalid MAP folder name: {folder.name}; "
                "expected <PDBID>_0 or <PDBID>_1"
            )

        pdbid = match.group("pdbid")
        index = int(match.group("index"))

        out.setdefault(pdbid, {})

        if index in out[pdbid]:
            raise ValueError(f"duplicate MAP folder: {pdbid}_{index}")

        out[pdbid][index] = folder

    for pdbid, pair in sorted(out.items()):
        if set(pair) != {0, 1}:
            found = " ".join(
                f"{pdbid}_{index}"
                for index in sorted(pair)
            )
            raise ValueError(
                f"MAP PDBID {pdbid} must contain both _0 and _1; "
                f"found: {found}"
            )

    return out


def collect_native_occs(native_root: Path) -> dict[str, Path]:
    out = {}

    for folder in child_dirs(native_root):
        if folder.name in out:
            raise ValueError(f"duplicate native OCC PDBID: {folder.name}")
        out[folder.name] = folder

    return out


def collect_decoy_occs(decoy_root: Path) -> list[tuple[str, Path]]:
    """
    Return decoys in folder-name order:
        0001_xxxx, 0002_xxxx, ... 1023_xxxx
    """
    dirs = child_dirs(decoy_root)

    if len(dirs) != EXPECTED_DECOYS:
        raise ValueError(
            f"{decoy_root} has {len(dirs)} decoy folders; "
            f"expected {EXPECTED_DECOYS}"
        )

    names = [p.name for p in dirs]
    if len(set(names)) != len(names):
        raise ValueError("duplicate decoy folder name found")

    return [(p.name, p) for p in dirs]


def map_type_names(folder: Path) -> set[str]:
    """Return file names in one receptor MAP pose folder."""
    return {p.name for p in folder.iterdir() if p.is_file()}


def build_layout(
    native_folder: Path,
    decoy_occs: list[tuple[str, Path]],
    map_pair: dict[int, Path],
) -> Layout:
    """
    Build the fixed eight-channel layout.

    Channel order is identical for MAP and OCC:
        e + seven ligand atom types from the selection log.

    Missing channels are represented by zero grids.
    """
    if len(ATOM_TYPE_ORDER) != N_SLOTS:
        raise AssertionError(
            f"fixed channel order must contain exactly {N_SLOTS} channels; "
            f"got {len(ATOM_TYPE_ORDER)}"
        )

    # Validate every OCC folder. Missing fixed channels are allowed;
    # extra/unexpected channels are rejected.
    folder_atom_types(native_folder)
    for _, folder in decoy_occs:
        folder_atom_types(folder)

    physical_size = N_SLOTS * GRID_SIZE
    full_size = 2 * physical_size
    aux0 = physical_size

    physical_qubits = int(round(math.log2(physical_size)))
    full_qubits = int(round(math.log2(full_size)))

    if physical_size != 2 ** 18:
        raise AssertionError(
            f"physical_size must be 2^18, got {physical_size}"
        )

    if full_size != 2 ** 19:
        raise AssertionError(
            f"full_size must be 2^19, got {full_size}"
        )

    return Layout(
        group_types=ATOM_TYPE_ORDER,
        n_slots=N_SLOTS,
        physical_size=physical_size,
        full_size=full_size,
        aux0=aux0,
        physical_qubits=physical_qubits,
        full_qubits=full_qubits,
    )


# ---------------------------------------------------------------------------
# Build physical vectors
# ---------------------------------------------------------------------------

def build_map_physical(
    folder: Path,
    layout: Layout,
) -> np.ndarray:
    """
    Build one MAP vector in the same fixed eight-channel order as OCC.

    Missing MAP channels are zero-filled.
    Extra MAP files are ignored.
    """
    present = map_type_names(folder)
    blocks = []

    for atom_type in layout.group_types:
        path = folder / atom_type
        if atom_type in present and path.is_file():
            blocks.append(read_grid(path))
        else:
            blocks.append(
                np.zeros(GRID_SIZE, dtype=np.float64)
            )

    out = np.concatenate(blocks)

    if out.size != layout.physical_size:
        raise AssertionError(
            f"MAP physical length error for {folder}: "
            f"{out.size} != {layout.physical_size}"
        )

    return out


def build_occ_physical(
    folder: Path,
    layout: Layout,
) -> np.ndarray:
    """
    Build OCC vector using the group's fixed channel order.

    If one ligand lacks a group channel, that channel is ZERO.
    """
    present = set(folder_atom_types(folder))

    blocks = []

    for atom_type in layout.group_types:
        if atom_type in present:
            blocks.append(read_grid(folder / atom_type))
        else:
            blocks.append(
                np.zeros(GRID_SIZE, dtype=np.float64)
            )

    while len(blocks) < layout.n_slots:
        blocks.append(
            np.zeros(GRID_SIZE, dtype=np.float64)
        )

    out = np.concatenate(blocks)

    if out.size != layout.physical_size:
        raise AssertionError(
            f"OCC physical length error for {folder}: "
            f"{out.size} != {layout.physical_size}"
        )

    return out


# ---------------------------------------------------------------------------
# Shift search: same 32 rotations and 512 cyclic translations
# ---------------------------------------------------------------------------

def ceil_1dp(value: float) -> float:
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"invalid nonnegative value: {value}")

    return math.ceil(value * 10.0 - 1e-12) / 10.0


def next_1dp(value: float) -> float:
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"invalid nonnegative value: {value}")

    return (
        math.floor(value * 10.0 + 1e-12) + 1
    ) / 10.0


def build_translation_shifts() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    selector = np.arange(TRANSLATION_STATES, dtype=np.int64)

    x0 = (selector >> 0) & 1
    x1 = (selector >> 1) & 1
    x2 = (selector >> 2) & 1

    y0 = (selector >> 3) & 1
    y1 = (selector >> 4) & 1
    y2 = (selector >> 5) & 1

    z0 = (selector >> 6) & 1
    z1 = (selector >> 7) & 1
    z2 = (selector >> 8) & 1

    shift_x = np.where(x2 == 1, 0, -4) + 2 * x1 + x0
    shift_y = np.where(y2 == 1, 0, -4) + 2 * y1 + y0
    shift_z = np.where(z2 == 1, 0, -4) + 2 * z1 + z0

    return (
        shift_z.astype(np.int64),
        shift_y.astype(np.int64),
        shift_x.astype(np.int64),
    )


def apply_rotation_numpy(
    vector: np.ndarray,
    rotation_selector: int,
) -> np.ndarray:
    """vector shape = (n_slots, z, y, x)."""
    output = vector

    if (rotation_selector >> 4) & 1:
        output = np.rot90(output, k=1, axes=(2, 3))

    if (rotation_selector >> 3) & 1:
        output = np.rot90(output, k=2, axes=(2, 3))

    if (rotation_selector >> 2) & 1:
        output = np.rot90(output, k=-1, axes=(1, 3))

    if (rotation_selector >> 1) & 1:
        output = np.rot90(output, k=2, axes=(1, 3))

    if rotation_selector & 1:
        output = np.rot90(output, k=1, axes=(1, 2))

    return output


def apply_rotation_torch(
    vector: object,
    rotation_selector: int,
    torch: object,
) -> object:
    output = vector

    if (rotation_selector >> 4) & 1:
        output = torch.rot90(output, k=1, dims=(2, 3))

    if (rotation_selector >> 3) & 1:
        output = torch.rot90(output, k=2, dims=(2, 3))

    if (rotation_selector >> 2) & 1:
        output = torch.rot90(output, k=-1, dims=(1, 3))

    if (rotation_selector >> 1) & 1:
        output = torch.rot90(output, k=2, dims=(1, 3))

    if rotation_selector & 1:
        output = torch.rot90(output, k=1, dims=(1, 2))

    return output.contiguous()


def select_backend(requested: str) -> Backend:
    if requested == "cpu":
        return Backend(name="numpy_cpu_fft")

    try:
        import torch
    except ImportError as exc:
        if requested == "cuda":
            raise ImportError(
                "--device cuda requires PyTorch with CUDA support"
            ) from exc
        return Backend(name="numpy_cpu_fft")

    if torch.cuda.is_available():
        return Backend(
            name="torch_cuda_fft_float64",
            torch_module=torch,
            torch_device=torch.device("cuda"),
        )

    if requested == "cuda":
        raise RuntimeError(
            "--device cuda requested, but CUDA is unavailable"
        )

    return Backend(name="numpy_cpu_fft")


def make_occ_sources(
    pdbid: str,
    native_folder: Path,
    decoy_occs: list[tuple[str, Path]],
) -> list[tuple[str, Path]]:
    """
    Native first, then 0001...1023 decoys.
    """
    return [(pdbid, native_folder)] + decoy_occs


def find_group_shift_numpy(
    pdbid: str,
    map_vecs: dict[str, np.ndarray],
    occ_sources: list[tuple[str, Path]],
    layout: Layout,
) -> tuple[
    float,
    float,
    tuple[str, str, int, int, int, int, int],
]:
    shift_z, shift_y, shift_x = build_translation_shifts()

    map_names = (
        f"{pdbid}_0",
        f"{pdbid}_1",
    )

    map_spectra = {
        name: np.fft.rfftn(
            map_vecs[name].reshape(
                layout.n_slots,
                N,
                N,
                N,
            ),
            axes=(1, 2, 3),
        )
        for name in map_names
    }

    group_min = math.inf
    best_detail = ("", "", -1, -1, 0, 0, 0)

    for occ_i, (occ_name, occ_folder) in enumerate(
        occ_sources,
        start=1,
    ):
        print(
            f"[{pdbid}] shift search OCC "
            f"{occ_i}/{len(occ_sources)}: {occ_name}"
        )

        occ = build_occ_physical(
            occ_folder,
            layout,
        ).reshape(
            layout.n_slots,
            N,
            N,
            N,
        )

        for rotation in range(ROTATION_STATES):
            rotated = apply_rotation_numpy(
                occ,
                rotation,
            )

            occ_spectrum = np.fft.rfftn(
                rotated,
                axes=(1, 2, 3),
            )

            for map_name in map_names:
                correlation = np.fft.irfftn(
                    np.sum(
                        map_spectra[map_name]
                        * np.conj(occ_spectrum),
                        axis=0,
                    ),
                    s=(N, N, N),
                    axes=(0, 1, 2),
                ).real

                values = correlation[
                    shift_z,
                    shift_y,
                    shift_x,
                ]

                local_index = int(np.argmin(values))
                local_min = float(values[local_index])

                if local_min < group_min:
                    group_min = local_min
                    best_detail = (
                        map_name,
                        occ_name,
                        rotation,
                        local_index,
                        int(shift_z[local_index]),
                        int(shift_y[local_index]),
                        int(shift_x[local_index]),
                    )

    if not math.isfinite(group_min):
        raise ArithmeticError(
            f"failed to determine minimum RAW for {pdbid}"
        )

    shift = (
        ceil_1dp(-group_min) + 0.1
        if group_min < 0.0
        else 0.0
    )

    if group_min + shift < -TOL:
        raise AssertionError(
            f"shift failed for {pdbid}: "
            f"min RAW={group_min}, shift={shift}"
        )

    return shift, group_min, best_detail


def find_group_shift_cuda(
    pdbid: str,
    map_vecs: dict[str, np.ndarray],
    occ_sources: list[tuple[str, Path]],
    layout: Layout,
    backend: Backend,
) -> tuple[
    float,
    float,
    tuple[str, str, int, int, int, int, int],
]:
    torch = backend.torch_module

    if torch is None:
        raise AssertionError("CUDA backend requested without torch")

    shift_z_np, shift_y_np, shift_x_np = (
        build_translation_shifts()
    )

    shift_z = torch.as_tensor(
        shift_z_np,
        device=backend.torch_device,
        dtype=torch.long,
    )
    shift_y = torch.as_tensor(
        shift_y_np,
        device=backend.torch_device,
        dtype=torch.long,
    )
    shift_x = torch.as_tensor(
        shift_x_np,
        device=backend.torch_device,
        dtype=torch.long,
    )

    map_names = (
        f"{pdbid}_0",
        f"{pdbid}_1",
    )

    map_spectra = {}

    for map_name in map_names:
        tensor = torch.from_numpy(
            map_vecs[map_name].reshape(
                layout.n_slots,
                N,
                N,
                N,
            )
        ).to(
            device=backend.torch_device,
            dtype=torch.float64,
        )

        map_spectra[map_name] = torch.fft.rfftn(
            tensor,
            dim=(1, 2, 3),
        )

    group_min = math.inf
    best_detail = ("", "", -1, -1, 0, 0, 0)

    with torch.no_grad():
        for occ_i, (occ_name, occ_folder) in enumerate(
            occ_sources,
            start=1,
        ):
            print(
                f"[{pdbid}] shift search OCC "
                f"{occ_i}/{len(occ_sources)}: {occ_name}"
            )

            occ_np = build_occ_physical(
                occ_folder,
                layout,
            ).reshape(
                layout.n_slots,
                N,
                N,
                N,
            )

            occ = torch.from_numpy(occ_np).to(
                device=backend.torch_device,
                dtype=torch.float64,
            )

            for rotation in range(ROTATION_STATES):
                rotated = apply_rotation_torch(
                    occ,
                    rotation,
                    torch,
                )

                occ_spectrum = torch.fft.rfftn(
                    rotated,
                    dim=(1, 2, 3),
                )

                for map_name in map_names:
                    correlation = torch.fft.irfftn(
                        torch.sum(
                            map_spectra[map_name]
                            * torch.conj(occ_spectrum),
                            dim=0,
                        ),
                        s=(N, N, N),
                        dim=(0, 1, 2),
                    ).real

                    values = correlation[
                        shift_z,
                        shift_y,
                        shift_x,
                    ]

                    local_min_tensor, local_index_tensor = (
                        torch.min(values, dim=0)
                    )

                    local_min = float(
                        local_min_tensor.item()
                    )
                    local_index = int(
                        local_index_tensor.item()
                    )

                    if local_min < group_min:
                        group_min = local_min
                        best_detail = (
                            map_name,
                            occ_name,
                            rotation,
                            local_index,
                            int(shift_z_np[local_index]),
                            int(shift_y_np[local_index]),
                            int(shift_x_np[local_index]),
                        )

    if not math.isfinite(group_min):
        raise ArithmeticError(
            f"failed to determine minimum RAW for {pdbid}"
        )

    shift = (
        ceil_1dp(-group_min) + 0.1
        if group_min < 0.0
        else 0.0
    )

    if group_min + shift < -TOL:
        raise AssertionError(
            f"shift failed for {pdbid}: "
            f"min RAW={group_min}, shift={shift}"
        )

    return shift, group_min, best_detail


def find_group_shift(
    pdbid: str,
    map_vecs: dict[str, np.ndarray],
    occ_sources: list[tuple[str, Path]],
    layout: Layout,
    backend: Backend,
):
    if backend.torch_module is None:
        return find_group_shift_numpy(
            pdbid,
            map_vecs,
            occ_sources,
            layout,
        )

    return find_group_shift_cuda(
        pdbid,
        map_vecs,
        occ_sources,
        layout,
        backend,
    )


# ---------------------------------------------------------------------------
# Group-specific normalization and output
# ---------------------------------------------------------------------------

def calculate_group_lo(
    occ_sources: list[tuple[str, Path]],
    layout: Layout,
) -> float:
    """
    One Lo shared only by this PDBID group's 1024 OCC vectors.
    """
    max_base_norm = 0.0

    for occ_name, occ_folder in occ_sources:
        vec = build_occ_physical(
            occ_folder,
            layout,
        )

        base_norm = math.sqrt(
            float(np.dot(vec, vec)) + 1.0
        )

        max_base_norm = max(
            max_base_norm,
            base_norm,
        )

    return next_1dp(max_base_norm)


def build_map_full(
    pdbid: str,
    map_vecs: dict[str, np.ndarray],
    layout: Layout,
    shift: float,
) -> tuple[dict[str, np.ndarray], float]:
    names = (
        f"{pdbid}_0",
        f"{pdbid}_1",
    )

    base_norm = {}

    for name in names:
        physical_norm2 = float(
            np.dot(
                map_vecs[name],
                map_vecs[name],
            )
        )

        base_norm[name] = math.sqrt(
            physical_norm2
            + shift * shift
        )

    Lm = next_1dp(
        max(base_norm.values())
    )

    out = {}

    for name in names:
        full = np.zeros(
            layout.full_size,
            dtype=np.float64,
        )

        full[:layout.physical_size] = (
            map_vecs[name]
        )

        # Common shift for this receptor's _0/_1 pair.
        full[layout.aux0] = shift

        pad2 = (
            Lm * Lm
            - base_norm[name] ** 2
        )

        if pad2 < -TOL:
            raise ArithmeticError(
                f"negative MAP padding for {name}: {pad2}"
            )

        # Keep MAP padding orthogonal to OCC padding.
        full[layout.aux0 + 1] = math.sqrt(
            max(pad2, 0.0)
        )

        norm = float(np.linalg.norm(full))

        if not math.isclose(
            norm,
            Lm,
            abs_tol=1e-9,
            rel_tol=0.0,
        ):
            raise AssertionError(
                f"MAP norm equalization failed: "
                f"{name}: {norm} != {Lm}"
            )

        out[name] = full / Lm

    return out, Lm


def build_one_occ_full(
    occ_folder: Path,
    layout: Layout,
    Lo: float,
) -> np.ndarray:
    vec = build_occ_physical(
        occ_folder,
        layout,
    )

    base_norm = math.sqrt(
        float(np.dot(vec, vec)) + 1.0
    )

    full = np.zeros(
        layout.full_size,
        dtype=np.float64,
    )

    full[:layout.physical_size] = vec

    # Pairs with MAP shift.
    full[layout.aux0] = 1.0

    pad2 = (
        Lo * Lo
        - base_norm * base_norm
    )

    if pad2 < -TOL:
        raise ArithmeticError(
            f"negative OCC padding for {occ_folder}: {pad2}"
        )

    # Keep OCC padding orthogonal to MAP padding.
    full[layout.aux0 + 2] = math.sqrt(
        max(pad2, 0.0)
    )

    norm = float(np.linalg.norm(full))

    if not math.isclose(
        norm,
        Lo,
        abs_tol=1e-9,
        rel_tol=0.0,
    ):
        raise AssertionError(
            f"OCC norm equalization failed: "
            f"{occ_folder}: {norm} != {Lo}"
        )

    return full / Lo


def clear_generated_files(folder: Path) -> None:
    """
    Remove only files generated by this script from an existing output folder.
    """
    folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    for p in folder.iterdir():
        if p.is_file() and (
            p.suffix == ".bin"
            or p.name == "len"
        ):
            p.unlink()


def write_group_outputs(
    output_root: Path,
    pdbid: str,
    map_full: dict[str, np.ndarray],
    occ_sources: list[tuple[str, Path]],
    layout: Layout,
    Lo: float,
    Lm: float,
    shift: float,
    group_min: float,
    best_detail: tuple[str, str, int, int, int, int, int],
) -> None:
    group_dir = output_root / pdbid
    map_dir = group_dir / "map"
    occ_dir = group_dir / "occ"

    clear_generated_files(map_dir)
    clear_generated_files(occ_dir)

    # MAP binaries
    for name in (
        f"{pdbid}_0",
        f"{pdbid}_1",
    ):
        map_full[name].astype(
            np.float64
        ).tofile(
            map_dir / f"{name}.bin"
        )

    # MAP length file: one receptor line.
    with (map_dir / "len").open(
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
            f"{pdbid} "
            f"{Lm:.12e} "
            f"{shift:.1f}\n"
        )

    # OCC binaries: native first, then decoys.
    with (occ_dir / "len").open(
        "w",
        encoding="utf-8",
    ) as len_file:
        for i, (occ_name, occ_folder) in enumerate(
            occ_sources,
            start=1,
        ):
            print(
                f"[{pdbid}] write OCC "
                f"{i}/{len(occ_sources)}: {occ_name}"
            )

            full = build_one_occ_full(
                occ_folder,
                layout,
                Lo,
            )

            full.astype(
                np.float64
            ).tofile(
                occ_dir / f"{occ_name}.bin"
            )

            len_file.write(
                f"{occ_name} "
                f"{Lo:.12e}\n"
            )

    (
        best_map,
        best_occ,
        best_rot,
        best_tra,
        best_dz,
        best_dy,
        best_dx,
    ) = best_detail

    shifted_min = group_min + shift
    min_normalized = shifted_min / (
        Lm * Lo
    )

    group_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    with (group_dir / "info").open(
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
            "allowed_order "
            + " ".join(ATOM_TYPE_ORDER)
            + "\n"
        )
        f.write(
            f"group_pdbid {pdbid}\n"
        )
        f.write(
            "group_types "
            + " ".join(layout.group_types)
            + "\n"
        )
        f.write(
            f"n_type_channels "
            f"{len(layout.group_types)}\n"
        )
        f.write(
            f"n_slots {layout.n_slots}\n"
        )
        f.write(
            f"grid_values_per_type "
            f"{GRID_SIZE}\n"
        )
        f.write(
            f"physical_values "
            f"{layout.physical_size}\n"
        )
        f.write(
            f"full_values "
            f"{layout.full_size}\n"
        )
        f.write(
            f"physical_qubits "
            f"{layout.physical_qubits}\n"
        )
        f.write(
            f"full_qubits "
            f"{layout.full_qubits}\n"
        )
        f.write(
            f"occ_count "
            f"{len(occ_sources)}\n"
        )
        f.write(
            f"decoy_count "
            f"{len(occ_sources) - 1}\n"
        )
        f.write(
            f"rotation_selectors "
            f"{ROTATION_STATES}\n"
        )
        f.write(
            f"translation_selectors "
            f"{TRANSLATION_STATES}\n"
        )
        f.write(
            "translation_range -4 +3\n"
        )
        f.write(
            "translation_boundary cyclic_modulo_32\n"
        )
        f.write(
            f"Lm {Lm:.12e}\n"
        )
        f.write(
            f"Lo {Lo:.12e}\n"
        )
        f.write(
            f"shift {shift:.1f}\n"
        )
        f.write(
            f"group_min_raw "
            f"{group_min:.12e} "
            f"{best_map} {best_occ} "
            f"rot={best_rot} "
            f"tra={best_tra} "
            f"dx={best_dx} "
            f"dy={best_dy} "
            f"dz={best_dz}\n"
        )
        f.write(
            f"global_min_normalized "
            f"{min_normalized:.12e}\n"
        )


# ---------------------------------------------------------------------------
# One receptor group
# ---------------------------------------------------------------------------

def process_group(
    pdbid: str,
    map_pair: dict[int, Path],
    native_folder: Path,
    decoy_occs: list[tuple[str, Path]],
    output_root: Path,
    backend: Backend,
) -> None:
    print()
    print("=" * 72)
    print(f"PROCESS GROUP: {pdbid}")
    print("=" * 72)

    layout = build_layout(
        native_folder,
        decoy_occs,
        map_pair,
    )

    print(
        f"[{pdbid}] types       : "
        f"{' '.join(layout.group_types)}"
    )
    print(
        f"[{pdbid}] slots       : "
        f"{layout.n_slots}"
    )
    print(
        f"[{pdbid}] full qubits : "
        f"{layout.full_qubits}"
    )

    map_vecs = {
        f"{pdbid}_0": build_map_physical(
            map_pair[0],
            layout,
        ),
        f"{pdbid}_1": build_map_physical(
            map_pair[1],
            layout,
        ),
    }

    occ_sources = make_occ_sources(
        pdbid,
        native_folder,
        decoy_occs,
    )

    if len(occ_sources) != 1024:
        raise AssertionError(
            f"{pdbid}: expected 1024 OCC vectors, "
            f"got {len(occ_sources)}"
        )

    shift, group_min, best_detail = (
        find_group_shift(
            pdbid,
            map_vecs,
            occ_sources,
            layout,
            backend,
        )
    )

    (
        best_map,
        best_occ,
        best_rot,
        best_tra,
        best_dz,
        best_dy,
        best_dx,
    ) = best_detail

    print(
        f"[shift] {pdbid}: "
        f"min RAW={group_min:.12e}, "
        f"shift={shift:.1f}, "
        f"map={best_map}, "
        f"occ={best_occ}, "
        f"rot={best_rot}, "
        f"tra={best_tra}, "
        f"xyz=({best_dx},{best_dy},{best_dz})"
    )

    Lo = calculate_group_lo(
        occ_sources,
        layout,
    )

    map_full, Lm = build_map_full(
        pdbid,
        map_vecs,
        layout,
        shift,
    )

    shifted_min = group_min + shift

    if shifted_min < -TOL:
        raise AssertionError(
            f"{pdbid}: shifted minimum is negative: "
            f"{shifted_min}"
        )

    min_normalized = shifted_min / (
        Lm * Lo
    )

    write_group_outputs(
        output_root,
        pdbid,
        map_full,
        occ_sources,
        layout,
        Lo,
        Lm,
        shift,
        group_min,
        best_detail,
    )

    print(
        f"[OK] {pdbid}"
    )
    print(
        f"     OCC vectors   : {len(occ_sources)}"
    )
    print(
        f"     Lm            : {Lm:.1f}"
    )
    print(
        f"     Lo            : {Lo:.1f}"
    )
    print(
        f"     shift         : {shift:.1f}"
    )
    print(
        f"     min normalized: "
        f"{min_normalized:.12e}"
    )
    print(
        f"     output        : "
        f"{output_root / pdbid}"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare one independent normalized MAP/OCC quantum dataset "
            "per receptor PDBID."
        )
    )

    parser.add_argument(
        "map_dir",
        type=Path,
        help=(
            "MAP root containing <PDBID>_0 and <PDBID>_1 folders"
        ),
    )

    parser.add_argument(
        "occ_root",
        type=Path,
        help=(
            "OCC parent containing occ/ and occ_decoy/"
        ),
    )

    parser.add_argument(
        "output_dir",
        type=Path,
        help=(
            "output root; creates one folder per receptor PDBID"
        ),
    )

    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help=(
            "FFT backend for shift search; "
            "auto uses CUDA PyTorch when available"
        ),
    )

    parser.add_argument(
        "--pdbid",
        default=None,
        help=(
            "optional: process only one receptor PDBID, "
            "for example --pdbid 2ybs"
        ),
    )

    return parser.parse_args()


def main() -> None:
    global ATOM_TYPE_ORDER, ATOM_TYPE_SET

    args = parse_args()

    ATOM_TYPE_ORDER = load_fixed_channel_order(TYPES_LOG)
    ATOM_TYPE_SET = set(ATOM_TYPE_ORDER)

    print(
        "fixed channels : "
        + " ".join(ATOM_TYPE_ORDER)
    )
    print(
        f"channel count  : {len(ATOM_TYPE_ORDER)}"
    )
    print(
        "physical size  : "
        f"{N_SLOTS} x {GRID_SIZE} = {N_SLOTS * GRID_SIZE} = 2^18"
    )
    print(
        "full size      : "
        f"{2 * N_SLOTS * GRID_SIZE} = 2^19"
    )

    map_dir = args.map_dir.resolve()
    occ_root = args.occ_root.resolve()
    output_dir = args.output_dir.resolve()

    native_root = occ_root / "occ"
    decoy_root = occ_root / "occ_decoy"

    if not map_dir.is_dir():
        raise FileNotFoundError(
            f"MAP directory not found: {map_dir}"
        )

    if not native_root.is_dir():
        raise FileNotFoundError(
            f"native OCC directory not found: {native_root}"
        )

    if not decoy_root.is_dir():
        raise FileNotFoundError(
            f"decoy OCC directory not found: {decoy_root}"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    maps = collect_maps(map_dir)
    native_occs = collect_native_occs(
        native_root
    )
    decoy_occs = collect_decoy_occs(
        decoy_root
    )

    missing_native = sorted(
        set(maps) - set(native_occs)
    )

    if missing_native:
        raise ValueError(
            "MAP without same-PDBID native OCC: "
            + " ".join(missing_native)
        )

    if args.pdbid is None:
        pdbids = sorted(maps)
    else:
        if args.pdbid not in maps:
            raise ValueError(
                f"PDBID not found in MAP root: "
                f"{args.pdbid}"
            )
        pdbids = [args.pdbid]

    backend = select_backend(
        args.device
    )

    print(
        f"shift backend : {backend.name}"
    )
    print(
        f"MAP PDBIDs    : {' '.join(pdbids)}"
    )
    print(
        f"decoys        : {len(decoy_occs)}"
    )
    print(
        f"output root   : {output_dir}"
    )

    for pdbid in pdbids:
        process_group(
            pdbid,
            maps[pdbid],
            native_occs[pdbid],
            decoy_occs,
            output_dir,
            backend,
        )

    print()
    print("Done")


if __name__ == "__main__":
    main()
