#!/usr/bin/env python3
"""
Unified pose-level and ligand-level quantum-threshold retrieval experiment for one or multiple receptor PDBIDs.

This program is the ligand-readout counterpart of the pose-level program.  The coherent
search space and the interaction calculation are unchanged:

    MAP selector x OCC selector x rotation selector x translation selector

with two receptor-map states, N_OCC ligand occupancies, 32 rotation-selector states,
and 512 cyclic translations.  The normalized target amplitude remains

    A0 = (ENERGY + shift) / (Lm * Lo)

and QAE/QPE plus the reversible comparator still sets FLAG according to the finite-
precision representation of A0 < A0_threshold.

In the bin-aligned experiment used here, the user supplies base evaluation-register sizes
m0 (for example 10, 14, 18, and 22).  Each base m0 defines the actual threshold

    A0_threshold = sin(pi / 2^m0).

For resolution multiples K = 1, 2, 4, and 8, the threshold is held fixed while the
actual QPE register uses m = m0 + log2(K).  Thus the comparator boundary lies exactly
at phase bin K rather than being matched approximately to an arbitrary decimal threshold.

The retrieval objective is different.  A ligand/OCC is defined as an exact-good ligand if
AT LEAST ONE of its MAP/rotation/translation configurations satisfies

    A0 < A0_threshold.

The final quantum readout therefore needs only FLAG and OCC_SEL.  MAP_SEL, ROT, and
TRANS remain coherent selector registers because they are still required to form A0 and
to decide whether a ligand has a threshold-qualified configuration; they are simply not
measured as part of the ligand-only output.

For finite-QPE resource accounting, a FLAG outcome is counted as a successful ligand
retrieval whenever its OCC selector belongs to the exact-good-ligand set, regardless of
which latent MAP/rotation/translation branch produced that FLAG outcome.  Therefore

    P(good ligand and FLAG)
        = sum_{configurations whose OCC is a good ligand} P_FLAG(configuration) / M

and

    P(good ligand | FLAG)
        = P(good ligand and FLAG) / P(FLAG).

QSEL/Grover amplification is unchanged: it amplifies the complete FLAG=1 subspace.
Under the ideal model, P(good ligand | FLAG) is preserved, so after r iterations

    P(good ligand; r)
        = P_FLAG(r) * P(good ligand | FLAG).

By default, the selected r is the value in the scanned range that MINIMIZES the total
coherent Q-operator calls needed for 95% retrieval, including r=0.  This matches the
resource objective used to decide whether QSEL is worthwhile.  If --qsel-shot-budget is
provided, the program instead selects the smallest r whose 95%-confidence shot count is
within that budget; if none is feasible, it falls back to the minimum-Q-call point.

The DATA register remains fixed at 19 qubits:

    x          5 qubits
    y          5 qubits
    z          5 qubits
    atom type  3 qubits
    auxiliary  1 qubit
    -------------------
    total     19 qubits

The eight physical channels remain fixed in this order:

    e, A, C, HD, N, NA, OA, S

The program retains the same ideal/noiseless analytical finite-QPE model, the same
symmetric QPE phase branches, and the same FFT pose-overlap calculation as the pose-level
program.  The generated Qiskit circuit remains a logical/high-level circuit: SELECT OCC,
ROT, SHIFT, SELECT U_MAP, QAE/QPE, and QSEL are logical blocks rather than fully
synthesized data-loading/oracle gates.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import dataclass, field
import math
import re
from pathlib import Path

import numpy as np


PROGRAM_VERSION = "20261008-unified-pose-ligand-bin-aligned-threshold"

# ---------------------------------------------------------------------------
# Fixed data-register definition: always 19 qubits.
# ---------------------------------------------------------------------------

CHANNEL_ORDER = ("e", "A", "C", "HD", "N", "NA", "OA", "S")
TYPE_SIZE = 8
GRID_SIZE = 32
PHYSICAL_DATA_SIZE = TYPE_SIZE * GRID_SIZE**3       # 262144 = 2^18
PHYSICAL_DATA_QUBITS = 18
AUXILIARY_QUBITS = 1
DATA_QUBITS = 19
FULL_DATA_SIZE = 2 * PHYSICAL_DATA_SIZE             # 524288 = 2^19
AUX0 = PHYSICAL_DATA_SIZE

# Pose selectors.
ROTATION_SELECTOR_QUBITS = 5
ROTATION_STATES = 32
UNIQUE_ROTATIONS = 24
TRANSLATION_SELECTOR_QUBITS = 9
TRANSLATION_STATES = 512
POSES_PER_PAIR = ROTATION_STATES * TRANSLATION_STATES  # 16384 = 2^14

# Requested nested OCC experiments.
OCC_COUNTS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)
EXPECTED_DECOYS = 1023

DEFAULT_AMPLITUDE_THRESHOLD = 1e-5  # retained only for legacy helper functions
DEFAULT_BASE_EVAL_QUBITS = (10, 14, 18, 22)
DEFAULT_RESOLUTION_MULTIPLES = (1, 2, 4, 8)
DEFAULT_NORM_ATOL = 1e-8
TOLERANCE = 1e-10

MAP_NAME_RE = re.compile(r"^(?P<pdbid>.+)_(?P<index>[01])$")
DECOY_RE = re.compile(r"^(?P<index>\d{4})_.+\.bin$")


@dataclass(frozen=True)
class MapData:
    name: str
    selector: int
    amplitudes: np.ndarray
    physical: np.ndarray


@dataclass(frozen=True)
class OccEntry:
    selector: int
    name: str
    path: Path
    source: str


@dataclass(frozen=True)
class PoseColumns:
    pose_selector: np.ndarray
    rotation_selector: np.ndarray
    translation_selector: np.ndarray
    shift_z: np.ndarray
    shift_y: np.ndarray
    shift_x: np.ndarray
    shift_z_512: np.ndarray
    shift_y_512: np.ndarray
    shift_x_512: np.ndarray


@dataclass(frozen=True)
class ComparatorSpec:
    modulus: int
    low_limit: int
    high_limit: int


@dataclass(frozen=True)
class Backend:
    name: str
    torch_module: object | None = None
    torch_device: object | None = None


@dataclass
class CumulativeStats:
    states: int = 0
    exact_good_configs: int = 0
    good_ligands: int = 0
    sum_flag: float = 0.0
    sum_flag_good_ligand: float = 0.0
    min_energy: float = math.inf
    min_shifted: float = math.inf
    max_a0: float = -math.inf



@dataclass
class PoseCumulativeStats:
    states: int = 0
    exact_good: int = 0
    sum_flag: float = 0.0
    sum_flag_true_good: float = 0.0
    min_energy: float = math.inf
    min_shifted: float = math.inf
    max_a0: float = -math.inf


@dataclass(frozen=True)
class PoseGroverPoint:
    occ_count: int
    grover_iterations: int
    amplified_flag_probability: float
    true_good_probability: float
    shots_95: int | None
    shots_99: int | None
    q_calls_per_shot: int
    total_q_calls_95: int | None
    total_q_calls_99: int | None


@dataclass(frozen=True)
class PoseScaleResult:
    occ_count: int
    occ_selector_qubits: int
    total_selector_qubits: int
    search_states: int
    exact_good_states: int
    exact_good_fraction: float
    initial_flag_probability: float
    true_good_probability_qae: float
    true_good_given_flag: float
    grover_max: int
    theoretical_opt_grover: int
    grover_iterations: int
    amplified_flag_probability: float
    true_good_probability_qsel: float
    qae_shots_95: int | None
    qae_shots_99: int | None
    qsel_shots_95: int | None
    qsel_shots_99: int | None
    qsel_shot_budget: int | None
    qsel_shot_budget_met: bool | None
    q_calls_per_qpe: int
    q_calls_per_threshold_oracle: int
    qsel_q_calls_per_shot: int
    qae_total_q_calls_95: int | None
    qae_total_q_calls_99: int | None
    qsel_total_q_calls_95: int | None
    qsel_total_q_calls_99: int | None
    logical_qubits: int
    work_qubits: int
    min_energy: float
    max_a0: float


@dataclass(frozen=True)
class LigandRecord:
    occ_selector: int
    occ_name: str
    source: str
    exact_good_configs: int
    min_energy: float
    min_a0: float
    best_map_selector: int
    best_pose_selector: int
    best_rotation_selector: int
    best_translation_selector: int
    best_shift_z: int
    best_shift_y: int
    best_shift_x: int
    sum_flag: float
    p_flag_given_occ: float


@dataclass(frozen=True)
class GroverPoint:
    occ_count: int
    grover_iterations: int
    amplified_flag_probability: float
    good_ligand_probability: float
    shots_95: int | None
    shots_99: int | None
    q_calls_per_shot: int
    total_q_calls_95: int | None
    total_q_calls_99: int | None


@dataclass(frozen=True)
class ScaleResult:
    occ_count: int
    occ_selector_qubits: int
    total_selector_qubits: int
    search_states: int
    exact_good_config_states: int
    exact_good_config_fraction: float
    good_ligands: int
    good_ligand_fraction: float
    initial_flag_probability: float
    good_ligand_probability_qae: float
    good_ligand_given_flag: float
    grover_max: int
    theoretical_opt_grover: int
    grover_iterations: int
    amplified_flag_probability: float
    good_ligand_probability_qsel: float
    qae_shots_95: int | None
    qae_shots_99: int | None
    qsel_shots_95: int | None
    qsel_shots_99: int | None
    qsel_shot_budget: int | None
    qsel_shot_budget_met: bool | None
    q_calls_per_qpe: int
    q_calls_per_threshold_oracle: int
    qsel_q_calls_per_shot: int
    qae_total_q_calls_95: int | None
    qae_total_q_calls_99: int | None
    qsel_total_q_calls_95: int | None
    qsel_total_q_calls_99: int | None
    logical_qubits: int
    work_qubits: int
    min_energy: float
    max_a0: float


@dataclass
class ExperimentConfig:
    amplitude_threshold: float
    energy_threshold: float
    eval_qubits: int
    resolution_multiple: int
    comparator: ComparatorSpec
    work_qubits: int
    output_root: Path
    # Ligand-only retrieval statistics.
    stats: CumulativeStats = field(default_factory=CumulativeStats)
    results: list[ScaleResult] = field(default_factory=list)
    grover_rows: list[GroverPoint] = field(default_factory=list)
    ligand_records: list[LigandRecord] = field(default_factory=list)

    # Pose/configuration retrieval statistics from the SAME A0/FLAG calculation.
    pose_stats: PoseCumulativeStats = field(default_factory=PoseCumulativeStats)
    pose_results: list[PoseScaleResult] = field(default_factory=list)
    pose_grover_rows: list[PoseGroverPoint] = field(default_factory=list)
    temp0: Path | None = None
    temp1: Path | None = None


# ---------------------------------------------------------------------------
# Input / metadata
# ---------------------------------------------------------------------------


def parse_map_name(name: str) -> tuple[str, int]:
    match = MAP_NAME_RE.match(name)
    if match is None:
        raise ValueError(
            f"invalid MAP name {name!r}; expected <PDBID>_0 or <PDBID>_1"
        )
    return match.group("pdbid"), int(match.group("index"))


def read_map_len(path: Path) -> tuple[str, float, float]:
    """Read exactly one group record: PDBID Lm shift."""
    if not path.is_file():
        raise FileNotFoundError(f"MAP len not found: {path}")

    rows: list[tuple[str, float, float]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line_number, line in enumerate(handle, start=1):
            fields = line.split()
            if not fields:
                continue
            if len(fields) != 3:
                raise ValueError(
                    f"invalid MAP len row at {path}:{line_number}; "
                    "expected: PDBID Lm shift"
                )
            pdbid = fields[0]
            try:
                Lm = float(fields[1])
                shift = float(fields[2])
            except ValueError as exc:
                raise ValueError(
                    f"non-numeric MAP len value at {path}:{line_number}"
                ) from exc

            if not math.isfinite(Lm) or Lm <= 0.0:
                raise ValueError(f"invalid Lm for {pdbid}: {Lm}")
            if not math.isfinite(shift) or shift < 0.0:
                raise ValueError(f"invalid shift for {pdbid}: {shift}")
            rows.append((pdbid, Lm, shift))

    if len(rows) != 1:
        raise ValueError(
            f"{path} must contain exactly one receptor PDBID; found {len(rows)}"
        )
    return rows[0]


def read_occ_len(path: Path) -> dict[str, float]:
    """Read OCC len rows: OCC_ID Lo."""
    if not path.is_file():
        raise FileNotFoundError(f"OCC len not found: {path}")

    result: dict[str, float] = {}
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line_number, line in enumerate(handle, start=1):
            fields = line.split()
            if not fields:
                continue
            if len(fields) != 2:
                raise ValueError(
                    f"invalid OCC len row at {path}:{line_number}; "
                    "expected: OCC_ID Lo"
                )
            name = fields[0]
            if name in result:
                raise ValueError(f"duplicate OCC len entry: {name}")
            try:
                Lo = float(fields[1])
            except ValueError as exc:
                raise ValueError(
                    f"non-numeric OCC Lo at {path}:{line_number}"
                ) from exc
            if not math.isfinite(Lo) or Lo <= 0.0:
                raise ValueError(f"invalid Lo for {name}: {Lo}")
            result[name] = Lo

    if not result:
        raise ValueError(f"no OCC len records found: {path}")
    return result


def read_full_vector(path: Path, norm_atol: float) -> np.ndarray:
    """Read one fixed 19-qubit normalized float64 state."""
    if not path.is_file():
        raise FileNotFoundError(f"vector not found: {path}")

    full = np.fromfile(path, dtype=np.float64)
    if full.size != FULL_DATA_SIZE:
        raise ValueError(
            f"wrong number of float64 values in {path}: "
            f"got {full.size}, expected {FULL_DATA_SIZE} (=2^19)"
        )
    if not np.all(np.isfinite(full)):
        raise ValueError(f"NaN or Inf found in: {path}")

    norm = float(np.linalg.norm(full))
    if not np.isclose(norm, 1.0, atol=norm_atol, rtol=0.0):
        raise ValueError(
            f"19-qubit vector norm is not 1 within atol={norm_atol}: "
            f"{path}, norm={norm:.16e}"
        )
    return full


def load_maps(
    map_dir: Path,
    pdbid: str,
    Lm: float,
    shift: float,
    norm_atol: float,
) -> list[MapData]:
    maps: list[MapData] = []

    for selector in (0, 1):
        path = map_dir / f"{pdbid}_{selector}.bin"
        full = read_full_vector(path, norm_atol)

        encoded_shift = float(full[AUX0] * Lm)
        if not np.isclose(encoded_shift, shift, atol=1e-8, rtol=1e-10):
            raise ValueError(
                f"MAP auxiliary shift mismatch for {path}: "
                f"bin*Lm={encoded_shift:.16e}, len shift={shift:.16e}"
            )

        # MAP norm padding lives at AUX0+1; OCC padding slot must be zero.
        if abs(float(full[AUX0 + 2])) > 1e-10:
            raise ValueError(
                f"MAP OCC-padding slot is not zero in {path}: "
                f"{full[AUX0 + 2]:.16e}"
            )

        maps.append(
            MapData(
                name=f"{pdbid}_{selector}",
                selector=selector,
                amplitudes=full,
                physical=np.ascontiguousarray(full[:PHYSICAL_DATA_SIZE]),
            )
        )

    return maps


def collect_occ_entries(occ_dir: Path, pdbid: str) -> list[OccEntry]:
    """
    Fixed selector mapping:
        0    = native <PDBID>.bin
        1    = 0001_*.bin
        2    = 0002_*.bin
        ...
        1023 = 1023_*.bin
    """
    native_path = occ_dir / f"{pdbid}.bin"
    if not native_path.is_file():
        raise FileNotFoundError(
            f"native OCC must exist and is always used: {native_path}"
        )

    by_index: dict[int, Path] = {}
    unexpected: list[str] = []

    for path in sorted(occ_dir.glob("*.bin"), key=lambda p: p.name):
        if path == native_path:
            continue

        match = DECOY_RE.match(path.name)
        if match is None:
            unexpected.append(path.name)
            continue

        index = int(match.group("index"))
        if index in by_index:
            raise ValueError(
                f"duplicate decoy numeric prefix {index:04d}: "
                f"{by_index[index].name}, {path.name}"
            )
        by_index[index] = path

    if unexpected:
        raise ValueError(
            "unexpected OCC .bin filename(s); expected native <PDBID>.bin "
            "or NNNN_*.bin: " + ", ".join(unexpected)
        )

    expected = set(range(1, EXPECTED_DECOYS + 1))
    found = set(by_index)
    missing = sorted(expected - found)
    extra = sorted(found - expected)

    if missing or extra:
        messages = []
        if missing:
            messages.append(
                "missing decoy prefixes: "
                + " ".join(f"{x:04d}" for x in missing[:30])
                + (" ..." if len(missing) > 30 else "")
            )
        if extra:
            messages.append(
                "unexpected decoy prefixes: "
                + " ".join(f"{x:04d}" for x in extra)
            )
        raise ValueError("\n".join(messages))

    entries = [
        OccEntry(
            selector=0,
            name=pdbid,
            path=native_path,
            source="native",
        )
    ]

    for decoy_index in range(1, EXPECTED_DECOYS + 1):
        path = by_index[decoy_index]
        entries.append(
            OccEntry(
                selector=decoy_index,
                name=path.stem,
                path=path,
                source="decoy",
            )
        )

    if len(entries) != 1024:
        raise AssertionError(f"expected 1024 OCC entries; got {len(entries)}")

    return entries


def validate_occ_lengths(
    entries: list[OccEntry],
    occ_lengths: dict[str, float],
) -> float:
    """All 1024 OCC states in one receptor group must share one Lo."""
    values: list[float] = []

    for entry in entries:
        if entry.name not in occ_lengths:
            raise KeyError(f"no OCC len entry for {entry.name}")
        values.append(float(occ_lengths[entry.name]))

    Lo = values[0]
    for entry, value in zip(entries[1:], values[1:]):
        if not np.isclose(value, Lo, atol=1e-10, rtol=1e-12):
            raise ValueError(
                "all 1024 OCC vectors in one receptor group must share one Lo; "
                f"{entries[0].name}={Lo}, {entry.name}={value}"
            )
    return Lo


def validate_occ_auxiliary(
    full: np.ndarray,
    path: Path,
    Lo: float,
) -> None:
    aux_one = float(full[AUX0] * Lo)
    if not np.isclose(aux_one, 1.0, atol=1e-8, rtol=1e-10):
        raise ValueError(
            f"OCC auxiliary first amplitude mismatch for {path}: "
            f"bin*Lo={aux_one:.16e}, expected 1"
        )

    # OCC norm padding lives at AUX0+2; MAP padding slot must be zero.
    if abs(float(full[AUX0 + 1])) > 1e-10:
        raise ValueError(
            f"OCC MAP-padding slot is not zero in {path}: "
            f"{full[AUX0 + 1]:.16e}"
        )


# ---------------------------------------------------------------------------
# Pose geometry / FFT
# ---------------------------------------------------------------------------


def build_pose_columns() -> PoseColumns:
    translation_selector = np.arange(TRANSLATION_STATES, dtype=np.int64)

    x0 = (translation_selector >> 0) & 1
    x1 = (translation_selector >> 1) & 1
    x2 = (translation_selector >> 2) & 1

    y0 = (translation_selector >> 3) & 1
    y1 = (translation_selector >> 4) & 1
    y2 = (translation_selector >> 5) & 1

    z0 = (translation_selector >> 6) & 1
    z1 = (translation_selector >> 7) & 1
    z2 = (translation_selector >> 8) & 1

    shift_x_512 = np.where(x2 == 1, 0, -4) + 2 * x1 + x0
    shift_y_512 = np.where(y2 == 1, 0, -4) + 2 * y1 + y0
    shift_z_512 = np.where(z2 == 1, 0, -4) + 2 * z1 + z0

    pose_selector = np.arange(POSES_PER_PAIR, dtype=np.int64)
    rotation = pose_selector & 31
    translation = pose_selector >> 5

    return PoseColumns(
        pose_selector=pose_selector,
        rotation_selector=rotation,
        translation_selector=translation,
        shift_z=shift_z_512[translation],
        shift_y=shift_y_512[translation],
        shift_x=shift_x_512[translation],
        shift_z_512=shift_z_512,
        shift_y_512=shift_y_512,
        shift_x_512=shift_x_512,
    )


def apply_rotation_numpy(vector: np.ndarray, rotation_selector: int) -> np.ndarray:
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


def apply_rotation_torch(vector: object, rotation_selector: int, torch: object) -> object:
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
        raise RuntimeError("--device cuda requested, but CUDA is unavailable")

    return Backend(name="numpy_cpu_fft")



def test_rotation_output(output_dir: Path, device: str) -> None:
    """Test the ORIGINAL NumPy/Torch ROT functions, without running the experiment."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # One identifiable input (type, z, y, x): a center and three directed axes.
    marks = (10, 20, 30, 40)
    locations = ((15, 15, 15), (16, 15, 15), (15, 16, 15), (15, 15, 16))
    test_input = np.zeros((TYPE_SIZE, GRID_SIZE, GRID_SIZE, GRID_SIZE), dtype=np.float64)
    for mark, (z, y, x) in zip(marks, locations):
        test_input[0, z, y, x] = mark

    with (output_dir / "rot_input.txt").open("w", encoding="utf-8") as handle:
        handle.write(f"{'Marker':>8} {'z':>6} {'y':>6} {'x':>6}\n")
        for mark, (z, y, x) in zip(marks, locations):
            handle.write(f"{mark:8d} {z:6d} {y:6d} {x:6d}\n")

    backend = select_backend(device)
    torch_input = None
    if backend.torch_module is not None:
        torch = backend.torch_module
        torch_input = torch.from_numpy(test_input).to(
            device=backend.torch_device, dtype=torch.float64
        )

    seen: dict[tuple, tuple[int, np.ndarray]] = {}
    repeats: list[tuple[int, int]] = []
    with (output_dir / "rot_outputs.txt").open("w", encoding="utf-8") as out, \
         (output_dir / "rot_summary.txt").open("w", encoding="utf-8") as summary:
        out.write(f"{'ROT':>5} {'Marker':>8} {'z':>6} {'y':>6} {'x':>6}\n")
        summary.write(f"{'ROT':>5} {'SameAs':>8} {'UniqueID':>9} {'Det':>6}\n")
        for selector in range(ROTATION_STATES):
            numpy_result = apply_rotation_numpy(test_input, selector)
            if torch_input is None:
                result = numpy_result
            else:
                result = apply_rotation_torch(
                    torch_input, selector, backend.torch_module
                ).cpu().numpy()
                if not np.array_equal(result, numpy_result):
                    raise AssertionError(f"ROT {selector:02d}: CPU/CUDA outputs differ")

            points = []
            for mark in marks:
                coords = np.argwhere(result[0] == mark)
                if coords.shape != (1, 3):
                    raise AssertionError(f"ROT {selector:02d}: marker {mark} not unique")
                z, y, x = (int(v) for v in coords[0])
                points.append((z, y, x))
                out.write(f"{selector:5d} {mark:8d} {z:6d} {y:6d} {x:6d}\n")

            out.write("\n")

            # Columns are the images of the original +z, +y, +x basis vectors.
            basis = np.column_stack([
                np.subtract(points[1], points[0]),
                np.subtract(points[2], points[0]),
                np.subtract(points[3], points[0]),
            ])
            determinant = int(round(float(np.linalg.det(basis))))
            if determinant != 1:
                raise AssertionError(f"ROT {selector:02d}: determinant != +1")

            fingerprint = tuple(points)
            if fingerprint in seen:
                first, first_result = seen[fingerprint]
                if not np.array_equal(result, first_result):
                    raise AssertionError("Same marker positions but different full data")
                repeats.append((selector, first))
                unique_id = list(seen).index(fingerprint)
            else:
                first = selector
                unique_id = len(seen)
                seen[fingerprint] = (selector, result.copy())
            summary.write(f"{selector:5d} {first:8d} {unique_id:9d} {determinant:6d}\n")

    print(f"ROT backend: {backend.name}")
    print(f"ROT selectors: {ROTATION_STATES}")
    print(f"Distinct rotations: {len(seen)}")
    print(f"Repeated selectors: {len(repeats)}")
    for selector, first in repeats:
        print(f"  ROT {selector:02d} = ROT {first:02d}")
    if len(seen) != UNIQUE_ROTATIONS or len(repeats) != ROTATION_STATES - UNIQUE_ROTATIONS:
        raise AssertionError("Unexpected number of distinct rotations")
    print(f"PASS: {ROTATION_STATES} selectors, {UNIQUE_ROTATIONS} distinct rotations")
    print(f"Input and all 32 outputs: {output_dir}")



def test_input_output(
    input_dir: Path,
    output_dir: Path,
    max_occ: int,
    norm_atol: float,
) -> None:
    """Audit the input used by this program, without modifying any amplitudes."""
    receptor_dirs = discover_receptor_dirs(input_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "input_check.txt"
    lines: list[str] = []
    errors: list[str] = []
    n_maps = 0
    n_occ = 0

    lines.append("INPUT DATA VALIDATION (original energy.py input functions)")
    lines.append(f"Input root : {input_dir.resolve()}")
    lines.append(f"Norm atol  : {norm_atol:.3e}")
    lines.append(f"OCC tested : first {max_occ} of 1024 for each receptor")
    lines.append("All values are read from disk; NO renormalization is performed.")
    lines.append("Norm = ||entire 19-qubit vector||; Phys2 = sum(physical amplitudes^2).")
    lines.append("AuxValue = bin[AUX0] * Lm (MAP) or bin[AUX0] * Lo (OCC).")
    lines.append("NormPad = bin[AUX0+1] (MAP) or bin[AUX0+2] (OCC).")
    lines.append("AuxTail = max(abs(bin[AUX0+3:])) (expected to be zero).")

    for receptor_dir in receptor_dirs:
        lines.append("")
        lines.append("=" * 120)
        lines.append(f"RECEPTOR {receptor_dir.name}")
        map_dir = receptor_dir / "map"
        occ_dir = receptor_dir / "occ"
        lines.append(f"MAP len file : {map_dir / 'len'}")
        lines.append(f"OCC len file : {occ_dir / 'len'}")
        try:
            pdbid, Lm, shift = read_map_len(map_dir / "len")
            occ_lengths = read_occ_len(occ_dir / "len")
            occ_entries = collect_occ_entries(occ_dir, pdbid)
            Lo = validate_occ_lengths(occ_entries, occ_lengths)
        except (OSError, ValueError, KeyError, AssertionError) as exc:
            message = f"{receptor_dir.name}: metadata/file mapping: {exc}"
            errors.append(message)
            lines.append(f"FAIL: {message}")
            continue

        lines.append(f"MAP metadata  PDBID={pdbid}  Lm={Lm:.16e}  shift={shift:.16e}")
        lines.append(f"OCC metadata  entries={len(occ_lengths)}  Lo={Lo:.16e}")
        if receptor_dir.name != pdbid:
            lines.append(f"NOTE: directory name {receptor_dir.name!r} differs from MAP PDBID {pdbid!r}")
        extra_names = sorted(set(occ_lengths) - {entry.name for entry in occ_entries})
        if extra_names:
            message = f"{pdbid}: OCC len has {len(extra_names)} extra name(s): {', '.join(extra_names[:5])}"
            errors.append(message)
            lines.append(f"FAIL: {message}")

        lines.append("")
        lines.append("MAP BIN: original load_maps() / read_full_vector()")
        lines.append(f"{'Sel':>3} {'MAP_ID':<20} {'Norm':>15} {'Phys2':>15} {'AuxValue':>15} {'NormPad':>12} {'AuxTail':>11} {'Status':>6}")
        try:
            maps = load_maps(map_dir, pdbid, Lm, shift, norm_atol)
        except (OSError, ValueError, KeyError, ArithmeticError) as exc:
            message = f"{pdbid}: MAP load: {exc}"
            errors.append(message)
            lines.append(f"FAIL: {message}")
        else:
            for receptor in maps:
                full = receptor.amplitudes
                phys2 = float(np.dot(receptor.physical, receptor.physical))
                norm = float(np.linalg.norm(full))
                aux_value = float(full[AUX0] * Lm)
                norm_pad = float(full[AUX0 + 1])
                tail = float(np.max(np.abs(full[AUX0 + 3:])))
                raw_physical_unchanged = np.array_equal(
                    receptor.physical, full[:PHYSICAL_DATA_SIZE]
                )
                ok = tail <= TOLERANCE and raw_physical_unchanged
                if not ok:
                    message = f"{pdbid}: MAP {receptor.name}: nonzero auxiliary tail or changed physical amplitudes"
                    errors.append(message)
                lines.append(
                    f"{receptor.selector:3d} {receptor.name:<20} {norm:15.9e} "
                    f"{phys2:15.9e} {aux_value:15.9e} {norm_pad:12.5e} {tail:11.3e} "
                    f"{'PASS' if ok else 'FAIL':>6}"
                )
                n_maps += 1
            lines.append(f"Expected for each MAP: Norm=1, AuxValue=shift={shift:.9e}, AuxTail=0")
            lines.append("load_maps() already checked shift encoding and MAP reserved padding.")

        lines.append("")
        lines.append("OCC BIN: original read_full_vector() / validate_occ_auxiliary()")
        lines.append(f"{'Sel':>4} {'OCC_ID':<31} {'Lo':>13} {'Norm':>13} {'Phys2':>13} {'AuxValue':>13} {'NormPad':>11} {'Tail':>9} {'Status':>6}")
        for entry in occ_entries[:max_occ]:
            current_lo = occ_lengths[entry.name]
            try:
                full = read_full_vector(entry.path, norm_atol)
                validate_occ_auxiliary(full, entry.path, Lo)
                norm = float(np.linalg.norm(full))
                physical = full[:PHYSICAL_DATA_SIZE]
                phys2 = float(np.dot(physical, physical))
                aux_value = float(full[AUX0] * current_lo)
                norm_pad = float(full[AUX0 + 2])
                tail = float(np.max(np.abs(full[AUX0 + 3:])))
                ok = tail <= TOLERANCE
                if not ok:
                    errors.append(f"{pdbid}: OCC {entry.name}: unexpected nonzero auxiliary tail={tail:.3e}")
                lines.append(
                    f"{entry.selector:4d} {entry.name:<31} {current_lo:13.6e} "
                    f"{norm:13.6e} {phys2:13.6e} {aux_value:13.6e} "
                    f"{norm_pad:11.4e} {tail:9.1e} {'PASS' if ok else 'FAIL':>6}"
                )
                n_occ += 1
            except (OSError, ValueError, KeyError, ArithmeticError) as exc:
                message = f"{pdbid}: OCC {entry.name} ({entry.path}): {exc}"
                errors.append(message)
                lines.append(f"{entry.selector:4d} {entry.name:<31} {'FAIL':>90}")
                lines.append(f"    ERROR: {exc}")
        lines.append(f"Expected for each OCC: Norm=1, AuxValue=1, Tail=0, common Lo={Lo:.9e}")
        lines.append("validate_occ_auxiliary() already checked encoded 1/Lo and OCC reserved padding.")

    summary = [
        "CHECK SUMMARY",
        f"Result    : {'PASS' if not errors else 'FAIL'}",
        f"Receptors : {len(receptor_dirs)}",
        f"MAP tested: {n_maps}",
        f"OCC tested: {n_occ}",
        f"Errors    : {len(errors)}",
    ]
    if errors:
        summary.append("")
        summary.append("ERRORS")
        summary.extend(f"- {message}" for message in errors)
    report_path.write_text("\n".join(summary + [""] + lines) + "\n", encoding="utf-8")
    print("\n".join(summary[:6]))
    print(f"Report    : {report_path}")
    if errors:
        raise AssertionError(f"input validation failed ({len(errors)} error(s)); see {report_path}")

def prepare_map_spectra(maps: list[MapData], backend: Backend) -> list[object]:
    spectra: list[object] = []

    for receptor in maps:
        shaped = receptor.physical.reshape(
            TYPE_SIZE,
            GRID_SIZE,
            GRID_SIZE,
            GRID_SIZE,
        )

        if backend.torch_module is None:
            spectra.append(np.fft.rfftn(shaped, axes=(1, 2, 3)))
        else:
            torch = backend.torch_module
            tensor = torch.from_numpy(shaped).to(
                device=backend.torch_device,
                dtype=torch.float64,
            )
            spectra.append(torch.fft.rfftn(tensor, dim=(1, 2, 3)))

    return spectra


def compute_occ_pose_overlaps_numpy(
    map_spectra: list[np.ndarray],
    occ_physical: np.ndarray,
    columns: PoseColumns,
) -> np.ndarray:
    """Return shape (2, 16384) normalized physical overlaps."""
    occ = occ_physical.reshape(TYPE_SIZE, GRID_SIZE, GRID_SIZE, GRID_SIZE)
    output = np.empty((2, POSES_PER_PAIR), dtype=np.float64)

    for rotation in range(ROTATION_STATES):
        rotated = apply_rotation_numpy(occ, rotation)
        occ_spectrum = np.fft.rfftn(rotated, axes=(1, 2, 3))

        for map_selector in (0, 1):
            correlation = np.fft.irfftn(
                np.sum(
                    map_spectra[map_selector] * np.conj(occ_spectrum),
                    axis=0,
                ),
                s=(GRID_SIZE, GRID_SIZE, GRID_SIZE),
                axes=(0, 1, 2),
            ).real

            output[map_selector, rotation::ROTATION_STATES] = correlation[
                columns.shift_z_512,
                columns.shift_y_512,
                columns.shift_x_512,
            ]

    return output


def compute_occ_pose_overlaps_torch(
    map_spectra: list[object],
    occ_physical: np.ndarray,
    columns: PoseColumns,
    backend: Backend,
) -> np.ndarray:
    torch = backend.torch_module
    if torch is None:
        raise AssertionError("Torch backend requested without torch")

    occ = torch.from_numpy(
        occ_physical.reshape(TYPE_SIZE, GRID_SIZE, GRID_SIZE, GRID_SIZE)
    ).to(
        device=backend.torch_device,
        dtype=torch.float64,
    )

    shift_z = torch.as_tensor(
        columns.shift_z_512,
        device=backend.torch_device,
        dtype=torch.long,
    )
    shift_y = torch.as_tensor(
        columns.shift_y_512,
        device=backend.torch_device,
        dtype=torch.long,
    )
    shift_x = torch.as_tensor(
        columns.shift_x_512,
        device=backend.torch_device,
        dtype=torch.long,
    )

    output = np.empty((2, POSES_PER_PAIR), dtype=np.float64)

    with torch.no_grad():
        for rotation in range(ROTATION_STATES):
            rotated = apply_rotation_torch(occ, rotation, torch)
            occ_spectrum = torch.fft.rfftn(rotated, dim=(1, 2, 3))

            for map_selector in (0, 1):
                correlation = torch.fft.irfftn(
                    torch.sum(
                        map_spectra[map_selector] * torch.conj(occ_spectrum),
                        dim=0,
                    ),
                    s=(GRID_SIZE, GRID_SIZE, GRID_SIZE),
                    dim=(0, 1, 2),
                ).real

                output[map_selector, rotation::ROTATION_STATES] = (
                    correlation[shift_z, shift_y, shift_x]
                    .detach()
                    .cpu()
                    .numpy()
                )

    return output


def compute_occ_pose_overlaps(
    map_spectra: list[object],
    occ_physical: np.ndarray,
    columns: PoseColumns,
    backend: Backend,
) -> np.ndarray:
    if backend.torch_module is None:
        overlaps = compute_occ_pose_overlaps_numpy(
            map_spectra,
            occ_physical,
            columns,
        )
    else:
        overlaps = compute_occ_pose_overlaps_torch(
            map_spectra,
            occ_physical,
            columns,
            backend,
        )

    maximum = float(np.max(np.abs(overlaps)))
    if maximum > 1.0 + 1e-9:
        raise ArithmeticError(
            f"normalized physical overlap exceeds one: max abs={maximum:.16e}"
        )
    return np.clip(overlaps, -1.0, 1.0)


# ---------------------------------------------------------------------------
# QAE/QPE threshold model
# ---------------------------------------------------------------------------


def comparator_spec(amplitude_threshold: float, eval_qubits: int) -> ComparatorSpec:
    """
    Define the two-sided QPE phase acceptance region.

    For amplitude A0, QAE/QPE contains the symmetric eigenphases

        +theta  and  -theta mod 1,

    where theta = asin(A0) / pi.

    Therefore A0 < threshold is represented by two phase regions:

        y < low_limit
        OR
        y >= high_limit

    with the upper region being the mirror image of the nonzero lower bins.
    """
    if not 0.0 < amplitude_threshold < 1.0:
        raise ValueError("amplitude threshold must be strictly between 0 and 1")
    if eval_qubits < 1:
        raise ValueError("evaluation qubits must be at least 1")

    modulus = 1 << eval_qubits
    cutoff = modulus * math.asin(amplitude_threshold) / math.pi

    # Strict lower-branch condition y < cutoff.
    low_limit = int(math.ceil(math.nextafter(cutoff, -math.inf)))
    low_limit = max(1, min(low_limit, modulus // 2))

    # Strict upper-branch condition y > modulus - cutoff.
    # IntegerComparator implements >=, hence:
    #
    #     y >= modulus - low_limit + 1
    #
    # When low_limit == 1, high_limit == modulus and there is no distinct
    # high-side integer bin; bin 0 already represents the phase wraparound.
    high_limit = modulus - low_limit + 1

    return ComparatorSpec(
        modulus=modulus,
        low_limit=low_limit,
        high_limit=high_limit,
    )


def qpe_probability_on_bins(
    amplitudes: np.ndarray,
    bins: np.ndarray,
    modulus: int,
    *,
    max_matrix_elements: int = 2_000_000,
) -> np.ndarray:
    """Analytical finite-QPE probability mass on selected phase bins."""
    amplitudes = np.asarray(amplitudes, dtype=np.float64)
    bins = np.asarray(bins, dtype=np.int64)

    if bins.size == 0:
        return np.zeros_like(amplitudes)

    accepted = bins.astype(np.float64) / float(modulus)
    rows_per_chunk = max(
        1,
        min(250_000, max_matrix_elements // max(1, bins.size)),
    )
    output = np.empty_like(amplitudes)

    for start in range(0, amplitudes.size, rows_per_chunk):
        stop = min(start + rows_per_chunk, amplitudes.size)

        phase = np.arcsin(
            np.clip(amplitudes[start:stop], 0.0, 1.0)
        ) / math.pi
        negative_phase = (-phase) % 1.0

        positive_delta = phase[:, None] - accepted[None, :]
        negative_delta = negative_phase[:, None] - accepted[None, :]

        positive_ratio = (
            np.sinc(modulus * positive_delta)
            / np.sinc(positive_delta)
        )
        negative_ratio = (
            np.sinc(modulus * negative_delta)
            / np.sinc(negative_delta)
        )

        output[start:stop] = 0.5 * (
            np.sum(np.square(np.abs(positive_ratio)), axis=1)
            + np.sum(np.square(np.abs(negative_ratio)), axis=1)
        )

    return np.clip(output, 0.0, 1.0)


def qae_flag_probability(
    amplitudes: np.ndarray,
    spec: ComparatorSpec,
) -> np.ndarray:
    """
    Probability that the two-sided comparator sets FLAG.

    Accepted QPE bins are

        y < low_limit
        OR
        y >= high_limit.

    The QAE phase distribution used by qpe_probability_on_bins() is the
    equal mixture of the +theta and -theta eigenphase branches and is exactly
    symmetric under y -> modulus-y.  Therefore

        P(low OR high) = 2 * P(low) - P(y=0),

    because bin 0 is shared by the two mirrored regions.

    This is mathematically equivalent to explicitly summing both branches,
    while avoiding an approximately two-fold increase in the expensive
    finite-QPE bin calculation.
    """
    amplitudes = np.asarray(amplitudes, dtype=np.float64)

    minimum = float(np.min(amplitudes))
    if minimum < -TOLERANCE:
        raise ArithmeticError(
            f"A0 must be nonnegative after stored shift; min={minimum:.16e}"
        )

    low_bins = np.arange(spec.low_limit, dtype=np.int64)
    low_probability = qpe_probability_on_bins(
        amplitudes,
        low_bins,
        spec.modulus,
    )

    # If low_limit == 1, the only accepted bin is y=0.  There is no distinct
    # high-side integer bin because high_limit == modulus.
    if spec.high_limit >= spec.modulus:
        return low_probability

    zero_probability = qpe_probability_on_bins(
        amplitudes,
        np.asarray([0], dtype=np.int64),
        spec.modulus,
    )

    return np.clip(
        2.0 * low_probability - zero_probability,
        0.0,
        1.0,
    )


def amplified_success_probability(
    initial_good_probability: float,
    grover_iterations: int,
) -> float:
    probability = float(np.clip(initial_good_probability, 0.0, 1.0))

    if probability <= 0.0:
        return 0.0
    if probability >= 1.0:
        return 1.0

    theta = math.asin(math.sqrt(probability))
    value = math.sin((2 * grover_iterations + 1) * theta) ** 2
    return float(np.clip(value, 0.0, 1.0))


def shots_for_at_least_one_success(
    success_probability: float,
    confidence: float,
) -> int | None:
    p = float(np.clip(success_probability, 0.0, 1.0))

    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be strictly between 0 and 1")
    if p <= 0.0:
        return None
    if p >= 1.0:
        return 1

    return max(
        1,
        int(math.ceil(math.log1p(-confidence) / math.log1p(-p))),
    )



def theoretical_optimal_grover(initial_flag_probability: float) -> int:
    """Return the integer Grover iteration nearest the first success peak."""
    p = float(np.clip(initial_flag_probability, 0.0, 1.0))
    if p <= 0.0 or p >= 1.0:
        return 0

    theta = math.asin(math.sqrt(p))
    continuous = math.pi / (4.0 * theta) - 0.5
    if continuous <= 0.0:
        return 0

    candidates = {
        max(0, int(math.floor(continuous))),
        max(0, int(math.ceil(continuous))),
    }
    return max(
        candidates,
        key=lambda r: (amplified_success_probability(p, r), -r),
    )


def resolve_grover_scan_max(
    requested: int | None,
    *,
    initial_flag_probability: float,
    retrieval_probability_qae: float,
) -> tuple[int, int]:
    """Return (scan_max, theoretical_optimum); None means --grover-max auto."""
    if retrieval_probability_qae <= 0.0 or initial_flag_probability <= 0.0:
        theoretical = 0
        auto_max = 0
    else:
        theoretical = theoretical_optimal_grover(initial_flag_probability)
        auto_max = max(1, 2 * theoretical)

    if requested is None:
        return auto_max, theoretical
    if requested < 0:
        raise ValueError("--grover-max must be >= 0 or 'auto'")
    return requested, theoretical


def safe_product(a: int | None, b: int) -> int | None:
    if a is None:
        return None
    return int(a) * int(b)


def scan_grover_iterations(
    *,
    occ_count: int,
    initial_flag_probability: float,
    good_ligand_given_flag: float,
    grover_max: int,
    q_calls_per_qpe: int,
    q_calls_per_threshold_oracle: int,
) -> list[GroverPoint]:
    """Evaluate ligand-retrieval probability for r=0..grover_max."""
    rows: list[GroverPoint] = []

    for r in range(grover_max + 1):
        amplified_flag = amplified_success_probability(
            initial_flag_probability,
            r,
        )
        good_ligand_probability = amplified_flag * good_ligand_given_flag
        shots_95 = shots_for_at_least_one_success(
            good_ligand_probability,
            0.95,
        )
        shots_99 = shots_for_at_least_one_success(
            good_ligand_probability,
            0.99,
        )
        q_calls_per_shot = (
            r * q_calls_per_threshold_oracle
            + q_calls_per_qpe
        )

        rows.append(
            GroverPoint(
                occ_count=occ_count,
                grover_iterations=r,
                amplified_flag_probability=amplified_flag,
                good_ligand_probability=good_ligand_probability,
                shots_95=shots_95,
                shots_99=shots_99,
                q_calls_per_shot=q_calls_per_shot,
                total_q_calls_95=safe_product(shots_95, q_calls_per_shot),
                total_q_calls_99=safe_product(shots_99, q_calls_per_shot),
            )
        )

    return rows


# ---------------------------------------------------------------------------
# Selector / circuit resources
# ---------------------------------------------------------------------------


def selector_qubits(count: int) -> int:
    if count < 1:
        raise ValueError("selector count must be >= 1")
    return 0 if count == 1 else int(math.ceil(math.log2(count)))


def is_power_of_two(value: int) -> bool:
    return value > 0 and value & (value - 1) == 0


def build_logical_circuit_ligand(
    *,
    receptor_pdbid: str,
    occ_count: int,
    eval_qubits: int,
    comparator: ComparatorSpec,
    grover_iterations: int,
) -> tuple[object, int, int]:
    """Build the high-level logical circuit for one OCC-count experiment."""
    try:
        from qiskit import (
            AncillaRegister,
            ClassicalRegister,
            QuantumCircuit,
            QuantumRegister,
        )
        from qiskit.circuit import Gate
        from qiskit.circuit.library import IntegerComparator
    except ImportError as exc:
        raise ImportError(
            "Qiskit is required to build the logical circuit. "
            "Use --skip-circuit to skip SVG generation."
        ) from exc

    map_count = 2
    map_bits = 1
    occ_bits = selector_qubits(occ_count)

    if not is_power_of_two(occ_count):
        raise ValueError("this scaling experiment expects OCC counts that are powers of two")

    # Fixed 19-qubit data register:
    # 5 x + 5 y + 5 z + 3 type + 1 auxiliary.
    x = QuantumRegister(5, "x")
    y = QuantumRegister(5, "y")
    z = QuantumRegister(5, "z")
    atom_type = QuantumRegister(3, "type")
    aux = QuantumRegister(1, "aux")

    shift_x = QuantumRegister(3, "shift_x")
    shift_y = QuantumRegister(3, "shift_y")
    shift_z = QuantumRegister(3, "shift_z")
    rotation = QuantumRegister(ROTATION_SELECTOR_QUBITS, "rot")

    occ_selector = QuantumRegister(occ_bits, "occ_sel") if occ_bits else None
    map_selector = QuantumRegister(map_bits, "map_sel")
    evaluation = QuantumRegister(eval_qubits, "eval")
    flags = QuantumRegister(1, "flag")

    registers: list = [x, y, z, atom_type, aux]
    if occ_selector is not None:
        registers.append(occ_selector)
    registers.append(rotation)
    registers.extend((shift_x, shift_y, shift_z, map_selector, evaluation, flags))

    low_comparator = IntegerComparator(
        num_state_qubits=eval_qubits,
        value=comparator.low_limit,
        geq=False,
    )
    low_work = low_comparator.num_qubits - eval_qubits - 1

    high_comparator = None
    high_work = 0
    if comparator.high_limit < comparator.modulus:
        high_comparator = IntegerComparator(
            num_state_qubits=eval_qubits,
            value=comparator.high_limit,
            geq=True,
        )
        high_work = high_comparator.num_qubits - eval_qubits - 1

    # The low and high comparisons are applied sequentially, so their work
    # ancillas can be reused.
    comparator_work = max(low_work, high_work)

    work = AncillaRegister(comparator_work, "work") if comparator_work else None
    if work is not None:
        registers.append(work)

    # Ligand-only readout: measure FLAG and OCC selector only.
    # MAP/ROT/TRANS remain coherent search registers but are not read out.
    flag_measure = ClassicalRegister(1, "flag_m")
    classical_registers: list = [flag_measure]

    occ_measure = None
    if occ_bits:
        occ_measure = ClassicalRegister(occ_bits, "occ_m")
        classical_registers.append(occ_measure)

    circuit = QuantumCircuit(*registers, *classical_registers)

    # Uniform selector superposition.
    circuit.h(rotation)
    circuit.h(shift_x)
    circuit.h(shift_y)
    circuit.h(shift_z)
    circuit.h(map_selector)
    if occ_selector is not None:
        circuit.h(occ_selector)

    physical_data_list = list(x) + list(y) + list(z) + list(atom_type)
    data_list = physical_data_list + list(aux)
    spatial_list = list(x) + list(y) + list(z)
    shift_list = list(shift_x) + list(shift_y) + list(shift_z)
    rotation_list = list(rotation)
    occ_list = list(occ_selector) if occ_selector is not None else []
    map_list = list(map_selector)
    work_list = list(work) if work is not None else []

    circuit.append(
        Gate(
            "select_occ",
            occ_bits + DATA_QUBITS,
            [],
            label=f"SELECT {occ_count} OCC",
        ),
        occ_list + data_list,
    )

    circuit.append(
        Gate(
            "pose_rotation",
            ROTATION_SELECTOR_QUBITS + DATA_QUBITS,
            [],
            label="ROT 32/24 AUX0",
        ),
        rotation_list + data_list,
    )

    circuit.append(
        Gate(
            "cyclic_translation",
            9 + 15 + 1,
            [],
            label="SHIFT MOD 32 AUX0",
        ),
        shift_list + spatial_list + list(aux),
    )

    circuit.append(
        Gate(
            "select_u_map",
            map_bits + DATA_QUBITS,
            [],
            label=f"SELECT {receptor_pdbid}_0/_1",
        ),
        map_list + data_list,
    )

    qae_system = (
        map_list
        + occ_list
        + rotation_list
        + shift_list
        + data_list
    )

    circuit.append(
        Gate(
            "pose_qae_qpe",
            eval_qubits + len(qae_system),
            [],
            label=f"QAE/QPE m={eval_qubits}",
        ),
        list(evaluation) + qae_system,
    )

    comparator_state = list(reversed(evaluation))

    # Low phase branch: y < low_limit.
    circuit.append(
        low_comparator.to_gate(label=f"y<{comparator.low_limit}"),
        comparator_state + [flags[0]] + work_list[:low_work],
    )

    # High phase branch: y >= high_limit.
    #
    # IntegerComparator toggles the compare qubit.  The two accepted intervals
    # are disjoint, so sequential toggles on the same FLAG implement the OR
    # exactly (XOR == OR for mutually exclusive conditions).
    if high_comparator is not None:
        circuit.append(
            high_comparator.to_gate(label=f"y>={comparator.high_limit}"),
            comparator_state + [flags[0]] + work_list[:high_work],
        )

    if grover_iterations:
        search_system = (
            data_list
            + shift_list
            + rotation_list
            + occ_list
            + map_list
            + list(evaluation)
            + list(flags)
            + work_list
        )
        circuit.append(
            Gate(
                "selector_amplification",
                len(search_system),
                [],
                label=f"Q_SEL x{grover_iterations}",
            ),
            search_system,
        )

    circuit.measure(flags[0], flag_measure[0])
    if occ_selector is not None and occ_measure is not None:
        circuit.measure(occ_selector, occ_measure)

    return circuit, circuit.num_qubits, comparator_work


def comparator_resource_counts(eval_qubits: int, comparator: ComparatorSpec) -> tuple[int, int]:
    """
    Return reusable comparator work-qubit count.

    The low-side and high-side IntegerComparators run sequentially and reuse
    the same ancillas, so the required work register is the larger of the two,
    not their sum.
    """
    try:
        from qiskit.circuit.library import IntegerComparator
    except ImportError as exc:
        raise ImportError(
            "Qiskit is required for exact comparator resource counting."
        ) from exc

    low_comparator = IntegerComparator(
        num_state_qubits=eval_qubits,
        value=comparator.low_limit,
        geq=False,
    )
    low_work = low_comparator.num_qubits - eval_qubits - 1

    high_work = 0
    if comparator.high_limit < comparator.modulus:
        high_comparator = IntegerComparator(
            num_state_qubits=eval_qubits,
            value=comparator.high_limit,
            geq=True,
        )
        high_work = high_comparator.num_qubits - eval_qubits - 1

    work = max(low_work, high_work)
    return work, 0


def logical_qubit_count(
    occ_count: int,
    eval_qubits: int,
    work_qubits: int,
) -> int:
    return (
        DATA_QUBITS
        + 1  # map selector
        + selector_qubits(occ_count)
        + ROTATION_SELECTOR_QUBITS
        + TRANSLATION_SELECTOR_QUBITS
        + eval_qubits
        + 1  # flag
        + work_qubits
    )


def build_logical_circuit_pose(
    *,
    receptor_pdbid: str,
    occ_count: int,
    eval_qubits: int,
    comparator: ComparatorSpec,
    grover_iterations: int,
) -> tuple[object, int, int]:
    """Build the high-level logical circuit for one OCC-count experiment."""
    try:
        from qiskit import (
            AncillaRegister,
            ClassicalRegister,
            QuantumCircuit,
            QuantumRegister,
        )
        from qiskit.circuit import Gate
        from qiskit.circuit.library import IntegerComparator
    except ImportError as exc:
        raise ImportError(
            "Qiskit is required to build the logical circuit. "
            "Use --skip-circuit to skip SVG generation."
        ) from exc

    map_count = 2
    map_bits = 1
    occ_bits = selector_qubits(occ_count)

    if not is_power_of_two(occ_count):
        raise ValueError("this scaling experiment expects OCC counts that are powers of two")

    # Fixed 19-qubit data register:
    # 5 x + 5 y + 5 z + 3 type + 1 auxiliary.
    x = QuantumRegister(5, "x")
    y = QuantumRegister(5, "y")
    z = QuantumRegister(5, "z")
    atom_type = QuantumRegister(3, "type")
    aux = QuantumRegister(1, "aux")

    shift_x = QuantumRegister(3, "shift_x")
    shift_y = QuantumRegister(3, "shift_y")
    shift_z = QuantumRegister(3, "shift_z")
    rotation = QuantumRegister(ROTATION_SELECTOR_QUBITS, "rot")

    occ_selector = QuantumRegister(occ_bits, "occ_sel") if occ_bits else None
    map_selector = QuantumRegister(map_bits, "map_sel")
    evaluation = QuantumRegister(eval_qubits, "eval")
    flags = QuantumRegister(1, "flag")

    registers: list = [x, y, z, atom_type, aux]
    if occ_selector is not None:
        registers.append(occ_selector)
    registers.append(rotation)
    registers.extend((shift_x, shift_y, shift_z, map_selector, evaluation, flags))

    low_comparator = IntegerComparator(
        num_state_qubits=eval_qubits,
        value=comparator.low_limit,
        geq=False,
    )
    low_work = low_comparator.num_qubits - eval_qubits - 1

    high_comparator = None
    high_work = 0
    if comparator.high_limit < comparator.modulus:
        high_comparator = IntegerComparator(
            num_state_qubits=eval_qubits,
            value=comparator.high_limit,
            geq=True,
        )
        high_work = high_comparator.num_qubits - eval_qubits - 1

    # The low and high comparisons are applied sequentially, so their work
    # ancillas can be reused.
    comparator_work = max(low_work, high_work)

    work = AncillaRegister(comparator_work, "work") if comparator_work else None
    if work is not None:
        registers.append(work)

    flag_measure = ClassicalRegister(1, "flag_m")
    rotation_measure = ClassicalRegister(ROTATION_SELECTOR_QUBITS, "rot_m")
    shift_measure = ClassicalRegister(TRANSLATION_SELECTOR_QUBITS, "shift_m")
    map_measure = ClassicalRegister(1, "map_m")

    classical_registers: list = [
        flag_measure,
        rotation_measure,
        shift_measure,
        map_measure,
    ]

    occ_measure = None
    if occ_bits:
        occ_measure = ClassicalRegister(occ_bits, "occ_m")
        classical_registers.append(occ_measure)

    circuit = QuantumCircuit(*registers, *classical_registers)

    # Uniform selector superposition.
    circuit.h(rotation)
    circuit.h(shift_x)
    circuit.h(shift_y)
    circuit.h(shift_z)
    circuit.h(map_selector)
    if occ_selector is not None:
        circuit.h(occ_selector)

    physical_data_list = list(x) + list(y) + list(z) + list(atom_type)
    data_list = physical_data_list + list(aux)
    spatial_list = list(x) + list(y) + list(z)
    shift_list = list(shift_x) + list(shift_y) + list(shift_z)
    rotation_list = list(rotation)
    occ_list = list(occ_selector) if occ_selector is not None else []
    map_list = list(map_selector)
    work_list = list(work) if work is not None else []

    circuit.append(
        Gate(
            "select_occ",
            occ_bits + DATA_QUBITS,
            [],
            label=f"SELECT {occ_count} OCC",
        ),
        occ_list + data_list,
    )

    circuit.append(
        Gate(
            "pose_rotation",
            ROTATION_SELECTOR_QUBITS + DATA_QUBITS,
            [],
            label="ROT 32/24 AUX0",
        ),
        rotation_list + data_list,
    )

    circuit.append(
        Gate(
            "cyclic_translation",
            9 + 15 + 1,
            [],
            label="SHIFT MOD 32 AUX0",
        ),
        shift_list + spatial_list + list(aux),
    )

    circuit.append(
        Gate(
            "select_u_map",
            map_bits + DATA_QUBITS,
            [],
            label=f"SELECT {receptor_pdbid}_0/_1",
        ),
        map_list + data_list,
    )

    qae_system = (
        map_list
        + occ_list
        + rotation_list
        + shift_list
        + data_list
    )

    circuit.append(
        Gate(
            "pose_qae_qpe",
            eval_qubits + len(qae_system),
            [],
            label=f"QAE/QPE m={eval_qubits}",
        ),
        list(evaluation) + qae_system,
    )

    comparator_state = list(reversed(evaluation))

    # Low phase branch: y < low_limit.
    circuit.append(
        low_comparator.to_gate(label=f"y<{comparator.low_limit}"),
        comparator_state + [flags[0]] + work_list[:low_work],
    )

    # High phase branch: y >= high_limit.
    #
    # IntegerComparator toggles the compare qubit.  The two accepted intervals
    # are disjoint, so sequential toggles on the same FLAG implement the OR
    # exactly (XOR == OR for mutually exclusive conditions).
    if high_comparator is not None:
        circuit.append(
            high_comparator.to_gate(label=f"y>={comparator.high_limit}"),
            comparator_state + [flags[0]] + work_list[:high_work],
        )

    if grover_iterations:
        search_system = (
            data_list
            + shift_list
            + rotation_list
            + occ_list
            + map_list
            + list(evaluation)
            + list(flags)
            + work_list
        )
        circuit.append(
            Gate(
                "selector_amplification",
                len(search_system),
                [],
                label=f"Q_SEL x{grover_iterations}",
            ),
            search_system,
        )

    circuit.measure(flags[0], flag_measure[0])
    circuit.measure(rotation, rotation_measure)
    circuit.measure(
        list(shift_x) + list(shift_y) + list(shift_z),
        shift_measure,
    )
    circuit.measure(map_selector, map_measure)

    if occ_selector is not None and occ_measure is not None:
        circuit.measure(occ_selector, occ_measure)

    return circuit, circuit.num_qubits, comparator_work




SELECTED_TEMP_HEADER = (
    "OCC_SELECTOR\tPOSE_SELECTOR\tROT_SELECTOR\tTRANS_SELECTOR\t"
    "SHIFT_Z\tSHIFT_Y\tSHIFT_X\tOCC\tENERGY\tA0\tP_FLAG\n"
)

def append_selected_rows(
    handle,
    *,
    occ_selector: int,
    occ_name: str,
    good_indices: np.ndarray,
    energy: np.ndarray,
    a0: np.ndarray,
    flag: np.ndarray,
    columns: PoseColumns,
) -> None:
    for pose_index in good_indices.tolist():
        handle.write(
            f"{occ_selector}\t"
            f"{int(columns.pose_selector[pose_index])}\t"
            f"{int(columns.rotation_selector[pose_index])}\t"
            f"{int(columns.translation_selector[pose_index])}\t"
            f"{int(columns.shift_z[pose_index])}\t"
            f"{int(columns.shift_y[pose_index])}\t"
            f"{int(columns.shift_x[pose_index])}\t"
            f"{occ_name}\t"
            f"{float(energy[pose_index]):.12e}\t"
            f"{float(a0[pose_index]):.12e}\t"
            f"{float(flag[pose_index]):.12e}\n"
        )


def iter_selected_temp(path: Path, occ_count: int):
    with path.open("r", encoding="utf-8") as handle:
        header = handle.readline()
        if header != SELECTED_TEMP_HEADER:
            raise ValueError(f"unexpected selected-temp header: {path}")

        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 11:
                raise ValueError(f"malformed selected-temp row in {path}")

            occ_selector = int(fields[0])
            if occ_selector >= occ_count:
                break

            yield (
                occ_selector,
                int(fields[1]),
                int(fields[2]),
                int(fields[3]),
                int(fields[4]),
                int(fields[5]),
                int(fields[6]),
                fields[7],
                float(fields[8]),
                float(fields[9]),
                float(fields[10]),
            )


def write_selected_indices(
    path: Path,
    *,
    pdbid: str,
    occ_count: int,
    map0_temp: Path,
    map1_temp: Path,
    result: ScaleResult,
) -> None:
    """Write only exact A0<threshold states, sorted by global selector index."""
    path.parent.mkdir(parents=True, exist_ok=True)

    M = result.search_states
    p_before = 1.0 / float(M)
    initial_flag = result.initial_flag_probability
    amplified_flag = result.amplified_flag_probability

    if initial_flag <= TOLERANCE or initial_flag >= 1.0 - TOLERANCE:
        good_scale = 1.0
        bad_scale = 1.0
    else:
        good_scale = amplified_flag / initial_flag
        bad_scale = (1.0 - amplified_flag) / (1.0 - initial_flag)

    flag_sum = initial_flag * float(M)

    with path.open("w", encoding="utf-8") as handle:
        handle.write(
            "GLOBAL_INDEX\tMAP_SELECTOR\tOCC_SELECTOR\tPOSE_SELECTOR\t"
            "ROT_SELECTOR\tTRANS_SELECTOR\tSHIFT_Z\tSHIFT_Y\tSHIFT_X\t"
            "MAP\tOCC\tENERGY\tA0\tP_FLAG\t"
            "P_QSEL_BEFORE\tP_QSEL_AFTER_BEST\tP_FLAG_AND_SELECTOR_AFTER_BEST\n"
        )

        written = 0

        # Global-index layout:
        # low 14 bits = pose (rot low 5 bits, then trans 9 bits)
        # next log2(N_OCC) bits = OCC selector
        # highest selector bit = MAP selector
        for map_selector, temp_path in ((0, map0_temp), (1, map1_temp)):
            for (
                occ_selector,
                pose_selector,
                rotation,
                translation,
                dz,
                dy,
                dx,
                occ_name,
                energy,
                a0,
                p_flag,
            ) in iter_selected_temp(temp_path, occ_count):

                global_index = (
                    pose_selector
                    + POSES_PER_PAIR * occ_selector
                    + POSES_PER_PAIR * occ_count * map_selector
                )

                p_after = p_before * (
                    p_flag * good_scale
                    + (1.0 - p_flag) * bad_scale
                )

                p_joint = (
                    amplified_flag * p_flag / flag_sum
                    if flag_sum > 0.0 and amplified_flag > 0.0
                    else 0.0
                )

                handle.write(
                    f"{global_index}\t"
                    f"{map_selector}\t"
                    f"{occ_selector}\t"
                    f"{pose_selector}\t"
                    f"{rotation}\t"
                    f"{translation}\t"
                    f"{dz}\t{dy}\t{dx}\t"
                    f"{pdbid}_{map_selector}\t"
                    f"{occ_name}\t"
                    f"{energy:.12e}\t"
                    f"{a0:.12e}\t"
                    f"{p_flag:.12e}\t"
                    f"{p_before:.12e}\t"
                    f"{p_after:.12e}\t"
                    f"{p_joint:.12e}\n"
                )
                written += 1

    if written != result.exact_good_states:
        raise AssertionError(
            f"selected-index count mismatch for N={occ_count}: "
            f"wrote {written}, expected {result.exact_good_states}"
        )




def scan_pose_grover_iterations(
    *,
    occ_count: int,
    initial_flag_probability: float,
    true_good_given_flag: float,
    grover_max: int,
    q_calls_per_qpe: int,
    q_calls_per_threshold_oracle: int,
) -> list[PoseGroverPoint]:
    rows: list[PoseGroverPoint] = []
    for r in range(grover_max + 1):
        amplified_flag = amplified_success_probability(initial_flag_probability, r)
        true_good_probability = amplified_flag * true_good_given_flag
        shots_95 = shots_for_at_least_one_success(true_good_probability, 0.95)
        shots_99 = shots_for_at_least_one_success(true_good_probability, 0.99)
        q_calls_per_shot = r * q_calls_per_threshold_oracle + q_calls_per_qpe
        rows.append(
            PoseGroverPoint(
                occ_count=occ_count,
                grover_iterations=r,
                amplified_flag_probability=amplified_flag,
                true_good_probability=true_good_probability,
                shots_95=shots_95,
                shots_99=shots_99,
                q_calls_per_shot=q_calls_per_shot,
                total_q_calls_95=safe_product(shots_95, q_calls_per_shot),
                total_q_calls_99=safe_product(shots_99, q_calls_per_shot),
            )
        )
    return rows


def calculate_pose_scale_result(
    *,
    occ_count: int,
    stats: PoseCumulativeStats,
    eval_qubits: int,
    grover_max_requested: int | None,
    qsel_shot_budget: int | None,
    work_qubits: int,
) -> tuple[PoseScaleResult, list[PoseGroverPoint]]:
    expected_states = 2 * occ_count * POSES_PER_PAIR
    if stats.states != expected_states:
        raise AssertionError(
            f"pose search-state accounting mismatch for N={occ_count}: "
            f"got {stats.states}, expected {expected_states}"
        )

    M = stats.states
    initial_flag = stats.sum_flag / float(M)
    true_good_qae = stats.sum_flag_true_good / float(M)
    true_good_given_flag = (
        true_good_qae / initial_flag if initial_flag > 0.0 else 0.0
    )

    qae_shots_95 = shots_for_at_least_one_success(true_good_qae, 0.95)
    qae_shots_99 = shots_for_at_least_one_success(true_good_qae, 0.99)

    q_calls_per_qpe = (1 << eval_qubits) - 1
    q_calls_per_threshold_oracle = 2 * q_calls_per_qpe

    grover_max, theoretical_opt = resolve_grover_scan_max(
        grover_max_requested,
        initial_flag_probability=initial_flag,
        retrieval_probability_qae=true_good_qae,
    )
    rows = scan_pose_grover_iterations(
        occ_count=occ_count,
        initial_flag_probability=initial_flag,
        true_good_given_flag=true_good_given_flag,
        grover_max=grover_max,
        q_calls_per_qpe=q_calls_per_qpe,
        q_calls_per_threshold_oracle=q_calls_per_threshold_oracle,
    )

    finite = [row for row in rows if row.total_q_calls_95 is not None]
    cost_best = (
        min(
            finite,
            key=lambda row: (
                row.total_q_calls_95,
                row.grover_iterations,
                row.shots_95 if row.shots_95 is not None else math.inf,
            ),
        )
        if finite
        else rows[0]
    )

    if qsel_shot_budget is None:
        best = cost_best
        qsel_shot_budget_met = None
    else:
        feasible = [
            row for row in rows
            if row.shots_95 is not None and row.shots_95 <= qsel_shot_budget
        ]
        if feasible:
            best = min(
                feasible,
                key=lambda row: (
                    row.grover_iterations,
                    row.total_q_calls_95 if row.total_q_calls_95 is not None else math.inf,
                ),
            )
            qsel_shot_budget_met = True
        else:
            best = cost_best
            qsel_shot_budget_met = False

    occ_bits = selector_qubits(occ_count)
    selector_total = (
        1 + occ_bits + ROTATION_SELECTOR_QUBITS + TRANSLATION_SELECTOR_QUBITS
    )
    if 1 << selector_total != M:
        raise AssertionError(
            f"pose selector-state mismatch for N={occ_count}: 2^{selector_total} != {M}"
        )

    logical = logical_qubit_count(occ_count, eval_qubits, work_qubits)

    result = PoseScaleResult(
        occ_count=occ_count,
        occ_selector_qubits=occ_bits,
        total_selector_qubits=selector_total,
        search_states=M,
        exact_good_states=stats.exact_good,
        exact_good_fraction=stats.exact_good / float(M),
        initial_flag_probability=initial_flag,
        true_good_probability_qae=true_good_qae,
        true_good_given_flag=true_good_given_flag,
        grover_max=grover_max,
        theoretical_opt_grover=theoretical_opt,
        grover_iterations=best.grover_iterations,
        amplified_flag_probability=best.amplified_flag_probability,
        true_good_probability_qsel=best.true_good_probability,
        qae_shots_95=qae_shots_95,
        qae_shots_99=qae_shots_99,
        qsel_shots_95=best.shots_95,
        qsel_shots_99=best.shots_99,
        qsel_shot_budget=qsel_shot_budget,
        qsel_shot_budget_met=qsel_shot_budget_met,
        q_calls_per_qpe=q_calls_per_qpe,
        q_calls_per_threshold_oracle=q_calls_per_threshold_oracle,
        qsel_q_calls_per_shot=best.q_calls_per_shot,
        qae_total_q_calls_95=safe_product(qae_shots_95, q_calls_per_qpe),
        qae_total_q_calls_99=safe_product(qae_shots_99, q_calls_per_qpe),
        qsel_total_q_calls_95=best.total_q_calls_95,
        qsel_total_q_calls_99=best.total_q_calls_99,
        logical_qubits=logical,
        work_qubits=work_qubits,
        min_energy=stats.min_energy,
        max_a0=stats.max_a0,
    )
    return result, rows


def write_pose_experiment_summary(
    path: Path,
    *,
    pdbid: str,
    cfg: ExperimentConfig,
    result: PoseScaleResult,
    Lm: float,
    Lo: float,
    shift: float,
    backend: Backend,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    resolution = amplitude_grid_resolution(cfg.eval_qubits)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(f"PROGRAM_VERSION {PROGRAM_VERSION}\n")
        handle.write("MODEL ideal_noiseless_quantum_circuit_analytic_simulation\n")
        handle.write("RETRIEVAL_TARGET LIGAND_AND_POSE_CONFIGURATION\n")
        handle.write("MEASURED_REGISTERS FLAG_OCC_MAP_ROT_TRANS\n")
        handle.write(f"RECEPTOR_PDBID {pdbid}\n")
        handle.write(f"OCC_COUNT {result.occ_count}\n")
        handle.write(f"SEARCH_STATES {result.search_states}\n")
        handle.write(f"AMPLITUDE_THRESHOLD {cfg.amplitude_threshold:.16e}\n")
        handle.write(f"ENERGY_THRESHOLD {cfg.energy_threshold:.16e}\n")
        handle.write(f"EVALUATION_QUBITS {cfg.eval_qubits}\n")
        handle.write(f"RESOLUTION_MULTIPLE_TARGET {cfg.resolution_multiple}\n")
        handle.write(f"AMPLITUDE_RESOLUTION {resolution:.16e}\n")
        handle.write(
            f"THRESHOLD_RESOLUTION_RATIO_ACTUAL "
            f"{cfg.amplitude_threshold / resolution:.16e}\n"
        )
        handle.write(f"LM {Lm:.16e}\n")
        handle.write(f"LO {Lo:.16e}\n")
        handle.write(f"MAP_SHIFT {shift:.16e}\n")
        handle.write(f"EXACT_GOOD_STATES {result.exact_good_states}\n")
        handle.write(f"EXACT_GOOD_FRACTION {result.exact_good_fraction:.16e}\n")
        handle.write(f"FLAG_PROBABILITY {result.initial_flag_probability:.16e}\n")
        handle.write(f"TRUE_GOOD_PROBABILITY_QAE {result.true_good_probability_qae:.16e}\n")
        handle.write(f"TRUE_GOOD_GIVEN_FLAG {result.true_good_given_flag:.16e}\n")
        handle.write(f"QAE_SHOTS95 {fmt_count(result.qae_shots_95)}\n")
        handle.write(f"QAE_TOTAL_Q_CALLS95 {fmt_count(result.qae_total_q_calls_95)}\n")
        handle.write(f"BEST_GROVER_ITERATIONS {result.grover_iterations}\n")
        handle.write(f"QSEL_FLAG_PROBABILITY {result.amplified_flag_probability:.16e}\n")
        handle.write(f"QSEL_TRUE_GOOD_PROBABILITY {result.true_good_probability_qsel:.16e}\n")
        handle.write(f"QSEL_SHOTS95 {fmt_count(result.qsel_shots_95)}\n")
        handle.write(f"QSEL_TOTAL_Q_CALLS95 {fmt_count(result.qsel_total_q_calls_95)}\n")
        handle.write(f"Q_CALLS_PER_QPE {result.q_calls_per_qpe}\n")
        handle.write(f"Q_CALLS_PER_THRESHOLD_ORACLE {result.q_calls_per_threshold_oracle}\n")
        handle.write(f"QSEL_Q_CALLS_PER_SHOT {result.qsel_q_calls_per_shot}\n")
        handle.write(f"TOTAL_LOGICAL_QUBITS {result.logical_qubits}\n")
        handle.write(f"COMPARATOR_WORK_QUBITS {result.work_qubits}\n")
        handle.write(f"MIN_ENERGY {result.min_energy:.16e}\n")
        handle.write(f"MAX_A0 {result.max_a0:.16e}\n")
        handle.write(f"CLASSICAL_OVERLAP_BACKEND {backend.name}\n")


def write_pose_scaling_summary(
    path: Path,
    results: list[PoseScaleResult],
    *,
    cfg: ExperimentConfig,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    resolution = amplitude_grid_resolution(cfg.eval_qubits)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(
            "OCC_COUNT\tOCC_SELECTOR_QUBITS\tTOTAL_SELECTOR_QUBITS\tSEARCH_STATES\t"
            "EXACT_GOOD_STATES\tEXACT_GOOD_FRACTION\tAMPLITUDE_THRESHOLD\t"
            "ENERGY_THRESHOLD\tEVAL_QUBITS\tRESOLUTION_MULTIPLE_TARGET\t"
            "AMPLITUDE_RESOLUTION\tTHRESHOLD_RESOLUTION_RATIO_ACTUAL\t"
            "QAE_FLAG_PROB\tQAE_TRUE_GOOD_PROB\tTRUE_GOOD_GIVEN_FLAG\t"
            "QAE_SHOTS95\tQAE_SHOTS99\tQAE_TOTAL_Q_CALLS95\tQAE_TOTAL_Q_CALLS99\t"
            "GROVER_MAX\tTHEORETICAL_OPT_GROVER\tBEST_GROVER_ITER\t"
            "BEST_QSEL_FLAG_PROB\tBEST_QSEL_TRUE_GOOD_PROB\tBEST_QSEL_SHOTS95\t"
            "BEST_QSEL_SHOTS99\tBEST_QSEL_TOTAL_Q_CALLS95\tBEST_QSEL_TOTAL_Q_CALLS99\t"
            "Q_CALLS_PER_QPE\tBEST_QSEL_Q_CALLS_PER_SHOT\tWORK_QUBITS\t"
            "TOTAL_LOGICAL_QUBITS\tMIN_ENERGY\tMAX_A0\n"
        )
        for r in results:
            handle.write(
                f"{r.occ_count}\t{r.occ_selector_qubits}\t{r.total_selector_qubits}\t"
                f"{r.search_states}\t{r.exact_good_states}\t{r.exact_good_fraction:.12e}\t"
                f"{cfg.amplitude_threshold:.12e}\t{cfg.energy_threshold:.12e}\t"
                f"{cfg.eval_qubits}\t{cfg.resolution_multiple}\t{resolution:.12e}\t"
                f"{cfg.amplitude_threshold / resolution:.12e}\t"
                f"{r.initial_flag_probability:.12e}\t{r.true_good_probability_qae:.12e}\t"
                f"{r.true_good_given_flag:.12e}\t{fmt_count(r.qae_shots_95)}\t"
                f"{fmt_count(r.qae_shots_99)}\t{fmt_count(r.qae_total_q_calls_95)}\t"
                f"{fmt_count(r.qae_total_q_calls_99)}\t{r.grover_max}\t"
                f"{r.theoretical_opt_grover}\t{r.grover_iterations}\t"
                f"{r.amplified_flag_probability:.12e}\t{r.true_good_probability_qsel:.12e}\t"
                f"{fmt_count(r.qsel_shots_95)}\t{fmt_count(r.qsel_shots_99)}\t"
                f"{fmt_count(r.qsel_total_q_calls_95)}\t{fmt_count(r.qsel_total_q_calls_99)}\t"
                f"{r.q_calls_per_qpe}\t{r.qsel_q_calls_per_shot}\t{r.work_qubits}\t"
                f"{r.logical_qubits}\t{r.min_energy:.12e}\t{r.max_a0:.12e}\n"
            )


def write_pose_grover_shots(path: Path, rows: list[PoseGroverPoint]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(
            "OCC_COUNT\tGROVER_ITERATIONS\tAMPLIFIED_FLAG_PROB\tTRUE_GOOD_PROB\t"
            "SHOTS95\tSHOTS99\tQ_CALLS_PER_SHOT\tTOTAL_Q_CALLS95\tTOTAL_Q_CALLS99\n"
        )
        for row in rows:
            handle.write(
                f"{row.occ_count}\t{row.grover_iterations}\t"
                f"{row.amplified_flag_probability:.12e}\t{row.true_good_probability:.12e}\t"
                f"{fmt_count(row.shots_95)}\t{fmt_count(row.shots_99)}\t"
                f"{row.q_calls_per_shot}\t{fmt_count(row.total_q_calls_95)}\t"
                f"{fmt_count(row.total_q_calls_99)}\n"
            )


def write_pose_batch_summary(
    path: Path,
    rows: list[tuple[str, ExperimentConfig, PoseScaleResult]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(
            "PDBID\tAMPLITUDE_THRESHOLD\tENERGY_THRESHOLD\tEVAL_QUBITS\t"
            "RESOLUTION_MULTIPLE_TARGET\tAMPLITUDE_RESOLUTION\t"
            "THRESHOLD_RESOLUTION_RATIO_ACTUAL\tOCC_COUNT\tSEARCH_STATES\t"
            "EXACT_GOOD_STATES\tEXACT_GOOD_FRACTION\tQAE_FLAG_PROB\t"
            "QAE_TRUE_GOOD_PROB\tTRUE_GOOD_GIVEN_FLAG\tFALSE_POSITIVE_GIVEN_FLAG\t"
            "QAE_SHOTS95\tQAE_SHOTS99\tQAE_TOTAL_Q_CALLS95\tQAE_TOTAL_Q_CALLS99\t"
            "THEORETICAL_OPT_GROVER\tBEST_GROVER_ITER\tQSEL_SHOTS95\tQSEL_SHOTS99\t"
            "QSEL_TOTAL_Q_CALLS95\tQSEL_TOTAL_Q_CALLS99\tQ_CALLS_PER_QPE\t"
            "QSEL_Q_CALLS_PER_SHOT\tTOTAL_LOGICAL_QUBITS\tMIN_ENERGY\tMAX_A0\n"
        )
        for pdbid, cfg, r in rows:
            resolution = amplitude_grid_resolution(cfg.eval_qubits)
            handle.write(
                f"{pdbid}\t{cfg.amplitude_threshold:.12e}\t{cfg.energy_threshold:.12e}\t"
                f"{cfg.eval_qubits}\t{cfg.resolution_multiple}\t{resolution:.12e}\t"
                f"{cfg.amplitude_threshold / resolution:.12e}\t{r.occ_count}\t"
                f"{r.search_states}\t{r.exact_good_states}\t{r.exact_good_fraction:.12e}\t"
                f"{r.initial_flag_probability:.12e}\t{r.true_good_probability_qae:.12e}\t"
                f"{r.true_good_given_flag:.12e}\t{1.0-r.true_good_given_flag:.12e}\t"
                f"{fmt_count(r.qae_shots_95)}\t{fmt_count(r.qae_shots_99)}\t"
                f"{fmt_count(r.qae_total_q_calls_95)}\t{fmt_count(r.qae_total_q_calls_99)}\t"
                f"{r.theoretical_opt_grover}\t{r.grover_iterations}\t"
                f"{fmt_count(r.qsel_shots_95)}\t{fmt_count(r.qsel_shots_99)}\t"
                f"{fmt_count(r.qsel_total_q_calls_95)}\t{fmt_count(r.qsel_total_q_calls_99)}\t"
                f"{r.q_calls_per_qpe}\t{r.qsel_q_calls_per_shot}\t{r.logical_qubits}\t"
                f"{r.min_energy:.12e}\t{r.max_a0:.12e}\n"
            )


# ---------------------------------------------------------------------------
# Ligand-level output
# ---------------------------------------------------------------------------


def write_selected_ligands(
    path: Path,
    *,
    records: list[LigandRecord],
    occ_count: int,
    result: ScaleResult,
) -> None:
    """Write exact-good ligands for the current nested OCC checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)

    if len(records) < occ_count:
        raise AssertionError(
            f"ligand-record count {len(records)} is smaller than OCC checkpoint {occ_count}"
        )

    M = result.search_states
    flag_sum = result.initial_flag_probability * float(M)
    amplified_flag = result.amplified_flag_probability

    written = 0
    with path.open("w", encoding="utf-8") as handle:
        handle.write(
            "OCC_SELECTOR\tSOURCE\tOCC\tEXACT_GOOD_CONFIGS\t"
            "EXACT_GOOD_CONFIG_FRACTION_WITHIN_LIGAND\tMIN_ENERGY\tMIN_A0\t"
            "REFERENCE_MIN_MAP_SELECTOR\tREFERENCE_MIN_POSE_SELECTOR\t"
            "REFERENCE_MIN_ROT_SELECTOR\tREFERENCE_MIN_TRANS_SELECTOR\t"
            "REFERENCE_MIN_SHIFT_Z\tREFERENCE_MIN_SHIFT_Y\tREFERENCE_MIN_SHIFT_X\t"
            "SUM_FLAG_WEIGHT\tP_FLAG_GIVEN_OCC\t"
            "P_FLAG_AND_OCC_BEFORE\tP_FLAG_AND_OCC_AFTER_BEST\n"
        )

        for record in records[:occ_count]:
            if record.exact_good_configs <= 0:
                continue

            p_joint_before = record.sum_flag / float(M)
            p_joint_after = (
                amplified_flag * record.sum_flag / flag_sum
                if flag_sum > 0.0 and amplified_flag > 0.0
                else 0.0
            )
            handle.write(
                f"{record.occ_selector}\t"
                f"{record.source}\t"
                f"{record.occ_name}\t"
                f"{record.exact_good_configs}\t"
                f"{record.exact_good_configs / float(2 * POSES_PER_PAIR):.12e}\t"
                f"{record.min_energy:.12e}\t"
                f"{record.min_a0:.12e}\t"
                f"{record.best_map_selector}\t"
                f"{record.best_pose_selector}\t"
                f"{record.best_rotation_selector}\t"
                f"{record.best_translation_selector}\t"
                f"{record.best_shift_z}\t"
                f"{record.best_shift_y}\t"
                f"{record.best_shift_x}\t"
                f"{record.sum_flag:.12e}\t"
                f"{record.p_flag_given_occ:.12e}\t"
                f"{p_joint_before:.12e}\t"
                f"{p_joint_after:.12e}\n"
            )
            written += 1

    if written != result.good_ligands:
        raise AssertionError(
            f"selected-ligand count mismatch for N={occ_count}: "
            f"wrote {written}, expected {result.good_ligands}"
        )


# ---------------------------------------------------------------------------
# Summary / output
# ---------------------------------------------------------------------------


def fmt_count(value: int | None) -> str:
    return "INF" if value is None else str(value)


def calculate_ligand_scale_result(
    *,
    occ_count: int,
    stats: CumulativeStats,
    eval_qubits: int,
    grover_max_requested: int | None,
    qsel_shot_budget: int | None,
    work_qubits: int,
) -> tuple[ScaleResult, list[GroverPoint]]:
    expected_states = 2 * occ_count * POSES_PER_PAIR
    if stats.states != expected_states:
        raise AssertionError(
            f"search-state accounting mismatch for N={occ_count}: "
            f"got {stats.states}, expected {expected_states}"
        )
    if not 0 <= stats.good_ligands <= occ_count:
        raise AssertionError(
            f"good-ligand accounting mismatch for N={occ_count}: {stats.good_ligands}"
        )

    M = stats.states
    initial_flag = stats.sum_flag / float(M)
    good_ligand_qae = stats.sum_flag_good_ligand / float(M)
    good_ligand_given_flag = (
        good_ligand_qae / initial_flag
        if initial_flag > 0.0
        else 0.0
    )

    qae_shots_95 = shots_for_at_least_one_success(good_ligand_qae, 0.95)
    qae_shots_99 = shots_for_at_least_one_success(good_ligand_qae, 0.99)

    q_calls_per_qpe = (1 << eval_qubits) - 1
    q_calls_per_threshold_oracle = 2 * q_calls_per_qpe

    grover_max, theoretical_opt = resolve_grover_scan_max(
        grover_max_requested,
        initial_flag_probability=initial_flag,
        retrieval_probability_qae=good_ligand_qae,
    )

    grover_rows = scan_grover_iterations(
        occ_count=occ_count,
        initial_flag_probability=initial_flag,
        good_ligand_given_flag=good_ligand_given_flag,
        grover_max=grover_max,
        q_calls_per_qpe=q_calls_per_qpe,
        q_calls_per_threshold_oracle=q_calls_per_threshold_oracle,
    )

    # Resource objective: select the r (including r=0) with the smallest total
    # coherent Q-operator calls required for 95% good-ligand retrieval.
    finite_cost = [
        row for row in grover_rows if row.total_q_calls_95 is not None
    ]
    cost_best = (
        min(
            finite_cost,
            key=lambda row: (
                row.total_q_calls_95,
                row.grover_iterations,
                row.shots_95 if row.shots_95 is not None else math.inf,
            ),
        )
        if finite_cost
        else grover_rows[0]
    )

    if qsel_shot_budget is None:
        best = cost_best
        qsel_shot_budget_met = None
    else:
        feasible = [
            row
            for row in grover_rows
            if row.shots_95 is not None and row.shots_95 <= qsel_shot_budget
        ]
        if feasible:
            best = min(
                feasible,
                key=lambda row: (
                    row.grover_iterations,
                    row.total_q_calls_95 if row.total_q_calls_95 is not None else math.inf,
                ),
            )
            qsel_shot_budget_met = True
        else:
            best = cost_best
            qsel_shot_budget_met = False

    occ_bits = selector_qubits(occ_count)
    selector_total = (
        1
        + occ_bits
        + ROTATION_SELECTOR_QUBITS
        + TRANSLATION_SELECTOR_QUBITS
    )

    if 1 << selector_total != M:
        raise AssertionError(
            f"selector-state mismatch for N={occ_count}: "
            f"2^{selector_total} != {M}"
        )

    logical = logical_qubit_count(
        occ_count,
        eval_qubits,
        work_qubits,
    )

    result = ScaleResult(
        occ_count=occ_count,
        occ_selector_qubits=occ_bits,
        total_selector_qubits=selector_total,
        search_states=M,
        exact_good_config_states=stats.exact_good_configs,
        exact_good_config_fraction=stats.exact_good_configs / float(M),
        good_ligands=stats.good_ligands,
        good_ligand_fraction=stats.good_ligands / float(occ_count),
        initial_flag_probability=initial_flag,
        good_ligand_probability_qae=good_ligand_qae,
        good_ligand_given_flag=good_ligand_given_flag,
        grover_max=grover_max,
        theoretical_opt_grover=theoretical_opt,
        grover_iterations=best.grover_iterations,
        amplified_flag_probability=best.amplified_flag_probability,
        good_ligand_probability_qsel=best.good_ligand_probability,
        qae_shots_95=qae_shots_95,
        qae_shots_99=qae_shots_99,
        qsel_shots_95=best.shots_95,
        qsel_shots_99=best.shots_99,
        qsel_shot_budget=qsel_shot_budget,
        qsel_shot_budget_met=qsel_shot_budget_met,
        q_calls_per_qpe=q_calls_per_qpe,
        q_calls_per_threshold_oracle=q_calls_per_threshold_oracle,
        qsel_q_calls_per_shot=best.q_calls_per_shot,
        qae_total_q_calls_95=safe_product(qae_shots_95, q_calls_per_qpe),
        qae_total_q_calls_99=safe_product(qae_shots_99, q_calls_per_qpe),
        qsel_total_q_calls_95=best.total_q_calls_95,
        qsel_total_q_calls_99=best.total_q_calls_99,
        logical_qubits=logical,
        work_qubits=work_qubits,
        min_energy=stats.min_energy,
        max_a0=stats.max_a0,
    )
    return result, grover_rows


def write_ligand_experiment_summary(
    path: Path,
    *,
    pdbid: str,
    result: ScaleResult,
    amplitude_threshold: float,
    energy_threshold: float,
    Lm: float,
    Lo: float,
    shift: float,
    eval_qubits: int,
    result_resolution_multiple: int,
    comparator: ComparatorSpec,
    backend: Backend,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as handle:
        handle.write(f"PROGRAM_VERSION {PROGRAM_VERSION}\n")
        handle.write("MODEL ideal_noiseless_quantum_circuit_analytic_simulation\n")
        handle.write("RETRIEVAL_TARGET LIGAND_ID_ONLY\n")
        handle.write("SUCCESS_DEFINITION OCC_HAS_AT_LEAST_ONE_EXACT_GOOD_MAP_ROT_TRANS_CONFIGURATION\n")
        handle.write("MEASURED_REGISTERS FLAG_AND_OCC_SELECTOR\n")
        handle.write(f"RECEPTOR_PDBID {pdbid}\n")
        handle.write(f"OCC_COUNT {result.occ_count}\n")
        handle.write("NATIVE_OCC_SELECTOR 0\n")
        handle.write("DECOY_SELECTION 0001_*_THROUGH_PREFIX_N_MINUS_1\n")
        handle.write(f"DATA_QUBITS {DATA_QUBITS}\n")
        handle.write(f"FULL_DATA_SIZE {FULL_DATA_SIZE}\n")
        handle.write("MAP_SELECTOR_QUBITS 1\n")
        handle.write(f"OCC_SELECTOR_QUBITS {result.occ_selector_qubits}\n")
        handle.write(f"ROTATION_SELECTOR_QUBITS {ROTATION_SELECTOR_QUBITS}\n")
        handle.write(f"TRANSLATION_SELECTOR_QUBITS {TRANSLATION_SELECTOR_QUBITS}\n")
        handle.write(f"TOTAL_SELECTOR_QUBITS {result.total_selector_qubits}\n")
        handle.write(f"SEARCH_STATES {result.search_states}\n")
        handle.write("CONFIGURATIONS_PER_LIGAND 32768\n")
        handle.write("ROTATION_SELECTOR_STATES 32\n")
        handle.write("UNIQUE_ROTATIONS 24\n")
        handle.write("TRANSLATION_SELECTOR_STATES 512\n")
        handle.write("TRANSLATION_RANGE_XYZ -4..3\n")
        handle.write("TRANSLATION_BOUNDARY CYCLIC_MOD_32\n")
        handle.write(f"LM {Lm:.16e}\n")
        handle.write(f"LO {Lo:.16e}\n")
        handle.write(f"MAP_SHIFT {shift:.16e}\n")
        handle.write(f"AMPLITUDE_THRESHOLD {amplitude_threshold:.16e}\n")
        handle.write(f"ENERGY_THRESHOLD {energy_threshold:.16e}\n")
        handle.write(f"MIN_ENERGY {result.min_energy:.16e}\n")
        handle.write(f"MAX_A0 {result.max_a0:.16e}\n")
        handle.write(f"EXACT_GOOD_CONFIG_STATES {result.exact_good_config_states}\n")
        handle.write(f"EXACT_GOOD_CONFIG_FRACTION {result.exact_good_config_fraction:.16e}\n")
        handle.write(f"GOOD_LIGAND_COUNT {result.good_ligands}\n")
        handle.write(f"GOOD_LIGAND_FRACTION {result.good_ligand_fraction:.16e}\n")
        handle.write(f"EVALUATION_QUBITS {eval_qubits}\n")
        handle.write(f"RESOLUTION_MULTIPLE_TARGET {result_resolution_multiple}\n")
        amplitude_resolution = amplitude_grid_resolution(eval_qubits)
        handle.write(f"AMPLITUDE_RESOLUTION {amplitude_resolution:.16e}\n")
        handle.write(
            "THRESHOLD_RESOLUTION_RATIO_ACTUAL "
            f"{amplitude_threshold / amplitude_resolution:.16e}\n"
        )
        handle.write(f"QPE_MODULUS {comparator.modulus}\n")
        handle.write(f"COMPARATOR_LOW y<{comparator.low_limit}\n")
        if comparator.high_limit < comparator.modulus:
            handle.write(f"COMPARATOR_HIGH y>={comparator.high_limit}\n")
            handle.write(
                f"COMPARATOR_ACCEPT y<{comparator.low_limit} OR "
                f"y>={comparator.high_limit}\n"
            )
        else:
            handle.write("COMPARATOR_HIGH NONE\n")
            handle.write(f"COMPARATOR_ACCEPT y<{comparator.low_limit}\n")
        handle.write(
            "MIN_NONZERO_AMPLITUDE_GRID "
            f"{math.sin(math.pi / comparator.modulus):.16e}\n"
        )
        handle.write(f"CLASSICAL_OVERLAP_BACKEND {backend.name}\n")
        handle.write("CLASSICAL_FFT_IS_QUANTUM_GATE NO\n")
        handle.write("HARDWARE_EXECUTION NO\n")
        handle.write("NOISE_MODEL NONE\n")

        handle.write("\nQAE_QPE_LIGAND_RETRIEVAL\n")
        handle.write(f"FLAG_PROBABILITY {result.initial_flag_probability:.16e}\n")
        handle.write(f"GOOD_LIGAND_PROBABILITY {result.good_ligand_probability_qae:.16e}\n")
        handle.write(f"GOOD_LIGAND_GIVEN_FLAG {result.good_ligand_given_flag:.16e}\n")
        handle.write(f"SHOTS_95_GOOD_LIGAND {fmt_count(result.qae_shots_95)}\n")
        handle.write(f"SHOTS_99_GOOD_LIGAND {fmt_count(result.qae_shots_99)}\n")
        handle.write(f"Q_CALLS_PER_SHOT {result.q_calls_per_qpe}\n")
        handle.write(f"TOTAL_Q_CALLS_95 {fmt_count(result.qae_total_q_calls_95)}\n")
        handle.write(f"TOTAL_Q_CALLS_99 {fmt_count(result.qae_total_q_calls_99)}\n")

        handle.write("\nQSEL_GROVER_SCAN\n")
        handle.write(
            "QSEL_SELECTION_MODE "
            + (
                "MIN_R_WITHIN_SHOTS95_BUDGET"
                if result.qsel_shot_budget is not None
                else "MIN_TOTAL_Q_CALLS95"
            )
            + "\n"
        )
        handle.write(
            "QSEL_SHOT_BUDGET "
            + ("NONE" if result.qsel_shot_budget is None else str(result.qsel_shot_budget))
            + "\n"
        )
        handle.write(
            "QSEL_SHOT_BUDGET_MET "
            + (
                "NA"
                if result.qsel_shot_budget_met is None
                else ("YES" if result.qsel_shot_budget_met else "NO")
            )
            + "\n"
        )
        handle.write(f"GROVER_SCAN_MAX {result.grover_max}\n")
        handle.write(f"THEORETICAL_OPT_GROVER {result.theoretical_opt_grover}\n")
        handle.write(f"BEST_GROVER_ITERATIONS {result.grover_iterations}\n")
        handle.write(f"FLAG_PROBABILITY {result.amplified_flag_probability:.16e}\n")
        handle.write(f"GOOD_LIGAND_PROBABILITY {result.good_ligand_probability_qsel:.16e}\n")
        handle.write(f"SHOTS_95_GOOD_LIGAND {fmt_count(result.qsel_shots_95)}\n")
        handle.write(f"SHOTS_99_GOOD_LIGAND {fmt_count(result.qsel_shots_99)}\n")
        handle.write(f"Q_CALLS_PER_SHOT {result.qsel_q_calls_per_shot}\n")
        handle.write(f"TOTAL_Q_CALLS_95 {fmt_count(result.qsel_total_q_calls_95)}\n")
        handle.write(f"TOTAL_Q_CALLS_99 {fmt_count(result.qsel_total_q_calls_99)}\n")

        handle.write("\nRESOURCE\n")
        handle.write(f"COMPARATOR_WORK_QUBITS {result.work_qubits}\n")
        handle.write(f"TOTAL_LOGICAL_QUBITS {result.logical_qubits}\n")
        handle.write(f"Q_CALLS_PER_QPE {result.q_calls_per_qpe}\n")
        handle.write(f"Q_CALLS_PER_THRESHOLD_ORACLE {result.q_calls_per_threshold_oracle}\n")
        handle.write(f"QSEL_Q_CALLS_PER_SHOT {result.qsel_q_calls_per_shot}\n")


def write_ligand_scaling_summary(
    path: Path,
    results: list[ScaleResult],
    *,
    amplitude_threshold: float,
    energy_threshold: float,
    eval_qubits: int,
    resolution_multiple: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as handle:
        handle.write(
            "OCC_COUNT\tOCC_SELECTOR_QUBITS\tTOTAL_SELECTOR_QUBITS\t"
            "SEARCH_STATES\tEXACT_GOOD_CONFIG_STATES\tEXACT_GOOD_CONFIG_FRACTION\t"
            "GOOD_LIGANDS\tGOOD_LIGAND_FRACTION\t"
            "AMPLITUDE_THRESHOLD\tENERGY_THRESHOLD\tEVAL_QUBITS\t"
            "RESOLUTION_MULTIPLE_TARGET\tAMPLITUDE_RESOLUTION\t"
            "THRESHOLD_RESOLUTION_RATIO_ACTUAL\t"
            "QAE_FLAG_PROB\tQAE_GOOD_LIGAND_PROB\tGOOD_LIGAND_GIVEN_FLAG\t"
            "QAE_SHOTS95\tQAE_SHOTS99\tQAE_TOTAL_Q_CALLS95\tQAE_TOTAL_Q_CALLS99\t"
            "GROVER_MAX\tTHEORETICAL_OPT_GROVER\tBEST_GROVER_ITER\t"
            "QSEL_SHOT_BUDGET\tQSEL_SHOT_BUDGET_MET\t"
            "BEST_QSEL_FLAG_PROB\tBEST_QSEL_GOOD_LIGAND_PROB\t"
            "BEST_QSEL_SHOTS95\tBEST_QSEL_SHOTS99\t"
            "BEST_QSEL_TOTAL_Q_CALLS95\tBEST_QSEL_TOTAL_Q_CALLS99\t"
            "Q_CALLS_PER_QPE\tBEST_QSEL_Q_CALLS_PER_SHOT\tWORK_QUBITS\t"
            "TOTAL_LOGICAL_QUBITS\tMIN_ENERGY\tMAX_A0\n"
        )

        for r in results:
            handle.write(
                f"{r.occ_count}\t"
                f"{r.occ_selector_qubits}\t"
                f"{r.total_selector_qubits}\t"
                f"{r.search_states}\t"
                f"{r.exact_good_config_states}\t"
                f"{r.exact_good_config_fraction:.12e}\t"
                f"{r.good_ligands}\t"
                f"{r.good_ligand_fraction:.12e}\t"
                f"{amplitude_threshold:.12e}\t"
                f"{energy_threshold:.12e}\t"
                f"{eval_qubits}\t"
                f"{resolution_multiple}\t"
                f"{amplitude_grid_resolution(eval_qubits):.12e}\t"
                f"{amplitude_threshold / amplitude_grid_resolution(eval_qubits):.12e}\t"
                f"{r.initial_flag_probability:.12e}\t"
                f"{r.good_ligand_probability_qae:.12e}\t"
                f"{r.good_ligand_given_flag:.12e}\t"
                f"{fmt_count(r.qae_shots_95)}\t"
                f"{fmt_count(r.qae_shots_99)}\t"
                f"{fmt_count(r.qae_total_q_calls_95)}\t"
                f"{fmt_count(r.qae_total_q_calls_99)}\t"
                f"{r.grover_max}\t"
                f"{r.theoretical_opt_grover}\t"
                f"{r.grover_iterations}\t"
                f"{'NONE' if r.qsel_shot_budget is None else r.qsel_shot_budget}\t"
                f"{'NA' if r.qsel_shot_budget_met is None else ('YES' if r.qsel_shot_budget_met else 'NO')}\t"
                f"{r.amplified_flag_probability:.12e}\t"
                f"{r.good_ligand_probability_qsel:.12e}\t"
                f"{fmt_count(r.qsel_shots_95)}\t"
                f"{fmt_count(r.qsel_shots_99)}\t"
                f"{fmt_count(r.qsel_total_q_calls_95)}\t"
                f"{fmt_count(r.qsel_total_q_calls_99)}\t"
                f"{r.q_calls_per_qpe}\t"
                f"{r.qsel_q_calls_per_shot}\t"
                f"{r.work_qubits}\t"
                f"{r.logical_qubits}\t"
                f"{r.min_energy:.12e}\t"
                f"{r.max_a0:.12e}\n"
            )


def write_ligand_grover_shots(
    path: Path,
    rows: list[GroverPoint],
) -> None:
    """Write the full Grover-iteration versus ligand-retrieval resource curve."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(
            "OCC_COUNT\tGROVER_ITERATIONS\tAMPLIFIED_FLAG_PROB\t"
            "GOOD_LIGAND_PROB\tSHOTS95\tSHOTS99\tQ_CALLS_PER_SHOT\t"
            "TOTAL_Q_CALLS95\tTOTAL_Q_CALLS99\n"
        )
        for row in rows:
            handle.write(
                f"{row.occ_count}\t"
                f"{row.grover_iterations}\t"
                f"{row.amplified_flag_probability:.12e}\t"
                f"{row.good_ligand_probability:.12e}\t"
                f"{fmt_count(row.shots_95)}\t"
                f"{fmt_count(row.shots_99)}\t"
                f"{row.q_calls_per_shot}\t"
                f"{fmt_count(row.total_q_calls_95)}\t"
                f"{fmt_count(row.total_q_calls_99)}\n"
            )


def write_occ_mapping(path: Path, entries: list[OccEntry]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("OCC_SELECTOR\tSOURCE\tOCC\n")
        for entry in entries:
            handle.write(
                f"{entry.selector}\t{entry.source}\t{entry.name}\n"
            )


def write_circuit_svg(circuit: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        figure = circuit.draw(
            output="mpl",
            fold=-1,
            reverse_bits=True,
        )
    except Exception as exc:
        raise RuntimeError(
            "failed to render circuit.svg. Install Qiskit visualization "
            "dependencies (matplotlib and pylatexenc), or use --skip-circuit."
        ) from exc

    try:
        figure.savefig(path, format="svg", bbox_inches="tight")
    finally:
        try:
            import matplotlib.pyplot as plt
            plt.close(figure)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Optional validations carried over from the old program
# ---------------------------------------------------------------------------


def validate_pose_fft() -> float:
    rng = np.random.default_rng(20260811)
    receptor = rng.normal(size=PHYSICAL_DATA_SIZE)
    ligand = rng.normal(size=PHYSICAL_DATA_SIZE)
    receptor /= np.linalg.norm(receptor)
    ligand /= np.linalg.norm(ligand)

    columns = build_pose_columns()
    map_spectrum = np.fft.rfftn(
        receptor.reshape(TYPE_SIZE, GRID_SIZE, GRID_SIZE, GRID_SIZE),
        axes=(1, 2, 3),
    )

    calculated = compute_occ_pose_overlaps_numpy(
        [map_spectrum, map_spectrum],
        ligand,
        columns,
    )[0]

    receptor_4d = receptor.reshape(TYPE_SIZE, GRID_SIZE, GRID_SIZE, GRID_SIZE)
    ligand_4d = ligand.reshape(TYPE_SIZE, GRID_SIZE, GRID_SIZE, GRID_SIZE)

    sample_poses = [0, 1, 31, 32, 511, 8191, 12004, 16383]
    maximum_error = 0.0

    for pose in sample_poses:
        rotation = int(columns.rotation_selector[pose])
        rotated = apply_rotation_numpy(ligand_4d, rotation)
        shifted = np.roll(
            rotated,
            shift=(
                int(columns.shift_z[pose]),
                int(columns.shift_y[pose]),
                int(columns.shift_x[pose]),
            ),
            axis=(1, 2, 3),
        )
        expected = float(np.vdot(receptor_4d, shifted).real)
        maximum_error = max(
            maximum_error,
            abs(expected - float(calculated[pose])),
        )

    marker = np.arange(27).reshape(1, 3, 3, 3)
    groups: dict[bytes, list[int]] = {}
    for rotation in range(ROTATION_STATES):
        key = apply_rotation_numpy(marker, rotation).tobytes()
        groups.setdefault(key, []).append(rotation)

    duplicates = sorted(group for group in groups.values() if len(group) > 1)
    expected_duplicates = sorted(
        [
            [4, 21],
            [5, 28],
            [6, 31],
            [7, 22],
            [12, 29],
            [13, 20],
            [14, 23],
            [15, 30],
        ]
    )

    if len(groups) != UNIQUE_ROTATIONS or duplicates != expected_duplicates:
        raise AssertionError(
            f"rotation-selector validation failed: {duplicates}"
        )
    if maximum_error > 1e-10:
        raise AssertionError(
            f"pose FFT validation failed: max error={maximum_error:.6e}"
        )

    return maximum_error


def validate_small_qiskit_qae() -> float:
    try:
        from qiskit import AncillaRegister, QuantumCircuit, QuantumRegister
        from qiskit.circuit.library import grover_operator, phase_estimation
        from qiskit.quantum_info import Statevector
        from qiskit.circuit.library import IntegerComparator
    except ImportError as exc:
        raise ImportError(
            "Qiskit is required for --validate-small"
        ) from exc

    state_qubits = 2
    eval_qubits = 4
    amplitude = 0.2
    threshold = 0.3
    comparator = comparator_spec(threshold, eval_qubits)

    other = math.sqrt(1.0 - amplitude**2)
    unitary = np.asarray(
        [
            [amplitude, -other, 0.0, 0.0],
            [other, amplitude, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.complex128,
    )

    preparation = QuantumCircuit(state_qubits, name="U")
    preparation.unitary(unitary, range(state_qubits), label="U")

    oracle = QuantumCircuit(state_qubits, name="S_good")
    oracle.x(range(state_qubits))
    target = state_qubits - 1
    oracle.h(target)
    oracle.mcx(list(range(state_qubits - 1)), target)
    oracle.h(target)
    oracle.x(range(state_qubits))

    grover = grover_operator(
        oracle=oracle,
        state_preparation=preparation,
    )

    qpe_gate = phase_estimation(
        num_evaluation_qubits=eval_qubits,
        unitary=grover,
        name="QPE",
    ).to_gate(label="QPE")

    low_comparator = IntegerComparator(
        num_state_qubits=eval_qubits,
        value=comparator.low_limit,
        geq=False,
    )
    low_work = low_comparator.num_qubits - eval_qubits - 1

    high_comparator = None
    high_work = 0
    if comparator.high_limit < comparator.modulus:
        high_comparator = IntegerComparator(
            num_state_qubits=eval_qubits,
            value=comparator.high_limit,
            geq=True,
        )
        high_work = high_comparator.num_qubits - eval_qubits - 1

    comparator_work = max(low_work, high_work)

    evaluation = QuantumRegister(eval_qubits, "eval")
    state = QuantumRegister(state_qubits, "state")
    flags = QuantumRegister(1, "flag")
    work = AncillaRegister(comparator_work, "work") if comparator_work else None

    registers = [evaluation, state, flags]
    if work is not None:
        registers.append(work)
    circuit = QuantumCircuit(*registers)

    circuit.append(preparation.to_gate(label="U"), list(state))
    circuit.append(qpe_gate, list(evaluation) + list(state))

    work_list = list(work) if work is not None else []
    comparator_state = list(reversed(evaluation))

    circuit.append(
        low_comparator.to_gate(label=f"y<{comparator.low_limit}"),
        comparator_state + [flags[0]] + work_list[:low_work],
    )

    if high_comparator is not None:
        circuit.append(
            high_comparator.to_gate(label=f"y>={comparator.high_limit}"),
            comparator_state + [flags[0]] + work_list[:high_work],
        )

    statevector = Statevector.from_instruction(circuit)
    flag_index = circuit.find_bit(flags[0]).index
    actual = float(statevector.probabilities(qargs=[flag_index])[1])
    expected = float(
        qae_flag_probability(
            np.asarray([amplitude]),
            comparator,
        )[0]
    )

    error = abs(actual - expected)
    if error > 1e-10:
        raise AssertionError(
            "small Qiskit QAE validation failed: "
            f"actual={actual:.16e}, expected={expected:.16e}, "
            f"error={error:.6e}"
        )
    return error


# ---------------------------------------------------------------------------
# Main calculation
# ---------------------------------------------------------------------------


def resolve_thresholds(
    *,
    amplitude_arg: float | None,
    energy_arg: float | None,
    Lm: float,
    Lo: float,
    shift: float,
) -> tuple[float, float]:
    denominator = Lm * Lo

    if amplitude_arg is None and energy_arg is None:
        amplitude = DEFAULT_AMPLITUDE_THRESHOLD
        energy = amplitude * denominator - shift
    elif amplitude_arg is not None:
        amplitude = float(amplitude_arg)
        energy = amplitude * denominator - shift
    else:
        assert energy_arg is not None
        energy = float(energy_arg)
        amplitude = (energy + shift) / denominator

    if not math.isfinite(amplitude) or not 0.0 < amplitude < 1.0:
        raise ValueError(
            "the chosen threshold maps to an invalid A0 threshold. "
            f"A0_threshold={amplitude}; it must satisfy 0 < A0 < 1. "
            f"Lm={Lm}, Lo={Lo}, shift={shift}, energy_threshold={energy}"
        )

    return amplitude, energy


def checkpoint_folder(output_root: Path, occ_count: int) -> Path:
    return output_root / f"occ_{occ_count:03d}"


def parse_grover_max(value: str) -> int | None:
    text = value.strip().lower()
    if text == "auto":
        return None
    try:
        number = int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "--grover-max must be 'auto' or a nonnegative integer"
        ) from exc
    if number < 0:
        raise argparse.ArgumentTypeError(
            "--grover-max must be 'auto' or a nonnegative integer"
        )
    return number


def _unique_preserve(values):
    seen = set()
    output = []
    for value in values:
        if value not in seen:
            seen.add(value)
            output.append(value)
    return output


def discover_receptor_dirs(input_path: Path) -> list[Path]:
    """Accept either one receptor directory or a parent containing receptors."""
    root = input_path.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"input directory not found: {root}")

    if (root / "map").is_dir() and (root / "occ").is_dir():
        return [root]

    receptors = [
        child
        for child in sorted(root.iterdir(), key=lambda p: p.name)
        if child.is_dir()
        and (child / "map").is_dir()
        and (child / "occ").is_dir()
    ]
    if not receptors:
        raise FileNotFoundError(
            f"no normalized receptor directories found under {root}; "
            "each receptor must contain map/ and occ/"
        )
    return receptors


def format_float_token(value: float) -> str:
    """Filesystem-friendly compact float, e.g. 1e-05 -> 1e-5."""
    text = f"{value:.12g}"
    text = re.sub(r"e([+-])0+(\d+)$", r"e\1\2", text)
    text = text.replace("e+", "e")
    return text


def format_budget_token(value: int | None) -> str:
    if value is None:
        return "none"
    if value > 0:
        exponent = int(round(math.log10(value)))
        if 10 ** exponent == value:
            return f"1e{exponent}"
    return str(value)


def amplitude_grid_resolution(eval_qubits: int) -> float:
    """Smallest nonzero A0 grid value for an m-qubit QPE register."""
    if not 1 <= eval_qubits <= 30:
        raise ValueError("evaluation qubits must be between 1 and 30")
    return math.sin(math.pi / float(1 << eval_qubits))


def paired_eval_qubits(base_eval_qubits: int, resolution_multiple: int) -> int:
    """
    Return the actual QPE-register size for a bin-aligned threshold experiment.

    The base register size m0 defines the threshold exactly as

        A0_threshold = sin(pi / 2^m0).

    For K in {1, 2, 4, 8}, keep that same threshold and increase the actual
    QPE register by log2(K):

        m = m0 + log2(K).

    Therefore the threshold phase lies exactly at QPE bin K:

        2^m * asin(A0_threshold) / pi = K.
    """
    if not 1 <= base_eval_qubits <= 30:
        raise ValueError("base evaluation qubits must be between 1 and 30")
    if resolution_multiple not in DEFAULT_RESOLUTION_MULTIPLES:
        raise ValueError(
            f"resolution multiple must be one of {DEFAULT_RESOLUTION_MULTIPLES}"
        )

    shift = int(math.log2(resolution_multiple))
    if (1 << shift) != resolution_multiple:
        raise ValueError("resolution multiple must be a power of two")

    eval_qubits = base_eval_qubits + shift
    if eval_qubits > 30:
        raise ValueError(
            "requested base-m/resolution pairing requires more than 30 "
            f"evaluation qubits: base_m={base_eval_qubits}, "
            f"multiple={resolution_multiple}"
        )
    return eval_qubits


def experiment_output_root(
    output_root: Path,
    pdbid: str,
    amplitude_threshold: float,
    eval_qubits: int,
    resolution_multiple: int,
    qsel_shot_budget: int | None,
) -> Path:
    name = (
        f"thr_{format_float_token(amplitude_threshold)}_"
        f"mult_{resolution_multiple}_eval_{eval_qubits}"
    )
    if qsel_shot_budget is not None:
        name += f"_shot_{format_budget_token(qsel_shot_budget)}"
    return output_root / pdbid / name


def write_ligand_batch_summary(
    path: Path,
    rows: list[tuple[str, ExperimentConfig, ScaleResult]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(
            "PDBID\tAMPLITUDE_THRESHOLD\tENERGY_THRESHOLD\tEVAL_QUBITS\t"
            "RESOLUTION_MULTIPLE_TARGET\tAMPLITUDE_RESOLUTION\t"
            "THRESHOLD_RESOLUTION_RATIO_ACTUAL\t"
            "OCC_COUNT\tSEARCH_STATES\tEXACT_GOOD_CONFIG_STATES\t"
            "EXACT_GOOD_CONFIG_FRACTION\tGOOD_LIGANDS\tGOOD_LIGAND_FRACTION\t"
            "QAE_FLAG_PROB\tQAE_GOOD_LIGAND_PROB\tGOOD_LIGAND_GIVEN_FLAG\t"
            "BAD_LIGAND_GIVEN_FLAG\tQAE_SHOTS95\tQAE_SHOTS99\t"
            "QAE_TOTAL_Q_CALLS95\tQAE_TOTAL_Q_CALLS99\t"
            "THEORETICAL_OPT_GROVER\tBEST_GROVER_ITER\tQSEL_SHOTS95\t"
            "QSEL_SHOTS99\tQSEL_TOTAL_Q_CALLS95\tQSEL_TOTAL_Q_CALLS99\t"
            "QSEL_SHOT_BUDGET\tQSEL_SHOT_BUDGET_MET\tQ_CALLS_PER_QPE\t"
            "QSEL_Q_CALLS_PER_SHOT\tTOTAL_LOGICAL_QUBITS\tMIN_ENERGY\tMAX_A0\n"
        )
        for pdbid, cfg, result in rows:
            bad_ligand = 1.0 - result.good_ligand_given_flag
            handle.write(
                f"{pdbid}\t"
                f"{cfg.amplitude_threshold:.12e}\t"
                f"{cfg.energy_threshold:.12e}\t"
                f"{cfg.eval_qubits}\t"
                f"{cfg.resolution_multiple}\t"
                f"{amplitude_grid_resolution(cfg.eval_qubits):.12e}\t"
                f"{cfg.amplitude_threshold / amplitude_grid_resolution(cfg.eval_qubits):.12e}\t"
                f"{result.occ_count}\t"
                f"{result.search_states}\t"
                f"{result.exact_good_config_states}\t"
                f"{result.exact_good_config_fraction:.12e}\t"
                f"{result.good_ligands}\t"
                f"{result.good_ligand_fraction:.12e}\t"
                f"{result.initial_flag_probability:.12e}\t"
                f"{result.good_ligand_probability_qae:.12e}\t"
                f"{result.good_ligand_given_flag:.12e}\t"
                f"{bad_ligand:.12e}\t"
                f"{fmt_count(result.qae_shots_95)}\t"
                f"{fmt_count(result.qae_shots_99)}\t"
                f"{fmt_count(result.qae_total_q_calls_95)}\t"
                f"{fmt_count(result.qae_total_q_calls_99)}\t"
                f"{result.theoretical_opt_grover}\t"
                f"{result.grover_iterations}\t"
                f"{fmt_count(result.qsel_shots_95)}\t"
                f"{fmt_count(result.qsel_shots_99)}\t"
                f"{fmt_count(result.qsel_total_q_calls_95)}\t"
                f"{fmt_count(result.qsel_total_q_calls_99)}\t"
                f"{'NONE' if result.qsel_shot_budget is None else result.qsel_shot_budget}\t"
                f"{'NA' if result.qsel_shot_budget_met is None else ('YES' if result.qsel_shot_budget_met else 'NO')}\t"
                f"{result.q_calls_per_qpe}\t"
                f"{result.qsel_q_calls_per_shot}\t"
                f"{result.logical_qubits}\t"
                f"{result.min_energy:.12e}\t"
                f"{result.max_a0:.12e}\n"
            )


# ---------------------------------------------------------------------------
# Real-data 19-qubit full-overlap verification
# ---------------------------------------------------------------------------


def test_full_overlap_output(
    input_dir: Path,
    output_dir: Path,
    max_occ: int,
    norm_atol: float,
    device: str,
) -> None:
    """Check real 19-qubit MAP/OCC overlaps against production FFT A0.

    Uses the original loaders and pose/FFT functions.  The independent
    reference explicitly transforms all physical OCC amplitudes with
    np.roll(), leaves the complete auxiliary half unchanged, and takes the
    direct dot product of the FULL 524288-element vectors.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "full_overlap_check.txt"
    tol = 1e-10
    rows: list[str] = []
    errors: list[str] = []
    checked = 0
    worst_phys = 0.0
    worst_a0 = 0.0
    worst_aux = 0.0

    receptor_dirs = discover_receptor_dirs(input_dir)
    backend = select_backend(device)
    columns = build_pose_columns()
    translation_ids = (0, 291, 292, 293, 356, 511)
    rotation_ids = (0, 1, 4, 31)

    # Independently decode the nine selector bits, not via FFT indexing.
    all_translations = set()
    for trans in range(TRANSLATION_STATES):
        dx = (trans & 7) - 4
        dy = ((trans >> 3) & 7) - 4
        dz = ((trans >> 6) & 7) - 4
        all_translations.add((dz, dy, dx))
        if ((int(columns.shift_z_512[trans]),
             int(columns.shift_y_512[trans]),
             int(columns.shift_x_512[trans])) != (dz, dy, dx)):
            raise AssertionError(f"TRANS selector {trans} encodes the wrong shift")
    if len(all_translations) != 512:
        raise AssertionError("512 TRANS selectors do not encode 512 unique translations")

    rows.extend([
        "REAL-DATA 19-QUBIT FULL-OVERLAP VALIDATION",
        f"Input       : {input_dir.resolve()}",
        f"Output      : {report_path.resolve()}",
        f"Backend     : {backend.name}",
        f"Tolerance   : {tol:.1e} absolute",
        "Full length : 524288 float64 amplitudes (19 qubits)",
        "Data layout : 8 channels x 32 x 32 x 32 physical + auxiliary",
        "Reference   : np.roll(original ROT OCC) + UNCHANGED auxiliary,",
        "              then np.dot(entire 19-qubit MAP, entire posed OCC)",
        "Production  : original compute_occ_pose_overlaps() (FFT) + shift/(Lm*Lo)",
        "No normalization or modification of the input binary files.",
        "The selector mapping was checked independently for all 512 shifts.",
        "Selected ROT : " + " ".join(map(str, rotation_ids)),
        "Selected TRANS: " + " ".join(map(str, translation_ids)),
        "Selected OCC : 0 (native), 1 (first decoy), and last in --max-occ,",
        "               with duplicates omitted when max-occ < 3.",
        "",
    ])

    for receptor_dir in receptor_dirs:
        rows.append("=" * 125)
        rows.append(f"RECEPTOR {receptor_dir.name}")
        try:
            map_dir = receptor_dir / "map"
            occ_dir = receptor_dir / "occ"
            pdbid, lm, energy_shift = read_map_len(map_dir / "len")
            occ_lengths = read_occ_len(occ_dir / "len")
            entries = collect_occ_entries(occ_dir, pdbid)
            lo = validate_occ_lengths(entries, occ_lengths)
            maps = load_maps(map_dir, pdbid, lm, energy_shift, norm_atol)
            denominator = lm * lo
            if not math.isfinite(denominator) or denominator <= 0:
                raise ValueError(f"invalid normalization product Lm*Lo: {denominator}")
            expected_aux = energy_shift / denominator
            for receptor in maps:
                if float(np.max(np.abs(receptor.amplitudes[AUX0 + 3:]))) > tol:
                    raise AssertionError(f"MAP {receptor.name} has nonzero auxiliary tail")
            map_spectra = prepare_map_spectra(maps, backend)
            occ_positions = sorted({0, min(1, max_occ - 1), max_occ - 1})
            rows.append(f"PDBID={pdbid} Lm={lm:.12e} Lo={lo:.12e} shift={energy_shift:.12e}")
            rows.append(f"Expected AUX dot = shift/(Lm*Lo) = {expected_aux:.12e}")
        except (OSError, ValueError, KeyError, AssertionError, ArithmeticError) as exc:
            errors.append(f"{receptor_dir.name}: input setup: {exc}")
            rows.append(f"FAIL: {exc}")
            rows.append("")
            continue

        for position in occ_positions:
            entry = entries[position]
            rows.append("")
            rows.append(f"OCC {entry.selector}: {entry.name} ({entry.path.name})")
            try:
                occ_full = read_full_vector(entry.path, norm_atol)
                validate_occ_auxiliary(occ_full, entry.path, lo)
                if float(np.max(np.abs(occ_full[AUX0 + 3:]))) > tol:
                    raise AssertionError("OCC has nonzero auxiliary tail")
                occ_aux = occ_full[AUX0:]
                aux_per_map = [
                    float(np.dot(receptor.amplitudes[AUX0:], occ_aux))
                    for receptor in maps
                ]
                occ_4d = occ_full[:PHYSICAL_DATA_SIZE].reshape(
                    TYPE_SIZE, GRID_SIZE, GRID_SIZE, GRID_SIZE
                )
                fft_overlap = compute_occ_pose_overlaps(
                    map_spectra, np.ascontiguousarray(occ_full[:PHYSICAL_DATA_SIZE]),
                    columns, backend
                )
            except (OSError, ValueError, KeyError, ArithmeticError, AssertionError) as exc:
                errors.append(f"{pdbid} OCC {entry.name}: input/FFT: {exc}")
                rows.append(f"FAIL: {exc}")
                continue

            rows.append(
                f"{'MAP':>3} {'ROT':>3} {'TR':>3} {'dz':>3} {'dy':>3} {'dx':>3} "
                f"{'Full_Dot':>15} {'Phys_Dot':>15} {'Aux_Dot':>13} "
                f"{'A0_FFT':>15} {'dPhys':>10} {'dA0':>10} {'Status':>6}"
            )
            for rot in rotation_ids:
                rotated = apply_rotation_numpy(occ_4d, rot)
                for trans in translation_ids:
                    dz = ((trans >> 6) & 7) - 4
                    dy = ((trans >> 3) & 7) - 4
                    dx = (trans & 7) - 4
                    # Independent direct cyclic translation; no FFT and
                    # no use of production shift columns for this reference.
                    shifted = np.roll(rotated, (dz, dy, dx), axis=(1, 2, 3))
                    posed = occ_full.copy()
                    posed[:PHYSICAL_DATA_SIZE] = shifted.reshape(-1)
                    pose_index = (trans << ROTATION_SELECTOR_QUBITS) | rot
                    for map_index, receptor in enumerate(maps):
                        full_dot = float(np.dot(receptor.amplitudes, posed))
                        physical_direct = float(np.dot(receptor.physical,
                                                       posed[:PHYSICAL_DATA_SIZE]))
                        aux_direct = aux_per_map[map_index]
                        fft_physical = float(fft_overlap[map_index, pose_index])
                        energy = fft_physical * denominator
                        a0_fft = (energy + energy_shift) / denominator
                        d_phys = abs(physical_direct - fft_physical)
                        d_a0 = abs(full_dot - a0_fft)
                        d_aux = abs(aux_direct - expected_aux)
                        worst_phys = max(worst_phys, d_phys)
                        worst_a0 = max(worst_a0, d_a0)
                        worst_aux = max(worst_aux, d_aux)
                        checks_ok = (
                            d_phys <= tol and d_a0 <= tol and d_aux <= tol
                            and abs(full_dot - physical_direct - aux_direct) <= tol
                            and np.array_equal(posed[AUX0:], occ_full[AUX0:])
                            and abs(float(np.linalg.norm(posed)) - 1.0) <= norm_atol
                        )
                        checked += 1
                        rows.append(
                            f"{map_index:3d} {rot:3d} {trans:3d} "
                            f"{dz:3d} {dy:3d} {dx:3d} "
                            f"{full_dot:15.7e} {physical_direct:15.7e} "
                            f"{aux_direct:13.5e} {a0_fft:15.7e} "
                            f"{d_phys:10.2e} {d_a0:10.2e} "
                            f"{'PASS' if checks_ok else 'FAIL':>6}"
                        )
                        if not checks_ok:
                            errors.append(
                                f"{pdbid}, OCC={entry.name}, MAP={map_index}, "
                                f"ROT={rot}, TRANS={trans}, dPhys={d_phys:.6e}, "
                                f"dA0={d_a0:.6e}, dAux={d_aux:.6e}"
                            )
        rows.append("")

    summary = [
        "CHECK SUMMARY",
        f"Result         : {'PASS' if not errors else 'FAIL'}",
        f"Receptors      : {len(receptor_dirs)}",
        f"Cases checked  : {checked}",
        f"Distinct TRANS : {len(all_translations)} of {TRANSLATION_STATES}",
        f"Max |PhysDirect - FFT| : {worst_phys:.12e}",
        f"Max |FullDot - A0_FFT| : {worst_a0:.12e}",
        f"Max |AuxDot - Shift/(LmLo)| : {worst_aux:.12e}",
        f"Errors         : {len(errors)}",
    ]
    if errors:
        summary.extend(["", "FAILURES"] + [f"- {x}" for x in errors])
    report_path.write_text("\n".join(summary + [""] + rows) + "\n", encoding="utf-8")
    print("\n".join(summary[:9]))
    print(f"Report         : {report_path}")
    if errors:
        raise AssertionError(f"full-overlap verification failed, see {report_path}")


# ---------------------------------------------------------------------------
# Small coherent-selector QAE/QPE and QSEL verification
# ---------------------------------------------------------------------------


def test_qae_selector_output(output_dir: Path) -> None:
    """Cross-check this program's ideal QAE/QSEL probabilities with a tiny statevector.

    Uses two normalized, 8-element MAP/OCC states with orthogonal padding,
    followed by selector-controlled 2x2 Grover operators and a 6-qubit QPE
    register.  This does NOT synthesize the production 19-qubit oracle.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    report = output_dir / "qae_selector_check.txt"
    tol = 1e-11

    # The auxiliary layout is the same as energy.py, but scaled down to
    # four physical basis states and four auxiliary slots.
    physical_size = 4
    aux0 = physical_size
    Lm = 2.0
    Lo = 2.0
    shift = 0.08
    wanted_amplitudes = np.array([0.11, 0.33], dtype=np.float64)

    receptor = np.zeros(8, dtype=np.float64)
    receptor[0:2] = [0.6, 0.1]
    receptor[aux0] = shift / Lm
    receptor[aux0 + 1] = math.sqrt(1.0 - np.dot(receptor, receptor))

    occupancies = []
    for target in wanted_amplitudes:
        occ = np.zeros(8, dtype=np.float64)
        occ[0] = (target - shift / (Lm * Lo)) / receptor[0]
        occ[aux0] = 1.0 / Lo
        occ[aux0 + 2] = math.sqrt(1.0 - np.dot(occ, occ))
        occupancies.append(occ)

    amplitudes = []
    source_rows = []
    for selector, occ in enumerate(occupancies):
        norm_map = float(np.linalg.norm(receptor))
        norm_occ = float(np.linalg.norm(occ))
        physical_overlap = float(np.dot(receptor[:physical_size], occ[:physical_size]))
        auxiliary_overlap = float(np.dot(receptor[physical_size:], occ[physical_size:]))
        full_overlap = float(np.dot(receptor, occ))
        energy = physical_overlap * Lm * Lo
        a0 = (energy + shift) / (Lm * Lo)
        if not np.isclose(norm_map, 1, rtol=0, atol=tol):
            raise AssertionError("MAP normalization failed")
        if not np.isclose(norm_occ, 1, rtol=0, atol=tol):
            raise AssertionError("OCC normalization failed")
        if abs(full_overlap - a0) > tol:
            raise AssertionError("Full-state overlap does not match A0")
        if abs(a0 - wanted_amplitudes[selector]) > tol:
            raise AssertionError("A0 does not match known input")
        amplitudes.append(a0)
        source_rows.append((selector, norm_map, norm_occ, physical_overlap,
                            auxiliary_overlap, full_overlap, energy, a0))
    amplitudes = np.asarray(amplitudes, dtype=np.float64)

    base_m = 4
    multiple = 4
    eval_qubits = paired_eval_qubits(base_m, multiple)
    modulus = 1 << eval_qubits
    threshold = math.sin(math.pi / (1 << base_m))
    comparator = comparator_spec(threshold, eval_qubits)
    accepted = (np.arange(modulus) < comparator.low_limit) | (
        np.arange(modulus) >= comparator.high_limit
    )
    exact_good = amplitudes < threshold
    if (int(np.count_nonzero(exact_good)) != 1 or
            comparator.low_limit != multiple or
            comparator.high_limit != modulus - multiple + 1):
        raise AssertionError("Unexpected toy threshold or accepted phase bins")

    # Q_j is CONSTRUCTED from the state-preparation and marking reflections,
    # not from energy.py's finite-QPE probability formula.
    s_zero = np.diag([-1.0, 1.0])
    s_good = np.diag([1.0, -1.0])
    q_operators = []
    prepared = []
    for amplitude in amplitudes:
        remaining = math.sqrt(1.0 - amplitude * amplitude)
        unitary = np.array([[remaining, -amplitude],
                            [amplitude, remaining]], dtype=np.float64)
        psi = unitary @ np.array([1.0, 0.0])
        q = -unitary @ s_zero @ unitary.T @ s_good
        if not np.allclose(q.T @ q, np.eye(2), atol=tol, rtol=0):
            raise AssertionError("Constructed Grover Q is not unitary")
        eigenvalues = np.linalg.eigvals(q)
        theta = math.asin(amplitude)
        expected_eigenvalues = [np.exp(2j * theta), np.exp(-2j * theta)]
        if max(min(abs(e - v) for e in eigenvalues)
               for v in expected_eigenvalues) > tol:
            raise AssertionError("Grover eigenphases differ from asin(A0)/pi")
        prepared.append(psi)
        q_operators.append(q)

    # Explicit joint statevector |evaluation>|selector>|data>.
    # Equal initial selector weights and an m-qubit Hadamard register.
    # Q acts as sum_j |j><j| tensor Q_j, thus preserving each selector.
    state = np.zeros((modulus, 2, 2), dtype=np.complex128)
    for selector in range(2):
        q_power = np.eye(2)
        for time_index in range(modulus):
            state[time_index, selector, :] = (
                q_power @ prepared[selector] / math.sqrt(2.0 * modulus)
            )
            q_power = q_operators[selector] @ q_power

    # Inverse QFT across the evaluation register, independently of energy.py's
    # sinc-based analytic QPE evaluation (numpy FFT implements this linear map).
    after_qpe = np.fft.fft(state, axis=0, norm="ortho")
    if abs(float(np.vdot(after_qpe, after_qpe).real) - 1.0) > tol:
        raise AssertionError("Joint QPE state is not normalized")

    simulated_bins = 2.0 * np.sum(np.abs(after_qpe) ** 2, axis=2)
    analytic_bins = np.stack([
        qpe_probability_on_bins(amplitudes, np.array([y]), modulus)
        for y in range(modulus)
    ], axis=0)
    max_bin_error = float(np.max(np.abs(simulated_bins - analytic_bins)))
    simulated_flags = np.sum(simulated_bins[accepted, :], axis=0)
    analytic_flags = qae_flag_probability(amplitudes, comparator)
    max_flag_error = float(np.max(np.abs(simulated_flags - analytic_flags)))
    if max_bin_error > tol or max_flag_error > tol:
        raise AssertionError(
            f"Finite-QPE mismatch: bin={max_bin_error:.3e}, "
            f"FLAG={max_flag_error:.3e}"
        )

    # Construct the full-state amplitude amplification, without calling the
    # analytical formula for the simulation.  The good subspace is FLAG=1.
    initial = after_qpe.reshape(-1)
    marked = np.broadcast_to(accepted[:, None, None], after_qpe.shape).reshape(-1)
    good_marked = np.broadcast_to(
        (accepted[:, None, None] & exact_good[None, :, None]),
        after_qpe.shape,
    ).reshape(-1)
    initial_flag = float(np.sum(np.abs(initial[marked]) ** 2))
    initial_joint = float(np.sum(np.abs(initial[good_marked]) ** 2))
    eta = initial_joint / initial_flag
    if abs(initial_flag - float(np.mean(analytic_flags))) > tol:
        raise AssertionError("Selector-marginal FLAG probability mismatch")
    if abs(initial_joint - float(np.sum(analytic_flags[exact_good]) / len(amplitudes))) > tol:
        raise AssertionError("Joint true-good/FLAG probability mismatch")

    qsel_rows = []
    amplified = initial.copy()
    for r in range(5):
        probability_flag = float(np.sum(np.abs(amplified[marked]) ** 2))
        probability_joint = float(np.sum(np.abs(amplified[good_marked]) ** 2))
        predicted_flag = amplified_success_probability(initial_flag, r)
        predicted_joint = predicted_flag * eta
        error_flag = abs(probability_flag - predicted_flag)
        error_joint = abs(probability_joint - predicted_joint)
        if error_flag > tol or error_joint > tol:
            raise AssertionError(
                f"QSEL r={r}: FLAG error={error_flag:.3e}, "
                f"joint error={error_joint:.3e}"
            )
        qsel_rows.append((r, probability_flag, predicted_flag,
                          probability_joint, predicted_joint,
                          probability_joint / probability_flag if probability_flag else 0.0))

        # One ideal Grover iteration: mark FLAG, then reflect around the
        # full initial state.  These are statevector operations, not formulas.
        amplified[marked] *= -1.0
        amplified = 2.0 * initial * np.vdot(initial, amplified) - amplified

    # Intentionally incorrect non-controlled Q illustrates the distinction:
    # applying Q_0 to both selectors erases selector-dependent QPE phase data.
    wrong = np.zeros_like(state)
    for selector in range(2):
        q_power = np.eye(2)
        for time_index in range(modulus):
            wrong[time_index, selector, :] = (
                q_power @ prepared[selector] / math.sqrt(2.0 * modulus)
            )
            q_power = q_operators[0] @ q_power
    wrong_after = np.fft.fft(wrong, axis=0, norm="ortho")
    wrong_bins = 2.0 * np.sum(np.abs(wrong_after) ** 2, axis=2)
    wrong_flags = np.sum(wrong_bins[accepted, :], axis=0)

    lines = [
        "SMALL SELECTOR-CONTROLLED QAE/QPE AND QSEL VALIDATION",
        "Result      : PASS",
        "Scope       : two selectors, toy 8-amplitude MAP/OCC vectors,",
        "              one data qubit, m=6 evaluation qubits, ideal operators",
        "Limitation  : does NOT validate the actual 19-qubit preparation/oracle,",
        "              compiled SELECT/ROT/TRANS gates, or hardware noise",
        "Input files : none (known synthetic vectors)",
        "",
        f"Lm          : {Lm:.12f}",
        f"Lo          : {Lo:.12f}",
        f"Shift       : {shift:.12f}",
        f"m0          : {base_m}",
        f"k           : {multiple}",
        f"m           : {eval_qubits}",
        f"A0_thr      : {threshold:.12e}",
        f"QCOMP bins  : y < {comparator.low_limit} OR y >= {comparator.high_limit}",
        "",
        "1. FULL MAP/OCC VECTOR OVERLAPS",
        f"{'Sel':>3} {'MAP_Norm':>12} {'OCC_Norm':>12} {'Phys_Dot':>13} "
        f"{'Aux_Dot':>13} {'Full_Dot':>13} {'Energy':>13} {'A0':>13} {'Good':>5}",
    ]
    for sel, mn, on, phys, aux, full, energy, a0 in source_rows:
        lines.append(f"{sel:3d} {mn:12.9f} {on:12.9f} {phys:13.9f} "
                     f"{aux:13.9f} {full:13.9f} {energy:13.9f} "
                     f"{a0:13.9f} {str(bool(exact_good[sel])):>5}")
    lines.extend([
        "",
        "2. SELECTOR-CONTROLLED QPE AND ORIGINAL energy.py ANALYTIC FLAG",
        f"{'Sel':>3} {'A0':>13} {'P_FLAG_SV':>16} {'P_FLAG_PY':>16} "
        f"{'Abs_Error':>13} {'NoCtrl_Q':>13}",
    ])
    for sel in range(2):
        lines.append(
            f"{sel:3d} {amplitudes[sel]:13.9f} {simulated_flags[sel]:16.12e} "
            f"{analytic_flags[sel]:16.12e} "
            f"{abs(simulated_flags[sel]-analytic_flags[sel]):13.3e} "
            f"{wrong_flags[sel]:13.9e}"
        )
    lines.extend([
        f"Max bin probability error  : {max_bin_error:.6e}",
        f"Max FLAG probability error : {max_flag_error:.6e}",
        f"P(FLAG) at r=0           : {initial_flag:.12e}",
        f"P(G AND FLAG) at r=0     : {initial_joint:.12e}",
        f"eta=P(G|FLAG)           : {eta:.12e}",
        "NoCtrl_Q = illustration of mistakenly applying Q_0 to BOTH selectors.",
        "",
        "3. IDEAL FULL-STATE QSEL, COMPARED WITH energy.py ANALYTIC MODEL",
        f"{'r':>3} {'P_FLAG_SV':>16} {'P_FLAG_PY':>16} "
        f"{'P_GOOD_SV':>16} {'P_GOOD_PY':>16} {'eta_SV':>13}",
    ])
    for r, psv, ppy, jsv, jpy, eta_sv in qsel_rows:
        lines.append(f"{r:3d} {psv:16.12e} {ppy:16.12e} "
                     f"{jsv:16.12e} {jpy:16.12e} {eta_sv:13.9e}")
    lines.extend([
        "",
        "ALL TESTS PASS within 1e-11 absolute error.",
        "This verifies mathematical consistency of the ideal conditional-Q model",
        "with energy.py's analytic QAE/QSEL formulas; it does not construct",
        "the 19-qubit selector-controlled Q oracle used by a real device.",
    ])
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"PASS: toy full-state overlaps, controlled QPE, QSEL")
    print(f"Max bin error : {max_bin_error:.3e}")
    print(f"Max FLAG error: {max_flag_error:.3e}")
    print(f"Report        : {report}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run unified pose-level and ligand-level nested OCC quantum-threshold retrieval "
            "experiments for one normalized receptor or every receptor under an input root."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "input_dir",
        type=Path,
        help=(
            "one receptor directory (contains map/ and occ/) or a parent such "
            "as ../1_data/2_norm/ containing multiple receptor directories"
        ),
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="output root for all receptor/threshold/resolution/OCC experiments",
    )

    parser.add_argument(
        "--max-occ",
        type=int,
        choices=OCC_COUNTS,
        default=max(OCC_COUNTS),
        help=(
            "maximum nested OCC size to evaluate; only OCC entries up to this "
            "count are processed"
        ),
    )

    parser.add_argument(
        "--eval-qubits",
        type=int,
        nargs="+",
        default=list(DEFAULT_BASE_EVAL_QUBITS),
        metavar="M0",
        help=(
            "base QPE-register sizes m0 that define the actual amplitude thresholds as "
            "A0_threshold = sin(pi / 2^m0); e.g. --eval-qubits 10 14 18 22"
        ),
    )
    parser.add_argument(
        "--resolution-multiples",
        type=int,
        nargs="+",
        choices=DEFAULT_RESOLUTION_MULTIPLES,
        default=list(DEFAULT_RESOLUTION_MULTIPLES),
        metavar="K",
        help=(
            "keep each base-m threshold fixed and use actual QPE size "
            "m = m0 + log2(K), so the threshold lies exactly at phase bin K; "
            "supported K values are 1, 2, 4, and 8"
        ),
    )
    parser.add_argument(
        "--grover-max",
        type=parse_grover_max,
        default=None,
        metavar="auto|N",
        help=(
            "scan Grover iterations from 0 through N; 'auto' uses 0..2*r_opt "
            "separately for each OCC size"
        ),
    )
    parser.add_argument(
        "--qsel-shot-budget",
        type=int,
        default=None,
        metavar="N",
        help=(
            "maximum allowed 95%% shots for the selected QSEL point; choose "
            "the smallest Grover iteration meeting this budget. If omitted, "
            "select the r with minimum total Q calls for 95%% retrieval"
        ),
    )
    parser.add_argument(
        "--norm-atol",
        type=float,
        default=DEFAULT_NORM_ATOL,
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="FFT backend for pose overlaps",
    )
    parser.add_argument(
        "--skip-circuit",
        action="store_true",
        help="do not generate circuit.svg files",
    )
    parser.add_argument(
        "--circuit-occ",
        type=int,
        choices=OCC_COUNTS,
        default=None,
        metavar="N",
        help=(
            "generate circuit.svg only at the selected OCC checkpoint N; "
            "if omitted, generate circuits at every OCC checkpoint unless "
            "--skip-circuit is used"
        ),
    )
    parser.add_argument(
        "--test-input",
        action="store_true",
        help="audit original MAP/OCC len files and full-vector normalization; skip experiment",
    )
    parser.add_argument(
        "--test-rot",
        action="store_true",
        help="test the original 32 ROT selector outputs only; skip the experiment",
    )
    parser.add_argument(
        "--test-full-overlap",
        action="store_true",
        help="compare actual full 19-qubit MAP/OCC dot products with FFT A0; skip experiment",
    )
    parser.add_argument(
        "--test-qae-selector",
        action="store_true",
        help="verify toy full-state overlap, controlled QPE, and ideal QSEL; skip experiment",
    )
    parser.add_argument(
        "--validate-small",
        action="store_true",
        help="validate pose FFT and analytic QAE against direct/small-Qiskit results",
    )
    parser.add_argument(
        "--retrieval-mode",
        choices=("both", "pose", "ligand"),
        default="both",
        help=(
            "write pose/configuration results, ligand-only results, or both. "
            "The A0 and finite-QPE calculations are shared."
        ),
    )

    return parser.parse_args()


def run_receptor(
    input_dir: Path,
    output_root: Path,
    args: argparse.Namespace,
    backend: Backend,
    columns: PoseColumns,
) -> tuple[
    list[tuple[str, ExperimentConfig, PoseScaleResult]],
    list[tuple[str, ExperimentConfig, ScaleResult]],
]:
    map_dir = input_dir / "map"
    occ_dir = input_dir / "occ"

    pdbid, Lm, shift = read_map_len(map_dir / "len")
    occ_lengths = read_occ_len(occ_dir / "len")
    occ_entries = collect_occ_entries(occ_dir, pdbid)
    Lo = validate_occ_lengths(occ_entries, occ_lengths)

    occ_entries_used = occ_entries[: args.max_occ]
    checkpoint_counts = tuple(n for n in OCC_COUNTS if n <= args.max_occ)
    checkpoint_set = set(checkpoint_counts)

    base_eval_qubits = _unique_preserve(args.eval_qubits)

    # Each base m0 defines one actual threshold exactly on the first QPE phase bin:
    #
    #     A0_threshold = sin(pi / 2^m0).
    #
    # K = 1,2,4,8 then keeps this threshold fixed while increasing the actual
    # QPE register to m = m0 + log2(K).  The comparator boundary therefore lies
    # exactly at integer phase bin K.
    threshold_specs: list[tuple[int, float, float]] = []
    denominator = Lm * Lo
    for base_m in base_eval_qubits:
        amplitude_threshold = amplitude_grid_resolution(base_m)
        energy_threshold = amplitude_threshold * denominator - shift
        threshold_specs.append(
            (base_m, amplitude_threshold, energy_threshold)
        )

    resolution_multiples = _unique_preserve(args.resolution_multiples)
    configs: list[ExperimentConfig] = []
    for base_m, amplitude_threshold, energy_threshold in threshold_specs:
        for resolution_multiple in resolution_multiples:
            eval_qubits = paired_eval_qubits(
                base_m,
                resolution_multiple,
            )
            comparator = comparator_spec(amplitude_threshold, eval_qubits)
            work_qubits, _ = comparator_resource_counts(eval_qubits, comparator)
            cfg_root = experiment_output_root(
                output_root,
                pdbid,
                amplitude_threshold,
                eval_qubits,
                resolution_multiple,
                args.qsel_shot_budget,
            )
            cfg_root.mkdir(parents=True, exist_ok=True)
            cfg = ExperimentConfig(
                amplitude_threshold=amplitude_threshold,
                energy_threshold=energy_threshold,
                eval_qubits=eval_qubits,
                resolution_multiple=resolution_multiple,
                comparator=comparator,
                work_qubits=work_qubits,
                output_root=cfg_root,
            )
            configs.append(cfg)
            write_occ_mapping(cfg_root / "occ_mapping.tsv", occ_entries_used)

    maps = load_maps(map_dir, pdbid, Lm, shift, args.norm_atol)
    map_spectra = prepare_map_spectra(maps, backend)

    print("=" * 78)
    print(f"PDBID                 {pdbid}")
    print(f"input                  {input_dir}")
    print(f"retrieval mode         {args.retrieval_mode}")
    print("shared coherent search MAP x OCC x ROT x TRANS")
    print(f"DATA qubits            {DATA_QUBITS}")
    print(f"OCCs used              {len(occ_entries_used)}")
    print(f"base eval qubits        {' '.join(str(x[0]) for x in threshold_specs)}")
    print(f"thresholds             {' '.join(format_float_token(x[1]) for x in threshold_specs)}")
    print(f"resolution multiples   {' '.join(str(x) for x in resolution_multiples)}")
    print(f"paired configurations  {len(configs)}")
    for cfg in configs:
        resolution = amplitude_grid_resolution(cfg.eval_qubits)
        print(
            f"  A0thr={format_float_token(cfg.amplitude_threshold):>8s} "
            f"target={cfg.resolution_multiple}x m={cfg.eval_qubits:2d} "
            f"resolution={resolution:.6e} "
            f"actual={cfg.amplitude_threshold/resolution:.6f}x"
        )
    print(f"FFT backend            {backend.name}")
    print()

    denominator = Lm * Lo
    pose_batch_rows: list[tuple[str, ExperimentConfig, PoseScaleResult]] = []
    ligand_batch_rows: list[tuple[str, ExperimentConfig, ScaleResult]] = []

    with ExitStack() as stack:
        # Pose statistics are accumulated in memory. No selected-pose temporary files.

        for occ_position, entry in enumerate(occ_entries_used, start=1):
            print(
                f"[{pdbid}] OCC {occ_position:4d}/{args.max_occ:<4d}  "
                f"selector={entry.selector:4d}  {entry.name}"
            )

            full_occ = read_full_vector(entry.path, args.norm_atol)
            validate_occ_auxiliary(full_occ, entry.path, Lo)
            occ_physical = np.ascontiguousarray(full_occ[:PHYSICAL_DATA_SIZE])

            # The expensive MAP x pose overlap calculation is done ONCE.
            normalized_physical = compute_occ_pose_overlaps(
                map_spectra,
                occ_physical,
                columns,
                backend,
            )

            energies: list[np.ndarray] = []
            a0_values: list[np.ndarray] = []
            min_energy_ligand = math.inf
            min_shifted_ligand = math.inf
            max_a0_ligand = -math.inf

            for map_selector in (0, 1):
                energy = normalized_physical[map_selector] * denominator
                shifted = energy + shift
                min_shifted = float(np.min(shifted))
                if min_shifted < -TOLERANCE:
                    min_energy_here = float(np.min(energy))
                    raise ArithmeticError(
                        "stored MAP shift does not keep every pose nonnegative: "
                        f"OCC={entry.name}, MAP={pdbid}_{map_selector}, "
                        f"min ENERGY={min_energy_here:.12e}, stored shift={shift:.12e}, "
                        f"minimum required shift is >={-min_energy_here:.12e}"
                    )
                a0 = np.clip(shifted / denominator, 0.0, 1.0)
                energies.append(energy)
                a0_values.append(a0)
                min_energy_ligand = min(min_energy_ligand, float(np.min(energy)))
                min_shifted_ligand = min(min_shifted_ligand, min_shifted)
                max_a0_ligand = max(max_a0_ligand, float(np.max(a0)))

            map_min_a0 = [float(np.min(a0_values[m])) for m in (0, 1)]
            best_map = 0 if map_min_a0[0] <= map_min_a0[1] else 1
            best_pose = int(np.argmin(a0_values[best_map]))
            min_a0_ligand = float(a0_values[best_map][best_pose])
            reference_min_energy = float(energies[best_map][best_pose])

            # Exact membership is threshold-dependent but m-independent.
            exact_cache: dict[float, tuple[list[np.ndarray], int, bool]] = {}
            for _, amplitude_threshold, _ in threshold_specs:
                masks = [
                    a0_values[m] < amplitude_threshold
                    for m in (0, 1)
                ]
                exact_count = int(np.count_nonzero(masks[0])) + int(np.count_nonzero(masks[1]))
                exact_cache[amplitude_threshold] = (masks, exact_count, exact_count > 0)

            for config_index, cfg in enumerate(configs):
                masks, exact_count, good_ligand = exact_cache[cfg.amplitude_threshold]

                # Finite-QPE probabilities are also shared by both retrieval objectives.
                flag0 = qae_flag_probability(a0_values[0], cfg.comparator)
                flag1 = qae_flag_probability(a0_values[1], cfg.comparator)
                ligand_flag_sum = float(
                    np.sum(flag0, dtype=np.float64) + np.sum(flag1, dtype=np.float64)
                )

                if args.retrieval_mode in ("both", "ligand"):
                    cfg.stats.states += 2 * POSES_PER_PAIR
                    cfg.stats.exact_good_configs += exact_count
                    cfg.stats.sum_flag += ligand_flag_sum
                    if good_ligand:
                        cfg.stats.good_ligands += 1
                        cfg.stats.sum_flag_good_ligand += ligand_flag_sum
                    cfg.stats.min_energy = min(cfg.stats.min_energy, min_energy_ligand)
                    cfg.stats.min_shifted = min(cfg.stats.min_shifted, min_shifted_ligand)
                    cfg.stats.max_a0 = max(cfg.stats.max_a0, max_a0_ligand)

                    cfg.ligand_records.append(
                        LigandRecord(
                            occ_selector=entry.selector,
                            occ_name=entry.name,
                            source=entry.source,
                            exact_good_configs=exact_count,
                            min_energy=reference_min_energy,
                            min_a0=min_a0_ligand,
                            best_map_selector=best_map,
                            best_pose_selector=best_pose,
                            best_rotation_selector=int(columns.rotation_selector[best_pose]),
                            best_translation_selector=int(columns.translation_selector[best_pose]),
                            best_shift_z=int(columns.shift_z[best_pose]),
                            best_shift_y=int(columns.shift_y[best_pose]),
                            best_shift_x=int(columns.shift_x[best_pose]),
                            sum_flag=ligand_flag_sum,
                            p_flag_given_occ=ligand_flag_sum / float(2 * POSES_PER_PAIR),
                        )
                    )

                if args.retrieval_mode in ("both", "pose"):
                    ps = cfg.pose_stats
                    ps.states += 2 * POSES_PER_PAIR
                    ps.exact_good += exact_count
                    ps.sum_flag += ligand_flag_sum
                    ps.sum_flag_true_good += float(
                        np.sum(flag0[masks[0]], dtype=np.float64)
                        + np.sum(flag1[masks[1]], dtype=np.float64)
                    )
                    ps.min_energy = min(ps.min_energy, min_energy_ligand)
                    ps.min_shifted = min(ps.min_shifted, min_shifted_ligand)
                    ps.max_a0 = max(ps.max_a0, max_a0_ligand)

            if occ_position in checkpoint_set:
                for cfg in configs:
                    if args.retrieval_mode in ("both", "ligand"):
                        lig_result, lig_rows = calculate_ligand_scale_result(
                            occ_count=occ_position,
                            stats=cfg.stats,
                            eval_qubits=cfg.eval_qubits,
                            grover_max_requested=args.grover_max,
                            qsel_shot_budget=args.qsel_shot_budget,
                            work_qubits=cfg.work_qubits,
                        )
                        cfg.results.append(lig_result)
                        cfg.grover_rows.extend(lig_rows)
                        ligand_batch_rows.append((pdbid, cfg, lig_result))

                        lig_dir = checkpoint_folder(cfg.output_root / "ligand", occ_position)
                        lig_dir.mkdir(parents=True, exist_ok=True)
                        write_ligand_experiment_summary(
                            lig_dir / "summary.txt",
                            pdbid=pdbid,
                            result=lig_result,
                            amplitude_threshold=cfg.amplitude_threshold,
                            energy_threshold=cfg.energy_threshold,
                            Lm=Lm,
                            Lo=Lo,
                            shift=shift,
                            eval_qubits=cfg.eval_qubits,
                            result_resolution_multiple=cfg.resolution_multiple,
                            comparator=cfg.comparator,
                            backend=backend,
                        )
                        write_selected_ligands(
                            lig_dir / "selected_ligands.tsv",
                            records=cfg.ligand_records,
                            occ_count=occ_position,
                            result=lig_result,
                        )
                        if not args.skip_circuit and (args.circuit_occ is None or occ_position == args.circuit_occ):
                            circuit, circuit_qubits, circuit_work = build_logical_circuit_ligand(
                                receptor_pdbid=pdbid,
                                occ_count=occ_position,
                                eval_qubits=cfg.eval_qubits,
                                comparator=cfg.comparator,
                                grover_iterations=lig_result.grover_iterations,
                            )
                            if circuit_work != lig_result.work_qubits:
                                raise AssertionError("ligand circuit work-qubit mismatch")
                            if circuit_qubits != lig_result.logical_qubits:
                                raise AssertionError("ligand circuit logical-qubit mismatch")
                            write_circuit_svg(circuit, lig_dir / "circuit.svg")

                    if args.retrieval_mode in ("both", "pose"):
                        pose_result, pose_rows = calculate_pose_scale_result(
                            occ_count=occ_position,
                            stats=cfg.pose_stats,
                            eval_qubits=cfg.eval_qubits,
                            grover_max_requested=args.grover_max,
                            qsel_shot_budget=args.qsel_shot_budget,
                            work_qubits=cfg.work_qubits,
                        )
                        cfg.pose_results.append(pose_result)
                        cfg.pose_grover_rows.extend(pose_rows)
                        pose_batch_rows.append((pdbid, cfg, pose_result))

                        pose_dir = checkpoint_folder(cfg.output_root / "pose", occ_position)
                        pose_dir.mkdir(parents=True, exist_ok=True)
                        write_pose_experiment_summary(
                            pose_dir / "summary.txt",
                            pdbid=pdbid,
                            cfg=cfg,
                            result=pose_result,
                            Lm=Lm,
                            Lo=Lo,
                            shift=shift,
                            backend=backend,
                        )
                        if not args.skip_circuit and (args.circuit_occ is None or occ_position == args.circuit_occ):
                            circuit, circuit_qubits, circuit_work = build_logical_circuit_pose(
                                receptor_pdbid=pdbid,
                                occ_count=occ_position,
                                eval_qubits=cfg.eval_qubits,
                                comparator=cfg.comparator,
                                grover_iterations=pose_result.grover_iterations,
                            )
                            if circuit_work != pose_result.work_qubits:
                                raise AssertionError("pose circuit work-qubit mismatch")
                            if circuit_qubits != pose_result.logical_qubits:
                                raise AssertionError("pose circuit logical-qubit mismatch")
                            write_circuit_svg(circuit, pose_dir / "circuit.svg")

                print(
                    f"  -> checkpoint N={occ_position:4d}: "
                    f"completed {len(configs)} paired threshold/resolution configurations"
                )

    for cfg in configs:
        if args.retrieval_mode in ("both", "ligand"):
            ligand_root = cfg.output_root / "ligand"
            write_ligand_scaling_summary(
                ligand_root / "scaling_summary.tsv",
                cfg.results,
                amplitude_threshold=cfg.amplitude_threshold,
                energy_threshold=cfg.energy_threshold,
                eval_qubits=cfg.eval_qubits,
                resolution_multiple=cfg.resolution_multiple,
            )
            write_ligand_grover_shots(
                ligand_root / "grover_shots.tsv",
                cfg.grover_rows,
            )

        if args.retrieval_mode in ("both", "pose"):
            pose_root = cfg.output_root / "pose"
            write_pose_scaling_summary(
                pose_root / "scaling_summary.tsv",
                cfg.pose_results,
                cfg=cfg,
            )
            write_pose_grover_shots(
                pose_root / "grover_shots.tsv",
                cfg.pose_grover_rows,
            )

    print(f"[{pdbid}] completed {len(configs)} paired configurations")
    return pose_batch_rows, ligand_batch_rows


def main() -> None:
    args = parse_args()
    if args.test_input:
        if args.norm_atol < 0.0:
            raise ValueError("--norm-atol must be >= 0")
        test_input_output(args.input_dir, args.output_dir, args.max_occ, args.norm_atol)
        return
    if args.test_rot:
        test_rotation_output(args.output_dir, args.device)
        return
    if args.test_qae_selector:
        test_qae_selector_output(args.output_dir)
        return
    if args.test_full_overlap:
        if args.norm_atol < 0.0:
            raise ValueError("--norm-atol must be >= 0")
        test_full_overlap_output(
            args.input_dir, args.output_dir, args.max_occ, args.norm_atol, args.device
        )
        return

    if args.norm_atol < 0.0:
        raise ValueError("--norm-atol must be >= 0")
    if args.qsel_shot_budget is not None and args.qsel_shot_budget < 1:
        raise ValueError("--qsel-shot-budget must be >= 1")

    args.resolution_multiples = _unique_preserve(args.resolution_multiples)
    for value in args.resolution_multiples:
        if value not in DEFAULT_RESOLUTION_MULTIPLES:
            raise ValueError(
                f"every --resolution-multiples value must be one of "
                f"{DEFAULT_RESOLUTION_MULTIPLES}"
            )

    args.eval_qubits = _unique_preserve(args.eval_qubits)
    for value in args.eval_qubits:
        if not 1 <= value <= 30:
            raise ValueError(
                "every --eval-qubits base value must be between 1 and 30"
            )
        max_eval = value + int(math.log2(max(args.resolution_multiples)))
        if max_eval > 30:
            raise ValueError(
                f"base m={value} with the requested resolution multiples "
                f"requires m={max_eval} > 30"
            )

    receptor_dirs = discover_receptor_dirs(args.input_dir)
    output_root = args.output_dir.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    backend = select_backend(args.device)
    columns = build_pose_columns()

    if args.validate_small:
        pose_error = validate_pose_fft()
        qae_error = validate_small_qiskit_qae()
        print(f"pose FFT validation max error  {pose_error:.6e}")
        print(f"small QAE validation max error {qae_error:.6e}")

    print(f"retrieval mode          {args.retrieval_mode}")
    print("QSEL objective          minimum total Q calls for 95% retrieval")
    print(f"receptors found         {len(receptor_dirs)}")
    for receptor_dir in receptor_dirs:
        print(f"  {receptor_dir.name}")
    print(f"max OCC                 {args.max_occ}")
    print(
        f"QSEL shots95 budget     "
        f"{'none' if args.qsel_shot_budget is None else args.qsel_shot_budget}"
    )
    print(f"output root             {output_root}")
    print()

    pose_batch_rows: list[tuple[str, ExperimentConfig, PoseScaleResult]] = []
    ligand_batch_rows: list[tuple[str, ExperimentConfig, ScaleResult]] = []

    for receptor_dir in receptor_dirs:
        pose_rows, ligand_rows = run_receptor(
            receptor_dir,
            output_root,
            args,
            backend,
            columns,
        )
        pose_batch_rows.extend(pose_rows)
        ligand_batch_rows.extend(ligand_rows)

    if args.retrieval_mode in ("both", "pose"):
        pose_summary = output_root / "batch_summary_pose.tsv"
        write_pose_batch_summary(pose_summary, pose_batch_rows)
        print(f"pose batch summary      {pose_summary}")

    if args.retrieval_mode in ("both", "ligand"):
        ligand_summary = output_root / "batch_summary_ligand.tsv"
        write_ligand_batch_summary(ligand_summary, ligand_batch_rows)
        print(f"ligand batch summary    {ligand_summary}")

    print(f"output root             {output_root}")


if __name__ == "__main__":
    main()
