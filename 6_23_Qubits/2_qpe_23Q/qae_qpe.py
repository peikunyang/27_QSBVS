#!/usr/bin/env python3

import argparse
import math
from pathlib import Path

import numpy as np


PROGRAM_VERSION = "20261025-dual-phase-qcomp"
from qiskit import AncillaRegister, QuantumCircuit, QuantumRegister, transpile
from qiskit.circuit.library import StatePreparation, UnitaryGate, grover_operator, phase_estimation
from qiskit.quantum_info import Statevector
from qiskit.synthesis import synth_integer_comparator_2s


EVAL_QUBITS = 4
ROT_QUBITS = 3
TRA_QUBITS = 3
SELECTOR_STATES = 2 * 2 * (1 << ROT_QUBITS) * (1 << TRA_QUBITS)
STATE_BOX_LABELS = {"O1", "O2", "MAP1", "MAP2"}
IBM_BASIS_GATES = ["rz", "sx", "x", "cx"]
ALGORITHM_GATES = {
    "id", "h", "x", "y", "z",
    "s", "sdg", "t", "tdg",
    "rx", "ry", "rz", "p", "u",
    "cx", "cy", "cz", "ch",
    "crx", "cry", "crz", "cp",
    "swap", "cswap", "ccx",
    "barrier", "measure", "reset",
}
TOL = 1e-10
SIM_DEVICE = "cpu"
SIM_PRECISION = "double"
_AER_GPU_BACKEND = None



def shots_for_at_least_one_success(probability, confidence):
    probability = float(probability)
    confidence = float(confidence)
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must satisfy 0 < confidence < 1")
    if probability <= 0.0:
        return None
    if probability >= 1.0:
        return 1
    return int(
        math.ceil(
            math.log1p(-confidence)
            / math.log1p(-probability)
        )
    )


def register_size(circuit, name):
    for register in circuit.qregs:
        if register.name == name:
            return len(register)
    return 0


def rotation_signatures(system):
    seed = np.arange(system["dimension"], dtype=np.float64)
    signatures = []
    for selector in range(1 << ROT_QUBITS):
        transformed = pose_vector(seed, system, selector, 0)
        signatures.append(
            tuple(
                int(value)
                for value in transformed[:system["physical_dim"]]
            )
        )
    return signatures


def translation_signatures(system):
    seed = np.arange(system["dimension"], dtype=np.float64)
    signatures = []
    for selector in range(1 << TRA_QUBITS):
        transformed = pose_vector(seed, system, 0, selector)
        signatures.append(
            tuple(
                int(value)
                for value in transformed[:system["physical_dim"]]
            )
        )
    return signatures


def true_good_metrics(flag_probability, selector_results, good_mask):
    true_good_probability = 0.0
    for m in range(2):
        for o in range(2):
            for r in range(1 << ROT_QUBITS):
                for t in range(1 << TRA_QUBITS):
                    if good_mask[m, o, r, t]:
                        true_good_probability += float(
                            selector_results[(m, o, r, t)]["joint"]
                        )

    true_good_given_flag = (
        true_good_probability / float(flag_probability)
        if flag_probability > 0.0
        else 0.0
    )

    return {
        "flag_probability": float(flag_probability),
        "true_good_probability": float(true_good_probability),
        "true_good_given_flag": float(true_good_given_flag),
        "shots_95_true_good": shots_for_at_least_one_success(
            true_good_probability, 0.95
        ),
        "shots_99_true_good": shots_for_at_least_one_success(
            true_good_probability, 0.99
        ),
    }


def paper_summary_text(
    *,
    system,
    threshold,
    low,
    qsel_iterations,
    classical_a0,
    qae_circuit,
    full_circuit,
    quantum_p_before,
    quantum_before,
    quantum_p_after,
    quantum_after,
    validation_lines,
    tolerance,
):
    if qae_circuit.num_qubits != full_circuit.num_qubits:
        raise AssertionError(
            "QAE and QSEL circuits use different total qubit counts"
        )

    good_mask = np.asarray(classical_a0 < threshold, dtype=bool)
    exact_good_states = int(np.count_nonzero(good_mask))

    comp = true_good_metrics(
        quantum_p_before,
        quantum_before,
        good_mask,
    )
    qsel = true_good_metrics(
        quantum_p_after,
        quantum_after,
        good_mask,
    )

    if not math.isclose(
        comp["true_good_given_flag"],
        qsel["true_good_given_flag"],
        rel_tol=0.0,
        abs_tol=tolerance,
    ):
        raise AssertionError(
            "COMP and QSEL true_good_given_flag differ beyond tolerance: "
            f"{comp['true_good_given_flag']:.16e} vs "
            f"{qsel['true_good_given_flag']:.16e}; "
            f"tolerance={tolerance:.3e}"
        )

    rotation_states = 1 << ROT_QUBITS
    translation_states = 1 << TRA_QUBITS
    unique_rotations = len(set(rotation_signatures(system)))
    unique_translations = len(set(translation_signatures(system)))

    physical_data_qubits = system["data_qubits"] - 1
    aux_qubits = 1

    work_qubits = register_size(qae_circuit, "work")
    flag_qubits = register_size(qae_circuit, "flag")
    eval_qubits = register_size(qae_circuit, "eval")
    map_qubits = register_size(qae_circuit, "map_sel")
    occ_qubits = register_size(qae_circuit, "occ_sel")
    rot_qubits = register_size(qae_circuit, "rot_sel")
    trans_qubits = register_size(qae_circuit, "tra_sel")

    component_total = (
        work_qubits
        + flag_qubits
        + eval_qubits
        + map_qubits
        + trans_qubits
        + rot_qubits
        + occ_qubits
        + aux_qubits
        + physical_data_qubits
    )
    if component_total != qae_circuit.num_qubits:
        raise AssertionError(
            "paper-summary qubit accounting mismatch: "
            f"components={component_total}, "
            f"circuit={qae_circuit.num_qubits}"
        )

    total_combinations = (
        2 * 2 * rotation_states * translation_states
    )
    if total_combinations != SELECTOR_STATES:
        raise AssertionError(
            "selector-space accounting mismatch: "
            f"computed={total_combinations}, "
            f"constant={SELECTOR_STATES}"
        )

    validation_pass = (
        bool(validation_lines)
        and validation_lines[-1].strip().endswith("PASS")
    )

    def fmt_count(value):
        return "INF" if value is None else str(value)

    lines = [
        f"PROGRAM_VERSION = {PROGRAM_VERSION}",
        "",
        "SYSTEM",
        f"grid_size = {system['grid_size']}",
        f"input_dimension = {system['dimension']}",
        f"qsel_iterations = {qsel_iterations}",
        "",
        "QUBITS",
        f"work = {work_qubits}",
        f"flag = {flag_qubits}",
        f"eval = {eval_qubits}",
        f"map_sel = {map_qubits}",
        f"trans = {trans_qubits}",
        f"rot = {rot_qubits}",
        f"occ_sel = {occ_qubits}",
        f"aux = {aux_qubits}",
        f"data = {physical_data_qubits}",
        f"total = {qae_circuit.num_qubits}",
        "",
        "SEARCH_SPACE",
        "MAP = 2",
        "OCC = 2",
        f"rotation_selector_qubits = {ROT_QUBITS}",
        f"rotation_selector_states = {rotation_states}",
        f"unique_rotations = {unique_rotations}",
        f"translation_selector_qubits = {TRA_QUBITS}",
        f"translation_selector_states = {translation_states}",
        f"unique_translations = {unique_translations}",
        f"total_combinations = {total_combinations}",
        "",
        "SEARCH_TARGET",
        f"a0_threshold = {threshold:.12e}",
        f"qpe_compare = {qcomp_label(low)}",
        f"exact_good_states = {exact_good_states}",
        "",
        "COMP_VS_QSEL",
    ]

    first_width = 13
    pair_width = 22
    single_width = 22
    metric_names = (
        "flag_probability",
        "true_good_probability",
        "shots_95_true_good",
        "shots_99_true_good",
    )

    header1 = f"{'':<{first_width}}"
    for name in metric_names:
        header1 += f"{name:^{pair_width}}"
    header1 += f"{'true_good_given_flag':^{single_width}}"
    lines.append(header1)

    header2 = f"{'SYSTEM':<{first_width}}"
    for _ in metric_names:
        header2 += f"{'COMP':>11}{'QSEL':>11}"
    header2 += f"{'VALUE':>{single_width}}"
    lines.append(header2)

    row = f"{'small_rot_tra':<{first_width}}"
    row += f"{comp['flag_probability']:11.3e}{qsel['flag_probability']:11.3e}"
    row += (
        f"{comp['true_good_probability']:11.3e}"
        f"{qsel['true_good_probability']:11.3e}"
    )
    row += (
        f"{fmt_count(comp['shots_95_true_good']):>11}"
        f"{fmt_count(qsel['shots_95_true_good']):>11}"
    )
    row += (
        f"{fmt_count(comp['shots_99_true_good']):>11}"
        f"{fmt_count(qsel['shots_99_true_good']):>11}"
    )
    row += f"{comp['true_good_given_flag']:{single_width}.3e}"
    lines.append(row)

    lines.extend([
        "",
        "VALIDATION",
        f"classical_vs_quantum = {'PASS' if validation_pass else 'FAIL'}",
    ])

    return "\n".join(lines).rstrip() + "\n"

def configure_sim_device(device, precision="double"):
    global SIM_DEVICE, SIM_PRECISION, _AER_GPU_BACKEND
    SIM_DEVICE = str(device).strip().lower()
    SIM_PRECISION = str(precision).strip().lower()
    if SIM_DEVICE not in {"cpu", "gpu"}:
        raise ValueError("--device must be cpu or gpu")
    if SIM_PRECISION not in {"single", "double"}:
        raise ValueError("--precision must be single or double")
    if SIM_DEVICE == "cpu":
        _AER_GPU_BACKEND = None
        return
    try:
        from qiskit_aer import AerSimulator
    except ImportError as exc:
        raise ImportError("--device gpu requires Qiskit Aer GPU support") from exc
    backend = AerSimulator(
        method="statevector",
        device="GPU",
        precision=SIM_PRECISION,
    )
    devices = tuple(str(x).upper() for x in backend.available_devices())
    if "GPU" not in devices:
        raise RuntimeError(f"Qiskit Aer GPU device unavailable: {devices}")
    _AER_GPU_BACKEND = backend


def simulate_statevector(circuit):
    if SIM_DEVICE == "cpu":
        return Statevector.from_instruction(circuit)
    if _AER_GPU_BACKEND is None:
        raise RuntimeError("GPU simulator has not been configured")
    run_circuit = circuit.copy()
    run_circuit.save_statevector(label="statevector")
    compiled = transpile(run_circuit, _AER_GPU_BACKEND, optimization_level=0)
    result = _AER_GPU_BACKEND.run(compiled, shots=1).result()
    if not result.success:
        raise RuntimeError(f"Aer GPU simulation failed: {result.status}")
    return Statevector(np.asarray(result.data(0)["statevector"], dtype=np.complex128))


def grid_parameters(grid_size):
    if grid_size not in (2, 4, 8):
        raise ValueError("--grid-size must be 2, 4, or 8")
    coord_qubits = int(round(math.log2(grid_size)))
    physical_dim = grid_size ** 3
    data_qubits = 3 * coord_qubits + 1
    dimension = 2 * physical_dim
    return coord_qubits, data_qubits, physical_dim, dimension


def parse_input(path, dimension):
    names = {"O1", "O2", "M1", "M2"}
    values = {}
    current = None
    buffer = []

    def commit():
        nonlocal current, buffer
        if current is None:
            return
        if len(buffer) != dimension:
            raise ValueError(f"{current} needs {dimension} values; got {len(buffer)}")
        values[current] = np.asarray(buffer, dtype=np.float64)
        current = None
        buffer = []

    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if parts[0] in names:
                commit()
                current = parts[0]
                buffer = [float(x) for x in parts[1:]]
            else:
                if current is None:
                    raise ValueError("data appears before O1/O2/M1/M2 label")
                buffer.extend(float(x) for x in parts)
    commit()

    for name in ("O1", "O2", "M1", "M2"):
        if name not in values:
            raise ValueError(f"input missing {name}")
    return values


def load_system(path, grid_size):
    coord_qubits, data_qubits, physical_dim, dimension = grid_parameters(grid_size)
    vectors = parse_input(path, dimension)
    raw_occs = np.vstack([vectors["O1"], vectors["O2"]])
    raw_maps = np.vstack([vectors["M1"], vectors["M2"]])
    occ_norms = np.linalg.norm(raw_occs, axis=1)
    map_norms = np.linalg.norm(raw_maps, axis=1)
    if not math.isclose(float(occ_norms[0]), float(occ_norms[1]), abs_tol=5e-10, rel_tol=0.0):
        raise ValueError(f"O1/O2 lengths differ: {occ_norms}")
    if not math.isclose(float(map_norms[0]), float(map_norms[1]), abs_tol=5e-10, rel_tol=0.0):
        raise ValueError(f"M1/M2 lengths differ: {map_norms}")
    Lo = float(np.mean(occ_norms))
    Lm = float(np.mean(map_norms))
    return {
        "grid_size": grid_size,
        "coord_qubits": coord_qubits,
        "data_qubits": data_qubits,
        "physical_dim": physical_dim,
        "dimension": dimension,
        "aux_index": data_qubits - 1,
        "raw_occs": raw_occs,
        "raw_maps": raw_maps,
        "occs": raw_occs / Lo,
        "maps": raw_maps / Lm,
        "Lo": Lo,
        "Lm": Lm,
    }


def choose_threshold(amplitudes):
    flat = np.asarray(amplitudes, dtype=float).reshape(-1)
    fallback = None
    for tenth in range(1, 11):
        threshold = tenth / 10.0
        count = int(np.sum(flat < threshold))
        if 2 <= count <= 16:
            return threshold
        if fallback is None and count >= 2:
            fallback = threshold
    return 1.0 if fallback is None else fallback


def threshold_low_limit(threshold):
    modulus = 1 << EVAL_QUBITS
    cutoff = modulus * math.asin(min(threshold, 1.0)) / math.pi
    return int(math.ceil(cutoff))


def threshold_high_limit(low):
    return (1 << EVAL_QUBITS) - low + 1


def qcomp_label(low):
    high = threshold_high_limit(low)
    return f"y<{low} OR y>={high}"


def coordinate_data_indices(system):
    k = system["coord_qubits"]
    return (
        list(range(0, k)),
        list(range(k, 2 * k)),
        list(range(2 * k, 3 * k)),
    )


def rotation_coordinate_indices(system, axis):
    x_bits, y_bits, z_bits = coordinate_data_indices(system)
    if axis == "x":
        return y_bits, z_bits
    if axis == "y":
        return x_bits, z_bits
    if axis == "z":
        return x_bits, y_bits
    raise ValueError(axis)


def translation_coordinate_indices(system, axis):
    x_bits, y_bits, z_bits = coordinate_data_indices(system)
    if axis == "x":
        return x_bits
    if axis == "y":
        return y_bits
    if axis == "z":
        return z_bits
    raise ValueError(axis)


def index_xyz(x, y, z, n):
    return int(x + n * y + n * n * z)


def xyz_from_index(index, n):
    return int(index % n), int((index // n) % n), int(index // (n * n))


def rotate_axis(vector, system, axis):
    n = system["grid_size"]
    physical_dim = system["physical_dim"]
    out = np.asarray(vector, dtype=np.float64).copy()
    rotated = np.zeros(physical_dim, dtype=np.float64)
    edge = n - 1
    for old_index, amplitude in enumerate(out[:physical_dim].copy()):
        x, y, z = xyz_from_index(old_index, n)
        if axis == "x":
            xn, yn, zn = x, z, edge - y
        elif axis == "y":
            xn, yn, zn = z, y, edge - x
        elif axis == "z":
            xn, yn, zn = y, edge - x, z
        else:
            raise ValueError(axis)
        rotated[index_xyz(xn, yn, zn, n)] = amplitude
    out[:physical_dim] = rotated
    return out


def translate_axis(vector, system, axis):
    n = system["grid_size"]
    physical_dim = system["physical_dim"]
    out = np.asarray(vector, dtype=np.float64).copy()
    translated = np.zeros(physical_dim, dtype=np.float64)
    for old_index, amplitude in enumerate(out[:physical_dim].copy()):
        x, y, z = xyz_from_index(old_index, n)
        if axis == "x":
            x = (x + 1) % n
        elif axis == "y":
            y = (y + 1) % n
        elif axis == "z":
            z = (z + 1) % n
        else:
            raise ValueError(axis)
        translated[index_xyz(x, y, z, n)] = amplitude
    out[:physical_dim] = translated
    return out


def pose_vector(vector, system, rot_selector, tra_selector):
    out = np.asarray(vector, dtype=np.float64).copy()
    if rot_selector & 0b001:
        out = rotate_axis(out, system, "x")
    if rot_selector & 0b010:
        out = rotate_axis(out, system, "y")
    if rot_selector & 0b100:
        out = rotate_axis(out, system, "z")
    if tra_selector & 0b001:
        out = translate_axis(out, system, "x")
    if tra_selector & 0b010:
        out = translate_axis(out, system, "y")
    if tra_selector & 0b100:
        out = translate_axis(out, system, "z")
    return out


def named_state_gate(vector, label, inverse=False):
    n = int(round(math.log2(len(vector))))
    circuit = QuantumCircuit(n, name=label)
    gate = StatePreparation(vector)
    if inverse:
        gate = gate.inverse()
    circuit.append(gate, range(n))
    return circuit.to_gate(label=label)


def controlled_state(vector, ctrl_state, label):
    return named_state_gate(vector, label, inverse=False).control(
        1, label=label, ctrl_state=ctrl_state, annotated=False
    )


def controlled_inverse_state(vector, ctrl_state, label):
    return named_state_gate(vector, label, inverse=True).control(
        1, label=label, ctrl_state=ctrl_state, annotated=False
    )


def build_rotation_gate(system, axis):
    k = system["coord_qubits"]
    circuit = QuantumCircuit(2 + 2 * k, name=f"R{axis.upper()}")
    selector = 0
    aux = 1
    reg_a = list(range(2, 2 + k))
    reg_b = list(range(2 + k, 2 + 2 * k))
    swap_base = QuantumCircuit(2)
    swap_base.swap(0, 1)
    controlled_swap = swap_base.to_gate().control(
        2, ctrl_state=1, annotated=False
    )
    x_base = QuantumCircuit(1)
    x_base.x(0)
    controlled_x = x_base.to_gate().control(
        2, ctrl_state=1, annotated=False
    )
    for bit in range(k):
        circuit.append(
            controlled_swap,
            [selector, aux, reg_a[bit], reg_b[bit]],
        )
    for bit in range(k):
        circuit.append(
            controlled_x,
            [selector, aux, reg_b[bit]],
        )
    return circuit.to_gate(label=f"R{axis.upper()}")


def build_translation_gate(system, axis):
    n = system["grid_size"]
    matrix = np.zeros((n, n), dtype=np.float64)
    for old in range(n):
        matrix[(old + 1) % n, old] = 1.0
    base = UnitaryGate(matrix, label=f"T{axis.upper()}")
    return base.control(
        2, label=f"T{axis.upper()}", ctrl_state=1, annotated=False
    )


def branch_layout(system):
    n = system["data_qubits"]
    occ_sel = n
    map_sel = n + 1
    rot = list(range(n + 2, n + 2 + ROT_QUBITS))
    tra = list(range(n + 2 + ROT_QUBITS, n + 2 + ROT_QUBITS + TRA_QUBITS))
    return list(range(n)), occ_sel, map_sel, rot, tra


def build_branch_preparation(system):
    n = system["data_qubits"]
    branch_qubits = n + 2 + ROT_QUBITS + TRA_QUBITS
    circuit = QuantumCircuit(branch_qubits, name="B_POSE")
    data, occ_sel, map_sel, rot, tra = branch_layout(system)
    aux = data[system["aux_index"]]

    circuit.append(controlled_state(system["occs"][0], 0, "O1"), [occ_sel] + data)
    circuit.append(controlled_state(system["occs"][1], 1, "O2"), [occ_sel] + data)

    for axis_index, axis in enumerate(("x", "y", "z")):
        reg_a, reg_b = rotation_coordinate_indices(system, axis)
        circuit.append(
            build_rotation_gate(system, axis),
            [rot[axis_index], aux]
            + [data[i] for i in reg_a]
            + [data[i] for i in reg_b],
        )

    for axis_index, axis in enumerate(("x", "y", "z")):
        coord = translation_coordinate_indices(system, axis)
        circuit.append(
            build_translation_gate(system, axis),
            [tra[axis_index], aux] + [data[i] for i in coord],
        )

    circuit.append(controlled_inverse_state(system["maps"][0], 0, "MAP1"), [map_sel] + data)
    circuit.append(controlled_inverse_state(system["maps"][1], 1, "MAP2"), [map_sel] + data)
    return circuit



def build_pose_checkpoint(system):
    """
    Prepare the normalized OCC state and apply the complete ROT/TRA
    transformation in a uniform OCC/ROT/TRA selector superposition.

    MAP is intentionally omitted: this checkpoint validates the pose
    transformation itself before receptor-map application.
    """
    n = system["data_qubits"]

    data = QuantumRegister(n, "data")
    occ = QuantumRegister(1, "occ_sel")
    rot = QuantumRegister(ROT_QUBITS, "rot_sel")
    tra = QuantumRegister(TRA_QUBITS, "tra_sel")

    circuit = QuantumCircuit(
        data,
        occ,
        rot,
        tra,
        name="POSE_CHECKPOINT",
    )

    circuit.h(occ)
    circuit.h(rot)
    circuit.h(tra)

    # Use exactly the same controlled OCC preparation as the production branch.
    circuit.append(
        controlled_state(system["occs"][0], 0, "O1"),
        [occ[0]] + list(data),
    )
    circuit.append(
        controlled_state(system["occs"][1], 1, "O2"),
        [occ[0]] + list(data),
    )

    aux = data[system["aux_index"]]

    # Use exactly the same controlled rotation gates and ordering.
    for axis_index, axis in enumerate(("x", "y", "z")):
        reg_a, reg_b = rotation_coordinate_indices(system, axis)
        circuit.append(
            build_rotation_gate(system, axis),
            [rot[axis_index], aux]
            + [data[i] for i in reg_a]
            + [data[i] for i in reg_b],
        )

    # Use exactly the same controlled translation gates and ordering.
    for axis_index, axis in enumerate(("x", "y", "z")):
        coord = translation_coordinate_indices(system, axis)
        circuit.append(
            build_translation_gate(system, axis),
            [tra[axis_index], aux]
            + [data[i] for i in coord],
        )

    return circuit


def pose_selector_index(o, r, t):
    """
    Integer selector index for qargs ordered as:
        occ, rot[0], rot[1], rot[2], tra[0], tra[1], tra[2]
    """
    return int(o + 2 * r + 16 * t)


def pose_selector_qargs(circuit):
    occ = qreg(circuit, "occ_sel")
    rot = qreg(circuit, "rot_sel")
    tra = qreg(circuit, "tra_sel")

    return [
        circuit.find_bit(occ[0]).index,
        *[circuit.find_bit(q).index for q in rot],
        *[circuit.find_bit(q).index for q in tra],
    ]


def classical_pose_states(system):
    """
    Classical normalized OCC state after ROT/TRA for every
    OCC x ROT x TRA selector branch.
    """
    dimension = system["dimension"]
    out = np.zeros(
        (2, 1 << ROT_QUBITS, 1 << TRA_QUBITS, dimension),
        dtype=np.float64,
    )

    for o in range(2):
        for r in range(1 << ROT_QUBITS):
            for t in range(1 << TRA_QUBITS):
                out[o, r, t, :] = pose_vector(
                    system["occs"][o],
                    system,
                    r,
                    t,
                )

    return out


def quantum_pose_states(circuit, system):
    """
    Extract the complete data-register state from each OCC/ROT/TRA
    selector branch of the Qiskit pose checkpoint.

    Since the selector registers are prepared uniformly and remain in
    computational-basis labels during the pose operation, division by the
    positive selector amplitude recovers the full branch data state,
    including any branch-dependent phase.
    """
    state = simulate_statevector(circuit)

    selector_prob = np.asarray(
        state.probabilities(
            qargs=pose_selector_qargs(circuit)
        ),
        dtype=float,
    )

    n = system["data_qubits"]
    dimension = 1 << n

    if dimension != system["dimension"]:
        raise AssertionError(
            f"pose-state dimension mismatch: "
            f"2^{n}={dimension}, "
            f"system dimension={system['dimension']}"
        )

    out = np.zeros(
        (2, 1 << ROT_QUBITS, 1 << TRA_QUBITS, dimension),
        dtype=np.complex128,
    )

    for o in range(2):
        for r in range(1 << ROT_QUBITS):
            for t in range(1 << TRA_QUBITS):
                s = pose_selector_index(o, r, t)
                selector_amplitude = math.sqrt(
                    float(selector_prob[s])
                )

                if selector_amplitude <= TOL:
                    raise AssertionError(
                        "zero pose-selector amplitude for "
                        f"OCC={o}, R={r}, T={t}"
                    )

                base = s << n
                out[o, r, t, :] = (
                    state.data[base : base + dimension]
                    / selector_amplitude
                )

    return out


def pose_state_max_error(classical_pose, quantum_pose):
    return float(
        np.max(
            np.abs(
                np.asarray(classical_pose)
                - np.asarray(quantum_pose)
            )
        )
    )


def build_occ_block(system):
    n = system["data_qubits"]
    circuit = QuantumCircuit(n + 1, name="SELECT_OCC")
    data = list(range(n))
    selector = n
    circuit.append(controlled_state(system["occs"][0], 0, "O1"), [selector] + data)
    circuit.append(controlled_state(system["occs"][1], 1, "O2"), [selector] + data)
    return circuit


def build_rot_block(system):
    n = system["data_qubits"]
    circuit = QuantumCircuit(n + ROT_QUBITS, name="ROT")
    data = list(range(n))
    rot = list(range(n, n + ROT_QUBITS))
    aux = data[system["aux_index"]]
    for axis_index, axis in enumerate(("x", "y", "z")):
        reg_a, reg_b = rotation_coordinate_indices(system, axis)
        circuit.append(
            build_rotation_gate(system, axis),
            [rot[axis_index], aux]
            + [data[i] for i in reg_a]
            + [data[i] for i in reg_b],
        )
    return circuit


def build_tra_block(system):
    n = system["data_qubits"]
    circuit = QuantumCircuit(n + TRA_QUBITS, name="TRA")
    data = list(range(n))
    tra = list(range(n, n + TRA_QUBITS))
    aux = data[system["aux_index"]]
    for axis_index, axis in enumerate(("x", "y", "z")):
        coord = translation_coordinate_indices(system, axis)
        circuit.append(
            build_translation_gate(system, axis),
            [tra[axis_index], aux] + [data[i] for i in coord],
        )
    return circuit


def build_map_block(system):
    n = system["data_qubits"]
    circuit = QuantumCircuit(n + 1, name="SELECT_MAP")
    data = list(range(n))
    selector = n
    circuit.append(controlled_inverse_state(system["maps"][0], 0, "MAP1"), [selector] + data)
    circuit.append(controlled_inverse_state(system["maps"][1], 1, "MAP2"), [selector] + data)
    return circuit


def make_selector_h_block():
    circuit = QuantumCircuit(2 + ROT_QUBITS + TRA_QUBITS, name="SELECTOR_H")
    circuit.h(range(circuit.num_qubits))
    return circuit


def make_qpe_block(qpe_gate):
    circuit = QuantumCircuit(qpe_gate.num_qubits, name="QAE_QPE_BLOCK")
    circuit.append(qpe_gate, range(qpe_gate.num_qubits))
    return circuit


def build_data_zero_oracle(system):
    n = system["data_qubits"]
    circuit = QuantumCircuit(n + 2 + ROT_QUBITS + TRA_QUBITS, name="S_DATA_ZERO")
    data = list(range(n))
    circuit.x(data)
    if n == 1:
        circuit.z(data[0])
    else:
        circuit.h(data[-1])
        circuit.mcx(data[:-1], data[-1])
        circuit.h(data[-1])
    circuit.x(data)
    return circuit


def build_inner_product(branch, system):
    data = QuantumRegister(system["data_qubits"], "data")
    occ = QuantumRegister(1, "occ_sel")
    map_sel = QuantumRegister(1, "map_sel")
    rot = QuantumRegister(ROT_QUBITS, "rot_sel")
    tra = QuantumRegister(TRA_QUBITS, "tra_sel")
    circuit = QuantumCircuit(data, occ, map_sel, rot, tra, name="INNER_PRODUCT")
    circuit.h(occ)
    circuit.h(map_sel)
    circuit.h(rot)
    circuit.h(tra)
    circuit.compose(
        branch,
        qubits=list(data) + list(occ) + list(map_sel) + list(rot) + list(tra),
        inplace=True,
    )
    return circuit


def build_qpe_checkpoint(branch, system):
    data = QuantumRegister(system["data_qubits"], "data")
    occ = QuantumRegister(1, "occ_sel")
    map_sel = QuantumRegister(1, "map_sel")
    rot = QuantumRegister(ROT_QUBITS, "rot_sel")
    tra = QuantumRegister(TRA_QUBITS, "tra_sel")
    evaluation = QuantumRegister(EVAL_QUBITS, "eval")
    circuit = QuantumCircuit(data, occ, map_sel, rot, tra, evaluation, name="QPE_CHECKPOINT")
    system_qubits = list(data) + list(occ) + list(map_sel) + list(rot) + list(tra)
    circuit.h(occ)
    circuit.h(map_sel)
    circuit.h(rot)
    circuit.h(tra)
    circuit.compose(branch, qubits=system_qubits, inplace=True)
    Q = grover_operator(
        oracle=build_data_zero_oracle(system),
        state_preparation=branch,
        reflection_qubits=list(range(system["data_qubits"])),
        name="Q_BRANCH",
    )
    QPE = phase_estimation(
        num_evaluation_qubits=EVAL_QUBITS,
        unitary=Q,
        name="QPE",
    ).to_gate(label="QAE/QPE")
    circuit.append(QPE, list(evaluation) + system_qubits)
    return circuit, QPE


def qreg(circuit, name):
    for register in circuit.qregs:
        if register.name == name:
            return register
    raise KeyError(name)


def build_qcomp_block(low):
    """Reversibly mark both QPE eigenphase tails with the same FLAG qubit.

    QPE bins are integers y in [0, 2**m - 1].  The threshold is strict:
        FLAG = (y < low) OR (y >= 2**m - low + 1).
    The two intervals are disjoint. Each clean-ancilla comparator XORs its
    result into FLAG and uncomputes its own workspace, so two comparators
    can share the same FLAG and reusable ancillas without adding qubits.
    """
    modulus = 1 << EVAL_QUBITS
    high = threshold_high_limit(low)
    low_cmp = synth_integer_comparator_2s(
        num_state_qubits=EVAL_QUBITS,
        value=low,
        geq=False,
    )
    high_cmp = (
        synth_integer_comparator_2s(
            num_state_qubits=EVAL_QUBITS,
            value=high,
            geq=True,
        )
        if high < modulus
        else None
    )
    low_work = low_cmp.num_qubits - EVAL_QUBITS - 1
    high_work = (
        high_cmp.num_qubits - EVAL_QUBITS - 1
        if high_cmp is not None else 0
    )
    work_count = max(low_work, high_work)
    comp = QuantumCircuit(EVAL_QUBITS + 1 + work_count, name="QCOMP")
    evaluation = list(range(EVAL_QUBITS))
    flag = EVAL_QUBITS
    work = list(range(EVAL_QUBITS + 1, comp.num_qubits))
    comp.append(
        low_cmp.to_gate(label=f"y<{low}"),
        evaluation + [flag] + work[:low_work],
    )
    if high_cmp is not None:
        comp.append(
            high_cmp.to_gate(label=f"y>={high}"),
            evaluation + [flag] + work[:high_work],
        )
    return comp


def add_comparator(qpe_circuit, low):
    comparator = build_qcomp_block(low)
    work_count = comparator.num_qubits - EVAL_QUBITS - 1
    circuit = qpe_circuit.copy(name="A_THRESHOLD")
    flag = QuantumRegister(1, "flag")
    circuit.add_register(flag)
    if work_count > 0:
        work = AncillaRegister(work_count, "work")
        circuit.add_register(work)
        work_list = list(work)
    else:
        work_list = []
    evaluation = qreg(circuit, "eval")
    circuit.append(
        comparator.to_gate(label="QCOMP"),
        list(reversed(evaluation)) + [flag[0]] + work_list,
    )
    return circuit, comparator


def build_qsel(qae_circuit, iterations):
    flag = qreg(qae_circuit, "flag")
    flag_index = qae_circuit.find_bit(flag[0]).index
    oracle = qae_circuit.copy_empty_like(name="S_FLAG")
    oracle.z(flag_index)
    QSEL = grover_operator(
        oracle=oracle,
        state_preparation=qae_circuit,
        reflection_qubits=list(range(qae_circuit.num_qubits)),
        name="Q_SEL",
    )
    circuit = qae_circuit.copy(name="FULL_SEARCH")
    gate = QSEL.to_gate(label="Q_SEL")
    for _ in range(iterations):
        circuit.append(gate, circuit.qubits)
    return circuit, QSEL


def selector_index(m, o, r, t):
    return o + 2 * m + 4 * r + 32 * t


def selector_label(m, o, r, t):
    return f"MAP{m + 1} OCC{o + 1} R={r:03b} T={t:03b}"


def selector_qargs(circuit):
    occ = qreg(circuit, "occ_sel")
    map_sel = qreg(circuit, "map_sel")
    rot = qreg(circuit, "rot_sel")
    tra = qreg(circuit, "tra_sel")
    return [
        circuit.find_bit(occ[0]).index,
        circuit.find_bit(map_sel[0]).index,
        *[circuit.find_bit(q).index for q in rot],
        *[circuit.find_bit(q).index for q in tra],
    ]


def classical_inner_products(system):
    out = np.zeros((2, 2, 8, 8), dtype=float)
    for m in range(2):
        for o in range(2):
            for r in range(8):
                for t in range(8):
                    posed = pose_vector(system["occs"][o], system, r, t)
                    out[m, o, r, t] = float(np.vdot(system["maps"][m], posed).real)
    return out


def quantum_inner_products(circuit, system):
    state = simulate_statevector(circuit)
    selector_prob = np.asarray(state.probabilities(qargs=selector_qargs(circuit)), dtype=float)
    n = system["data_qubits"]
    out = np.zeros((2, 2, 8, 8), dtype=np.complex128)
    for m in range(2):
        for o in range(2):
            for r in range(8):
                for t in range(8):
                    s = selector_index(m, o, r, t)
                    selector_amplitude = math.sqrt(float(selector_prob[s]))
                    if selector_amplitude <= TOL:
                        raise AssertionError("zero selector amplitude")
                    out[m, o, r, t] = state.data[s << n] / selector_amplitude
    return out


def qpe_probability_at_bins(phase, bins):
    modulus = 1 << EVAL_QUBITS
    phase = float(phase % 1.0)
    delta = phase - bins.astype(np.float64) / float(modulus)
    ratio = np.sinc(modulus * delta) / np.sinc(delta)
    return np.square(np.abs(ratio))


def classical_qpe_distribution(amplitude):
    modulus = 1 << EVAL_QUBITS
    amplitude = float(np.clip(abs(amplitude), 0.0, 1.0))
    phase = math.asin(amplitude) / math.pi
    bins = np.arange(modulus, dtype=np.int64)
    positive = qpe_probability_at_bins(phase, bins)
    negative = qpe_probability_at_bins((-phase) % 1.0, bins)
    probabilities = 0.5 * (positive + negative)
    return probabilities / float(np.sum(probabilities))


def classical_qpe_matrix(inner_values):
    modulus = 1 << EVAL_QUBITS
    out = np.zeros((modulus, SELECTOR_STATES), dtype=float)
    for m in range(2):
        for o in range(2):
            for r in range(8):
                for t in range(8):
                    s = selector_index(m, o, r, t)
                    out[:, s] = classical_qpe_distribution(inner_values[m, o, r, t])
    return out


def quantum_qpe_matrix(circuit):
    state = simulate_statevector(circuit)
    evaluation = qreg(circuit, "eval")
    qargs = selector_qargs(circuit) + [
        circuit.find_bit(q).index for q in reversed(evaluation)
    ]
    p = np.asarray(state.probabilities(qargs=qargs), dtype=float)
    matrix = p.reshape(1 << EVAL_QUBITS, SELECTOR_STATES)
    totals = np.sum(matrix, axis=0)
    return matrix / totals[np.newaxis, :]


def classical_flag_probabilities(qpe_matrix, low):
    high = threshold_high_limit(low)
    return (
        np.sum(qpe_matrix[:low, :], axis=0)
        + np.sum(qpe_matrix[high:, :], axis=0)
    )


def quantum_selector_flag_results(circuit):
    state = simulate_statevector(circuit)
    flag = qreg(circuit, "flag")
    qargs = selector_qargs(circuit) + [circuit.find_bit(flag[0]).index]
    p = np.asarray(state.probabilities(qargs=qargs), dtype=float)
    matrix = p.reshape(2, SELECTOR_STATES)
    selector_total = np.sum(matrix, axis=0)
    result = {}
    for m in range(2):
        for o in range(2):
            for r in range(8):
                for t in range(8):
                    s = selector_index(m, o, r, t)
                    total = float(selector_total[s])
                    joint = float(matrix[1, s])
                    result[(m, o, r, t)] = {
                        "selector": total,
                        "joint": joint,
                        "conditional": joint / total if total > 0.0 else 0.0,
                    }
    return float(np.sum(matrix[1, :])), result


def classical_selector_results(flag_conditional, iterations):
    flag_conditional = np.asarray(flag_conditional, dtype=float)
    p0 = 1.0 / SELECTOR_STATES
    initial_good = float(np.mean(flag_conditional))
    theta = math.asin(math.sqrt(float(np.clip(initial_good, 0.0, 1.0))))
    amplified_good = math.sin((2 * iterations + 1) * theta) ** 2
    if initial_good <= TOL or initial_good >= 1.0 - TOL:
        good_scale = 1.0
        bad_scale = 1.0
        amplified_good = initial_good
    else:
        good_scale = amplified_good / initial_good
        bad_scale = (1.0 - amplified_good) / (1.0 - initial_good)

    before = {}
    after = {}
    for m in range(2):
        for o in range(2):
            for r in range(8):
                for t in range(8):
                    s = selector_index(m, o, r, t)
                    f = float(flag_conditional[s])
                    joint_before = p0 * f
                    before[(m, o, r, t)] = {
                        "selector": p0,
                        "joint": joint_before,
                        "conditional": f,
                    }
                    joint_after = p0 * f * good_scale
                    bad_after = p0 * (1.0 - f) * bad_scale
                    selector_after = joint_after + bad_after
                    after[(m, o, r, t)] = {
                        "selector": selector_after,
                        "joint": joint_after,
                        "conditional": joint_after / selector_after if selector_after > 0.0 else 0.0,
                    }
    return initial_good, amplified_good, before, after


def format_a0_results(title, values):
    lines = [title, "MAP OCC ROT TRA              A0"]
    for m in range(2):
        for o in range(2):
            for r in range(8):
                for t in range(8):
                    value = values[m, o, r, t]
                    if np.iscomplexobj(values):
                        text = f"{value.real: .10f}{value.imag:+.3e}j"
                    else:
                        text = f"{float(value): .10f}"
                    lines.append(
                        f"{m + 1:3d} {o + 1:3d} {r:03b} {t:03b}  {text}"
                    )
    return "\n".join(lines)



def format_pose_state_validation(classical_pose, quantum_pose):
    lines = [
        "POSE STATE VALIDATION: CLASSICAL vs QUANTUM",
        "OCC ROT TRA     MAX_STATE_ERROR",
    ]

    overall = 0.0

    for o in range(2):
        for r in range(1 << ROT_QUBITS):
            for t in range(1 << TRA_QUBITS):
                error = float(
                    np.max(
                        np.abs(
                            classical_pose[o, r, t, :]
                            - quantum_pose[o, r, t, :]
                        )
                    )
                )
                overall = max(overall, error)

                lines.append(
                    f"{o + 1:3d} {r:03b} {t:03b}  "
                    f"{error:18.3e}"
                )

    lines.append("")
    lines.append(
        f"MAX_POSE_STATE_ERROR {overall:.3e}"
    )

    return "\\n".join(lines)


def format_rot_tra_only(classical_a0, quantum_a0):
    lines = [
        "ROT ONLY (TRA=000)",
        "MAP OCC ROT TRA      CLASSICAL_A0       QUANTUM_A0          ERROR",
    ]
    for m in range(2):
        for o in range(2):
            for r in range(8):
                c = float(classical_a0[m, o, r, 0])
                q = complex(quantum_a0[m, o, r, 0])
                lines.append(
                    f"{m + 1:3d} {o + 1:3d} {r:03b} 000  "
                    f"{c:16.10f}  {q.real:16.10f}  {abs(c - q):12.3e}"
                )
    lines.extend([
        "",
        "TRANS ONLY (ROT=000)",
        "MAP OCC ROT TRA      CLASSICAL_A0       QUANTUM_A0          ERROR",
    ])
    for m in range(2):
        for o in range(2):
            for t in range(8):
                c = float(classical_a0[m, o, 0, t])
                q = complex(quantum_a0[m, o, 0, t])
                lines.append(
                    f"{m + 1:3d} {o + 1:3d} 000 {t:03b}  "
                    f"{c:16.10f}  {q.real:16.10f}  {abs(c - q):12.3e}"
                )
    return "\n".join(lines)


def format_selector_results(title, results):
    lines = [title]
    for m in range(2):
        for o in range(2):
            for r in range(8):
                for t in range(8):
                    row = results[(m, o, r, t)]
                    lines.append(
                        f"{selector_label(m, o, r, t)}  "
                        f"P(selector)={row['selector']:10.7f}  "
                        f"P(selector,FLAG=1)={row['joint']:10.7f}  "
                        f"P(FLAG=1|selector)={row['conditional']:10.7f}"
                    )
    return "\n".join(lines)



def format_qsel_comparison(title, quantum_results, classical_results):
    ordered = sorted(
        quantum_results.keys(),
        key=lambda key: quantum_results[key]["joint"],
        reverse=True,
    )
    header = (
        f"{'MAP':<6}"
        f"{'OCC':<6}"
        f"{'ROT':<7}"
        f"{'TRA':<7}"
        f"{'P(selector)':>14}  "
        f"{'Q_P(selector,FLAG=1)':>22}  "
        f"{'Q_P(FLAG=1|selector)':>23}  "
        f"{'C_P(selector,FLAG=1)':>22}  "
        f"{'C_P(FLAG=1|selector)':>23}"
    )
    lines = [title, header]
    for m, o, r, t in ordered:
        q = quantum_results[(m, o, r, t)]
        c = classical_results[(m, o, r, t)]
        lines.append(
            f"{'MAP' + str(m + 1):<6}"
            f"{'OCC' + str(o + 1):<6}"
            f"{'R=' + format(r, '03b'):<7}"
            f"{'T=' + format(t, '03b'):<7}"
            f"{q['selector']:14.7f}  "
            f"{q['joint']:22.7f}  "
            f"{q['conditional']:23.7f}  "
            f"{c['joint']:22.7f}  "
            f"{c['conditional']:23.7f}"
        )
    return "\n".join(lines)

def qpe_validation_lines(classical_qpe, quantum_qpe):
    lines = ["QPE DISTRIBUTION: CLASSICAL vs QUANTUM"]
    modulus = 1 << EVAL_QUBITS
    for m in range(2):
        for o in range(2):
            for r in range(8):
                for t in range(8):
                    s = selector_index(m, o, r, t)
                    lines.append("")
                    lines.append(selector_label(m, o, r, t))
                    lines.append("BIN      CLASSICAL       QUANTUM          ERROR")
                    for y in range(modulus):
                        c = float(classical_qpe[y, s])
                        q = float(quantum_qpe[y, s])
                        lines.append(
                            f"{y:3d}  {c:14.10f}  {q:14.10f}  {abs(c - q):12.3e}"
                        )
    return lines


def validation_summary(
    classical_a0,
    quantum_a0,
    classical_qpe,
    quantum_qpe,
    classical_before,
    classical_after,
    quantum_before,
    quantum_after,
    classical_p_before,
    classical_p_after,
    quantum_p_before,
    quantum_p_after,
    pose_error,
    tolerance,
):
    a0_error = float(np.max(np.abs(classical_a0 - quantum_a0)))
    qpe_error = float(np.max(np.abs(classical_qpe - quantum_qpe)))
    flag_error = 0.0
    qsel_before_error = 0.0
    qsel_after_error = 0.0
    for key in classical_before:
        c0 = classical_before[key]
        q0 = quantum_before[key]
        c1 = classical_after[key]
        q1 = quantum_after[key]
        flag_error = max(flag_error, abs(c0["conditional"] - q0["conditional"]))
        qsel_before_error = max(qsel_before_error, abs(c0["selector"] - q0["selector"]))
        qsel_after_error = max(qsel_after_error, abs(c1["selector"] - q1["selector"]))

    errors = {
        "POSE ROT/TRA": float(pose_error),
        "INNER A0": a0_error,
        "QPE": qpe_error,
        "FLAG": flag_error,
        "Q_SEL before": qsel_before_error,
        "Q_SEL after": qsel_after_error,
        "PFLAG before": abs(classical_p_before - quantum_p_before),
        "PFLAG after": abs(classical_p_after - quantum_p_after),
    }
    lines = ["VALIDATION SUMMARY"]
    for name, error in errors.items():
        status = "PASS" if error <= tolerance else "FAIL"
        lines.append(f"{name:<14s} max_error={error:12.3e}  {status}")
    overall = all(error <= tolerance for error in errors.values())
    lines.append(f"tolerance      = {tolerance:.3e}")
    lines.append(f"OVERALL        = {'PASS' if overall else 'FAIL'}")
    return lines


def state_box_name(operation):
    candidates = [getattr(operation, "label", None), getattr(operation, "name", None)]
    base_gate = getattr(operation, "base_gate", None)
    if base_gate is not None:
        candidates.extend([getattr(base_gate, "label", None), getattr(base_gate, "name", None)])
    for value in candidates:
        if value is None:
            continue
        text = str(value).upper()
        for label in STATE_BOX_LABELS:
            if label in text:
                return label
    return None


def append_algorithmic_operation(target, operation, qargs, cargs, depth=0):
    if depth > 100:
        raise RuntimeError("algorithmic decomposition exceeded 100 levels")
    box_name = state_box_name(operation)
    if box_name is not None:
        target.append(operation, qargs, cargs)
        return
    name = str(operation.name).lower()
    if name in ALGORITHM_GATES:
        target.append(operation, qargs, cargs)
        return
    definition = getattr(operation, "definition", None)
    if definition is None or len(definition.data) == 0:
        target.append(operation, qargs, cargs)
        return
    for instruction in definition.data:
        local_qargs = [qargs[definition.find_bit(qubit).index] for qubit in instruction.qubits]
        local_cargs = [cargs[definition.find_bit(clbit).index] for clbit in instruction.clbits]
        append_algorithmic_operation(
            target,
            instruction.operation,
            local_qargs,
            local_cargs,
            depth + 1,
        )


def algorithmic_decompose(circuit):
    out = circuit.copy_empty_like(name=f"{circuit.name}_ALG")
    for instruction in circuit.data:
        append_algorithmic_operation(
            out,
            instruction.operation,
            list(instruction.qubits),
            list(instruction.clbits),
        )
    return out


def algorithmic_gate_name(operation):
    box_name = state_box_name(operation)
    return box_name if box_name is not None else str(operation.name)


def algorithmic_stats(circuit):
    alg = algorithmic_decompose(circuit)
    types = {}
    for instruction in alg.data:
        name = algorithmic_gate_name(instruction.operation)
        types[name] = types.get(name, 0) + 1
    return {
        "qubits": alg.num_qubits,
        "gates": len(alg.data),
        "depth": alg.depth(),
        "types": types,
    }



def ibm_basis_stats(circuit):
    ibm = transpile(
        circuit,
        basis_gates=IBM_BASIS_GATES,
        optimization_level=0,
    )
    return {
        "qubits": ibm.num_qubits,
        "gates": ibm.size(),
        "depth": ibm.depth(),
        "types": dict(ibm.count_ops()),
    }


def format_stats(name, stats):
    return (
        f"{name}\n"
        f"  qubits = {stats['qubits']}\n"
        f"  gates  = {stats['gates']}\n"
        f"  depth  = {stats['depth']}\n"
        f"  types  = {stats['types']}\n"
    )



def visual_controlled_block(num_targets, label, ctrl_state):
    matrix = np.eye(1 << num_targets, dtype=np.complex128)
    base = UnitaryGate(matrix, label=label)
    return base.control(2, label=label, ctrl_state=ctrl_state, annotated=False)


def normalize_wire_label(text):
    label = str(text)
    for old, new in (("$", ""), ("\\", ""), ("{", ""), ("}", ""), (" ", "")):
        label = label.replace(old, new)
    label = label.replace("mathrm", "")
    label = label.replace("text", "")
    return label.lower()


def find_wire_y_positions(ax):
    out = {}
    targets = ("occ_sel", "map_sel", "rot_", "trans_", "flag")
    for text in ax.texts:
        raw = text.get_text()
        label = normalize_wire_label(raw)
        if not any(key in label for key in targets):
            continue
        x, y = text.get_position()
        if label not in out or x < out[label][0]:
            out[label] = (x, y)
    return {key: value[1] for key, value in out.items()}


def draw_meter_icon(ax, x, y, size=0.78):
    from matplotlib.patches import Arc, FancyArrowPatch, Rectangle

    width = size
    height = size * 0.72
    rect = Rectangle(
        (x - width / 2.0, y - height / 2.0),
        width,
        height,
        fill=False,
        linewidth=1.1,
        color="black",
        clip_on=False,
    )
    arc = Arc(
        (x, y - 0.02 * size),
        width=width * 0.58,
        height=height * 0.56,
        theta1=0,
        theta2=180,
        linewidth=1.0,
        color="black",
        clip_on=False,
    )
    arrow = FancyArrowPatch(
        (x, y - 0.02 * size),
        (x + width * 0.18, y + height * 0.18),
        arrowstyle='-|>',
        mutation_scale=8,
        linewidth=0.9,
        color="black",
        clip_on=False,
    )
    ax.add_patch(rect)
    ax.add_patch(arc)
    ax.add_patch(arrow)


def add_measurement_meters(ax):
    positions = find_wire_y_positions(ax)
    labels = [
        "occ_sel",
        "map_sel",
        "rot_0",
        "rot_1",
        "rot_2",
        "trans_0",
        "trans_1",
        "trans_2",
        "flag",
    ]
    x_min, x_max = ax.get_xlim()
    icon_x = x_max + 1.00
    ax.set_xlim(x_min, x_max + 1.75)
    for name in labels:
        y = positions.get(name)
        if y is None:
            continue
        ax.plot(
            [x_max - 0.05, icon_x - 0.42],
            [y, y],
            color="black",
            linewidth=1.0,
            clip_on=False,
        )
        draw_meter_icon(ax, icon_x, y)



def build_box_drawing_circuit(
    system,
    qpe_gate,
    comparator,
    low,
    qsel,
    iterations,
):
    physical_data_qubits = system["data_qubits"] - 1

    data = QuantumRegister(physical_data_qubits, "data")
    aux = QuantumRegister(1, "aux")
    occ = QuantumRegister(1, "occ_sel")
    rot = QuantumRegister(ROT_QUBITS, "rot")
    trans = QuantumRegister(TRA_QUBITS, "trans")
    map_sel = QuantumRegister(1, "map_sel")
    evaluation = QuantumRegister(EVAL_QUBITS, "eval")
    flag = QuantumRegister(1, "flag")

    work_count = (
        comparator.num_qubits - EVAL_QUBITS - 1
    )
    work = (
        AncillaRegister(work_count, "work")
        if work_count > 0
        else None
    )

    # reverse_bits=True reverses register display order.
    # This creation order therefore renders top-to-bottom as:
    # work, flag, eval, map_sel, trans, rot, occ_sel, aux, data.
    qregs = [
        data,
        aux,
        occ,
        rot,
        trans,
        map_sel,
        evaluation,
        flag,
    ]
    if work is not None:
        qregs.append(work)

    circuit = QuantumCircuit(*qregs, name="CIRCUIT_BOX")

    circuit.h(occ)
    circuit.h(map_sel)
    circuit.h(rot)
    circuit.h(trans)

    combined_data = list(data) + [aux[0]]

    circuit.append(
        controlled_state(system["occs"][0], 0, "O1"),
        [occ[0]] + combined_data,
    )
    circuit.append(
        controlled_state(system["occs"][1], 1, "O2"),
        [occ[0]] + combined_data,
    )

    for axis_index, axis in enumerate(("x", "y", "z")):
        reg_a, reg_b = rotation_coordinate_indices(system, axis)
        targets = (
            [combined_data[i] for i in reg_a]
            + [combined_data[i] for i in reg_b]
        )
        gate = visual_controlled_block(
            len(targets),
            f"ROT-{axis.upper()}",
            1,
        )
        circuit.append(
            gate,
            [rot[axis_index], aux[0]] + targets,
        )

    for axis_index, axis in enumerate(("x", "y", "z")):
        coord = translation_coordinate_indices(system, axis)
        targets = [combined_data[i] for i in coord]
        gate = visual_controlled_block(
            len(targets),
            f"T{axis.upper()}",
            1,
        )
        circuit.append(
            gate,
            [trans[axis_index], aux[0]] + targets,
        )

    circuit.append(
        controlled_inverse_state(system["maps"][0], 0, "MAP1"),
        [map_sel[0]] + combined_data,
    )
    circuit.append(
        controlled_inverse_state(system["maps"][1], 1, "MAP2"),
        [map_sel[0]] + combined_data,
    )

    # Preserve the operational qubit ordering expected by QPE/QSEL.
    system_qubits = (
        combined_data
        + list(occ)
        + list(map_sel)
        + list(rot)
        + list(trans)
    )
    circuit.append(
        qpe_gate,
        list(evaluation) + system_qubits,
    )

    work_list = list(work) if work is not None else []
    circuit.append(
        comparator.to_gate(label="QCOMP"),
        list(reversed(evaluation))
        + [flag[0]]
        + work_list,
    )

    qsel_gate = qsel.to_gate(label="Q_SEL")
    qsel_qargs = (
        system_qubits
        + list(evaluation)
        + [flag[0]]
        + work_list
    )
    for _ in range(iterations):
        circuit.append(qsel_gate, qsel_qargs)

    return circuit

def write_circuit_svg(circuit, path, low):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    figure = circuit.draw(output="mpl", fold=-1, reverse_bits=True)
    axes = figure.axes
    if axes:
        ax = axes[0]
        add_measurement_meters(ax)
        small_control_labels = {
            "O1", "O2", "MAP1", "MAP2",
            "ROT-X", "ROT-Y", "ROT-Z",
            "TX", "TY", "TZ",
        }
        for text in ax.texts:
            if text.get_text().strip() in small_control_labels and text.get_fontsize() <= 8.5:
                text.set_visible(False)
    figure.savefig(path, format="svg", bbox_inches="tight")
    plt.close(figure)

def main():
    global EVAL_QUBITS
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {PROGRAM_VERSION}",
    )
    parser.add_argument("--grid-size", type=int, choices=(2, 4, 8), default=2)
    parser.add_argument("--eval-qubits", type=int, default=4)
    parser.add_argument("--input-file", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--qsel-iterations", type=int, default=1)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--precision", choices=("single", "double"), default="double")
    parser.add_argument("--validation-tolerance", type=float, default=None)
    parser.add_argument("--circuit-file", type=Path, required=True)
    parser.add_argument("--circuit-svg-file", type=Path, default=None)
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument("--rot-tra-file", type=Path, default=None)
    parser.add_argument("--qsel-file", type=Path, default=None)
    parser.add_argument(
        "--paper-summary-file",
        type=Path,
        default=None,
        help="optional compact paper-level summary",
    )
    args = parser.parse_args()

    EVAL_QUBITS = args.eval_qubits
    if EVAL_QUBITS < 1:
        raise ValueError("--eval-qubits must be >= 1")
    if args.qsel_iterations < 0:
        raise ValueError("--qsel-iterations must be >= 0")
    configure_sim_device(args.device, args.precision)
    tolerance = (
        float(args.validation_tolerance)
        if args.validation_tolerance is not None
        else (1e-5 if args.precision == "single" else 1e-8)
    )

    system = load_system(args.input_file, args.grid_size)
    classical_a0 = classical_inner_products(system)
    threshold = float(args.threshold) if args.threshold is not None else choose_threshold(classical_a0)
    if not 0.0 < threshold <= 1.0:
        raise ValueError("threshold must satisfy 0 < threshold <= 1")
    low = threshold_low_limit(threshold)

    branch = build_branch_preparation(system)

    # Direct pose-state validation checkpoint, before MAP.
    pose_checkpoint = build_pose_checkpoint(system)
    classical_pose = classical_pose_states(system)
    quantum_pose = quantum_pose_states(
        pose_checkpoint,
        system,
    )
    pose_error = pose_state_max_error(
        classical_pose,
        quantum_pose,
    )

    inner = build_inner_product(branch, system)
    qpe, qpe_gate = build_qpe_checkpoint(branch, system)
    qae, low_cmp = add_comparator(qpe, low)
    full, qsel = build_qsel(qae, args.qsel_iterations)

    classical_qpe = classical_qpe_matrix(classical_a0)
    classical_flag = classical_flag_probabilities(classical_qpe, low)
    (
        classical_p_before,
        classical_p_after,
        classical_before,
        classical_after,
    ) = classical_selector_results(classical_flag, args.qsel_iterations)

    quantum_a0 = quantum_inner_products(inner, system)
    quantum_qpe = quantum_qpe_matrix(qpe)
    quantum_p_before, quantum_before = quantum_selector_flag_results(qae)
    quantum_p_after, quantum_after = quantum_selector_flag_results(full)

    lines = [
        f"PROGRAM_VERSION = {PROGRAM_VERSION}",
        "",
        "CLASSICAL ANALYTICAL RESULTS",
        "",
        format_a0_results("INNER PRODUCT A0", classical_a0),
        f"threshold       = {threshold:10.7f}",
        f"compare         = {qcomp_label(low)}",
        f"selector states = {SELECTOR_STATES}",
        f"Q_SEL iterations= {args.qsel_iterations}",
        f"P(FLAG=1) before Q_SEL = {classical_p_before:10.7f}",
        f"P(FLAG=1) after  Q_SEL = {classical_p_after:10.7f}",
        "",
        "QUANTUM CIRCUIT RESULTS",
        "",
        format_a0_results("INNER PRODUCT A0", quantum_a0),
        f"threshold       = {threshold:10.7f}",
        f"compare         = {qcomp_label(low)}",
        f"selector states = {SELECTOR_STATES}",
        f"Q_SEL iterations= {args.qsel_iterations}",
        f"P(FLAG=1) before Q_SEL = {quantum_p_before:10.7f}",
        f"P(FLAG=1) after  Q_SEL = {quantum_p_after:10.7f}",
        "",
    ]
    lines.extend(qpe_validation_lines(classical_qpe, quantum_qpe))
    lines.append("")
    validation_lines = validation_summary(
        classical_a0,
        quantum_a0,
        classical_qpe,
        quantum_qpe,
        classical_before,
        classical_after,
        quantum_before,
        quantum_after,
        classical_p_before,
        classical_p_after,
        quantum_p_before,
        quantum_p_after,
        pose_error,
        tolerance,
    )
    lines.extend(validation_lines)

    selector_h = make_selector_h_block()
    occ_block = build_occ_block(system)
    rot_block = build_rot_block(system)
    tra_block = build_tra_block(system)
    map_block = build_map_block(system)
    qpe_block = make_qpe_block(qpe_gate)

    blocks = (
        ("SELECTOR H", selector_h),
        ("SELECT OCC", occ_block),
        ("ROT", rot_block),
        ("TRA", tra_block),
        ("SELECT MAP inverse", map_block),
        ("QAE/QPE", qpe_block),
        ("COMPARATOR", low_cmp),
        ("Q_SEL ONE ITERATION", qsel),
    )

    for title, stats_fn in (
        ("BLOCK ALGORITHMIC GATE ESTIMATE", algorithmic_stats),
        ("BLOCK IBM BASIS GATE ESTIMATE", ibm_basis_stats),
    ):
        lines.extend(["", title, ""])
        for name, block in blocks:
            lines.append(format_stats(name, stats_fn(block)).rstrip())
            lines.append("")

        before_stats = stats_fn(qae)
        lines.extend([
            "TOTAL BEFORE Q_SEL",
            f"  qubits = {before_stats['qubits']}",
            f"  gates  = {before_stats['gates']}",
            f"  depth  = {before_stats['depth']}",
            f"  types  = {before_stats['types']}",
            "",
        ])

        after_stats = stats_fn(full)
        lines.extend([
            "TOTAL AFTER Q_SEL",
            f"  qubits = {after_stats['qubits']}",
            f"  gates  = {after_stats['gates']}",
            f"  depth  = {after_stats['depth']}",
            f"  types  = {after_stats['types']}",
        ])

    box_circuit = build_box_drawing_circuit(
        system,
        qpe_gate,
        low_cmp,
        low,
        qsel,
        args.qsel_iterations,
    )
    svg_path = (
        args.circuit_svg_file
        if args.circuit_svg_file is not None
        else Path(str(args.circuit_file) + ".svg")
    )

    rot_tra_path = (
        args.rot_tra_file
        if args.rot_tra_file is not None
        else Path(str(args.output_file) + ".rot_tra")
    )
    qsel_path = (
        args.qsel_file
        if args.qsel_file is not None
        else Path(str(args.output_file) + ".qsel")
    )

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    args.circuit_file.parent.mkdir(parents=True, exist_ok=True)
    rot_tra_path.parent.mkdir(parents=True, exist_ok=True)
    qsel_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_file.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    rot_tra_path.write_text(
        format_pose_state_validation(
            classical_pose,
            quantum_pose,
        )
        + "\n\n"
        + format_rot_tra_only(
            classical_a0,
            quantum_a0,
        )
        + "\n",
        encoding="utf-8",
    )
    qsel_path.write_text(
        format_qsel_comparison("BEFORE Q_SEL", quantum_before, classical_before)
        + "\n\n"
        + format_qsel_comparison("AFTER Q_SEL", quantum_after, classical_after)
        + "\n",
        encoding="utf-8",
    )

    if args.paper_summary_file is not None:
        args.paper_summary_file.parent.mkdir(parents=True, exist_ok=True)
        args.paper_summary_file.write_text(
            paper_summary_text(
                system=system,
                threshold=threshold,
                low=low,
                qsel_iterations=args.qsel_iterations,
                classical_a0=classical_a0,
                qae_circuit=qae,
                full_circuit=full,
                quantum_p_before=quantum_p_before,
                quantum_before=quantum_before,
                quantum_p_after=quantum_p_after,
                quantum_after=quantum_after,
                validation_lines=validation_lines,
                tolerance=tolerance,
            ),
            encoding="utf-8",
        )

    args.circuit_file.write_text(
        str(box_circuit.draw(output="text", fold=-1, reverse_bits=True)) + "\n",
        encoding="utf-8",
    )
    write_circuit_svg(box_circuit, svg_path, low)


if __name__ == "__main__":
    main()
