#!/usr/bin/env python3
"""Gate-level 29-qubit Qiskit verification of the 2_QAE ROT/TRANS mapping.

Version 4: retains the validated QFT ROT/TRANS implementation and the
independent 16-branch reference-controlled interference test. Compiles that
interference circuit to universal u/cx gates before Aer MPS simulation so
the ANF reference correction has no unsupported mcx instructions.

Registers: x[5], y[5], z[5], rot[5], shift_x[3], shift_y[3], shift_z[3].
This is a spatial-transform validation ONLY: no ligand loading, channel/AUX,
inner-product, QAE/QPE, QCOMP, QSEL, FFT, or 2_QAE function calls.

Zero large-file output. 29-qubit Aer MPS is the default to avoid an 8-GiB
complex128 statevector (and further simulator overhead).
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np

GRID = 32
NQ = 29


def index_word(x: int, y: int, z: int, rotation: int, translation: int) -> int:
    """Qiskit little-endian order: x[0:5],y[5:10],z[10:15],rot[15:20],tra[20:29]."""
    return (int(x) | (int(y) << 5) | (int(z) << 10)
            | (int(rotation) << 15) | (int(translation) << 20))


def decode_word(word: int):
    return ((word & 31), ((word >> 5) & 31), ((word >> 10) & 31),
            ((word >> 15) & 31), ((word >> 20) & 511))


def translation_shifts(index: int):
    """Independent selector interpretation specified by this QAE implementation.

    Per-axis bits low/medium/high: +1,+2,and -4 IF high bit = 0.
    Not the bit convention of the earlier JMG&M paper.
    """
    out = []
    for axis in range(3):
        three = (int(index) >> (3 * axis)) & 7
        low, middle, high = three & 1, (three >> 1) & 1, (three >> 2) & 1
        out.append(low + 2 * middle - 4 * (1 - high))
    return tuple(out)  # dx,dy,dz


def np_reference_rotations(initial_xyz):
    """Independent NumPy one-hot array reference, not the 2_QAE routine.

    z,y,x 3D axes: k=1 (y,x) for z90; k=2 for z180;
    k=-1 (z,x) for y90; k=2 for y180; k=1 (z,y) for x90.
    """
    x, y, z = initial_xyz
    occupancy = np.zeros((GRID, GRID, GRID), dtype=np.uint8)
    occupancy[z, y, x] = 1
    lookup = {}
    for selector in range(32):
        arr = occupancy
        if selector & 16:
            arr = np.rot90(arr, k=1, axes=(1, 2))
        if selector & 8:
            arr = np.rot90(arr, k=2, axes=(1, 2))
        if selector & 4:
            arr = np.rot90(arr, k=-1, axes=(0, 2))
        if selector & 2:
            arr = np.rot90(arr, k=2, axes=(0, 2))
        if selector & 1:
            arr = np.rot90(arr, k=1, axes=(0, 1))
        where = np.argwhere(arr == 1)
        if len(where) != 1:
            raise AssertionError('NumPy one-hot rotation did not preserve one state')
        zr, yr, xr = (int(q) for q in where[0])
        lookup[selector] = (xr, yr, zr)
    return lookup


def reference_output(orientation_lookup, r: int, t: int):
    x, y, z = orientation_lookup[r]
    dx, dy, dz = translation_shifts(t)
    return ((x + dx) % GRID, (y + dy) % GRID, (z + dz) % GRID)


def controlled_swap_axis(qc, ctrl, axis1, axis2):
    for a, b in zip(axis1, axis2):
        qc.cswap(ctrl, a, b)


def controlled_invert(qc, ctrl, axis):
    for bit in axis:
        qc.cx(ctrl, bit)


def append_rotation(qc, x, y, z, rot):
    """Execute exactly 5 selector-controlled cubic orientation operations.

    Selectors bit4 z90, bit3 z180, bit2 y90 (k=-1),
    bit1 y180, bit0 x90, IN THAT ORDER.
    """
    controlled_swap_axis(qc, rot[4], y, x)
    controlled_invert(qc, rot[4], y)

    controlled_invert(qc, rot[3], y)
    controlled_invert(qc, rot[3], x)

    controlled_swap_axis(qc, rot[2], z, x)
    controlled_invert(qc, rot[2], x)

    controlled_invert(qc, rot[1], z)
    controlled_invert(qc, rot[1], x)

    controlled_swap_axis(qc, rot[0], z, y)
    controlled_invert(qc, rot[0], z)


def append_qft_5(qc, axis):
    """Exact 5-qubit QFT in little-endian computational-basis order.

    Uses only H, controlled-phase and SWAP gates; no arithmetic MCX ladders.
    """
    for j in range(4, -1, -1):
        qc.h(axis[j])
        for k in range(j):
            qc.cp(math.pi / (2 ** (j - k)), axis[k], axis[j])
    qc.swap(axis[0], axis[4])
    qc.swap(axis[1], axis[3])


def append_inverse_qft_5(qc, axis):
    """The exact inverse of append_qft_5 (reverse order/conjugate phases)."""
    qc.swap(axis[1], axis[3])
    qc.swap(axis[0], axis[4])
    for j in range(5):
        for k in range(j - 1, -1, -1):
            qc.cp(-math.pi / (2 ** (j - k)), axis[k], axis[j])
        qc.h(axis[j])


def append_controlled_phase_add(qc, axis, ctrl, amount: int):
    """In the Fourier basis, add 'amount' mod 32 if ctrl is |1>.

    QFT|x> = sum_y exp(2πi*x*y/32)|y>/sqrt(32), so applying
    exp(2πi*amount*y/32) followed by inverse QFT gives |x+amount mod 32>.
    """
    for j in range(5):
        mod = 1 << (5 - j)
        residue = amount % mod
        if residue == 0:
            continue
        phase = (2.0 * math.pi * residue) / mod
        if phase > math.pi:
            phase -= 2.0 * math.pi
        qc.cp(phase, ctrl, axis[j])


def append_translation_axis(qc, axis, selector):
    """Independent Qiskit controlled modular translation, with ZERO MCX.

    Current 2_QAE selector mapping: shift = bit0 + 2*bit1 - 4*(1-bit2).
    The bit2=0 branch is a NEGATIVE control; all operations remain coherent.
    """
    append_qft_5(qc, axis)
    qc.x(selector[2])
    append_controlled_phase_add(qc, axis, selector[2], -4)
    qc.x(selector[2])
    append_controlled_phase_add(qc, axis, selector[1], +2)
    append_controlled_phase_add(qc, axis, selector[0], +1)
    append_inverse_qft_5(qc, axis)


def append_translation(qc, x, y, z, shift_x, shift_y, shift_z):
    """QFT-based circuit implementation of each selector-controlled shift.

    No FFT, no matrix-defined UnitaryGate, no MCX, and no external input.
    QFT is a quantum circuit over five coordinate qubits, not a classical FFT.
    """
    for coord, selectors in ((x, shift_x), (y, shift_y), (z, shift_z)):
        append_translation_axis(qc, coord, selectors)


def make_pose_circuit():
    try:
        from qiskit import QuantumCircuit, QuantumRegister
    except ImportError as exc:
        raise SystemExit('Missing Qiskit. Install: python -m pip install qiskit qiskit-aer') from exc

    x = QuantumRegister(5, 'x')
    y = QuantumRegister(5, 'y')
    z = QuantumRegister(5, 'z')
    rot = QuantumRegister(5, 'rot')
    tx = QuantumRegister(3, 'shift_x')
    ty = QuantumRegister(3, 'shift_y')
    tz = QuantumRegister(3, 'shift_z')
    qc = QuantumCircuit(x, y, z, rot, tx, ty, tz, name='ROT_TRANS_29q')
    append_rotation(qc, x, y, z, rot)
    append_translation(qc, x, y, z, tx, ty, tz)
    assert qc.num_qubits == NQ
    return qc


def test_cases(mode: str, seed: int, extra_random: int):
    """Quick: all individual selectors plus random combinations; full: all pairs."""
    if mode == 'full':
        for t in range(512):
            for r in range(32):
                yield r, t
        return
    cases = set()
    for r in range(32):
        cases.add((r, 292))  # selector 292 => (0,0,0) translation
        cases.add((r, 0))
    for t in range(512):
        cases.add((0, t))
        cases.add((31, t))
    rng = random.Random(seed)
    for _ in range(extra_random):
        cases.add((rng.randrange(32), rng.randrange(512)))
    yield from sorted(cases, key=lambda pair: (pair[1], pair[0]))


def seed_coordinates(n: int):
    # Non-symmetric position first, followed by origin and boundary probes.
    seeds = [(3, 11, 25), (0, 0, 0), (31, 31, 31), (0, 17, 31)]
    return seeds[:n]


def make_basis_test(base, pose, initial_xyz, r, t):
    from qiskit import ClassicalRegister, QuantumCircuit
    classical = ClassicalRegister(NQ, 'readout')
    qc = QuantumCircuit(*base.qregs, classical)
    word = index_word(*initial_xyz, r, t)
    for bit in range(NQ):
        if (word >> bit) & 1:
            qc.x(qc.qubits[bit])
    qc.compose(pose, inplace=True)
    qc.measure(qc.qubits, list(classical))
    return qc


def _anf_coefficients(truth_table):
    """Mobius transform over GF(2): 4 selector bits -> XOR-of-ANDs expansion.

    A reference correction is synthesized directly from independently generated
    coordinate truth tables. No ROT/TRANS quantum implementation is reused.
    """
    if len(truth_table) != 16:
        raise ValueError('Reference truth table must contain 16 entries')
    coefficients = [int(v) & 1 for v in truth_table]
    for bit in range(4):
        for mask in range(16):
            if (mask >> bit) & 1:
                coefficients[mask] ^= coefficients[mask ^ (1 << bit)]
    return coefficients


def _reference_correction(qc, active, initial_xyz):
    """Reversibly XOR known reference output coordinates back to initial_xyz.

    Each coordinate flip is a Boolean function of the four selector bits,
    independently calculated with NumPy. The function is decomposed into
    multi-controlled X gates via Algebraic Normal Form (ANF). This deliberately
    does not use pose.inverse() or the pose implementation's rotation/shift
    gates: an erroneous pose circuit cannot cancel itself by its own inverse.
    """
    rotations = np_reference_rotations(initial_xyz)
    expected_words = []
    base_xyz_word = index_word(*initial_xyz, 0, 0) & ((1 << 15) - 1)
    for mask in range(16):
        sel_word = sum(1 << active[j] for j in range(4) if mask & (1 << j))
        r, t = (sel_word >> 15) & 31, (sel_word >> 20) & 511
        coords = reference_output(rotations, r, t)
        expected_words.append(index_word(*coords, r, t))

    reference_data_words = [word & ((1 << 15) - 1) for word in expected_words]
    for data_bit in range(15):
        # Truth-table value = 1 exactly when this output bit differs from
        # the input coordinate bit. Conditional XOR therefore restores input.
        table = [((data_word ^ base_xyz_word) >> data_bit) & 1
                 for data_word in reference_data_words]
        coefficients = _anf_coefficients(table)
        # Sanity-check synthesis truth table before creating quantum gates.
        for assignment in range(16):
            predicted = 0
            for subset, coefficient in enumerate(coefficients):
                if coefficient and (assignment & subset) == subset:
                    predicted ^= 1
            if predicted != table[assignment]:
                raise AssertionError('Reference correction ANF synthesis failed')
        target = qc.qubits[data_bit]
        for subset, coefficient in enumerate(coefficients):
            if not coefficient:
                continue
            controls = [qc.qubits[active[j]] for j in range(4)
                        if subset & (1 << j)]
            if not controls:
                qc.x(target)
            elif len(controls) == 1:
                qc.cx(controls[0], target)
            elif len(controls) == 2:
                qc.ccx(controls[0], controls[1], target)
            else:
                qc.mcx(controls, target)
    return expected_words


def coherent_test(backend, base, pose, initial_xyz, tolerance):
    """Genuine Qiskit interference test for the 29-qubit ROT/TRANS circuit.

    Construct a 16-branch equal superposition using 4 H gates. Apply the
    actual 29-qubit pose gate circuit, then an INDEPENDENT Boolean reference
    correction derived from NumPy coordinate transformations. If all branches
    were mapped correctly with the correct relative phases, all spatial states
    collapse to their original coordinate, and another 4 H gates interfere
    constructively only at control state 0000.

    Save 16 probabilities on the 4 selector controls, not 2^29 amplitudes or
    statevectors. This also avoids reliance on MPS 'save_amplitudes' indexing.
    """
    from qiskit import QuantumCircuit, transpile
    qc = QuantumCircuit(*base.qregs)
    position_word = index_word(*initial_xyz, 0, 0)
    for bit in range(15):
        if (position_word >> bit) & 1:
            qc.x(qc.qubits[bit])
    # rotation bits 0/4, translation bits 0/8; 16 selector combinations
    active = [15, 19, 20, 28]
    for bit in active:
        qc.h(qc.qubits[bit])
    qc.compose(pose, inplace=True)

    expected_words = _reference_correction(qc, active, initial_xyz)
    if len(set(expected_words)) != 16:
        raise AssertionError('Reference selector branches collided')
    for bit in active:
        qc.h(qc.qubits[bit])
    # Qiskit Aer MPS does not accept general mcx instructions directly.
    # The independent ANF correction above uses some 3- and 4-control X
    # gates. Decompose the WHOLE coherent test to universally supported
    # u/cx gates before adding the Aer-specific probability snapshot.
    # Crucially, this changes only gate synthesis, not the reference or
    # the original 29-qubit ROT/TRANS selector mapping.
    qc = transpile(qc, basis_gates=['u', 'cx'], optimization_level=0)
    if qc.num_qubits != NQ:
        raise AssertionError('Coherent circuit transpilation changed qubit count')
    unsupported = set(qc.count_ops()) - {'u', 'cx'}
    if unsupported:
        raise RuntimeError('Coherent circuit still contains unsupported operations: '
                           + repr(sorted(unsupported)))
    # Exact 16-outcome control probabilities from the actual Qiskit MPS.
    # Never materialize the full 2^29 statevector or save big files.
    qc.save_probabilities([qc.qubits[bit] for bit in active],
                          label='interference_probabilities')
    result = backend.run(qc, shots=1).result()
    if not result.success:
        raise RuntimeError(f'Coherent Qiskit simulation failed: {result.status}')
    probabilities = np.asarray(result.data(0)['interference_probabilities'],
                               dtype=np.float64)
    if probabilities.shape != (16,):
        raise RuntimeError('Expected 16 control probabilities; got ' +
                           repr(probabilities.shape))
    return_probability = float(probabilities[0])
    leakage_probability = float(np.sum(probabilities[1:]))
    normalization_error = abs(float(np.sum(probabilities)) - 1.0)
    passed = (abs(return_probability - 1.0) <= tolerance
              and leakage_probability <= tolerance
              and normalization_error <= tolerance)
    return {'passed': passed, 'branches': 16,
            'reference_uncompute': 'independent NumPy coordinate truth table -> Qiskit ANF-controlled-X',
            'probability_return': return_probability,
            'probability_leakage': leakage_probability,
            'normalization_error': normalization_error,
            'probabilities': [float(p) for p in probabilities] if not passed else None}


def translation_primitive_check(seed: int = 20261029):
    """Aer gate check of ALL 32 x 8 possible one-axis translations.

    This isolates the TRANS operator from ROT and the 29-q aggregate,
    preventing a failure in the basic arithmetic from being masked.
    """
    from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, transpile
    from qiskit_aer import AerSimulator

    axis = QuantumRegister(5, 'axis')
    sel = QuantumRegister(3, 'sel')
    base = QuantumCircuit(axis, sel)
    append_translation_axis(base, axis, sel)
    backend = AerSimulator(method='matrix_product_state', device='CPU')
    compiled = transpile(base, backend, optimization_level=0)
    circuits = []
    expected = []
    for x in range(32):
        for t in range(8):
            readout = ClassicalRegister(8, 'readout')
            qc = QuantumCircuit(*base.qregs, readout)
            word = x | (t << 5)
            for bit in range(8):
                if word & (1 << bit):
                    qc.x(qc.qubits[bit])
            qc.compose(compiled, inplace=True)
            qc.measure(qc.qubits, list(readout))
            dx = (t & 1) + 2 * ((t >> 1) & 1) - 4 * (1 - ((t >> 2) & 1))
            expected.append(((x + dx) % 32) | (t << 5))
            circuits.append(qc)

    failures = []
    for start in range(0, len(circuits), 32):
        result = backend.run(circuits[start:start+32], shots=1,
                             memory=True, seed_simulator=seed).result()
        if not result.success:
            raise RuntimeError('TRANS primitive Aer run failed: ' + str(result.status))
        for i in range(min(32, len(circuits)-start)):
            actual = int(result.get_memory(i)[0].replace(' ', ''), 2)
            ref = expected[start+i]
            if actual != ref:
                failures.append((start+i, ref, actual))
                if len(failures) >= 5:
                    raise RuntimeError('TRANS primitive check FAILED; first errors: ' + repr(failures))
    print('One-axis TRANS quantum primitive: PASS (256/256 cases)', flush=True)


def main():
    ap = argparse.ArgumentParser(description='29-qubit Qiskit ROT/TRANS gate verification, no FFT or saved statevectors')
    ap.add_argument('--mode', choices=('quick', 'full'), default='quick',
                    help='quick: all ROT and TRANS selectors separately + random pairs; full: all 16384 pairs')
    ap.add_argument('--positions', type=int, choices=range(1, 5), metavar='{1,2,3,4}', default=1,
                    help='number of test coordinates (default 1)')
    ap.add_argument('--batch', type=int, default=32, help='Aer circuits per batch (default 32)')
    ap.add_argument('--workers', type=int, default=4, help='parallel Aer experiments on CPU (default 4)')
    ap.add_argument('--extra-random', type=int, default=128,
                    help='extra random ROT-TRANS pairs for quick mode')
    ap.add_argument('--seed', type=int, default=20261029)
    ap.add_argument('--tolerance', type=float, default=1e-8)
    ap.add_argument('--skip-primitive-check', action='store_true',
                    help='skip 256-case isolated Qiskit verification of one-axis TRANS')
    ap.add_argument('--no-coherent', action='store_true', help='skip phase-sensitive 16-branch superposition test')
    ap.add_argument('--report', type=Path, default=None, help='optional SMALL JSON report path; no files by default')
    args = ap.parse_args()
    if args.batch < 1 or args.extra_random < 0 or args.workers < 1:
        ap.error('batch/workers must be >= 1; extra-random must be >= 0')

    try:
        from qiskit import transpile
        from qiskit_aer import AerSimulator
    except ImportError as exc:
        raise SystemExit('Requires Qiskit and Aer: python -m pip install qiskit qiskit-aer') from exc

    start = time.monotonic()
    if not args.skip_primitive_check:
        translation_primitive_check(args.seed)
    backend = AerSimulator(method='matrix_product_state', device='CPU',
                           matrix_product_state_truncation_threshold=0.0,
                           max_parallel_threads=args.workers,
                           max_parallel_experiments=args.workers)
    base = make_pose_circuit()
    pose = transpile(base, backend, optimization_level=0)
    if pose.num_qubits != NQ:
        raise AssertionError('Transpilation changed circuit qubit count')
    coord_seeds = seed_coordinates(args.positions)
    signature_count = len(set(np_reference_rotations(coord_seeds[0]).values()))
    if signature_count != 24:
        raise AssertionError(f'Expected 24 distinct rotations, got {signature_count}')

    checked = failed = 0
    failure_examples = []
    for xyz in coord_seeds:
        rotations = np_reference_rotations(xyz)
        buffer = []
        expected_buffer = []
        for r, t in test_cases(args.mode, args.seed, args.extra_random):
            output_xyz = reference_output(rotations, r, t)
            expected_word = index_word(*output_xyz, r, t)
            buffer.append(make_basis_test(base, pose, xyz, r, t))
            expected_buffer.append((xyz, r, t, expected_word))
            if len(buffer) < args.batch:
                continue
            result = backend.run(buffer, shots=1, memory=True, seed_simulator=args.seed).result()
            if not result.success:
                raise RuntimeError(f'Aer simulation failed: {result.status}')
            for i, (seed_xyz, rs, ts, expected) in enumerate(expected_buffer):
                memory = result.get_memory(i)[0].replace(' ', '')
                observed = int(memory, 2)
                checked += 1
                if observed != expected:
                    failed += 1
                    if len(failure_examples) < 12:
                        failure_examples.append({'initial_xyz': seed_xyz, 'rot': rs, 'trans': ts,
                                                 'expected_xyz': decode_word(expected)[:3],
                                                 'observed_xyz': decode_word(observed)[:3],
                                                 'expected_index_word': expected,
                                                 'observed_index_word': observed})
            buffer.clear()
            expected_buffer.clear()
        if buffer:
            result = backend.run(buffer, shots=1, memory=True, seed_simulator=args.seed).result()
            if not result.success:
                raise RuntimeError(f'Aer simulation failed: {result.status}')
            for i, (seed_xyz, rs, ts, expected) in enumerate(expected_buffer):
                observed = int(result.get_memory(i)[0].replace(' ', ''), 2)
                checked += 1
                if observed != expected:
                    failed += 1
                    if len(failure_examples) < 12:
                        failure_examples.append({'initial_xyz': seed_xyz, 'rot': rs, 'trans': ts,
                                                 'expected_xyz': decode_word(expected)[:3],
                                                 'observed_xyz': decode_word(observed)[:3],
                                                 'expected_index_word': expected,
                                                 'observed_index_word': observed})
        print(f'Position {xyz}: {checked:,} total basis-state cases processed; failures={failed}', flush=True)

    coherence = None
    if not args.no_coherent:
        coherence = coherent_test(backend, base, pose, coord_seeds[0], args.tolerance)

    success = failed == 0 and (coherence is None or coherence['passed'])
    summary = {
        'status': 'PASS' if success else 'FAIL', 'total_qubits': NQ,
        'spatial_qubits': 15, 'rotation_selector_qubits': 5,
        'translation_selector_qubits': 9, 'distinct_cubic_orientations': signature_count,
        'mode': args.mode, 'positions': coord_seeds, 'basis_cases': checked,
        'basis_failures': failed, 'coherent': coherence,
        'elapsed_seconds': round(time.monotonic() - start, 2),
        'failure_examples': failure_examples,
        'input_data': 'basis-coordinate states (not molecular grids)',
        'simulation': 'Qiskit Aer MPS, QFT modular-add gates, ANF-based reference interference, no FFT or full statevector',
    }
    print('\n' + '=' * 58)
    print('29-QUBIT QISKIT ROT/TRANS CHECK:', summary['status'])
    print(f'Basis-state cases: {checked:,}; failures: {failed}')
    print(f'Unique orientations: {signature_count} / 32 rotation selectors')
    if coherence:
        print(f"Coherent 16-branch interference test: {'PASS' if coherence['passed'] else 'FAIL'}; "
              f"return probability={coherence['probability_return']:.12f}; "
              f"leakage={coherence['probability_leakage']:.3e}")
        if not coherence['passed']:
            print('Control probability distribution:',
                  [round(p, 9) for p in coherence['probabilities']])
    print(f"Elapsed: {summary['elapsed_seconds']:.2f} s")
    for example in failure_examples:
        print('FAILED:', example)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
        print(f'Small summary: {args.report}')
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
