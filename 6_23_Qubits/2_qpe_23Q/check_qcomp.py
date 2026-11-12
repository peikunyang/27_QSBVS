#!/usr/bin/env python3

import argparse
import math

import numpy as np
from qiskit.quantum_info import Statevector

import qae_qpe


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval-qubits', type=int, default=4)
    parser.add_argument('--thresholds', type=float, nargs='+', default=[0.1, 0.2, 0.3])
    args = parser.parse_args()

    qae_qpe.EVAL_QUBITS = args.eval_qubits
    modulus = 1 << args.eval_qubits
    tested = 0

    for threshold in args.thresholds:
        if not 0.0 < threshold <= 1.0:
            raise ValueError('threshold must satisfy 0 < threshold <= 1')
        low = qae_qpe.threshold_low_limit(threshold)
        high = qae_qpe.threshold_high_limit(low)
        comp = qae_qpe.build_qcomp_block(low)
        if comp.num_qubits < args.eval_qubits + 1:
            raise AssertionError('missing FLAG qubit')
        if any(y < low and y >= high for y in range(modulus)):
            raise AssertionError('QCOMP regions overlap')

        for y in range(modulus):
            expected_mark = int(y < low or y >= high)
            for flag_in in (0, 1):
                input_index = y | (flag_in << args.eval_qubits)
                state = Statevector.from_int(input_index, 1 << comp.num_qubits)
                output = state.evolve(comp)
                probabilities = np.abs(output.data) ** 2
                expected_index = y | ((flag_in ^ expected_mark) << args.eval_qubits)
                if abs(probabilities[expected_index] - 1.0) > 1e-10:
                    actual_index = int(np.argmax(probabilities))
                    raise AssertionError(
                        f'threshold={threshold} bin={y} FLAG={flag_in}: '
                        f'expected basis={expected_index}, actual basis={actual_index}, '
                        f'expected probability={probabilities[expected_index]:.12g}'
                    )
                tested += 1

        print(
            f'PASS  threshold={threshold:.7g}  eval_qubits={args.eval_qubits}  '
            f'low=[0,{low - 1}]  high=[{high},{modulus - 1}]  '
            f'comparator_qubits={comp.num_qubits}'
        )
    print(f'OVERALL PASS  basis-state tests={tested}  clean ancillas and FLAG XOR verified')


if __name__ == '__main__':
    main()
