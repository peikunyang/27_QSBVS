# Quantum Amplitude Estimation for Ligand and Pose Index Retrieval

Research code accompanying **“Quantum Amplitude Estimation for Ligand and Pose Index Retrieval”** by **Pei-Kun Yang**.

This repository investigates an **index-first strategy for structure-based virtual screening (SBVS)**. Instead of measuring and reconstructing an interaction score for every receptor–ligand–pose configuration, the framework uses quantum amplitude estimation based on quantum phase estimation (**QAE/QPE**) and a reversible comparator (**QCOMP**) to mark configurations below a screening threshold. Measuring the `FLAG` and selector registers retrieves a candidate ligand and pose index. Optional Grover-style quantum selection (**QSEL**) amplifies the marked subspace.

> **Scope:** The large-library experiments are numerical benchmarks based on FFT-computed grid overlaps and an *ideal, noiseless analytical* QAE/QPE–QCOMP–QSEL model. Separate Qiskit simulations validate reduced-scale circuitry and pose transformations.

## Research overview

For each indexed configuration, the method combines:

- **Receptor state:** two receptor conformations per protein system.
- **Ligand occupancy:** nested libraries containing 1 to 1,024 ligand occupancies.
- **Rotation:** 32 selector states representing 24 distinct cubic orientations.
- **Translation:** 512 cyclic translations on a Cartesian lattice.

Receptor interaction maps and ligand charge/occupancy grids are represented on a **32 × 32 × 32** spatial lattice with eight channels (`e`, `A`, `C`, `HD`, `N`, `NA`, `OA`, `S`). The physical representation uses 18 qubits; an auxiliary qubit extends the encoded data register to **19 qubits**.

The logical workflow is:

```text
Receptor / ligand inputs
        |
        v
AutoDock4 interaction maps + ligand occupancy grids
        |
        v
Normalized, selector-indexed receptor / ligand states
        |
        v
SELECT-OCC -> ROT -> TRANS -> SELECT-MAP
        |
        v
Interaction amplitude A0
        |
        v
QAE/QPE -> QCOMP -> FLAG
                     |
                     +--> optional QSEL amplification
                     |
                     v
       Measure FLAG and selector registers
                     |
                     v
       Candidate ligand / receptor / pose index
```

The method targets **candidate index retrieval**, not reconstruction of all numerical interaction scores. The AutoDock grid terms provide a controlled proof-of-concept interaction model.

## Repository structure

```text
2026_09_23_github/
├── 1_data/          # PDBbind ligand preparation, geometric filtering, selections
├── 2_MD/            # CHARMM-GUI/OpenMM MD scripts and PDB/PDBQT processing
├── 3_decoy/         # Distractor ligand selection, centering, and grid alignment
├── 4_Autodock/      # AutoGrid4 maps, ligand charge/occupancy grids, grid energies
├── 5_Ene/
│   ├── 1_data/
│   │   ├── 2_norm/   # MAP/OCC normalization and dataset assembly (comb.py)
│   │   └── 3_ene/    # Independent overlap/energy calculations (ene.py)
│   ├── 2_QAE/        # Main analytical QAE/QPE + QCOMP + QSEL sweep (energy.py)
│   ├── 3_QAE_verify/ # Component checks, overlap comparisons, energy validation
│   └── 4_rot_trans_verify/ # 29-qubit Qiskit ROT/TRANS validation
├── 6_23_Qubits/     # Synthetic inputs and reduced-scale Qiskit circuit validation
└── 7_paper/         # Figure/table
```

## Requirements

**Core numerical and circuit analysis**

- Python 3
- NumPy, PyTorch
- Qiskit and Qiskit Aer for gate-level validation
- Optional CUDA-capable PyTorch

**Molecular preparation / grid generation**, when rebuilding the input data:

- PDBbind molecular structures (subject to the dataset's access conditions)
- CHARMM-GUI and OpenMM, with the relevant CHARMM force-field inputs
- RDKit and Meeko, where used for ligand preparation
- AutoDockTools/MGLTools, AutoGrid4, and the applicable AutoDock4 parameter files

## Independent validation

The repository separates analytical large-library calculations from several numerical or gate-level checks:

- **23-qubit reduced-scale circuit:** `6_23_Qubits/2_qpe_23Q/qae_qpe.py` tests the integrated logical operations on synthetic inputs.
- **29-qubit full-resolution pose circuit:** `5_Ene/4_rot_trans_verify/verify_rot_trans_29q.py` checks grid-coordinate ROT/TRANS operations with Qiskit Aer and reference transformations. This is a *pose subcircuit*, not a full 29-qubit molecular QAE implementation.
- **Component and numerical checks:** `5_Ene/3_QAE_verify/` contains independent overlap, QAE/QPE, QSEL, and energy-validation scripts.

## Manuscript and citation

**Pei-Kun Yang.** *Quantum Amplitude Estimation for Ligand and Pose Index Retrieval.* Manuscript and accompanying Supporting Information.

Please cite the associated publication when a definitive bibliographic reference becomes available.

## Contact

Pei-Kun Yang  
`peikun@isu.edu.tw`  
`peikun6416@gmail.com`
