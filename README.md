# Quantum Amplitude Estimation for Ligand and Pose Index Retrieval

Research code accompanying **“Quantum Amplitude Estimation for Ligand and Pose Index Retrieval”** by **Pei-Kun Yang**.

This repository investigates an **index-first strategy for structure-based virtual screening (SBVS)**. Instead of measuring and reconstructing an interaction score for every receptor–ligand–pose configuration, the framework uses quantum amplitude estimation based on quantum phase estimation (**QAE/QPE**) and a reversible comparator (**QCOMP**) to mark configurations below a screening threshold. Measuring the `FLAG` and selector registers retrieves a candidate ligand and pose index. Optional Grover-style quantum selection (**QSEL**) amplifies the marked subspace.

> **Scope:** The large-library experiments are numerical benchmarks based on FFT-computed grid overlaps and an *ideal, noiseless analytical* QAE/QPE–QCOMP–QSEL model. Separate Qiskit simulations validate reduced-scale circuitry and pose transformations. The repository does **not** demonstrate a fully fault-tolerant, gate-decomposed, end-to-end quantum screening implementation or a quantum-hardware speedup.

## Research overview

For each indexed configuration, the method combines:

- **Receptor state:** two receptor conformations per protein system.
- **Ligand occupancy:** nested libraries containing 1 to 1,024 ligand occupancies.
- **Rotation:** 32 selector states representing 24 distinct cubic orientations.
- **Translation:** 512 cyclic translations on a Cartesian lattice.

The resulting search space contains

\[
N_{\mathrm{search}}=2\times N_{\mathrm{OCC}}\times32\times512
=32{,}768\,N_{\mathrm{OCC}},
\]

or **32,768 to 33,554,432 indexed configurations** per receptor system.

Receptor interaction maps and ligand charge/occupancy grids are represented on a **32 × 32 × 32** spatial lattice with eight channels (`e`, `A`, `C`, `HD`, `N`, `NA`, `OA`, `S`). The physical representation uses 18 qubits; an auxiliary qubit extends the encoded data register to **19 qubits**. A normalized interaction amplitude is defined as

\[
A_0=\frac{E+E_+}{L_U L_N},
\]

where `E` is the selected grid-interaction score and `E_+`, `L_U`, and `L_N` are the energy offset and normalization factors. Configurations qualify when `A₀ < A₀,thr`.

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

The method targets **candidate index retrieval**, not reconstruction of all numerical interaction scores. The AutoDock grid terms provide a controlled proof-of-concept interaction model; they are **not** rigorous binding free energies or the full AutoDock4 scoring function.

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
└── 7_paper/         # Figure/table helper code and orchestration scripts
```

Most `exe` files are shell commands or small launchers for individual pipeline stages. They generally assume the **working directory is the directory containing that launcher**; some also explicitly request CUDA.

## Requirements

**Core numerical and circuit analysis**

- Python 3 (package versions are not pinned in the supplied archive)
- NumPy, PyTorch
- Qiskit and Qiskit Aer for gate-level validation
- Matplotlib, pandas, mpmath, NetworkX for supporting analyses and plotting
- Optional CUDA-capable PyTorch for GPU FFT processing

**Molecular preparation / grid generation**, when rebuilding the input data:

- PDBbind molecular structures (subject to the dataset's access conditions)
- CHARMM-GUI and OpenMM, with the relevant CHARMM force-field inputs
- RDKit and Meeko, where used for ligand preparation
- AutoDockTools/MGLTools, AutoGrid4, and the applicable AutoDock4 parameter files

A possible starting point for a fresh Python environment is:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy torch qiskit qiskit-aer matplotlib pandas mpmath networkx
```

The installation command covers common Python dependencies for the numerical and Qiskit sections, **not** the complete molecular-preparation toolchain. Adapt it to your hardware and installed Qiskit version. No environment lockfile or package-version compatibility matrix is included in this archive.

## Quick start: synthetic Qiskit validation

The synthetic input generator in `6_23_Qubits/` can be used without PDBbind or AutoGrid outputs. From the repository root:

```bash
mkdir -p demo_output

python 6_23_Qubits/1_gen_map_occ/gen_map_occ.py \
  --grid-size 4 \
  --threshold 0.02 \
  --output demo_output/map_occ.txt

python 6_23_Qubits/2_qpe_23Q/qae_qpe.py \
  --grid-size 4 \
  --eval-qubits 4 \
  --input-file demo_output/map_occ.txt \
  --threshold 0.02 \
  --qsel-iterations 1 \
  --device cpu \
  --circuit-file demo_output/circuit.txt \
  --output-file demo_output/validation.txt
```

The generator produces normalized synthetic `MAP` and `OCC` inputs with a specified threshold. The validator builds a reduced-scale circuit and compares circuit-level results with classical references, including overlap, QAE/QPE, comparator, and QSEL behavior. **This example may require substantial RAM and computation; it has not been run as part of preparing this README.**

## Main numerical benchmark

The principal benchmark program is `5_Ene/2_QAE/energy.py`. It expects **previously generated, normalized** receptor directories, each containing `map/` and `occ/`, including the normalization metadata and binary arrays created by the earlier preprocessing stages.

From the repository root, **after preparing those inputs**, the manuscript's pose-retrieval sweep can be invoked as:

```bash
python 5_Ene/2_QAE/energy.py \
  5_Ene/1_data/2_norm \
  5_Ene/2_QAE/out \
  --eval-qubits 10 11 12 13 14 15 16 17 18 19 20 21 22 \
  --resolution-multiples 1 2 4 8 \
  --retrieval-mode pose \
  --grover-max auto \
  --max-occ 1024 \
  --device auto \
  --skip-circuit
```

This is a CPU/CUDA-adaptive version of the parameters in `5_Ene/2_QAE/exe`, which uses `--device cuda`. The sweep uses

\[
A_{0,\mathrm{thr}}=\sin\left(\frac{\pi}{2^{m_0}}\right),
\qquad m=m_0+\log_2(k), \quad k\in\{1,2,4,8\}.
\]

Increasing `k` increases evaluation-register resolution **while keeping the screening threshold fixed** for a given `m₀`.

Useful options:

| Option | Purpose |
| --- | --- |
| `--retrieval-mode pose` | Write configuration/pose-retrieval results used in the manuscript. |
| `--retrieval-mode ligand` | Evaluate ligand-only retrieval (also supported by the code). |
| `--retrieval-mode both` | Produce both sets of results. |
| `--max-occ N` | Set the largest nested occupancy library (`1` to `1024`, powers of two). |
| `--eval-qubits ...` | Set base evaluation-register sizes `m₀`. |
| `--resolution-multiples ...` | Set resolution factors `k` (`1`, `2`, `4`, `8`). |
| `--grover-max auto` | Scan the QSEL iteration range selected by the program. |
| `--device auto` | Choose CUDA if available, otherwise CPU. |
| `--skip-circuit` | Omit circuit SVG generation. |

The pose run writes **`batch_summary_pose.tsv`** under the supplied output directory. Ligand-only or combined runs also write **`batch_summary_ligand.tsv`**. Other experiment-specific results are placed in subdirectories under the output root.

### Normalized input preparation

The normalization entry point is `5_Ene/1_data/2_norm/comb.py`:

```bash
python 5_Ene/1_data/2_norm/comb.py \
  <MAP_ROOT> <OCC_PARENT> <NORMALIZED_OUTPUT_ROOT> --device auto
```

Here `<MAP_ROOT>` must contain receptor-state map folders named like `<PDBID>_0` and `<PDBID>_1`; `<OCC_PARENT>` must contain `occ/` and `occ_decoy/`. This step creates per-receptor normalized input groups for the QAE benchmark. The upstream AutoGrid4 mapping and occupancy-generation scripts are in `4_Autodock/`.

**Important:** The supplied repository archive contains the preparation and analysis scripts but **does not include a complete set of precomputed normalized MAP/OCC inputs**. The main benchmark command therefore cannot run directly on a fresh extraction without preparing or supplying the missing inputs.

## Independent validation

The repository separates analytical large-library calculations from several numerical or gate-level checks:

- **23-qubit reduced-scale circuit:** `6_23_Qubits/2_qpe_23Q/qae_qpe.py` tests the integrated logical operations on synthetic inputs.
- **29-qubit full-resolution pose circuit:** `5_Ene/4_rot_trans_verify/verify_rot_trans_29q.py` checks grid-coordinate ROT/TRANS operations with Qiskit Aer and reference transformations. This is a *pose subcircuit*, not a full 29-qubit molecular QAE implementation.
- **Component and numerical checks:** `5_Ene/3_QAE_verify/` contains independent overlap, QAE/QPE, QSEL, and energy-validation scripts.

For a reduced workload on the standalone 29-qubit pose checker:

```bash
python 5_Ene/4_rot_trans_verify/verify_rot_trans_29q.py \
  --mode quick --positions 1 --workers 4
```

The included `exe` launcher selects `--mode full --positions 4 --workers 4`, which is more computationally demanding.

## Quantities reported

Let `G` denote the event that a configuration truly meets the amplitude threshold. The benchmark reports:

- **Initial acceptance:** `p = Pr(FLAG = 1 | r = 0)`.
- **Classification precision:** `η = Pr(G | FLAG = 1)`, which captures finite-QPE false positives at the benchmark level.
- **Acceptance after QSEL:** `pᵣ = sin²((2r + 1) arcsin(√p))` in the ideal amplification model.
- **95% FLAG-detection shots:** `S₉₅(r) = ceil[ln(0.05) / ln(1 − pᵣ)]` for `0 < pᵣ < 1`.
- **Logical Q-operator calls:** `C₉₅(r) = S₉₅(r) (2r + 1) (2^m − 1)`.

`S₉₅` measures the shots needed to observe **at least one `FLAG = 1`**, not the probability of obtaining a true qualifying configuration. Precision `η` is tracked separately. The logical call count is **not** an elementary gate count, circuit depth, or execution-time estimate. QSEL can reduce the number of shots while increasing the coherent work per shot.

## Benchmark systems and modeling assumptions

The manuscript examines eight PDB receptor systems:

`1ecq`, `1h0r`, `2aa9`, `2q9y`, `2ybk`, `2ybs`, `4gui`, `6o4x`.

The reported interaction model uses AutoGrid4 electrostatics and seven atom-type affinity channels on a 0.375 Å lattice. It omits the separate charge-dependent desolvation map and non-grid torsional terms, and therefore should **not** be interpreted as a complete AutoDock4 docking score. The two receptor states were generated using short molecular dynamics trajectories as contrasting structures, not as converged equilibrium ensembles.

## Release and reproducibility notes

This archive is **a research-code snapshot, not a fully self-contained reproduction package**:

1. It includes selected example molecular-preparation files (notably for `1ecq`), but not the full original PDBbind/CHARMM-GUI inputs and processed data for all eight receptors.
2. Intermediate normalized maps, ligand occupancies, and completed large-library benchmark outputs are not bundled. The data-generation workflow must be completed before running the main sweep.
3. `7_paper/run_all.sh` and `7_paper/exe` reference `7_paper/main/`, `7_paper/supplement/`, and `supplement/build_tables_word.py`, which are **not present** in the supplied archive. Those paper-generation launchers are therefore **not runnable as provided**.
4. Some stage-specific shell launchers reference paths and external executables that must be configured in a local environment. Python package versions and input data checksums are not specified.
5. The large-library resource study assumes ideal coherent operators; no claim of a practical end-to-end runtime advantage is made.

These limitations are stated so the included scripts are not mistaken for an independently executable, complete publication-reproduction bundle.

## Manuscript and citation

**Pei-Kun Yang.** *Quantum Amplitude Estimation for Ligand and Pose Index Retrieval.* Manuscript and accompanying Supporting Information.

Please cite the associated publication when a definitive bibliographic reference becomes available. No DOI or publication URL is supplied here to avoid implying that a particular published version has been verified.

## Contact

Pei-Kun Yang  
`peikun@isu.edu.tw`  
`peikun6416@gmail.com`

## License

**No `LICENSE` file is included in the provided code archive.** This README does not assign an open-source license or grant additional permission to reuse or redistribute the source. Please consult the repository owner regarding licensing.
