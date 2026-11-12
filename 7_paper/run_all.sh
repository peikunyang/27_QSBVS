#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo 'Usage: ./run_all.sh /ABS/PATH/2_QAE/out [SUPPLEMENT_WORD_MASTER.docx]' >&2
    exit 2
fi
DATA_ROOT="$(realpath "$1")"
if [[ ! -f "$DATA_ROOT/batch_summary_pose.tsv" ]]; then
    echo "Missing $DATA_ROOT/batch_summary_pose.tsv" >&2
    exit 2
fi
printf 'Main Figure 1–5:\n'
"$ROOT/main/Fig1/run.sh"
for n in 2 3 4 5; do
    "$ROOT/main/Fig$n/run.sh" "$DATA_ROOT"
done
printf '\nSupplementary Figures S1–S4:\n'
for n in 1 2 3 4; do
    "$ROOT/supplement/FigS$n/run.sh" "$DATA_ROOT"
done
printf '\nSupplementary Tables S1–S2:\n'
"$ROOT/supplement/TableS1/run.sh"
"$ROOT/supplement/TableS2/run.sh"
printf '\nSupplementary native Word tables:\n'
if [[ $# -eq 2 ]]; then
    "${PYTHON_BIN:-python3}" "$ROOT/supplement/build_tables_word.py" --supplement-docx "$2"
else
    "${PYTHON_BIN:-python3}" "$ROOT/supplement/build_tables_word.py"
fi
printf '\nAll outputs generated under main/ and supplement/.\n'
