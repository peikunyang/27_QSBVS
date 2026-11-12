#!/usr/bin/env python3
"""Compare real molecular pose-index validation with the *existing* production output.

Usage:
  python check_pose_figures.py ../6_chain_chk/out2/real_chain_check.txt ../../2_QAE/out --output out/pose_figure_check.txt

Only POSE rows are used. It never recomputes the molecular data, changes energy.py,
rewrites production TSV files, or claims that rendered figures were compared.
"""

import argparse
import csv
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

KEY_COLS = ("pdbid", "m0", "k", "N")
FLOAT_FIELDS = {
    "q_exact": "EXACT_GOOD_FRACTION",
    "p_flag": "QAE_FLAG_PROB",
    "eta": "TRUE_GOOD_GIVEN_FLAG",
    "good0": "QAE_TRUE_GOOD_PROB",
}
INTEGER_FIELDS = {
    "r": "BEST_GROVER_ITER",
    "shots0": "QAE_SHOTS95",
    "shotsR": "QSEL_SHOTS95",
    "callsR": "QSEL_TOTAL_Q_CALLS95",
}
SCALING_INTEGER_FIELDS = {
    "r": "BEST_GROVER_ITER",
    "shots0": "QAE_SHOTS95",
    "shotsR": "BEST_QSEL_SHOTS95",
    "callsR": "BEST_QSEL_TOTAL_Q_CALLS95",
}


def parse_validation(path):
    records = {}
    status = None
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.startswith("Result "):
            status = line.split(":", 1)[1].strip()
        chunks = line.split()
        if len(chunks) != 16 or chunks[0] not in ("PASS", "FAIL") or chunks[6] != "POSE":
            continue
        _, pdbid, m0, k, m, n, _, q, p, eta, good0, goodr, r, shots0, shotsr, callsr = chunks
        key = (pdbid.lower(), int(m0), int(k), int(n))
        if key in records:
            raise ValueError(f"Duplicate POSE row {key} at line {line_number}")
        if int(m) != int(m0) + (int(k).bit_length() - 1):
            raise ValueError(f"Inconsistent QPE-register size for {key}")
        records[key] = dict(q_exact=q, p_flag=p, eta=eta, good0=good0,
                            goodR=goodr, r=r, shots0=shots0, shotsR=shotsr,
                            callsR=callsr, source_status=chunks[0], line=line_number)
    if not records:
        raise ValueError(f"No POSE records found in {path}")
    return records, status


def read_tsv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t", skipinitialspace=True)
        if not reader.fieldnames:
            raise ValueError(f"Empty TSV: {path}")
        reader.fieldnames = [name.strip() for name in reader.fieldnames]
        if len(reader.fieldnames) == 1 and len(reader.fieldnames[0].split()) > 1:
            # Some exports are whitespace-separated despite having a .tsv suffix.
            handle.seek(0)
            raw_lines = [line.strip().split() for line in handle if line.strip()]
            fields = raw_lines[0]
            if any(len(line) != len(fields) for line in raw_lines[1:]):
                raise ValueError(f"Inconsistent whitespace-delimited columns: {path}")
            return [dict(zip(fields, values)) for values in raw_lines[1:]]
        records = []
        for number, raw in enumerate(reader, 2):
            if None in raw:
                raise ValueError(f"Extra TSV cells in {path}, line {number}: {raw[None]}")
            records.append({k: v.strip() if v is not None else v for k, v in raw.items()})
        return records


def key_from_row(row, pdbid=None):
    k = int(row["RESOLUTION_MULTIPLE_TARGET"])
    if k not in (1, 2, 4, 8):
        raise ValueError(f"Unexpected k={k}")
    m = int(row["EVAL_QUBITS"])
    m0 = m - int(math.log2(k))
    return ((pdbid or row["PDBID"]).lower(), m0, k, int(row["OCC_COUNT"]))


def formatted_number_agrees(report_token, prod_token):
    """Compare floats at the precision *printed* in the validation report.

    The validation text uses fewer significant digits than the production TSV.
    Allow 0.51 of a unit in the final displayed decimal digit, plus a tiny
    floating point guard; this avoids a false mismatch from report formatting.
    """
    try:
        expected = float(report_token)
        actual = float(prod_token)
    except (ValueError, TypeError):
        return False, float("inf"), 0
    if not (math.isfinite(expected) and math.isfinite(actual)):
        return False, float("inf"), 0
    match = re.fullmatch(r"[-+]?\d+(?:\.(\d+))?[eE]([-+]?\d+)", report_token)
    if match:
        decimal_places = len(match.group(1) or "")
        scale = 10.0 ** (int(match.group(2)) - decimal_places)
        tolerance = max(0.51 * scale, 5e-16)
    else:
        tolerance = 5e-12
    difference = abs(actual - expected)
    return difference <= tolerance, difference, tolerance



def read_pose_summary_probabilities(path):
    """Read the high-precision p and eta written by production for one OCC size."""
    if not path.is_file():
        return None
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        pieces = line.split(maxsplit=1)
        if len(pieces) == 2 and pieces[0] in ("FLAG_PROBABILITY", "TRUE_GOOD_GIVEN_FLAG"):
            values[pieces[0]] = float(pieces[1])
    if len(values) != 2:
        raise ValueError(f"Missing FLAG_PROBABILITY / TRUE_GOOD_GIVEN_FLAG in {path}")
    return values["FLAG_PROBABILITY"], values["TRUE_GOOD_GIVEN_FLAG"]


def independent_shots(probability, confidence):
    """Binomial repeated-trial target; independent of energy.py."""
    if probability <= 0:
        return None
    if probability >= 1:
        return 1
    return max(1, math.ceil(math.log(1.0 - confidence) / math.log1p(-probability)))


def fmt_integer(value):
    return "INF" if value is None else str(value)


def curve_probability_agrees(reference, observed, r):
    """Accept TSV rounding and the accumulated roundoff of a 2D rotation."""
    if not (math.isfinite(reference) and math.isfinite(observed)):
        return False, math.inf, 0.0
    delta = abs(reference - observed)
    tolerance = 3e-12 + 3e-10 * abs(reference) + 1.5e-14 * (r + 1)
    return delta <= tolerance, delta, tolerance


def audit_grover_curve(key, grover_rows, scaling_row, batch_row, pose_dir,
                       issues, disagreements, counts, precision_notes):
    """Check *every* saved r by 2D Grover amplitude rotation, not energy.py calls.

    The input p, eta are independently verified for these (PDBID, m0, k, N)
    by 6_chain_chk and batch/scaling checks. The per-OCC production summary
    stores 16-digit probabilities and avoids rounding-induced shot differences.
    """
    pdbid, m0, k, n = key
    m = m0 + (k.bit_length() - 1)
    qpe_cost = (1 << m) - 1
    expected_rmax = int(scaling_row["GROVER_MAX"])
    if len(grover_rows) != expected_rmax + 1:
        issues.append(f"Grover curve length {key}: got {len(grover_rows)}, expected {expected_rmax + 1}")
        return

    summary_path = pose_dir / f"occ_{n:03d}" / "summary.txt"
    high_precision = read_pose_summary_probabilities(summary_path)
    if high_precision is None:
        # Fallback still works but 12-digit production summary values may cause
        # a borderline integer-shot rounding difference.
        p = float(batch_row["QAE_FLAG_PROB"])
        eta = float(batch_row["TRUE_GOOD_GIVEN_FLAG"])
        counts["curves_using_rounded_probability"] += 1
    else:
        p, eta = high_precision
        for name, expected, printed in (
            ("p0", p, batch_row["QAE_FLAG_PROB"]),
            ("eta", eta, batch_row["TRUE_GOOD_GIVEN_FLAG"]),
        ):
            agrees, delta, tolerance = formatted_number_agrees(printed, f"{expected:.17e}")
            if not agrees:
                disagreements.append((key, "summary_" + name, printed, expected, delta, tolerance))
    if not (math.isfinite(p) and math.isfinite(eta) and 0 <= p <= 1 and 0 <= eta <= 1):
        issues.append(f"Invalid p/eta for {key}: p={p}, eta={eta}")
        return

    # Grover = reflection in FLAG subspace, followed by reflection about
    # the initial state. Restricted to 2D it is a rotation by 2*theta.
    # Update the (FLAG, non-FLAG) amplitude vector iteratively. This is
    # independent of the sin^2((2r+1)*theta) routine used by production.
    s = math.sqrt(p)
    c = math.sqrt(1.0 - p)
    cosine_twice = 1.0 - 2.0 * p
    sine_twice = 2.0 * s * c
    counts["grover_curve_groups"] += 1

    for r, point in enumerate(grover_rows):
        try:
            observed_r = int(point["GROVER_ITERATIONS"])
        except (KeyError, TypeError, ValueError) as exc:
            issues.append(f"Malformed Grover r in {key}: {exc}")
            return
        if observed_r != r:
            issues.append(f"Missing, extra, or nonsequential Grover r in {key}: position {r}, observed {observed_r}")
            return

        flag_probability = max(0.0, min(1.0, s * s))
        good_probability = flag_probability * eta
        for col, expected in (("AMPLIFIED_FLAG_PROB", flag_probability),
                              ("TRUE_GOOD_PROB", good_probability)):
            actual = float(point[col])
            agree, delta, tolerance = curve_probability_agrees(expected, actual, r)
            counts["grover_curve_probability_checks"] += 1
            counts["grover_curve_max_prob_delta"] = max(
                counts["grover_curve_max_prob_delta"], delta
            )
            if not agree:
                # A direct-angle formula may agree when successive 2D rotations
                # accumulate IEEE754 phase error (especially if p is near 1).
                direct_theta = math.asin(math.sqrt(p))
                direct_flag = math.sin((2 * r + 1) * direct_theta) ** 2
                direct_probability = direct_flag if col == "AMPLIFIED_FLAG_PROB" else direct_flag * eta
                direct_agree, _, _ = curve_probability_agrees(direct_probability, actual, r)
                if direct_agree:
                    precision_notes.append(f"{key} r={r} {col}: 2D-roundoff, direct-angle agrees")
                else:
                    disagreements.append((key + (r,), col, f"{expected:.17e}", point[col], delta, tolerance))

        per_shot_calls = (2 * r + 1) * qpe_cost
        counts["grover_curve_cost_checks"] += 1
        if int(point["Q_CALLS_PER_SHOT"]) != per_shot_calls:
            disagreements.append((key + (r,), "Q_CALLS_PER_SHOT", per_shot_calls, point["Q_CALLS_PER_SHOT"], None, 0))
        for confidence, shot_field, call_field in (
            (0.95, "SHOTS95", "TOTAL_Q_CALLS95"),
            (0.99, "SHOTS99", "TOTAL_Q_CALLS99"),
        ):
            expected_shots = independent_shots(good_probability, confidence)
            actual_shots = point[shot_field].upper()
            expected_cost = None if expected_shots is None else expected_shots * per_shot_calls
            counts["grover_curve_shot_checks"] += 1
            counts["grover_curve_total_call_checks"] += 1
            if actual_shots != fmt_integer(expected_shots):
                # A 2D iterative rotation and the production direct-angle
                # calculation can straddle an exact integer-shot boundary by
                # sub-ULP error. Re-evaluate using a noniterative angle.
                theta = math.atan2(math.sqrt(p), math.sqrt(1.0 - p))
                direct_good = math.sin((2 * r + 1) * theta) ** 2 * eta
                direct_shots = independent_shots(direct_good, confidence)
                if (fmt_integer(direct_shots) == actual_shots and
                        abs(direct_good - good_probability) <= 2e-10):
                    precision_notes.append(f"{key} r={r} {shot_field}: iterative={fmt_integer(expected_shots)}, direct={actual_shots}")
                    expected_shots = direct_shots
                    expected_cost = None if expected_shots is None else expected_shots * per_shot_calls
                else:
                    disagreements.append((key + (r,), shot_field, fmt_integer(expected_shots), actual_shots, None, 0))
            if point[call_field].upper() != fmt_integer(expected_cost):
                disagreements.append((key + (r,), call_field, fmt_integer(expected_cost), point[call_field], None, 0))

        counts["grover_curve_points"] += 1
        s, c = (cosine_twice * s + sine_twice * c,
                cosine_twice * c - sine_twice * s)


def compare(report, batch_path=None, experiment_root=None):
    rows, status = parse_validation(report)
    issues = []
    notes = []
    counts = Counter()
    disagreements = []
    precision_notes = []
    if status is None or not status.startswith("PASS"):
        issues.append(f"Validation report summary is not PASS: {status}")
    for key, item in rows.items():
        if item["source_status"] != "PASS":
            issues.append(f"Validation test not PASS: {key}")
        m = key[1] + int(math.log2(key[2]))
        q = (1 << m) - 1
        shots = None if item["shotsR"] in ("INF", "-") else int(item["shotsR"])
        calls = None if item["callsR"] in ("INF", "-") else int(item["callsR"])
        expected_calls = None if shots is None else shots * (2 * int(item["r"]) + 1) * q
        counts["validation_pose_rows"] += 1
        if expected_calls != calls:
            issues.append(f"Validation report Q-call accounting: {key}: {calls} != {expected_calls}")
        else:
            counts["validation_cost_checks"] += 1

    batch_rows = None
    if batch_path is not None and batch_path.is_file():
        batch_rows = {}
        for raw in read_tsv(batch_path):
            key = key_from_row(raw)
            if key in batch_rows:
                issues.append(f"Duplicate production row {key}")
            batch_rows[key] = raw
        for key, item in rows.items():
            prod = batch_rows.get(key)
            if prod is None:
                issues.append(f"Missing in production batch: {key}")
                continue
            for source_field, production_field in FLOAT_FIELDS.items():
                match, delta, tolerance = formatted_number_agrees(item[source_field], prod[production_field])
                counts["numeric_field_checks"] += 1
                if not match:
                    disagreements.append((key, source_field, item[source_field], prod[production_field], delta, tolerance))
            for source_field, production_field in INTEGER_FIELDS.items():
                counts["integer_field_checks"] += 1
                if int(item[source_field]) != int(prod[production_field]):
                    disagreements.append((key, source_field, item[source_field], prod[production_field], None, 0))
            m = key[1] + int(math.log2(key[2]))
            if int(prod["Q_CALLS_PER_QPE"]) != (1 << m) - 1:
                issues.append(f"Production QPE Q-call count invalid: {key}")
            if int(prod["QSEL_Q_CALLS_PER_SHOT"]) != (2 * int(prod["BEST_GROVER_ITER"]) + 1) * ((1 << m) - 1):
                issues.append(f"Production per-shot Q-call count invalid: {key}")
            counts["matching_batch_rows"] += 1
        extras = set(batch_rows) - set(rows)
        if extras:
            extra_m0 = sorted({key[1] for key in extras})
            notes.append(f"Production contains {len(extras)} extra parameter configurations not in the validation report")
            notes.append(f"Additional production m0 values requiring validation: {' '.join(map(str, extra_m0))}")
        counts["production_batch_rows"] = len(batch_rows)
    else:
        notes.append("Production batch_summary_pose.tsv was NOT supplied; no production-to-validation comparison has been performed")

    if experiment_root is not None and experiment_root.is_dir() and batch_rows is not None:
        scaling_paths = sorted(experiment_root.rglob("scaling_summary.tsv"))
        seen = set()
        for file in scaling_paths:
            if file.parent.name != "pose":
                continue
            pdbid = file.parents[2].name
            for prod in read_tsv(file):
                key = key_from_row(prod, pdbid=pdbid)
                if key not in rows:
                    continue
                seen.add(key)
                reference = rows[key]
                match, delta, tolerance = formatted_number_agrees(reference["goodR"], prod["BEST_QSEL_TRUE_GOOD_PROB"])
                counts["scaling_goodR_checks"] += 1
                if not match:
                    disagreements.append((key, "goodR", reference["goodR"], prod["BEST_QSEL_TRUE_GOOD_PROB"], delta, tolerance))
                for source_field, production_field in SCALING_INTEGER_FIELDS.items():
                    counts["scaling_integer_checks"] += 1
                    if int(reference[source_field]) != int(prod[production_field]):
                        disagreements.append((key, source_field, reference[source_field], prod[production_field], None, 0))
            grover_path = file.with_name("grover_shots.tsv")
            if grover_path.is_file():
                groups = defaultdict(list)
                for row in read_tsv(grover_path):
                    groups[int(row["OCC_COUNT"])].append(row)
                sample = read_tsv(file)
                if sample:
                    k_sample = key_from_row(sample[0], pdbid=pdbid)
                    for n, iterations in groups.items():
                        key = (k_sample[0], k_sample[1], k_sample[2], n)
                        if key not in rows:
                            continue
                        audit_grover_curve(
                            key, iterations, next((row for row in sample if int(row["OCC_COUNT"]) == n), sample[0]),
                            batch_rows[key], file.parent,
                            issues, disagreements, counts, precision_notes,
                        )
                        chosen_r = int(rows[key]["r"])
                        actual = [item for item in iterations if int(item["GROVER_ITERATIONS"]) == chosen_r]
                        if len(actual) != 1:
                            issues.append(f"Missing/duplicate chosen Grover iteration {key} r={chosen_r}")
                            continue
                        row = actual[0]
                        counts["grover_selected_checks"] += 1
                        for source, field in (("shotsR", "SHOTS95"), ("callsR", "TOTAL_Q_CALLS95")):
                            if int(rows[key][source]) != int(row[field]):
                                disagreements.append((key, "grover_" + source, rows[key][source], row[field], None, 0))
                        if "_shot_" not in str(file.parent.parent.name):
                            feasible = [it for it in iterations if it["TOTAL_Q_CALLS95"] != "-"]
                            if feasible:
                                optimum = min(feasible, key=lambda it: (int(it["TOTAL_Q_CALLS95"]), int(it["GROVER_ITERATIONS"])))
                                counts["minimum_cost_r_checks"] += 1
                                if chosen_r != int(optimum["GROVER_ITERATIONS"]):
                                    issues.append(f"Nonminimum chosen Grover r: {key} chosen={chosen_r}, optimum={optimum['GROVER_ITERATIONS']}")
        if not seen:
            notes.append("No matching pose/scaling_summary.tsv found; QSEL single-shot true-good probability not cross-compared")
        else:
            missing = set(rows) - seen
            if missing:
                notes.append(f"{len(missing)} validation rows lack individual pose/scaling_summary.tsv comparisons")
            counts["scaling_files"] = sum(file.parent.name == "pose" for file in scaling_paths)

    complete = (
        batch_rows is not None
        and counts["matching_batch_rows"] == len(rows)
        and counts["scaling_goodR_checks"] == len(rows)
        and counts["minimum_cost_r_checks"] == len(rows)
        and counts["grover_curve_groups"] == len(rows)
    )
    extra_configurations = len(set(batch_rows) - set(rows)) if batch_rows is not None else 0
    status = ("FAIL" if issues or disagreements else
              "INCOMPLETE" if batch_rows is None else
              "PASS_VERIFIED_SUBSET" if complete and extra_configurations else
              "PASS" if complete else "PARTIAL")
    receptors = sorted({key[0] for key in rows})
    checkpoints = sorted({key[3] for key in rows})
    base_m0_values = sorted({key[1] for key in rows})
    body = [
        "POSE-LEVEL PRODUCTION OUTPUT VS INDEPENDENT REAL-CHAIN VALIDATION",
        f"Result                : {status}",
        f"Validation source     : {report.resolve()}",
        f"Production batch      : {batch_path.resolve() if batch_path else 'NOT PROVIDED'}",
        f"Receptors             : {len(receptors)} ({' '.join(receptors)})",
        f"Nested OCC checkpoints: {' '.join(map(str, checkpoints))}",
        f"Validation base m0    : {' '.join(map(str, base_m0_values))}",
        f"Verified pose rows    : {len(rows)}",
        f"Production pose rows  : {counts['production_batch_rows'] if batch_rows is not None else 'NOT CHECKED'}",
        f"Matched pose rows     : {counts['matching_batch_rows'] if batch_rows is not None else 'NOT CHECKED'}",
        f"Unverified production configurations: {extra_configurations}",
        f"Float field checks    : {counts['numeric_field_checks'] if batch_rows is not None else 'NOT CHECKED'}",
        f"Integer field checks  : {counts['integer_field_checks'] if batch_rows is not None else 'NOT CHECKED'}",
        f"Single-shot QSEL checks: {counts['scaling_goodR_checks'] if batch_rows is not None else 'NOT CHECKED'}",
        f"Validation cost checks: {counts['validation_cost_checks']}",
        f"Grover chosen-point checks: {counts['grover_selected_checks'] if batch_rows is not None else 'NOT CHECKED'}",
        f"Minimum-cost r checks  : {counts['minimum_cost_r_checks'] if batch_rows is not None else 'NOT CHECKED'}",
        f"Full Grover curves      : {counts['grover_curve_groups'] if batch_rows is not None else 'NOT CHECKED'}",
        f"Every-r curve points    : {counts['grover_curve_points'] if batch_rows is not None else 'NOT CHECKED'}",
        f"Every-r probability checks : {counts['grover_curve_probability_checks'] if batch_rows is not None else 'NOT CHECKED'}",
        f"Every-r shots95/99 checks : {counts['grover_curve_shot_checks'] if batch_rows is not None else 'NOT CHECKED'}",
        f"Every-r per-shot/total Q-call checks : {counts['grover_curve_cost_checks'] + counts['grover_curve_total_call_checks'] if batch_rows is not None else 'NOT CHECKED'}",
        f"Every-r floating-error max : {counts['grover_curve_max_prob_delta']:.3e}",
        f"Curves missing 16-digit pose summary: {counts['curves_using_rounded_probability']}",
        f"Every-r numerical boundary notes: {len(precision_notes)}",
        f"Mismatched fields     : {len(disagreements)}",
        f"Structural errors     : {len(issues)}",
        "",
        "Compared fields: qualified fraction q, FLAG probability p, precision eta,",
        "QAE true-good probability, best QSEL r, QAE/QSEL shots95 and total Q calls95.",
        "When pose/scaling_summary.tsv exists, also compare QSEL true-good probability.",
        "All r in each validated pose/grover_shots.tsv are independently checked",
        "against 2D Grover-rotation amplitudes, 95%/99% shots, and logical calls.",
        "Curve checks cover validated m0 only, not unverified production m0.",
        "FLOAT comparison respects the precision displayed in real_chain_check.txt.",
        "INDEX FIRST: only threshold-qualified configuration indices are considered.",
        "This audit compares numerical TSV outputs. It does NOT inspect rendered plots,",
        "plotting scripts, or the numerical data arrays inside Figure 2-5 artwork.",
        "",
    ]
    if len(receptors) > 0:
        body.append("REFERENCE MEDIANS FROM VALIDATION (not yet checked against figure artwork)")
        body.append("  N   m0  k   median_q           median_pflag       median_eta      median_r    median_calls95")
        for n in (1, 16, 128, 1024):
            for m0 in base_m0_values:
                for k in (1, 8):
                    group = [v for key, v in rows.items() if key[1] == m0 and key[2] == k and key[3] == n]
                    if not group:
                        continue
                    median_q = statistics.median(float(v["q_exact"]) for v in group)
                    median_p = statistics.median(float(v["p_flag"]) for v in group)
                    median_eta = statistics.median(float(v["eta"]) for v in group)
                    median_r = statistics.median(int(v["r"]) for v in group)
                    calls = [float("inf") if v["callsR"] in ("INF", "-") else int(v["callsR"]) for v in group]
                    median_calls = statistics.median(calls)
                    cost_token = "INF" if math.isinf(median_calls) else f"{median_calls:.6e}"
                    body.append(f"{n:4d}  {m0:3d}  {k:1d}  {median_q:17.9e}  {median_p:17.9e}  {median_eta:12.7f}  {median_r:10.1f}  {cost_token:>15s}")
        body.append("Note: medians are calculated separately for each column across receptors.")
        body.append("")
    if issues:
        body.extend(["STRUCTURAL ERRORS", *issues, ""])
    if disagreements:
        body.append("FIELD MISMATCHES (max first 200)")
        for key, field, reference, production, delta, tolerance in disagreements[:200]:
            body.append(f"{key} {field} check={reference} prod={production} difference={delta} tolerance={tolerance}")
        body.append("")
    if batch_rows is not None and not complete and not issues and not disagreements:
        notes.append("PARTIAL means some validated rows lack complete production/scaling/Grover crosschecks")
    if complete and extra_configurations:
        notes.append("PASS_VERIFIED_SUBSET means all crosschecks passed for validation m0 values, NOT for extra production m0 values")
    if precision_notes:
        body.extend(["NUMERICAL SHOT-BOUNDARY NOTES (first 100)", *precision_notes[:100], ""])
    if notes:
        body.extend(["NOTES", *notes, ""])
    return "\n".join(body) + "\n", status


def main():
    parser = argparse.ArgumentParser(description="Compare pose-only validated real data to the actual production TSV (no recomputation)")
    parser.add_argument("validation_report", type=Path, help="Previously generated real_chain_check.txt")
    parser.add_argument("production_root", nargs="?", type=Path, help="Directory containing batch_summary_pose.tsv and receptor outputs")
    parser.add_argument("--batch", type=Path, help="Exact production batch_summary_pose.tsv path (overrides auto-discovery)")
    parser.add_argument("--output", type=Path, help="Plaintext audit output; default next to validation report")
    args = parser.parse_args()
    report = args.validation_report
    if not report.is_file():
        parser.error(f"missing validation report: {report}")
    root = args.production_root
    batch = args.batch if args.batch is not None else (root / "batch_summary_pose.tsv" if root is not None else None)
    out = args.output or (report.parent / "pose_figure_check.txt")
    data, status = compare(report, batch, root)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(data, encoding="utf-8")
    print(f"Pose production-vs-validation check: {status}")
    print(f"Report: {out}")
    if status == "FAIL":
        raise SystemExit(1)
    if status in ("INCOMPLETE", "PARTIAL"):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
