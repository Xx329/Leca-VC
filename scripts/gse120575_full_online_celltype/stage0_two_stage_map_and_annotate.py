#!/usr/bin/env python3
"""Authorized two-stage deterministic mapping and frozen GSE120575 annotation.

Stage 1 freezes identical raw IDs. Stage 2 removes those records and performs
an exact join only among the normalized residual IDs. The exact:: and
residual:: namespaces are never collapsed. No fuzzy, distance, order-based,
cluster-based, expression-based, or manually selected pairing is implemented.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from stage0_normalize_and_annotate import (
    ALL_MARKERS,
    BROAD_TYPES,
    DIRECT_BROAD,
    NK_MARKERS,
    PATIENT_SAMPLE_COLUMN,
    PUBLISHED_POPULATIONS,
    REQUIRED_METADATA_COLUMNS,
    RESPONSE_COLUMN,
    THERAPY_COLUMN,
    T_MARKERS,
    annotation_canonical,
    duplicate_count,
    expression_canonical,
    load_cluster_rows,
    load_expression_ids,
    load_metadata,
    natural_sample_key,
    scan_frozen_markers,
    sha256_file,
    write_csv,
    write_json,
)


PASS_STATUS = "PASS_WITH_TWO_STAGE_DETERMINISTIC_ID_MAPPING"
MAPPING_FIELDS = [
    "final_match_key",
    "match_stage",
    "expression_raw_id",
    "annotation_raw_id",
    "residual_canonical_id",
    "cluster_number",
    "expression_normalization_rule",
    "annotation_normalization_rule",
    "originally_exact_match",
    "one_to_one_valid",
]


def expression_rule_name(internal_rule: str) -> str:
    return {
        "unchanged": "unchanged",
        "remove_expression_T_enriched": "remove_terminal_T_enriched",
        "remove_expression_myeloid_enriched": "remove_terminal_myeloid_enriched",
    }[internal_rule]


def annotation_rule_name(internal_rule: str) -> str:
    return {
        "unchanged": "unchanged",
        "remove_annotation_DN": "remove_terminal_DN",
        "remove_annotation_DN1": "remove_terminal_DN1",
        "remove_annotation_DP1": "remove_terminal_DP1",
    }[internal_rule]


def write_audit(path: Path, summary: dict[str, object]) -> None:
    lines = [
        "# GSE120575 two-stage deterministic ID mapping audit",
        "",
        f"- Status: **{summary['status']}**",
        f"- All safety gates passed: **{summary['all_safety_gates_passed']}**",
        "- Pairing implementation: exact unique dictionary keys only",
        "- Prohibited methods used: none",
        "",
        "## Frozen two-stage procedure",
        "",
        "1. Freeze every identical expression/annotation raw ID as `exact::<raw_id>`.",
        "2. Remove all stage-1 records from both pending sets.",
        "3. Normalize only the two residual sets using the authorized terminal suffix rules.",
        "4. Require unique residual canonical IDs and exact residual-set equality.",
        "5. Join residuals as `residual::<canonical_id>`.",
        "6. Never collapse the `exact::` and `residual::` namespaces.",
        "",
        "Fuzzy matching, edit distance, closest-string matching, row-order pairing, "
        "cluster-based selection, expression-based selection and manual positional pairing are absent.",
        "",
        "## Safety gates",
        "",
        "| Gate | Observed | Expected | Pass |",
        "|---|---:|---:|---|",
    ]
    for name, result in summary["safety_gates"].items():
        observed = result["observed"]
        expected = result["expected"]
        if isinstance(observed, (dict, list)):
            observed = json.dumps(observed, ensure_ascii=False, sort_keys=True)
        if isinstance(expected, (dict, list)):
            expected = json.dumps(expected, ensure_ascii=False, sort_keys=True)
        lines.append(f"| `{name}` | {observed} | {expected} | {result['pass']} |")
    lines.extend(
        [
            "",
            "## Residual diagnostics (not used for pairing)",
            "",
            "```json",
            json.dumps(summary["residual_diagnostics"], ensure_ascii=False, indent=2, sort_keys=True),
            "```",
            "",
            "## Collision-group preservation",
            "",
            f"- Collision groups checked: {summary['collision_resolution']['group_count']}",
            f"- Groups retaining two unique keys and two clusters: {summary['collision_resolution']['valid_group_count']}",
            "- Full records: `two_stage_collision_resolution.csv`",
            "",
            "## Decision",
            "",
            (
                "All frozen gates passed. Stage 0 may proceed to the frozen two-layer annotation and sample inventory."
                if summary["all_safety_gates_passed"]
                else "At least one frozen gate failed. No annotation or sample inventory may be generated."
            ),
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expression", type=Path, required=True)
    parser.add_argument("--cluster-info", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    audit_dir = args.output_root / "audit"
    annotation_dir = args.output_root / "annotation"
    audit_dir.mkdir(parents=True, exist_ok=True)
    annotation_dir.mkdir(parents=True, exist_ok=True)

    expression_ids, expression_header_sha = load_expression_ids(args.expression)
    cluster_rows = load_cluster_rows(args.cluster_info)
    metadata_rows, _ = load_metadata(args.metadata)
    annotation_ids = [str(row["raw_id"]) for row in cluster_rows]
    metadata_ids = [row["title"] for row in metadata_rows]
    expression_set = set(expression_ids)
    annotation_set = set(annotation_ids)
    annotation_by_raw = {str(row["raw_id"]): row for row in cluster_rows}
    metadata_by_id = {row["title"]: row for row in metadata_rows}

    # Stage 1: freeze all raw exact matches before any normalization.
    raw_exact_ids = expression_set & annotation_set
    residual_expression_ids = [raw_id for raw_id in expression_ids if raw_id not in raw_exact_ids]
    residual_annotation_rows = [
        row for row in cluster_rows if str(row["raw_id"]) not in raw_exact_ids
    ]
    residual_annotation_ids = [str(row["raw_id"]) for row in residual_annotation_rows]

    # Stage 2: normalize only residuals, then exact-join unique canonical keys.
    residual_expression_records = [
        (raw_id, *expression_canonical(raw_id)) for raw_id in residual_expression_ids
    ]
    residual_annotation_records = [
        (str(row["raw_id"]), *annotation_canonical(str(row["raw_id"])), int(row["cluster"]))
        for row in residual_annotation_rows
    ]
    residual_expression_canonical = [record[1] for record in residual_expression_records]
    residual_annotation_canonical = [record[1] for record in residual_annotation_records]
    residual_expression_canonical_set = set(residual_expression_canonical)
    residual_annotation_canonical_set = set(residual_annotation_canonical)
    residual_annotation_by_canonical = {
        record[1]: record for record in residual_annotation_records
    }

    mapping_rows: list[dict[str, object]] = []
    if duplicate_count(expression_ids) == 0 and duplicate_count(annotation_ids) == 0:
        for expression_raw_id in expression_ids:
            if expression_raw_id in raw_exact_ids:
                source = annotation_by_raw[expression_raw_id]
                mapping_rows.append(
                    {
                        "final_match_key": f"exact::{expression_raw_id}",
                        "match_stage": "raw_exact",
                        "expression_raw_id": expression_raw_id,
                        "annotation_raw_id": expression_raw_id,
                        "residual_canonical_id": "",
                        "cluster_number": int(source["cluster"]),
                        "expression_normalization_rule": "unchanged",
                        "annotation_normalization_rule": "unchanged",
                        "originally_exact_match": "true",
                        "one_to_one_valid": "true",
                    }
                )
            else:
                canonical, expression_rule = expression_canonical(expression_raw_id)
                annotation_record = residual_annotation_by_canonical.get(canonical)
                if annotation_record is None:
                    continue
                annotation_raw_id, annotation_can, annotation_rule, cluster = annotation_record
                mapping_rows.append(
                    {
                        "final_match_key": f"residual::{canonical}",
                        "match_stage": "residual_normalized",
                        "expression_raw_id": expression_raw_id,
                        "annotation_raw_id": annotation_raw_id,
                        "residual_canonical_id": annotation_can,
                        "cluster_number": cluster,
                        "expression_normalization_rule": expression_rule_name(expression_rule),
                        "annotation_normalization_rule": annotation_rule_name(annotation_rule),
                        "originally_exact_match": "false",
                        "one_to_one_valid": "true",
                    }
                )

    mapped_expression_ids = [str(row["expression_raw_id"]) for row in mapping_rows]
    mapped_annotation_ids = [str(row["annotation_raw_id"]) for row in mapping_rows]
    final_keys = [str(row["final_match_key"]) for row in mapping_rows]
    raw_cluster_counts = Counter(int(row["cluster"]) for row in cluster_rows)
    mapped_cluster_counts = Counter(int(row["cluster_number"]) for row in mapping_rows)
    stage_counts = Counter(str(row["match_stage"]) for row in mapping_rows)
    metadata_na_by_field = {
        column: sum(
            1
            for expression_id in mapped_expression_ids
            if expression_id not in metadata_by_id
            or not metadata_by_id[expression_id].get(column, "").strip()
        )
        for column in REQUIRED_METADATA_COLUMNS
    }

    # Identify the previously observed global canonical collision groups, then
    # verify that two-stage namespaces preserve both cells and their clusters.
    all_expression_by_global_canonical: dict[str, list[str]] = defaultdict(list)
    all_annotation_by_global_canonical: dict[str, list[dict[str, object]]] = defaultdict(list)
    for raw_id in expression_ids:
        all_expression_by_global_canonical[expression_canonical(raw_id)[0]].append(raw_id)
    for row in cluster_rows:
        all_annotation_by_global_canonical[annotation_canonical(str(row["raw_id"]))[0]].append(row)
    global_collision_keys = sorted(
        canonical
        for canonical in set(all_expression_by_global_canonical) | set(all_annotation_by_global_canonical)
        if len(all_expression_by_global_canonical[canonical]) > 1
        or len(all_annotation_by_global_canonical[canonical]) > 1
    )
    mapping_by_key = {str(row["final_match_key"]): row for row in mapping_rows}
    collision_rows: list[dict[str, object]] = []
    valid_collision_groups = 0
    for canonical in global_collision_keys:
        exact_key = f"exact::{canonical}"
        residual_key = f"residual::{canonical}"
        exact_row = mapping_by_key.get(exact_key)
        residual_row = mapping_by_key.get(residual_key)
        valid = bool(
            exact_row
            and residual_row
            and exact_row["expression_raw_id"] != residual_row["expression_raw_id"]
            and exact_row["annotation_raw_id"] != residual_row["annotation_raw_id"]
            and exact_key != residual_key
        )
        valid_collision_groups += int(valid)
        collision_rows.append(
            {
                "global_canonical_id": canonical,
                "exact_final_match_key": exact_key,
                "exact_expression_raw_id": "" if not exact_row else exact_row["expression_raw_id"],
                "exact_annotation_raw_id": "" if not exact_row else exact_row["annotation_raw_id"],
                "exact_cluster_number": "" if not exact_row else exact_row["cluster_number"],
                "residual_final_match_key": residual_key,
                "residual_expression_raw_id": "" if not residual_row else residual_row["expression_raw_id"],
                "residual_annotation_raw_id": "" if not residual_row else residual_row["annotation_raw_id"],
                "residual_cluster_number": "" if not residual_row else residual_row["cluster_number"],
                "two_distinct_cells_preserved": str(valid).lower(),
            }
        )

    expected = {
        "expression_raw_count": 16291,
        "expression_raw_unique": 16291,
        "annotation_raw_count": 16291,
        "annotation_raw_unique": 16291,
        "raw_exact_matches": 15300,
        "residual_expression_count": 991,
        "residual_annotation_count": 991,
        "residual_expression_canonical_unique": 991,
        "residual_annotation_canonical_unique": 991,
        "residual_canonical_intersection": 991,
        "residual_expression_only": 0,
        "residual_annotation_only": 0,
        "final_mapping_rows": 16291,
        "unique_mapped_expression_raw_id": 16291,
        "unique_mapped_annotation_raw_id": 16291,
        "unique_final_match_key": 16291,
        "raw_exact_stage_rows": 15300,
        "residual_normalized_stage_rows": 991,
        "unmatched_expression": 0,
        "unmatched_annotation": 0,
        "duplicate_expression_assignment": 0,
        "duplicate_annotation_assignment": 0,
        "clusters_present": list(range(1, 12)),
        "metadata_expression_exact_intersection": 16291,
        "metadata_na_by_required_field": {column: 0 for column in REQUIRED_METADATA_COLUMNS},
        "global_collision_groups": 67,
        "valid_two_namespace_collision_groups": 67,
    }
    observed = {
        "expression_raw_count": len(expression_ids),
        "expression_raw_unique": len(expression_set),
        "annotation_raw_count": len(annotation_ids),
        "annotation_raw_unique": len(annotation_set),
        "raw_exact_matches": len(raw_exact_ids),
        "residual_expression_count": len(residual_expression_ids),
        "residual_annotation_count": len(residual_annotation_ids),
        "residual_expression_canonical_unique": len(residual_expression_canonical_set),
        "residual_annotation_canonical_unique": len(residual_annotation_canonical_set),
        "residual_canonical_intersection": len(
            residual_expression_canonical_set & residual_annotation_canonical_set
        ),
        "residual_expression_only": len(
            residual_expression_canonical_set - residual_annotation_canonical_set
        ),
        "residual_annotation_only": len(
            residual_annotation_canonical_set - residual_expression_canonical_set
        ),
        "final_mapping_rows": len(mapping_rows),
        "unique_mapped_expression_raw_id": len(set(mapped_expression_ids)),
        "unique_mapped_annotation_raw_id": len(set(mapped_annotation_ids)),
        "unique_final_match_key": len(set(final_keys)),
        "raw_exact_stage_rows": stage_counts["raw_exact"],
        "residual_normalized_stage_rows": stage_counts["residual_normalized"],
        "unmatched_expression": len(expression_set - set(mapped_expression_ids)),
        "unmatched_annotation": len(annotation_set - set(mapped_annotation_ids)),
        "duplicate_expression_assignment": duplicate_count(mapped_expression_ids),
        "duplicate_annotation_assignment": duplicate_count(mapped_annotation_ids),
        "clusters_present": sorted(mapped_cluster_counts),
        "metadata_expression_exact_intersection": len(set(metadata_ids) & set(mapped_expression_ids)),
        "metadata_na_by_required_field": metadata_na_by_field,
        "global_collision_groups": len(global_collision_keys),
        "valid_two_namespace_collision_groups": valid_collision_groups,
    }
    safety_gates = {
        name: {"observed": observed[name], "expected": expected_value, "pass": observed[name] == expected_value}
        for name, expected_value in expected.items()
    }
    safety_gates["cluster_counts_unchanged"] = {
        "observed": dict(sorted(mapped_cluster_counts.items())),
        "expected": dict(sorted(raw_cluster_counts.items())),
        "pass": mapped_cluster_counts == raw_cluster_counts,
    }

    residual_mapping_rows = [row for row in mapping_rows if row["match_stage"] == "residual_normalized"]
    suffix_cluster_counts: dict[str, Counter[int]] = {
        "_T_enriched": Counter(),
        "_myeloid_enriched": Counter(),
    }
    residual_group_counts: Counter[tuple[str, str, str]] = Counter()
    for row in residual_mapping_rows:
        expression_id = str(row["expression_raw_id"])
        suffix = "_T_enriched" if expression_id.endswith("_T_enriched") else "_myeloid_enriched"
        suffix_cluster_counts[suffix][int(row["cluster_number"])] += 1
        metadata = metadata_by_id[expression_id]
        sample_id = metadata[PATIENT_SAMPLE_COLUMN]
        match = re.fullmatch(r"(Pre|Post)_(P\d+)(?:_\d+)?", sample_id)
        if not match:
            raise ValueError(f"Unexpected sample token {sample_id!r}")
        treatment, patient = match.groups()
        residual_group_counts[(patient, treatment, metadata[RESPONSE_COLUMN])] += 1

    residual_diagnostics = {
        "used_for_pairing_decisions": False,
        "expression_suffix_to_cluster_counts": {
            suffix: {f"G{cluster}": counts[cluster] for cluster in range(1, 12)}
            for suffix, counts in suffix_cluster_counts.items()
        },
        "patient_treatment_response_counts": [
            {
                "patient_id": patient,
                "treatment_status": treatment,
                "response": response,
                "residual_count": count,
            }
            for (patient, treatment, response), count in sorted(residual_group_counts.items())
        ],
        "mapping_changed_cluster_distribution": mapped_cluster_counts != raw_cluster_counts,
        "interpretation": (
            "No mapping-induced cluster distribution change: every annotation raw ID is used exactly once."
            if mapped_cluster_counts == raw_cluster_counts
            else "FAIL: mapped cluster counts differ from the source annotation."
        ),
    }
    all_passed = all(bool(result["pass"]) for result in safety_gates.values())
    summary: dict[str, object] = {
        "status": PASS_STATUS if all_passed else "FAIL_TWO_STAGE_DETERMINISTIC_ID_MAPPING",
        "all_safety_gates_passed": all_passed,
        "inputs": {
            "expression_matrix": str(args.expression.resolve()),
            "cluster_info": str(args.cluster_info.resolve()),
            "geo_metadata": str(args.metadata.resolve()),
            "expression_header_sha256": expression_header_sha,
            "cluster_info_sha256": sha256_file(args.cluster_info),
            "geo_metadata_sha256": sha256_file(args.metadata),
        },
        "pairing_method": "raw_exact_freeze_then_residual_unique_canonical_exact_join",
        "prohibited_methods_used": [],
        "safety_gates": safety_gates,
        "match_stage_counts": dict(sorted(stage_counts.items())),
        "cluster_counts": {f"G{k}": mapped_cluster_counts[k] for k in range(1, 12)},
        "collision_resolution": {
            "group_count": len(global_collision_keys),
            "valid_group_count": valid_collision_groups,
        },
        "residual_diagnostics": residual_diagnostics,
    }

    write_csv(audit_dir / "two_stage_id_mapping.csv", MAPPING_FIELDS, mapping_rows)
    write_csv(
        audit_dir / "two_stage_collision_resolution.csv",
        [
            "global_canonical_id",
            "exact_final_match_key",
            "exact_expression_raw_id",
            "exact_annotation_raw_id",
            "exact_cluster_number",
            "residual_final_match_key",
            "residual_expression_raw_id",
            "residual_annotation_raw_id",
            "residual_cluster_number",
            "two_distinct_cells_preserved",
        ],
        collision_rows,
    )
    write_json(audit_dir / "two_stage_mapping_summary.json", summary)
    write_audit(audit_dir / "two_stage_mapping_audit.md", summary)
    if not all_passed:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        print("STOP: two-stage deterministic mapping safety gate failed", file=sys.stderr)
        return 2

    # Only now may frozen annotation begin.
    marker_values = scan_frozen_markers(args.expression, len(expression_ids))
    mapping_by_expression = {str(row["expression_raw_id"]): row for row in mapping_rows}
    cell_annotations: list[dict[str, object]] = []
    broad_counts: Counter[str] = Counter()
    marker_split_counts: Counter[str] = Counter()
    marker_ties = 0
    for index, expression_id in enumerate(expression_ids):
        mapping = mapping_by_expression[expression_id]
        cluster = int(mapping["cluster_number"])
        metadata = metadata_by_id[expression_id]
        t_score: float | str = ""
        nk_score: float | str = ""
        if cluster in (5, 8):
            t_score = sum(marker_values[gene][index] for gene in T_MARKERS) / len(T_MARKERS)
            nk_score = sum(marker_values[gene][index] for gene in NK_MARKERS) / len(NK_MARKERS)
            if t_score >= nk_score:
                broad = "T cell"
                annotation_rule = "frozen_marker_mean_T_ge_NK"
                marker_ties += int(t_score == nk_score)
            else:
                broad = "NK cell"
                annotation_rule = "frozen_marker_mean_NK_gt_T"
            marker_split_counts[broad] += 1
        else:
            broad = DIRECT_BROAD[cluster]
            annotation_rule = "published_cluster_direct"
        broad_counts[broad] += 1
        cell_annotations.append(
            {
                "cell_id": expression_id,
                "final_match_key": mapping["final_match_key"],
                "annotation_source_cell_id": mapping["annotation_raw_id"],
                "published_cluster": f"G{cluster}",
                "published_population": PUBLISHED_POPULATIONS[cluster],
                "broad_cell_type": broad,
                "broad_annotation_rule": annotation_rule,
                "t_marker_mean": "" if t_score == "" else f"{t_score:.8g}",
                "nk_marker_mean": "" if nk_score == "" else f"{nk_score:.8g}",
                "patient_sample": metadata[PATIENT_SAMPLE_COLUMN],
                "response": metadata[RESPONSE_COLUMN],
                "therapy": metadata[THERAPY_COLUMN],
            }
        )

    write_csv(
        annotation_dir / "cell_level_annotations.csv",
        [
            "cell_id", "final_match_key", "annotation_source_cell_id",
            "published_cluster", "published_population", "broad_cell_type",
            "broad_annotation_rule", "t_marker_mean", "nk_marker_mean",
            "patient_sample", "response", "therapy",
        ],
        cell_annotations,
    )
    write_csv(
        annotation_dir / "fine_population_mapping.csv",
        ["published_cluster", "published_population", "broad_mapping"],
        [
            {
                "published_cluster": f"G{k}",
                "published_population": PUBLISHED_POPULATIONS[k],
                "broad_mapping": DIRECT_BROAD.get(k, "marker_split_T_or_NK"),
            }
            for k in range(1, 12)
        ],
    )
    write_csv(
        annotation_dir / "broad_cell_type_mapping.csv",
        ["published_cluster", "broad_cell_type_or_rule", "mapping_rule"],
        [
            {
                "published_cluster": f"G{k}",
                "broad_cell_type_or_rule": DIRECT_BROAD.get(k, "T cell or NK cell"),
                "mapping_rule": (
                    "mean(T markers) >= mean(NK markers) => T; otherwise NK"
                    if k in (5, 8) else "published_cluster_direct"
                ),
            }
            for k in range(1, 12)
        ],
    )

    annotation_by_expression = {str(row["cell_id"]): row for row in cell_annotations}
    sample_cells: dict[str, list[str]] = defaultdict(list)
    for expression_id in expression_ids:
        sample_cells[metadata_by_id[expression_id][PATIENT_SAMPLE_COLUMN]].append(expression_id)
    patient_treatments: dict[str, set[str]] = defaultdict(set)
    sample_properties: dict[str, tuple[str, str, str, str, int]] = {}
    for sample_id, cell_ids in sample_cells.items():
        match = re.fullmatch(r"(Pre|Post)_(P\d+)(?:_(\d+))?", sample_id)
        if not match:
            raise ValueError(f"Unexpected GEO sample token: {sample_id!r}")
        treatment, patient, biopsy = match.groups()
        responses = {metadata_by_id[cell_id][RESPONSE_COLUMN] for cell_id in cell_ids}
        therapies = {metadata_by_id[cell_id][THERAPY_COLUMN] for cell_id in cell_ids}
        if len(responses) != 1 or len(therapies) != 1:
            raise ValueError(f"Inconsistent metadata within sample {sample_id}")
        patient_treatments[patient].add(treatment)
        sample_properties[sample_id] = (
            patient, treatment, next(iter(responses)), next(iter(therapies)), int(biopsy or 1)
        )

    sample_rows: list[dict[str, object]] = []
    for sample_id in sorted(sample_cells, key=natural_sample_key):
        patient, treatment, response, therapy, biopsy = sample_properties[sample_id]
        cell_ids = sample_cells[sample_id]
        sample_rows.append(
            {
                "sample_id": sample_id,
                "patient_id": patient,
                "treatment_status": treatment,
                "biopsy_index": biopsy,
                "response": response,
                "therapy": therapy,
                "n_cells": len(cell_ids),
                "n_unique_cell_ids": len(set(cell_ids)),
                "residual_normalized_cells": sum(
                    mapping_by_expression[cell_id]["match_stage"] == "residual_normalized"
                    for cell_id in cell_ids
                ),
                "patient_has_pre_and_post": str(patient_treatments[patient] == {"Pre", "Post"}).lower(),
                "proportion_denominator": "all annotated cells in this sample_id x treatment_status biopsy",
            }
        )
    write_csv(
        audit_dir / "sample_inventory.csv",
        [
            "sample_id", "patient_id", "treatment_status", "biopsy_index",
            "response", "therapy", "n_cells", "n_unique_cell_ids",
            "residual_normalized_cells", "patient_has_pre_and_post", "proportion_denominator",
        ],
        sample_rows,
    )

    inventory_rows: list[dict[str, object]] = []
    for sample in sample_rows:
        sample_id = str(sample["sample_id"])
        cell_ids = sample_cells[sample_id]
        counts = Counter(str(annotation_by_expression[cell_id]["broad_cell_type"]) for cell_id in cell_ids)
        denominator = len(cell_ids)
        for broad in BROAD_TYPES:
            inventory_rows.append(
                {
                    "sample_id": sample_id,
                    "patient_id": sample["patient_id"],
                    "treatment_status": sample["treatment_status"],
                    "response": sample["response"],
                    "broad_cell_type": broad,
                    "cell_count": counts[broad],
                    "sample_treatment_denominator": denominator,
                    "cell_type_proportion": f"{counts[broad] / denominator:.12g}",
                }
            )
    write_csv(
        audit_dir / "annotation_inventory.csv",
        [
            "sample_id", "patient_id", "treatment_status", "response",
            "broad_cell_type", "cell_count", "sample_treatment_denominator", "cell_type_proportion",
        ],
        inventory_rows,
    )

    annotation_summary = {
        "status": "PASS_FROZEN_TWO_LAYER_ANNOTATION",
        "cell_count": len(cell_annotations),
        "published_cluster_counts": {f"G{k}": mapped_cluster_counts[k] for k in range(1, 12)},
        "broad_cell_type_counts": dict(sorted(broad_counts.items())),
        "g5_g8_marker_split_counts": dict(sorted(marker_split_counts.items())),
        "g5_g8_marker_score_ties_assigned_to_T": marker_ties,
        "t_markers": list(T_MARKERS),
        "nk_markers": list(NK_MARKERS),
        "marker_score": "arithmetic mean of supplied expression values for each frozen marker set",
        "marker_decision": "T when T mean >= NK mean; NK otherwise",
        "ground_truth_scgpt_used": False,
        "response_or_treatment_used_for_annotation": False,
        "sample_count": len(sample_rows),
        "sample_group_counts": dict(
            sorted(Counter(f"{row['treatment_status']}_{row['response']}" for row in sample_rows).items())
        ),
        "patients_with_pre_and_post": sorted(
            patient for patient, treatments in patient_treatments.items() if treatments == {"Pre", "Post"}
        ),
    }
    write_json(audit_dir / "annotation_summary.json", annotation_summary)
    print(json.dumps({"mapping": summary["status"], "annotation": annotation_summary}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
