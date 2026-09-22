#!/usr/bin/env python3
"""Audited deterministic ID normalization and frozen GSE120575 annotation.

This script intentionally does not implement fuzzy matching, edit distance,
row-order pairing, nearest-string matching, or manual positional pairing.
Canonical IDs are joined only by exact dictionary lookup after applying the
explicit suffix rules authorized for this experiment.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


STATUS = "PASS_WITH_AUDITED_DETERMINISTIC_ID_NORMALIZATION"
T_MARKERS = ("CD3D", "CD3E", "TRAC", "CD247")
NK_MARKERS = ("NKG7", "GNLY", "KLRD1", "FCGR3A", "TYROBP")
ALL_MARKERS = set(T_MARKERS + NK_MARKERS)
BROAD_TYPES = (
    "B cell",
    "Plasma cell",
    "Monocyte/Macrophage",
    "Dendritic cell",
    "T cell",
    "NK cell",
)

PUBLISHED_POPULATIONS = {
    1: "B cells",
    2: "Plasma cells",
    3: "Monocytes/Macrophages",
    4: "Dendritic cells",
    5: "Lymphocytes",
    6: "Exhausted CD8+ T cells",
    7: "Regulatory T cells",
    8: "Cytotoxicity lymphocytes",
    9: "Exhausted/heat-shock CD8+ T cells",
    10: "Memory T cells",
    11: "Lymphocytes exhausted/cell cycle",
}

DIRECT_BROAD = {
    1: "B cell",
    2: "Plasma cell",
    3: "Monocyte/Macrophage",
    4: "Dendritic cell",
    6: "T cell",
    7: "T cell",
    9: "T cell",
    10: "T cell",
    11: "T cell",
}

PATIENT_SAMPLE_COLUMN = (
    "characteristics: patinet ID (Pre=baseline; Post= on treatment)"
)
RESPONSE_COLUMN = "characteristics: response"
THERAPY_COLUMN = "characteristics: therapy"
REQUIRED_METADATA_COLUMNS = ("title", PATIENT_SAMPLE_COLUMN, RESPONSE_COLUMN, THERAPY_COLUMN)
ALLOWED_MAPPING_RULES = {
    "unchanged_exact",
    "remove_expression_T_enriched_and_annotation_DN",
    "remove_expression_T_enriched_and_annotation_DN1",
    "remove_expression_T_enriched_and_annotation_DP1",
    "remove_expression_myeloid_enriched",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def duplicate_count(values: Iterable[str]) -> int:
    counts = Counter(values)
    return sum(count - 1 for count in counts.values() if count > 1)


def expression_canonical(raw_id: str) -> tuple[str, str]:
    if raw_id.endswith("_myeloid_enriched"):
        return raw_id[: -len("_myeloid_enriched")], "remove_expression_myeloid_enriched"
    if raw_id.endswith("_T_enriched"):
        return raw_id[: -len("_T_enriched")], "remove_expression_T_enriched"
    return raw_id, "unchanged"


def annotation_canonical(raw_id: str) -> tuple[str, str]:
    # Longest suffixes are tested before _DN for clarity and auditability.
    for suffix, rule in (
        ("_DN1", "remove_annotation_DN1"),
        ("_DP1", "remove_annotation_DP1"),
        ("_DN", "remove_annotation_DN"),
    ):
        if raw_id.endswith(suffix):
            return raw_id[: -len(suffix)], rule
    return raw_id, "unchanged"


def load_expression_ids(path: Path) -> tuple[list[str], str]:
    with path.open("rb") as handle:
        header = handle.readline().rstrip(b"\r\n")
    fields = header.decode("utf-8").split("\t")
    if len(fields) < 2:
        raise ValueError("Expression matrix header has fewer than two columns")
    return fields[1:], hashlib.sha256(header).hexdigest()


def load_cluster_rows(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not reader.fieldnames or "Cell Name" not in reader.fieldnames or "Cluster number" not in reader.fieldnames:
            raise ValueError("Cluster file must contain 'Cell Name' and 'Cluster number'")
        rows = []
        for row in reader:
            raw_id = row["Cell Name"]
            try:
                cluster = int(row["Cluster number"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Non-integer cluster for {raw_id!r}") from exc
            rows.append({"raw_id": raw_id, "cluster": cluster})
    return rows


def open_metadata(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="latin-1", newline="")
    return path.open("r", encoding="latin-1", newline="")


def load_metadata(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with open_metadata(path) as handle:
        reader = csv.reader(handle, delimiter="\t")
        header = None
        data_rows = []
        for row in reader:
            if header is None:
                if row and row[0] == "Sample name":
                    header = row
                continue
            # The GEO template contains protocol sections after the 16,291 samples.
            if not row or not row[0].startswith("Sample "):
                continue
            padded = row + [""] * (len(header) - len(row))
            data_rows.append(dict(zip(header, padded)))
    if header is None:
        raise ValueError("Could not locate GEO sample metadata header")
    missing_columns = [column for column in REQUIRED_METADATA_COLUMNS if column not in header]
    if missing_columns:
        raise ValueError(f"Missing GEO metadata columns: {missing_columns}")
    return data_rows, header


def derive_mapping_rule(expression_raw: str, annotation_raw: str) -> str:
    if expression_raw == annotation_raw:
        return "unchanged_exact"
    if expression_raw.endswith("_T_enriched"):
        for suffix in ("_DN1", "_DP1", "_DN"):
            if annotation_raw.endswith(suffix):
                return f"remove_expression_T_enriched_and_annotation_{suffix[1:]}"
    if expression_raw.endswith("_myeloid_enriched"):
        return "remove_expression_myeloid_enriched"
    return "UNAUTHORIZED_RULE"


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_normalization_audit(path: Path, summary: dict[str, object]) -> None:
    checks = summary["checks"]
    lines = [
        "# GSE120575 deterministic ID normalization audit",
        "",
        f"- Status: **{summary['status']}**",
        f"- All safety checks passed: **{summary['all_checks_passed']}**",
        f"- Expression input: `{summary['inputs']['expression_matrix']}`",
        f"- Cluster input: `{summary['inputs']['cluster_info']}`",
        f"- GEO metadata input: `{summary['inputs']['geo_metadata']}`",
        "",
        "## Authorized canonicalization",
        "",
        "- Expression: remove only complete terminal `_T_enriched` or `_myeloid_enriched`.",
        "- Annotation: remove only complete terminal `_DN`, `_DN1`, or `_DP1`.",
        "- Annotation `_L001` is never removed.",
        "- Pairing is an exact dictionary join on unique canonical IDs.",
        "- Fuzzy matching, edit distance, closest-string matching, row-order pairing and manual positional pairing are not implemented.",
        "",
        "## Safety checks",
        "",
        "| Check | Observed | Pass |",
        "|---|---:|---|",
    ]
    for name, result in checks.items():
        observed = result["observed"]
        if isinstance(observed, (dict, list)):
            observed = json.dumps(observed, ensure_ascii=False, sort_keys=True)
        lines.append(f"| `{name}` | {observed} | {result['pass']} |")
    lines.extend(
        [
            "",
            "## Mapping-rule counts",
            "",
            "```json",
            json.dumps(summary.get("mapping_rule_counts", {}), ensure_ascii=False, indent=2, sort_keys=True),
            "```",
            "",
            "## Canonical collision examples",
            "",
            "```json",
            json.dumps(summary.get("canonical_collision_examples", []), ensure_ascii=False, indent=2),
            "```",
            "",
            "## Unmatched-residual diagnostic (not used for pairing)",
            "",
            "```json",
            json.dumps(summary.get("unmatched_residual_diagnostic", {}), ensure_ascii=False, indent=2),
            "```",
            "",
            "## Decision",
            "",
            (
                "Canonical matching reached 16,291 / 16,291 one-to-one exact matches. "
                "Stage 0 may proceed to frozen annotation and sample inventory."
                if summary["all_checks_passed"]
                else "At least one frozen safety check failed. Downstream annotation must not proceed."
            ),
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def scan_frozen_markers(expression_path: Path, n_cells: int) -> dict[str, list[float]]:
    found: dict[str, list[float]] = {}
    with expression_path.open("r", encoding="utf-8", newline="") as handle:
        handle.readline()
        for line in handle:
            gene, separator, rest = line.partition("\t")
            if separator and gene in ALL_MARKERS:
                if gene in found:
                    raise ValueError(f"Frozen marker appears more than once: {gene}")
                values = rest.rstrip("\r\n").split("\t")
                # The source matrix keeps one terminal tab on gene rows. It is
                # accepted only when it produces exactly one final empty field;
                # no non-empty value or additional column is discarded.
                if len(values) == n_cells + 1 and values[-1] == "":
                    values = values[:-1]
                if len(values) != n_cells:
                    raise ValueError(
                        f"Marker {gene} has {len(values)} values; expected {n_cells}"
                    )
                try:
                    found[gene] = [float(value) for value in values]
                except ValueError as exc:
                    raise ValueError(f"Marker {gene} contains a non-numeric value") from exc
    missing = sorted(ALL_MARKERS - set(found))
    if missing:
        raise ValueError(f"Frozen marker genes missing from expression matrix: {missing}")
    return found


def natural_sample_key(sample_id: str) -> tuple[int, int, int]:
    match = re.fullmatch(r"(Pre|Post)_(P\d+)(?:_(\d+))?", sample_id)
    if not match:
        return (9, 999999, 999999)
    treatment, patient, biopsy = match.groups()
    return (0 if treatment == "Pre" else 1, int(patient[1:]), int(biopsy or 1))


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

    expression_canonical_rows = [expression_canonical(raw) for raw in expression_ids]
    annotation_canonical_rows = [annotation_canonical(raw) for raw in annotation_ids]
    expression_canonical_ids = [item[0] for item in expression_canonical_rows]
    annotation_canonical_ids = [item[0] for item in annotation_canonical_rows]

    expression_raw_set = set(expression_ids)
    annotation_raw_set = set(annotation_ids)
    expression_canonical_set = set(expression_canonical_ids)
    annotation_canonical_set = set(annotation_canonical_ids)
    original_exact_ids = expression_raw_set & annotation_raw_set
    residual_expression_ids = [raw_id for raw_id in expression_ids if raw_id not in original_exact_ids]
    residual_annotation_ids = [raw_id for raw_id in annotation_ids if raw_id not in original_exact_ids]
    residual_expression_canonical = [expression_canonical(raw_id)[0] for raw_id in residual_expression_ids]
    residual_annotation_canonical = [annotation_canonical(raw_id)[0] for raw_id in residual_annotation_ids]
    residual_expression_canonical_set = set(residual_expression_canonical)
    residual_annotation_canonical_set = set(residual_annotation_canonical)

    expression_raws_by_canonical: dict[str, list[str]] = defaultdict(list)
    annotation_rows_by_canonical: dict[str, list[dict[str, object]]] = defaultdict(list)
    for raw_id, canonical_row in zip(expression_ids, expression_canonical_rows):
        expression_raws_by_canonical[canonical_row[0]].append(raw_id)
    for source, canonical_row in zip(cluster_rows, annotation_canonical_rows):
        annotation_rows_by_canonical[canonical_row[0]].append(source)
    collision_keys = sorted(
        canonical
        for canonical in expression_canonical_set | annotation_canonical_set
        if len(expression_raws_by_canonical[canonical]) > 1
        or len(annotation_rows_by_canonical[canonical]) > 1
    )
    collision_rows = [
        {
            "canonical_id": canonical,
            "expression_raw_ids": json.dumps(
                expression_raws_by_canonical[canonical], ensure_ascii=False
            ),
            "annotation_raw_ids": json.dumps(
                [str(row["raw_id"]) for row in annotation_rows_by_canonical[canonical]],
                ensure_ascii=False,
            ),
            "annotation_clusters": json.dumps(
                [int(row["cluster"]) for row in annotation_rows_by_canonical[canonical]]
            ),
        }
        for canonical in collision_keys
    ]

    ann_by_canonical: dict[str, dict[str, object]] = {}
    if duplicate_count(annotation_canonical_ids) == 0:
        for source, (canonical, _) in zip(cluster_rows, annotation_canonical_rows):
            ann_by_canonical[canonical] = source

    metadata_by_id: dict[str, dict[str, str]] = {}
    if duplicate_count(metadata_ids) == 0:
        metadata_by_id = {row["title"]: row for row in metadata_rows}

    mapping_rows: list[dict[str, object]] = []
    unexpected_rules = 0
    stable_original_exact = 0
    if (
        duplicate_count(expression_canonical_ids) == 0
        and duplicate_count(annotation_canonical_ids) == 0
        and expression_canonical_set == annotation_canonical_set
    ):
        for expression_raw, (canonical, _) in zip(expression_ids, expression_canonical_rows):
            annotation = ann_by_canonical[canonical]
            annotation_raw = str(annotation["raw_id"])
            annotation_can, _ = annotation_canonical(annotation_raw)
            rule = derive_mapping_rule(expression_raw, annotation_raw)
            if rule not in ALLOWED_MAPPING_RULES:
                unexpected_rules += 1
            originally_exact = expression_raw == annotation_raw and expression_raw in original_exact_ids
            if originally_exact and canonical == expression_raw and annotation_can == annotation_raw:
                stable_original_exact += 1
            mapping_rows.append(
                {
                    "expression_raw_id": expression_raw,
                    "expression_canonical_id": canonical,
                    "annotation_raw_id": annotation_raw,
                    "annotation_canonical_id": annotation_can,
                    "cluster_number": int(annotation["cluster"]),
                    "mapping_rule": rule,
                    "originally_exact_match": str(originally_exact).lower(),
                    "one_to_one_valid": "true",
                }
            )

    raw_cluster_counts = Counter(int(row["cluster"]) for row in cluster_rows)
    mapped_cluster_counts = Counter(int(row["cluster_number"]) for row in mapping_rows)
    metadata_na_by_field = {
        column: sum(
            1
            for expression_id in expression_ids
            if expression_id not in metadata_by_id or not metadata_by_id[expression_id].get(column, "").strip()
        )
        for column in REQUIRED_METADATA_COLUMNS
    }

    observed = {
        "expression_raw_count": len(expression_ids),
        "expression_raw_unique": len(expression_raw_set),
        "expression_raw_duplicate_count": duplicate_count(expression_ids),
        "annotation_raw_count": len(annotation_ids),
        "annotation_raw_unique": len(annotation_raw_set),
        "annotation_raw_duplicate_count": duplicate_count(annotation_ids),
        "expression_canonical_count": len(expression_canonical_ids),
        "expression_canonical_unique": len(expression_canonical_set),
        "expression_canonical_duplicate_count": duplicate_count(expression_canonical_ids),
        "annotation_canonical_count": len(annotation_canonical_ids),
        "annotation_canonical_unique": len(annotation_canonical_set),
        "annotation_canonical_duplicate_count": duplicate_count(annotation_canonical_ids),
        "canonical_intersection": len(expression_canonical_set & annotation_canonical_set),
        "expression_only_canonical": len(expression_canonical_set - annotation_canonical_set),
        "annotation_only_canonical": len(annotation_canonical_set - expression_canonical_set),
        "expression_to_annotation_one_to_one": len(mapping_rows),
        "annotation_to_expression_one_to_one": len({row["annotation_raw_id"] for row in mapping_rows}),
        "original_raw_exact_intersection": len(original_exact_ids),
        "original_exact_matches_stable": stable_original_exact,
        "unexpected_or_unauthorized_mapping_rules": unexpected_rules,
        "raw_cluster_counts": dict(sorted(raw_cluster_counts.items())),
        "mapped_cluster_counts": dict(sorted(mapped_cluster_counts.items())),
        "clusters_present": sorted(raw_cluster_counts),
        "metadata_raw_count": len(metadata_ids),
        "metadata_unique": len(set(metadata_ids)),
        "metadata_duplicate_count": duplicate_count(metadata_ids),
        "metadata_expression_exact_intersection": len(set(metadata_ids) & expression_raw_set),
        "metadata_na_by_required_field": metadata_na_by_field,
        "canonical_collision_group_count": len(collision_keys),
    }

    expected_checks = {
        "expression_raw_count": 16291,
        "expression_raw_unique": 16291,
        "expression_raw_duplicate_count": 0,
        "annotation_raw_count": 16291,
        "annotation_raw_unique": 16291,
        "annotation_raw_duplicate_count": 0,
        "expression_canonical_count": 16291,
        "expression_canonical_unique": 16291,
        "expression_canonical_duplicate_count": 0,
        "annotation_canonical_count": 16291,
        "annotation_canonical_unique": 16291,
        "annotation_canonical_duplicate_count": 0,
        "canonical_intersection": 16291,
        "expression_only_canonical": 0,
        "annotation_only_canonical": 0,
        "expression_to_annotation_one_to_one": 16291,
        "annotation_to_expression_one_to_one": 16291,
        "original_raw_exact_intersection": 15300,
        "original_exact_matches_stable": 15300,
        "unexpected_or_unauthorized_mapping_rules": 0,
        "clusters_present": list(range(1, 12)),
        "metadata_raw_count": 16291,
        "metadata_unique": 16291,
        "metadata_duplicate_count": 0,
        "metadata_expression_exact_intersection": 16291,
        "metadata_na_by_required_field": {column: 0 for column in REQUIRED_METADATA_COLUMNS},
        "canonical_collision_group_count": 0,
    }

    checks: dict[str, dict[str, object]] = {}
    for name, expected in expected_checks.items():
        checks[name] = {"observed": observed[name], "expected": expected, "pass": observed[name] == expected}
    checks["cluster_counts_unchanged"] = {
        "observed": dict(sorted(mapped_cluster_counts.items())),
        "expected": dict(sorted(raw_cluster_counts.items())),
        "pass": mapped_cluster_counts == raw_cluster_counts,
    }
    all_checks_passed = all(bool(check["pass"]) for check in checks.values())
    summary: dict[str, object] = {
        "status": STATUS if all_checks_passed else "FAIL_DETERMINISTIC_ID_NORMALIZATION",
        "all_checks_passed": all_checks_passed,
        "inputs": {
            "expression_matrix": str(args.expression.resolve()),
            "cluster_info": str(args.cluster_info.resolve()),
            "geo_metadata": str(args.metadata.resolve()),
            "expression_header_sha256": expression_header_sha,
            "cluster_info_sha256": sha256_file(args.cluster_info),
            "geo_metadata_sha256": sha256_file(args.metadata),
        },
        "prohibited_methods_used": [],
        "canonical_join_method": "exact_unique_dictionary_key",
        "checks": checks,
        "mapping_rule_counts": dict(sorted(Counter(str(row["mapping_rule"]) for row in mapping_rows).items())),
        "mapping_rows_written": len(mapping_rows),
        "canonical_collision_examples": collision_rows[:10],
        "unmatched_residual_diagnostic": {
            "used_for_pairing": False,
            "raw_exact_ids_preserved": len(original_exact_ids),
            "residual_expression_ids": len(residual_expression_ids),
            "residual_annotation_ids": len(residual_annotation_ids),
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
            "residual_canonicals_colliding_with_preserved_exact_ids": len(
                residual_expression_canonical_set & original_exact_ids
            ),
        },
    }

    write_json(audit_dir / "id_normalization_summary.json", summary)
    write_normalization_audit(audit_dir / "id_normalization_audit.md", summary)
    write_csv(
        audit_dir / "id_normalization_collisions.csv",
        ["canonical_id", "expression_raw_ids", "annotation_raw_ids", "annotation_clusters"],
        collision_rows,
    )
    mapping_fields = [
        "expression_raw_id",
        "expression_canonical_id",
        "annotation_raw_id",
        "annotation_canonical_id",
        "cluster_number",
        "mapping_rule",
        "originally_exact_match",
        "one_to_one_valid",
    ]
    # On failure this is deliberately header-only: manufacturing pairs inside
    # a duplicated canonical key would violate the frozen one-to-one gate.
    write_csv(audit_dir / "id_normalization_mapping.csv", mapping_fields, mapping_rows)

    if not all_checks_passed:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        print("STOP: deterministic ID normalization safety gate failed", file=sys.stderr)
        return 2

    # Annotation begins only after every normalization safety gate has passed.
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
                split_rule = "frozen_marker_mean_T_ge_NK"
                if t_score == nk_score:
                    marker_ties += 1
            else:
                broad = "NK cell"
                split_rule = "frozen_marker_mean_NK_gt_T"
            marker_split_counts[broad] += 1
        else:
            broad = DIRECT_BROAD[cluster]
            split_rule = "published_cluster_direct"
        broad_counts[broad] += 1
        cell_annotations.append(
            {
                "cell_id": expression_id,
                "canonical_cell_id": mapping["expression_canonical_id"],
                "annotation_source_cell_id": mapping["annotation_raw_id"],
                "published_cluster": f"G{cluster}",
                "published_population": PUBLISHED_POPULATIONS[cluster],
                "broad_cell_type": broad,
                "broad_annotation_rule": split_rule,
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
            "cell_id",
            "canonical_cell_id",
            "annotation_source_cell_id",
            "published_cluster",
            "published_population",
            "broad_cell_type",
            "broad_annotation_rule",
            "t_marker_mean",
            "nk_marker_mean",
            "patient_sample",
            "response",
            "therapy",
        ],
        cell_annotations,
    )

    fine_rows = [
        {
            "published_cluster": f"G{cluster}",
            "published_population": PUBLISHED_POPULATIONS[cluster],
            "broad_mapping": DIRECT_BROAD.get(cluster, "marker_split_T_or_NK"),
        }
        for cluster in range(1, 12)
    ]
    write_csv(
        annotation_dir / "fine_population_mapping.csv",
        ["published_cluster", "published_population", "broad_mapping"],
        fine_rows,
    )
    broad_mapping_rows = [
        {
            "published_cluster": f"G{cluster}",
            "broad_cell_type_or_rule": DIRECT_BROAD.get(cluster, "T cell or NK cell"),
            "mapping_rule": (
                "frozen marker mean: T if mean(CD3D,CD3E,TRAC,CD247) >= "
                "mean(NKG7,GNLY,KLRD1,FCGR3A,TYROBP), else NK"
                if cluster in (5, 8)
                else "published_cluster_direct"
            ),
        }
        for cluster in range(1, 12)
    ]
    write_csv(
        annotation_dir / "broad_cell_type_mapping.csv",
        ["published_cluster", "broad_cell_type_or_rule", "mapping_rule"],
        broad_mapping_rows,
    )

    annotations_by_id = {str(row["cell_id"]): row for row in cell_annotations}
    sample_cells: dict[str, list[str]] = defaultdict(list)
    for expression_id in expression_ids:
        sample_cells[metadata_by_id[expression_id][PATIENT_SAMPLE_COLUMN]].append(expression_id)

    sample_rows: list[dict[str, object]] = []
    patient_treatments: dict[str, set[str]] = defaultdict(set)
    sample_properties: dict[str, tuple[str, str, str, str]] = {}
    for sample_id, cell_ids in sample_cells.items():
        match = re.fullmatch(r"(Pre|Post)_(P\d+)(?:_(\d+))?", sample_id)
        if not match:
            raise ValueError(f"Unexpected GEO patient/sample token: {sample_id!r}")
        treatment, patient_id, biopsy = match.groups()
        responses = {metadata_by_id[cell_id][RESPONSE_COLUMN] for cell_id in cell_ids}
        therapies = {metadata_by_id[cell_id][THERAPY_COLUMN] for cell_id in cell_ids}
        if len(responses) != 1 or len(therapies) != 1:
            raise ValueError(f"Inconsistent metadata within sample {sample_id}")
        patient_treatments[patient_id].add(treatment)
        sample_properties[sample_id] = (
            patient_id,
            treatment,
            next(iter(responses)),
            next(iter(therapies)),
        )

    for sample_id in sorted(sample_cells, key=natural_sample_key):
        cell_ids = sample_cells[sample_id]
        patient_id, treatment, response, therapy = sample_properties[sample_id]
        sample_rows.append(
            {
                "sample_id": sample_id,
                "patient_id": patient_id,
                "treatment_status": treatment,
                "biopsy_index": re.fullmatch(r"(?:Pre|Post)_P\d+(?:_(\d+))?", sample_id).group(1) or "1",
                "response": response,
                "therapy": therapy,
                "n_cells": len(cell_ids),
                "n_unique_cell_ids": len(set(cell_ids)),
                "has_T_enriched_cell_ids": str(any(cell.endswith("_T_enriched") for cell in cell_ids)).lower(),
                "has_myeloid_enriched_cell_ids": str(any(cell.endswith("_myeloid_enriched") for cell in cell_ids)).lower(),
                "patient_has_pre_and_post": str(patient_treatments[patient_id] == {"Pre", "Post"}).lower(),
                "proportion_denominator": "sample_id x treatment_status (all annotated cells in this biopsy)",
            }
        )
    write_csv(
        audit_dir / "sample_inventory.csv",
        [
            "sample_id",
            "patient_id",
            "treatment_status",
            "biopsy_index",
            "response",
            "therapy",
            "n_cells",
            "n_unique_cell_ids",
            "has_T_enriched_cell_ids",
            "has_myeloid_enriched_cell_ids",
            "patient_has_pre_and_post",
            "proportion_denominator",
        ],
        sample_rows,
    )

    inventory_rows: list[dict[str, object]] = []
    for sample in sample_rows:
        sample_id = str(sample["sample_id"])
        cell_ids = sample_cells[sample_id]
        counts = Counter(str(annotations_by_id[cell_id]["broad_cell_type"]) for cell_id in cell_ids)
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
                    "sample_total_cells": denominator,
                    "cell_type_proportion": f"{counts[broad] / denominator:.12g}",
                }
            )
    write_csv(
        audit_dir / "annotation_inventory.csv",
        [
            "sample_id",
            "patient_id",
            "treatment_status",
            "response",
            "broad_cell_type",
            "cell_count",
            "sample_total_cells",
            "cell_type_proportion",
        ],
        inventory_rows,
    )

    annotation_summary = {
        "status": "PASS_FROZEN_TWO_LAYER_ANNOTATION",
        "cell_count": len(cell_annotations),
        "published_cluster_counts": {f"G{k}": raw_cluster_counts[k] for k in range(1, 12)},
        "broad_cell_type_counts": dict(sorted(broad_counts.items())),
        "g5_g8_marker_split_counts": dict(sorted(marker_split_counts.items())),
        "g5_g8_marker_score_ties_assigned_to_T": marker_ties,
        "t_markers": list(T_MARKERS),
        "nk_markers": list(NK_MARKERS),
        "marker_score": "arithmetic mean of the expression-matrix values for the frozen marker set",
        "marker_decision": "T cell when T marker mean >= NK marker mean; NK cell otherwise",
        "ground_truth_scgpt_used": False,
        "response_or_treatment_used_for_annotation": False,
        "sample_count": len(sample_rows),
        "sample_group_counts": {
            f"{treatment}_{response}": sum(
                1
                for row in sample_rows
                if row["treatment_status"] == treatment and row["response"] == response
            )
            for treatment in ("Pre", "Post")
            for response in ("Responder", "Non-responder")
        },
        "patients_with_pre_and_post": sorted(
            patient for patient, treatments in patient_treatments.items() if treatments == {"Pre", "Post"}
        ),
    }
    write_json(audit_dir / "annotation_summary.json", annotation_summary)
    print(json.dumps({"normalization": summary["status"], "annotation": annotation_summary}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
