#!/usr/bin/env python3
"""Freeze sample-level proportions, pairing status and preregistered subsets.

This stage uses only the audited cell annotation and official GEO metadata
already carried into the inventories. It does not read post-treatment values
to select samples and does not run PhysiCell, an LLM, or scGPT.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


BROAD_TYPES = (
    "B cell",
    "Plasma cell",
    "Monocyte/Macrophage",
    "Dendritic cell",
    "T cell",
    "NK cell",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    samples = read_csv(args.audit_dir / "sample_inventory.csv")
    inventory = read_csv(args.audit_dir / "annotation_inventory.csv")
    sample_by_id = {row["sample_id"]: row for row in samples}
    inv_by_sample: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in inventory:
        inv_by_sample[row["sample_id"]].append(row)

    wide_rows: list[dict[str, object]] = []
    proportion_errors: list[dict[str, object]] = []
    for sample_id in sorted(sample_by_id):
        source = sample_by_id[sample_id]
        rows = inv_by_sample[sample_id]
        observed_types = {row["broad_cell_type"] for row in rows}
        denominator_values = {int(row["sample_treatment_denominator"]) for row in rows}
        proportions = {row["broad_cell_type"]: float(row["cell_type_proportion"]) for row in rows}
        total = sum(proportions.values())
        valid = (
            observed_types == set(BROAD_TYPES)
            and len(denominator_values) == 1
            and next(iter(denominator_values)) == int(source["n_cells"])
            and abs(total - 1.0) < 1e-6
        )
        if not valid:
            proportion_errors.append(
                {
                    "sample_id": sample_id,
                    "observed_types": sorted(observed_types),
                    "denominators": sorted(denominator_values),
                    "proportion_sum": total,
                }
            )
        wide_rows.append(
            {
                "sample_id": sample_id,
                "patient_id": source["patient_id"],
                "treatment_status": source["treatment_status"],
                "response": source["response"],
                "therapy": source["therapy"],
                "sample_treatment_denominator": int(source["n_cells"]),
                **{broad: f"{proportions[broad]:.12g}" for broad in BROAD_TYPES},
                "proportion_sum": f"{total:.12g}",
            }
        )
    if proportion_errors:
        raise RuntimeError(f"Sample-level proportion validation failed: {proportion_errors[:3]}")

    fields = [
        "sample_id", "patient_id", "treatment_status", "response", "therapy",
        "sample_treatment_denominator", *BROAD_TYPES, "proportion_sum",
    ]
    write_csv(
        args.output_dir / "pretreatment_sample_celltype_proportions.csv",
        fields,
        [row for row in wide_rows if row["treatment_status"] == "Pre"],
    )
    write_csv(
        args.output_dir / "posttreatment_sample_celltype_proportions.csv",
        fields,
        [row for row in wide_rows if row["treatment_status"] == "Post"],
    )

    by_patient: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in samples:
        by_patient[row["patient_id"]].append(row)
    pairing_rows: list[dict[str, object]] = []
    for patient_id in sorted(by_patient, key=lambda value: int(value[1:])):
        rows = by_patient[patient_id]
        pre = sorted(row["sample_id"] for row in rows if row["treatment_status"] == "Pre")
        post = sorted(row["sample_id"] for row in rows if row["treatment_status"] == "Post")
        pairing_rows.append(
            {
                "patient_id": patient_id,
                "pre_sample_ids": ";".join(pre),
                "post_sample_ids": ";".join(post),
                "has_pre": str(bool(pre)).lower(),
                "has_post": str(bool(post)).lower(),
                "paired_patient": str(bool(pre and post)).lower(),
                "n_pre_biopsies": len(pre),
                "n_post_biopsies": len(post),
                "response_values": ";".join(sorted({row["response"] for row in rows})),
                "therapy_values": ";".join(sorted({row["therapy"] for row in rows})),
            }
        )
    write_csv(
        args.output_dir / "sample_pairing_status.csv",
        [
            "patient_id", "pre_sample_ids", "post_sample_ids", "has_pre", "has_post",
            "paired_patient", "n_pre_biopsies", "n_post_biopsies",
            "response_values", "therapy_values",
        ],
        pairing_rows,
    )

    pre_by_response: dict[str, list[str]] = defaultdict(list)
    for row in samples:
        if row["treatment_status"] == "Pre":
            pre_by_response[row["response"]].append(row["sample_id"])
    for values in pre_by_response.values():
        values.sort()
    subset_rows: list[dict[str, object]] = []
    for response in ("Responder", "Non-responder"):
        ordered = pre_by_response[response]
        subset_rows.append(
            {
                "phase": "smoke",
                "response": response,
                "sample_id": ordered[0],
                "selection_rank": 1,
                "selection_rule": "lexicographic sample_id within pre-treatment response group",
            }
        )
        for rank, sample_id in enumerate(ordered[:3], 1):
            subset_rows.append(
                {
                    "phase": "quick_3seed",
                    "response": response,
                    "sample_id": sample_id,
                    "selection_rank": rank,
                    "selection_rule": "first 3 lexicographic sample_id within pre-treatment response group",
                }
            )
    write_csv(
        args.output_dir / "preregistered_sample_subset.csv",
        ["phase", "response", "sample_id", "selection_rank", "selection_rule"],
        subset_rows,
    )

    validation = {
        "status": "PASS_SAMPLE_TREATMENT_DENOMINATOR_VALIDATION",
        "sample_count": len(samples),
        "patient_count": len(by_patient),
        "pre_sample_count": sum(row["treatment_status"] == "Pre" for row in samples),
        "post_sample_count": sum(row["treatment_status"] == "Post" for row in samples),
        "paired_patient_count": sum(
            {row["treatment_status"] for row in rows} == {"Pre", "Post"}
            for rows in by_patient.values()
        ),
        "all_samples_have_six_broad_types": True,
        "all_sample_proportion_sums_within_1e_minus_6": True,
        "all_denominators_equal_sample_treatment_cell_count": True,
        "cross_treatment_denominator_used": False,
        "posttreatment_used_for_sample_selection": False,
        "smoke_samples": [row for row in subset_rows if row["phase"] == "smoke"],
        "quick_3seed_samples": [row for row in subset_rows if row["phase"] == "quick_3seed"],
        "pretreatment_expression_profiles_status": "not generated in this audit-only task; must be generated from pre-treatment cells after the scGPT gene panel is frozen",
    }
    (args.output_dir / "preprocessing_validation.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(validation, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
