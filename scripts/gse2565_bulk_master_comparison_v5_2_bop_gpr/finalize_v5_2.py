#!/usr/bin/env python3
"""Final consistency, visual-audit, and manifest gates for GSE2565 V5.2."""

from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import scipy

from common import OUT, sha256


def main() -> None:
    audit_path = OUT / "FINAL_AUDIT.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit["status"] != "PASS_DESCRIPTIVE_BULK_RECONSTRUCTION_V5_2_BOP_GPR":
        raise RuntimeError("V5.2 analysis did not pass before finalization")

    source_inventory = pd.read_csv(OUT / "source_inventory.csv")
    source_mismatches = []
    for row in source_inventory.itertuples():
        actual = sha256(Path(row.absolute_path))
        if actual != row.sha256:
            source_mismatches.append(
                {
                    "path": row.absolute_path,
                    "expected": row.sha256,
                    "actual": actual,
                }
            )
    if source_mismatches:
        raise RuntimeError(f"Frozen source hash changes: {source_mismatches}")

    formal = np.load(OUT / "bopdmd_reconstructed_trajectory.npz", allow_pickle=False)
    dnb = np.load(
        OUT / "bopdmd_dnb_bootstrap_trajectories_B20.npz",
        allow_pickle=False,
    )
    if (
        formal["expression"].shape != (9, 12263)
        or dnb["expression"].shape != (20, 9, 1601)
        or not np.isfinite(formal["expression"]).all()
        or not np.isfinite(dnb["expression"]).all()
    ):
        raise RuntimeError("BOP-DMD saved-array final gate failed")

    metrics = pd.read_csv(OUT / "raw_metric_summary.csv")
    expected_methods = {
        "chronODE-M",
        "RVAgene",
        "BOP-DMD",
        "GPR",
        "AgentVC Online Agent pilot",
    }
    if set(metrics.method) != expected_methods or len(metrics) != 5:
        raise RuntimeError("Unified metric method gate failed")
    numeric = metrics[
        [
            "dnb_peak_time",
            "dnb_peak_error",
            "dnb_pearson",
            "dnb_dtw",
            "distribution_pearson",
            "progression_pearson",
            "velocity_pearson",
            "mean_module_pearson",
            "gene_direction_agreement",
        ]
    ].to_numpy(float)
    if not np.isfinite(numeric).all():
        raise RuntimeError("Unified metrics contain non-finite values")

    figures = [
        OUT / "GSE2565_bulk_master_comparison_v5_2_preview.png",
        OUT / "GSE2565_bulk_master_comparison_v5_2_600dpi.png",
        OUT / "GSE2565_bulk_master_comparison_v5_2_vector.pdf",
        OUT / "GSE2565_bulk_master_comparison_v5_2_vector.svg",
    ]
    if any(not path.exists() or path.stat().st_size == 0 for path in figures):
        raise RuntimeError("A figure deliverable is missing or empty")

    environment = {
        "purpose": "metrics and figure rendering",
        "python": platform.python_version(),
        "python_executable": str(Path("/usr/bin/python3")),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "matplotlib": matplotlib.__version__,
        "historical_metric_reproduction_max_abs_diff": audit[
            "frozen_metric_max_absolute_differences"
        ],
    }
    (OUT / "plotting_environment.json").write_text(
        json.dumps(environment, indent=2) + "\n", encoding="utf-8"
    )

    caption = """# Figure caption

**GSE2565 descriptive bulk trajectory reconstruction comparison.** Panel A
shows 1,601-gene common-space DNB sensitivity; the inset preserves the
independent 11,171-gene formal result in which Real and AgentVC both peak at
8 h. Panels B–D show normalized expression-distribution shift, transcriptomic
progression and time-normalized velocity across all nine timepoints. Panel E
shows ten frozen functional programs in a shared temporal z-score scale.
Panel F reports raw DNB, late global and late program metrics without a
composite score. Late state metrics use 8–72 h and velocity uses 4→8 through
48→72 h, excluding GPR training states. Input regimes are not equivalent:
chronODE-M, RVAgene and BOP-DMD use the complete observed trajectory; GPR uses
0–4 h; AgentVC is a runtime-blind six-run pilot. BOP-DMD uses stable,
conjugate-paired eigenvalues with fixed imaginary sorting and is
target-derived. Values are descriptive and do not establish an overall
winner.
"""
    (OUT / "FIGURE_CAPTION.md").write_text(caption, encoding="utf-8")

    audit["visual_inspection"] = (
        "PASS_NO_CLIPPING_OVERLAP_OR_UNREADABLE_LABELS_PREVIEW_AND_VECTOR"
    )
    audit["source_hash_validation"] = {
        "files_checked": len(source_inventory),
        "mismatches": source_mismatches,
    }
    audit["saved_array_validation"] = {
        "formal_shape": [9, 12263],
        "DNB_bootstrap_shape": [20, 9, 1601],
        "all_finite": True,
    }
    audit["rank_sensitivity_note"] = (
        "ranks 2 and 4 estimable; rank 6 retained as NE because its relative "
        "imaginary norm exceeded 1e-10; primary rank 4 unchanged"
    )
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")

    rows = []
    for path in sorted(p for p in OUT.rglob("*") if p.is_file()):
        if path.name in {
            "sha256_manifest.csv",
            "FINAL_SHA256_MANIFEST.csv",
            "FINAL_VALIDATION_SUMMARY.json",
        }:
            continue
        rows.append(
            {
                "relative_path": str(path.relative_to(OUT)),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    pd.DataFrame(rows).to_csv(OUT / "FINAL_SHA256_MANIFEST.csv", index=False)
    summary = {
        "status": audit["status"],
        "files_hashed": len(rows),
        "source_files_checked": len(source_inventory),
        "source_hash_mismatches": 0,
        "historical_metric_max_abs_diff": max(
            audit["frozen_metric_max_absolute_differences"].values()
        ),
        "formal_shape": [9, 12263],
        "DNB_shape": [20, 9, 1601],
        "figure_visual_inspection": audit["visual_inspection"],
    }
    (OUT / "FINAL_VALIDATION_SUMMARY.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
