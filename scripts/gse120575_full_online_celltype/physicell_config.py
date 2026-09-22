#!/usr/bin/env python3
"""Freeze 600-worker discretization and deterministic synthetic registries."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


BROAD_TYPES = (
    "B cell", "Plasma cell", "Monocyte/Macrophage",
    "Dendritic cell", "T cell", "NK cell",
)
SEEDS = (12057501, 12057502, 12057503)
WORKERS = 600
MAX_ERROR_THRESHOLD = 0.002


def allocate(proportions: np.ndarray) -> np.ndarray:
    raw = proportions * WORKERS
    counts = np.floor(raw).astype(int)
    nonzero = proportions > 0
    counts[(nonzero) & (counts == 0)] = 1
    while counts.sum() > WORKERS:
        candidates = np.flatnonzero(counts > 1)
        if not len(candidates):
            raise RuntimeError("Cannot preserve nonzero cell types within worker budget")
        surplus = counts[candidates] - raw[candidates]
        counts[candidates[np.argmax(surplus)]] -= 1
    while counts.sum() < WORKERS:
        deficit = raw - counts
        counts[int(np.argmax(deficit))] += 1
    return counts


def stable_rng(sample_id: str, seed: int) -> np.random.Generator:
    digest = hashlib.sha256(f"{sample_id}|{seed}|synthetic_well_mixed_v1".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--proportions", type=Path, required=True)
    p.add_argument("--subset", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    a = p.parse_args()
    audit_dir = a.output_root / "audit"
    registry_dir = a.output_root / "preprocessed" / "worker_registries"
    audit_dir.mkdir(parents=True, exist_ok=True)
    registry_dir.mkdir(parents=True, exist_ok=True)

    prereg = {
        "status": "PREREGISTERED_BEFORE_DISCRETIZATION_AUDIT",
        "worker_count": WORKERS,
        "allocation": "floor plus minimum one for every nonzero type, then deterministic largest deficit/surplus correction",
        "maximum_absolute_initialization_error_threshold": MAX_ERROR_THRESHOLD,
        "stop_if_any_nonzero_type_dropped_to_zero": True,
        "seeds": list(SEEDS),
        "position_initialization": "uniform disk radius 900; RNG from SHA256(sample_id|seed|synthetic_well_mixed_v1)",
    }
    prereg_path = audit_dir / "worker_discretization_preregistration.json"
    if not prereg_path.exists():
        prereg_path.write_text(json.dumps(prereg, indent=2) + "\n")
    elif json.loads(prereg_path.read_text()) != prereg:
        raise RuntimeError("Worker discretization preregistration changed")

    with a.proportions.open(newline="", encoding="utf-8") as f:
        rows = {row["sample_id"]: row for row in csv.DictReader(f)}
    with a.subset.open(newline="", encoding="utf-8") as f:
        selected = sorted({row["sample_id"] for row in csv.DictReader(f)})
    audit_rows = []
    allocations: dict[str, np.ndarray] = {}
    for sample_id in selected:
        source = rows[sample_id]
        target = np.asarray([float(source[cell_type]) for cell_type in BROAD_TYPES])
        counts = allocate(target)
        allocations[sample_id] = counts
        initialized = counts / WORKERS
        for cell_type, target_p, count, init_p in zip(BROAD_TYPES, target, counts, initialized):
            audit_rows.append({
                "sample_id": sample_id,
                "broad_cell_type": cell_type,
                "target_proportion": f"{target_p:.12g}",
                "assigned_worker_count": int(count),
                "initialized_proportion": f"{init_p:.12g}",
                "absolute_initialization_error": f"{abs(init_p - target_p):.12g}",
                "nonzero_type_dropped_to_zero": str(target_p > 0 and count == 0).lower(),
            })
    audit_path = audit_dir / "worker_discretization_audit.csv"
    with audit_path.open("w", newline="", encoding="utf-8") as f:
        fields = list(audit_rows[0])
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(audit_rows)

    for sample_id, counts in allocations.items():
        for seed in SEEDS:
            rng = stable_rng(sample_id, seed)
            radius = 900.0 * np.sqrt(rng.random(WORKERS))
            angle = 2 * np.pi * rng.random(WORKERS)
            types = np.concatenate([
                np.repeat(cell_type, count) for cell_type, count in zip(BROAD_TYPES, counts)
            ])
            permutation = rng.permutation(WORKERS)
            path = registry_dir / f"{sample_id}__seed_{seed}.csv"
            with path.open("w", newline="", encoding="utf-8") as f:
                fields = ["worker_id", "cell_type", "x", "y", "z", "initial_represented_abundance"]
                w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
                for worker_id, index in enumerate(permutation):
                    w.writerow({
                        "worker_id": f"worker_{worker_id:04d}",
                        "cell_type": str(types[index]),
                        "x": f"{radius[index] * np.cos(angle[index]):.12g}",
                        "y": f"{radius[index] * np.sin(angle[index]):.12g}",
                        "z": "0",
                        "initial_represented_abundance": f"{1 / WORKERS:.12g}",
                    })

    max_error = max(float(row["absolute_initialization_error"]) for row in audit_rows)
    dropped = [row for row in audit_rows if row["nonzero_type_dropped_to_zero"] == "true"]
    passed = not dropped and max_error <= MAX_ERROR_THRESHOLD
    summary = {
        "status": "PASS_600_WORKER_DISCRETIZATION" if passed else "FAIL_600_WORKER_DISCRETIZATION",
        "all_gates_passed": passed,
        "samples": selected,
        "worker_count": WORKERS,
        "maximum_absolute_initialization_error": max_error,
        "preregistered_threshold": MAX_ERROR_THRESHOLD,
        "nonzero_types_dropped_to_zero": dropped,
        "registry_count": len(selected) * len(SEEDS),
        "same_sample_seed_registry_shared_by_all_models": True,
        "synthetic_spatial_initialization": True,
        "real_spatial_reconstruction_claimed": False,
    }
    (audit_dir / "worker_discretization_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
