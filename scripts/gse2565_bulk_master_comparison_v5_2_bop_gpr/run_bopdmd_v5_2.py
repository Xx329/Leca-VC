#!/usr/bin/env python3
"""Execute V5.2 by reusing V5.1 orchestration with fixed imaginary eig sorting."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
from pathlib import Path
import sys
import time
import warnings

import numpy as np
from pydmd import BOPDMD

from common import (
    BOP_EIG_CONSTRAINTS,
    BOP_EIG_SORT,
    BOP_NUM_TRIALS,
    BOP_PRIMARY_SEED,
    BOP_RANK,
    BOP_RANK_SENSITIVITY,
    BOP_TRIAL_SIZE,
    BOP_VARPRO_TOL,
    COMPLEX_TOLERANCE,
    OUT,
    TIMES,
    array_sha256,
)


ROOT = Path(__file__).resolve().parents[2]
SOURCE_PATH = (
    ROOT
    / "scripts/gse2565_bulk_master_comparison_v5_1_bop_gpr/run_bopdmd_v5_1.py"
)


def load_source():
    spec = importlib.util.spec_from_file_location("gse2565_v5_2_runner_base", SOURCE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {SOURCE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SOURCE = load_source()


def fit_bop(
    matrix_time_by_gene: np.ndarray,
    seed: int,
    rank: int = BOP_RANK,
) -> tuple[np.ndarray, dict]:
    matrix = np.asarray(matrix_time_by_gene, float)
    np.random.seed(seed)
    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        model = BOPDMD(
            svd_rank=rank,
            num_trials=BOP_NUM_TRIALS,
            trial_size=BOP_TRIAL_SIZE,
            eig_sort=BOP_EIG_SORT,
            eig_constraints=set(BOP_EIG_CONSTRAINTS),
            varpro_opts_dict={"tol": BOP_VARPRO_TOL},
        )
        model.fit(matrix.T, TIMES)
    reconstructed = np.asarray(model.reconstructed_data).T
    relative_imaginary_norm = float(
        np.linalg.norm(reconstructed.imag)
        / max(np.linalg.norm(reconstructed.real), np.finfo(float).eps)
    )
    if relative_imaginary_norm > COMPLEX_TOLERANCE:
        raise RuntimeError(
            "BLOCKED_BOPDMD_COMPLEX_OUTPUT: "
            f"relative imaginary norm {relative_imaginary_norm:.3e}"
        )
    output = reconstructed.real
    if output.shape != matrix.shape or not np.isfinite(output).all():
        raise RuntimeError("BLOCKED_BOPDMD_INVALID_RECONSTRUCTION")
    metadata = {
        "seed": seed,
        "rank": rank,
        "num_trials": BOP_NUM_TRIALS,
        "trial_size": BOP_TRIAL_SIZE,
        "eig_sort": BOP_EIG_SORT,
        "eig_constraints": sorted(BOP_EIG_CONSTRAINTS),
        "varpro_tol": BOP_VARPRO_TOL,
        "time_coordinate": "true hours",
        "runtime_seconds": time.perf_counter() - started,
        "relative_imaginary_norm": relative_imaginary_norm,
        "warning_messages": sorted({str(item.message) for item in captured}),
        "eigenvalues_real": np.asarray(model.eigs).real.tolist(),
        "eigenvalues_imag": np.asarray(model.eigs).imag.tolist(),
    }
    print(
        f"DONE V5.2 seed={seed} rank={rank} shape={matrix.shape} "
        f"seconds={metadata['runtime_seconds']:.3f}",
        flush=True,
    )
    return output, metadata


def freeze_protocol(inputs: dict) -> Path:
    inventory = SOURCE.input_inventory(inputs)
    payload = {
        "status": "FROZEN_BEFORE_STABLE_SORTED_BOPDMD_EXECUTION",
        "version": "V5.2",
        "supersedes_blocked_protocols": {
            "V5": "unconstrained smoke timeout",
            "V5.1": "auto eig sorting produced complex DNB bootstrap output",
        },
        "selection_rule": (
            "configuration selected only by smoke/bootstrap finite-real and "
            "determinism gates; no benchmark performance metrics read"
        ),
        "benchmark_semantics": "descriptive bulk trajectory reconstruction comparison",
        "equal_input_claim_allowed": False,
        "overall_winner_claim_allowed": False,
        "times_hours": TIMES.tolist(),
        "gene_count": len(inputs["genes"]),
        "gene_order_sha256": array_sha256(np.asarray(inputs["genes"], dtype="U")),
        "bopdmd": {
            "package": "pydmd",
            "version": importlib.metadata.version("pydmd"),
            "input": "9-timepoint Real group mean log2(signal+1)",
            "svd_rank": BOP_RANK,
            "num_trials": BOP_NUM_TRIALS,
            "trial_size": BOP_TRIAL_SIZE,
            "eig_constraints": sorted(BOP_EIG_CONSTRAINTS),
            "eig_sort": BOP_EIG_SORT,
            "varpro_tol": BOP_VARPRO_TOL,
            "seed": BOP_PRIMARY_SEED,
            "time_coordinate": "true hours",
            "rank_sensitivity": BOP_RANK_SENSITIVITY,
            "negative_clipping": False,
            "observed_refill": False,
        },
        "method_input_regimes": {
            "chronODE-M": "complete observed 0-72 h trajectory",
            "RVAgene": "complete observed 0-72 h trajectory",
            "BOP-DMD": "complete observed 0-72 h trajectory",
            "GPR": "observed 0-4 h only",
            "AgentVC Online Agent pilot": "no observed expression at runtime",
        },
        "late_metric_scope": {
            "states_hours": [8, 12, 24, 48, 72],
            "velocity_intervals": [
                "4->8",
                "8->12",
                "12->24",
                "24->48",
                "48->72",
            ],
        },
        "source_inventory": inventory,
    }
    path = OUT / "FROZEN_PROTOCOL_V5_2.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    SOURCE.pd.DataFrame(inventory).to_csv(OUT / "source_inventory.csv", index=False)
    return path


def main() -> None:
    SOURCE.fit_bop = fit_bop
    SOURCE.freeze_protocol = freeze_protocol
    SOURCE.main()
    audit_path = OUT / "BOPDMD_FORMAL_AUDIT.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["status"] = "PASS_BOPDMD_STABLE_CONJUGATE_SORTED_V5_2"
    audit["version"] = "V5.2"
    audit["eig_sort"] = BOP_EIG_SORT
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
