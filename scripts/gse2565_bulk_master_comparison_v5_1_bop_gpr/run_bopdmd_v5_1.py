#!/usr/bin/env python3
"""Run stable/conjugate BOP-DMD for GSE2565 V5.1.

This is a new protocol. It does not modify or resume the blocked unconstrained
V5. The main configuration is frozen before execution and never selected using
late-trajectory performance.
"""

from __future__ import annotations

import importlib.metadata
import json
import platform
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from pydmd import BOPDMD

from common import (
    BASE,
    BOP_BOOTSTRAPS,
    BOP_BOOTSTRAP_SEED_BASE,
    BOP_EIG_CONSTRAINTS,
    BOP_NUM_TRIALS,
    BOP_PRIMARY_SEED,
    BOP_RANK,
    BOP_RANK_SENSITIVITY,
    BOP_TRIAL_SIZE,
    BOP_VARPRO_TOL,
    CHRON_PARAMETERS,
    CHRON_PATH,
    COMMON_DNB,
    COMPLEX_TOLERANCE,
    DETERMINISM_TOLERANCE,
    ENV_DIR,
    EVAL,
    FORMAL_DNB_PATH,
    GENE_PATH,
    GPR_DNB_PATH,
    GPR_PATH,
    MODULE_PATH,
    OUT,
    REAL_PATH,
    RVA_PATH,
    RVA_SEEDS_PATH,
    TIMES,
    array_sha256,
    ensure_dirs,
    sha256,
)


def fit_bop(
    matrix_time_by_gene: np.ndarray,
    seed: int,
    rank: int = BOP_RANK,
) -> tuple[np.ndarray, dict]:
    matrix = np.asarray(matrix_time_by_gene, float)
    if matrix.ndim != 2 or matrix.shape[0] != len(TIMES):
        raise RuntimeError(f"BOP-DMD input shape invalid: {matrix.shape}")
    np.random.seed(seed)
    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        model = BOPDMD(
            svd_rank=rank,
            num_trials=BOP_NUM_TRIALS,
            trial_size=BOP_TRIAL_SIZE,
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
        f"DONE seed={seed} rank={rank} shape={matrix.shape} "
        f"seconds={metadata['runtime_seconds']:.3f}",
        flush=True,
    )
    return output, metadata


def input_inventory(inputs: dict) -> list[dict]:
    sources = [
        GENE_PATH,
        REAL_PATH,
        MODULE_PATH,
        CHRON_PATH,
        CHRON_PARAMETERS,
        RVA_PATH,
        RVA_SEEDS_PATH,
        GPR_PATH,
        GPR_DNB_PATH,
        FORMAL_DNB_PATH,
        COMMON_DNB / "panelA_unified_dnb_trajectories.csv",
        COMMON_DNB / "panelA_unified_dnb_metrics.csv",
        COMMON_DNB / "panelA_common_dnb_gene_set.txt",
        COMMON_DNB / "panelA_common_dnb_module_genes.txt",
        COMMON_DNB / "panelA_common_dnb_background_genes.txt",
        *inputs["agent_paths"],
    ]
    return [
        {
            "absolute_path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in sources
    ]


def freeze_protocol(inputs: dict) -> Path:
    inventory = input_inventory(inputs)
    payload = {
        "status": "FROZEN_BEFORE_STABLE_BOPDMD_EXECUTION",
        "version": "V5.1",
        "supersedes_blocked_protocol": (
            "V5 remains BLOCKED_BOPDMD_SMOKE_TIMEOUT and is not overwritten"
        ),
        "selection_rule": (
            "configuration selected only by smoke completion, finite real "
            "output, no warnings, and determinism; no benchmark metrics read"
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
            "varpro_tol": BOP_VARPRO_TOL,
            "seed": BOP_PRIMARY_SEED,
            "time_coordinate": "true hours",
            "rank_sensitivity": BOP_RANK_SENSITIVITY,
            "rank_sensitivity_note": (
                "even ranks only because conjugate-pair constraints are not "
                "well-defined for the requested odd ranks 3 and 5"
            ),
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
    path = OUT / "FROZEN_PROTOCOL_V5_1.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame(inventory).to_csv(OUT / "source_inventory.csv", index=False)
    return path


def environment_audit() -> None:
    freeze = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    (ENV_DIR / "pip_freeze.txt").write_text(freeze, encoding="utf-8")
    payload = {
        "python_executable": sys.executable,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "pydmd": importlib.metadata.version("pydmd"),
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    (ENV_DIR / "environment_audit.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )


def common_dnb_sets(inputs: dict) -> dict:
    common = [
        x
        for x in (COMMON_DNB / "panelA_common_dnb_gene_set.txt").read_text().splitlines()
        if x
    ]
    module = [
        x
        for x in (COMMON_DNB / "panelA_common_dnb_module_genes.txt").read_text().splitlines()
        if x
    ]
    background = [
        x
        for x in (COMMON_DNB / "panelA_common_dnb_background_genes.txt").read_text().splitlines()
        if x
    ]
    if (len(common), len(module), len(background)) != (1601, 86, 1515):
        raise RuntimeError("Frozen common DNB gene counts changed")
    position = {gene: index for index, gene in enumerate(inputs["genes"])}
    return {
        "genes": common,
        "module": module,
        "background": background,
        "indices": np.asarray([position[gene] for gene in common], int),
    }


def dnb_bootstrap(inputs: dict, sets: dict) -> tuple[Path, Path]:
    replicates = inputs["real_replicates"][:, :, sets["indices"]]
    trajectories, draw_rows, fit_rows = [], [], []
    for bootstrap in range(BOP_BOOTSTRAPS):
        seed = BOP_BOOTSTRAP_SEED_BASE + bootstrap
        rng = np.random.default_rng(seed)
        draws = rng.integers(0, 3, size=(len(TIMES), 3))
        sampled = np.vstack(
            [replicates[t, draws[t]].mean(axis=0) for t in range(len(TIMES))]
        )
        reconstruction, metadata = fit_bop(sampled, seed)
        trajectories.append(reconstruction)
        fit_rows.append({"bootstrap": bootstrap + 1, **metadata})
        for time_index, hour in enumerate(TIMES):
            draw_rows.append(
                {
                    "bootstrap": bootstrap + 1,
                    "seed": seed,
                    "time_hours": hour,
                    "draw_indices_zero_based": "|".join(map(str, draws[time_index])),
                }
            )
    trajectories = np.stack(trajectories)
    if trajectories.shape != (20, 9, 1601) or not np.isfinite(trajectories).all():
        raise RuntimeError("BLOCKED_BOPDMD_DNB_BOOTSTRAP_INVALID")
    raw = np.asarray(
        [
            EVAL.dnb_value(
                trajectories[:, time_index, :],
                sets["genes"],
                sets["module"],
                sets["background"],
            )
            for time_index in range(len(TIMES))
        ]
    )
    normalized = EVAL.mm(raw)
    if not np.isfinite(normalized).all():
        raise RuntimeError("BLOCKED_BOPDMD_DNB_NOT_ESTIMABLE")
    trajectory_path = OUT / "bopdmd_dnb_bootstrap_trajectories_B20.npz"
    np.savez_compressed(
        trajectory_path,
        expression=trajectories,
        genes=np.asarray(sets["genes"]),
        times=TIMES,
    )
    curve_path = OUT / "bopdmd_dnb_bootstrap_curve.csv"
    pd.DataFrame(
        {
            "time_hours": TIMES,
            "raw_DNB": raw,
            "normalized_DNB": normalized,
            "bootstrap_refits": BOP_BOOTSTRAPS,
            "common_DNB_genes": len(sets["genes"]),
        }
    ).to_csv(curve_path, index=False)
    pd.DataFrame(draw_rows).to_csv(
        OUT / "bopdmd_dnb_bootstrap_draws.csv", index=False
    )
    pd.DataFrame(fit_rows).to_json(
        OUT / "bopdmd_dnb_bootstrap_fit_audit.json",
        orient="records",
        indent=2,
    )
    return trajectory_path, curve_path


def main() -> None:
    ensure_dirs()
    environment_audit()
    inputs = BASE.load_frozen_inputs()
    protocol = freeze_protocol(inputs)

    smoke_outputs, smoke_meta = [], []
    for run in range(2):
        output, metadata = fit_bop(inputs["real"][:, :500], BOP_PRIMARY_SEED)
        smoke_outputs.append(output)
        smoke_meta.append(metadata)
    smoke_diff = float(np.max(np.abs(smoke_outputs[0] - smoke_outputs[1])))
    if (
        smoke_outputs[0].shape != (9, 500)
        or smoke_diff > DETERMINISM_TOLERANCE
        or any(item["warning_messages"] for item in smoke_meta)
    ):
        raise RuntimeError("BLOCKED_STABLE_BOPDMD_SMOKE")
    (OUT / "BOPDMD_SMOKE_AUDIT.json").write_text(
        json.dumps(
            {
                "status": "PASS",
                "shape": [9, 500],
                "finite": True,
                "determinism_max_abs_diff": smoke_diff,
                "runs": smoke_meta,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    formal_outputs, formal_meta = [], []
    for run in range(2):
        output, metadata = fit_bop(inputs["real"], BOP_PRIMARY_SEED)
        formal_outputs.append(output)
        formal_meta.append(metadata)
    formal_diff = float(np.max(np.abs(formal_outputs[0] - formal_outputs[1])))
    formal = formal_outputs[0]
    if (
        formal.shape != (9, 12263)
        or formal_diff > DETERMINISM_TOLERANCE
        or not np.isfinite(formal).all()
        or any(item["warning_messages"] for item in formal_meta)
    ):
        raise RuntimeError("BLOCKED_STABLE_BOPDMD_FORMAL")
    formal_path = OUT / "bopdmd_reconstructed_trajectory.npz"
    np.savez_compressed(
        formal_path,
        expression=formal,
        genes=np.asarray(inputs["genes"]),
        times=TIMES,
        observed_group_mean=inputs["real"],
    )
    pd.DataFrame(formal, columns=inputs["genes"]).assign(
        time_hours=TIMES
    )[["time_hours", *inputs["genes"]]].to_csv(
        OUT / "bopdmd_reconstructed_trajectory.csv.gz",
        index=False,
        compression="gzip",
    )

    sensitivity = []
    for rank in BOP_RANK_SENSITIVITY:
        try:
            reconstruction, metadata = fit_bop(
                inputs["real"], BOP_PRIMARY_SEED, rank
            )
            sensitivity.append(
                {
                    "rank": rank,
                    "is_primary_rank": rank == BOP_RANK,
                    "estimable": True,
                    "non_estimable_reason": "",
                    "native_rmse": float(
                        np.sqrt(np.mean((reconstruction - inputs["real"]) ** 2))
                    ),
                    "native_pearson": EVAL.safe_pearson(
                        inputs["real"].ravel(), reconstruction.ravel()
                    ),
                    "output_sha256": array_sha256(reconstruction),
                    **metadata,
                }
            )
        except RuntimeError as error:
            if rank == BOP_RANK:
                raise
            sensitivity.append(
                {
                    "rank": rank,
                    "is_primary_rank": False,
                    "estimable": False,
                    "non_estimable_reason": str(error),
                    "native_rmse": np.nan,
                    "native_pearson": np.nan,
                    "output_sha256": "",
                    "seed": BOP_PRIMARY_SEED,
                    "num_trials": BOP_NUM_TRIALS,
                    "trial_size": BOP_TRIAL_SIZE,
                    "eig_constraints": "|".join(sorted(BOP_EIG_CONSTRAINTS)),
                    "varpro_tol": BOP_VARPRO_TOL,
                }
            )
    pd.DataFrame(sensitivity).to_csv(
        OUT / "bopdmd_rank_sensitivity_even_ranks.csv", index=False
    )

    sets = common_dnb_sets(inputs)
    dnb_trajectory, dnb_curve = dnb_bootstrap(inputs, sets)
    audit = {
        "status": "PASS_BOPDMD_STABLE_CONJUGATE_V5_1",
        "protocol": str(protocol),
        "formal_shape": list(formal.shape),
        "formal_finite": bool(np.isfinite(formal).all()),
        "formal_determinism_max_abs_diff": formal_diff,
        "formal_expression_sha256": array_sha256(formal),
        "native_rmse": float(np.sqrt(np.mean((formal - inputs["real"]) ** 2))),
        "native_pearson": EVAL.safe_pearson(inputs["real"].ravel(), formal.ravel()),
        "negative_values": int(np.count_nonzero(formal < 0)),
        "negative_values_clipped": 0,
        "observed_values_refilled": False,
        "formal_runs": formal_meta,
        "formal_output": str(formal_path),
        "dnb_bootstrap_output": str(dnb_trajectory),
        "dnb_curve": str(dnb_curve),
    }
    (OUT / "BOPDMD_FORMAL_AUDIT.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
