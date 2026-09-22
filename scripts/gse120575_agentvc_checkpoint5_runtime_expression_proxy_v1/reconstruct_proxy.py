#!/usr/bin/env python3
"""Reconstruct and audit the frozen AgentVC V2.1 checkpoint-5 expression proxy.

This is an offline deterministic reconstruction. It does not call an LLM,
PhysiCell, scGPT, scGen, WOT, or any other model.
"""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import os
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(os.environ.get("GSE120575_PROXY_OUT", str(ROOT / "outputs/GSE120575_agentvc_checkpoint5_runtime_expression_proxy_v1"))).resolve()
FEASIBILITY = ROOT / "outputs/GSE120575_external_expression_benchmark_feasibility_v1"
COMMON_AUDIT = FEASIBILITY / "common_gene_audit.json"
FORMULA_FILE = ROOT / "scripts/gse120575_full_online_celltype_v21/online_inference_v21.py"
MODULES_FILE = ROOT / "scripts/gse120575_full_online_celltype/expression_profiles.py"
PROFILES = ROOT / (
    "outputs/GSE120575_full_online_celltype/preprocessed/"
    "pretreatment_celltype_expression_profiles.npz"
)
RUNS = Path(os.environ.get(
    "GSE120575_RUNS_ROOT",
    str(ROOT / "outputs/GSE120575_full_online_celltype_v21_agent_rerun/unified_18_agent_physicell_v21/runs"),
)).resolve()
ANNOTATION = ROOT / "outputs/GSE120575_full_online_celltype/annotation/cell_level_annotations.csv"
OBSERVED = ROOT / (
    "data/GSE120575_Sade_Feldman_melanoma_single_cells_TPM_GEO(1).txt/"
    "GSE120575_Sade_Feldman_melanoma_single_cells_TPM_GEO.txt"
)
SCGEN = ROOT / (
    "data/wot_scgen_cellrank_native_pre_post_reports/"
    "scgen_gse120575_pre_to_post/adapter/generated_post_from_pre.h5ad"
)
WOT = ROOT / (
    "data/wot_scgen_cellrank_native_pre_post_reports/"
    "wot_gse120575_pre_to_post_native/wot_pre_to_T1_transport_status_all_genes.h5ad"
)

SAMPLES = ("Pre_P24", "Pre_P29", "Pre_P35", "Pre_P2", "Pre_P3", "Pre_P27")
SEEDS = (12057501, 12057502, 12057503)
CELL_TYPES = (
    "B cell", "Plasma cell", "Monocyte/Macrophage",
    "Dendritic cell", "T cell", "NK cell",
)
RESPONSES = {
    "Pre_P24": "Responder", "Pre_P29": "Responder", "Pre_P35": "Responder",
    "Pre_P2": "Non-responder", "Pre_P3": "Non-responder", "Pre_P27": "Non-responder",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def frozen_modules() -> dict[str, tuple[str, ...]]:
    """Read the exact MODULES literal used by the frozen runtime formula."""
    source = MODULES_FILE.read_text(encoding="utf-8")
    tree = ast.parse(source)
    assignments = [
        node for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "MODULES" for target in node.targets)
    ]
    if len(assignments) != 1:
        raise RuntimeError("Could not isolate one frozen MODULES assignment")
    value = ast.literal_eval(assignments[0].value)
    if not isinstance(value, dict):
        raise RuntimeError("Frozen MODULES is not a dictionary")
    return value


def frozen_formula() -> tuple[object, str]:
    """Compile only the original expression_for_checkpoint AST unchanged."""
    source = FORMULA_FILE.read_text(encoding="utf-8")
    tree = ast.parse(source)
    nodes = [
        node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "expression_for_checkpoint"
    ]
    if len(nodes) != 1:
        raise RuntimeError("Could not isolate one frozen expression_for_checkpoint function")
    node = nodes[0]
    segment = ast.get_source_segment(source, node)
    if segment is None:
        raise RuntimeError("Could not recover exact formula source")
    module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    namespace: dict[str, object] = {"np": np, "MODULES": frozen_modules()}
    exec(compile(module, str(FORMULA_FILE), "exec"), namespace)
    return namespace["expression_for_checkpoint"], segment


def read_checkpoint_state(path: Path) -> dict[str, dict[str, object]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != len(CELL_TYPES) or {row["cell_type"] for row in rows} != set(CELL_TYPES):
        raise RuntimeError(f"Unexpected checkpoint state: {path}")
    numeric = (
        "live_cells", "dead_cells", "cumulative_baseline_divisions",
        "cumulative_actuator_divisions", "cumulative_baseline_apoptosis",
        "cumulative_actuator_apoptosis", "cumulative_actuator_recruitment",
        "cumulative_actuator_clearance", "mean_birth_rate", "mean_death_rate",
        "treatment_signal", "inflammatory_signal", "suppressive_signal",
        "survival_signal", "stress_signal",
    )
    return {
        str(row["cell_type"]): {
            **row,
            **{field: float(row[field]) for field in numeric},
        }
        for row in rows
    }


def read_endpoint_weights(path: Path) -> dict[str, float]:
    frame = pd.read_csv(path)
    frame = frame[frame["checkpoint"].eq(5)].copy()
    if len(frame) != len(CELL_TYPES) or set(frame["cell_type"]) != set(CELL_TYPES):
        raise RuntimeError(f"Unexpected endpoint composition: {path}")
    weights = frame.set_index("cell_type")["proportion"].astype(float).to_dict()
    if not np.isclose(sum(weights.values()), 1.0, atol=1e-9):
        raise RuntimeError(f"Endpoint composition does not sum to one: {path}")
    return weights


def reconstruct_all(
    formula: object,
    frozen_genes: list[str],
    profile_data: np.lib.npyio.NpzFile,
) -> tuple[np.ndarray, list[dict[str, object]], dict[str, str]]:
    panel_genes = profile_data["genes"].astype(str).tolist()
    gene_index = {gene: index for index, gene in enumerate(panel_genes)}
    frozen_indices = np.asarray([gene_index[gene] for gene in frozen_genes], dtype=int)
    profiles = profile_data["profiles"].astype(np.float32)
    profile_samples = profile_data["sample_ids"].astype(str)
    profile_types = profile_data["broad_cell_types"].astype(str)

    vectors: list[np.ndarray] = []
    records: list[dict[str, object]] = []
    input_hashes: dict[str, str] = {}
    for sample in SAMPLES:
        baseline_by_type: dict[str, np.ndarray] = {}
        for cell_type in CELL_TYPES:
            indexes = np.flatnonzero(
                (profile_samples == sample) & (profile_types == cell_type)
            )
            if len(indexes) != 1:
                raise RuntimeError(f"Missing unique profile: {sample}/{cell_type}")
            baseline_by_type[cell_type] = profiles[indexes[0]]

        for seed in SEEDS:
            run_id = f"agent_v21_unified__{sample}__seed_{seed}"
            run_dir = RUNS / run_id
            if not (run_dir / "RUN_COMPLETE.json").is_file():
                raise RuntimeError(f"Incomplete historical run: {run_id}")
            manifest = json.loads((run_dir / "run_manifest.json").read_text())
            if (
                manifest["sample_id"] != sample
                or int(manifest["seed"]) != seed
                or manifest["model"] != "agent_only"
                or manifest["therapy"] != "anti-PD1"
                or manifest["response"] != RESPONSES[sample]
            ):
                raise RuntimeError(f"Historical run manifest mismatch: {run_id}")

            state = read_checkpoint_state(run_dir / "state_checkpoint_5.csv")
            weights = read_endpoint_weights(run_dir / "composition_trajectory.csv")
            history = json.loads((run_dir / "runtime_history.json").read_text())
            cell_type_vectors: dict[str, np.ndarray] = {}
            for cell_type in CELL_TYPES:
                entries = history.get(cell_type, [])
                if len(entries) != 5 or int(entries[-1]["checkpoint"]) != 4:
                    raise RuntimeError(f"Missing checkpoint-4 action history: {run_id}/{cell_type}")
                previous_action = entries[-1]["executed_action"]
                endpoint = formula(
                    baseline_by_type[cell_type],
                    panel_genes,
                    state[cell_type],
                    5,
                    previous_action,
                )
                endpoint = np.asarray(endpoint)
                if endpoint.shape != (len(panel_genes),):
                    raise RuntimeError(f"Bad endpoint profile shape: {run_id}/{cell_type}")
                cell_type_vectors[cell_type] = endpoint

            pseudobulk = sum(
                weights[cell_type] * cell_type_vectors[cell_type].astype(np.float64)
                for cell_type in CELL_TYPES
            )
            selected = np.asarray(pseudobulk[frozen_indices], dtype=np.float64)
            vectors.append(selected)
            records.append(
                {
                    "run_id": run_id,
                    "sample_id": sample,
                    "response": RESPONSES[sample],
                    "therapy": "anti-PD1",
                    "seed": seed,
                    "checkpoint": 5,
                    "gene_count": len(selected),
                    "endpoint_composition_sum": float(sum(weights.values())),
                    "proxy_scale": "TPM-like before three-seed averaging and log1p",
                    "vector_sha256": sha256_bytes(selected.tobytes()),
                }
            )
            input_hashes[run_id] = sha256_bytes(
                (run_dir / "state_checkpoint_5.csv").read_bytes()
                + (run_dir / "composition_trajectory.csv").read_bytes()
                + (run_dir / "runtime_history.json").read_bytes()
            )
    return np.vstack(vectors), records, input_hashes


def observed_harmonized(
    genes: list[str],
    annotation: pd.DataFrame,
) -> tuple[np.ndarray, list[str], list[str]]:
    by_cell = annotation.set_index("cell_id")
    post = annotation[
        annotation["patient_sample"].astype(str).str.startswith("Post_")
        & annotation["therapy"].eq("anti-PD1")
    ].copy()
    biopsy_ids = sorted(post["patient_sample"].unique())
    biopsy_index = {sample: index for index, sample in enumerate(biopsy_ids)}

    with OBSERVED.open("rb") as handle:
        header = handle.readline().rstrip(b"\r\n").decode("utf-8").split("\t")[1:]
    selected_positions = []
    selected_groups = []
    for position, cell_id in enumerate(header):
        if cell_id not in by_cell.index:
            raise RuntimeError(f"Observed cell absent from frozen annotation: {cell_id}")
        row = by_cell.loc[cell_id]
        sample = str(row["patient_sample"])
        if sample in biopsy_index and row["therapy"] == "anti-PD1":
            selected_positions.append(position)
            selected_groups.append(biopsy_index[sample])
    positions = np.asarray(selected_positions, dtype=int)
    groups = np.asarray(selected_groups, dtype=int)
    counts = np.bincount(groups, minlength=len(biopsy_ids))
    if np.any(counts == 0):
        raise RuntimeError("Observed biopsy with zero cells")

    wanted = set(genes)
    matrix_by_gene: dict[str, np.ndarray] = {}
    with OBSERVED.open("r", encoding="utf-8", newline="") as handle:
        next(handle)
        next(handle)
        for line in handle:
            gene, separator, rest = line.partition("\t")
            if not separator or gene not in wanted:
                continue
            values = rest.rstrip("\r\n").split("\t")
            if len(values) == len(header) + 1 and values[-1] == "":
                values.pop()
            vector = np.asarray(values, dtype=np.float64)[positions]
            sums = np.bincount(groups, weights=vector, minlength=len(biopsy_ids))
            matrix_by_gene[gene] = sums / counts
            if len(matrix_by_gene) == len(genes):
                break
    if set(matrix_by_gene) != wanted:
        raise RuntimeError("Observed expression is missing frozen genes")
    tpm = np.column_stack([matrix_by_gene[gene] for gene in genes])
    responses = [
        str(post[post["patient_sample"].eq(sample)]["response"].iloc[0])
        for sample in biopsy_ids
    ]
    return np.log1p(tpm), biopsy_ids, responses


def external_harmonized(
    path: Path,
    genes: list[str],
    method: str,
) -> tuple[np.ndarray, np.ndarray | None]:
    data = ad.read_h5ad(path, backed="r")
    mask = np.asarray(data.obs["sample_label"].astype(str).isin(SAMPLES))
    obs = data.obs.loc[mask].copy()
    gene_positions = data.var_names.get_indexer(genes)
    if np.any(gene_positions < 0):
        raise RuntimeError(f"{method} is missing frozen genes")
    sorted_order = np.argsort(gene_positions)
    sorted_genes = [genes[index] for index in sorted_order]
    inverse_order = np.argsort(sorted_order)
    # A backed HDF5 dataset permits only one fancy index at a time. Load the
    # sorted frozen columns first, then perform the row and column reordering
    # in memory. This does not alter values.
    loaded = data[:, sorted_genes].X
    if hasattr(loaded, "toarray"):
        loaded = loaded.toarray()
    matrix = np.asarray(loaded, dtype=np.float64)[mask][:, inverse_order]
    if method == "scGen":
        linear = np.expm1(np.maximum(matrix, 0.0))
    elif method == "WOT":
        if np.any(matrix < 0):
            raise RuntimeError("WOT projected TPM contains negative values")
        linear = matrix
    else:
        raise RuntimeError(method)

    weighted_rows = []
    unweighted_rows = []
    for sample in SAMPLES:
        sample_mask = obs["sample_label"].astype(str).to_numpy() == sample
        if not np.any(sample_mask):
            raise RuntimeError(f"{method} missing {sample}")
        selected = linear[sample_mask]
        unweighted_rows.append(np.log1p(selected.mean(axis=0)))
        if method == "WOT":
            weights = obs.loc[sample_mask, "wot_row_mass"].astype(float).to_numpy()
            if not np.all(weights > 0) or not np.isfinite(weights).all():
                raise RuntimeError(f"Invalid WOT row mass: {sample}")
            weighted_rows.append(np.log1p(np.average(selected, axis=0, weights=weights)))
    data.file.close()
    unweighted = np.vstack(unweighted_rows)
    if method == "WOT":
        return np.vstack(weighted_rows), unweighted
    return unweighted, None


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    completed_names = {
        "scale_harmonization_audit.json",
        "PROXY_RECONSTRUCTION_SUMMARY.md",
    }
    existing = completed_names.intersection(path.name for path in OUT.iterdir())
    if existing:
        raise RuntimeError(f"Refusing to overwrite existing reconstruction: {sorted(existing)}")

    prior = json.loads(COMMON_AUDIT.read_text())
    genes = list(prior["conditional_gene_list"])
    if len(genes) != 834 or len(set(genes)) != 834 or genes != sorted(genes):
        raise RuntimeError("Prior frozen candidate gene list is not exactly 834 sorted unique genes")
    panel_path = OUT / "frozen_834_gene_panel.txt"
    expected_panel = "".join(f"{gene}\n" for gene in genes)
    if panel_path.exists():
        if panel_path.read_text(encoding="utf-8") != expected_panel:
            raise RuntimeError("Existing partial frozen gene panel differs; refusing resume")
    else:
        panel_path.write_text(expected_panel, encoding="utf-8")
    panel_hash = sha256(panel_path)
    gene_hash_record = {
        "status": "FROZEN_EXACT_834_GENE_PANEL",
        "gene_count": len(genes),
        "ordering": "lexicographic order inherited unchanged from feasibility V1 conditional_gene_list",
        "matching": "case-sensitive exact symbols; no aliases, fuzzy matching, or result-based filtering",
        "sha256": panel_hash,
        "source_audit": str(COMMON_AUDIT.resolve()),
        "source_audit_sha256": sha256(COMMON_AUDIT),
    }
    gene_hash_path = OUT / "gene_panel_hash.json"
    if gene_hash_path.exists():
        if json.loads(gene_hash_path.read_text()) != gene_hash_record:
            raise RuntimeError("Existing partial gene-panel audit differs; refusing resume")
    else:
        write_json(gene_hash_path, gene_hash_record)

    formula, formula_source = frozen_formula()
    formula_hash = sha256_bytes(formula_source.encode("utf-8"))
    spec = {
        "status": "FROZEN_BEFORE_PROXY_RECONSTRUCTION",
        "version": "GSE120575_AGENTVC_CHECKPOINT5_RUNTIME_EXPRESSION_PROXY_V1",
        "model_rerun": False,
        "llm_calls": 0,
        "physicell_runs": 0,
        "scgpt_calls": 0,
        "external_model_runs": 0,
        "samples": list(SAMPLES),
        "seeds": list(SEEDS),
        "checkpoint": 5,
        "gene_panel": str(panel_path.resolve()),
        "gene_panel_sha256": panel_hash,
        "formula_file": str(FORMULA_FILE.resolve()),
        "formula_file_sha256": sha256(FORMULA_FILE),
        "expression_for_checkpoint_source_sha256": formula_hash,
        "formula_constraints": [
            "use original expression_for_checkpoint AST unchanged",
            "use pre-only sample×broad-cell-type TPM profiles",
            "use saved checkpoint-5 PhysiCell state",
            "use saved checkpoint-4 executed action/history",
            "use real PhysiCell checkpoint-5 composition weights",
            "reconstruct every sample×seed independently",
        ],
        "sample_level_scale": (
            "cell-type TPM-like proxy weighted by endpoint composition; average three seeds "
            "within source sample; then log1p"
        ),
        "forbidden": [
            "Observed Post values in proxy reconstruction",
            "scGen or WOT expression values in proxy reconstruction",
            "metric-driven formula changes",
            "gene deletion based on performance",
            "claiming proxy was directly serialized by the original run",
            "claiming proxy is a PhysiCell-executed gene-expression field",
        ],
    }
    spec_path = OUT / "agentvc_proxy_reconstruction_spec.json"
    if spec_path.exists():
        if json.loads(spec_path.read_text()) != spec:
            raise RuntimeError("Existing partial reconstruction spec differs; refusing resume")
    else:
        write_json(spec_path, spec)

    profile_data = np.load(PROFILES)
    first, records, input_hashes = reconstruct_all(formula, genes, profile_data)
    second, second_records, second_input_hashes = reconstruct_all(formula, genes, profile_data)
    deterministic_bytes = first.tobytes() == second.tobytes()
    if not deterministic_bytes or records != second_records or input_hashes != second_input_hashes:
        raise RuntimeError("Repeated proxy reconstruction was not deterministic")
    if first.shape != (len(SAMPLES) * len(SEEDS), len(genes)):
        raise RuntimeError(f"Unexpected reconstructed matrix shape: {first.shape}")
    if not np.isfinite(first).all() or np.any(first < 0):
        raise RuntimeError("Reconstructed proxy contains invalid values")

    sample_ids = np.asarray([row["sample_id"] for row in records])
    seeds = np.asarray([row["seed"] for row in records], dtype=np.int64)
    responses = np.asarray([row["response"] for row in records])
    npz_path = OUT / "agentvc_checkpoint5_proxy_by_sample_seed.npz"
    npz_values = {
        "genes": np.asarray(genes),
        "sample_ids": sample_ids,
        "responses": responses,
        "therapy": np.asarray(["anti-PD1"] * len(records)),
        "seeds": seeds,
        "checkpoint": np.asarray([5] * len(records), dtype=np.int64),
        "proxy_tpm_like": first,
    }
    if npz_path.exists():
        with np.load(npz_path) as existing_npz:
            if set(existing_npz.files) != set(npz_values) or any(
                not (
                    np.array_equal(existing_npz[key], value, equal_nan=True)
                    if np.issubdtype(value.dtype, np.number)
                    else np.array_equal(existing_npz[key], value)
                )
                for key, value in npz_values.items()
            ):
                raise RuntimeError("Existing partial sample-seed proxy differs; refusing resume")
    else:
        np.savez_compressed(npz_path, **npz_values)

    sample_tpm = np.vstack([
        first[sample_ids == sample].mean(axis=0)
        for sample in SAMPLES
    ])
    sample_log1p = np.log1p(sample_tpm)
    sample_frame = pd.DataFrame(sample_log1p, columns=genes)
    sample_frame.insert(0, "therapy", "anti-PD1")
    sample_frame.insert(0, "response", [RESPONSES[sample] for sample in SAMPLES])
    sample_frame.insert(0, "sample_id", SAMPLES)
    sample_frame.insert(3, "seed_aggregation", "mean of three TPM-like seed proxies before log1p")
    sample_csv = OUT / "agentvc_checkpoint5_proxy_by_sample.csv"
    if sample_csv.exists():
        existing_frame = pd.read_csv(sample_csv)
        pd.testing.assert_frame_equal(
            existing_frame,
            sample_frame,
            check_exact=False,
            rtol=1e-14,
            atol=1e-14,
        )
    else:
        sample_frame.to_csv(sample_csv, index=False)

    reconstruction_audit = {
        "status": "PASS_AGENTVC_CHECKPOINT5_PROXY_RECONSTRUCTION",
        "interpretation": (
            "offline deterministic runtime-expression proxy reconstructed from frozen V2.1 inputs; "
            "not directly saved by the original run and not a PhysiCell gene-expression field"
        ),
        "checks": {
            "six_samples": set(sample_ids) == set(SAMPLES),
            "three_seeds_per_sample": all(
                set(seeds[sample_ids == sample].tolist()) == set(SEEDS) for sample in SAMPLES
            ),
            "all_18_sample_seed_pairs": len(records) == 18,
            "each_vector_exactly_834_genes": first.shape[1] == 834,
            "no_nan": not np.isnan(first).any(),
            "no_inf": not np.isinf(first).any(),
            "no_negative": not np.any(first < 0),
            "repeat_reconstruction_byte_identical": deterministic_bytes,
            "all_endpoint_composition_sums_one": all(
                np.isclose(row["endpoint_composition_sum"], 1.0, atol=1e-9) for row in records
            ),
            "three_seed_mean_before_log1p": True,
            "no_observed_scgen_wot_values_used_in_proxy": True,
            "original_formula_AST_used_unchanged": True,
        },
        "matrix_shape_sample_seed_by_gene": list(first.shape),
        "sample_matrix_shape_sample_by_gene": list(sample_log1p.shape),
        "records": records,
        "input_bundle_sha256_by_run": input_hashes,
        "hashes": {
            "gene_panel": panel_hash,
            "formula_file": sha256(FORMULA_FILE),
            "expression_for_checkpoint_source": formula_hash,
            "modules_definition_file": sha256(MODULES_FILE),
            "pre_only_profiles": sha256(PROFILES),
            "sample_seed_npz": sha256(npz_path),
            "sample_csv": sha256(sample_csv),
            "sample_seed_matrix_bytes": sha256_bytes(first.tobytes()),
            "sample_log1p_matrix_bytes": sha256_bytes(sample_log1p.tobytes()),
        },
    }
    if not all(reconstruction_audit["checks"].values()):
        raise RuntimeError("Proxy reconstruction audit failed")
    reconstruction_audit_path = OUT / "reconstruction_audit.json"
    if reconstruction_audit_path.exists():
        if json.loads(reconstruction_audit_path.read_text()) != reconstruction_audit:
            raise RuntimeError("Existing partial reconstruction audit differs; refusing resume")
    else:
        write_json(reconstruction_audit_path, reconstruction_audit)

    # External values are read only after the AgentVC reconstruction is frozen.
    annotation = pd.read_csv(ANNOTATION)
    observed_values, observed_ids, observed_responses = observed_harmonized(genes, annotation)
    scgen_values, _ = external_harmonized(SCGEN, genes, "scGen")
    wot_weighted, wot_unweighted = external_harmonized(WOT, genes, "WOT")
    assert wot_unweighted is not None

    methods: list[str] = []
    harmonized_ids: list[str] = []
    harmonized_responses: list[str] = []
    roles: list[str] = []
    matrices: list[np.ndarray] = []

    methods.extend(["Observed_Post"] * len(observed_ids))
    harmonized_ids.extend(observed_ids)
    harmonized_responses.extend(observed_responses)
    roles.extend(["evaluation_reference"] * len(observed_ids))
    matrices.append(observed_values)
    for method, values, role in (
        ("scGen", scgen_values, "generated_expression_baseline"),
        ("WOT_weighted", wot_weighted, "primary_target_derived_transport_baseline"),
        ("WOT_unweighted", wot_unweighted, "sensitivity_target_derived_transport_baseline"),
        ("AgentVC_proxy", sample_log1p, "offline_reconstructed_runtime_expression_proxy"),
    ):
        methods.extend([method] * len(SAMPLES))
        harmonized_ids.extend(SAMPLES)
        harmonized_responses.extend([RESPONSES[sample] for sample in SAMPLES])
        roles.extend([role] * len(SAMPLES))
        matrices.append(values)
    harmonized = np.vstack(matrices)
    harmonized_npz = OUT / "harmonized_834_expression_by_method_sample.npz"
    np.savez_compressed(
        harmonized_npz,
        genes=np.asarray(genes),
        methods=np.asarray(methods),
        sample_ids=np.asarray(harmonized_ids),
        responses=np.asarray(harmonized_responses),
        therapy=np.asarray(["anti-PD1"] * len(methods)),
        roles=np.asarray(roles),
        expression_log1p_sample_pseudobulk=harmonized,
    )
    harmonized_frame = pd.DataFrame(harmonized, columns=genes)
    harmonized_frame.insert(0, "role", roles)
    harmonized_frame.insert(0, "therapy", "anti-PD1")
    harmonized_frame.insert(0, "response", harmonized_responses)
    harmonized_frame.insert(0, "sample_id", harmonized_ids)
    harmonized_frame.insert(0, "method", methods)
    harmonized_csv = OUT / "harmonized_834_expression_by_method_sample.csv"
    harmonized_frame.to_csv(harmonized_csv, index=False)

    scale_audit = {
        "status": "PASS_FOUR_METHOD_834_GENE_SCALE_HARMONIZATION",
        "gene_count": len(genes),
        "gene_panel_sha256": panel_hash,
        "common_scale": "log1p(sample-level pseudo-bulk TPM-like expression)",
        "methods": {
            "Observed_Post": {
                "rows": len(observed_ids),
                "construction": "mean cell-level TPM within each anti-PD1 Post biopsy, then log1p",
            },
            "scGen": {
                "rows": len(SAMPLES),
                "construction": (
                    "clip generated log1p values to >=0, expm1, mean within source sample, then log1p"
                ),
            },
            "WOT_weighted": {
                "rows": len(SAMPLES),
                "construction": "wot_row_mass-weighted source-sample mean projected TPM, then log1p",
                "semantics": "primary target-derived transport baseline",
            },
            "WOT_unweighted": {
                "rows": len(SAMPLES),
                "construction": "unweighted source-sample mean projected TPM, then log1p",
                "semantics": "sensitivity only",
            },
            "AgentVC_proxy": {
                "rows": len(SAMPLES),
                "construction": (
                    "checkpoint-5 cell-type TPM-like profiles weighted by real PhysiCell endpoint composition; "
                    "mean three seeds within source sample; then log1p"
                ),
                "semantics": "offline reconstructed proxy, not directly saved or executed as a gene field",
            },
        },
        "checks": {
            "all_rows_exactly_834_genes": harmonized.shape[1] == 834,
            "all_values_finite": bool(np.isfinite(harmonized).all()),
            "all_values_nonnegative": bool(np.all(harmonized >= 0)),
            "all_six_source_samples_present_per_virtual_method": all(
                set(np.asarray(harmonized_ids)[np.asarray(methods) == method]) == set(SAMPLES)
                for method in ("scGen", "WOT_weighted", "WOT_unweighted", "AgentVC_proxy")
            ),
            "wot_weighted_primary_and_unweighted_sensitivity_retained": True,
            "no_metrics_or_rankings_computed": True,
        },
        "output_hashes": {
            "harmonized_npz": sha256(harmonized_npz),
            "harmonized_csv": sha256(harmonized_csv),
            "harmonized_matrix_bytes": sha256_bytes(harmonized.tobytes()),
        },
    }
    if not all(scale_audit["checks"].values()):
        raise RuntimeError("Scale harmonization audit failed")
    write_json(OUT / "scale_harmonization_audit.json", scale_audit)

    summary = f"""# AgentVC checkpoint-5 runtime-expression proxy reconstruction V1

## Status

**PASS_AGENTVC_CHECKPOINT5_PROXY_RECONSTRUCTION**

The reconstruction used the original frozen `expression_for_checkpoint`
function, pre-only sample×cell-type TPM profiles, each run's saved checkpoint-4
action/history, checkpoint-5 state, and real PhysiCell endpoint composition.
No LLM, PhysiCell, scGPT, scGen, WOT or external model was rerun.

This output is an **offline reconstructed runtime-expression proxy**. It was
not directly serialized in the original V2.1 runs and is not a
PhysiCell-executed per-gene expression field.

## Frozen inputs

- Samples: {", ".join(SAMPLES)}
- Seeds: {", ".join(map(str, SEEDS))}
- Gene panel: 834 exact genes
- Gene panel SHA256: `{panel_hash}`
- Formula source SHA256: `{formula_hash}`

## Reconstruction checks

- 18/18 sample×seed vectors reconstructed.
- Every vector contains exactly 834 genes.
- No NaN, infinity or negative TPM-like proxy values.
- Repeated reconstruction is byte-identical.
- All endpoint composition weight sums equal one.
- Three seeds were averaged within source sample before `log1p`.

## Harmonization

Observed Post, scGen, WOT weighted, WOT unweighted sensitivity and AgentVC proxy
are stored on the same 834-gene order and the same
`log1p(sample-level pseudo-bulk TPM-like expression)` scale.

The files are technically ready for final metric computation and the proposed
2×2 figure. No metric, ranking, significance test or final figure was generated
in this phase.
"""
    (OUT / "PROXY_RECONSTRUCTION_SUMMARY.md").write_text(summary, encoding="utf-8")

    final_hashes = {
        path.name: sha256(path)
        for path in OUT.iterdir()
        if path.is_file() and path.name != "reconstruction_output_hashes.json"
    }
    write_json(
        OUT / "reconstruction_output_hashes.json",
        {"status": "FROZEN_OUTPUT_HASHES", "files": final_hashes},
    )
    print(json.dumps({
        "status": reconstruction_audit["status"],
        "harmonization": scale_audit["status"],
        "output_dir": str(OUT.resolve()),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
