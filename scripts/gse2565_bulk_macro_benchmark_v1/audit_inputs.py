#!/usr/bin/env python3
"""Leakage-safe stage-0 audit for GSE2565 bulk macro benchmark V1.

The script reads all arrays only for per-array distributions and paired technical
replicate QC.  It deliberately does not aggregate or compare 4--72 h biological
expression by condition or time.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/literature/GSE2565"
MATRIX = DATA / "GSE2565_series_matrix.txt.gz"
SOFT = DATA / "GSE2565_family.soft.gz"
ZIP = ROOT / "data/literature/GSE2565.zip"
G141 = ROOT / "data/GSE141259"
OUT = ROOT / "outputs/GSE2565_bulk_macro_benchmark_v1/audit"

QC = {
    "transform": "log2(signal_intensity + 1)",
    "pair_correlation_min": 0.90,
    "pair_median_abs_difference_max": 0.50,
    "pair_iqr_abs_difference_max": 0.50,
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def split_values(line: str) -> list[str]:
    return [x.strip().strip('"') for x in line.rstrip("\n").split("\t")][1:]


def read_matrix() -> tuple[pd.DataFrame, dict[str, list[list[str]]]]:
    meta: dict[str, list[list[str]]] = defaultdict(list)
    samples: list[str] = []
    probes: list[str] = []
    rows: list[list[float]] = []
    in_table = False
    with gzip.open(MATRIX, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("!series_matrix_table_begin"):
                in_table = True
                continue
            if line.startswith("!series_matrix_table_end"):
                break
            if not in_table and line.startswith("!"):
                key = line.split("\t", 1)[0].lstrip("!")
                meta[key].append(split_values(line))
                continue
            if not in_table:
                continue
            fields = [x.strip().strip('"') for x in line.rstrip("\n").split("\t")]
            if not samples:
                samples = fields[1:]
            else:
                probes.append(fields[0])
                rows.append([float(x) for x in fields[1:]])
    return pd.DataFrame(np.asarray(rows), index=probes, columns=samples), meta


def description_rows(meta: dict[str, list[list[str]]], n: int) -> list[list[str]]:
    rows = meta.get("Sample_description", [])
    return [[r[i] if i < len(r) else "" for r in rows] for i in range(n)]


def sample_table(expr: pd.DataFrame, meta: dict[str, list[list[str]]]) -> pd.DataFrame:
    n = expr.shape[1]
    titles = meta["Sample_title"][0]
    desc = description_rows(meta, n)
    out = []
    for i, gsm in enumerate(expr.columns):
        d = desc[i]
        treatment = d[1] if len(d) > 1 else ""
        dose = d[2] if len(d) > 2 else ""
        time = d[3] if len(d) > 3 else ""
        tech = d[4] if len(d) > 4 else ""
        condition = "phosgene" if treatment == "CG" else "air" if treatment in {"Air", "None"} else "unknown"
        title = titles[i]
        bio = re.sub(r"[AB]$", "", title)
        out.append({
            "gsm": gsm,
            "sample_title": title,
            "biological_sample_id": bio,
            "technical_replicate": "A" if tech.endswith("A") else "B" if tech.endswith("B") else "unknown",
            "condition": condition,
            "dose_metadata": dose,
            "time_hours": float(time),
            "source": "whole lung",
            "organism": "Mus musculus",
            "sex": "male",
            "strain": "CD-1",
            "platform": "GPL339",
            "is_biological_replicate": False,
            "technical_merge_group": bio,
            "cel_unavailable": any("CEL file unavailable" in x for x in d),
            "late_expression_access_policy": "metadata_only_before_model_freeze" if float(time) >= 4 else "calibration_allowed",
        })
    return pd.DataFrame(out)


def platform_mapping() -> pd.DataFrame:
    inside = False
    header: list[str] | None = None
    rows = []
    with gzip.open(SOFT, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if line == "!platform_table_begin":
                inside = True
                continue
            if line == "!platform_table_end":
                break
            if not inside:
                continue
            fields = line.split("\t")
            if header is None:
                header = fields
                continue
            fields += [""] * (len(header) - len(fields))
            r = dict(zip(header, fields))
            raw_symbol = r.get("Gene Symbol", "").strip()
            raw_entrez = r.get("ENTREZ_GENE_ID", "").strip()
            primary_symbol = raw_symbol.split(" /// ")[0].strip() if raw_symbol else ""
            primary_entrez = raw_entrez.split(" /// ")[0].strip() if raw_entrez else ""
            rows.append({
                "probe_id": r.get("ID", ""),
                "gene_symbol_raw": raw_symbol,
                "primary_gene_symbol": primary_symbol,
                "entrez_id_raw": raw_entrez,
                "primary_entrez_id": primary_entrez,
                "symbol_mapping_status": "unmapped" if not raw_symbol else "ambiguous_multi_symbol" if " /// " in raw_symbol else "mapped",
            })
    m = pd.DataFrame(rows)
    counts = m.loc[m.primary_gene_symbol.ne(""), "primary_gene_symbol"].value_counts()
    m["probe_count_for_primary_symbol"] = m.primary_gene_symbol.map(counts).fillna(0).astype(int)
    genes = {x.strip().upper() for x in gzip.open(G141 / "GSE141259_WholeLung_genes.txt.gz", "rt") if x.strip()}
    m["present_in_gse141259_wholelung"] = m.primary_gene_symbol.str.upper().isin(genes)
    m["planned_probe_collapse"] = "mean_after_log2_and_AB_technical_merge"
    return m


def technical_qc(expr: pd.DataFrame, samples: pd.DataFrame) -> pd.DataFrame:
    x = np.log2(expr.to_numpy(dtype=float) + 1.0)
    col = {s: i for i, s in enumerate(expr.columns)}
    rows = []
    for bio, g in samples.groupby("biological_sample_id", sort=False):
        if len(g) != 2:
            rows.append({"biological_sample_id": bio, "qc_pass": False, "flag_reason": f"expected_2_chips_found_{len(g)}"})
            continue
        ids = g.gsm.tolist()
        a, b = x[:, col[ids[0]]], x[:, col[ids[1]]]
        corr = float(np.corrcoef(a, b)[0, 1])
        med_a, med_b = float(np.median(a)), float(np.median(b))
        iqr_a = float(np.quantile(a, .75) - np.quantile(a, .25))
        iqr_b = float(np.quantile(b, .75) - np.quantile(b, .25))
        reasons = []
        if corr < QC["pair_correlation_min"]:
            reasons.append("low_probe_correlation")
        if abs(med_a - med_b) > QC["pair_median_abs_difference_max"]:
            reasons.append("median_distribution_shift")
        if abs(iqr_a - iqr_b) > QC["pair_iqr_abs_difference_max"]:
            reasons.append("iqr_distribution_shift")
        rows.append({
            "biological_sample_id": bio,
            "condition_metadata_only": g.condition.iloc[0],
            "time_hours_metadata_only": g.time_hours.iloc[0],
            "chip_1": ids[0], "chip_2": ids[1],
            "probe_level_pearson_log2": corr,
            "chip_1_log2_median": med_a, "chip_2_log2_median": med_b,
            "abs_log2_median_difference": abs(med_a - med_b),
            "chip_1_log2_iqr": iqr_a, "chip_2_log2_iqr": iqr_b,
            "abs_log2_iqr_difference": abs(iqr_a - iqr_b),
            "qc_pass": not reasons,
            "flag_reason": ";".join(reasons),
            "merge_rule_if_pass": "probe-wise arithmetic mean after log2(signal+1)",
            "late_biological_expression_compared": False,
        })
    return pd.DataFrame(rows)


def write_manifest(paths: list[Path]) -> None:
    rows = []
    for p in paths:
        rows.append({"path": str(p.relative_to(ROOT)), "bytes": p.stat().st_size, "sha256": sha256(p), "role": "input", "modified": False})
    pd.DataFrame(rows).to_csv(OUT / "data_manifest.csv", index=False)
    (OUT / "input_hashes.json").write_text(json.dumps({str(p.relative_to(ROOT)): sha256(p) for p in paths}, indent=2) + "\n")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    expr, meta = read_matrix()
    samples = sample_table(expr, meta)
    mapping = platform_mapping()
    tech = technical_qc(expr, samples)
    samples.to_csv(OUT / "sample_metadata.csv", index=False)
    mapping.to_csv(OUT / "gene_mapping_report.csv", index=False)
    tech.to_csv(OUT / "technical_replicate_qc.csv", index=False)
    inputs = [ZIP, MATRIX, SOFT, G141 / "GSE141259_WholeLung_rawcounts.mtx.gz", G141 / "GSE141259_WholeLung_genes.txt.gz", G141 / "GSE141259_WholeLung_barcodes.txt.gz", G141 / "GSE141259_WholeLung_cellinfo.csv.gz"]
    write_manifest(inputs)

    biological = samples.drop_duplicates("biological_sample_id")
    counts = biological.groupby(["time_hours", "condition"]).size().to_dict()
    q = np.quantile(expr.to_numpy(dtype=float), [0, .01, .25, .5, .75, .99, 1])
    audit = f"""# GSE2565 Bulk Macro Benchmark V1 — Stage-0 data audit

## Status and leakage boundary

- GSE2565 is bulk whole-lung microarray, not single-cell data.
- Stage 0 inspected all arrays only for per-chip distribution and A/B technical-replicate QC.
- No 4–72 h condition/time biological expression aggregate, DNB, KL, module trajectory, PCA or outcome plot was computed.
- Late expression remains evaluation-only until `model_frozen.json` exists.

## Files and platform

- Processed series matrix: `{MATRIX.relative_to(ROOT)}`
- Family SOFT/platform annotation: `{SOFT.relative_to(ROOT)}`
- Platform: GPL339, Affymetrix Mouse Expression 430A Array, 22,690 probes.
- Local archive contains no CEL files. GEO metadata advertises raw CEL files, with seven chip records marked unavailable.
- SOFT labels values as `Signal Intensity` but does not document the exact normalization algorithm.

## Expression scale and registered transform

- Matrix shape: {expr.shape[0]} probes × {expr.shape[1]} chips; missing numeric values: {int(expr.isna().sum().sum())}.
- Raw signal quantiles (global distribution audit only): {dict(zip(['min','p01','p25','median','p75','p99','max'], [round(float(v),3) for v in q]))}.
- Positive, strongly right-skewed intensities support the preregistered `log2(signal + 1)` transform.
- A/B are merged probe-wise by arithmetic mean after log2 transformation, only when the frozen QC gates pass.

## Samples

- 104 chip profiles form 52 biological samples; A/B are technical replicates, not biological replicates.
- Biological counts: {json.dumps({f'{k[0]:g}h_{k[1]}': int(v) for k,v in sorted(counts.items())}, sort_keys=True)}.
- 0 h has air only. From 0.5–72 h, air and phosgene are time matched; 48 h air has four biological samples and other nonzero time-condition groups have three.

## Technical replicate QC

- Passing pairs: {int(tech.qc_pass.sum())}/{len(tech)}.
- Correlation range after log2: {tech.probe_level_pearson_log2.min():.6f}–{tech.probe_level_pearson_log2.max():.6f}.
- Frozen gates: correlation ≥ {QC['pair_correlation_min']}; absolute median difference ≤ {QC['pair_median_abs_difference_max']}; absolute IQR difference ≤ {QC['pair_iqr_abs_difference_max']}.
- Any failing pair must be reported and is not silently merged or deleted.

## Gene mapping and reference overlap

- Symbol-mapped probes: {int(mapping.primary_gene_symbol.ne('').sum())}; unique primary symbols: {mapping.loc[mapping.primary_gene_symbol.ne(''),'primary_gene_symbol'].nunique()}.
- Ambiguous multi-symbol rows: {int(mapping.symbol_mapping_status.eq('ambiguous_multi_symbol').sum())}.
- Unique mapped symbols shared with GSE141259 WholeLung: {mapping.loc[mapping.present_in_gse141259_wholelung & mapping.primary_gene_symbol.ne(''),'primary_gene_symbol'].str.upper().nunique()}.
- Multiple probes are collapsed by mean only after log2 transformation and A/B technical merging; ambiguity is retained in the mapping audit.

## Known limitations

- Exact historical series-matrix normalization is undocumented locally.
- Three biological samples per group make ordinary PCC/DNB unstable; original, leave-one-out/bootstrap and shrinkage sensitivity results must be separated.
- GSE141259 PBS supplies independent healthy cell identities/prototypes but is a bleomycin study and cannot supply phosgene injury dynamics.
"""
    (OUT / "data_audit.md").write_text(audit, encoding="utf-8")

    prereg = {
        "experiment_id": "gse2565_bulk_macro_benchmark_v1",
        "status": "stage0_preregistered",
        "data_split": {"initialization": [0], "early_direction_calibration": [0.5, 1], "locked_evaluation": [4, 8, 12, 24, 48, 72]},
        "technical_replicates": QC,
        "final_virtual_seeds": [256501, 256502, 256503, 256504, 256505, 256506],
        "pilot_virtual_seeds": [256501, 256502, 256503],
        "time_points_hours": [0, 0.5, 1, 4, 8, 12, 24, 48, 72],
        "models": ["static_initial", "traditional_rule_physicell", "agent_physicell_full"],
        "conditions": ["virtual_air_control", "virtual_phosgene_injury"],
        "pseudobulk": {"method": "represented_abundance_weighted_mean", "explicit_worker_count_double_weighting": False},
        "dnb": {
            "published_aligned": "pending_reliable_220_gene_and_algorithm_extraction",
            "label": "real-data-defined published-aligned post-hoc benchmark",
            "robustness_network": "STRING Mus musculus external PPI, version and hash required before freeze",
            "required_components": ["SD_in", "PCC_in", "PCC_out"],
            "old_dnb_like_proxy_forbidden": True,
        },
        "skld": {"published_method": "Zhong et al. BMC Genomics 2020; exact algorithm pending implementation verification", "simple_expression_KL_must_not_be_called_skld": True},
        "qualification_gates": {
            "real_physicell_exit_code": 0,
            "all_required_output_snapshots_present": True,
            "finite_nonnegative_substrates_fraction": 1.0,
            "agent_policy_worker_coverage_min": 1.0,
            "represented_abundance_conservation_relative_error_max": 1e-6,
            "common_gene_coverage_min": 10000,
            "rule_agent_outputs_identical_forbidden": True,
            "pilot_seed_variance_min": 1e-10,
            "pilot_seed_cv_max": 0.50,
            "virtual_air_max_mean_injury_change": 0.15,
            "virtual_injury_min_early_injury_change": 0.05,
        },
        "late_expression_access_before_model_freeze": False,
        "global_tissue_coordinator": False,
        "llm_failure_policy": "stop_without_rule_fallback",
    }
    (OUT / "preregistration.json").write_text(json.dumps(prereg, indent=2) + "\n")
    print(json.dumps({"chips": len(samples), "biological_samples": len(biological), "technical_pairs_passed": int(tech.qc_pass.sum()), "common_genes": int(mapping.loc[mapping.present_in_gse141259_wholelung & mapping.primary_gene_symbol.ne(''),'primary_gene_symbol'].str.upper().nunique())}))


if __name__ == "__main__":
    main()
