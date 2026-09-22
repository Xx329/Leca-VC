#!/usr/bin/env python3
"""Build pre-treatment sample × broad-cell-type profiles and freeze scGPT genes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np


BROAD_TYPES = (
    "B cell", "Plasma cell", "Monocyte/Macrophage",
    "Dendritic cell", "T cell", "NK cell",
)
MODULES = {
    "identity_B": ("CD79A", "MS4A1", "CD37", "CD74"),
    "identity_plasma": ("MZB1", "JCHAIN", "SDC1", "CD79A"),
    "identity_monocyte_macrophage": ("LST1", "CTSS", "FCGR3A", "TYROBP"),
    "identity_dendritic": ("FCER1A", "CD1C", "HLA-DRA", "CST3"),
    "identity_T": ("CD3D", "CD3E", "TRAC", "CD247"),
    "identity_NK": ("NKG7", "GNLY", "KLRD1", "FCGR3A", "TYROBP"),
    "activation": ("IFNG", "GZMB", "PRF1", "CD69", "HLA-DRA"),
    "stress": ("FOS", "JUN", "HSPA1A", "DDIT3", "ATF4"),
    "exhaustion": ("PDCD1", "LAG3", "HAVCR2", "TIGIT", "TOX"),
    "treatment_shift": ("IFNG", "GZMB", "NKG7", "CXCL13", "ISG15", "STAT1"),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--expression", type=Path, required=True)
    p.add_argument("--annotations", type=Path, required=True)
    p.add_argument("--vocab", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--panel-size", type=int, default=1200)
    a = p.parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=True)

    with a.annotations.open(newline="", encoding="utf-8") as f:
        annotations = list(csv.DictReader(f))
    by_cell = {row["cell_id"]: row for row in annotations}
    pre_samples = sorted({row["patient_sample"] for row in annotations if row["patient_sample"].startswith("Pre_")})
    groups = [(sample, cell_type) for sample in pre_samples for cell_type in BROAD_TYPES]
    group_index = {group: i for i, group in enumerate(groups)}

    with a.expression.open("rb") as f:
        header = f.readline().rstrip(b"\r\n").decode("utf-8").split("\t")[1:]
    if len(header) != 16291 or len(set(header)) != 16291:
        raise RuntimeError("Unexpected expression cell header")
    cell_group = np.full(len(header), -1, dtype=np.int32)
    for i, cell_id in enumerate(header):
        row = by_cell[cell_id]
        if row["patient_sample"].startswith("Pre_"):
            cell_group[i] = group_index[(row["patient_sample"], row["broad_cell_type"])]
    pre_indices = np.flatnonzero(cell_group >= 0)
    pre_group_codes = cell_group[pre_indices]
    group_counts = np.bincount(pre_group_codes, minlength=len(groups)).astype(np.int64)

    vocab = json.loads(a.vocab.read_text())
    vocab_genes = set(vocab)
    genes: list[str] = []
    means: list[np.ndarray] = []
    seen: set[str] = set()
    total_gene_rows = 0
    trailing_empty_rows = 0
    with a.expression.open("r", encoding="utf-8", newline="") as f:
        f.readline()
        for line in f:
            total_gene_rows += 1
            gene, sep, rest = line.partition("\t")
            if not sep or gene not in vocab_genes:
                continue
            if gene in seen:
                raise RuntimeError(f"Duplicate expression gene in scGPT vocab intersection: {gene}")
            values = rest.rstrip("\r\n").split("\t")
            if len(values) == len(header) + 1 and values[-1] == "":
                values.pop()
                trailing_empty_rows += 1
            if len(values) != len(header):
                raise RuntimeError(f"Gene {gene}: {len(values)} values != {len(header)}")
            vector = np.asarray(values, dtype=np.float32)
            sums = np.bincount(pre_group_codes, weights=vector[pre_indices], minlength=len(groups))
            mean = np.divide(sums, group_counts, out=np.full(len(groups), np.nan), where=group_counts > 0)
            genes.append(gene)
            means.append(mean.astype(np.float32))
            seen.add(gene)

    all_means = np.vstack(means).T  # group × gene
    gene_to_col = {gene: i for i, gene in enumerate(genes)}
    required = sorted({gene for module in MODULES.values() for gene in module if gene in gene_to_col})
    module_coverage = {
        module: {
            "requested": list(module_genes),
            "available": [gene for gene in module_genes if gene in gene_to_col],
            "missing": [gene for gene in module_genes if gene not in gene_to_col],
            "fraction": sum(gene in gene_to_col for gene in module_genes) / len(module_genes),
        }
        for module, module_genes in MODULES.items()
    }
    if any(info["fraction"] < 0.6 for info in module_coverage.values()):
        raise RuntimeError(f"Frozen module gene coverage below 60%: {module_coverage}")
    if len(required) > a.panel_size:
        raise RuntimeError("Required marker genes exceed panel size")

    # Only observed pre-treatment groups contribute to variance selection.
    observed = all_means[group_counts > 0]
    variances = np.nanvar(observed, axis=0)
    order = np.argsort(-variances, kind="stable")
    selected = [gene_to_col[gene] for gene in required]
    selected_set = set(selected)
    for index in order:
        if int(index) not in selected_set:
            selected.append(int(index))
            selected_set.add(int(index))
        if len(selected) == a.panel_size:
            break
    panel_genes = [genes[index] for index in selected]
    profiles = all_means[:, selected].copy()

    # Zero-cell sample/type rows remain explicit but use a pre-only global
    # cell-type prototype for runtime initialization, with source audited.
    sources = []
    for row_index, (_, cell_type) in enumerate(groups):
        if group_counts[row_index] > 0:
            sources.append("sample_cell_type_pre_cells")
            continue
        same_type = [i for i, (_, ct) in enumerate(groups) if ct == cell_type and group_counts[i] > 0]
        weights = group_counts[same_type].astype(np.float64)
        profiles[row_index] = np.average(profiles[same_type], axis=0, weights=weights)
        sources.append("pre_only_global_cell_type_fallback_for_zero_count")
    if not np.isfinite(profiles).all():
        raise RuntimeError("Non-finite frozen expression profile")

    np.savez_compressed(
        a.output_dir / "pretreatment_celltype_expression_profiles.npz",
        profiles=profiles.astype(np.float32),
        sample_ids=np.asarray([sample for sample, _ in groups]),
        broad_cell_types=np.asarray([cell_type for _, cell_type in groups]),
        genes=np.asarray(panel_genes),
        n_cells=group_counts,
        profile_sources=np.asarray(sources),
    )
    with (a.output_dir / "sample_celltype_profile_inventory.csv").open("w", newline="", encoding="utf-8") as f:
        fields = ["sample_id", "broad_cell_type", "n_cells", "profile_source"]
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for (sample, cell_type), count, source in zip(groups, group_counts, sources):
            w.writerow({"sample_id": sample, "broad_cell_type": cell_type, "n_cells": int(count), "profile_source": source})
    required_set = set(required)
    with (a.output_dir / "scgpt_gene_panel.csv").open("w", newline="", encoding="utf-8") as f:
        fields = ["panel_rank", "gene", "vocab_id", "pre_profile_variance", "selection_reason"]
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for rank, index in enumerate(selected, 1):
            gene = genes[index]
            w.writerow({
                "panel_rank": rank, "gene": gene, "vocab_id": vocab[gene],
                "pre_profile_variance": f"{variances[index]:.12g}",
                "selection_reason": "required_frozen_module" if gene in required_set else "top_pre_only_profile_variance",
            })
    manifest = {
        "status": "PASS_PRETREATMENT_SAMPLE_CELLTYPE_PROFILES_AND_GENE_PANEL",
        "pre_only": True,
        "post_cells_used": False,
        "pre_sample_count": len(pre_samples),
        "broad_cell_type_count": len(BROAD_TYPES),
        "profile_count": len(groups),
        "observed_sample_celltype_profiles": int((group_counts > 0).sum()),
        "zero_count_profiles_with_pre_only_celltype_fallback": int((group_counts == 0).sum()),
        "pre_cell_count": int(len(pre_indices)),
        "expression_gene_rows": total_gene_rows,
        "expression_vocab_intersection_genes": len(genes),
        "panel_size": len(panel_genes),
        "max_seq_len": 1200,
        "panel_selection": "all available frozen module genes plus highest variance genes across observed pre-treatment sample×cell-type profiles",
        "module_coverage": module_coverage,
        "terminal_empty_fields_removed": trailing_empty_rows,
        "expression_sha256": sha256(a.expression),
        "annotation_sha256": sha256(a.annotations),
        "vocab_sha256": sha256(a.vocab),
        "profile_npz_sha256": sha256(a.output_dir / "pretreatment_celltype_expression_profiles.npz"),
        "gene_panel_sha256": sha256(a.output_dir / "scgpt_gene_panel.csv"),
    }
    (a.output_dir / "expression_profile_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
