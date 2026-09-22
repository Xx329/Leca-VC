#!/usr/bin/env python3
"""Generate auditable worker expression and represented-abundance pseudobulk."""
from __future__ import annotations

import argparse
import gzip
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/GSE2565_bulk_macro_benchmark_v1"
SOFT = ROOT / "data/literature/GSE2565/GSE2565_family.soft.gz"
PROGRAMS = {
    "oxidative_stress": ["oxidative stress", "oxidation-reduction", "reactive oxygen"],
    "epithelial_injury": ["response to wounding", "response to toxic substance", "cellular response to chemical"],
    "inflammation": ["inflammatory response", "immune response", "nf-kappa"],
    "edema_proxy": ["endothelial", "blood vessel", "cell junction", "vascular"],
    "death_signal": ["apoptotic process", "programmed cell death"],
    "repair_signal": ["wound healing", "tissue regeneration", "cell proliferation"],
}


def gene_program_sets(genes: np.ndarray) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    inside = False
    header = None
    ann: dict[str, str] = {}
    with gzip.open(SOFT, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if line == "!platform_table_begin": inside = True; continue
            if line == "!platform_table_end": break
            if not inside: continue
            z = line.split("\t")
            if header is None: header = z; continue
            z += [""] * (len(header) - len(z)); r = dict(zip(header, z))
            symbol = r.get("Gene Symbol", "").split(" /// ")[0].strip()
            if symbol: ann[symbol.upper()] = r.get("Gene Ontology Biological Process", "").lower()
    sets = {}
    rows = []
    for program, terms in PROGRAMS.items():
        idx = np.array([i for i, gene in enumerate(genes) if any(t in ann.get(str(gene).upper(), "") for t in terms)], dtype=int)
        sets[program] = idx
        rows.append({"program": program, "source": "GPL339 Gene Ontology Biological Process annotation, GEO annotation date 2014-10-06", "selection_terms": "|".join(terms), "gene_count": len(idx), "evaluation_primary_module": False})
    return sets, pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--stage", choices=["smoke", "pilot", "final"], required=True); a = p.parse_args()
    proto = np.load(OUT / "initialization/healthy_expression_prototypes.npz", allow_pickle=False)
    genes, agents, base = proto["genes"], proto["agent_types"], proto["expression"]
    base_by_agent = {str(agent): base[i] for i, agent in enumerate(agents)}
    program_sets, program_audit = gene_program_sets(genes)
    program_audit.to_csv(OUT / "audit/expression_program_gene_sets.csv", index=False)
    run_manifest = json.loads((OUT / f"simulation/{a.stage}_run_manifest.json").read_text())
    cell_root = OUT / "virtual_expression/cell_level"
    bulk_root = OUT / "virtual_expression/pseudobulk"
    cell_root.mkdir(parents=True, exist_ok=True); bulk_root.mkdir(parents=True, exist_ok=True)
    coverage_rows, weight_rows = [], []
    for run in run_manifest:
        run_id = run["run_id"]; run_dir = OUT / "simulation" / run_id
        times = sorted(int(re.search(r"minute_(\d+)", x.name).group(1)) for x in run_dir.glob("work_cells_minute_*.csv"))
        bulk_rows = []
        run_cell = cell_root / run_id; run_cell.mkdir(parents=True, exist_ok=True)
        for minute in times:
            cells = pd.read_csv(run_dir / f"work_cells_minute_{minute}.csv")
            expression = np.vstack([base_by_agent[a].copy() for a in cells.agent_id])
            signal_columns = {"oxidative_stress":"oxidative_stress", "epithelial_injury":"injury_state", "inflammation":"inflammation_state", "edema_proxy":"edema_proxy", "death_signal":"death_signal", "repair_signal":"repair_state"}
            coefficients = {"oxidative_stress":.35,"epithelial_injury":.45,"inflammation":.35,"edema_proxy":.22,"death_signal":.30,"repair_signal":.28}
            for program, col in signal_columns.items():
                idx = program_sets[program]
                if len(idx): expression[:, idx] += coefficients[program] * cells[col].to_numpy()[:, None]
            expression = np.maximum(expression, 0).astype(np.float32)
            weights = cells.represented_abundance.to_numpy(float)
            weighted = np.average(expression, axis=0, weights=weights)
            np.savez_compressed(run_cell / f"worker_expression_minute_{minute}.npz", genes=genes, worker_ids=cells.worker_id.astype(str).to_numpy(), agent_ids=cells.agent_id.astype(str).to_numpy(), represented_abundance=weights, expression=expression)
            bulk_rows.append({"minute": minute, "hour": minute/60, **{str(g): float(v) for g, v in zip(genes, weighted)}})
            weight_rows.append({"run_id":run_id,"minute":minute,"worker_count":len(cells),"represented_abundance_sum":float(weights.sum()),"pseudobulk_method":"represented_abundance_weighted_mean","worker_count_applied_as_extra_weight":False})
        pd.DataFrame(bulk_rows).to_csv(bulk_root / f"{run_id}.csv.gz", index=False, compression="gzip")
        coverage_rows.append({"run_id":run_id,"common_genes":len(genes),"all_workers_have_expression":True,"cell_level_format":"compressed_npz","pseudobulk_format":"csv.gz","late_real_expression_used":False})
    pd.DataFrame(coverage_rows).to_csv(OUT / "virtual_expression/gene_coverage_audit.csv", index=False)
    pd.DataFrame(weight_rows).to_csv(OUT / "audit/pseudobulk_weight_audit.csv", index=False)
    print(json.dumps({"stage":a.stage,"runs":len(run_manifest),"genes":len(genes),"expression_programs":{k:len(v) for k,v in program_sets.items()}}))


if __name__ == "__main__": main()

