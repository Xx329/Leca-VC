#!/usr/bin/env python3
"""Build a 50 micro-agent registry from calibration spatial spots."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from common import CELL_TYPES, MICRO_AGENT_ALLOCATION, dump_json, ensure_dirs, outpath


def import_anndata():
    import anndata as ad
    return ad


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--project-root", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=Path("outputs/GSE267904_spatial_commot_agent"))
    p.add_argument("--num-agents", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    root = a.project_root.resolve()
    out = outpath(root, a.out_dir)
    ensure_dirs(out)
    rng = np.random.default_rng(a.seed)
    ad = import_anndata()
    h5ads = [p for p in [out / "real_benchmark/d7_ctrl.h5ad", out / "real_benchmark/d7_bleo.h5ad"] if p.exists()]
    if not h5ads:
        raise RuntimeError("Need d7_ctrl/d7_bleo h5ad files with marker annotations to initialize micro-agents.")

    agents, expr_rows, pos_rows = [], [], []
    agent_idx = 0
    for h5 in h5ads:
        adata = ad.read_h5ad(h5)
        if "dominant_cell_type" not in adata.obs:
            raise RuntimeError(f"{h5} lacks dominant_cell_type; run annotate_spots_by_markers.py first.")
    # Merge references by cell type.
    refs = {}
    all_genes = None
    for h5 in h5ads:
        adata = ad.read_h5ad(h5)
        all_genes = list(map(str, adata.var_names))
        xy = adata.obsm["spatial"]
        for ct in CELL_TYPES:
            idx = np.where(adata.obs["dominant_cell_type"].astype(str).to_numpy() == ct)[0]
            if len(idx) == 0:
                continue
            sub = adata.X[idx]
            arr = sub.toarray() if hasattr(sub, "toarray") else np.asarray(sub)
            refs.setdefault(ct, {"expr": [], "xy": []})
            refs[ct]["expr"].append(np.asarray(arr.mean(axis=0)).ravel())
            refs[ct]["xy"].append(xy[idx])
    if all_genes is None:
        raise RuntimeError("Could not read genes from h5ad.")

    for ct, n in MICRO_AGENT_ALLOCATION.items():
        if agent_idx >= a.num_agents:
            break
        n = min(n, a.num_agents - agent_idx)
        if ct in refs:
            expr = np.mean(np.vstack(refs[ct]["expr"]), axis=0)
            xy_pool = np.vstack(refs[ct]["xy"])
        else:
            expr = np.zeros(len(all_genes))
            xy_pool = np.array([[0.0, 0.0]])
        for _ in range(n):
            agent_id = f"micro_agent_{agent_idx:03d}"
            base_xy = xy_pool[rng.integers(0, len(xy_pool))]
            jitter = rng.normal(0, 30, size=2)
            x, y = (base_xy + jitter).tolist()
            agents.append({
                "agent_id": agent_id,
                "cell_type": ct,
                "initial_position_x": float(x),
                "initial_position_y": float(y),
                "neighbor_radius": 150.0,
                "initial_memory": "low",
                "possible_programs": "maintain_identity;injury_response;recruitment;fibrosis;resolution;migration;apoptosis",
            })
            expr_rows.append({"agent_id": agent_id, **{g: float(v) for g, v in zip(all_genes, np.log1p(expr))}})
            pos_rows.append({"agent_id": agent_id, "cell_type": ct, "x": float(x), "y": float(y)})
            agent_idx += 1
    # Fill if allocation changed.
    while agent_idx < a.num_agents:
        ct = "other"
        agent_id = f"micro_agent_{agent_idx:03d}"
        agents.append({"agent_id": agent_id, "cell_type": ct, "initial_position_x": 0.0, "initial_position_y": 0.0, "neighbor_radius": 150.0, "initial_memory": "low", "possible_programs": "maintain_identity"})
        expr_rows.append({"agent_id": agent_id, **{g: 0.0 for g in all_genes}})
        pos_rows.append({"agent_id": agent_id, "cell_type": ct, "x": 0.0, "y": 0.0})
        agent_idx += 1
    reg = pd.DataFrame(agents)
    expr = pd.DataFrame(expr_rows)
    pos = pd.DataFrame(pos_rows)
    reg.to_csv(out / "micro_agents/micro_agent_registry.csv", index=False)
    expr.to_csv(out / "micro_agents/micro_agent_expression_prototypes.csv", index=False)
    pos.to_csv(out / "micro_agents/micro_agent_initial_positions.csv", index=False)
    dump_json(out / "micro_agents/micro_agent_registry_summary.json", {
        "num_micro_agents": len(reg),
        "allocation": reg["cell_type"].value_counts().to_dict(),
        "expression_genes": len(all_genes),
        "source_h5ad": [str(x) for x in h5ads],
    })
    print(f"Wrote {len(reg)} micro-agents to {out/'micro_agents'}")


if __name__ == "__main__":
    main()

