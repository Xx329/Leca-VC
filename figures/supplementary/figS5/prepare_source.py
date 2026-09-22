#!/usr/bin/env python3
"""Freeze the minimal numeric arrays needed to redraw final Supplementary Fig. S5.

This one-time source builder reads the completed K=100 COMMOT run. The quick
renderer uses only the compact NPZ it creates and never requires the large
AnnData files.
"""
from pathlib import Path
import argparse
import json
import anndata as ad
import numpy as np
import pandas as pd

STAGES=[("real","d7_bleo"),("virtual","virtual_early_bleo"),("real","d21_bleo"),("virtual","virtual_late_bleo")]
FAMILIES=["TGFB","CCL","CXCL"]
MODES=["sender","receiver"]

def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--input-root",type=Path,required=True,help="Completed K_100 output directory")
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args(); root=args.input_root.resolve(); arrays={}; meta={"stages":[],"families":FAMILIES,"modes":MODES}
    edge_cache={}
    for source in ("real","virtual"):
        filename="real_commot/real_commot_edges_long.csv" if source=="real" else "virtual_commot/virtual_commot_edges_long.csv"
        edge_cache[source]=pd.read_csv(root/filename)
    for index,(source,stage) in enumerate(STAGES):
        filename=(root/"real_benchmark"/f"{stage}.h5ad") if source=="real" else (root/"virtual_commot_input"/f"{stage}.h5ad")
        data=ad.read_h5ad(filename)
        xy=np.asarray(data.obsm["spatial"],dtype=float)
        labels=data.obs["dominant_cell_type" if source=="real" else "cell_type"].astype(str)
        arrays[f"xy_{index}"]=xy
        edges=edge_cache[source]; current=edges[edges.stage.astype(str).eq(stage)]
        for family in FAMILIES:
            subset=current[current.pathway.astype(str).str.contains(family,case=False,na=False,regex=False)]
            for mode in MODES:
                group="sender" if mode=="sender" else "receiver"
                strengths=subset.groupby(group).weight.sum().to_dict()
                arrays[f"score_{index}_{family}_{mode}"]=labels.map(lambda x:float(strengths.get(x,0.0))).to_numpy(float)
        meta["stages"].append({"index":index,"source":source,"stage":stage,"n":int(len(xy))})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.output,**arrays)
    args.output.with_suffix(".json").write_text(json.dumps(meta,indent=2)+"\n",encoding="utf-8")

if __name__=="__main__": main()
