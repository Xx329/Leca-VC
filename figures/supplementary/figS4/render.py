#!/usr/bin/env python3
from pathlib import Path
import sys
import pandas as pd
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from heatmap_base import plot_cci,setup
def main():
    out=ROOT/"build/figures/figS4"; out.mkdir(parents=True,exist_ok=True); setup()
    plot_cci(pd.read_csv(ROOT/"source_data/figS4/GSE267904_CCI_network_heatmaps_source_data.csv"),out)
if __name__=="__main__": main()
