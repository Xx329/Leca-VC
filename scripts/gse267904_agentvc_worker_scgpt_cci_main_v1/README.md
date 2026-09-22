# GSE267904 AgentVC worker vs external-scGPT CCI v1

This directory implements the prospectively frozen worker benchmark after the
historical Agent-level analyses. It never overwrites the K=100 or scGPT v1
outputs and never substitutes a Python spatial executor for PhysiCell.

The complete resumable run is:

```bash
bash scripts/gse267904_agentvc_worker_scgpt_cci_main_v1/run_pipeline.sh
```

Formal results are written only to:

`outputs/GSE267904_agentvc_worker_scgpt_cci_main_v1/`

COMMOT stages must run with `/path/to/commot-env/bin/python`. The formal
runner executes all 58 frozen pairs in one call for every input and supplies a
distance matrix whose diagonal is outside the frozen 500-pixel threshold.

The single 2x3 publication main figure can be regenerated without rerunning
PhysiCell or COMMOT:

```bash
MPLCONFIGDIR=/tmp/agentvc_paper_main_mpl \
  /path/to/commot-env/bin/python \
  scripts/gse267904_agentvc_worker_scgpt_cci_main_v1/plot_paper_main_v1.py
```

It writes only to `08_figures/paper_main_v1/`.
