# GSE267904 Spatial COMMOT + 50 Micro-Agent + PhysiCell Executor

This experiment is intentionally strict:

- real COMMOT requires `commot` and `scanpy`;
- virtual COMMOT is computed from newly generated virtual h5ad inputs;
- real PhysiCell-backed execution is required by default;
- the optional debug spatial generator is explicitly marked `REAL_PHYSICELL_USED=false`.

## Preflight

```bash
python scripts/gse267904_spatial_commot_agent/preflight_commot.py \
  --project-root /path/to/Leca-VC \
  --out-dir outputs/GSE267904_spatial_commot_agent \
  --require-data
```

## Download

```bash
python scripts/gse267904_spatial_commot_agent/download_gse267904.py \
  --project-root /path/to/Leca-VC \
  --out-dir outputs/GSE267904_spatial_commot_agent
```

## Full run

```bash
NUMBA_DISABLE_JIT=1 MPLCONFIGDIR=/tmp/matplotlib \
python scripts/gse267904_spatial_commot_agent/run_pipeline.py \
  --project-root /path/to/Leca-VC \
  --out-dir outputs/GSE267904_spatial_commot_agent \
  --num-agents 50
```

If a dedicated real PhysiCell scenario is missing, the pipeline stops with
`audit/physicell_blocking_report.json`. Do not use `--allow-debug-model-spatial-generator`
for claims; it is only for debugging downstream COMMOT formatting.

