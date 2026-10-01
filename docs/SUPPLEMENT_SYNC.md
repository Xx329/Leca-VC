# Supplement synchronization record

The canonical Supplement is `Leca_VC_supplemental_overleaf_v1 (1).pdf` (SHA256
recorded in `manifests/paper_figures.yaml`) and contains Figures S1–S8.
All eight figures are mapped to release renderers and source data and are
pixel-identical to their frozen release references. The S6 input/output values
are additionally validated against the collaborator-delivered original
renderer and CRC-validated step-60 prompt/response record; full raw records
remain assigned to the separately scanned runtime-audit artifact.

Final numbering is S1 functional heatmaps, S2 GSE120575 composition,
S3 GSE230538 composition, S4 CCI heatmaps, S5 spatial hotspots, S6 step-60
policy, S7 clean program traces, and S8 contamination probe. Caption text
follows the final PDF. Both supplementary tables have frozen inputs and
independent offline numerical verification via `workflows/reproduce_tables.py`.
Detailed lineage and manuscript wording caveats are in `FINAL_PAPER_SYNC.md`.
