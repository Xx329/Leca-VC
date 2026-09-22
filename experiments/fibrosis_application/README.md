# GSE267904 pulmonary-fibrosis application

This directory contains the collaborator-delivered source snapshot for the
downstream fibrosis application associated with manuscript Figure 6. The
delivery archive is recorded in `provenance/delivery_audit.json`; raw datasets,
compiled binaries, workstation configs, and the 506 MiB archive are not
committed.

## Current status

`CODE_AND_AUTHOR_CONFIRMED_MANUAL_PAPER_ASSEMBLY_INTEGRATED`

The delivery includes the Python pipeline, PhysiCell/BioFVM custom project,
environment specification, a metadata-only manifest for 1,200 cached LLM
decisions, the exact 7200-by-3600 Figure 6 raster used by the manuscript, a
418-row requested-run matrix, a 427-row aggregate result table, and four audit
JSON files. The authors confirm that the final Figure 6 layout was assembled
manually. The quick figure tier therefore preserves and repackages that exact
paper asset; a unified six-panel renderer is neither available nor expected.
The separate full-rerun evidence is evaluated below and remains fail-closed.

Two source issues require resolution by the contributing team before a formal
rerun:

1. `application/qualify_pre_benchmark.py` defines the same field-boost helper
   twice, labels it as 30-fold, implements 500-fold, and invokes it twice. The
   release preserves this scientifically material code for review instead of
   guessing the intended multiplier.
2. The delivered online client and `llm_cell_agent_policy_audit.json` identify
   a DashScope/Qwen configuration, while the earlier cache manifest and a
   recovered step-60 raw response identify `deepseek-v4-flash`. Provider
   provenance must be reconciled before a formal rerun claim.
3. The delivered formal logs state `Freeze skipped (temporary bypass)`. Other
   logs contain unknown-run failures, a 401 call, and failed result collection.
   In addition, the qualification JSON marks an agent-versus-diffusion
   `>=20%` gate true while recording `0.09011247225692984`. The 418-row run
   matrix is therefore treated as a requested-run manifest, not proof that all
   runs completed under the frozen protocol.

Only a portability defect was corrected in the integrated snapshot:
`application/physicell_engine.py` now resolves PhysiCell from
`PHYSICELL_ROOT` (or its project-relative fallback) instead of a collaborator
workstation path. No simulation, qualification, or evaluation was run during
release assembly. Only the CRC-validated frozen raster was repackaged.

## Layout

- `application/`: contributor analysis and simulation pipeline.
- `tools/`: portable handoff entrypoints retained for review.
- `configs/local_paths.example.yaml`: empty path template only.
- `provenance/`: archive/source hashes and cache metadata.
- `../../physicell/fibrosis_application/`: custom C++/XML/Makefile.

Large inputs belong in the processed-input artifact or are downloaded from
GEO; see `manifests/datasets.yaml`. Do not place them in Git.
