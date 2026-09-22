# V3 Application Source

This directory is the transferred GSE267904 Fibrosis Application V3 source snapshot.

- Use the retained `../tools/*.sh` entrypoints only after the unresolved
  qualification issues documented in `../README.md` have been reviewed.
- Runtime paths come from `configs/local_paths.yaml` and environment variables.
- V3 is paused because pre-benchmark qualification failed. Do not run the formal matrix until qualification passes and a new model lock is created.
- The source contains no scGPT or COMMOT runtime import. Those components are documented as not used by this V3 implementation.
