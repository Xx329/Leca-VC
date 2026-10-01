# Private release-candidate checklist

Historical 2026-09-22 checklist; this is not the current public release status.
See `FINAL_PAPER_SYNC.md` and `../manifests/release_status.json` for the
author-designated final PDFs and current offline reproduction status.

- [x] Isolated release directory; historical project untouched.
- [x] Canonical manuscript SHA256 recorded.
- [x] Figures 2–6 and S1–S8 reproduce from frozen release inputs; Figure 6 is an author-confirmed manual assembly asset.
- [x] Ready figures are pixel-identical to their frozen references; Figure 6 comparison is against the exact recovered raster.
- [x] Prompt firewall, schema/range, no-fallback, probe, and hygiene tests.
- [x] Four included PhysiCell custom scenarios compile against 1.14.2.
- [x] Figure 1 recorded as an author-created design asset exempt from code reproduction.
- [x] Receive and provenance-audit collaborator-owned Figure 6 pipeline code.
- [x] Receive and CRC-validate the exact frozen Figure 6 reference raster plus partial table/audit inputs.
- [x] Record that Figure 6 was assembled manually and therefore has no missing unified six-panel renderer.
- [ ] Optional for the later Zenodo runtime archive: obtain a non-truncated outputs archive and reconciled scientific-gate evidence.
- [x] Map final Supplement S1–S8 and validate S6 values against the recovered step-60 runtime record.
- [ ] Stage and audit the three Zenodo archives; record DOI and hashes.
- [ ] Fresh-directory environment installation and full quick-tier test.
- [ ] License audit sign-off.
- [ ] Initialize local Git and create `v1.0.0-rc1` only after the above gates.
- [ ] Create private remote only after explicit user confirmation.
- [ ] Publish `v1.0.0` only after final manuscript/caption/numeric audit.
