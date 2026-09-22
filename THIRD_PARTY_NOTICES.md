# Third-party notices

This repository contains Leca-VC glue code, evaluation code, plotting code, and
small frozen source-data tables. It does **not** vendor full copies of
PhysiCell, BioFVM, COMMOT, scGPT, scGen, CellRank, WOT, or their model weights.

Full simulation requires separately installed PhysiCell/BioFVM 1.14.2.
COMMOT and the Python/R baseline packages are installed into separate
environments under their upstream licenses. Dataset accessions and download
locations are documented in `manifests/datasets.yaml`; original dataset terms
continue to apply. Before public release, the private release candidate must
complete the license audit recorded in `manifests/release_status.json`.

The collaborator-delivered fibrosis application references GSE141259,
GSE264278, the GSE267904 data, and BioStudies accession S-BSST1409. Their raw
files are intentionally excluded from Git and remain governed by their source
repositories' terms. The delivery's original top-level MIT badge/README is not
used to relicense third-party dependencies; the integrated Leca-VC source is
released under the repository's BSD-3-Clause license subject to these notices.
