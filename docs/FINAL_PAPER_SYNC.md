# Final manuscript synchronization — 2026-10-02

The author-designated authorities are `Short_Article_Title (7).pdf` and
`Leca_VC_supplemental_overleaf_v1 (1).pdf`, recorded by SHA256 in the figure
manifest. Reference assets and captions reproduce those documents; scientific
sources retain their actual experiment versions. No LLM, PhysiCell, or COMMOT
experiment was rerun during synchronization.

## Numbering

Figure 3 now places responder expression at A–F, non-responder expression at
G–L, and CCI at M–N. The panel mapping CSV is stored beside the renderer.

| Previous repository ID | Final Supplement ID | Content |
|---|---|---|
| S4 | S1 | GSE2565 functional-program heatmaps |
| S1 | S2 | GSE120575 composition |
| S2 | S3 | GSE230538 composition |
| S3 | S4 | GSE267904 CCI heatmaps |
| S5 | S5 | Spatial COMMOT hotspots |
| S6 | S6 | Step-60 myofibroblast policy |
| S8 | S7 | Simplified agent-program traces |
| S7 | S8 | Exact-runtime contamination probe |

S1 and S4 use the visible label Leca-VC. S2 omits the previous V2.1 footer.
S7 uses the final clean layout, and the S8 caption defines Wilson intervals.
The command names, plotting inputs, verification paths, and table lineage use
the final numbering.

## Numerical provenance and manuscript wording

The author's requested final-PDF numerical values and captions are retained.
This preserves manuscript correspondence, not a claim that every original
manuscript statement is supported by the underlying files.

- Figure 2 uses common 1,601-gene Panel A curves and peak-time errors of
  12/12/4/4 h. Leca-VC's Pearson 0.523543 and DTW 1.515170 remain the formal
  11,171-gene metrics. The formal full-space peak error is 0 h. The Results'
  formal/common wording and the caption's reference-line description need
  reconciliation; see `source_data/fig2/dnb_metric_scopes.json`.
- Figure 3 M–N and Supplement S4 retain the frozen V5 diagnostic CCI source.
  That source failed its scientific gate and lacks passed Visium pseudo-spot
  confirmation. This status is not changed by successful figure rendering.
- Supplementary Table 1 comes from the historical GSE120575 V7 external
  validation, with 23,848 filtered genes and 39 signature genes. It is not
  reassigned to the de-identified 834-gene experiment in Figures 3–4.
- Supplementary Table 2 reuses frozen GSE230538 V7 trajectories and the
  6,336-gene old-scale analysis. WOT is target-derived; CellRank is a
  pre-derived expression proxy; scGen uses a Day0-dispersion-adjusted
  pseudo-bulk proxy because the native generated-cell file was absent.
- Figure 6 is the exact author-approved manual assembly asset. Contributor
  source, partial numerical results, and delivered runtime limitations remain
  recorded. This synchronization does not resume the paused V3 experiment.

The main text uses both “Supplementary Table 2” and “Supplementary Table S2”,
while the final Supplement labels the table “Table 2”. Code Availability also
describes future public release although this repository is already public.
Large-input and complete runtime-audit artifact URLs remain pending; the
offline figure tier is available without those artifacts.

## Verification

`workflows/reproduce_figures.py --all` recreates Figures 2–6 and S1–S8,
copies their final-PDF captions, and records execution reports.
`workflows/reproduce_tables.py --all` rebuilds both tables, independently
checks full-gene Table 1 correlations and all Table 2 Pearson/CCC/RMSE values.
`workflows/verify_figures.py` compares rendered pixels with the updated frozen
references; it is a reproducibility test, not an independent scientific gate.
`manifests/canonical_pdf_alignment.json` records the separate final-PDF audit.
The four custom C++ scenarios compile against a fresh PhysiCell 1.14.2 checkout.
Their Makefiles generate a build-only main file from the upstream template so
they do not depend on the mutable default sample project's `main.cpp`.
Repository hygiene tests exclude Git's pack files and ignored vendor/build
directories while continuing to inspect all release files.
