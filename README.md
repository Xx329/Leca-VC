# Leca-VC paper code

This is the isolated private release candidate for the Leca-VC manuscript.
`Short_Article_Title (6).pdf` and
`Leca_VC_supplemental_overleaf_v1.pdf` are the final visual, numerical, and
figure-numbering authorities; the private PDFs are not distributed here.

## Current release status

The quick tier is implemented for Figures 2–6 and Supplementary Figures
S1–S8. Figure 1 is an author-created design asset and is intentionally excluded
from code reproduction. Supplementary Figure S6 is now tied to the
collaborator-delivered original renderer and a CRC-validated step-60 runtime
record. Collaborator source for the Figure 6 pulmonary-fibrosis application is
also integrated, together with the exact 7200-by-3600 raster shown in the
canonical manuscript, a partial aggregate table, run matrix, and audit files.
The authors confirm that Figure 6 was assembled manually, so the exact frozen
image is the authoritative assembly asset and no unified six-panel plotting
script is expected. The completion logs nevertheless contain a freeze bypass
and failed exploratory runs; this is retained as a runtime-provenance
limitation rather than being conflated with the manual layout.
See
`manifests/release_status.json` and
`experiments/fibrosis_application/provenance/delivery_audit.json`.

No remote repository has been created and this directory has not been
initialized as Git. Public release remains disabled until the final RC checks
and author approval are complete.

The top-level `experiments/` directories are the supported benchmark
entrypoints. `scripts/` contains the current scientific implementation and the
small set of earlier base modules that those entrypoints import. The exact
dependency contract is recorded in `manifests/code_entrypoints.json`; files in
`scripts/` are not separate claims that every historical experiment is a
supported release workflow.

## Quick figure reproduction

```bash
conda env create -f environment/figure-environment.yml
conda activate lecavc-figures
python workflows/reproduce_figures.py --figure fig4
python workflows/reproduce_figures.py --figure figS8
python workflows/reproduce_figures.py --all
```

During the private RC, `--all` renders every `READY` entry. Figure 6 verifies
and repackages the exact 300-DPI collaborator raster; its PDF/SVG wrappers are
not native-vector reconstructions. Outputs go to `build/figures/`; immutable
visual references remain under `figures/`.

## Full benchmark preflight and execution

```bash
python workflows/run_benchmark.py --dataset gse2565 --preflight
python workflows/run_benchmark.py --dataset gse120575 --preflight
python workflows/run_benchmark.py --dataset gse267904 --preflight
python workflows/run_benchmark.py --dataset fibrosis_application --preflight

export DEEPSEEK_API_KEY='enter-key-in-your-shell-only'
export PHYSICELL_ROOT=/path/to/PhysiCell-1.14.2
export COMMOT_PYTHON=/path/to/lecavc-commot/bin/python
python workflows/run_benchmark.py --dataset gse2565 --run
```

The fibrosis-application full-rerun preflight intentionally reports `BLOCKED`
until the contributor reconciles the scientific qualification,
completion-log, archive-integrity, and provider-provenance issues. This does
not block publication of the author-confirmed manual Figure 6 assembly. The
release workflow will not resume that paused V3 experiment implicitly.

The key is read only from `DEEPSEEK_API_KEY`. Missing credentials, invalid
schema/ranges, failed PhysiCell/BioFVM execution, or absent COMMOT stop the run;
there is no rule-based fallback. Large processed inputs are not in Git and
must be installed from the verified Zenodo artifact once its draft URL and
hash are frozen.

## Contamination probe

```bash
python workflows/run_contamination_probe.py --offline
python workflows/run_contamination_probe.py --online
```

The offline command verifies 30 frozen exact-runtime probes (1 correct, 6
incorrect, 23 unknown; accuracy 1/30) and redraws Fig. S7. The probe supports
the operational claim that deployed de-identified prompts did not enable
reliable recall of held-out outcomes. It cannot prove that the underlying
studies were absent from LLM pretraining.

## Data and artifact policy

- GitHub: code, small source-data, reference vector figures, manifests, tests.
- GEO: original public datasets and accession-level provenance.
- Zenodo: minimal processed inputs, de-identified runtime audit records, and
  frozen reference-result intermediates.

Exact prompts/responses are eligible for the runtime-audit archive only after
the second secret and private-path scan. API keys, authorization headers,
cookies, and local user paths must never be included.

## Scientific invariants

Agents represent cell types or state prototypes and can instantiate multiple
PhysiCell worker cells. Worker cells execute shared conditional programs in
their own local BioFVM environments. This release does not add a global tissue
coordinator. Held-out data are evaluation-only and must not influence prompt,
parameter, threshold, or metric selection.

License: BSD-3-Clause. Third-party software and datasets remain under their
upstream terms; see `THIRD_PARTY_NOTICES.md`.
