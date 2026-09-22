# GSE267904 AgentVC-LRRA V5

This independent post-hoc version calibrates low-dimensional ligand/receptor
sender and receiver roles using GSE267904 d7 only. GSE267904 d21 is inaccessible
until the cross-validated d7 parameters are frozen.

Commands are deliberately gated:

```bash
python pipeline.py freeze
python pipeline.py prepare-d7
python pipeline.py smoke
python pipeline.py run-cv-base
python pipeline.py prepare-candidates
python pipeline.py run-cv-candidates
python pipeline.py evaluate-cv
python pipeline.py refit-freeze
python pipeline.py prepare-late
python pipeline.py commot-affine
python pipeline.py evaluate-affine
python pipeline.py commot-median
python pipeline.py evaluate-median
python pipeline.py finalize
python plot_main.py
```

Every command verifies the preceding immutable PASS marker. No figure can be
generated unless `PASS_AGENTVC_LRRA_TOP1_LOCAL_NO_REGRESSION.marker` exists.
