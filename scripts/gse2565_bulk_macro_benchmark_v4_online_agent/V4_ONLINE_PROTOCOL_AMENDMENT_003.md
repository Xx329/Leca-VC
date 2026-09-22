# V4 Online Protocol Amendment 003

## Problem

The semantic probe showed that DeepSeek now produced non-zero compatible program strengths, but repeatedly returned zero redundant phenotype multipliers. The validator rejected the response after four attempts with:

`active secretion-related program requires secretion_multiplier >=0.05`

This is not an all-zero Agent decision. It is an inconsistency between the primary biological decision (`program_strengths`) and redundant execution couplers (`phenotype_action` multipliers).

## Frozen correction

`program_strengths` remain the primary LLM decision and are never altered.
The raw API response remains preserved verbatim.
Before semantic execution validation, Python deterministically canonicalizes only the redundant executor couplers:

- `secretion_multiplier = max(raw secretion_multiplier, max secretion-related program strengths)`
- `uptake_multiplier = max(raw uptake_multiplier, toxicant_uptake)`
- `motility_multiplier = max(raw motility_multiplier, motility, immune_recruitment)`
- `death_tendency = max(raw death_tendency, death_signal)`
- `repair_tendency = max(raw repair_tendency, repair_signal, ecm_remodeling)`

The correction is fixed, condition-independent, seed-independent and time-independent. It does not use real expression, target time, DNB results or future state. It is an execution adapter, not a rule fallback.

## Audit

The validator records original, required and canonical executor fields, whether canonicalization was applied, and confirms that program strengths were unchanged and fallback was not used.
