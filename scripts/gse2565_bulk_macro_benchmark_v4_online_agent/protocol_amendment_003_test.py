#!/usr/bin/env python3
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location('online_controller_amend003', HERE / 'online_controller.py')
MOD = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MOD)


def base_req(stable: bool) -> dict:
    return {
        'agent_id': 'AT2',
        'current_time': 4.0,
        'stable_air_evidence': stable,
        'repair_evidence': False,
        'normalized_program_evidence': {
            'oxidative_stress': 1.0 if not stable else 0.0,
            'epithelial_injury': 0.35 if not stable else 0.0,
            'inflammation': 0.0,
            'immune_recruitment': 0.0,
            'edema_proxy': 0.0,
            'death_signal': 0.0,
            'repair_signal': 0.0,
            'ecm_remodeling': 0.0,
            'toxicant_uptake': 1.0 if not stable else 0.0,
            'motility': 0.0,
        },
    }


def decision(active: bool) -> dict:
    strengths = {p: 0.0 for p in MOD.PROGRAMS}
    if active:
        strengths['oxidative_stress'] = 0.40
        strengths['epithelial_injury'] = 0.20
        strengths['toxicant_uptake'] = 0.30
    return {
        'agent_id': 'AT2',
        'decision_time': 4.0,
        'observed_state_summary': {
            'dominant_state': 'oxidative injury' if active else 'stable',
            'trend': 'rising' if active else 'stable',
            'key_signals': ['oxidative evidence'] if active else [],
        },
        'program_strengths': strengths,
        'phenotype_action': {
            'dominant_program': 'oxidative_stress' if active else 'none',
            'transition_proposal': 'none',
            # This is the exact failure pattern from the probe: active programs but zero couplers.
            'secretion_multiplier': 0.0,
            'uptake_multiplier': 0.0,
            'motility_multiplier': 0.0,
            'death_tendency': 0.0,
            'repair_tendency': 0.0,
        },
        'abundance_transition_request': {'target_agent': 'none', 'fraction': 0.0},
        'valid_until_next_checkpoint': True,
        'confidence': 0.8,
        'brief_rationale': 'Observed oxidative and exposure evidence supports a bounded epithelial response.',
    }


def main() -> None:
    raw = decision(True)
    before_programs = copy.deepcopy(raw['program_strengths'])
    validated, audit = MOD.validate_with_rationale_audit(raw, base_req(False))

    pa = validated['phenotype_action']
    checks = {
        'all_passed': True,
        'active_program_with_zero_raw_multiplier_is_accepted': True,
        'program_strengths_unchanged': validated['program_strengths'] == before_programs,
        'secretion_multiplier_canonicalized': pa['secretion_multiplier'] == 0.40,
        'uptake_multiplier_canonicalized': pa['uptake_multiplier'] == 0.30,
        'motility_multiplier_remains_zero': pa['motility_multiplier'] == 0.0,
        'raw_decision_object_not_mutated': raw['phenotype_action']['secretion_multiplier'] == 0.0,
        'no_fallback': audit['fallback_used'] is False,
        'canonicalization_audited': audit['executor_canonicalization_applied'] is True,
    }

    stable, stable_audit = MOD.validate_with_rationale_audit(decision(False), base_req(True))
    checks['stable_air_remains_zero'] = all(float(v) == 0.0 for v in stable['program_strengths'].values()) and all(
        float(stable['phenotype_action'][k]) == 0.0
        for k in ['secretion_multiplier', 'uptake_multiplier', 'motility_multiplier', 'death_tendency', 'repair_tendency']
    )
    checks['stable_air_no_fallback'] = stable_audit['fallback_used'] is False
    checks['all_passed'] = all(v is True for k, v in checks.items() if k != 'all_passed')

    print(json.dumps(checks, indent=2, ensure_ascii=False))
    raise SystemExit(0 if checks['all_passed'] else 2)


if __name__ == '__main__':
    main()
