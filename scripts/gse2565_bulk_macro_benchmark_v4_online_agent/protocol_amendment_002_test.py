#!/usr/bin/env python3
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1] if HERE.name == 'gse2565_bulk_macro_benchmark_v4_online_agent' else Path.cwd()
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from online_controller import PROGRAMS, model_visible_request, validate
from runtime_bridge import enrich_request_with_evidence

OUT = ROOT / 'outputs/GSE2565_bulk_macro_benchmark_v4_online_agent'
RUNS = OUT / 'pilot/runs'


def find_run(condition: str) -> Path:
    matches = sorted(RUNS.glob(f'agent_physicell_v4_online__{condition}__seed_256501__to_72h'))
    if not matches:
        raise FileNotFoundError(f'cannot find completed seed 256501 run for {condition}: {RUNS}')
    return matches[0]


def load_agent(condition: str, minute: int, agent_id: str) -> dict:
    rd = find_run(condition)
    request_path = rd / 'runtime_requests' / rd.name / str(minute) / 'request.json'
    payload = json.loads(request_path.read_text())
    item = next(x for x in payload['agents'] if x['agent_id'] == agent_id)
    return enrich_request_with_evidence(item)


def zero_decision(req: dict) -> dict:
    return {
        'agent_id': req['agent_id'],
        'decision_time': req['current_time'],
        'observed_state_summary': {
            'dominant_state': 'stable',
            'trend': 'stable',
            'key_signals': [],
        },
        'program_strengths': {p: 0.0 for p in PROGRAMS},
        'phenotype_action': {
            'dominant_program': 'none',
            'transition_proposal': 'none',
            'secretion_multiplier': 0.0,
            'uptake_multiplier': 0.0,
            'motility_multiplier': 0.0,
            'death_tendency': 0.0,
            'repair_tendency': 0.0,
        },
        'abundance_transition_request': {'target_agent': 'none', 'fraction': 0.0},
        'valid_until_next_checkpoint': True,
        'confidence': 0.5,
        'brief_rationale': 'stable baseline',
    }


def plausible_decision(req: dict, strengths: dict, dominant: str, secretion: float, uptake: float = 0.0, motility: float = 0.0) -> dict:
    obj = zero_decision(req)
    obj['program_strengths'].update(strengths)
    obj['observed_state_summary'] = {
        'dominant_state': 'evidence-responsive',
        'trend': 'state-dependent',
        'key_signals': list(strengths),
    }
    obj['phenotype_action'].update({
        'dominant_program': dominant,
        'secretion_multiplier': secretion,
        'uptake_multiplier': uptake,
        'motility_multiplier': motility,
        'death_tendency': 0.1,
        'repair_tendency': 0.1,
    })
    obj['confidence'] = 0.8
    obj['brief_rationale'] = 'normalized evidence supports a compatible non-zero response'
    return obj


def main() -> None:
    air = load_agent('virtual_air_control', 240, 'AT2')
    at2 = load_agent('virtual_phosgene_injury', 240, 'AT2')
    macrophage = load_agent('virtual_phosgene_injury', 720, 'resident_macrophage')
    fibroblast = load_agent('virtual_phosgene_injury', 1440, 'fibroblast_stromal')

    hidden = {'condition', 'seed', 'run_id', 'run_phase', 'minute', 'input_hash'}
    visible = model_visible_request(at2)
    identity_hidden = hidden.isdisjoint(visible)
    if not identity_hidden:
        raise AssertionError(f'runtime identity leaked to model: {sorted(hidden & set(visible))}')

    validate(zero_decision(air), air)

    rejected = []
    for label, req in [
        ('phosgene_AT2_4h', at2),
        ('phosgene_resident_macrophage_12h', macrophage),
        ('phosgene_fibroblast_24h', fibroblast),
    ]:
        try:
            validate(zero_decision(req), req)
        except ValueError as e:
            rejected.append({'case': label, 'rejected': True, 'error': str(e)})
        else:
            raise AssertionError(f'all-zero decision was incorrectly accepted: {label}')

    valid_examples = [
        plausible_decision(
            at2,
            {'oxidative_stress': 0.60, 'epithelial_injury': 0.20, 'toxicant_uptake': 0.50},
            'oxidative_stress',
            secretion=0.50,
            uptake=0.50,
        ),
        plausible_decision(
            macrophage,
            {'inflammation': 0.55, 'immune_recruitment': 0.45, 'repair_signal': 0.35, 'motility': 0.30},
            'inflammation',
            secretion=0.50,
            motility=0.40,
        ),
        plausible_decision(
            fibroblast,
            {'repair_signal': 0.70, 'ecm_remodeling': 0.65, 'motility': 0.20},
            'repair_signal',
            secretion=0.50,
            motility=0.20,
        ),
    ]
    for req, obj in zip([at2, macrophage, fibroblast], valid_examples):
        validate(copy.deepcopy(obj), req)

    result = {
        'protocol_amendment': '002',
        'all_passed': True,
        'deepseek_called': False,
        'runtime_identity_hidden_from_model': identity_hidden,
        'air_zero_decision_accepted': True,
        'phosgene_all_zero_decisions_rejected': rejected,
        'plausible_nonzero_decisions_accepted': True,
        'evidence_examples': {
            'air_AT2_4h': air['normalized_program_evidence'],
            'phosgene_AT2_4h': at2['normalized_program_evidence'],
            'phosgene_resident_macrophage_12h': macrophage['normalized_program_evidence'],
            'phosgene_fibroblast_24h': fibroblast['normalized_program_evidence'],
        },
    }
    audit = OUT / 'audit/protocol_amendment_002_regression.json'
    audit.parent.mkdir(parents=True, exist_ok=True)
    audit.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
