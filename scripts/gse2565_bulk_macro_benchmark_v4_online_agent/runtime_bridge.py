#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from online_controller import PROGRAMS, STATE, atomic_json, call_deepseek

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(
    os.environ.get(
        'LECAVC_OUTPUT_ROOT',
        str(ROOT / 'outputs/GSE2565_bulk_macro_benchmark_v4_online_agent'),
    )
).resolve()
ALPHA = 0.6

EVIDENCE_THRESHOLDS = {
    'cumulative_exposure': (0.005, 0.060),
    'oxidative_stress': (0.002, 0.020),
    'damage_memory': (0.010, 0.080),
    'epithelial_injury': (0.002, 0.030),
    'inflammatory_memory': (0.001, 0.020),
    'inflammation_state': (0.001, 0.020),
    'inflammation': (0.002, 0.020),
    'death_signal': (0.002, 0.020),
    'resolution_memory': (0.005, 0.025),
    'repair_state': (0.002, 0.020),
    'repair_signal': (0.002, 0.020),
    'edema_proxy': (0.002, 0.020),
}


def append_csv(path: Path | str, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = pd.DataFrame(rows)
    data.to_csv(path, mode='a', header=not path.exists(), index=False)


def stats(g: pd.DataFrame) -> dict:
    out = {}
    for c in STATE:
        x = pd.to_numeric(g[c], errors='coerce').to_numpy(float)
        if not np.isfinite(x).all():
            raise RuntimeError(f'nonfinite current state {c}')
        out[c] = {
            'mean': float(x.mean()),
            'median': float(np.median(x)),
            'minimum': float(x.min()),
            'maximum': float(x.max()),
            'standard_deviation': float(x.std(ddof=1)) if len(x) > 1 else 0.0,
        }
    out['stage_fractions'] = {str(i): float((g.stage_code == i).mean()) for i in range(4)}
    return out


def _norm(value: float, variable: str) -> float:
    off, on = EVIDENCE_THRESHOLDS[variable]
    return float(np.clip((float(value) - off) / (on - off), 0.0, 1.0))


def normalized_program_evidence(s: dict) -> dict:
    """Convert small raw model-state values into interpretable [0,1] evidence.

    This is a frozen monotonic transformation, not a target-time rule and not a replacement for
    the Agent. The Agent still chooses the action strength from evidence, trend, history and identity.
    """
    mean = lambda name: float(s[name]['mean'])
    exposure = _norm(mean('cumulative_exposure'), 'cumulative_exposure')
    oxidative = _norm(mean('oxidative_stress'), 'oxidative_stress')
    damage = _norm(mean('damage_memory'), 'damage_memory')
    injury_field = _norm(mean('epithelial_injury'), 'epithelial_injury')
    inflammatory = max(
        _norm(mean('inflammatory_memory'), 'inflammatory_memory'),
        _norm(mean('inflammation_state'), 'inflammation_state'),
        _norm(mean('inflammation'), 'inflammation'),
    )
    death = max(_norm(mean('death_signal'), 'death_signal'), damage)
    repair_core = max(
        _norm(mean('resolution_memory'), 'resolution_memory'),
        _norm(mean('repair_state'), 'repair_state'),
        _norm(mean('repair_signal'), 'repair_signal'),
    )
    post_exposure = float(np.clip(mean('post_exposure'), 0.0, 1.0))
    repair = repair_core * post_exposure
    edema = max(_norm(mean('edema_proxy'), 'edema_proxy'), damage)
    injury = max(damage, injury_field)
    oxidative_evidence = max(exposure, oxidative)
    uptake = exposure
    motility = max(inflammatory, injury, repair)

    return {
        'oxidative_stress': oxidative_evidence,
        'epithelial_injury': injury,
        'inflammation': inflammatory,
        'immune_recruitment': inflammatory,
        'edema_proxy': edema,
        'death_signal': death,
        'repair_signal': repair,
        'ecm_remodeling': repair,
        'toxicant_uptake': uptake,
        'motility': motility,
    }


def state_is_stable_baseline(s: dict) -> bool:
    """State-only stability test; intentionally independent of condition label."""
    return (
        s['toxicant_proxy']['mean'] < 0.005
        and s['cumulative_exposure']['mean'] < 0.005
        and s['damage_memory']['mean'] < 0.01
        and s['inflammatory_memory']['mean'] < 0.01
        and s['inflammation']['mean'] < 0.01
        and s['death_signal']['mean'] < 0.01
    )


def repair_evidence_present(s: dict) -> bool:
    return bool(
        s['post_exposure']['mean'] > 0.5
        and (
            s['damage_memory']['mean'] > 0.01
            or s['resolution_memory']['mean'] > 0.005
            or s['repair_signal']['mean'] > 0.002
        )
    )


def enrich_request_with_evidence(item: dict) -> dict:
    enriched = dict(item)
    s = enriched['current_worker_state_statistics']
    enriched['normalized_program_evidence'] = normalized_program_evidence(s)
    enriched['evidence_scale'] = {
        'range': '[0,1]',
        '0.00': 'no evidence',
        '0.25': 'mild/actionable evidence',
        '0.50': 'moderate evidence',
        '0.75': 'strong evidence',
        '1.00': 'saturated evidence',
        'normalization': 'clip((mean-off)/(on-off),0,1) with preregistered thresholds',
    }
    enriched['stable_air_evidence'] = state_is_stable_baseline(s)
    enriched['repair_evidence'] = repair_evidence_present(s)
    return enriched


def smooth(raw: dict, previous: dict | None) -> dict:
    if previous is None:
        return raw.copy()
    return {k: ALPHA * float(raw[k]) + (1 - ALPHA) * float(previous[k]) for k in raw}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--minute', type=int, required=True)
    parser.add_argument('--mock-profile', choices=['low', 'high'])
    parser.add_argument('--schema-test-agent')
    args = parser.parse_args()

    rd = Path(args.run_dir)
    rid = rd.name
    parts = rid.split('__')
    condition = parts[1]
    seed = int(parts[2].split('_')[-1])
    phase = 'schema_test' if args.schema_test_agent else rd.parents[1].name
    minute = args.minute

    if args.mock_profile and os.getenv('V4_MOCK_SLEEP_SECONDS'):
        time.sleep(float(os.environ['V4_MOCK_SLEEP_SECONDS']))

    cells = pd.read_csv(rd / f'work_cells_minute_{minute}.csv').sort_values('worker_id')
    history_path = rd / 'runtime_history.json'
    history = json.loads(history_path.read_text()) if history_path.exists() else {}
    agents = []
    input_rows = []

    for agent, group in cells.groupby('agent_id'):
        s = stats(group)
        old = history.get(agent, [])
        abundance = float(group.represented_abundance.sum())
        trends = [
            {
                'from_minute': q['minute'],
                'to_minute': minute,
                'mean_changes': {
                    k: s[k]['mean'] - q['state_statistics'][k]['mean']
                    for k in STATE
                },
            }
            for q in old[-2:]
        ]

        item = {
            'experiment_version': 'gse2565_v4_online_agent_amendment_002',
            'run_phase': phase,
            'run_id': rid,
            'condition': condition,
            'seed': seed,
            'current_time': minute / 60,
            'minute': minute,
            'agent_id': agent,
            'biological_cell_type': agent,
            'worker_count': len(group),
            'represented_abundance': abundance,
            'current_worker_state_statistics': s,
            'previous_agent_output': old[-1]['raw_decision'] if old else None,
            'previous_executed_action': old[-1]['executed_action'] if old else None,
            'recent_state_trends': trends,
            'cell_type_abundance_change': abundance - (old[-1]['represented_abundance'] if old else abundance),
            'allowed_program_enum': PROGRAMS,
            'biological_boundaries': [
                'current/past evidence only',
                'no future real expression and no target time',
                'cell-type compatibility enforced',
                'stable baseline pathological programs <=0.20',
                'all-zero action is invalid when compatible normalized evidence is actionable',
            ],
        }
        item = enrich_request_with_evidence(item)
        agents.append(item)
        input_rows.append(
            {
                'run_id': rid,
                'condition': condition,
                'seed': seed,
                'minute': minute,
                'agent_id': agent,
                'worker_count': len(group),
                'represented_abundance': abundance,
                'stable_air_evidence': item['stable_air_evidence'],
                'history_points': len(old),
                'state_json': json.dumps(s, sort_keys=True),
                'normalized_program_evidence_json': json.dumps(
                    item['normalized_program_evidence'], sort_keys=True
                ),
            }
        )

    payload = {
        'run_id': rid,
        'run_phase': phase,
        'condition': condition,
        'seed': seed,
        'minute': minute,
        'agents': agents,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(',', ':'))
    payload['input_hash'] = hashlib.sha256(canonical.encode()).hexdigest()
    reqpath = rd / 'runtime_requests' / rid / str(minute) / 'request.json'
    atomic_json(reqpath, payload)
    append_csv(OUT / 'audit/agent_input_state_audit.csv', input_rows)

    code_hash = hashlib.sha256(
        Path(__file__).read_bytes()
        + (Path(__file__).parent / 'online_controller.py').read_bytes()
    ).hexdigest()
    config_hash = hashlib.sha256((OUT / 'audit/online_agent_config.json').read_bytes()).hexdigest()

    def decide(item: dict):
        if args.schema_test_agent and item['agent_id'] != args.schema_test_agent:
            return None
        if args.mock_profile:
            value = 0.15 if args.mock_profile == 'low' else 0.65
            obj = {
                'agent_id': item['agent_id'],
                'decision_time': minute / 60,
                'observed_state_summary': {
                    'dominant_state': 'mock',
                    'trend': 'protocol',
                    'key_signals': [],
                },
                'program_strengths': {x: value for x in PROGRAMS},
                'phenotype_action': {
                    'dominant_program': 'oxidative_stress',
                    'transition_proposal': 'none',
                    'secretion_multiplier': value,
                    'uptake_multiplier': value,
                    'motility_multiplier': value,
                    'death_tendency': 0.0,
                    'repair_tendency': 0.0,
                },
                'abundance_transition_request': {
                    'target_agent': 'none',
                    'fraction': 0.0,
                },
                'valid_until_next_checkpoint': True,
                'confidence': 1.0,
                'brief_rationale': 'mock',
            }
            return item, obj, {
                'model': 'mock',
                'schema_valid': True,
                'fallback_used': False,
                'fresh_call': False,
                'cache_reused': False,
                'attempt': 1,
                'input_hash': payload['input_hash'],
                'prompt_hash': 'mock',
            }

        cache = (
            rd
            / 'runtime_cache'
            / phase
            / condition
            / str(seed)
            / str(minute)
            / f'{item["agent_id"]}.json'
        )
        raw = rd / 'agent_decisions' / str(minute) / item['agent_id']
        obj, meta = call_deepseek(item, cache, raw, code_hash, config_hash)
        return item, obj, meta

    selected = [
        x for x in agents
        if not args.schema_test_agent or x['agent_id'] == args.schema_test_agent
    ]
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(decide, x) for x in selected]
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())

    if args.schema_test_agent:
        item, obj, meta = results[0]
        envelope = {
            'run_id': rid,
            'run_phase': phase,
            'condition': condition,
            'seed': seed,
            'minute': minute,
            'input_hash': payload['input_hash'],
            'schema_valid': True,
            'fallback_used': False,
            'agents': [{'agent_id': item['agent_id'], 'raw_decision': obj}],
        }
        atomic_json(rd / 'schema_test_response.json', envelope)
        print(json.dumps({
            'schema_test_passed': True,
            'agent_id': item['agent_id'],
            'attempt': meta['attempt'],
        }))
        return

    if len(results) != 10:
        raise RuntimeError('checkpoint incomplete; refusing execution payload')

    flat = []
    full = []
    call_rows = []
    cache_rows = []
    rationale_rows = []

    for item, obj, meta in sorted(results, key=lambda x: x[0]['agent_id']):
        agent = item['agent_id']
        previous = history.get(agent, [])
        previous_executed = previous[-1]['executed_action'] if previous else None
        raw = {
            **obj['program_strengths'],
            **{
                k: obj['phenotype_action'][k]
                for k in [
                    'secretion_multiplier',
                    'uptake_multiplier',
                    'motility_multiplier',
                    'death_tendency',
                    'repair_tendency',
                ]
            },
        }
        executed = smooth(raw, previous_executed)
        row = {
            'agent_id': agent,
            'schema_valid': True,
            'fallback_used': False,
            **{'raw_' + k: float(v) for k, v in raw.items()},
            **{'executed_' + k: float(v) for k, v in executed.items()},
            'abundance_target': obj['abundance_transition_request']['target_agent'],
            'abundance_fraction': float(obj['abundance_transition_request']['fraction']),
            'confidence': float(obj['confidence']),
        }
        flat.append(row)
        full.append({
            'agent_id': agent,
            'raw_decision': obj,
            'executed_action': executed,
            'normalized_program_evidence': item['normalized_program_evidence'],
            'meta': meta,
        })
        history.setdefault(agent, []).append({
            'minute': minute,
            'state_statistics': item['current_worker_state_statistics'],
            'represented_abundance': item['represented_abundance'],
            'normalized_program_evidence': item['normalized_program_evidence'],
            'raw_decision': obj,
            'executed_action': executed,
        })
        call_rows.append({
            'run_phase': phase,
            'run_id': rid,
            'condition': condition,
            'seed': seed,
            'minute': minute,
            'agent_id': agent,
            'max_compatible_evidence': max(item['normalized_program_evidence'].values()),
            **meta,
        })
        cache_rows.append({
            'run_id': rid,
            'condition': condition,
            'seed': seed,
            'minute': minute,
            'agent_id': agent,
            'cache_reused': meta.get('cache_reused', False),
            'allowed_exact_resume': meta.get('cache_reused', False),
            'cross_run_reuse': False,
        })
        rationale_rows.append({
            'run_phase': phase,
            'run_id': rid,
            'condition': condition,
            'seed': seed,
            'minute': minute,
            'agent_id': agent,
            'rationale_original_length': meta.get('rationale_original_length', len(obj['brief_rationale'])),
            'rationale_stored_length': meta.get('rationale_stored_length', len(obj['brief_rationale'])),
            'rationale_truncated': meta.get('rationale_truncated', False),
            'rationale_original_hash': meta.get('rationale_original_hash', ''),
            'rationale_stored_hash': meta.get('rationale_stored_hash', ''),
            'rationale_original_text': meta.get('rationale_original_text', obj['brief_rationale']),
            'rationale_stored_text': obj['brief_rationale'],
            'fallback_used': False,
        })

    envelope = {
        'run_id': rid,
        'run_phase': phase,
        'condition': condition,
        'seed': seed,
        'minute': minute,
        'input_hash': payload['input_hash'],
        'schema_valid': True,
        'fallback_used': False,
        'agents': full,
    }
    response_dir = rd / 'runtime_responses' / rid / str(minute)
    atomic_json(response_dir / 'full_response.json', envelope)
    execution = {
        'run_id': rid,
        'condition': condition,
        'seed': seed,
        'minute': minute,
        'input_hash': payload['input_hash'],
        'schema_valid': True,
        'fallback_used': False,
        'agent_count': 10,
        'agents': flat,
    }
    atomic_json(response_dir / 'execution_payload.json', execution)
    atomic_json(history_path, history)
    append_csv(OUT / 'audit/deepseek_runtime_call_audit.csv', call_rows)
    append_csv(OUT / 'audit/cache_reuse_audit.csv', cache_rows)
    append_csv(OUT / 'audit/rationale_processing_audit.csv', rationale_rows)


if __name__ == '__main__':
    main()
