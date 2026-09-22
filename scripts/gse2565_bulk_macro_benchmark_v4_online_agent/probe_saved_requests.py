#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1] if HERE.name == 'gse2565_bulk_macro_benchmark_v4_online_agent' else Path.cwd()
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from online_controller import allowed_programs, call_deepseek
from runtime_bridge import enrich_request_with_evidence

OUT = ROOT / 'outputs/GSE2565_bulk_macro_benchmark_v4_online_agent'
RUNS = OUT / 'pilot/runs'

CASES = [
    ('air_AT2_4h', 'virtual_air_control', 240, 'AT2'),
    ('phosgene_AT2_4h', 'virtual_phosgene_injury', 240, 'AT2'),
    ('phosgene_resident_macrophage_12h', 'virtual_phosgene_injury', 720, 'resident_macrophage'),
    ('phosgene_fibroblast_24h', 'virtual_phosgene_injury', 1440, 'fibroblast_stromal'),
]


def find_run(condition: str) -> Path:
    rd = RUNS / f'agent_physicell_v4_online__{condition}__seed_256501__to_72h'
    if not rd.exists():
        raise FileNotFoundError(f'missing completed stage1 run: {rd}')
    return rd


def load_request(condition: str, minute: int, agent_id: str, label: str) -> dict:
    rd = find_run(condition)
    payload = json.loads((rd / 'runtime_requests' / rd.name / str(minute) / 'request.json').read_text())
    item = next(x for x in payload['agents'] if x['agent_id'] == agent_id)
    item = enrich_request_with_evidence(item)
    item['run_phase'] = 'semantic_probe_002'
    item['run_id'] = f'{item["run_id"]}__{label}'
    return item


def main() -> None:
    if not os.getenv('DEEPSEEK_API_KEY'):
        raise RuntimeError('DEEPSEEK_API_KEY is not set')

    config = OUT / 'audit/online_agent_config.json'
    if not config.exists():
        raise FileNotFoundError(config)
    code_hash = hashlib.sha256(
        (HERE / 'online_controller.py').read_bytes() + (HERE / 'runtime_bridge.py').read_bytes()
    ).hexdigest()
    config_hash = hashlib.sha256(config.read_bytes()).hexdigest()

    probe_root = OUT / 'audit/semantic_probe_002'
    results = []
    for label, condition, minute, agent_id in CASES:
        req = load_request(condition, minute, agent_id, label)
        case_dir = probe_root / label
        obj, meta = call_deepseek(
            req,
            case_dir / 'cache/decision.json',
            case_dir / 'raw',
            code_hash,
            config_hash,
        )
        allowed = allowed_programs(agent_id)
        compatible_strengths = {
            p: float(v)
            for p, v in obj['program_strengths'].items()
            if p in allowed
        }
        row = {
            'label': label,
            'condition_for_audit_only': condition,
            'minute': minute,
            'agent_id': agent_id,
            'attempt': meta['attempt'],
            'normalized_program_evidence': req['normalized_program_evidence'],
            'program_strengths': obj['program_strengths'],
            'phenotype_action': obj['phenotype_action'],
            'dominant_program': obj['phenotype_action']['dominant_program'],
            'max_compatible_strength': max(compatible_strengths.values()),
            'brief_rationale': obj['brief_rationale'],
            'fallback_used': False,
        }
        results.append(row)
        print(json.dumps(row, indent=2, ensure_ascii=False))

    by_label = {x['label']: x for x in results}
    passed = (
        by_label['air_AT2_4h']['max_compatible_strength'] <= 0.20
        and by_label['phosgene_AT2_4h']['max_compatible_strength'] >= 0.15
        and by_label['phosgene_resident_macrophage_12h']['max_compatible_strength'] >= 0.15
        and by_label['phosgene_fibroblast_24h']['max_compatible_strength'] >= 0.15
    )
    report = {
        'all_passed': passed,
        'deepseek_calls_expected': 4,
        'fallback_used': False,
        'cases': results,
    }
    (probe_root / 'semantic_probe_002_result.json').write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + '\n'
    )
    print(json.dumps({'all_passed': passed, 'report': str(probe_root / 'semantic_probe_002_result.json')}, ensure_ascii=False))
    raise SystemExit(0 if passed else 2)


if __name__ == '__main__':
    main()
