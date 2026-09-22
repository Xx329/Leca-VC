#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from urllib import request
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/leca_vc_prompt_isolation_v1'))
from prompt_firewall import (  # noqa: E402
    PROTOCOL_ID,
    sanitize_gse2565_request,
    validate_response_text,
    write_message_audit,
)

PROGRAMS = [
    'oxidative_stress',
    'epithelial_injury',
    'inflammation',
    'immune_recruitment',
    'edema_proxy',
    'death_signal',
    'repair_signal',
    'ecm_remodeling',
    'toxicant_uptake',
    'motility',
]
STATE = [
    'toxicant_proxy',
    'oxidative_stress',
    'epithelial_injury',
    'inflammation',
    'edema_proxy',
    'death_signal',
    'repair_signal',
    'cumulative_exposure',
    'exposure_peak',
    'damage_memory',
    'inflammatory_memory',
    'resolution_memory',
    'injury_state',
    'inflammation_state',
    'repair_state',
    'sustained_injury_duration',
    'post_exposure',
]

TEMPLATE = {
    'agent_id': 'AT1',
    'decision_time': 0.0,
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
    'abundance_transition_request': {
        'target_agent': 'none',
        'fraction': 0.0,
    },
    'valid_until_next_checkpoint': True,
    'confidence': 0.0,
    'brief_rationale': '不超过80字',
}

MODEL_VISIBLE_KEYS = {
    'current_time',
    'agent_id',
    'biological_cell_type',
    'worker_count',
    'represented_abundance',
    'current_worker_state_statistics',
    'previous_agent_output',
    'previous_executed_action',
    'recent_state_trends',
    'cell_type_abundance_change',
    'allowed_program_enum',
    'biological_boundaries',
    'stable_air_evidence',
    'repair_evidence',
    'normalized_program_evidence',
    'evidence_scale',
}

SECRETION_PROGRAMS = {
    'oxidative_stress',
    'epithelial_injury',
    'inflammation',
    'immune_recruitment',
    'edema_proxy',
    'death_signal',
    'repair_signal',
    'ecm_remodeling',
}


def sha_bytes(x: bytes) -> str:
    return hashlib.sha256(x).hexdigest()


def atomic_json(path: Path | str, obj: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n')
    os.replace(tmp, path)


def allowed_programs(agent: str) -> set[str]:
    if agent in {'AT1', 'AT2', 'airway_epithelial', 'injured_Krt8_epithelial'}:
        return {
            'oxidative_stress',
            'epithelial_injury',
            'death_signal',
            'toxicant_uptake',
            'repair_signal',
            'motility',
            'inflammation',
            'immune_recruitment',
        }
    if agent == 'endothelial':
        return {
            'oxidative_stress',
            'epithelial_injury',
            'edema_proxy',
            'death_signal',
            'repair_signal',
            'toxicant_uptake',
            'motility',
        }
    if agent in {
        'resident_macrophage',
        'recruited_monocyte_macrophage',
        'neutrophil',
        'lymphoid_immune',
    }:
        return {
            'inflammation',
            'immune_recruitment',
            'motility',
            'death_signal',
            'repair_signal',
            'toxicant_uptake',
            'oxidative_stress',
        }
    if agent == 'fibroblast_stromal':
        return {
            'repair_signal',
            'ecm_remodeling',
            'motility',
            'oxidative_stress',
        }
    return set(PROGRAMS)


def cell_type_guidance(agent: str) -> str:
    if agent in {'AT1', 'AT2', 'airway_epithelial', 'injured_Krt8_epithelial'}:
        return (
            'Epithelial Agent: prioritize oxidative stress, epithelial injury, toxicant uptake '
            'and death when supported; repair is allowed only with resolution/post-exposure evidence.'
        )
    if agent == 'endothelial':
        return (
            'Endothelial Agent: prioritize oxidative stress, barrier injury, edema and death; '
            'repair is allowed only with recovery evidence.'
        )
    if agent == 'resident_macrophage':
        return (
            'Resident macrophage Agent: inflammation and immune recruitment are appropriate during '
            'injury; transition toward repair is appropriate only when resolution evidence rises.'
        )
    if agent in {'recruited_monocyte_macrophage', 'neutrophil', 'lymphoid_immune'}:
        return (
            'Immune Agent: prioritize inflammation, recruitment and motility when inflammatory '
            'evidence is present; avoid unsupported repair or epithelial programs.'
        )
    if agent == 'fibroblast_stromal':
        return (
            'Fibroblast/stromal Agent: repair and ECM are appropriate only with post-exposure or '
            'resolution evidence; do not activate strong early ECM without such evidence.'
        )
    return 'Use only programs compatible with the biological cell type.'


def model_visible_request(req: dict) -> dict:
    """Return the biology-only prompt payload.

    Runtime identity (condition, seed, run_id, phase, minute and hashes) remains in the trusted
    Python envelope/cache key and is intentionally hidden from the model.
    """
    visible = {k: req[k] for k in MODEL_VISIBLE_KEYS if k in req}
    if 'stable_air_evidence' in visible:
        visible['stable_baseline_evidence'] = visible.pop('stable_air_evidence')
    visible['cell_type_decision_guidance'] = cell_type_guidance(req['agent_id'])
    visible['decision_semantics'] = {
        'evidence_interpretation': {
            '0.00': 'no evidence',
            '0.25': 'mild but actionable evidence',
            '0.50': 'moderate evidence',
            '0.75': 'strong evidence',
            '1.00': 'very strong/saturated evidence',
        },
        'requirements': [
            'Use current and past observations only.',
            'All-zero actions are acceptable only when all compatible evidence is below 0.25 and no compatible trend is rising.',
            'When compatible evidence is at least 0.25, choose at least one biologically compatible non-zero program.',
            'When compatible evidence is at least 0.75, at least one compatible program should be clearly active (normally at least 0.15).',
            'If a secretion-related program is active, secretion_multiplier must also be non-zero.',
            'If toxicant_uptake is active, uptake_multiplier must also be non-zero.',
            'If motility or immune_recruitment is active, motility_multiplier must also be non-zero.',
            'Do not simply copy the evidence values; make a bounded biological decision proportional to evidence, trend and cell identity.',
        ],
    }
    return visible


def _normalize_brief_rationale(o: dict) -> tuple[dict, dict]:
    if 'brief_rationale' not in o:
        raise ValueError('brief_rationale missing')
    original = o['brief_rationale']
    if not isinstance(original, str):
        raise ValueError('brief_rationale must be string')
    stripped = original.strip()
    truncated = len(stripped) > 80
    stored = stripped[:79] + '…' if truncated else stripped
    normalized = dict(o)
    normalized['brief_rationale'] = stored
    audit = {
        'rationale_original_length': len(original),
        'rationale_stored_length': len(stored),
        'rationale_truncated': truncated,
        'rationale_original_hash': sha_bytes(original.encode('utf-8')),
        'rationale_stored_hash': sha_bytes(stored.encode('utf-8')),
        'rationale_original_text': original,
        'rationale_stored_text': stored,
    }
    return normalized, audit


def _canonicalize_executor_fields(o: dict) -> tuple[dict, dict]:
    """Derive redundant executor couplers from the LLM's primary program decisions.

    The DeepSeek program_strengths remain unchanged and are the biological decision. The raw API
    response is already preserved on disk. These phenotype fields are executor couplers used by
    PhysiCell; canonicalising them prevents a valid non-zero program from being silently multiplied
    by a contradictory zero multiplier.
    """
    normalized = dict(o)
    pa = dict(o['phenotype_action'])
    strengths = {p: float(o['program_strengths'][p]) for p in PROGRAMS}

    original = {
        'secretion_multiplier': float(pa['secretion_multiplier']),
        'uptake_multiplier': float(pa['uptake_multiplier']),
        'motility_multiplier': float(pa['motility_multiplier']),
        'death_tendency': float(pa['death_tendency']),
        'repair_tendency': float(pa['repair_tendency']),
    }
    required = {
        'secretion_multiplier': max(strengths[p] for p in SECRETION_PROGRAMS),
        'uptake_multiplier': strengths['toxicant_uptake'],
        'motility_multiplier': max(strengths['motility'], strengths['immune_recruitment']),
        'death_tendency': strengths['death_signal'],
        'repair_tendency': max(strengths['repair_signal'], strengths['ecm_remodeling']),
    }
    canonical = {k: max(original[k], required[k]) for k in original}
    for k, v in canonical.items():
        pa[k] = float(v)
    normalized['phenotype_action'] = pa
    audit = {
        'executor_canonicalization_applied': any(abs(canonical[k] - original[k]) > 1e-12 for k in original),
        'executor_fields_original': original,
        'executor_fields_required_from_programs': required,
        'executor_fields_canonical': canonical,
        'program_strengths_unchanged': True,
        'fallback_used': False,
    }
    return normalized, audit


def _semantic_consistency(o: dict, req: dict) -> None:
    evidence = req.get('normalized_program_evidence') or {}
    allowed = allowed_programs(req['agent_id'])
    strengths = {p: float(o['program_strengths'][p]) for p in PROGRAMS}

    compatible_evidence = {
        p: float(evidence.get(p, 0.0))
        for p in PROGRAMS
        if p in allowed and p != 'motility'
    }
    if not compatible_evidence:
        return

    strongest_program = max(compatible_evidence, key=compatible_evidence.get)
    strongest_evidence = compatible_evidence[strongest_program]
    responsive_programs = [p for p, e in compatible_evidence.items() if e >= 0.25]
    responsive_strength = max((strengths[p] for p in responsive_programs), default=0.0)

    stable = bool(req.get('stable_air_evidence', False))
    if not stable and strongest_evidence >= 0.25 and responsive_strength < 0.05:
        raise ValueError(
            'decision-state inconsistency: compatible evidence is actionable '
            f'({strongest_program}={strongest_evidence:.3f}) but all supported programs are <0.05'
        )
    if not stable and strongest_evidence >= 0.75 and responsive_strength < 0.15:
        raise ValueError(
            'decision-state inconsistency: strong compatible evidence '
            f'({strongest_program}={strongest_evidence:.3f}) requires at least one compatible program >=0.15'
        )

    max_program = max(PROGRAMS, key=lambda p: strengths[p])
    max_strength = strengths[max_program]
    dominant = o['phenotype_action']['dominant_program']
    if max_strength > 0.05 and dominant == 'none':
        raise ValueError('dominant_program cannot be none when a program strength exceeds 0.05')
    if dominant != 'none':
        if dominant not in allowed:
            raise ValueError(f'dominant_program {dominant} is incompatible with {req["agent_id"]}')
        if strengths[dominant] + 0.05 < max_strength:
            raise ValueError('dominant_program is not one of the strongest active programs')

    pa = o['phenotype_action']
    secretion_active = max(strengths[p] for p in SECRETION_PROGRAMS) > 0.05
    if secretion_active and float(pa['secretion_multiplier']) < 0.05:
        raise ValueError('active secretion-related program requires secretion_multiplier >=0.05')
    if strengths['toxicant_uptake'] > 0.05 and float(pa['uptake_multiplier']) < 0.05:
        raise ValueError('active toxicant_uptake requires uptake_multiplier >=0.05')
    if max(strengths['motility'], strengths['immune_recruitment']) > 0.05 and float(pa['motility_multiplier']) < 0.05:
        raise ValueError('active motility/recruitment requires motility_multiplier >=0.05')


def validate_with_rationale_audit(o: dict, req: dict) -> tuple[dict, dict]:
    top = {
        'agent_id',
        'decision_time',
        'observed_state_summary',
        'program_strengths',
        'phenotype_action',
        'abundance_transition_request',
        'valid_until_next_checkpoint',
        'confidence',
        'brief_rationale',
    }
    if not isinstance(o, dict) or set(o) != top:
        got = sorted(o) if isinstance(o, dict) else type(o).__name__
        raise ValueError(f'top-level keys expected {sorted(top)}, got {got}')
    o, rationale_audit = _normalize_brief_rationale(o)

    if o['agent_id'] != req['agent_id']:
        raise ValueError('agent_id conflict')
    if abs(float(o['decision_time']) - float(req['current_time'])) > 1e-9:
        raise ValueError('decision_time conflict')
    if set(o['program_strengths']) != set(PROGRAMS):
        raise ValueError('program_strength keys')

    pa = o['phenotype_action']
    expected_pa = {
        'dominant_program',
        'transition_proposal',
        'secretion_multiplier',
        'uptake_multiplier',
        'motility_multiplier',
        'death_tendency',
        'repair_tendency',
    }
    if set(pa) != expected_pa:
        raise ValueError('phenotype_action keys')

    ar = o['abundance_transition_request']
    if set(ar) != {'target_agent', 'fraction'}:
        raise ValueError('abundance keys')

    vals = [
        *o['program_strengths'].values(),
        pa['secretion_multiplier'],
        pa['uptake_multiplier'],
        pa['motility_multiplier'],
        pa['death_tendency'],
        pa['repair_tendency'],
        ar['fraction'],
        o['confidence'],
    ]
    if not all(np.isfinite(float(v)) and 0 <= float(v) <= 1 for v in vals):
        raise ValueError('finite [0,1] bounds')
    if float(ar['fraction']) > 0.02:
        raise ValueError('abundance fraction >0.02')

    allowed_targets = {
        'AT1': 'injured_Krt8_epithelial',
        'AT2': 'injured_Krt8_epithelial',
        'airway_epithelial': 'injured_Krt8_epithelial',
        'resident_macrophage': 'recruited_monocyte_macrophage',
    }
    if ar['target_agent'] != 'none' and ar['target_agent'] != allowed_targets.get(req['agent_id']):
        raise ValueError('illegal abundance target')
    if pa['dominant_program'] not in PROGRAMS + ['none']:
        raise ValueError('dominant_program enum')

    allowed = allowed_programs(req['agent_id'])
    stable = bool(req.get('stable_air_evidence', False))
    for p, v in o['program_strengths'].items():
        if p not in allowed and float(v) > 0.15:
            raise ValueError(f'{req["agent_id"]} incompatible high program {p}')

    if stable:
        for p in ['inflammation', 'immune_recruitment', 'death_signal', 'repair_signal', 'ecm_remodeling']:
            if float(o['program_strengths'][p]) > 0.20:
                raise ValueError(f'stable baseline boundary: {p}>0.20')
        if float(ar['fraction']) > 0.005:
            raise ValueError('stable baseline abundance fraction>0.005')

    if req['agent_id'] == 'fibroblast_stromal' and not req.get('repair_evidence', False):
        if float(o['program_strengths']['ecm_remodeling']) > 0.15:
            raise ValueError('fibroblast ECM without repair evidence')

    o, executor_audit = _canonicalize_executor_fields(o)
    _semantic_consistency(o, req)
    return o, {**rationale_audit, **executor_audit}


def validate(o: dict, req: dict) -> dict:
    return validate_with_rationale_audit(o, req)[0]


def call_deepseek(req: dict, cache_path: Path | str, raw_dir: Path | str, code_hash: str, config_hash: str):
    key = os.getenv('DEEPSEEK_API_KEY')
    if not key:
        raise RuntimeError('DEEPSEEK_API_KEY is not set')

    model = 'deepseek-chat'
    temperature = 0.1
    deidentified = os.getenv('LECAVC_PROMPT_MODE') == 'deidentified_v1'
    example = json.loads(json.dumps(TEMPLATE))
    example['agent_id'] = req['agent_id']
    base_model_req = model_visible_request(req)
    if deidentified:
        model_req, normalized_step = sanitize_gse2565_request(base_model_req, req)
        example['decision_time'] = normalized_step
    else:
        model_req = base_model_req
        normalized_step = float(req['current_time'])
        example['decision_time'] = req['current_time']
    schema = json.dumps(example, ensure_ascii=False, indent=2)

    domain = (
        'a generic multicellular perturbation simulation'
        if deidentified else 'a virtual lung injury model'
    )
    system = (
        'You are the online biological decision layer for one cell-type Agent in ' + domain + '. '
        'Return only strict JSON matching the request-specific template exactly. Keep agent_id and decision_time '
        'exactly as shown, preserve every key, and do not add runtime-envelope identity. Use only the current/past '
        'state in the user payload. normalized_program_evidence is already scaled to [0,1]: 0=no evidence, '
        '0.25=mild actionable, 0.50=moderate, 0.75=strong, 1=saturated. Do not output an all-zero decision when '
        'compatible evidence is >=0.25. When compatible evidence is >=0.75, at least one compatible program should '
        'normally be >=0.15. Program strength is not the same as evidence: integrate evidence, trend, previous action '
        'and cell identity. program_strengths are the primary biological decision. Phenotype multipliers should be '
        'consistent with active programs; the trusted controller will deterministically canonicalize redundant executor '
        'couplers from program strengths while preserving the raw response and every program strength unchanged. '
        'Do not reveal hidden reasoning; brief_rationale is a short auditable conclusion.\nTemplate:\n'
        + schema
        + '\nPrograms enum: '
        + ','.join(PROGRAMS)
    )

    prompt = json.dumps(model_req, ensure_ascii=False, sort_keys=True)
    input_hash = sha_bytes(prompt.encode())
    prompt_hash = sha_bytes((system + '\n' + prompt).encode())

    cache_path = Path(cache_path)
    meta_path = cache_path.with_suffix('.meta.json')
    expected = {
        'run_phase': req['run_phase'],
        'run_id': req['run_id'],
        'condition': req['condition'],
        'seed': req['seed'],
        'minute': req['minute'],
        'agent_id': req['agent_id'],
        'input_hash': input_hash,
        'prompt_hash': prompt_hash,
        'model': model,
        'code_hash': code_hash,
        'configuration_hash': config_hash,
        'prompt_protocol': PROTOCOL_ID if deidentified else 'historical_v4',
    }

    if cache_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        obj = json.loads(cache_path.read_text())
        if all(meta.get(k) == v for k, v in expected.items()):
            obj, rationale_audit = validate_with_rationale_audit(obj, req)
            return obj, {**rationale_audit, **meta, 'fresh_call': False, 'cache_reused': True}

    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(raw_dir / 'model_visible_request.json', model_req)
    messages = [
        {'role': 'system', 'content': system},
        {'role': 'user', 'content': prompt},
    ]
    last = None

    for attempt in range(1, 5):
        content = None
        try:
            if deidentified:
                write_message_audit(
                    raw_dir,
                    messages,
                    dataset_role='GSE2565',
                    trusted_envelope=expected,
                )
            body = {
                'model': model,
                'temperature': temperature,
                'response_format': {'type': 'json_object'},
                'messages': messages,
            }
            rq = request.Request(
                'https://api.deepseek.com/chat/completions',
                data=json.dumps(body).encode(),
                headers={
                    'Authorization': 'Bearer ' + key,
                    'Content-Type': 'application/json',
                },
                method='POST',
            )
            with request.urlopen(rq, timeout=120) as r:
                raw = r.read()
            (raw_dir / f'attempt_{attempt}_raw.json').write_bytes(raw + b'\n')
            content = json.loads(raw)['choices'][0]['message']['content']
            (raw_dir / f'attempt_{attempt}_content.json').write_text(content + '\n')
            obj = json.loads(content)
            if deidentified:
                if abs(float(obj.get('decision_time', -1)) - normalized_step) > 1e-9:
                    raise ValueError('decision_time does not match normalized step')
                validate_response_text(obj, dataset_role='GSE2565')
                # The executor uses trusted physical time.  Only this deterministic
                # orchestration mapping restores it after model-output validation.
                obj['decision_time'] = req['current_time']
            obj, rationale_audit = validate_with_rationale_audit(obj, req)
            atomic_json(cache_path, obj)
            meta = {
                **expected,
                'response_hash': sha_bytes(cache_path.read_bytes()),
                'temperature': temperature,
                'timestamp': time.time(),
                'schema_valid': True,
                'fresh_call': True,
                'cache_reused': False,
                'fallback_used': False,
                'attempt': attempt,
                **rationale_audit,
            }
            atomic_json(meta_path, meta)
            atomic_json(raw_dir / 'rationale_processing_audit.json', rationale_audit)
            return obj, meta
        except Exception as e:
            last = e
            (raw_dir / f'attempt_{attempt}_error.txt').write_text(type(e).__name__ + ': ' + str(e) + '\n')
            if content is not None:
                messages.extend(
                    [
                        {'role': 'assistant', 'content': content},
                        {
                            'role': 'user',
                            'content': (
                                'Validation error: '
                                + str(e)
                                + '. Re-read normalized_program_evidence, current state, trend and cell-type guidance. '
                                'Return one complete corrected JSON object matching the template. Preserve agent_id and decision_time. '
                                'Do not change the schema and do not return unsupported all-zero actions.'
                            ),
                        },
                    ]
                )

    raise RuntimeError(
        f'DeepSeek schema/semantic validation failed after 4 attempts for {req["agent_id"]}: {last}; no fallback'
    )
