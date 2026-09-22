#!/usr/bin/env python3
"""Import only V2's pre-d21-frozen t=0 cell policies into the independent V3 schedule."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
from common import DEFAULT_OUT, dump_json, ensure_dirs, hash_paths, read_json, resolve_out, sha256
from generate_cell_agent_policies import POLICY_FIELDS, first_person_payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--source-dir", type=Path, help="Validated pre-d21 t=0 policy cache to revalidate against this run's d7 prompts")
    args = parser.parse_args()
    root = args.project_root.resolve()
    out = resolve_out(root, args.out_dir)
    ensure_dirs(out)
    source = (args.source_dir or root / "outputs/GSE267904_fibrosis_application_v2").resolve()
    frozen = read_json(source / "model_frozen.json", {})
    source_policies = sorted((source / "llm/policies").glob("*_cell_agent_policies.csv"))
    cache_audit = read_json(source / "audit/llm_cell_agent_policy_audit.json", {})
    is_v2_frozen = frozen.get("completed_real_physicell_runs") == 418
    is_prompt_revalidated_cache = args.source_dir is not None and cache_audit.get("heldout_d21_visible") is False
    if len(source_policies) != 12 or not (is_v2_frozen or is_prompt_revalidated_cache):
        raise RuntimeError("Source must contain 12 frozen V2 policies or a 12-section audit-confirmed pre-d21 t=0 cache")
    actual_hash = hash_paths(source_policies)
    expected_hash = frozen.get("model_hashes", {}).get("llm_cell_agent_policies")
    if is_v2_frozen and actual_hash != expected_hash:
        raise RuntimeError("V2 t=0 policy hash no longer matches model_frozen.json")
    rows = []
    source_files = []
    for path in source_policies:
        table = pd.read_csv(path)
        if len(table) != 50 or not set(POLICY_FIELDS).issubset(table.columns):
            raise RuntimeError(f"Invalid frozen V2 policy file {path}")
        sample_id = path.name.removesuffix("_cell_agent_policies.csv")
        registry_path = out / f"agents/{sample_id}_cell_agents.csv"
        if not registry_path.exists():
            raise FileNotFoundError(f"Missing R1 registry for prompt validation: {registry_path}")
        registry = pd.read_csv(registry_path)
        if set(table.agent_id.astype(str)) != set(registry.agent_id.astype(str)):
            raise RuntimeError(f"{sample_id} source policy agent IDs do not match the R1 registry")
        prompt_matches = []
        for agent in registry.itertuples(index=False):
            source_prompt = source / "llm/raw" / sample_id / f"{agent.agent_id}_prompt.json"
            if not source_prompt.exists():
                raise FileNotFoundError(f"Missing source prompt required for cache validation: {source_prompt}")
            expected_prompt = hashlib.sha256(json.dumps(first_person_payload(pd.Series(agent._asdict())), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
            prompt_matches.append(read_json(source_prompt, {}).get("prompt_sha256") == expected_prompt)
        if not all(prompt_matches):
            raise RuntimeError(f"{sample_id} source t=0 cache does not match the R1 initial prompts")
        table.insert(2, "checkpoint", 0)
        destination = out / f"llm/policies/{sample_id}_cell_agent_policy_schedule.csv"
        table[["agent_id", "cell_type", "checkpoint", *POLICY_FIELDS]].to_csv(destination, index=False)
        rows.append({"sample_id": sample_id, "n_agents": 50, "prompt_hashes_match_r1": True, "schedule_sha256": sha256(destination)})
        source_files.append({"path": str(path.resolve()), "sha256": sha256(path)})
    dump_json(out / "audit/imported_t0_policy_audit.json", {
        "source_experiment": "GSE267904 fibrosis application V2" if is_v2_frozen else "GSE267904 fibrosis application V3 pre-d21 t=0 cache",
        "source_model_frozen_sha256": sha256(source / "model_frozen.json") if (source / "model_frozen.json").exists() else None,
        "source_policy_set_sha256": actual_hash,
        "source_generated_before_d21_access": True,
        "source_kind": "frozen_v2" if is_v2_frozen else "prompt_revalidated_t0_cache",
        "V3_d21_opened": False,
        "source_files": source_files,
        "samples": rows,
    })
    print("Imported 600 hash-verified frozen t=0 cell policies into V3 schedules")


if __name__ == "__main__":
    main()
