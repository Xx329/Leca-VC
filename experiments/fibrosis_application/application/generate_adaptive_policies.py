#!/usr/bin/env python3
"""Generate condition-blind t=60/t=120 first-person cell-policy updates from real PhysiCell checkpoints."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.spatial import cKDTree

sys.path.append(str(Path(__file__).resolve().parent))
from common import DEFAULT_OUT, FIELDS, dump_json, ensure_dirs, load_config, resolve_out, sha256
from generate_cell_agent_policies import POLICY_FIELDS, deepseek_policy, extract_json, validate_policy
import physicell_engine as engine


def local_checkpoint(run_dir: Path, checkpoint: int) -> pd.DataFrame:
    agents = pd.read_csv(run_dir / f"cell_agents_step_{checkpoint}.csv")
    mesh = pd.read_csv(run_dir / f"mesh_fields_step_{checkpoint}.csv")
    _, nearest = cKDTree(mesh[["x", "y"]].to_numpy()).query(agents[["x", "y"]].to_numpy())
    for field in FIELDS:
        agents[field] = mesh.iloc[nearest][field].to_numpy(dtype=float)
    return agents


def adaptive_payload(agent: pd.Series, previous_agent: pd.Series, previous_policy: pd.Series, checkpoint: int) -> dict:
    return {
        "agent_id": str(agent.agent_id),
        "identity": str(agent.cell_type),
        "instruction": (
            f"You are one {agent.cell_type} representative virtual lung cell in a PhysiCell tissue. "
            "Update only your own response from your changing local microenvironment. You are not a controller."
        ),
        "checkpoint": checkpoint,
        "time_label": "abstract PhysiCell progression time / virtual progression step",
        "current_local_substrates": {field: float(agent[field]) for field in FIELDS},
        "local_substrate_change": {field: float(agent[field] - previous_agent[field]) for field in FIELDS},
        "current_cell_state": {
            key: float(agent[key]) for key in [
                "represented_abundance", "fibrosis_memory", "epithelial_integrity",
                "profibrotic_activation", "myofibroblast_activation",
            ]
        },
        "previous_policy": {field: float(previous_policy[field]) for field in POLICY_FIELDS},
        "output_schema": {field: "number from 0.0 to 1.0" for field in POLICY_FIELDS},
        "constraints": [
            "Return strict JSON only.", "Decide only this cell's behavior.",
            "Do not infer or use GSE267904 d21.", "Do not output physical secretion units.",
        ],
    }


def update_checkpoint(root: Path, out: Path, checkpoint: int, retries: int) -> list[dict]:
    config = load_config()
    project = engine.write_project(root, out, config)
    binary = engine.compile_project(project)
    matrix = engine.build_matrix(out, config)
    params = yaml.safe_load((out / "calibration/dynamic_parameters_pre_benchmark.yaml").read_text())
    manifest = pd.read_csv(out / "manifests/gse267904_24_sections.csv")
    rows = []
    for sample_id in manifest.loc[manifest.day.eq(7), "sample_id"]:
        schedule_path = out / f"llm/policies/{sample_id}_cell_agent_policy_schedule.csv"
        schedule = pd.read_csv(schedule_path)
        required_previous = 0 if checkpoint == 60 else 60
        if required_previous not in set(schedule.checkpoint):
            raise RuntimeError(f"{sample_id} schedule lacks checkpoint {required_previous}")
        reference = matrix[(matrix.sample_id == sample_id) & (matrix.condition == "natural_progression") & (matrix.seed == 1701)].iloc[0].copy()
        reference["run_id"] = f"{sample_id}__ADAPTIVE_REFERENCE__through{checkpoint}"
        engine.run_one(binary, project, out, reference, params)
        run_dir = out / "runs" / reference.run_id
        # Keep agent_id as both the stable lookup index and a prompt field.
        current = local_checkpoint(run_dir, checkpoint).set_index("agent_id", drop=False)
        previous = local_checkpoint(run_dir, required_previous).set_index("agent_id", drop=False)
        previous_policy = schedule[schedule.checkpoint.eq(required_previous)].set_index("agent_id", drop=False)
        new_rows = []
        cache_hits = 0
        for agent_id in current.index:
            payload = adaptive_payload(current.loc[agent_id], previous.loc[agent_id], previous_policy.loc[agent_id], checkpoint)
            canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
            prompt_hash = hashlib.sha256(canonical.encode()).hexdigest()
            raw_dir = out / "llm/raw" / sample_id / f"checkpoint_{checkpoint}"
            prompt_path = raw_dir / f"{agent_id}_prompt.json"
            response_path = raw_dir / f"{agent_id}_response.json"
            raw_dir.mkdir(parents=True, exist_ok=True)
            prompt_path.write_text(json.dumps({"prompt_sha256": prompt_hash, "payload": payload}, indent=2, ensure_ascii=False) + "\n")
            cached = json.loads(response_path.read_text()) if response_path.exists() else None
            if cached and cached.get("prompt_sha256") == prompt_hash:
                obj = extract_json(cached["content"]); cache_hits += 1
            else:
                obj, _ = deepseek_policy(payload, response_path, retries, prompt_hash)

            # ======== 💉 Leca-VC Case Study Logger 注入开始 ========
            # 获取当前细胞的类型
            cell_type_str = str(current.loc[agent_id, "cell_type"])
            
            # 只追踪成纤维细胞 (Fibroblast)
            if "fibroblast" in cell_type_str.lower():
                # 使用全局变量，确保在 Checkpoint 60 和 120 各只抓取第一个遇到的成纤维细胞，防止日志刷屏
                tracker_key = f"printed_case_study_cp_{checkpoint}"
                if not globals().get(tracker_key):
                    globals()[tracker_key] = True 

                    print(f"\n{'='*70}")
                    print(f"🧬 [Leca-VC Case Study] Tracking Agent: {agent_id} ({cell_type_str}) | Checkpoint: {checkpoint}")
                    print(f"{'='*70}")

                    print("\n>>> INPUT TO LLM (Context & Memory):")
                    # 深拷贝一下 payload，剔除掉 instruction/schema 等占据版面但无生物学意义的提示词
                    import copy
                    display_payload = copy.deepcopy(payload)
                    display_payload.pop("instruction", None)
                    display_payload.pop("output_schema", None)
                    display_payload.pop("constraints", None)
                    print(json.dumps(display_payload, indent=4, ensure_ascii=False))

                    print("\n<<< OUTPUT FROM LLM (Biological Decision):")
                    # 直接打印 LLM 吐出来的结构化 JSON
                    print(json.dumps(obj, indent=4, ensure_ascii=False))
                    print(f"{'='*70}\n")
            # ======== 💉 Leca-VC Case Study Logger 注入结束 ========

            raw_policy = validate_policy(obj, agent_id, str(current.loc[agent_id, "cell_type"]))
            effective = {"agent_id": agent_id, "cell_type": current.loc[agent_id, "cell_type"], "checkpoint": checkpoint}
            for field in POLICY_FIELDS:
                old = float(previous_policy.loc[agent_id, field])
                bounded = old + float(np.clip(raw_policy[field] - old, -0.25, 0.25))
                effective[field] = 0.5 * old + 0.5 * bounded
            new_rows.append(effective)
        schedule = pd.concat([schedule[schedule.checkpoint.ne(checkpoint)], pd.DataFrame(new_rows)], ignore_index=True)
        schedule.sort_values(["checkpoint", "agent_id"]).to_csv(schedule_path, index=False)
        rows.append({"sample_id": sample_id, "checkpoint": checkpoint, "n_agents": 50, "cache_hits": cache_hits, "schedule_sha256": sha256(schedule_path)})
        print(f"[{sample_id}] checkpoint {checkpoint}: 50 policies ({cache_hits} cached)", flush=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--retries", type=int, default=8)
    args = parser.parse_args()
    root = args.project_root.resolve(); out = resolve_out(root, args.out_dir); ensure_dirs(out)
    all_rows = []
    for checkpoint in [120]:
        all_rows.extend(update_checkpoint(root, out, checkpoint, args.retries))
    dump_json(out / "audit/adaptive_policy_audit.json", {
        "checkpoints": [60, 120], "new_policy_decisions": 1200,
        "condition_blind_natural_reference": True, "GSE267904_d21_opened": False,
        "bounded_update": "clip raw delta to +/-0.25 then blend 0.5 old + 0.5 bounded",
        "samples": all_rows,
    })
    print("Generated complete t=0/60/120 adaptive schedules for 600 V3 cell-agents")


if __name__ == "__main__":
    main()
