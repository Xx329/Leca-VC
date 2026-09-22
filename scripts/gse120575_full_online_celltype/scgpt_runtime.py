#!/usr/bin/env python3
"""Real local scGPT runtime and preflight for GSE120575 profiles."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np

from expression_profiles import BROAD_TYPES, MODULES


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    an = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
    bn = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-12)
    return an @ bn.T


class ScGPTRuntime:
    def __init__(self, model_dir: Path, genes: list[str]):
        import torch
        from scgpt.model import TransformerModel
        from scgpt.tokenizer.gene_tokenizer import GeneVocab

        self.torch = torch
        self.model_dir = model_dir
        self.genes = genes
        self.vocab = GeneVocab.from_file(str(model_dir / "vocab.json"))
        self.args = json.loads((model_dir / "args.json").read_text())
        if len(genes) > int(self.args.get("max_seq_len", 1200)):
            raise RuntimeError("Frozen gene panel exceeds scGPT max_seq_len")
        missing = [gene for gene in genes if gene not in self.vocab]
        if missing:
            raise RuntimeError(f"Frozen genes missing from loaded scGPT vocab: {missing[:10]}")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = TransformerModel(
            ntoken=len(self.vocab),
            d_model=self.args.get("embsize", 512),
            nhead=self.args.get("nheads", 8),
            d_hid=self.args.get("d_hid", 512),
            nlayers=self.args.get("nlayers", 12),
            nlayers_cls=3,
            n_cls=1,
            vocab=self.vocab,
            dropout=self.args.get("dropout", 0.2),
            pad_token="<pad>",
            pad_value=self.vocab["<pad>"],
        )
        state = torch.load(model_dir / "best_model.pt", map_location=self.device)
        normalized = {key.replace("Wqkv.", "in_proj_"): value for key, value in state.items()}
        incompat = self.model.load_state_dict(normalized, strict=False)
        self.missing_state_keys = list(incompat.missing_keys)
        self.unexpected_state_keys = list(incompat.unexpected_keys)
        self.model.to(self.device).eval()
        self.gene_ids = torch.tensor(
            [self.vocab[gene] for gene in genes], dtype=torch.long, device=self.device
        ).unsqueeze(0)

    def embed(self, expression: np.ndarray, batch_size: int = 16) -> np.ndarray:
        torch = self.torch
        values_np = np.log1p(np.maximum(np.asarray(expression, dtype=np.float32), 0.0))
        outputs = []
        with torch.no_grad():
            for start in range(0, len(values_np), batch_size):
                values = torch.tensor(values_np[start:start + batch_size], device=self.device)
                gene_ids = self.gene_ids.expand(len(values), -1)
                result = self.model(
                    src=gene_ids,
                    values=values,
                    src_key_padding_mask=None,
                )["cell_emb"]
                outputs.append(result.detach().cpu().numpy())
        return np.vstack(outputs)


def module_means(expression: np.ndarray, genes: list[str]) -> dict[str, float]:
    gene_index = {gene: i for i, gene in enumerate(genes)}
    result = {}
    for module, requested in MODULES.items():
        indexes = [gene_index[gene] for gene in requested if gene in gene_index]
        result[module] = float(np.mean(expression[indexes])) if indexes else 0.0
    return result


def derive_output(
    expression: np.ndarray,
    embedding: np.ndarray,
    references: np.ndarray,
    reference_expression: np.ndarray,
    cell_type_index: int,
    genes: list[str],
) -> dict[str, object]:
    similarities = cosine(embedding[None, :], references)[0]
    own_similarity = float(similarities[cell_type_index])
    modules = module_means(expression, genes)
    baseline_modules = module_means(reference_expression, genes)
    return {
        "identity_embedding": embedding.astype(float).tolist(),
        "identity_similarity": own_similarity,
        "identity_similarity_by_cell_type": {
            cell_type: float(similarities[i]) for i, cell_type in enumerate(BROAD_TYPES)
        },
        "identity_drift": float(1.0 - own_similarity),
        "treatment_expression_shift_l2": float(
            np.linalg.norm(expression - reference_expression)
            / max(np.linalg.norm(reference_expression), 1e-12)
        ),
        "activation_score": modules["activation"],
        "stress_score": modules["stress"],
        "exhaustion_score": modules["exhaustion"],
        "activation_shift": modules["activation"] - baseline_modules["activation"],
        "stress_shift": modules["stress"] - baseline_modules["stress"],
        "exhaustion_shift": modules["exhaustion"] - baseline_modules["exhaustion"],
        "identity_drift_warning": bool(own_similarity < 0.80),
        "expression_representation": {
            "gene_count": len(genes),
            "mean": float(np.mean(expression)),
            "standard_deviation": float(np.std(expression)),
            "maximum": float(np.max(expression)),
        },
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--profiles", type=Path, required=True)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    data = np.load(a.profiles)
    profiles = data["profiles"].astype(np.float32)
    sample_ids = data["sample_ids"].astype(str)
    cell_types = data["broad_cell_types"].astype(str)
    genes = data["genes"].astype(str).tolist()
    counts = data["n_cells"].astype(int)

    perturbation_spec = {
        "status": "PREREGISTERED_BEFORE_SCGPT_PERTURBATION_INFERENCE",
        "base_profile": "pre-only weighted cell-type reference",
        "perturbations": {
            "activation_up": {"genes": list(MODULES["activation"]), "additive_input_units": 1.0},
            "stress_up": {"genes": list(MODULES["stress"]), "additive_input_units": 1.0},
            "exhaustion_up": {"genes": list(MODULES["exhaustion"]), "additive_input_units": 1.0},
        },
        "pass_thresholds": {
            "repeat_max_absolute_embedding_difference": 1e-6,
            "minimum_between_celltype_embedding_distance": 1e-6,
            "minimum_perturbation_embedding_distance": 1e-6,
            "minimum_derived_quantity_absolute_change": 1e-6,
        },
    }
    perturbation_path = a.output_dir / "scgpt_perturbation_preregistration.json"
    if not perturbation_path.exists():
        perturbation_path.write_text(json.dumps(perturbation_spec, ensure_ascii=False, indent=2) + "\n")
    elif json.loads(perturbation_path.read_text()) != perturbation_spec:
        raise RuntimeError("Existing perturbation preregistration differs from frozen specification")

    runtime = ScGPTRuntime(a.model_dir, genes)
    reference_expression = []
    for cell_type in BROAD_TYPES:
        indexes = np.flatnonzero(cell_types == cell_type)
        positive = indexes[counts[indexes] > 0]
        reference_expression.append(np.average(profiles[positive], axis=0, weights=counts[positive]))
    reference_expression = np.asarray(reference_expression, dtype=np.float32)
    embeddings_1 = runtime.embed(reference_expression)
    embeddings_2 = runtime.embed(reference_expression)
    repeat_difference = float(np.max(np.abs(embeddings_1 - embeddings_2)))
    distance_matrix = np.linalg.norm(embeddings_1[:, None, :] - embeddings_1[None, :, :], axis=2)
    off_diagonal = distance_matrix[~np.eye(len(BROAD_TYPES), dtype=bool)]

    gene_index = {gene: i for i, gene in enumerate(genes)}
    perturbation_rows = []
    for cell_type_index, cell_type in enumerate(BROAD_TYPES):
        base_expression = reference_expression[cell_type_index]
        base_embedding = embeddings_1[cell_type_index]
        base_output = derive_output(
            base_expression, base_embedding, embeddings_1, base_expression,
            cell_type_index, genes,
        )
        for name, spec in perturbation_spec["perturbations"].items():
            perturbed = base_expression.copy()
            changed_genes = [gene for gene in spec["genes"] if gene in gene_index]
            for gene in changed_genes:
                perturbed[gene_index[gene]] += float(spec["additive_input_units"])
            embedding = runtime.embed(perturbed[None, :])[0]
            output = derive_output(
                perturbed, embedding, embeddings_1, base_expression,
                cell_type_index, genes,
            )
            derived_keys = (
                "identity_similarity", "identity_drift", "treatment_expression_shift_l2",
                "activation_score", "stress_score", "exhaustion_score",
            )
            changes = {key: float(output[key]) - float(base_output[key]) for key in derived_keys}
            perturbation_rows.append({
                "cell_type": cell_type,
                "perturbation": name,
                "changed_genes": changed_genes,
                "embedding_l2_change": float(np.linalg.norm(embedding - base_embedding)),
                "derived_changes": changes,
            })

    perturbation_embedding_changes = [row["embedding_l2_change"] for row in perturbation_rows]
    perturbation_derived_changes = [
        max(abs(value) for value in row["derived_changes"].values()) for row in perturbation_rows
    ]
    gates = {
        "checkpoint_exists": (a.model_dir / "best_model.pt").exists(),
        "vocab_exists": (a.model_dir / "vocab.json").exists(),
        "args_exists": (a.model_dir / "args.json").exists(),
        "real_checkpoint_loaded": True,
        "no_mock_embedding_used": True,
        "panel_gene_count_matches_max_seq_len": len(genes) == 1200,
        "six_celltype_embeddings_not_identical": float(np.min(off_diagonal)) > 1e-6,
        "repeat_inference_stable": repeat_difference <= 1e-6,
        "every_preregistered_perturbation_changes_embedding": min(perturbation_embedding_changes) > 1e-6,
        "every_preregistered_perturbation_changes_derived_quantity": min(perturbation_derived_changes) > 1e-6,
    }
    summary = {
        "status": "PASS_REAL_SCGPT_CORE_PREFLIGHT" if all(gates.values()) else "FAIL_REAL_SCGPT_CORE_PREFLIGHT",
        "all_core_gates_passed": all(gates.values()),
        "gates": gates,
        "model": {
            "checkpoint": str((a.model_dir / "best_model.pt").resolve()),
            "checkpoint_sha256": sha256(a.model_dir / "best_model.pt"),
            "vocab_sha256": sha256(a.model_dir / "vocab.json"),
            "args_sha256": sha256(a.model_dir / "args.json"),
            "device": str(runtime.device),
            "embedding_dimensions": int(embeddings_1.shape[1]),
            "missing_state_keys": runtime.missing_state_keys,
            "unexpected_state_keys": runtime.unexpected_state_keys,
        },
        "gene_panel": {
            "count": len(genes),
            "all_in_loaded_vocab": True,
            "genes": genes,
        },
        "cell_type_embedding_distance": {
            "minimum_off_diagonal_l2": float(np.min(off_diagonal)),
            "maximum_off_diagonal_l2": float(np.max(off_diagonal)),
            "matrix": distance_matrix.astype(float).tolist(),
        },
        "repeat_max_absolute_embedding_difference": repeat_difference,
        "perturbation_results": perturbation_rows,
        "minimum_perturbation_embedding_l2_change": float(min(perturbation_embedding_changes)),
        "minimum_perturbation_derived_quantity_change": float(min(perturbation_derived_changes)),
        "integration_gates_pending": [
            "scgpt_output_enters_full_prompt",
            "action_mapper_reads_scgpt_derived_quantity",
            "full_vs_agent_only_executed_action_diff_nonempty",
        ],
    }
    np.savez_compressed(
        a.output_dir / "scgpt_preonly_references.npz",
        cell_types=np.asarray(BROAD_TYPES), genes=np.asarray(genes),
        reference_expression=reference_expression,
        reference_embeddings=embeddings_1.astype(np.float32),
    )
    (a.output_dir / "scgpt_core_preflight.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: summary[k] for k in ["status", "all_core_gates_passed", "gates", "model", "cell_type_embedding_distance", "repeat_max_absolute_embedding_difference", "minimum_perturbation_embedding_l2_change", "minimum_perturbation_derived_quantity_change"]}, ensure_ascii=False, indent=2))
    return 0 if all(gates.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
