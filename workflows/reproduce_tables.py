#!/usr/bin/env python3
"""Rebuild final Supplement tables and verify frozen expression metrics offline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "build/tables"
METHODS = ("scGen", "CellRank", "AgentVC", "WOT")
LABELS = {"scGen": "scGen‡", "CellRank": "CellRank†", "AgentVC": "Leca-VC", "WOT": "WOT*"}
REGIMES = {"scGen": "External LOCO", "CellRank": "Pre-derived proxy",
           "AgentVC": "Frozen forward reconstruction", "WOT": "Target-derived reference"}


def concordance(prediction: np.ndarray, observed: np.ndarray) -> float:
    dp = prediction - prediction.mean()
    do = observed - observed.mean()
    denominator = np.mean(dp ** 2) + np.mean(do ** 2) + (prediction.mean() - observed.mean()) ** 2
    return float(2 * np.mean(dp * do) / denominator)


def table1() -> tuple[pd.DataFrame, dict]:
    source = ROOT / "source_data/table1"
    overall = pd.read_csv(source / "gse120575_v7_real_vs_virtual_metrics.csv").set_index("metric")
    states = pd.read_csv(source / "gse120575_v7_state_proportion_comparison.csv").set_index("scenario_name")
    signature = pd.read_csv(source / "gse120575_v7_signature_gene_recovery.csv").set_index("response_group")
    full = pd.read_csv(source / "gse120575_v7_full_gene_pseudobulk_correlation.csv").set_index("response_group")
    vectors = pd.read_csv(source / "gse120575_v7_full_gene_pseudobulk_real_virtual.csv")
    direction = str(overall.loc["response_score_direction_agreement", "value"]).lower() == "true"
    response = overall.loc["response_score_direction_agreement"]
    if direction != bool(np.sign(response.real_post_delta) == np.sign(response.virtual_delta)):
        raise RuntimeError("Frozen response-score direction differs from its reported deltas")
    errors = [abs(float(overall.loc["mean_state_proportion_rmse", "value"]) - states.state_proportion_rmse.mean()),
              abs(float(overall.loc["mean_marker_gene_pearson", "value"]) - signature.pearson.mean())]
    for group in ("Responder", "Non-responder"):
        x = np.log1p(vectors[f"{group}_real_post"].to_numpy(float))
        y = np.log1p(vectors[f"{group}_virtual_post"].to_numpy(float))
        valid = np.isfinite(x) & np.isfinite(y)
        if int(valid.sum()) != int(full.loc[group, "n_genes_after_expression_filter"]):
            raise RuntimeError("Table 1 full-gene filter count changed")
        errors.append(abs(float(pearsonr(x[valid], y[valid]).statistic) - full.loc[group, "pearson"]))
    if max(errors) > 1e-12:
        raise RuntimeError(f"Table 1 independent metric check failed: {max(errors)}")
    rows = [
        ("Overall", "Response-score direction agreement", "True" if direction else "False"),
        ("Overall", "Mean virtual state proportion RMSE", f'{float(overall.loc["mean_state_proportion_rmse", "value"]):.4f}'),
        ("Overall", "Mean signature gene Pearson", f'{float(overall.loc["mean_marker_gene_pearson", "value"]):.4f}'),
    ]
    for group, scenario in (("Responder", "virtual_responder_like"), ("Non-responder", "virtual_non_responder_like")):
        rows.extend([
            (group, "Virtual state proportion RMSE", f'{states.loc[scenario, "state_proportion_rmse"]:.4f}'),
            (group, "Full-gene pseudo-bulk Pearson", f'{full.loc[group, "pearson"]:.4f}'),
            (group, "Signature gene Pearson", f'{signature.loc[group, "pearson"]:.4f}'),
        ])
    frame = pd.DataFrame(rows, columns=["scenario", "evaluation_metric", "leca_vc_performance"])
    expected = ["True", "0.0366", "0.9753", "0.0446", "0.9931", "0.9696", "0.0286", "0.9957", "0.9810"]
    if frame.leca_vc_performance.tolist() != expected:
        raise RuntimeError("Table 1 displayed values differ from the final Supplement")
    return frame, {"status": "PASS", "rows": len(frame), "maximum_recomputation_error": float(max(errors)),
                   "experiment": "historical GSE120575 V7 external validation", "full_gene_count": 23848,
                   "signature_gene_count": 39}


def table2() -> tuple[pd.DataFrame, dict]:
    source = ROOT / "source_data/table2"
    frozen = pd.read_csv(source / "expression_metrics.csv").set_index(["day", "method"])
    rows = []
    errors = []
    with np.load(source / "harmonized_6336_old_scale_expression.npz", allow_pickle=False) as arrays:
        if len(arrays["genes"]) != 6336 or arrays["days"].tolist() != [4, 33]:
            raise RuntimeError("Table 2 frozen gene panel or day grid changed")
        for index, day in enumerate(arrays["days"]):
            observed = arrays["observed"][index].mean(axis=0)
            for method in METHODS:
                prediction = (np.log1p(arrays["agentvc_linear"][index].mean(axis=0)) if method == "AgentVC"
                              else arrays[method.lower()][index].mean(axis=0))
                measured = {"Pearson": float(pearsonr(prediction, observed).statistic),
                            "CCC": concordance(prediction, observed),
                            "RMSE": float(np.sqrt(np.mean((prediction - observed) ** 2)))}
                original = frozen.loc[(int(day), method)]
                errors.extend(abs(measured[key] - float(original[key])) for key in measured)
                rows.append({"stage": f"Day {day}", "source_method_id": method, "method": LABELS[method],
                             "information_regime": REGIMES[method],
                             **{key: f"{float(original[key]):.4f}" for key in measured}})
    if max(errors) > 1e-12:
        raise RuntimeError(f"Table 2 independent metric check failed: {max(errors)}")
    frame = pd.DataFrame(rows)
    expected = [
        ("0.9682", "0.9636", "0.1192"), ("0.7793", "0.5645", "0.3775"),
        ("0.9819", "0.9799", "0.0953"), ("0.9993", "0.9993", "0.0167"),
        ("0.9744", "0.9733", "0.1002"), ("0.8252", "0.6316", "0.3298"),
        ("0.9694", "0.9604", "0.1309"), ("0.9994", "0.9993", "0.0159"),
    ]
    if list(frame[["Pearson", "CCC", "RMSE"]].itertuples(index=False, name=None)) != expected:
        raise RuntimeError("Table 2 displayed values differ from the final Supplement")
    return frame, {"status": "PASS", "rows": len(frame), "gene_count": 6336,
                   "maximum_recomputation_error": float(max(errors)), "method_information_regimes_equal": False}


def latex(frame: pd.DataFrame) -> str:
    def escape(value: object) -> str:
        return str(value).replace("_", r"\_").replace("&", r"\&").replace("‡", r"$\ddagger$").replace("†", r"$\dagger$").replace("*", r"$\ast$")
    columns = [name for name in frame.columns if name != "source_method_id"]
    lines = [r"\begin{tabular}{" + "l" * len(columns) + "}", r"\toprule",
             " & ".join(escape(name.replace("_", " ")) for name in columns) + r" \\", r"\midrule"]
    lines.extend(" & ".join(escape(row[name]) for name in columns) + r" \\" for _, row in frame.iterrows())
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--all", action="store_true")
    selection.add_argument("--table", choices=("table1", "table2"))
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    results = {}
    for name in ("table1", "table2") if args.all else (args.table,):
        frame, audit = {"table1": table1, "table2": table2}[name]()
        frame.to_csv(OUT / f"{name}.csv", index=False)
        (OUT / f"{name}.tex").write_text(latex(frame), encoding="utf-8")
        results[name] = audit
    (OUT / "verification.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
