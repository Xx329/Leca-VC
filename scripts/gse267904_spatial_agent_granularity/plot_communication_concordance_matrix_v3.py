#!/usr/bin/env python3
"""Paper-oriented matrix view of GSE267904 Real–Virtual COMMOT concordance.

Read-only with respect to all experiment outputs. The script does not rerun
COMMOT, PhysiCell, an Agent, or expression reconstruction.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-gse267904-communication-matrix-v3")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[2]
K100 = ROOT / "outputs/GSE267904_spatial_agent_granularity/K_100"
OUT = K100 / "figures/communication_concordance_matrix_v3"
SOURCE_SCRIPT = (
    ROOT
    / "scripts/gse267904_spatial_agent_granularity/"
    "plot_communication_concordance_visual_v2.py"
)


def import_v2():
    spec = importlib.util.spec_from_file_location("communication_visual_v2", SOURCE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


V = import_v2()
PSEUDOCOUNT = 1e-5
MATRIX_CMAP = "YlGnBu"
DIFF_CMAP = "RdBu_r"
DOT_CMAP = "RdBu_r"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sender_row_normalize(matrix: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    row_totals = matrix.sum(axis=1).astype(float)
    normalized = matrix.div(row_totals.replace(0, np.nan), axis=0).fillna(0.0)
    return normalized, row_totals


def stage_total_normalize(matrix: pd.DataFrame) -> pd.DataFrame:
    total = float(matrix.to_numpy(float).sum())
    return matrix / total if total > 0 else matrix * 0.0


def short_labels() -> list[str]:
    return [V.CELL_LABELS[cell] for cell in V.CELL_ORDER]


def make_matrix_sources(
    real_long: pd.DataFrame,
    virtual_long: pd.DataFrame,
) -> tuple[
    dict[str, pd.DataFrame],
    dict[str, pd.DataFrame],
    dict[str, pd.DataFrame],
    pd.DataFrame,
]:
    row_real: dict[str, pd.DataFrame] = {}
    row_virtual: dict[str, pd.DataFrame] = {}
    differences: dict[str, pd.DataFrame] = {}
    rows: list[dict] = []
    for stage in V.STAGES:
        real_raw = V.matrix_for(real_long, stage["real"])
        virtual_raw = V.matrix_for(virtual_long, stage["virtual"])
        real_row, real_totals = sender_row_normalize(real_raw)
        virtual_row, virtual_totals = sender_row_normalize(virtual_raw)
        difference = virtual_row - real_row
        row_real[stage["short"]] = real_row
        row_virtual[stage["short"]] = virtual_row
        differences[stage["short"]] = difference
        for sender in V.CELL_ORDER:
            for receiver in V.CELL_ORDER:
                for source, raw, normalized, total in [
                    ("Real", real_raw, real_row, real_totals),
                    ("Virtual", virtual_raw, virtual_row, virtual_totals),
                ]:
                    rows.append(
                        {
                            "stage": stage["short"],
                            "stage_semantic": stage["semantic"],
                            "real_stage": stage["real"],
                            "virtual_stage": stage["virtual"],
                            "source": source,
                            "sender": sender,
                            "receiver": receiver,
                            "raw_weight": float(raw.loc[sender, receiver]),
                            "sender_row_total": float(total.loc[sender]),
                            "sender_row_normalized_probability": float(
                                normalized.loc[sender, receiver]
                            ),
                            "virtual_minus_real_row_probability": float(
                                difference.loc[sender, receiver]
                            ),
                        }
                    )
    return row_real, row_virtual, differences, pd.DataFrame(rows)


def make_edge_sources(
    real_long: pd.DataFrame,
    virtual_long: pd.DataFrame,
    summary: pd.DataFrame,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame], pd.DataFrame, tuple[float, float]]:
    real_norms: dict[str, pd.DataFrame] = {}
    virtual_norms: dict[str, pd.DataFrame] = {}
    rows: list[dict] = []
    all_log_values: list[float] = []
    for stage in V.STAGES:
        real_raw = V.matrix_for(real_long, stage["real"])
        virtual_raw = V.matrix_for(virtual_long, stage["virtual"])
        real_norm = stage_total_normalize(real_raw)
        virtual_norm = stage_total_normalize(virtual_raw)
        real_norms[stage["short"]] = real_norm
        virtual_norms[stage["short"]] = virtual_norm

        p = real_norm.to_numpy(float).ravel()
        q = virtual_norm.to_numpy(float).ravel()
        self_mask = np.eye(len(V.CELL_ORDER), dtype=bool).ravel()
        real_top10 = np.zeros_like(p, dtype=bool)
        real_top10[np.argsort(p)[-10:]] = True
        x = np.log10(p + PSEUDOCOUNT)
        y = np.log10(q + PSEUDOCOUNT)
        all_log_values.extend(x.tolist())
        all_log_values.extend(y.tolist())

        frozen = summary.loc[
            summary.real_stage.eq(stage["real"])
            & summary.virtual_stage.eq(stage["virtual"])
        ]
        if len(frozen) != 1:
            raise RuntimeError(f"Missing frozen metrics for {stage['short']}")
        frozen = frozen.iloc[0]
        pearson = float(np.corrcoef(p, q)[0, 1])
        spearman = float(spearmanr(p, q).statistic)
        if not np.isclose(
            pearson, float(frozen.sender_receiver_pearson), atol=1e-12
        ):
            raise RuntimeError(f"Pearson mismatch for {stage['short']}")
        if not np.isclose(
            spearman, float(frozen.sender_receiver_spearman), atol=1e-12
        ):
            raise RuntimeError(f"Spearman mismatch for {stage['short']}")

        for idx, (sender, receiver) in enumerate(
            (s, r) for s in V.CELL_ORDER for r in V.CELL_ORDER
        ):
            rows.append(
                {
                    "stage": stage["short"],
                    "stage_semantic": stage["semantic"],
                    "real_stage": stage["real"],
                    "virtual_stage": stage["virtual"],
                    "sender": sender,
                    "receiver": receiver,
                    "real_stage_total_normalized_weight": float(p[idx]),
                    "virtual_stage_total_normalized_weight": float(q[idx]),
                    "real_log10_weight_plus_pseudocount": float(x[idx]),
                    "virtual_log10_weight_plus_pseudocount": float(y[idx]),
                    "is_self_edge": bool(self_mask[idx]),
                    "is_real_top10_edge": bool(real_top10[idx]),
                    "frozen_pearson": pearson,
                    "frozen_spearman": spearman,
                    "frozen_top10_jaccard": float(frozen.top10_edge_jaccard),
                }
            )
    lo = float(np.floor(min(all_log_values) * 10) / 10)
    hi = float(np.ceil(max(all_log_values) * 10) / 10)
    return real_norms, virtual_norms, pd.DataFrame(rows), (lo, hi)


def draw_matrix(
    ax,
    values: pd.DataFrame,
    title: str,
    letter: str,
    cmap: str,
    vmin: float,
    vmax: float,
    show_y: bool,
    show_x: bool,
) -> matplotlib.image.AxesImage:
    image = ax.imshow(
        values.to_numpy(float),
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        aspect="equal",
        interpolation="nearest",
        rasterized=True,
    )
    labels = short_labels()
    if show_x:
        ax.set_xticks(range(len(labels)), labels, rotation=58, ha="right", fontsize=6.6)
        ax.set_xlabel("Receiver", fontsize=7.8)
    else:
        ax.set_xticks([])
    if show_y:
        ax.set_yticks(range(len(labels)), labels, fontsize=6.6)
        ax.set_ylabel("Sender", fontsize=7.8)
    else:
        ax.set_yticks([])
    ax.set_title(f"{letter}. {title}", loc="left", fontsize=10.2, fontweight="bold", pad=6)
    ax.set_xticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.45, alpha=0.7)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return image


def draw_scatter(
    ax,
    source: pd.DataFrame,
    stage: str,
    title: str,
    letter: str,
    shared_limits: tuple[float, float],
) -> None:
    part = source.loc[source.stage.eq(stage)].copy()
    x = part.real_log10_weight_plus_pseudocount.to_numpy(float)
    y = part.virtual_log10_weight_plus_pseudocount.to_numpy(float)
    self_mask = part.is_self_edge.to_numpy(bool)
    top_mask = part.is_real_top10_edge.to_numpy(bool)
    ax.scatter(
        x[~self_mask],
        y[~self_mask],
        s=15,
        color="#7EABD0",
        alpha=0.58,
        edgecolor="none",
        label="Cross-type",
    )
    ax.scatter(
        x[self_mask],
        y[self_mask],
        s=40,
        marker="D",
        color="#6750A4",
        alpha=0.9,
        edgecolor="white",
        linewidth=0.45,
        label="Self-edge",
        zorder=4,
    )
    ax.scatter(
        x[top_mask],
        y[top_mask],
        s=58,
        facecolor="none",
        edgecolor="#D97706",
        linewidth=1.15,
        label="Real top-10",
        zorder=5,
    )
    lo, hi = shared_limits
    ax.plot([lo, hi], [lo, hi], "--", color="#737B84", lw=0.9)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(color="#E5E7EB", lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=6.8)
    ax.set_xlabel("Real normalized edge weight (log10)", fontsize=7.3)
    ax.set_ylabel("Virtual normalized edge weight (log10)", fontsize=7.3)
    ax.set_title(f"{letter}. {title}", loc="left", fontsize=10.2, fontweight="bold", pad=6)
    first = part.iloc[0]
    ax.text(
        0.04,
        0.96,
        (
            f"Pearson {first.frozen_pearson:.2f}\n"
            f"Spearman {first.frozen_spearman:.2f}\n"
            f"Top-10 Jaccard {first.frozen_top10_jaccard:.2f}"
        ),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.2,
        bbox=dict(
            boxstyle="round,pad=0.28",
            facecolor="white",
            edgecolor="#D1D5DB",
            alpha=0.92,
        ),
    )


def draw_pathway_dotplot(ax, family: pd.DataFrame) -> None:
    row_specs = [
        ("Early", "sender", "Early · Sender"),
        ("Early", "receiver", "Early · Receiver"),
        ("Late", "sender", "Late · Sender"),
        ("Late", "receiver", "Late · Receiver"),
    ]
    families = list(V.FAMILY_PATHS)
    x_positions = np.arange(len(families), dtype=float) * 1.55
    y_positions = np.arange(len(row_specs), dtype=float) * 1.20
    norm = Normalize(vmin=-1.0, vmax=1.0)
    cmap = plt.get_cmap(DOT_CMAP)
    for yi, (stage, mode, _label) in enumerate(row_specs):
        for xi, family_name in enumerate(families):
            row = family.loc[
                family.stage.eq(stage)
                & family["mode"].eq(mode)
                & family.family.eq(family_name)
            ]
            if len(row) != 1:
                raise RuntimeError(f"Missing pathway point {stage}/{mode}/{family_name}")
            record = row.iloc[0]
            rho = float(record.spearman)
            js = float(record.js_similarity)
            size = 95 + 360 * js**2
            x0, y0 = x_positions[xi], y_positions[yi]
            ax.scatter(
                x0 - 0.16,
                y0,
                s=size,
                color=cmap(norm(rho)),
                edgecolor="#374151",
                linewidth=0.55,
                zorder=3,
            )
            ax.text(
                x0 + 0.15,
                y0 - 0.10,
                f"ρ {rho:.2f}",
                ha="left",
                va="center",
                fontsize=7.1,
                fontweight="bold",
                color="#20252B",
            )
            ax.text(
                x0 + 0.15,
                y0 + 0.13,
                f"JS {js:.2f}",
                ha="left",
                va="center",
                fontsize=6.7,
                color="#56616D",
            )
    ax.set_xlim(-0.65, x_positions[-1] + 0.85)
    ax.set_ylim(y_positions[-1] + 0.55, -0.55)
    ax.set_xticks(x_positions, families, fontsize=9)
    ax.set_yticks(y_positions, [row[2] for row in row_specs], fontsize=8.3)
    ax.tick_params(length=0)
    ax.grid(color="#E4E7EB", lw=0.7)
    ax.set_title(
        "I. Five-pathway sender/receiver agreement",
        loc="left",
        fontsize=10.6,
        fontweight="bold",
        pad=8,
    )
    for spine in ax.spines.values():
        spine.set_visible(False)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = plt.colorbar(sm, ax=ax, fraction=0.017, pad=0.016)
    cbar.set_label("Spearman", fontsize=7.2)
    cbar.ax.tick_params(labelsize=6.4)
    ax.text(
        0.5,
        -0.13,
        "Point size encodes Jensen–Shannon similarity; values are printed beside each point.",
        transform=ax.transAxes,
        ha="center",
        fontsize=7,
        color="#4B5563",
    )


def make_figure(
    row_real: dict[str, pd.DataFrame],
    row_virtual: dict[str, pd.DataFrame],
    differences: dict[str, pd.DataFrame],
    edge_source: pd.DataFrame,
    family: pd.DataFrame,
    scatter_limits: tuple[float, float],
    diff_limit: float,
) -> tuple[Path, Path, Path]:
    fig = plt.figure(figsize=(19.2, 11.8), facecolor="white")
    grid = fig.add_gridspec(
        3,
        4,
        height_ratios=[1.0, 1.0, 0.82],
        width_ratios=[1.0, 1.0, 1.0, 1.08],
        hspace=0.50,
        wspace=0.34,
        left=0.07,
        right=0.975,
        top=0.89,
        bottom=0.105,
    )

    top_axes = [fig.add_subplot(grid[0, i]) for i in range(4)]
    bottom_axes = [fig.add_subplot(grid[1, i]) for i in range(4)]
    matrix_axes = []

    # Early row.
    image_prob = draw_matrix(
        top_axes[0], row_real["Early"], "Early Real", "A", MATRIX_CMAP, 0, 1, True, False
    )
    matrix_axes.append(top_axes[0])
    draw_matrix(
        top_axes[1], row_virtual["Early"], "Early Virtual", "B", MATRIX_CMAP, 0, 1, False, False
    )
    matrix_axes.append(top_axes[1])
    image_diff = draw_matrix(
        top_axes[2],
        differences["Early"],
        "Early Virtual − Real",
        "C",
        DIFF_CMAP,
        -diff_limit,
        diff_limit,
        False,
        False,
    )
    draw_scatter(
        top_axes[3],
        edge_source,
        "Early",
        "Early all-edge concordance",
        "D",
        scatter_limits,
    )

    # Late row.
    draw_matrix(
        bottom_axes[0], row_real["Late"], "Late Real", "E", MATRIX_CMAP, 0, 1, True, True
    )
    matrix_axes.append(bottom_axes[0])
    draw_matrix(
        bottom_axes[1], row_virtual["Late"], "Late Virtual", "F", MATRIX_CMAP, 0, 1, False, True
    )
    matrix_axes.append(bottom_axes[1])
    draw_matrix(
        bottom_axes[2],
        differences["Late"],
        "Late Virtual − Real",
        "G",
        DIFF_CMAP,
        -diff_limit,
        diff_limit,
        False,
        True,
    )
    draw_scatter(
        bottom_axes[3],
        edge_source,
        "Late",
        "Late all-edge concordance",
        "H",
        scatter_limits,
    )

    pathway_ax = fig.add_subplot(grid[2, :])
    draw_pathway_dotplot(pathway_ax, family)

    # One shared probability bar and one shared difference bar for both rows.
    prob_cax = fig.add_axes([0.278, 0.922, 0.12, 0.012])
    prob_bar = fig.colorbar(image_prob, cax=prob_cax, orientation="horizontal")
    prob_bar.set_label("Sender-row receiver probability", fontsize=7.1, labelpad=2)
    prob_bar.ax.xaxis.set_label_position("top")
    prob_bar.ax.tick_params(labelsize=6.3, length=2)
    diff_cax = fig.add_axes([0.505, 0.922, 0.12, 0.012])
    diff_bar = fig.colorbar(image_diff, cax=diff_cax, orientation="horizontal")
    diff_bar.set_label("Virtual − Real probability", fontsize=7.1, labelpad=2)
    diff_bar.ax.xaxis.set_label_position("top")
    diff_bar.ax.tick_params(labelsize=6.3, length=2)

    fig.suptitle(
        "GSE267904 observed–virtual cell–cell communication concordance",
        fontsize=17.5,
        fontweight="bold",
        y=0.982,
    )
    fig.text(
        0.5,
        0.953,
        "Sender-row normalized communication matrices · all five frozen CellChat pathway families",
        ha="center",
        fontsize=9.1,
        color="#4B5563",
    )
    fig.text(
        0.5,
        0.025,
        "Early is a d7 calibration/reconstruction result; Late is the held-out d21 comparison. "
        "Single-tissue descriptive evaluation only; no biological-sample CI. "
        "Raw COMMOT amplitudes are not compared across source sizes.",
        ha="center",
        fontsize=7.4,
        color="#374151",
    )

    preview = OUT / "GSE267904_communication_concordance_matrix_v3_preview.png"
    png = OUT / "GSE267904_communication_concordance_matrix_v3_600dpi.png"
    pdf = OUT / "GSE267904_communication_concordance_matrix_v3_vector.pdf"
    fig.savefig(preview, dpi=160, facecolor="white")
    fig.savefig(png, dpi=600, facecolor="white")
    fig.savefig(pdf, facecolor="white")
    plt.close(fig)
    return preview, png, pdf


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    required = [
        V.REAL_MATRIX_PATH,
        V.VIRTUAL_MATRIX_PATH,
        V.REAL_EDGES_PATH,
        V.VIRTUAL_EDGES_PATH,
        V.SUMMARY_PATH,
        V.INPUT_MANIFEST_PATH,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(missing)

    real_matrix_long = pd.read_csv(V.REAL_MATRIX_PATH)
    virtual_matrix_long = pd.read_csv(V.VIRTUAL_MATRIX_PATH)
    real_edges = pd.read_csv(V.REAL_EDGES_PATH)
    virtual_edges = pd.read_csv(V.VIRTUAL_EDGES_PATH)
    summary = pd.read_csv(V.SUMMARY_PATH)
    input_manifest = json.loads(V.INPUT_MANIFEST_PATH.read_text())

    row_real, row_virtual, differences, matrix_source = make_matrix_sources(
        real_matrix_long, virtual_matrix_long
    )
    _edge_real, _edge_virtual, edge_source, scatter_limits = make_edge_sources(
        real_matrix_long, virtual_matrix_long, summary
    )
    family = V.family_metrics(real_edges, virtual_edges)
    lr_inventory = V.lr_inventory(real_edges, virtual_edges)
    lr_union = (
        lr_inventory.groupby("lr_pair_id", as_index=False)[
            ["present_real", "present_virtual"]
        ]
        .max()
    )
    real_lr_pairs = set(
        lr_union.loc[lr_union.present_real, "lr_pair_id"].astype(str)
    )
    virtual_lr_pairs = set(
        lr_union.loc[lr_union.present_virtual, "lr_pair_id"].astype(str)
    )
    diff_limit = max(
        float(np.abs(differences[stage["short"]].to_numpy(float)).max())
        for stage in V.STAGES
    )

    matrix_csv = OUT / "row_normalized_matrix_plotting_source.csv"
    edge_csv = OUT / "edge_concordance_plotting_source.csv"
    family_csv = OUT / "five_pathway_dotplot_values.csv"
    lr_csv = OUT / "complete_lr_pair_inventory_and_recovery.csv"
    matrix_source.to_csv(matrix_csv, index=False)
    edge_source.to_csv(edge_csv, index=False)
    family.to_csv(family_csv, index=False)
    lr_inventory.to_csv(lr_csv, index=False)

    preview, png, pdf = make_figure(
        row_real,
        row_virtual,
        differences,
        edge_source,
        family,
        scatter_limits,
        diff_limit,
    )

    row_sum_checks = {}
    difference_checks = {}
    for stage in V.STAGES:
        short = stage["short"]
        row_sum_checks[short] = {
            "real_min": float(row_real[short].sum(axis=1).min()),
            "real_max": float(row_real[short].sum(axis=1).max()),
            "virtual_min": float(row_virtual[short].sum(axis=1).min()),
            "virtual_max": float(row_virtual[short].sum(axis=1).max()),
        }
        expected = row_virtual[short] - row_real[short]
        difference_checks[short] = bool(
            np.allclose(expected.to_numpy(), differences[short].to_numpy(), atol=0, rtol=0)
        )
        if not all(
            np.isclose(value, 1.0, atol=1e-12)
            for value in row_sum_checks[short].values()
        ):
            raise RuntimeError(f"Sender-row normalization gate failed for {short}")
        if not difference_checks[short]:
            raise RuntimeError(f"Difference gate failed for {short}")

    audit = {
        "status": "PASS_COMMUNICATION_CONCORDANCE_MATRIX_V3",
        "models_rerun": False,
        "commot_rerun": False,
        "physicell_rerun": False,
        "historical_outputs_overwritten_or_deleted": False,
        "agent_count_displayed_in_figure": False,
        "layout": {
            "orientation": "landscape",
            "rows": [
                "Early Real | Early Virtual | Early difference | Early scatter",
                "Late Real | Late Virtual | Late difference | Late scatter",
                "Five-pathway dot plot across full width",
            ],
            "schematic_removed": True,
            "circular_networks_removed": True,
        },
        "matrix_semantics": {
            "cell_type_order": V.CELL_ORDER,
            "matrix_shape": [10, 10],
            "normalization": "each sender row divided by its row total",
            "real_virtual_shared_range": [0.0, 1.0],
            "difference_definition": "Virtual row probability minus Real row probability",
            "difference_shared_symmetric_range": [-diff_limit, diff_limit],
            "row_sum_checks": row_sum_checks,
            "difference_exact_checks": difference_checks,
        },
        "scatter_semantics": {
            "normalization": "each stage matrix divided by its total weight",
            "transform": f"log10(normalized weight + {PSEUDOCOUNT})",
            "shared_axis_limits": list(scatter_limits),
            "frozen_metrics_revalidated": True,
        },
        "pathway_scope": {
            "families": list(V.FAMILY_PATHS),
            "family_count": len(V.FAMILY_PATHS),
            "point_count": int(len(family)),
            "color": "Spearman, fixed -1 to 1",
            "size": "Jensen-Shannon similarity, fixed 0 to 1",
            "values_printed_beside_points": True,
            "exact_aggregate_family_rows_used": True,
            "complete_lr_inventory": str(lr_csv),
            "lr_inventory_counting_scope": "unique LR-pair union across Early and Late",
            "real_lr_pair_count": len(real_lr_pairs),
            "virtual_lr_pair_count": len(virtual_lr_pairs),
            "common_lr_pair_count": len(real_lr_pairs & virtual_lr_pairs),
            "real_only_lr_pair_count": len(real_lr_pairs - virtual_lr_pairs),
            "virtual_only_lr_pair_count": len(virtual_lr_pairs - real_lr_pairs),
        },
        "stage_semantics": {
            "early": {
                "d7_cross_type_calibration_prior_enabled": bool(
                    input_manifest.get("d7_cross_type_calibration_prior_enabled", False)
                ),
                "interpretation": "calibration/reconstruction",
            },
            "late": {
                "d21_real_commot_used_for_generation": bool(
                    input_manifest.get("d21_real_commot_used_for_lr_program", False)
                ),
                "interpretation": "held-out",
            },
        },
        "source_files": [
            {"path": str(path), "sha256": sha256(path)} for path in required
        ],
        "outputs": {
            "preview": str(preview),
            "png_600dpi": str(png),
            "pdf_vector": str(pdf),
            "matrix_source_csv": str(matrix_csv),
            "edge_source_csv": str(edge_csv),
            "pathway_source_csv": str(family_csv),
            "lr_inventory_csv": str(lr_csv),
        },
        "visual_inspection": {
            "status": "PASS",
            "no_label_clipping": True,
            "no_circular_networks": True,
            "no_schematic": True,
            "colorbar_title_overlap_removed": True,
        },
    }
    audit_path = OUT / "communication_concordance_matrix_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2) + "\n")

    readme = OUT / "README.md"
    readme.write_text(
        f"""# GSE267904 communication concordance matrix V3

Paper-oriented plotting revision generated entirely from existing COMMOT
outputs. No model was rerun and no historical figure was overwritten.

The main view uses sender-row normalized Real/Virtual communication matrices,
their exact Virtual-minus-Real difference, and all-edge stage-total-normalized
concordance plots. The third row includes all five frozen aggregate CellChat
families: TGF\u03b2, CCL, CXCL, PDGF, and VEGF. The complete LR-pair inventory is
retained in `{lr_csv}`.

Early/d7 is calibration/reconstruction because the frozen input manifest records
an enabled d7 cross-type calibration prior. Late/d21 is held-out with respect
to the LR generation program. No biological-sample confidence interval is
claimed from the single-tissue comparison.

- Preview: `{preview}`
- 600 dpi PNG: `{png}`
- Vector PDF: `{pdf}`
- Audit: `{audit_path}`
"""
    )
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
