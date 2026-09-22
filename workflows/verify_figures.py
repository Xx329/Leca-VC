#!/usr/bin/env python3
"""Pixel-check all currently reproducible figures against frozen references."""

from __future__ import annotations
import json
from pathlib import Path
from PIL import Image, ImageChops

ROOT=Path(__file__).resolve().parents[1]
PAIRS={
    "fig2": ("figures/main/fig2/GSE2565_bulk_time_resolved_transcriptomic_response_benchmark_deidentified_v1_preview.png","build/figures/fig2/GSE2565_bulk_time_resolved_transcriptomic_response_benchmark_deidentified_v1_preview.png"),
    "fig3": ("figures/main/fig3/Leca_VC_multibenchmark_compact_no_WOT_v4_4_preview.png","build/figures/fig3/Leca_VC_multibenchmark_compact_no_WOT_v4_4_preview.png"),
    "fig4": ("figures/main/fig4/GSE120575_expression_accuracy_and_heterogeneity_horizontal_preview.png","build/figures/fig4/figure/GSE120575_expression_accuracy_and_heterogeneity_horizontal_preview.png"),
    "fig5": ("figures/main/fig5/GSE267904_balanced_metrics_and_spatial_maps_preview.png","build/figures/fig5/GSE267904_balanced_metrics_and_spatial_maps_preview.png"),
    "fig6": ("figures/main/fig6/GSE267904_fibrosis_application_reference.png","build/figures/fig6/GSE267904_fibrosis_application_frozen_reference_300dpi.png"),
    "figS1": ("figures/supplementary/figS1/GSE120575_celltype_composition_with_CellRank_proxy_preview.png","build/figures/figS1/GSE120575_celltype_composition_with_CellRank_proxy_preview.png"),
    "figS2": ("figures/supplementary/figS2/GSE230538_cell_state_composition_multimethod_proxy_preview.png","build/figures/figS2/GSE230538_cell_state_composition_multimethod_proxy_preview.png"),
    "figS3": ("figures/supplementary/figS3/GSE267904_CCI_network_heatmaps_Leca_AC_preview.png","build/figures/figS3/GSE267904_CCI_network_heatmaps_Leca_AC_preview.png"),
    "figS4": ("figures/supplementary/figS4/GSE2565_functional_program_heatmaps_Leca_AC_preview.png","build/figures/figS4/GSE2565_functional_program_heatmaps_Leca_AC_preview.png"),
    "figS5": ("figures/supplementary/figS5/gse267904_k100_pathway_specific_spatial_commot_hotspots_smoothed.png","build/figures/figS5/gse267904_k100_pathway_specific_spatial_commot_hotspots_smoothed.png"),
    "figS6": ("figures/supplementary/figS6/representative_myofibroblast_agent_step60_preview.png","build/figures/figS6/representative_myofibroblast_agent_step60_preview.png"),
    "figS7": ("figures/supplementary/figS7/LecaVC_exact_runtime_prompt_contamination_probe_preview.png","build/figures/figS7/figure/LecaVC_exact_runtime_prompt_contamination_probe_preview.png"),
    "figS8": ("figures/supplementary/figS8/GSE267904_traceable_cellular_program_changes_clean_preview.png","build/figures/figS8/figure/GSE267904_traceable_cellular_program_changes_clean_preview.png")
}

def main() -> int:
    report={}; failed=False
    for name,(reference_name,candidate_name) in PAIRS.items():
        reference_path,candidate_path=ROOT/reference_name,ROOT/candidate_name
        if not reference_path.is_file() or not candidate_path.is_file():
            report[name]={"status":"MISSING"}; failed=True; continue
        reference=Image.open(reference_path).convert("RGB"); candidate=Image.open(candidate_path).convert("RGB")
        bbox=None if reference.size!=candidate.size else ImageChops.difference(reference,candidate).getbbox()
        identical=reference.size==candidate.size and bbox is None
        report[name]={"status":"PASS" if identical else "FAIL","reference_dimensions":reference.size,"candidate_dimensions":candidate.size,"pixel_identical":identical}
        if name == "fig6":
            report[name]["reference_scope"]="exact author-confirmed manual assembly asset"
        if name == "figS6":
            report[name]["reference_scope"]="release renderer values validated against the collaborator step-60 runtime record"
        failed|=not identical
    output=ROOT/"build/figure_visual_verification.json"; output.parent.mkdir(parents=True,exist_ok=True); output.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2)); return 2 if failed else 0

if __name__=="__main__": raise SystemExit(main())
