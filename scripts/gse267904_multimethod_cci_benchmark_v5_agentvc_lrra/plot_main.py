#!/usr/bin/env python3
"""Guarded V5 renderer; it cannot run without the final PASS marker."""
from __future__ import annotations

import importlib.util
import os
import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(os.environ.get("GSE267904_LRRA_OUT", str(ROOT / "outputs/GSE267904_multimethod_cci_benchmark_v5_agentvc_lrra"))).resolve()
FINAL_PASS = OUT / "PASS_AGENTVC_LRRA_TOP1_LOCAL_NO_REGRESSION.marker"
NO_FIGURE = OUT / "06_figure/NO_FIGURE_GENERATED.md"
V3_SCRIPT = ROOT / "scripts/gse267904_multimethod_cci_benchmark_v3_agentvc_hw"


def load_common():
    spec = importlib.util.spec_from_file_location("_v5_plot_common", V3_SCRIPT / "common.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load frozen V3 plotting constants")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.OUT = OUT
    module.PREDICTED_METHODS = [
        "scgpt", "scgen", "cellrank_proxy", "worker_lrra"
    ]
    module.ALL_SOURCES = [
        "observed", "scgpt", "scgen", "cellrank_proxy", "worker_lrra"
    ]
    module.METHOD_LABELS["worker_lrra"] = (
        "AgentVC d7-calibrated LR-role expression adapter (AgentVC-LRRA)"
    )
    module.COLORS["worker_lrra"] = "#CC3F8D"
    return module


def main() -> int:
    if not FINAL_PASS.is_file():
        raise RuntimeError("Figure generation forbidden: final V5 PASS marker is absent")
    if NO_FIGURE.is_file():
        raise RuntimeError("Inconsistent state: failure notice exists with final PASS marker")
    source = (V3_SCRIPT / "plot_main.py").read_text(encoding="utf-8")
    replacements = {
        '"worker_hw": "AgentVC-HW"': '"worker_lrra": "AgentVC-LRRA"',
        '"worker_hw": "AgentVC heterogeneous-worker\\nexpression proxy"':
            '"worker_lrra": "AgentVC d7-calibrated\\nLR-role adapter"',
        '"worker_hw"': '"worker_lrra"',
        "AgentVC-HW": "AgentVC-LRRA",
        "old→HW": "old→LRRA",
        "PASS_AGENTVC_HW_TOP1_GNRS_AND_LOCAL":
            "PASS_AGENTVC_LRRA_TOP1_LOCAL_NO_REGRESSION",
        "GSE267904 heterogeneous-worker cell–cell communication benchmark":
            "GSE267904 d7-calibrated LR-role cell–cell communication benchmark",
        "GSE267904_AgentVC_HW_CCI_main_v3": "GSE267904_AgentVC_LRRA_CCI_main_v5",
        "PASS_AGENTVC_HW_V3_SINGLE_3X3_FIGURE":
            "PASS_AGENTVC_LRRA_V5_SINGLE_3X3_FIGURE",
        "deterministically preserves d7 worker heterogeneity while conserving each frozen parent target":
            "uses frozen d7-calibrated pathway sender/receiver factors on the historical worker proxy",
        "post-hoc target-blind proxy": "post-hoc GSE267904 d7-calibrated proxy",
        "Post-hoc target-blind architectural repair":
            "Post-hoc GSE267904 d7-calibrated LR-role repair",
    }
    for old, new in replacements.items():
        source = source.replace(old, new)
    common = load_common()
    shim = types.ModuleType("common")
    for name in dir(common):
        if not name.startswith("__"):
            setattr(shim, name, getattr(common, name))
    previous = sys.modules.get("common")
    sys.modules["common"] = shim
    namespace = {
        "__name__": "_agentvc_lrra_v5_plot_impl",
        "__file__": str(V3_SCRIPT / "plot_main.py"),
    }
    try:
        exec(compile(source, namespace["__file__"], "exec"), namespace)
        return int(namespace["main"]())
    finally:
        if previous is None:
            sys.modules.pop("common", None)
        else:
            sys.modules["common"] = previous


if __name__ == "__main__":
    raise SystemExit(main())
