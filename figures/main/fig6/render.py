#!/usr/bin/env python3
"""Verify and repackage the exact manually assembled Figure 6 raster.

The authors confirmed that the final six-panel layout was assembled manually,
so no unified six-panel plotting script is expected. This renderer preserves
the canonical assembly without inferring scientific arrays from its pixels.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-lecavc-fig6")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "source_data/fig6/figure6_frozen_reference.png"
OUT = ROOT / "build/figures/fig6"
EXPECTED_SHA256 = "a1d7e849d42c56eb7010f033e5c5103b97c963c4fef2d3f36443ccbf62dcc3cf"
EXPECTED_SIZE = (7200, 3600)


def main() -> None:
    digest = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    if digest != EXPECTED_SHA256:
        raise RuntimeError(f"Figure 6 source hash mismatch: {digest}")

    with Image.open(SOURCE) as image:
        if image.size != EXPECTED_SIZE:
            raise RuntimeError(f"Figure 6 dimensions changed: {image.size}")
        image.verify()

    OUT.mkdir(parents=True, exist_ok=True)
    stem = OUT / "GSE267904_fibrosis_application"
    exact = stem.with_name(stem.name + "_frozen_reference_300dpi.png")
    shutil.copyfile(SOURCE, exact)

    with Image.open(SOURCE).convert("RGB") as image:
        preview = image.resize((2000, 1000), Image.Resampling.LANCZOS)
        preview.save(stem.with_name(stem.name + "_preview.png"), optimize=False)

        plt.rcParams.update({
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "svg.hashsalt": "lecavc-figure6-frozen-raster",
        })
        fig = plt.figure(figsize=(12, 6), frameon=False)
        ax = fig.add_axes((0, 0, 1, 1))
        ax.imshow(image, interpolation="none")
        ax.set_axis_off()
        fig.savefig(stem.with_suffix(".pdf"), dpi=600, metadata={"CreationDate": None})
        fig.savefig(stem.with_suffix(".svg"), dpi=600, metadata={"Date": None})
        plt.close(fig)

    audit = {
        "status": "PASS_FROZEN_RASTER_REPACKAGE",
        "source_sha256": digest,
        "source_dimensions": list(EXPECTED_SIZE),
        "source_dpi": 300,
        "scientific_values_recomputed": False,
        "manual_assembly_confirmed": True,
        "unified_six_panel_renderer_required": False,
        "pdf_and_svg": "raster-wrapped; not native vector reconstructions",
        "provenance": "source_data/fig6/provenance.json",
    }
    (OUT / "COMPLETE.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
