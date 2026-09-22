#!/usr/bin/env python3
"""Write Chinese report for GSE267904 spatial COMMOT micro-agent experiment."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from common import read_json, ensure_dirs, outpath


def maybe_csv(path: Path) -> str:
    if not path.exists():
        return "未生成。"
    try:
        return pd.read_csv(path).to_markdown(index=False)
    except Exception:
        return path.read_text(encoding="utf-8")[:2000]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--project-root", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=Path("outputs/GSE267904_spatial_commot_agent"))
    a = p.parse_args()
    root = a.project_root.resolve()
    out = outpath(root, a.out_dir)
    ensure_dirs(out)
    preflight = read_json(out / "audit/commot_preflight.json", {})
    phys = read_json(out / "audit/physicell_writeback_audit.json", read_json(out / "audit/physicell_blocking_report.json", {}))
    metrics = maybe_csv(out / "evaluation/commot_summary_metrics.csv")
    audit = read_json(out / "audit/gse267904_spatial_commot_audit.json", {})
    text = f"""# GSE267904 Spatial COMMOT + Micro-Agent + PhysiCell Executor 实验报告

## 1. 这个实验想证明什么

这个实验用于验证虚拟细胞系统是否能在空间通讯层面接近真实组织。逻辑是：

```text
真实 GSE267904 Visium 空间转录组 → 真实 COMMOT 网络
50 个 micro-cell Agents + 空间执行器 → 虚拟空间表达/坐标
虚拟表达/坐标 → 虚拟 COMMOT 网络
最后比较真实与虚拟的 sender-receiver / pathway / key interactions
```

## 2. 数据和限制

- 数据集：GSE267904 pulmonary fibrosis spatial transcriptomics。
- 该数据是 Visium-like spot-level 空间转录组，不是纯单细胞空间数据。
- spot cell type 通过 marker score 推断，因此结论应写成 cell-type / pathway 层面的通讯模式比较，而不是逐 spot 精确复刻。

## 3. COMMOT / 环境 preflight

```json
{preflight}
```

## 4. PhysiCell 执行层状态

```json
{phys}
```

如果 `PHYSICELL_USED_AS_SPATIAL_EXECUTOR=false`，说明当前只完成了数据/COMMOT脚手架，不能声称真实 PhysiCell-backed micro-agent simulation 已完成。

## 5. COMMOT real-vs-virtual 指标

{metrics}

## 6. 可以怎么解释

如果 COMMOT 指标非空且相关性/Top edge overlap 为正，可以谨慎说：

```text
The spatial micro-agent system partially recovers cell-type-level communication structure.
```

不能说：

```text
The model exactly reconstructs spatial cell-cell communication.
```

## 7. 主要图

- `figures/gse267904_real_spatial_celltype_maps.png`
- `figures/gse267904_real_vs_virtual_sender_receiver_heatmap_d7_bleo.png`
- `figures/gse267904_real_vs_virtual_sender_receiver_heatmap_d21_bleo.png`
- `figures/gse267904_commot_edge_agreement_d7_bleo.png`
- `figures/gse267904_commot_edge_agreement_d21_bleo.png`
- `figures/gse267904_commot_metric_summary.png`

## 8. 数据谱系审计

```json
{audit}
```
"""
    report = out / "reports/gse267904_spatial_commot_agent_report.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(text, encoding="utf-8")
    print(f"Wrote report to {report}")


if __name__ == "__main__":
    main()

