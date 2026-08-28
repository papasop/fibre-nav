#!/usr/bin/env python3
"""Recompute v3.1e leave-one-radius-out exponent robustness."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "evidence/confirmed/moving_fibre_v3_1e/results/cner_cnn_mnist_realisability_flat_confirm_v3_1e_results"
OUTDIR = ROOT / "docs/robustness_v3_1e"
RADII = [0.08, 0.04, 0.02, 0.01]


def slope(xs: list[float], ys: list[float]) -> float:
    return float(np.polyfit(np.log(np.asarray(xs)), np.log(np.asarray(ys)), 1)[0])


records: list[dict[str, float | str | int]] = []
for omitted in [None, *range(4)]:
    moving: list[float] = []
    fixed: list[float] = []
    keep = [i for i in range(4) if i != omitted]
    for path in sorted(RESULTS.glob("seed_*.json")):
        data = json.loads(path.read_text())
        if not (data["source_gate"] and data["source_chart_gate"] and data["source_metric_gate"]):
            continue
        arms = data["arms"]
        if not all(arms[f"moving_r{i}"]["admissible"] and arms[f"fixed_r{i}"]["admissible"] for i in range(4)):
            continue
        moving.append(slope([RADII[i] for i in keep], [arms[f"moving_r{i}"]["cumulative_retraction_fisher_length"] for i in keep]))
        fixed.append(slope([RADII[i] for i in keep], [arms[f"fixed_r{i}"]["cumulative_retraction_fisher_length"] for i in keep]))
    mm, fm = float(np.median(moving)), float(np.median(fixed))
    records.append({"fit": "all_four" if omitted is None else f"omit_{RADII[omitted]:.2f}", "omitted_radius": "" if omitted is None else RADII[omitted], "comparable_seeds": len(moving), "moving_alpha_median": mm, "fixed_alpha_median": fm, "median_separation": mm - fm})

baseline = records[0]
for row in records:
    row["moving_shift_from_full"] = float(row["moving_alpha_median"]) - float(baseline["moving_alpha_median"])
    row["fixed_shift_from_full"] = float(row["fixed_alpha_median"]) - float(baseline["fixed_alpha_median"])

summary = {"analysis_status": "post_confirmation_robustness", "source": str(RESULTS.relative_to(ROOT)), "records": records, "max_abs_moving_shift": max(abs(float(r["moving_shift_from_full"])) for r in records[1:]), "max_abs_fixed_shift": max(abs(float(r["fixed_shift_from_full"])) for r in records[1:]), "minimum_loo_median_separation": min(float(r["median_separation"]) for r in records[1:]), "frozen_separation_gate": 0.25}
OUTDIR.mkdir(parents=True, exist_ok=True)
(OUTDIR / "V31E_RADIUS_LOO_SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n")
with (OUTDIR / "V31E_RADIUS_LOO_SUMMARY.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(records[0]))
    writer.writeheader()
    writer.writerows(records)
print(json.dumps(summary, indent=2))
