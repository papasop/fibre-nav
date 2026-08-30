#!/usr/bin/env python3
"""Recompute the load-bearing Section 5.3 and Section 6 statistics.

This script intentionally ignores the aggregate values in report.json and
reconstructs every displayed statistic from the 32 per-seed JSON records.
"""
from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
V41 = ROOT / "evidence/confirmed/resnet18_cifar10_dual_scaling_v4_1b/results"
V42 = ROOT / "evidence/confirmed/resnet18_cifar10_transverse_v4_2d/results"
OUT = ROOT / "docs/supplementary/tables"


def load_rows(folder: Path) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(folder.glob("seed_*.json"))]


def median(values):
    return statistics.median(values)


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    r41 = load_rows(V41)
    r42 = load_rows(V42)
    if len(r41) != 16 or len(r42) != 16:
        raise SystemExit("Expected 16 per-seed records for each protocol")

    table41 = []
    for row in r41:
        moving = row["fits"]["moving"]["alpha"]
        fixed = row["fits"]["fixed"]["alpha"]
        table41.append({
            "seed": row["seed"],
            "geometric_candidate": row["geometric_candidate"],
            "heldout_generalization_pass": row["heldout_generalization_pass"],
            "moving_alpha": moving,
            "fixed_alpha": fixed,
            "moving_minus_fixed_alpha": moving - fixed,
            "smallest_radius_fixed_over_moving": row["smallest_radius_ratios"]["fixed_over_moving"],
            "raw_dev_confirm_gradient_cosine": row["gradient_transfer_diagnostic"]["raw_gradient_cosine"],
            "projected_dev_confirm_gradient_cosine": row["gradient_transfer_diagnostic"]["projected_gradient_cosine"],
        })

    table42 = []
    for row in r42:
        s = row["transverse_amplification_summary"]
        table42.append({
            "seed": row["seed"],
            "candidate": row["candidate"],
            "direction_cosine": s["median_moving_fixed_direction_cosine"],
            "active_residual_ratio": s["median_active_residual_ratio_fixed_over_moving"],
            "transverse_response_gain": s["median_transverse_response_gain"],
            "transverse_gain_contrast": s["median_transverse_over_moving_gain"],
            "finest_finite_over_jvp": s["median_finest_finite_over_linear_prediction"],
            "maximum_jvp_additivity_error": s["maximum_jvp_linearity_relative_error"],
        })

    summary = {
        "section_5_3": {
            "records": len(table41),
            "passing_every_geometric_and_heldout_gate": sum(
                r["geometric_candidate"] and r["heldout_generalization_pass"] for r in r41
            ),
            "median_moving_minus_fixed_slope": median([r["moving_minus_fixed_alpha"] for r in table41]),
            "median_smallest_radius_fixed_over_moving_cost": median([r["smallest_radius_fixed_over_moving"] for r in table41]),
            "median_raw_dev_confirm_gradient_cosine": median([r["raw_dev_confirm_gradient_cosine"] for r in table41]),
            "median_projected_dev_confirm_gradient_cosine": median([r["projected_dev_confirm_gradient_cosine"] for r in table41]),
        },
        "section_6": {
            "records": len(table42),
            "passing_every_gate": sum(r["candidate"] for r in r42),
            "median_direction_cosine": median([r["direction_cosine"] for r in table42]),
            "median_active_residual_ratio": median([r["active_residual_ratio"] for r in table42]),
            "median_transverse_response_gain": median([r["transverse_response_gain"] for r in table42]),
            "median_transverse_gain_contrast": median([r["transverse_gain_contrast"] for r in table42]),
            "median_finest_finite_over_jvp": median([r["finest_finite_over_jvp"] for r in table42]),
            "median_maximum_jvp_additivity_error": median([r["maximum_jvp_additivity_error"] for r in table42]),
        },
    }

    write_csv(OUT / "SI_RESNET_V41B_SEEDS_16.csv", table41)
    write_csv(OUT / "SI_RESNET_V42D_SEEDS_16.csv", table42)
    (OUT / "SI_RESNET_SECTIONS_5_3_6_SUMMARY.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
