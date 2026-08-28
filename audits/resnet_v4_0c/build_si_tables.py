#!/usr/bin/env python3
"""Build machine-readable SI tables from the archived v4.0c-r1 report."""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULT_ROOT = (
    ROOT
    / "evidence/confirmed/resnet18_cifar10_fibre_v4_0c_r1/results"
    / "cner_resnet18_cifar10_fibre_external_precision_confirm_v4_0c_r1_results"
)
OUT = ROOT / "docs/si_v4_0c"


def median(values: list[float]) -> float:
    return float(statistics.median(values))


def main() -> None:
    report = json.loads((RESULT_ROOT / "report.json").read_text(encoding="utf-8"))
    rows: list[dict[str, object]] = []
    for seed_row in report["per_seed"]:
        for anchor_key, anchor in seed_row["anchor_results"].items():
            loss = anchor["confirm_loss_deltas"]
            leakage = anchor["response_leakage_max"]
            residual = anchor["kernel_relative_residual"]
            gates = anchor["gates"]
            rows.append(
                {
                    "seed": seed_row["seed"],
                    "base_confirm_accuracy": seed_row["base_confirm_accuracy"],
                    "accuracy_gate": seed_row["accuracy_gate"],
                    "anchor_count": int(anchor_key),
                    "response_rank": anchor["response_rank"],
                    "null_dimension": anchor["null_dimension"],
                    "projected_gradient_share": anchor["projected_gradient_share"],
                    "true_loss_delta": loss["true"],
                    "anti_loss_delta": loss["anti"],
                    "shuffled_loss_delta": loss["shuffled"],
                    "random_best_loss_delta": loss["random_best"],
                    "ambient_raw_loss_delta": loss["ambient_raw"],
                    "true_kernel_relative_residual": residual["true"],
                    "true_response_leakage_max": leakage["true"],
                    **{f"gate_{name}": value for name, value in gates.items()},
                    "candidate": anchor["candidate"],
                }
            )

    OUT.mkdir(parents=True, exist_ok=True)
    seed_table = OUT / "SI_V40C_SEED_GATES_48.csv"
    with seed_table.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: (row["anchor_count"], row["seed"])))

    summary: list[dict[str, object]] = []
    for anchor_count in sorted({int(row["anchor_count"]) for row in rows}):
        group = [row for row in rows if row["anchor_count"] == anchor_count]
        summary.append(
            {
                "anchor_count": anchor_count,
                "candidate_seeds": sum(bool(row["candidate"]) for row in group),
                "total_seeds": len(group),
                "true_loss_delta_median": median([float(row["true_loss_delta"]) for row in group]),
                "anti_loss_delta_median": median([float(row["anti_loss_delta"]) for row in group]),
                "shuffled_loss_delta_median": median([float(row["shuffled_loss_delta"]) for row in group]),
                "random_best_loss_delta_median": median([float(row["random_best_loss_delta"]) for row in group]),
                "projected_gradient_share_median": median(
                    [float(row["projected_gradient_share"]) for row in group]
                ),
                "max_true_kernel_relative_residual": max(
                    float(row["true_kernel_relative_residual"]) for row in group
                ),
                "max_true_response_leakage": max(
                    float(row["true_response_leakage_max"]) for row in group
                ),
            }
        )

    summary_table = OUT / "SI_V40C_ANCHOR_SUMMARY.csv"
    with summary_table.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)

    print(seed_table.relative_to(ROOT))
    print(summary_table.relative_to(ROOT))


if __name__ == "__main__":
    main()
