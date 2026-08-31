#!/usr/bin/env python3
"""Verify the GPT-2 R2 and PC1-PC4 evidence overlay without recomputation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent

EXPECTED = {
    "r2": (
        ROOT / "evidence/gpt2_lora_r2_strict/results",
        "GPT2_LORA_LOW_RESPONSE_PARETO_STRICT_CONFIRMED",
        8,
    ),
    "pc1": (
        ROOT / "evidence/predictive_causal_boundaries/pc1/results",
        "PREDICTIVE_RESPONSE_FIBRE_CAPACITY_NOT_SUPPORTED",
        8,
    ),
    "pc2": (
        ROOT / "evidence/predictive_causal_boundaries/pc2/results",
        "RESPONSE_FIBRE_CAPACITY_OPTIMIZER_ACCESS_NOT_SUPPORTED",
        8,
    ),
    "pc3": (
        ROOT / "evidence/predictive_causal_boundaries/pc3/results",
        "DYNAMIC_RESPONSE_FIBRE_CAPACITY_PREDICTIVE_STATE_NOT_SUPPORTED",
        16,
    ),
    "pc4": (
        ROOT / "evidence/predictive_causal_boundaries/pc4/results",
        "CAUSAL_RESPONSE_FIBRE_CAPACITY_NOT_SUPPORTED",
        8,
    ),
}


def one(root: Path, pattern: str) -> Path | None:
    found = sorted(root.rglob(pattern))
    return found[0] if len(found) == 1 else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    failures: list[str] = []
    incomplete: list[str] = []

    for name, (root, decision, seed_count) in EXPECTED.items():
        report = one(root, "report.json")
        if report is None:
            incomplete.append(f"{name}: exactly one report.json required")
            continue
        data = json.loads(report.read_text(encoding="utf-8"))
        if data.get("decision") != decision:
            failures.append(
                f"{name}: decision {data.get('decision')!r} != {decision!r}"
            )
        seeds = sorted(root.rglob("seed_*.json"))
        if len(seeds) != seed_count:
            failures.append(f"{name}: found {len(seeds)} seed records; expected {seed_count}")
        if one(root, "protocol.json") is None:
            failures.append(f"{name}: exactly one protocol.json required")
        if one(root, "node_metrics.csv") is None:
            failures.append(f"{name}: exactly one node_metrics.csv required")
        print(f"[ok] {name}: {data.get('decision')} ({len(seeds)} seed records)")

    for item in incomplete:
        print(f"[incomplete] {item}")
    for item in failures:
        print(f"[fail] {item}")

    if failures or (args.require_complete and incomplete):
        return 1
    print("Overlay is structurally valid." if not incomplete else "Available evidence is valid; PC2 remains blocked.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
