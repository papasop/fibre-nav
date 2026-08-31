#!/usr/bin/env python3
"""Verify the Picard fine-tuning v0.2.6 evidence snapshot."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SNAPSHOT = ROOT / "evidence" / "picard_finetune_v0_2_6"
EXPECTED_STATUS = "PICARD_V0_2_6_REPEATED_TIMING_DUAL_10PCT_SPEEDUP_SUPPORTED"
EXPECTED_PROTOCOL = "GEOMETRIC_INTRINSIC_PICARD_FINETUNE_V0_2_6_REPEATED_TIMING_CONFIRMATORY"
EXPECTED_SEEDS = [22229, 22247, 22259, 22271, 22277]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fail(message: str) -> None:
    print(f"FAILED: {message}", file=sys.stderr)
    raise SystemExit(1)


def verify_sha256s() -> None:
    sums = SNAPSHOT / "SHA256SUMS"
    if not sums.exists():
        fail("missing SHA256SUMS")
    for line in sums.read_text().splitlines():
        if not line.strip():
            continue
        expected, rel = line.split(None, 1)
        rel = rel[2:] if rel.startswith("./") else rel
        path = SNAPSHOT / rel
        if not path.exists():
            fail(f"manifest path missing: {rel}")
        observed = sha256(path)
        if observed != expected:
            fail(f"hash mismatch for {rel}: {observed} != {expected}")


def verify_summary() -> None:
    summary_path = SNAPSHOT / "evidence" / "v0_2_6" / "results" / "run_summary.json"
    manifest_path = SNAPSHOT / "MANIFEST.json"
    summary = json.loads(summary_path.read_text())
    manifest = json.loads(manifest_path.read_text())

    if summary.get("protocol") != EXPECTED_PROTOCOL:
        fail("unexpected protocol in run_summary.json")
    if summary.get("scientific_status") != EXPECTED_STATUS:
        fail("unexpected scientific status in run_summary.json")
    if manifest.get("scientific_status") != EXPECTED_STATUS:
        fail("unexpected scientific status in MANIFEST.json")

    pairs = summary.get("pairs", [])
    seeds = [pair.get("seed") for pair in pairs]
    if seeds != EXPECTED_SEEDS:
        fail(f"unexpected seed list: {seeds}")
    if not all(pair.get("valid") for pair in pairs):
        fail("not all seed pairs are valid")
    failed_gates = [
        name for name, passed in summary.get("gates", {}).items() if not passed
    ]
    if failed_gates:
        fail(f"failed gates: {failed_gates}")

    if summary.get("median_time_to_equal_loss_speedup_fraction", 0) < 0.10:
        fail("median time-to-equal-loss speedup below 10%")
    if summary.get("median_fixed_budget_speedup_fraction", 0) < 0.10:
        fail("median fixed-budget speedup below 10%")
    if summary.get("float64_response_leakage", 1) > 1e-10:
        fail("float64 response leakage above gate")


def main() -> None:
    if not SNAPSHOT.exists():
        fail("missing evidence/picard_finetune_v0_2_6")
    verify_sha256s()
    verify_summary()
    print("VERIFIED: Picard fine-tuning v0.2.6 supported evidence snapshot")


if __name__ == "__main__":
    main()
