#!/usr/bin/env python3
"""Verify the GPT-2 LoRA global response-budget R10/R11 snapshot."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SNAP = ROOT / "evidence" / "gpt2_lora_global_response_budget_r10_r11"
R10_PROTOCOL = "GPT2_LORA_GLOBAL_RESPONSE_BUDGET_PARETO_V0_3_R10_DEVELOPMENT"
R11_PROTOCOL = "GPT2_LORA_GLOBAL_RESPONSE_BUDGET_PARETO_V0_3_R11_CONFIRMATORY"
R10_STATUS = "R10_GLOBAL_BUDGET_PARETO_DIAGNOSTIC_COMPLETE"
R11_STATUS = "R11_CURRENT_KERNEL_GLOBAL_BUDGET_PARETO_CONFIRMED"
R10_SEEDS = [25211, 25217, 25229, 25237, 25247]
R11_SEEDS = [27211, 27217, 27229, 27241, 27253]
R11_MEDIANS = {
    5e-05: 0.0037626028060913086,
    1e-04: 0.0036696195602416992,
    2e-04: 0.0035785436630249023,
    5e-04: 0.0031867027282714844,
}


def fail(message: str) -> None:
    print(f"FAILED: {message}", file=sys.stderr)
    raise SystemExit(1)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def verify_manifest() -> None:
    manifest = SNAP / "SHA256SUMS"
    if not manifest.is_file():
        fail("missing SHA256SUMS")
    for line in manifest.read_text().splitlines():
        if not line.strip():
            continue
        expected, rel = line.split(None, 1)
        path = SNAP / rel.lstrip(" *")
        if not path.is_file():
            fail(f"missing manifest file: {rel}")
        observed = sha256(path)
        if observed != expected:
            fail(f"hash mismatch: {rel}")


def verify_stage(summary: dict, protocol: str, status: str, seeds: list[int]) -> None:
    if summary.get("protocol") != protocol:
        fail(f"{protocol}: protocol mismatch")
    if summary.get("scientific_status") != status:
        fail(f"{protocol}: status mismatch")
    if summary.get("seeds") != seeds:
        fail(f"{protocol}: seed mismatch")
    if summary.get("supporting_budget_count") != 4:
        fail(f"{protocol}: expected four supporting budgets")
    if len(summary.get("pairs", [])) != 4:
        fail(f"{protocol}: expected four budget pairs")


def verify_r11(summary: dict) -> None:
    gates = summary.get("gates", {})
    if not gates or not all(gates.values()):
        fail("r11: not all frozen gates passed")
    for pair in summary["pairs"]:
        budget = pair.get("budget")
        if budget not in R11_MEDIANS:
            fail(f"r11: unexpected budget {budget}")
        if pair.get("current_positive_seeds") != 5:
            fail(f"r11: budget {budget} does not have 5/5 positive seeds")
        if pair.get("supports_frozen_gate") is not True:
            fail(f"r11: budget {budget} did not support frozen gate")
        observed = pair.get("median_source_minus_current_loss")
        expected = R11_MEDIANS[budget]
        if not math.isclose(observed, expected, rel_tol=0, abs_tol=1e-15):
            fail(f"r11: median mismatch for budget {budget}")


def main() -> None:
    if not SNAP.is_dir():
        fail("snapshot directory missing")
    verify_manifest()
    r10 = load_json(SNAP / "results" / "r10" / "run_summary.json")
    r11 = load_json(SNAP / "results" / "r11" / "run_summary.json")
    verify_stage(r10, R10_PROTOCOL, R10_STATUS, R10_SEEDS)
    verify_stage(r11, R11_PROTOCOL, R11_STATUS, R11_SEEDS)
    if r10.get("r7_r9_seed_reuse_intentional") is not True:
        fail("r10: seed reuse flag missing")
    verify_r11(r11)
    print("VERIFIED: GPT-2 LoRA global response-budget R10/R11 snapshot")


if __name__ == "__main__":
    main()
