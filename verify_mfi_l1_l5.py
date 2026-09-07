#!/usr/bin/env python3
"""Verify Moving Fibre Intelligence L1-L5 archive status without recomputation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def load_json(path: str) -> dict:
    p = ROOT / path
    if not p.exists():
        raise AssertionError(f"missing JSON: {path}")
    return json.loads(p.read_text(encoding="utf-8"))


def require(path: str) -> Path:
    p = ROOT / path
    if not p.exists():
        raise AssertionError(f"missing required path: {path}")
    return p


def check_manifest(directory: str) -> int:
    root = require(directory)
    manifest = root / "SHA256SUMS.txt"
    if not manifest.exists():
        raise AssertionError(f"{directory}: missing SHA256SUMS.txt")
    checked = 0
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, rel = line.split("  ", 1)
        target = root / rel
        if not target.exists():
            raise AssertionError(f"{directory}: missing manifest target {rel}")
        actual = hashlib.sha256(target.read_bytes()).hexdigest()
        if actual != expected:
            raise AssertionError(f"{directory}: hash mismatch for {rel}")
        checked += 1
    return checked


def main() -> int:
    layers: list[tuple[str, str]] = []

    l1 = load_json("evidence/l1_exact_anchoring_v1_3_2/evidence/v1_3_2/summary.json")
    assert l1["scientific_status"] == "DEVELOPMENT_NOT_CONFIRMATION"
    assert l1["all_seeds_primary_pass"] is True
    assert l1["all_seeds_secondary_pass"] is False
    layers.append(("L1", "single-seed mechanism supported; secondary paraphrase gates failed"))

    require("evidence/l2_heldout_expression_v1_5_6/RESULT_ARCHIVE_REQUIRED.md")
    require("evidence/l2_heldout_expression_v1_5_6/release_assets/gpt2_fibre_memory_l2_v1_5_6.zip")
    layers.append(("L2", "source/protocol archived; standalone result archive unavailable"))

    l3_writer = load_json("evidence/l3_writer_v1_7_8/summary.json")
    assert l3_writer["scientific_status"] == "DEVELOPMENT_NOT_CONFIRMATION"
    assert l3_writer["all_seeds_pass"] is True
    layers.append(("L3 writer", "v1.7.8 single-seed all gates passed"))

    l3 = load_json("evidence/l3_category_v1_9_6_0/evidence/v1_9_6_0/summary.json")
    routing = load_json("evidence/l3_category_v1_9_6_0/evidence/v1_9_6_0/routing/summary.json")
    assert l3["scientific_status"] == "PROSPECTIVE_SINGLE_SEED_L3_CONFIRMATION_CANDIDATE"
    assert l3["all_gates_pass"] is True
    assert routing["all_gates_pass"] is True
    layers.append(("L3 category", "v1.9.6.0 single-seed confirmation candidate"))

    l4 = load_json("evidence/l4_transport_v2_1_2/evidence/v2_1_2/summary.json")
    assert l4["scientific_status"] == "PROSPECTIVELY_FROZEN_SINGLE_SEED_L4_CONFIRMATION_CANDIDATE"
    assert l4["all_gates_pass"] is True
    require("evidence/l4_transport_v2_1_2/ERRATA.md")
    require("evidence/l4_transport_v2_1_2/STATISTICAL_NOTE.md")
    layers.append(("L4", "v2.1.2 single-seed local transport candidate"))

    l5 = load_json("evidence/l5_programmable_v2_2_4/results/v2_2_4/SUMMARY.json")
    assert l5["scientific_status"] == "PROSPECTIVELY_FROZEN_SINGLE_SEED_L5_CONFIRMATION_CANDIDATE"
    assert l5["all_gates_pass"] is True
    assert len(l5["gates"]) == 16
    layers.append(("L5", "v2.2.4 single-seed programmable-address confirmation candidate"))

    joint = load_json("evidence/joint_l1_l5_v2_3_2_1/SUMMARY.json")
    assert joint["scientific_status"] == "POST_FAILURE_DEVELOPMENT_REPAIR_NOT_CONFIRMATION"
    assert joint["seed"] == 83001
    assert joint["all_layer_smoke_gates_pass"] is True
    assert joint["all_gates_pass"] is True
    layers.append(("Joint L1-L5 smoke", "v2.3.2.1 development PASS; not confirmation"))

    for directory in [
        "evidence/l1_exact_anchoring_v1_3_2",
        "evidence/l2_heldout_expression_v1_5_6",
        "evidence/l3_writer_v1_7_8",
        "evidence/l3_category_v1_9_6_0",
        "evidence/l4_transport_v2_1_2",
        "evidence/l5_programmable_v2_2_4",
        "evidence/joint_l1_l5_v2_3_2_1",
    ]:
        count = check_manifest(directory)
        print(f"[ok] {directory}: SHA256SUMS.txt verified ({count} entries)")

    for name, status in layers:
        print(f"[ok] {name}: {status}")

    print("MFI L1-L5 archive status verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
