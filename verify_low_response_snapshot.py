#!/usr/bin/env python3
"""Verify the archived low-response Pareto evidence without rerunning models."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def verify_manifest() -> None:
    manifest = ROOT / "MANIFEST_LOW_RESPONSE_PARETO_V1.sha256"
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative = line.split("  ", 1)
        path = ROOT / relative
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f"SHA-256 mismatch: {relative}")


def load(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def verify_reports() -> None:
    resnet = load(
        "evidence/low_response_pareto_v1/resnet_v4_6/results/report.json"
    )
    gpt2 = load(
        "evidence/low_response_pareto_v1/gpt2_lora_b_v1/results/report.json"
    )

    assert resnet["protocol"] == "CNER_RESNET18_CIFAR10_LOW_RESPONSE_PARETO_V4_6_CPU"
    assert resnet["decision"] == "LOW_RESPONSE_PARETO_ADVANTAGE_SUPPORTED"
    assert resnet["attempted"] == 8
    assert resnet["supporting_seeds"] == 7

    assert gpt2["protocol"] == "GPTW_GPT2_NATIVE_LORA_B_LOW_RESPONSE_PARETO_CPU_V1_R1_8SEED"
    assert gpt2["decision"] == "GPT2_LORA_LOW_RESPONSE_PARETO_CONFIRMED"
    assert gpt2["attempted"] == 8
    assert gpt2["supporting_seeds"] == 8
    noninitial = sum(
        node["step"] > 0
        for record in gpt2["records"]
        for node in record["nodes"]
    )
    assert noninitial == 24


if __name__ == "__main__":
    verify_manifest()
    verify_reports()
    print("LOW_RESPONSE_PARETO_V1 snapshot verification: PASS")
