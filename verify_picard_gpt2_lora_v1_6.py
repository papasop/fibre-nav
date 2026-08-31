#!/usr/bin/env python3
"""Verify the GPT-2 LoRA Picard r4/r5 v1.6.0 evidence layout."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
R4 = ROOT / "evidence" / "confirmed" / "picard_gpt2_lora_r4"
R5 = ROOT / "evidence" / "audits" / "picard_gpt2_lora_r5_ten_step"
DEV = ROOT / "evidence" / "developmental" / "picard_gpt2_lora"
EXPECTED_SEEDS = [23311, 23321, 23327, 23333, 23339]
EXPECTED_DATA_SHA = "86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed"

EXPECTED_RAW_SHA = {
    R4 / "raw" / "picard_gpt2_lora_confirm_v0_2_6_r4.zip": "8b1aae50ba28e981d49f4a2c40eac60dd193fd88139565f597ec8afc5ca8a105",
    R4 / "raw" / "picard_gpt2_lora_confirm_v0_2_6_r4_results.zip": "d1e18f8cc12c60bc13d1a72b4c56a55efff2a9babbb7b8bf19bcb3dc0a108547",
    R5 / "raw" / "picard_gpt2_lora_confirm_v0_2_6_r5.zip": "0ed7bc923ffae5fff596b5ceb90b215834281cbb3dd528dd8c75105adaa67ba1",
    R5 / "raw" / "picard_gpt2_lora_confirm_v0_2_6_r5_results.zip": "c86e3ab46e3dc65893b6d5cbf6dda8acc6b78ad589aa4c8cce8029b580d8e0b1",
    DEV / "r1" / "raw" / "picard_gpt2_lora_quick_v0_2_6_r1.zip": "b8d42803560621e6801458b12f5f016bf233ce26ec7a55778b05ae708dac33b1",
    DEV / "r1" / "raw" / "picard_gpt2_lora_quick_v0_2_6_r1_results.zip": "56e86103f0e3527db9a72a1ef03cf8ca1f48e3081bdba8d1c47d61d65ac73e4e",
    DEV / "r2" / "raw" / "picard_gpt2_lora_quick_v0_2_6_r2.zip": "85f4cbbe2ecf93c0b279edbac26e53099262fe35f73e1ef122e1db9d6ccb7100",
    DEV / "r2" / "raw" / "picard_gpt2_lora_quick_v0_2_6_r2_results.zip": "1f2728501d7582d6e562f3621cad0b36212e7853966f5b61cc0efd3a3adaad3f",
    DEV / "r3" / "raw" / "picard_gpt2_lora_quick_v0_2_6_r3.zip": "e000a2217ae8baa3130a44955d5cc50bd434daba2c129a118dec1c547cb4caeb",
    DEV / "r3" / "raw" / "picard_gpt2_lora_quick_v0_2_6_r3_results.zip": "79f2f638a6b4834847513eca016a3ee78d9106d7fb17f6451d2abf0680a01a73",
}


def fail(message: str) -> None:
    print(f"FAILED: {message}", file=sys.stderr)
    raise SystemExit(1)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def verify_raw_archives() -> None:
    for path, expected in EXPECTED_RAW_SHA.items():
        if not path.exists():
            fail(f"missing raw archive: {path.relative_to(ROOT)}")
        observed = sha256(path)
        if observed != expected:
            fail(f"hash mismatch for {path.relative_to(ROOT)}: {observed} != {expected}")


def verify_summary(root: Path, status: str, protocol: str, *, r5: bool = False) -> dict:
    summary = load_json(root / "run_summary.json")
    protocol_json = load_json(root / "protocol.json")
    if summary["scientific_status"] != status:
        fail(f"unexpected status in {root.relative_to(ROOT)}")
    if summary["protocol"] != protocol:
        fail(f"unexpected protocol in {root.relative_to(ROOT)}")
    if protocol_json["protocol"] != protocol:
        fail(f"protocol.json mismatch in {root.relative_to(ROOT)}")
    if summary["eval_seeds"] != EXPECTED_SEEDS:
        fail(f"unexpected seed list in {root.relative_to(ROOT)}")
    if summary["data_identity"]["sha256"] != EXPECTED_DATA_SHA:
        fail(f"dataset hash mismatch in {root.relative_to(ROOT)}")
    if summary["positive_seed_count"] != 5:
        fail(f"positive seed count is not 5 in {root.relative_to(ROOT)}")
    if not all(pair["valid"] for pair in summary["pairs"]):
        fail(f"invalid seed pair in {root.relative_to(ROOT)}")
    failed = [name for name, passed in summary["gates"].items() if not passed]
    if failed:
        fail(f"failed gates in {root.relative_to(ROOT)}: {failed}")
    if summary["median_time_to_equal_loss_speedup_fraction"] < 0.10:
        fail(f"equal-loss speedup below gate in {root.relative_to(ROOT)}")
    if r5 and summary.get("target_evaluation_interval_steps") != 10:
        fail("r5 is not the ten-step resolution audit")
    return summary


def verify_developmental_history() -> None:
    expected = {
        "r1": "GPT2_LORA_PICARD_V0_2_6_R1_QUICK_INCONCLUSIVE_FAIL_CLOSED",
        "r2": "GPT2_LORA_PICARD_V0_2_6_R2_WARM_CHART_INCONCLUSIVE_FAIL_CLOSED",
        "r3": "GPT2_LORA_PICARD_V0_2_6_R3_METRIC_EXPONENT_POSITIVE_SIGNAL",
    }
    for stage, status in expected.items():
        summary = load_json(DEV / stage / "raw" / "results" / "run_summary.json")
        if summary["scientific_status"] != status:
            fail(f"unexpected developmental status for {stage}")


def verify_comparison(r4: dict, r5: dict) -> None:
    comparison = load_json(R5 / "comparison_r4_r5.json")
    if comparison["r4"]["scientific_status"] != r4["scientific_status"]:
        fail("comparison_r4_r5.json r4 status mismatch")
    if comparison["r5"]["scientific_status"] != r5["scientific_status"]:
        fail("comparison_r4_r5.json r5 status mismatch")
    headline = comparison["headline_interpretation"]
    if headline["featured_speedup_percent"] != 37.46:
        fail("unexpected featured r5 speedup percent")
    if headline["featured_positive_seeds"] != "5/5":
        fail("unexpected featured positive seed count")
    if headline["fixed_600_step_time_advantage_percent_diagnostic"] != 1.17:
        fail("unexpected fixed-budget diagnostic percent")


def main() -> None:
    verify_raw_archives()
    r4 = verify_summary(
        R4,
        "GPT2_LORA_PICARD_V0_2_6_R4_FROZEN_CONFIRMATORY_SUPPORTED",
        "GPT2_LORA_PICARD_V0_2_6_R4_FROZEN_CONFIRMATORY",
    )
    r5 = verify_summary(
        R5,
        "GPT2_LORA_PICARD_V0_2_6_R5_TEN_STEP_RESOLUTION_SUPPORTED",
        "GPT2_LORA_PICARD_V0_2_6_R5_TEN_STEP_RESOLUTION_AUDIT",
        r5=True,
    )
    verify_developmental_history()
    verify_comparison(r4, r5)
    print("VERIFIED: GPT-2 LoRA Picard r4 confirmation and r5 audit")


if __name__ == "__main__":
    main()
