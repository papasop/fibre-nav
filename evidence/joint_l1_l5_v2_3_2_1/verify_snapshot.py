#!/usr/bin/env python3
"""Verify archived v2.3.2.1 artifacts and load-bearing gates."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXPECTED = {
    "results.zip": "e4e5f85c444a9ddc2a42d343bdde3d9d186fedfe3b31641220d1dbb356c9960e",
    "release_assets/moving_fibre_intelligence_l1_l5_independent_smoke_v2_3_2_1.zip": "323c72ad695f4b17fbaa6ca176c43dc49a3cd8278dae746b76429ee89b6a07a1",
    "release_assets/run_moving_fibre_intelligence_l1_l5_independent_smoke_v2_3_2_1.py": "c3d95c5aafb3d7ed2d483273e112f21c9dc5a5d944192528d2f6b31dfe9c9ecb",
}

for relative, expected in EXPECTED.items():
    actual = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
    if actual != expected:
        raise SystemExit(f"SHA mismatch: {relative}: {actual}")

summary = json.loads((ROOT / "SUMMARY.json").read_text(encoding="utf-8"))
if summary["scientific_status"] != "POST_FAILURE_DEVELOPMENT_REPAIR_NOT_CONFIRMATION":
    raise SystemExit("Unexpected scientific status")
if not summary["all_layer_smoke_gates_pass"] or not summary["all_gates_pass"]:
    raise SystemExit("Archived gates do not all pass")
print("v2.3.2.1 snapshot verification: PASS")
