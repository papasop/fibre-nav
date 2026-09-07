#!/usr/bin/env python3
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent

for line in (ROOT / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    expected, relative = line.split("  ", 1)
    actual = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
    if actual != expected:
        raise SystemExit(f"SHA mismatch: {relative}: {actual} != {expected}")

summary = json.loads((ROOT / "THREE_SEED_SUMMARY.json").read_text(encoding="utf-8"))
assert summary["seeds_complete"] == 3
assert summary["seeds_all_gates_pass"] == 0
assert summary["all_seeds_pass"] is False
assert summary["layer_pass_counts"] == {"L1": 3, "L2": 0, "L3": 1, "L4": 1, "L5": 3}
print("MFI unified L1-L5 v2.4.1-r1 snapshot verified (0/3 full passes)")
