#!/usr/bin/env python3
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


expected = {
    "release_assets/moving_fibre_intelligence_l5_v2_2_4.zip": "7979dd3084a446c7ec9efb2303b105966951614ccc9459c0f335b6c8319abd5e",
    "release_assets/run_moving_fibre_intelligence_l5_v2_2_4.py": "f42ac98a7ebcecfac646b35f05bd5a130b013e1f5406431288d1cbfff20ef8e9",
    "results/v2_2_4/results.zip": "ca6753c0143f7a583094b8b5f31b1ed079eed90440ff4423a8c8c766234c2166",
    "development/v2_2_3_results.zip": "ea0f9886f9fe227b80684adbacfe1e19cfa09a3b5fa3bbc3d270bb1ee7591ec5",
}

for relative, digest in expected.items():
    actual = sha256(ROOT / relative)
    if actual != digest:
        raise SystemExit(f"SHA mismatch: {relative}: {actual} != {digest}")

summary = json.loads((ROOT / "results/v2_2_4/SUMMARY.json").read_text())
if summary["protocol"] != "MOVING_FIBRE_INTELLIGENCE_L5_FRESH_SEED_EXPRESSION_CONFIRMATION_V2_2_4":
    raise SystemExit("Protocol mismatch")
if summary["config_sha256"] != "1165673fbe1a993ca86369b26df246baa8b3939fe0bd26caa1b49db6716a1369":
    raise SystemExit("Config SHA mismatch")
if not summary["all_gates_pass"] or not all(summary["gates"].values()):
    raise SystemExit("Not all frozen gates passed")

print("MFI L5 v2.2.4 snapshot verified")
