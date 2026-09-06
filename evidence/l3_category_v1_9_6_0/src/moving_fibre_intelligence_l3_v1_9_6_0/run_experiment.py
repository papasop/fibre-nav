#!/usr/bin/env python3
"""MFI L3 v1.9.6.0 prospectively frozen single-seed confirmation."""
import argparse
import importlib.util
import json
import sys
from pathlib import Path


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config_quick.json")
    parser.add_argument("--output", default="results")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    bundle = root.parent
    bridge = load("mfi_bridge_v1960", bundle / "moving_fibre_intelligence_l3_bridge_v1_9_6_0" / "integration_bridge.py")
    repair = load("mfi_per_address_repair_v1952", root / "per_address_repair.py")
    writer = load("mfi_rank8_writer_v1960", root / "rank8_writer.py")
    repair.install(writer)
    original_loader = bridge.load_module

    def prospective_loader(name, path):
        if name == "mfi_writer_v178":
            return writer
        return original_loader(name, path)

    bridge.load_module = prospective_loader
    previous = sys.argv
    sys.argv = [previous[0], "--config", str(Path(args.config).resolve()),
                "--output", str(Path(args.output).resolve())]
    try:
        bridge.main()
    finally:
        sys.argv = previous

    output = Path(args.output)
    freeze = {
        "prospective_freeze": True,
        "baseline": "v1.9.5.7 rank-8 all-gates-pass development configuration",
        "new_writer_seed": 82601,
        "new_router_split_seed": 196001,
        "new_bridge_concepts": ["dog", "tokyo", "japan", "apple"],
        "hyperparameters_changed_after_freeze": False,
        "checkpoint_uses_heldout_expressions": False,
        "confirmation_scope": "single-seed candidate; multi-seed confirmation remains required"
    }
    for filename in ["integration_bridge.json", "summary.json"]:
        path = output / filename
        record = json.loads(path.read_text())
        record["prospective_freeze_record"] = freeze
        path.write_text(json.dumps(record, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
