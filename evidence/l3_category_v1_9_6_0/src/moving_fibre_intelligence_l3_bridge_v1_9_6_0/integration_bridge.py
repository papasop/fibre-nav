#!/usr/bin/env python3
"""MFI v1.9.6.0: prospective router -> compositional fibre-writer bridge."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import torch
import transformers


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_module(name, path):
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
    started = time.time()
    root = Path(__file__).resolve().parent
    bundle = root.parent
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    route_out = output / "routing"
    writer_out = output / "writer"
    route_out.mkdir(exist_ok=True)
    writer_out.mkdir(exist_ok=True)
    config_path = Path(args.config)
    cfg = json.loads(config_path.read_text())

    router_root = bundle / "moving_fibre_intelligence_l3_router_v1_9_6_0"
    writer_root = bundle / "moving_fibre_intelligence_l3_v1_7_8"
    route_cmd = [
        sys.executable, "-u", router_root / "routing_preflight.py",
        "--config", router_root / "config_quick.json", "--output", route_out,
    ]
    print("[stage 1/2] prospectively frozen fresh-seed/fresh-split router", flush=True)
    subprocess.run(list(map(str, route_cmd)), check=True)
    route_record = json.loads((route_out / "routing_preflight.json").read_text())

    predictions = {}
    for fold in route_record["folds"]:
        for result in fold["heldout_predictions"]:
            row = route_record["cohort"][result["index"]]
            predictions[row["name"]] = {**result, "description": row["description"]}

    address_by_category = cfg["address_by_category"]
    bridge_nodes = []
    for item in cfg["bridge_concepts"]:
        routed = predictions[item["name"]]
        bridge_nodes.append({
            "name": item["name"],
            "description": routed["description"],
            "true_category": item["category"],
            "routed_category": routed["prediction"],
            "routing_margin": routed["signed_score_margin"],
            "bits": address_by_category[routed["prediction"]],
        })

    print("[bridge] routed category -> compositional address", flush=True)
    for node in bridge_nodes:
        print(f"  {node['name']}: {node['routed_category']} -> {node['bits']} "
              f"(margin={node['routing_margin']:.4f})", flush=True)

    writer = load_module("mfi_writer_v178", writer_root / "fibre_memory_audit.py")
    writer.L3_NODES = [
        {"name": n["name"], "description": n["description"], "bits": n["bits"]}
        for n in bridge_nodes
    ]
    writer_cfg = dict(cfg)
    writer_cfg["protocol"] = cfg["protocol"]
    print("[stage 2/2] v1.7.8 slot-factorized writer on routed addresses", flush=True)
    writer_record = writer.one_seed_l3_writer(writer_cfg, cfg["seed"], writer_out)

    routed_correct = all(n["routed_category"] == n["true_category"] for n in bridge_nodes)
    route_positive = all(n["routing_margin"] > 0 for n in bridge_nodes)
    initial_assignment = writer_record["code_assignments"][0]
    assignment_matches = all(initial_assignment[n["name"]] == n["bits"] for n in bridge_nodes)
    gates = {
        "large_cohort_router_all_gates": bool(route_record["all_gates_pass"]),
        "bridge_concepts_routed_correctly": routed_correct,
        "bridge_concepts_positive_routing_margin": route_positive,
        "routed_categories_determine_initial_addresses": assignment_matches,
        "slot_writer_all_gates": bool(writer_record["all_gates_pass"]),
    }
    record = {
        "protocol": cfg["protocol"],
        "scientific_status": "PROSPECTIVE_SINGLE_SEED_L3_CONFIRMATION_CANDIDATE",
        "seed": cfg["seed"],
        "routing_protocol": route_record["protocol"],
        "routing_config_sha256": route_record["config_sha256"],
        "writer_source_protocol": "MOVING_FIBRE_INTELLIGENCE_L3_RANK8_PROSPECTIVE_V1_9_6_0",
        "bridge_nodes": bridge_nodes,
        "address_by_category": address_by_category,
        "writer_gates": writer_record["tertiary_gates"],
        "integration_gates": gates,
        "all_gates_pass": all(gates.values()),
        "config": cfg,
        "config_sha256": sha256_file(config_path),
        "elapsed_seconds": time.time() - started,
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
    }
    (output / "integration_bridge.json").write_text(
        json.dumps(record, indent=2, default=writer.json_default), encoding="utf-8"
    )
    summary = {k: record[k] for k in [
        "protocol", "scientific_status", "seed", "bridge_nodes",
        "writer_gates", "integration_gates", "all_gates_pass",
        "config_sha256", "elapsed_seconds", "python", "platform", "torch", "transformers",
    ]}
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, default=writer.json_default), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, default=writer.json_default))


if __name__ == "__main__":
    main()
