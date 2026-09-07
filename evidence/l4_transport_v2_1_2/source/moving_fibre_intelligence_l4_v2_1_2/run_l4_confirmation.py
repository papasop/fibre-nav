#!/usr/bin/env python3
"""Prospective large-atlas confirmation and round-trip audit for MFI L4."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
import random
import sys
import time
import math
from pathlib import Path

import numpy as np
import torch
import transformers


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pair_distances(endpoints):
    names = list(endpoints)
    return {
        "--".join(sorted((a, b))): float((endpoints[a] - endpoints[b]).norm().item())
        for i, a in enumerate(names) for b in names[i + 1:]
    }


def relative_distortion(source, moved):
    return float(np.mean([
        abs(moved[k] / max(source[k], 1e-12) - 1.0) for k in source]))


def one_sided_sign_p(wins, total):
    """Exact P[X >= wins] for X~Binomial(total, 0.5)."""
    return sum(math.comb(total, k) for k in range(wins, total + 1)) / (2 ** total)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config_audit.json")
    parser.add_argument("--output", default="results")
    args = parser.parse_args()
    started = time.time()
    root = Path(__file__).resolve().parent
    config_path = Path(args.config).resolve()
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)

    w = load("mfi_l4_writer_v211", root / "writer_core.py")
    repair = load("mfi_l4_repair_v211", root / "per_address_repair.py")
    repair.install(w)
    # Eight concepts, with the four binary two-slot addresses repeated twice.
    # Cyclic rotations ensure every concept is tested under every address.
    w.L3_NODES = [
        {"name": "paris", "description": "Paris, the capital city of France", "bits": [0, 0]},
        {"name": "france", "description": "France, a country in Europe", "bits": [0, 1]},
        {"name": "europe", "description": "Europe, the continent containing France", "bits": [1, 0]},
        {"name": "banana", "description": "banana, a yellow fruit", "bits": [1, 1]},
        {"name": "tokyo", "description": "Tokyo, the capital city of Japan", "bits": [0, 0]},
        {"name": "japan", "description": "Japan, an island country in East Asia", "bits": [0, 1]},
        {"name": "asia", "description": "Asia, the continent containing Japan", "bits": [1, 0]},
        {"name": "apple", "description": "apple, a round crisp fruit", "bits": [1, 1]}
    ]
    seed = cfg["seed"]
    w.set_seed(seed)
    device, _ = w.device_dtype()
    tok, model, params = w.make_model(cfg, device)
    sym0, sym1 = w.single_token_pair(tok)
    candidate_ids = [sym0[1], sym1[1]]
    rng = random.Random(seed)
    anchors = rng.sample(w.ANCHORS, cfg["anchor_count"])
    response_ids = rng.sample(
        list(range(100, min(50000, tok.vocab_size))), cfg["response_dim"])
    with torch.no_grad():
        logits0 = w.next_logits(model, tok, anchors, device).detach()
        response0 = logits0[:, response_ids].mean(0).detach()
    origin = w.param_flat(params)
    base_codes = [node["bits"] for node in w.L3_NODES]
    source_by_rotation = []
    assignments = []
    source_runs = []

    print("[stage 1/2] constructing frozen source-fibre atlases", flush=True)
    for rotation in range(cfg["l3_code_rotations"]):
        codes = base_codes[rotation:] + base_codes[:rotation]
        endpoints = {}
        records = {}
        assignment = {}
        for idx, (node, bits) in enumerate(zip(w.L3_NODES, codes)):
            w.assign_flat(params, origin)
            exact, train, held = w.concept_prompts(node)
            targets = [candidate_ids[b] for b in bits]
            train_targets = targets * (len(train) // len(exact))
            held_targets = targets * (len(held) // len(exact))
            result = w.run_write(
                model, tok, params, cfg, anchors, response_ids, response0, logits0,
                exact + train, targets + train_targets, exact, targets, train,
                held, held_targets, candidate_ids, "semantic_current",
                seed + 900000 + rotation * 10007 + idx * 1009, device,
                steps=cfg["l3_write_steps"])
            endpoint = w.param_flat(params).detach().cpu()
            endpoints[node["name"]] = endpoint
            records[node["name"]] = result
            assignment[node["name"]] = bits
            source_runs.append({
                "rotation": rotation, "concept": node["name"], "bits": bits,
                "response_drift": result["response_drift"],
                "anchor_kl": result["anchor_kl_diagnostic"],
                "exact_accuracy": result["exact_accuracy"],
                "heldout_accuracy": result["paraphrase_accuracy"]})
        source_by_rotation.append({
            "endpoints": endpoints, "records": records,
            "distances": pair_distances(endpoints), "codes": codes})
        assignments.append(assignment)

    move_sign = torch.tensor(
        [1.0 if i % 2 == 0 else -1.0 for i in range(response0.numel())],
        device=device)
    move_sign = move_sign / move_sign.norm()
    scale_records = []
    print("[stage 2/2] multi-scale transport, random controls and round trips", flush=True)

    for scale_index, shift in enumerate(cfg["l4_response_shifts"]):
        print(f"[L4] scale {scale_index + 1}/{len(cfg['l4_response_shifts'])}: {shift}", flush=True)
        target_response = response0 + float(shift) * move_sign
        true_samples = {}
        roundtrip_samples = {}
        random_samples = [dict() for _ in range(cfg["l4_random_repeats"])]
        endpoint_runs = []

        for rotation, atlas in enumerate(source_by_rotation):
            true_endpoints = {}
            roundtrip_endpoints = {}
            random_endpoints = [dict() for _ in range(cfg["l4_random_repeats"])]
            for idx, (node, bits) in enumerate(zip(w.L3_NODES, atlas["codes"])):
                name = node["name"]
                original = atlas["endpoints"][name].to(device)
                exact, _train, held = w.concept_prompts(node)
                targets = [candidate_ids[b] for b in bits]

                w.assign_flat(params, original)
                with torch.no_grad():
                    old_logits = w.next_logits(model, tok, anchors, device).detach()
                source_target_residual = float((
                    w.response_vector(model, tok, anchors, response_ids, device)
                    - target_response).abs().max().item())
                source_response = w.response_vector(
                    model, tok, anchors, response_ids, device)
                jacobian = w.jacobian_rows(source_response, params)

                w.retract(model, tok, params, anchors, response_ids, target_response,
                          device, cfg["l4_retraction_steps"])
                moved = w.param_flat(params)
                true_endpoints[name] = moved.detach().cpu()
                true_residual, true_kl = w.anchor_metrics(
                    model, tok, anchors, response_ids, target_response, old_logits, device)
                true_eval = w.evaluate_content(
                    model, tok, exact, held, targets, candidate_ids, device)
                transport_norm = float((moved - original).norm().item())

                # Return along the same response-normal retraction rule.
                with torch.no_grad():
                    moved_logits = w.next_logits(model, tok, anchors, device).detach()
                w.retract(model, tok, params, anchors, response_ids, response0,
                          device, cfg["l4_retraction_steps"])
                roundtrip = w.param_flat(params)
                roundtrip_endpoints[name] = roundtrip.detach().cpu()
                roundtrip_residual, roundtrip_kl = w.anchor_metrics(
                    model, tok, anchors, response_ids, response0, moved_logits, device)
                roundtrip_eval = w.evaluate_content(
                    model, tok, exact, held, targets, candidate_ids, device)

                random_rows = []
                for repeat in range(cfg["l4_random_repeats"]):
                    generator = torch.Generator(device=device).manual_seed(
                        seed + 7000000 + scale_index * 1000003
                        + rotation * 10007 + idx * 1009 + repeat * 97)
                    z = w.kernel_project(
                        torch.randn(original.shape, generator=generator, device=device),
                        jacobian)
                    z = z * (transport_norm / (z.norm() + 1e-12))
                    w.assign_flat(params, original + z)
                    w.retract(model, tok, params, anchors, response_ids, target_response,
                              device, cfg["l4_retraction_steps"])
                    random_point = w.param_flat(params)
                    random_endpoints[repeat][name] = random_point.detach().cpu()
                    random_residual, random_kl = w.anchor_metrics(
                        model, tok, anchors, response_ids, target_response,
                        old_logits, device)
                    random_eval = w.evaluate_content(
                        model, tok, exact, held, targets, candidate_ids, device)
                    random_rows.append({
                        "repeat": repeat,
                        "response_residual": random_residual,
                        "collateral_kl": random_kl,
                        "exact_accuracy": random_eval["exact_accuracy"],
                        "heldout_accuracy": random_eval["paraphrase_accuracy"]})

                endpoint_runs.append({
                    "rotation": rotation, "concept": name, "bits": bits,
                    "source_residual_to_target_fibre": source_target_residual,
                    "transport_norm": transport_norm,
                    "transport_response_residual": true_residual,
                    "transport_collateral_kl": true_kl,
                    "transport_exact_accuracy": true_eval["exact_accuracy"],
                    "transport_heldout_accuracy": true_eval["paraphrase_accuracy"],
                    "roundtrip_response_residual": roundtrip_residual,
                    "roundtrip_collateral_kl": roundtrip_kl,
                    "roundtrip_exact_accuracy": roundtrip_eval["exact_accuracy"],
                    "roundtrip_heldout_accuracy": roundtrip_eval["paraphrase_accuracy"],
                    "roundtrip_parameter_error": float((roundtrip - original).norm().item()),
                    "random_controls": random_rows})

            true_d = pair_distances(true_endpoints)
            roundtrip_d = pair_distances(roundtrip_endpoints)
            for key, value in true_d.items():
                true_samples.setdefault(key, []).append(value)
                roundtrip_samples.setdefault(key, []).append(roundtrip_d[key])
            for repeat, endpoint_set in enumerate(random_endpoints):
                random_d = pair_distances(endpoint_set)
                for key, value in random_d.items():
                    random_samples[repeat].setdefault(key, []).append(value)

        source_distances = {
            key: float(np.mean([atlas["distances"][key] for atlas in source_by_rotation]))
            for key in source_by_rotation[0]["distances"]}
        true_distances = {k: float(np.mean(v)) for k, v in true_samples.items()}
        roundtrip_distances = {k: float(np.mean(v)) for k, v in roundtrip_samples.items()}
        random_distance_sets = [
            {k: float(np.mean(v)) for k, v in sample.items()}
            for sample in random_samples]
        true_distortion = relative_distortion(source_distances, true_distances)
        roundtrip_distortion = relative_distortion(source_distances, roundtrip_distances)
        random_distortions = [
            relative_distortion(source_distances, d) for d in random_distance_sets]
        median_random_distortion = float(np.median(random_distortions))
        ratio = true_distortion / max(median_random_distortion, 1e-12)
        pair_wins = 0
        for key in source_distances:
            true_error = abs(true_distances[key] / source_distances[key] - 1.0)
            random_errors = [
                abs(d[key] / source_distances[key] - 1.0)
                for d in random_distance_sets]
            pair_wins += true_error < float(np.median(random_errors))
        pair_win_fraction = pair_wins / len(source_distances)
        sign_test_p = one_sided_sign_p(pair_wins, len(source_distances))
        scale_gates = {
            "source_outside_target_fibre": all(
                r["source_residual_to_target_fibre"] > cfg["l4_response_budget"]
                for r in endpoint_runs),
            "all_transports_reach_target_fibre": all(
                r["transport_response_residual"] <= cfg["l4_response_budget"]
                for r in endpoint_runs),
            "all_transports_within_collateral_kl": all(
                r["transport_collateral_kl"] <= cfg["l4_collateral_kl_gate"]
                for r in endpoint_runs),
            "transport_preserves_exact_access": all(
                r["transport_exact_accuracy"] >= cfg["train_accuracy_gate"]
                for r in endpoint_runs),
            "transport_preserves_heldout_access": all(
                r["transport_heldout_accuracy"] >= cfg["paraphrase_accuracy_gate"]
                for r in endpoint_runs),
            "transport_geometry_within_absolute_gate": (
                true_distortion <= cfg["l4_geometry_distortion_gate"]),
            "transport_beats_median_random_ratio": (
                ratio <= cfg["l4_distortion_ratio_gate"]),
            "transport_wins_pairwise_comparisons": (
                pair_win_fraction >= cfg["l4_pairwise_win_gate"]),
            "transport_pairwise_sign_test": (
                sign_test_p <= cfg["l4_sign_test_alpha"]),
            "roundtrip_reaches_source_fibre": all(
                r["roundtrip_response_residual"] <= cfg["l4_response_budget"]
                for r in endpoint_runs),
            "roundtrip_preserves_exact_access": all(
                r["roundtrip_exact_accuracy"] >= cfg["train_accuracy_gate"]
                for r in endpoint_runs),
            "roundtrip_preserves_heldout_access": all(
                r["roundtrip_heldout_accuracy"] >= cfg["paraphrase_accuracy_gate"]
                for r in endpoint_runs),
            "roundtrip_preserves_pair_geometry": (
                roundtrip_distortion <= cfg["l4_roundtrip_geometry_distortion_gate"]),
        }
        scale_records.append({
            "response_shift": shift,
            "source_pairwise_distances": source_distances,
            "transported_pairwise_distances": true_distances,
            "roundtrip_pairwise_distances": roundtrip_distances,
            "random_pairwise_distance_sets": random_distance_sets,
            "transport_distortion": true_distortion,
            "roundtrip_distortion": roundtrip_distortion,
            "random_distortions": random_distortions,
            "median_random_distortion": median_random_distortion,
            "transport_to_median_random_ratio": ratio,
            "pairwise_wins": pair_wins,
            "pairwise_total": len(source_distances),
            "pairwise_win_fraction": pair_win_fraction,
            "pairwise_sign_test_p": sign_test_p,
            "endpoint_runs": endpoint_runs,
            "gates": scale_gates,
            "all_gates_pass": all(scale_gates.values())})
        print(
            f"[L4 scale={shift}] distortion={true_distortion:.6g}; "
            f"random_median={median_random_distortion:.6g}; ratio={ratio:.4f}; "
            f"pair_wins={pair_wins}/{len(source_distances)}; p={sign_test_p:.6g}; "
            f"pass={all(scale_gates.values())}", flush=True)

    source_gates = {
        "source_endpoints_response_eligible": all(
            r["response_drift"] <= cfg["response_budget"] for r in source_runs),
        "source_endpoints_kl_eligible": all(
            r["anchor_kl"] <= cfg["endpoint_kl_gate"] for r in source_runs),
        "source_exact_access": all(
            r["exact_accuracy"] >= cfg["train_accuracy_gate"] for r in source_runs),
        "source_heldout_access": all(
            r["heldout_accuracy"] >= cfg["paraphrase_accuracy_gate"] for r in source_runs),
        "declared_large_atlas_size": all(
            s["pairwise_total"] == cfg["l4_expected_pairwise_total"]
            for s in scale_records),
    }
    gates = {
        **source_gates,
        "all_transport_scales_pass": all(s["all_gates_pass"] for s in scale_records),
    }
    freeze = {
        "prospective_freeze": True,
        "seed": seed,
        "response_shifts": cfg["l4_response_shifts"],
        "random_repeats": cfg["l4_random_repeats"],
        "distortion_ratio_gate": cfg["l4_distortion_ratio_gate"],
        "pairwise_win_gate": cfg["l4_pairwise_win_gate"],
        "roundtrip_required": True,
        "heldout_used_for_checkpoint_selection": False,
        "hyperparameters_changed_after_freeze": False,
    }
    record = {
        "protocol": cfg["protocol"],
        "scientific_status": cfg["scientific_status"],
        "claim_boundary": "Single-seed local transport discrimination across three finite-response level sets in one GPT-2 LoRA chart; not global parallel transport or cross-model transport.",
        "prospective_freeze_record": freeze,
        "seed": seed,
        "symbols": [sym0, sym1],
        "code_assignments": assignments,
        "source_runs": source_runs,
        "scales": scale_records,
        "gates": gates,
        "all_gates_pass": all(gates.values()),
        "config": cfg,
        "config_sha256": sha256(config_path),
        "elapsed_seconds": time.time() - started,
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
    }
    (output / f"seed_{seed}.json").write_text(
        json.dumps(record, indent=2, default=w.json_default), encoding="utf-8")
    (output / "prospective_freeze.json").write_text(
        json.dumps(freeze, indent=2), encoding="utf-8")
    summary = {
        "protocol": record["protocol"],
        "scientific_status": record["scientific_status"],
        "claim_boundary": record["claim_boundary"],
        "seed": seed,
        "scale_summaries": [{k: s[k] for k in [
            "response_shift", "transport_distortion", "roundtrip_distortion",
            "median_random_distortion", "transport_to_median_random_ratio",
            "pairwise_wins", "pairwise_total", "pairwise_sign_test_p",
            "gates", "all_gates_pass"]}
            for s in scale_records],
        "gates": gates,
        "all_gates_pass": record["all_gates_pass"],
        "config_sha256": record["config_sha256"],
        "elapsed_seconds": record["elapsed_seconds"],
        "python": record["python"], "platform": record["platform"],
        "torch": record["torch"], "transformers": record["transformers"],
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, default=w.json_default), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=w.json_default))


if __name__ == "__main__":
    main()
