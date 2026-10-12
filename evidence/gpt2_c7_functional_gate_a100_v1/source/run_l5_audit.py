#!/usr/bin/env python3
"""Prospective fresh-seed/fresh-expression confirmation candidate for MFI L5."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
import random
import time
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


def prompts():
    cells = ["00", "01", "10", "11"]
    exact = [f"Programmable fibre cell [address={a}] contains:" for a in cells]
    prefixes = [
        "Read the value stored at",
        "Memory-controller lookup for",
        "Return only the stored symbol from",
        "Audit the current content of",
        "The programmable address bus selects",
        "After this unrelated sentence, retrieve",
        "What value is held at",
        "Recover the current content from",
        "Which stored symbol is associated with",
        "Query the selected fibre cell",
        "For this audit, report the content of",
        "Use the address register to retrieve",
    ]
    train = [f"{p} [address={a}]:" for p in prefixes for a in cells]
    # Prospectively frozen fresh family: none of these prefixes appeared in
    # the v2.2.0--v2.2.3 held-out sets or in the training prefix list above.
    held_prefixes = [
        "Independent evaluation: disclose the programmed symbol from",
        "Without restating any write command, inspect",
        "A new operator requests the retained payload at",
    ]
    # The address token and terminal colon define the audited read location.
    # Held-out expressions vary only before that canonical suffix; adding a
    # question mark after it would move GPT-2's causal prediction site and
    # confound expression invariance with a different address/read position.
    held = [f"{p} [address={a}]:" for p in held_prefixes for a in cells]
    return cells, exact, train, held


def expand_targets(targets, prompt_count):
    if prompt_count % len(targets):
        raise ValueError("Prompt count must contain complete address views")
    return targets * (prompt_count // len(targets))


def evaluate(w, model, tok, exact, held, targets, candidate_ids, device):
    held_targets = expand_targets(targets, len(held))
    exact_margins = w.signed_margins(model, tok, exact, targets, candidate_ids, device)
    held_margins = w.signed_margins(model, tok, held, held_targets, candidate_ids, device)
    return {
        "exact_accuracy": w.read_accuracy(model, tok, exact, targets, candidate_ids, device),
        "heldout_accuracy": w.read_accuracy(model, tok, held, held_targets, candidate_ids, device),
        "exact_margins": exact_margins,
        "heldout_margins": held_margins,
        "minimum_exact_margin": float(min(exact_margins)),
        "minimum_heldout_margin": float(min(held_margins)),
    }


def write_program(w, model, tok, params, cfg, anchors, response_ids, response0,
                  logits0, exact, train, held, targets, candidate_ids, seed,
                  device, steps):
    train_targets = expand_targets(targets, len(train))
    held_targets = expand_targets(targets, len(held))
    return w.run_write(
        model, tok, params, cfg, anchors, response_ids, response0, logits0,
        exact + train, targets + train_targets, exact, targets, train, held,
        held_targets, candidate_ids, "semantic_current", seed, device,
        steps=steps)


def content_bits(w, model, tok, exact, candidate_ids, device):
    logits = w.next_logits(model, tok, exact, device)[:, candidate_ids]
    return [int(x) for x in logits.argmax(-1).detach().cpu().tolist()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config_audit.json")
    ap.add_argument("--output", default="results")
    args = ap.parse_args()
    started = time.time()
    root = Path(__file__).resolve().parent
    config_path = Path(args.config).resolve()
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    out = Path(args.output).resolve(); out.mkdir(parents=True, exist_ok=True)
    w = load("mfi_l5_writer", root / "writer_core.py")
    repair = load("mfi_l5_repair", root / "per_address_repair.py")
    repair.install(w)

    seed = int(cfg["seed"])
    w.set_seed(seed)
    device, _ = w.device_dtype()
    tok, model, params = w.make_model(cfg, device)
    sym0, sym1 = w.single_token_pair(tok)
    candidate_ids = [sym0[1], sym1[1]]
    rng = random.Random(seed)
    anchors = rng.sample(w.ANCHORS, cfg["anchor_count"])
    response_ids = rng.sample(list(range(100, min(50000, tok.vocab_size))), cfg["response_dim"])
    with torch.no_grad():
        logits0 = w.next_logits(model, tok, anchors, device).detach()
        response0 = logits0[:, response_ids].mean(0).detach()
    origin = w.param_flat(params)
    cells, exact, train, held = prompts()

    programs = [
        {"name": "WRITE", "targets": [0, 0, 1, 1], "steps": cfg["l5_initial_write_steps"]},
        {"name": "OVERWRITE address=01", "targets": [0, 1, 1, 1], "steps": cfg["l5_operation_steps"]},
        {"name": "SWAP address=00,address=11", "targets": [1, 1, 1, 0], "steps": cfg["l5_swap_steps"]},
    ]
    stage_records = []
    snapshots = []
    previous_bits = None
    print("[L5] executing shared-state program", flush=True)
    for index, program in enumerate(programs):
        target_ids = [candidate_ids[b] for b in program["targets"]]
        result = write_program(
            w, model, tok, params, cfg, anchors, response_ids, response0, logits0,
            exact, train, held, target_ids, candidate_ids,
            seed + 100003 * (index + 1), device, program["steps"])
        state = w.param_flat(params).detach().clone()
        snapshots.append(state)
        ev = evaluate(w, model, tok, exact, held, target_ids, candidate_ids, device)
        observed = content_bits(w, model, tok, exact, candidate_ids, device)
        changed = [] if previous_bits is None else [i for i, (a, b) in enumerate(zip(previous_bits, observed)) if a != b]
        expected_changed = [] if index == 0 else [i for i, (a, b) in enumerate(zip(programs[index-1]["targets"], program["targets"])) if a != b]
        stage_records.append({
            "operation": program["name"], "declared_target_bits": program["targets"],
            "observed_exact_bits": observed, "changed_indices": changed,
            "expected_changed_indices": expected_changed,
            "unselected_addresses_preserved": index == 0 or all(
                observed[i] == previous_bits[i] for i in range(4) if i not in expected_changed),
            "response_drift": result["response_drift"],
            "anchor_kl": result["anchor_kl_diagnostic"], **ev})
        # Checkpoint selection remains training-only. The compact trace is
        # retained so the final selection can be audited without relying on
        # held-out expressions.
        stage_records[-1]["selected_checkpoint_step"] = result["selected_checkpoint_step"]
        stage_records[-1]["selected_training_tuple"] = result["selected_training_tuple"]
        stage_records[-1]["selected_train_score"] = result["selected_train_score"]
        stage_records[-1]["accepted_steps"] = result["accepted_steps"]
        stage_records[-1]["training_checkpoint_trace"] = [
            {
                "step": row["step"],
                "training_selection_tuple": row["training_selection_tuple"],
                "train_selection_score": row["train_selection_score"],
                "write_objective_accuracy": row["write_objective_accuracy"],
                "response_drift": row["response_drift"],
                "anchor_kl_diagnostic": row["anchor_kl_diagnostic"],
                "accepted": row["accepted"],
            }
            for row in result["history"]
        ]
        previous_bits = observed
        print(f"  {program['name']}: bits={observed} exact={ev['exact_accuracy']:.3f} heldout={ev['heldout_accuracy']:.3f}", flush=True)

    # Wrong-instruction control starts from the post-WRITE state and edits address 10 instead of 01.
    w.assign_flat(params, snapshots[0])
    wrong_bits = [0, 0, 0, 1]
    wrong_ids = [candidate_ids[b] for b in wrong_bits]
    write_program(w, model, tok, params, cfg, anchors, response_ids, response0, logits0,
                  exact, train, held, wrong_ids, candidate_ids, seed + 900001,
                  device, cfg["l5_operation_steps"])
    requested_after_overwrite = [candidate_ids[b] for b in programs[1]["targets"]]
    wrong_requested_eval = evaluate(
        w, model, tok, exact, held, requested_after_overwrite, candidate_ids, device)

    # Matched-norm random-kernel control from post-WRITE state.
    requested_delta_norm = float((snapshots[1] - snapshots[0]).norm().item())
    w.assign_flat(params, snapshots[0])
    r = w.response_vector(model, tok, anchors, response_ids, device)
    J = w.jacobian_rows(r, params)
    gen = torch.Generator(device=device).manual_seed(seed + 7000001)
    z = w.kernel_project(torch.randn(snapshots[0].shape, generator=gen, device=device), J)
    z = z * (requested_delta_norm / (z.norm() + 1e-12))
    w.assign_flat(params, snapshots[0] + z)
    w.retract(model, tok, params, anchors, response_ids, response0, device, cfg["retraction_steps"])
    random_requested_eval = evaluate(
        w, model, tok, exact, held, requested_after_overwrite, candidate_ids, device)

    # SWAP controls start from the post-OVERWRITE shared state.  The wrong
    # instruction swaps 00/10 rather than 00/11.  The random control matches
    # the norm of the requested SWAP update in the local response kernel.
    desired_swap_ids = [candidate_ids[b] for b in programs[2]["targets"]]
    w.assign_flat(params, snapshots[1])
    wrong_swap_bits = [1, 1, 0, 1]
    wrong_swap_ids = [candidate_ids[b] for b in wrong_swap_bits]
    write_program(w, model, tok, params, cfg, anchors, response_ids, response0,
                  logits0, exact, train, held, wrong_swap_ids, candidate_ids,
                  seed + 900101, device, cfg["l5_swap_steps"])
    wrong_swap_eval = evaluate(
        w, model, tok, exact, held, desired_swap_ids, candidate_ids, device)
    swap_delta_norm = float((snapshots[2] - snapshots[1]).norm().item())
    w.assign_flat(params, snapshots[1])
    r_swap = w.response_vector(model, tok, anchors, response_ids, device)
    J_swap = w.jacobian_rows(r_swap, params)
    gen_swap = torch.Generator(device=device).manual_seed(seed + 7000101)
    z_swap = w.kernel_project(
        torch.randn(snapshots[1].shape, generator=gen_swap, device=device), J_swap)
    z_swap = z_swap * (swap_delta_norm / (z_swap.norm() + 1e-12))
    w.assign_flat(params, snapshots[1] + z_swap)
    w.retract(model, tok, params, anchors, response_ids, response0, device,
              cfg["retraction_steps"])
    random_swap_eval = evaluate(
        w, model, tok, exact, held, desired_swap_ids, candidate_ids, device)

    # Transport and return the final shared memory state.
    final_state = snapshots[-1]
    w.assign_flat(params, final_state)
    move_sign = torch.tensor([1.0 if i % 2 == 0 else -1.0 for i in range(response0.numel())], device=device)
    target_response = response0 + cfg["l5_response_shift"] * move_sign / move_sign.norm()
    final_ids = [candidate_ids[b] for b in programs[-1]["targets"]]
    w.retract(model, tok, params, anchors, response_ids, target_response, device, cfg["l5_transport_retraction_steps"])
    transported_eval = evaluate(w, model, tok, exact, held, final_ids, candidate_ids, device)
    transported_state = w.param_flat(params)
    transport_residual = float((w.response_vector(model, tok, anchors, response_ids, device) - target_response).abs().max().item())
    w.retract(model, tok, params, anchors, response_ids, response0, device, cfg["l5_transport_retraction_steps"])
    returned_eval = evaluate(w, model, tok, exact, held, final_ids, candidate_ids, device)
    return_residual = float((w.response_vector(model, tok, anchors, response_ids, device) - response0).abs().max().item())

    gates = {
        "all_program_stages_exact": all(x["exact_accuracy"] == 1.0 for x in stage_records),
        "all_program_stages_heldout": all(x["heldout_accuracy"] == 1.0 for x in stage_records),
        "all_program_stages_positive_heldout_margin": all(
            x["minimum_heldout_margin"] >= cfg["l5_minimum_heldout_margin_gate"]
            for x in stage_records),
        "all_unselected_addresses_preserved": all(x["unselected_addresses_preserved"] for x in stage_records),
        "all_program_stages_within_response_budget": all(x["response_drift"] <= cfg["response_budget"] for x in stage_records),
        "all_program_stages_within_kl_budget": all(x["anchor_kl"] <= cfg["endpoint_kl_gate"] for x in stage_records),
        "requested_operation_beats_wrong_instruction": stage_records[1]["heldout_accuracy"] > wrong_requested_eval["heldout_accuracy"],
        "requested_operation_beats_matched_random": stage_records[1]["heldout_accuracy"] > random_requested_eval["heldout_accuracy"],
        "swap_beats_wrong_instruction": stage_records[2]["heldout_accuracy"] > wrong_swap_eval["heldout_accuracy"],
        "swap_beats_matched_random": stage_records[2]["heldout_accuracy"] > random_swap_eval["heldout_accuracy"],
        "transport_reaches_target_response": transport_residual <= cfg["response_budget"],
        "transport_preserves_all_addresses": transported_eval["exact_accuracy"] == 1.0 and transported_eval["heldout_accuracy"] == 1.0,
        "transport_preserves_positive_heldout_margin": transported_eval["minimum_heldout_margin"] >= cfg["l5_minimum_heldout_margin_gate"],
        "return_reaches_source_response": return_residual <= cfg["response_budget"],
        "return_preserves_all_addresses": returned_eval["exact_accuracy"] == 1.0 and returned_eval["heldout_accuracy"] == 1.0,
        "return_preserves_positive_heldout_margin": returned_eval["minimum_heldout_margin"] >= cfg["l5_minimum_heldout_margin_gate"],
    }
    record = {
        "protocol": cfg["protocol"],
        "scientific_status": cfg["scientific_status"],
        "claim_boundary": "Prospectively frozen fresh-seed and fresh-expression single-seed confirmation candidate for shared-state four-cell programmable address operations in one GPT-2 LoRA chart; not multi-seed confirmation, instruction-conditioned natural-language control, or a general instruction-set architecture.",
        "seed": seed, "addresses": cells,
        "symbols": {"0": sym0[0], "1": sym1[0]},
        "program": stage_records,
        "controls": {"wrong_instruction_requested_eval": wrong_requested_eval,
                     "matched_random_requested_eval": random_requested_eval,
                     "matched_update_norm": requested_delta_norm,
                     "wrong_swap_requested_eval": wrong_swap_eval,
                     "matched_random_swap_requested_eval": random_swap_eval,
                     "matched_swap_update_norm": swap_delta_norm},
        "transport": {"parameter_displacement": float((transported_state-final_state).norm().item()),
                      "target_response_residual": transport_residual,
                      "transported_eval": transported_eval,
                      "return_response_residual": return_residual,
                      "returned_eval": returned_eval},
        "gates": gates, "all_gates_pass": all(gates.values()),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "elapsed_seconds": time.time() - started,
        "python": platform.python_version(), "platform": platform.platform(),
        "torch": torch.__version__, "transformers": transformers.__version__,
    }
    (out / "l5_audit.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    (out / "SUMMARY.json").write_text(json.dumps({k: record[k] for k in ["protocol", "scientific_status", "claim_boundary", "seed", "gates", "all_gates_pass", "config_sha256", "elapsed_seconds"]}, indent=2), encoding="utf-8")
    print(json.dumps({"protocol": record["protocol"], "gates": gates, "all_gates_pass": record["all_gates_pass"], "elapsed_seconds": record["elapsed_seconds"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
