#!/usr/bin/env python3
"""GPTW-PC4 prospective paired causal capacity intervention (CPU)."""
from __future__ import annotations
import argparse, csv, hashlib, json, math, random, shutil, time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import gptw_lora_low_response_pareto_cpu as v

PROTOCOL = "GPTW_PC4_PAIRED_CAUSAL_RESPONSE_FIBRE_CAPACITY_CPU_V1"
SEEDS = list(range(90841, 90849))
AUDIT_STEPS = [6, 18]
CANDIDATES = 12
BRANCH_HORIZON = 4
INTERVENTION_NORM_MULTIPLIER = 1.0
RESPONSE_BUDGET_FRACTION = 0.10
ALPHA_GRID = [0.0, 0.125, 0.25, 0.5, 0.75, 1.0]
HARD_LIMIT = 7200
EPS = 1e-12

SELECT_TEMPLATES = [
    "Under the declared mapping, {code} identifies",
    "Within this artificial vocabulary, {code} stands for",
]
BRANCH_TEMPLATES = [
    "For continued learning, the assigned color for {code} is",
    "During the next training phase, {code} denotes",
]
EVAL_TEMPLATES = [
    "In a completely held-out wording, {code} means",
    "After subsequent learning, the correct color for {code} is",
]


def loss(model, adapter, theta, batch, targets):
    return v.loss_float(model, adapter, theta, batch, targets)


def candidate_endpoint(model, adapter, theta, direction, alpha, anchors, response_ids,
                       select, select_y, base_select):
    q = theta + alpha * direction
    return {
        "alpha": alpha,
        "theta": q.detach().clone(),
        "response_cost": v.response_cost(model, adapter, theta, q, anchors, response_ids),
        "selection_utility": base_select - loss(model, adapter, q, select, select_y),
    }


def make_candidates(model, adapter, theta, row, update_norm, response_budget,
                    anchors, response_ids, select, select_y, seed, step):
    base_select = loss(model, adapter, theta, select, select_y)
    gen = torch.Generator().manual_seed(seed * 1000 + step + 417)
    candidates = []
    for index in range(CANDIDATES):
        raw = torch.randn(theta.numel(), generator=gen, dtype=theta.dtype)
        direction = v.normalized(v.project_kernel(row, raw), update_norm * INTERVENTION_NORM_MULTIPLIER)
        points = [candidate_endpoint(model, adapter, theta, direction, alpha, anchors,
                                     response_ids, select, select_y, base_select)
                  for alpha in ALPHA_GRID]
        feasible = [p for p in points if p["response_cost"] <= response_budget + EPS]
        # Use the largest feasible displacement. Capacity is compared across
        # directions only after a second, explicit response-cost matching step.
        chosen = max(feasible, key=lambda p: (p["alpha"], p["response_cost"]))
        chosen["candidate"] = index
        chosen["direction_norm"] = float(direction.norm())
        candidates.append(chosen)
    return candidates


def branch(model, adapter, root, batch, targets, eval_batch, eval_targets, started):
    q = torch.nn.Parameter(root.detach().clone())
    optimizer = torch.optim.AdamW([q], lr=2e-2, weight_decay=1e-4)
    initial = loss(model, adapter, q, eval_batch, eval_targets)
    for _ in range(BRANCH_HORIZON):
        if time.time() - started > HARD_LIMIT:
            raise TimeoutError("Frozen two-hour CPU hard limit exceeded")
        objective = v.task_loss(model, adapter, q, batch, targets)
        optimizer.zero_grad(); objective.backward(); optimizer.step()
    final = loss(model, adapter, q, eval_batch, eval_targets)
    return {"initial_eval_loss": initial, "final_eval_loss": final, "gain": initial - final}


def exact_sign_flip_p(values):
    x = np.asarray(values, dtype=float)
    observed = abs(float(x.mean()))
    count = 0
    for mask in range(1 << len(x)):
        signs = np.asarray([1.0 if mask & (1 << i) else -1.0 for i in range(len(x))])
        count += abs(float((x * signs).mean())) >= observed - 1e-15
    return count / float(1 << len(x))


def bootstrap_ci(values, seed=20260830, draws=10000):
    x = np.asarray(values, dtype=float); rng = np.random.default_rng(seed)
    means = np.asarray([rng.choice(x, len(x), replace=True).mean() for _ in range(draws)])
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def run_seed(seed, model, data, started):
    anchors, response_ids, train, train_y, select, select_y, branch_x, branch_y, eva, eva_y = data
    v.seed_all(seed)
    adapter = v.NativeLoraB(model, seed)
    theta = torch.nn.Parameter(torch.zeros(adapter.dimension, dtype=torch.float32))
    try:
        optimizer = torch.optim.AdamW([theta], lr=2e-2, weight_decay=1e-4)
        snapshots = {}
        for step in range(v.ADAPT_STEPS):
            if time.time() - started > HARD_LIMIT:
                raise TimeoutError("Frozen two-hour CPU hard limit exceeded")
            before = theta.detach().clone()
            objective = v.task_loss(model, adapter, theta, train, train_y)
            optimizer.zero_grad(); objective.backward(); optimizer.step()
            update = theta.detach() - before
            if step in AUDIT_STEPS:
                snapshots[step] = (before, update.detach().clone(), float(objective.detach()))

        nodes = []
        for step in AUDIT_STEPS:
            root, update, train_loss = snapshots[step]
            row, rank, _ = v.jacobian(model, adapter, root, anchors, response_ids)
            actual_cost = v.response_cost(model, adapter, root, root + update, anchors, response_ids)
            budget = RESPONSE_BUDGET_FRACTION * actual_cost
            candidates = make_candidates(model, adapter, root, row, update.norm(), budget,
                                         anchors, response_ids, select, select_y, seed, step)
            pairs = [(a, b) for i, a in enumerate(candidates) for b in candidates[i + 1:]
                     if abs(a["response_cost"] - b["response_cost"]) <= 0.25 * budget + EPS]
            if not pairs:
                pairs = [(a, b) for i, a in enumerate(candidates) for b in candidates[i + 1:]]
                minimum_gap = min(abs(a["response_cost"] - b["response_cost"]) for a, b in pairs)
                pairs = [(a, b) for a, b in pairs
                         if abs(abs(a["response_cost"] - b["response_cost"]) - minimum_gap) <= EPS]
            a, b = max(pairs, key=lambda pair: abs(pair[0]["selection_utility"] - pair[1]["selection_utility"]))
            low, high = sorted((a, b), key=lambda p: p["selection_utility"])
            rng = random.Random(seed * 10000 + step + 91)
            random_a, random_b = rng.choice(pairs)
            arms = {"high": high, "low": low, "random_a": random_a, "random_b": random_b}
            outcomes = {name: branch(model, adapter, arm["theta"], branch_x, branch_y,
                                     eva, eva_y, started) for name, arm in arms.items()}
            node = {
                "seed": seed, "step": step, "train_loss": train_loss,
                "response_rank": rank, "update_norm": float(update.norm()),
                "actual_update_response_cost": actual_cost, "response_budget": budget,
                "high_candidate": high["candidate"], "low_candidate": low["candidate"],
                "high_selection_utility": high["selection_utility"],
                "low_selection_utility": low["selection_utility"],
                "selection_contrast": high["selection_utility"] - low["selection_utility"],
                "high_response_cost": high["response_cost"], "low_response_cost": low["response_cost"],
                "response_cost_gap": abs(high["response_cost"] - low["response_cost"]),
                "high_gain": outcomes["high"]["gain"], "low_gain": outcomes["low"]["gain"],
                "causal_gain_contrast": outcomes["high"]["gain"] - outcomes["low"]["gain"],
                "high_final_eval_loss": outcomes["high"]["final_eval_loss"],
                "low_final_eval_loss": outcomes["low"]["final_eval_loss"],
                "random_a_gain": outcomes["random_a"]["gain"],
                "random_b_gain": outcomes["random_b"]["gain"],
                "random_control_contrast": outcomes["random_a"]["gain"] - outcomes["random_b"]["gain"],
            }
            nodes.append(node)
        return {"seed": seed, "nodes": nodes}
    finally:
        adapter.active = None
        for handle in adapter.handles: handle.remove()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="gptw_pc4_causal_capacity_results")
    parser.add_argument("--smoke", action="store_true")
    args, unknown = parser.parse_known_args()
    if unknown: print("[notice] ignored notebook arguments:", unknown, flush=True)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    started = time.time(); out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    protocol = json.loads(Path(__file__).with_name("protocol.json").read_text())
    seeds = SEEDS[:1] if args.smoke else SEEDS
    print(f"[preflight] {PROTOCOL} device=cpu seeds={seeds}", flush=True)
    tok = AutoTokenizer.from_pretrained(v.MODEL_ID); tok.pad_token = tok.eos_token; tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(v.MODEL_ID); model.eval(); model.config.use_cache = False
    for parameter in model.parameters(): parameter.requires_grad_(False)
    data = (v.encode_prompts(tok, v.ANCHORS), v.target_ids(tok, v.RESPONSE_WORDS),
            *v.build_task(tok, v.TRAIN_TEMPLATES), *v.build_task(tok, SELECT_TEMPLATES),
            *v.build_task(tok, BRANCH_TEMPLATES), *v.build_task(tok, EVAL_TEMPLATES))
    records = []
    for index, seed in enumerate(seeds, 1):
        print(f"[seed {index}/{len(seeds)}] {seed}", flush=True)
        record = run_seed(seed, model, data, started); records.append(record)
        (out / f"seed_{seed}.json").write_text(json.dumps(record, indent=2))
        print(json.dumps({"seed": seed, "median_causal_contrast": float(np.median(
            [n["causal_gain_contrast"] for n in record["nodes"]]))}, indent=2), flush=True)
    rows = [n for r in records for n in r["nodes"]]
    seed_contrasts = [float(np.mean([n["causal_gain_contrast"] for n in r["nodes"]])) for r in records]
    seed_controls = [float(np.mean([n["random_control_contrast"] for n in r["nodes"]])) for r in records]
    ci = bootstrap_ci(seed_contrasts) if len(seed_contrasts) > 1 else [seed_contrasts[0]] * 2
    wins = sum(x > 0 for x in seed_contrasts)
    median_gap_ratio = float(np.median([n["response_cost_gap"] / max(n["response_budget"], EPS) for n in rows]))
    primary = {
        "positive_seed_count": wins,
        "mean_paired_causal_gain_contrast": float(np.mean(seed_contrasts)),
        "median_paired_causal_gain_contrast": float(np.median(seed_contrasts)),
        "paired_bootstrap_mean_95ci": ci,
        "exact_two_sided_sign_flip_p": exact_sign_flip_p(seed_contrasts),
        "mean_absolute_random_control_contrast": float(np.mean(np.abs(seed_controls))),
        "median_response_cost_gap_over_budget": median_gap_ratio,
        "positive_selection_contrast_nodes": sum(n["selection_contrast"] > 0 for n in rows),
    }
    gates = {
        "complete_seeds": len(records) == 8,
        "complete_nodes": len(rows) == 16,
        "selection_non_degenerate": primary["positive_selection_contrast_nodes"] == len(rows),
        "response_matched": median_gap_ratio <= 0.25,
        "at_least_6_of_8_seed_blocks_positive": wins >= 6,
        "mean_causal_contrast_positive": primary["mean_paired_causal_gain_contrast"] > 0,
        "bootstrap_ci_lower_positive": ci[0] > 0,
        "effect_exceeds_random_pairing": abs(primary["mean_paired_causal_gain_contrast"]) > primary["mean_absolute_random_control_contrast"],
    }
    supported = (not args.smoke) and all(gates.values())
    decision = "CAUSAL_RESPONSE_FIBRE_CAPACITY_SUPPORTED" if supported else (
        "SMOKE_TEST_ONLY" if args.smoke else "CAUSAL_RESPONSE_FIBRE_CAPACITY_NOT_SUPPORTED")
    report = {"protocol": PROTOCOL, "prospective": True, "decision": decision,
              "attempted_seeds": len(records), "nodes": len(rows), "primary": primary,
              "gates": gates, "elapsed_seconds": time.time() - started,
              "claim_boundary": protocol["claim_boundary"]}
    (out / "report.json").write_text(json.dumps(report, indent=2))
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2))
    with (out / "node_metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    archive = shutil.make_archive(str(out), "zip", out.parent, out.name)
    print("=" * 88); print(json.dumps({"decision": decision, "primary": primary, "gates": gates}, indent=2))
    print("RESULT ZIP:", archive); print("SHA-256:", hashlib.sha256(Path(archive).read_bytes()).hexdigest())


if __name__ == "__main__": main()
