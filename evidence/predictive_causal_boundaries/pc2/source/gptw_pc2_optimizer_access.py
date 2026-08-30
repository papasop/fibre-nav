#!/usr/bin/env python3
"""GPTW-PC2 prospective optimizer-access branch audit."""
from __future__ import annotations
import argparse, csv, json, shutil, time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import gptw_lora_low_response_pareto_cpu as v
import gptw_pc1_predictive_capacity as pc1

PROTOCOL = "GPTW_PC2_RESPONSE_FIBRE_CAPACITY_OPTIMIZER_ACCESS_CPU_V1"
SEEDS = list(range(88841, 88849))
AUDIT_STEPS = [6, 12, 18]
BRANCH_HORIZON = 4
HARD_LIMIT = 7200
EPS = 1e-12


def estimate_capacities(model, adapter, theta, proposal, row, row0, seed, step, data):
    anchors, response_ids, cap_select, cap_select_y, cap_score, cap_score_y = data
    norm = proposal.norm()
    perm = v.permuted_row(row, seed * 1000 + step)
    directions = {
        "current": v.normalized(v.project_kernel(row, proposal), norm),
        "source": v.normalized(v.project_kernel(row0, proposal), norm),
        "permuted": v.normalized(v.project_kernel(perm, proposal), norm),
        "actual": proposal,
    }
    curves = {
        name: pc1.capacity_curve(model, adapter, theta, direction, anchors, response_ids,
                                 cap_select, cap_select_y, cap_score, cap_score_y)
        for name, direction in directions.items()
    }
    actual_full = next(p for p in curves["actual"] if p["alpha"] == 1.0)
    budgets = [fraction * actual_full["response_cost"] for fraction in v.BUDGETS]
    capacities, alphas = {}, {}
    for name, points in curves.items():
        selected = [pc1.select_point(points, budget) for budget in budgets]
        capacities[name] = pc1.auc([p["score_utility"] for p in selected])
        alphas[name] = [p["alpha"] for p in selected]
    return capacities, alphas, actual_full["response_cost"]


def run_branch(mode, model, adapter, theta0, source_row, anchors, response_ids,
               train, train_y, future_eval, future_eval_y, seed, step, started):
    q = torch.nn.Parameter(theta0.detach().clone())
    optimizer = torch.optim.AdamW([q], lr=2e-2, weight_decay=1e-4)
    base_loss = v.loss_float(model, adapter, q, future_eval, future_eval_y)
    proposal_norms = []
    for branch_step in range(BRANCH_HORIZON):
        if time.time() - started > HARD_LIMIT:
            raise TimeoutError("Two-hour frozen hard limit exceeded")
        before = q.detach().clone()
        loss = v.task_loss(model, adapter, q, train, train_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        proposal = q.detach() - before
        proposal_norms.append(float(proposal.norm()))
        if mode != "actual":
            if mode == "source":
                use_row = source_row
            else:
                current_row, _, _ = v.jacobian(model, adapter, before, anchors, response_ids)
                if mode == "current":
                    use_row = current_row
                elif mode == "permuted":
                    use_row = v.permuted_row(current_row, seed * 100000 + step * 100 + branch_step)
                else:
                    raise RuntimeError(f"Unknown branch: {mode}")
            adjusted = v.normalized(v.project_kernel(use_row, proposal), proposal.norm())
            with torch.no_grad():
                q.copy_(before + adjusted)
    final_loss = v.loss_float(model, adapter, q, future_eval, future_eval_y)
    response_cost = v.response_cost(model, adapter, theta0, q.detach(), anchors, response_ids)
    return {
        "gain": base_loss - final_loss,
        "response_cost": response_cost,
        "mean_proposal_norm": float(np.mean(proposal_norms)),
    }


def run_seed(seed, model, tok, data, started):
    (anchors, response_ids, train, train_y, cap_select, cap_select_y,
     cap_score, cap_score_y, future_eval, future_eval_y) = data
    v.seed_all(seed)
    adapter = v.NativeLoraB(model, seed)
    theta = torch.nn.Parameter(torch.zeros(adapter.dimension, dtype=torch.float32))
    try:
        row0, _, _ = v.jacobian(model, adapter, theta, anchors, response_ids)
        optimizer = torch.optim.AdamW([theta], lr=2e-2, weight_decay=1e-4)
        snapshots = {}
        for step in range(v.ADAPT_STEPS):
            if time.time() - started > HARD_LIMIT:
                raise TimeoutError("Two-hour frozen hard limit exceeded")
            before = theta.detach().clone()
            loss = v.task_loss(model, adapter, theta, train, train_y)
            optimizer.zero_grad()
            loss.backward()
            gradient_norm = float(theta.grad.detach().norm())
            optimizer.step()
            proposal = theta.detach() - before
            if step in AUDIT_STEPS:
                snapshots[step] = {
                    "theta": before,
                    "proposal": proposal.detach().clone(),
                    "train_loss": float(loss.detach()),
                    "gradient_norm": gradient_norm,
                    "update_norm": float(proposal.norm()),
                }
        nodes = []
        cap_data = (anchors, response_ids, cap_select, cap_select_y, cap_score, cap_score_y)
        for step in AUDIT_STEPS:
            snap = snapshots[step]
            th, proposal = snap["theta"], snap["proposal"]
            row, rank, _ = v.jacobian(model, adapter, th, anchors, response_ids)
            capacities, selected_alphas, actual_response = estimate_capacities(
                model, adapter, th, proposal, row, row0, seed, step, cap_data)
            branches = {
                mode: run_branch(mode, model, adapter, th, row, anchors, response_ids,
                                 train, train_y, future_eval, future_eval_y, seed, step, started)
                for mode in ("actual", "current", "source", "permuted")
            }
            node = {
                "seed": seed,
                "step": step,
                "step_fraction": step / float(v.ADAPT_STEPS),
                "train_loss": snap["train_loss"],
                "gradient_norm": snap["gradient_norm"],
                "update_norm": snap["update_norm"],
                "capacity_select_loss": v.loss_float(model, adapter, th, cap_select, cap_select_y),
                "actual_full_response_cost": actual_response,
                "response_rank": rank,
                "row_space_rotation": v.rotation(row, row0),
                "current_capacity": capacities["current"],
                "source_capacity": capacities["source"],
                "permuted_capacity": capacities["permuted"],
                "actual_capacity": capacities["actual"],
                "actual_branch_gain": branches["actual"]["gain"],
                "current_branch_gain": branches["current"]["gain"],
                "source_branch_gain": branches["source"]["gain"],
                "permuted_branch_gain": branches["permuted"]["gain"],
                "actual_branch_response_cost": branches["actual"]["response_cost"],
                "current_branch_response_cost": branches["current"]["response_cost"],
                "source_branch_response_cost": branches["source"]["response_cost"],
                "permuted_branch_response_cost": branches["permuted"]["response_cost"],
                "selected_alphas": selected_alphas,
            }
            nodes.append(node)
        return {"seed": seed, "nodes": nodes}
    finally:
        adapter.active = None
        for handle in adapter.handles:
            handle.remove()


def analyze_target(rows, target, predictor):
    working = [dict(row, future_heldout_gain=row[target]) for row in rows]
    baseline = pc1.loo_predictions(working)
    augmented = pc1.loo_predictions(working, predictor)
    b = pc1.predictive_metrics(working, baseline)
    a = pc1.predictive_metrics(working, augmented)
    wins = sum(a["seed_mae"][str(seed)] < b["seed_mae"][str(seed)] for seed in SEEDS)
    return {
        "baseline": b,
        "augmented": a,
        "incremental_loo_r2": a["loo_r2"] - b["loo_r2"],
        "seedwise_mae_improvement_count": wins,
        "predictions": augmented,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="gptw_pc2_optimizer_access_results")
    args, unknown = parser.parse_known_args()
    if unknown:
        print("[notice] ignored notebook arguments:", unknown, flush=True)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    started = time.time()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    frozen_protocol = json.loads(Path(__file__).with_name("protocol.json").read_text())
    print(f"[preflight] {PROTOCOL} device=cpu seeds={SEEDS}", flush=True)
    tok = AutoTokenizer.from_pretrained(v.MODEL_ID)
    tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(v.MODEL_ID)
    model.eval(); model.config.use_cache = False
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    data = (
        v.encode_prompts(tok, v.ANCHORS), v.target_ids(tok, v.RESPONSE_WORDS),
        *v.build_task(tok, v.TRAIN_TEMPLATES), *v.build_task(tok, v.CAL_TEMPLATES),
        *v.build_task(tok, pc1.CAPACITY_SCORE_TEMPLATES),
        *v.build_task(tok, pc1.FUTURE_EVAL_TEMPLATES),
    )
    records = []
    for index, seed in enumerate(SEEDS, 1):
        print(f"[seed {index}/8] {seed}", flush=True)
        record = run_seed(seed, model, tok, data, started)
        records.append(record)
        (out / f"seed_{seed}.json").write_text(json.dumps(record, indent=2))
        print(json.dumps({"seed": seed,
                          "median_current_capacity": float(np.median([n["current_capacity"] for n in record["nodes"]])),
                          "median_current_branch_gain": float(np.median([n["current_branch_gain"] for n in record["nodes"]]))}, indent=2), flush=True)
    rows = [node for record in records for node in record["nodes"]]
    current_current = analyze_target(rows, "current_branch_gain", "current_capacity")
    actual_current = analyze_target(rows, "actual_branch_gain", "current_capacity")
    current_source = analyze_target(rows, "current_branch_gain", "source_capacity")
    current_permuted = analyze_target(rows, "current_branch_gain", "permuted_capacity")
    raw_spearman = pc1.spearman([r["current_capacity"] for r in rows], [r["current_branch_gain"] for r in rows])
    current_minus_actual = current_current["incremental_loo_r2"] - actual_current["incremental_loo_r2"]
    current_minus_source_predictor = current_current["augmented"]["loo_r2"] - current_source["augmented"]["loo_r2"]
    current_minus_permuted_predictor = current_current["augmented"]["loo_r2"] - current_permuted["augmented"]["loo_r2"]
    ratios = [r["current_branch_response_cost"] / max(r["actual_branch_response_cost"], EPS) for r in rows]
    median_response_ratio = float(np.median(ratios))
    primary = {
        "current_capacity_to_current_branch_spearman": raw_spearman,
        "current_branch_incremental_loo_r2": current_current["incremental_loo_r2"],
        "current_branch_seedwise_mae_improvement_count": current_current["seedwise_mae_improvement_count"],
        "current_branch_minus_unprojected_branch_incremental_r2": current_minus_actual,
        "current_predictor_minus_source_predictor_r2_on_current_branch": current_minus_source_predictor,
        "current_predictor_minus_permuted_predictor_r2_on_current_branch": current_minus_permuted_predictor,
        "median_current_to_unprojected_response_cost_ratio": median_response_ratio,
    }
    gates = {
        "complete_seeds": len(records) == 8,
        "complete_nodes": len(rows) == 24,
        "current_capacity_to_current_branch_spearman": raw_spearman >= 0.30,
        "current_branch_incremental_loo_r2": current_current["incremental_loo_r2"] >= 0.05,
        "current_branch_seedwise_mae_improvement": current_current["seedwise_mae_improvement_count"] >= 6,
        "mechanism_access_specificity": current_minus_actual >= 0.03,
        "current_predictor_beats_source": current_minus_source_predictor >= 0.02,
        "current_predictor_beats_permuted": current_minus_permuted_predictor >= 0.02,
        "current_branch_response_preservation": median_response_ratio <= 0.50,
    }
    decision = "RESPONSE_FIBRE_CAPACITY_PREDICTS_IMPOSED_CURRENT_KERNEL_ACCESS" if all(gates.values()) else "RESPONSE_FIBRE_CAPACITY_OPTIMIZER_ACCESS_NOT_SUPPORTED"
    report = {
        "protocol": PROTOCOL, "prospective": True, "decision": decision,
        "attempted_seeds": len(records), "prediction_nodes": len(rows),
        "primary": primary, "gates": gates,
        "models": {
            "current_capacity_on_current_branch": {k: val for k, val in current_current.items() if k != "predictions"},
            "current_capacity_on_unprojected_branch": {k: val for k, val in actual_current.items() if k != "predictions"},
            "source_capacity_on_current_branch": {k: val for k, val in current_source.items() if k != "predictions"},
            "permuted_capacity_on_current_branch": {k: val for k, val in current_permuted.items() if k != "predictions"},
        },
        "elapsed_seconds": time.time() - started,
        "claim_boundary": frozen_protocol["claim_boundary"],
    }
    (out / "report.json").write_text(json.dumps(report, indent=2))
    (out / "protocol.json").write_text(json.dumps(frozen_protocol, indent=2))
    scalar = [k for k, val in rows[0].items() if not isinstance(val, dict)]
    with (out / "node_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=scalar); writer.writeheader()
        writer.writerows([{k: row[k] for k in scalar} for row in rows])
    print("=" * 88)
    print(json.dumps({"decision": decision, "primary": primary, "gates": gates}, indent=2), flush=True)
    print("RESULT DIRECTORY:", out, flush=True)


if __name__ == "__main__":
    main()
