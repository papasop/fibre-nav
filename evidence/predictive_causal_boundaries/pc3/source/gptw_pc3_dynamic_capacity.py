#!/usr/bin/env python3
"""GPTW-PC3 prospective dynamic persistent-capacity prediction audit."""
from __future__ import annotations
import argparse, csv, json, time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import gptw_lora_low_response_pareto_cpu as v
import gptw_pc1_predictive_capacity as pc1

PROTOCOL = "GPTW_PC3_DYNAMIC_PERSISTENT_RESPONSE_FIBRE_CAPACITY_CPU_V1"
SEEDS = list(range(89841, 89857))
CAPACITY_STEPS = [6, 12, 18]
PREDICTION_STEPS = [12, 18]
LAG = 6
HORIZON = 6
HARD_LIMIT = 7200
EPS = 1e-12
BASELINE = list(pc1.BASELINE_COLUMNS)


def projector_distance(row_a, row_b):
    # For orthonormal response-row bases, ||Pa-Pb||_F^2 = ra+rb-2||Aa Bb^T||_F^2.
    overlap_sq = float(((row_a @ row_b.T) ** 2).sum())
    value = max(0.0, row_a.shape[0] + row_b.shape[0] - 2.0 * overlap_sq)
    return value ** 0.5


def estimate_capacity(model, adapter, theta, proposal, row, row0, seed, step, data):
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
    capacities = {}
    for name, points in curves.items():
        selected = [pc1.select_point(points, budget) for budget in budgets]
        capacities[name] = float(np.trapezoid(
            [p["score_utility"] for p in selected], v.BUDGETS
        ) / (v.BUDGETS[-1] - v.BUDGETS[0]))
    return capacities, actual_full["response_cost"]


def run_seed(seed, model, tok, data, started):
    (anchors, response_ids, train, train_y, cap_select, cap_select_y,
     cap_score, cap_score_y, future_eval, future_eval_y) = data
    v.seed_all(seed)
    adapter = v.NativeLoraB(model, seed)
    theta = torch.nn.Parameter(torch.zeros(adapter.dimension, dtype=torch.float32))
    try:
        row0, _, _ = v.jacobian(model, adapter, theta, anchors, response_ids)
        optimizer = torch.optim.AdamW([theta], lr=2e-2, weight_decay=1e-4)
        snapshots, future_losses = {}, {}
        cumulative_length = 0.0
        for step in range(v.ADAPT_STEPS):
            if time.time() - started > HARD_LIMIT:
                raise TimeoutError("Two-hour frozen hard limit exceeded")
            before = theta.detach().clone()
            if step in CAPACITY_STEPS or step in PREDICTION_STEPS:
                future_losses[step] = v.loss_float(model, adapter, before, future_eval, future_eval_y)
            loss = v.task_loss(model, adapter, theta, train, train_y)
            optimizer.zero_grad(); loss.backward()
            gradient_norm = float(theta.grad.detach().norm())
            optimizer.step()
            proposal = theta.detach() - before
            if step in CAPACITY_STEPS:
                snapshots[step] = {
                    "theta": before, "proposal": proposal.detach().clone(),
                    "train_loss": float(loss.detach()), "gradient_norm": gradient_norm,
                    "update_norm": float(proposal.norm()), "path_length": cumulative_length,
                }
            cumulative_length += float(proposal.norm())
            if step + 1 in [s + HORIZON for s in PREDICTION_STEPS]:
                future_losses[step + 1] = v.loss_float(model, adapter, theta, future_eval, future_eval_y)

        measurements = {}
        cap_data = (anchors, response_ids, cap_select, cap_select_y, cap_score, cap_score_y)
        for step in CAPACITY_STEPS:
            snap = snapshots[step]
            row, rank, _ = v.jacobian(model, adapter, snap["theta"], anchors, response_ids)
            capacities, actual_response = estimate_capacity(
                model, adapter, snap["theta"], snap["proposal"], row, row0, seed, step, cap_data)
            measurements[step] = {
                **snap, "row": row, "rank": rank, "capacities": capacities,
                "actual_full_response_cost": actual_response,
                "capacity_select_loss": v.loss_float(model, adapter, snap["theta"], cap_select, cap_select_y),
                "row_space_rotation": v.rotation(row, row0),
            }

        nodes = []
        for step in PREDICTION_STEPS:
            now, past = measurements[step], measurements[step - LAG]
            ds = max(now["path_length"] - past["path_length"], EPS)
            current_rate = (now["capacities"]["current"] - past["capacities"]["current"]) / ds
            source_rate = (now["capacities"]["source"] - past["capacities"]["source"]) / ds
            drift_rate = projector_distance(now["row"], past["row"]) / ds
            nodes.append({
                "seed": seed, "step": step, "step_fraction": step / float(v.ADAPT_STEPS),
                "train_loss": now["train_loss"], "gradient_norm": now["gradient_norm"],
                "update_norm": now["update_norm"], "capacity_select_loss": now["capacity_select_loss"],
                "actual_full_response_cost": now["actual_full_response_cost"],
                "response_rank": now["rank"], "row_space_rotation": now["row_space_rotation"],
                "history_path_length": ds,
                "current_capacity": now["capacities"]["current"],
                "current_capacity_rate": current_rate,
                "kernel_projector_drift_rate": drift_rate,
                "source_capacity": now["capacities"]["source"],
                "source_capacity_rate": source_rate,
                "source_kernel_drift_rate": 0.0,
                "permuted_capacity": now["capacities"]["permuted"],
                "future_heldout_gain": future_losses[step] - future_losses[step + HORIZON],
            })
        return {"seed": seed, "nodes": nodes}
    finally:
        adapter.active = None
        for handle in adapter.handles: handle.remove()


def loo_predict(rows, columns):
    predictions = np.empty(len(rows), dtype=float)
    for seed in SEEDS:
        train_idx = [i for i, row in enumerate(rows) if row["seed"] != seed]
        test_idx = [i for i, row in enumerate(rows) if row["seed"] == seed]
        x_train = np.asarray([[rows[i][c] for c in columns] for i in train_idx], dtype=float)
        y_train = np.asarray([rows[i]["future_heldout_gain"] for i in train_idx], dtype=float)
        x_test = np.asarray([[rows[i][c] for c in columns] for i in test_idx], dtype=float)
        predictions[test_idx] = pc1.fit_ridge_predict(x_train, y_train, x_test)
    return predictions


def model_metrics(rows, columns):
    predictions = loo_predict(rows, columns)
    metrics = pc1.predictive_metrics(rows, predictions)
    return predictions, metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="gptw_pc3_dynamic_capacity_results")
    args, unknown = parser.parse_known_args()
    if unknown: print("[notice] ignored notebook arguments:", unknown, flush=True)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    started = time.time(); out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    frozen_protocol = json.loads(Path(__file__).with_name("protocol.json").read_text())
    print(f"[preflight] {PROTOCOL} device=cpu seeds={SEEDS}", flush=True)
    tok = AutoTokenizer.from_pretrained(v.MODEL_ID); tok.pad_token = tok.eos_token; tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(v.MODEL_ID); model.eval(); model.config.use_cache = False
    for parameter in model.parameters(): parameter.requires_grad_(False)
    data = (
        v.encode_prompts(tok, v.ANCHORS), v.target_ids(tok, v.RESPONSE_WORDS),
        *v.build_task(tok, v.TRAIN_TEMPLATES), *v.build_task(tok, v.CAL_TEMPLATES),
        *v.build_task(tok, pc1.CAPACITY_SCORE_TEMPLATES), *v.build_task(tok, pc1.FUTURE_EVAL_TEMPLATES),
    )
    records = []
    for index, seed in enumerate(SEEDS, 1):
        print(f"[seed {index}/16] {seed}", flush=True)
        record = run_seed(seed, model, tok, data, started); records.append(record)
        (out / f"seed_{seed}.json").write_text(json.dumps(record, indent=2))
        print(json.dumps({"seed": seed,
                          "median_capacity": float(np.median([n["current_capacity"] for n in record["nodes"]])),
                          "median_capacity_rate": float(np.median([n["current_capacity_rate"] for n in record["nodes"]])),
                          "median_future_gain": float(np.median([n["future_heldout_gain"] for n in record["nodes"]]))}, indent=2), flush=True)
    rows = [node for record in records for node in record["nodes"]]
    # Same-step cyclic mismatch preserves marginal step distributions but breaks seed history.
    for step in PREDICTION_STEPS:
        group = [row for row in rows if row["step"] == step]
        group.sort(key=lambda row: row["seed"])
        for i, row in enumerate(group):
            donor = group[(i + 1) % len(group)]
            row["mismatched_capacity_rate"] = donor["current_capacity_rate"]
            row["mismatched_drift_rate"] = donor["kernel_projector_drift_rate"]

    baseline_cols = BASELINE
    static_cols = BASELINE + ["current_capacity"]
    dynamic_cols = BASELINE + ["current_capacity", "current_capacity_rate", "kernel_projector_drift_rate"]
    source_cols = BASELINE + ["source_capacity", "source_capacity_rate", "source_kernel_drift_rate"]
    mismatch_cols = BASELINE + ["current_capacity", "mismatched_capacity_rate", "mismatched_drift_rate"]
    predictions, metrics = {}, {}
    for name, columns in {
        "baseline": baseline_cols, "static_current": static_cols,
        "dynamic_current": dynamic_cols, "dynamic_source": source_cols,
        "mismatched_history": mismatch_cols,
    }.items():
        predictions[name], metrics[name] = model_metrics(rows, columns)
    delta_baseline = metrics["dynamic_current"]["loo_r2"] - metrics["baseline"]["loo_r2"]
    delta_static = metrics["dynamic_current"]["loo_r2"] - metrics["static_current"]["loo_r2"]
    delta_source = metrics["dynamic_current"]["loo_r2"] - metrics["dynamic_source"]["loo_r2"]
    delta_mismatch = metrics["dynamic_current"]["loo_r2"] - metrics["mismatched_history"]["loo_r2"]
    seed_wins = sum(metrics["dynamic_current"]["seed_mae"][str(seed)] < metrics["baseline"]["seed_mae"][str(seed)] for seed in SEEDS)
    prediction_spearman = metrics["dynamic_current"]["spearman_prediction_target"]
    primary = {
        "dynamic_prediction_spearman": prediction_spearman,
        "dynamic_incremental_loo_r2_over_baseline": delta_baseline,
        "dynamic_minus_static_loo_r2": delta_static,
        "seedwise_mae_improvement_count": seed_wins,
        "dynamic_current_minus_dynamic_source_loo_r2": delta_source,
        "dynamic_current_minus_mismatched_history_loo_r2": delta_mismatch,
    }
    gates = {
        "complete_seeds": len(records) == 16,
        "complete_prediction_nodes": len(rows) == 32,
        "dynamic_prediction_spearman": prediction_spearman >= 0.30,
        "dynamic_incremental_loo_r2": delta_baseline >= 0.05,
        "dynamic_beats_static": delta_static >= 0.03,
        "seedwise_mae_improvement": seed_wins >= 11,
        "dynamic_current_beats_dynamic_source": delta_source >= 0.02,
        "dynamic_current_beats_mismatched_history": delta_mismatch >= 0.02,
    }
    decision = "DYNAMIC_RESPONSE_FIBRE_CAPACITY_PREDICTIVE_STATE_SUPPORTED" if all(gates.values()) else "DYNAMIC_RESPONSE_FIBRE_CAPACITY_PREDICTIVE_STATE_NOT_SUPPORTED"
    report = {
        "protocol": PROTOCOL, "prospective": True, "decision": decision,
        "attempted_seeds": len(records), "prediction_nodes": len(rows),
        "primary": primary, "models": metrics, "gates": gates,
        "elapsed_seconds": time.time() - started,
        "claim_boundary": frozen_protocol["claim_boundary"],
    }
    (out / "report.json").write_text(json.dumps(report, indent=2))
    (out / "protocol.json").write_text(json.dumps(frozen_protocol, indent=2))
    fields = list(rows[0].keys())
    with (out / "node_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    with (out / "loo_predictions.csv").open("w", newline="") as handle:
        names = list(predictions)
        writer = csv.DictWriter(handle, fieldnames=["seed", "step", "future_heldout_gain"] + names); writer.writeheader()
        for i, row in enumerate(rows):
            writer.writerow({"seed": row["seed"], "step": row["step"], "future_heldout_gain": row["future_heldout_gain"],
                             **{name: predictions[name][i] for name in names}})
    print("=" * 88)
    print(json.dumps({"decision": decision, "primary": primary, "gates": gates}, indent=2), flush=True)
    print("RESULT DIRECTORY:", out, flush=True)


if __name__ == "__main__": main()
