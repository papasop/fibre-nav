#!/usr/bin/env python3
"""GPTW-PC1 prospective predictive response-fibre capacity audit."""
from __future__ import annotations
import argparse
import csv
import json
import math
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import gptw_lora_low_response_pareto_cpu as v

PROTOCOL = "GPTW_PC1_PREDICTIVE_RESPONSE_FIBRE_CAPACITY_CPU_V1"
SEEDS = list(range(87841, 87849))
AUDIT_STEPS = [0, 6, 12, 18]
HORIZON = 6
HARD_LIMIT = 7200
RIDGE_ALPHA = 1.0
CAPACITY_SCORE_TEMPLATES = [
    "In the frozen auxiliary wording, {code} designates",
    "For the held-aside capacity probe, {code} maps to",
]
FUTURE_EVAL_TEMPLATES = [
    "Under a future evaluation phrasing, the color for {code} is",
    "When tested after further training, {code} must indicate",
]
BASELINE_COLUMNS = [
    "step_fraction", "train_loss", "capacity_select_loss", "gradient_norm",
    "update_norm", "actual_full_response_cost",
]
EPS = 1e-12


def capacity_curve(model, adapter, theta, direction, anchors, response_ids,
                   select, select_y, score, score_y):
    base_select = v.loss_float(model, adapter, theta, select, select_y)
    base_score = v.loss_float(model, adapter, theta, score, score_y)
    rows = []
    for alpha in v.ALPHAS:
        q = theta + alpha * direction
        rows.append({
            "alpha": alpha,
            "response_cost": v.response_cost(model, adapter, theta, q, anchors, response_ids),
            "selection_utility": base_select - v.loss_float(model, adapter, q, select, select_y),
            "score_utility": base_score - v.loss_float(model, adapter, q, score, score_y),
        })
    return rows


def select_point(points, budget):
    feasible = [p for p in points if p["response_cost"] <= budget + EPS]
    return max(feasible, key=lambda p: (p["selection_utility"], -p["alpha"]))


def auc(values):
    return float(np.trapz(values, v.BUDGETS) / (v.BUDGETS[-1] - v.BUDGETS[0]))


def rankdata(values):
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and values[order[j]] == values[order[i]]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + j - 1) + 1.0
        i = j
    return ranks


def spearman(x, y):
    rx, ry = rankdata(x), rankdata(y)
    if np.std(rx) < EPS or np.std(ry) < EPS:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


def fit_ridge_predict(x_train, y_train, x_test):
    mean = x_train.mean(axis=0)
    scale = x_train.std(axis=0)
    scale[scale < EPS] = 1.0
    z_train = (x_train - mean) / scale
    z_test = (x_test - mean) / scale
    design = np.column_stack([np.ones(len(z_train)), z_train])
    test_design = np.column_stack([np.ones(len(z_test)), z_test])
    penalty = np.eye(design.shape[1]) * RIDGE_ALPHA
    penalty[0, 0] = 0.0
    beta = np.linalg.solve(design.T @ design + penalty, design.T @ y_train)
    return test_design @ beta


def loo_predictions(rows, extra_column=None):
    columns = list(BASELINE_COLUMNS)
    if extra_column:
        columns.append(extra_column)
    predictions = np.empty(len(rows), dtype=float)
    seeds = sorted({r["seed"] for r in rows})
    for seed in seeds:
        train_idx = [i for i, r in enumerate(rows) if r["seed"] != seed]
        test_idx = [i for i, r in enumerate(rows) if r["seed"] == seed]
        x_train = np.asarray([[rows[i][c] for c in columns] for i in train_idx], dtype=float)
        y_train = np.asarray([rows[i]["future_heldout_gain"] for i in train_idx], dtype=float)
        x_test = np.asarray([[rows[i][c] for c in columns] for i in test_idx], dtype=float)
        predictions[test_idx] = fit_ridge_predict(x_train, y_train, x_test)
    return predictions


def predictive_metrics(rows, predictions):
    y = np.asarray([r["future_heldout_gain"] for r in rows], dtype=float)
    residual = y - predictions
    sse = float(residual @ residual)
    denom = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - sse / max(denom, EPS)
    maes = {}
    for seed in sorted({r["seed"] for r in rows}):
        idx = [i for i, r in enumerate(rows) if r["seed"] == seed]
        maes[str(seed)] = float(np.mean(np.abs(residual[idx])))
    return {"loo_r2": r2, "mae": float(np.mean(np.abs(residual))),
            "spearman_prediction_target": spearman(predictions, y),
            "seed_mae": maes}


def run_seed(seed, model, tok, data, started):
    if time.time() - started > HARD_LIMIT:
        raise TimeoutError("Two-hour frozen hard limit exceeded")
    v.seed_all(seed)
    anchors, response_ids, train, train_y, cap_select, cap_select_y, cap_score, cap_score_y, future_eval, future_eval_y = data
    adapter = v.NativeLoraB(model, seed)
    theta = torch.nn.Parameter(torch.zeros(adapter.dimension, dtype=torch.float32))
    try:
        row0, _, _ = v.jacobian(model, adapter, theta, anchors, response_ids)
        optimizer = torch.optim.AdamW([theta], lr=2e-2, weight_decay=1e-4)
        states = {0: theta.detach().clone()}
        updates = {}
        state_metrics = {}
        future_losses = {0: v.loss_float(model, adapter, theta, future_eval, future_eval_y)}
        for step in range(v.ADAPT_STEPS):
            if time.time() - started > HARD_LIMIT:
                raise TimeoutError("Two-hour frozen hard limit exceeded")
            before = theta.detach().clone()
            loss = v.task_loss(model, adapter, theta, train, train_y)
            optimizer.zero_grad()
            loss.backward()
            gradient_norm = float(theta.grad.detach().norm())
            optimizer.step()
            delta = theta.detach() - before
            updates[step] = delta.detach().clone()
            if step in AUDIT_STEPS:
                states[step] = before
                state_metrics[step] = {
                    "train_loss": float(loss),
                    "gradient_norm": gradient_norm,
                    "update_norm": float(delta.norm()),
                }
            if step + 1 in [s + HORIZON for s in AUDIT_STEPS]:
                states[step + 1] = theta.detach().clone()
                future_losses[step + 1] = v.loss_float(model, adapter, theta, future_eval, future_eval_y)

        nodes = []
        for step in AUDIT_STEPS:
            th = states[step]
            delta = updates[step]
            row, rank, _ = v.jacobian(model, adapter, th, anchors, response_ids)
            norm = delta.norm()
            perm = v.permuted_row(row, seed * 1000 + step)
            directions = {
                "current": v.normalized(v.project_kernel(row, delta), norm),
                "source": v.normalized(v.project_kernel(row0, delta), norm),
                "permuted": v.normalized(v.project_kernel(perm, delta), norm),
                "actual": delta,
            }
            curves = {
                name: capacity_curve(model, adapter, th, direction, anchors, response_ids,
                                     cap_select, cap_select_y, cap_score, cap_score_y)
                for name, direction in directions.items()
            }
            actual_full = next(p for p in curves["actual"] if p["alpha"] == 1.0)
            budgets = [fraction * actual_full["response_cost"] for fraction in v.BUDGETS]
            capacities = {}
            selected_alphas = {}
            for name, points in curves.items():
                chosen = [select_point(points, budget) for budget in budgets]
                capacities[name] = auc([p["score_utility"] for p in chosen])
                selected_alphas[name] = [p["alpha"] for p in chosen]
            node = {
                "seed": seed,
                "step": step,
                "step_fraction": step / float(v.ADAPT_STEPS),
                **state_metrics[step],
                "capacity_select_loss": v.loss_float(model, adapter, th, cap_select, cap_select_y),
                "actual_full_response_cost": actual_full["response_cost"],
                "response_rank": rank,
                "row_space_rotation": v.rotation(row, row0),
                "current_capacity": capacities["current"],
                "source_capacity": capacities["source"],
                "permuted_capacity": capacities["permuted"],
                "actual_capacity": capacities["actual"],
                "future_heldout_gain": future_losses[step] - future_losses[step + HORIZON],
                "future_horizon": HORIZON,
                "selected_alphas": selected_alphas,
            }
            nodes.append(node)
        return {"seed": seed, "nodes": nodes}
    finally:
        adapter.active = None
        for handle in adapter.handles:
            handle.remove()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="gptw_pc1_predictive_capacity_results")
    args, unknown = parser.parse_known_args()
    if unknown:
        print("[notice] ignored notebook arguments:", unknown, flush=True)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    started = time.time()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    protocol_path = Path(__file__).with_name("protocol.json")
    frozen_protocol = json.loads(protocol_path.read_text())
    print(f"[preflight] {PROTOCOL} device=cpu seeds={SEEDS}", flush=True)
    tok = AutoTokenizer.from_pretrained(v.MODEL_ID)
    tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(v.MODEL_ID)
    model.eval()
    model.config.use_cache = False
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    data = (
        v.encode_prompts(tok, v.ANCHORS),
        v.target_ids(tok, v.RESPONSE_WORDS),
        *v.build_task(tok, v.TRAIN_TEMPLATES),
        *v.build_task(tok, v.CAL_TEMPLATES),
        *v.build_task(tok, CAPACITY_SCORE_TEMPLATES),
        *v.build_task(tok, FUTURE_EVAL_TEMPLATES),
    )
    records = []
    for index, seed in enumerate(SEEDS, 1):
        print(f"[seed {index}/8] {seed}", flush=True)
        record = run_seed(seed, model, tok, data, started)
        records.append(record)
        (out / f"seed_{seed}.json").write_text(json.dumps(record, indent=2))
        summary = {"seed": seed,
                   "median_capacity": float(np.median([n["current_capacity"] for n in record["nodes"]])),
                   "median_future_gain": float(np.median([n["future_heldout_gain"] for n in record["nodes"]]))}
        print(json.dumps(summary, indent=2), flush=True)

    rows = [node for record in records for node in record["nodes"]]
    baseline_pred = loo_predictions(rows)
    current_pred = loo_predictions(rows, "current_capacity")
    source_pred = loo_predictions(rows, "source_capacity")
    permuted_pred = loo_predictions(rows, "permuted_capacity")
    metrics = {
        "baseline": predictive_metrics(rows, baseline_pred),
        "baseline_plus_current": predictive_metrics(rows, current_pred),
        "baseline_plus_source": predictive_metrics(rows, source_pred),
        "baseline_plus_permuted": predictive_metrics(rows, permuted_pred),
    }
    delta_r2 = metrics["baseline_plus_current"]["loo_r2"] - metrics["baseline"]["loo_r2"]
    current_minus_source = metrics["baseline_plus_current"]["loo_r2"] - metrics["baseline_plus_source"]["loo_r2"]
    current_minus_permuted = metrics["baseline_plus_current"]["loo_r2"] - metrics["baseline_plus_permuted"]["loo_r2"]
    seed_mae_wins = sum(
        metrics["baseline_plus_current"]["seed_mae"][str(seed)]
        < metrics["baseline"]["seed_mae"][str(seed)]
        for seed in SEEDS
    )
    raw_spearman = spearman([r["current_capacity"] for r in rows],
                            [r["future_heldout_gain"] for r in rows])
    gates = {
        "complete_seeds": len(records) == 8,
        "complete_nodes": len(rows) == 32,
        "current_capacity_positive_spearman": raw_spearman >= 0.30,
        "incremental_loo_r2": delta_r2 >= 0.05,
        "seedwise_mae_improvement": seed_mae_wins >= 6,
        "current_beats_source_predictor": current_minus_source >= 0.02,
        "current_beats_permuted_predictor": current_minus_permuted >= 0.02,
    }
    decision = "PREDICTIVE_RESPONSE_FIBRE_CAPACITY_SUPPORTED" if all(gates.values()) else "PREDICTIVE_RESPONSE_FIBRE_CAPACITY_NOT_SUPPORTED"
    primary = {
        "raw_current_capacity_spearman": raw_spearman,
        "incremental_loo_r2": delta_r2,
        "seedwise_mae_improvement_count": seed_mae_wins,
        "current_minus_source_incremental_r2": current_minus_source,
        "current_minus_permuted_incremental_r2": current_minus_permuted,
    }
    report = {
        "protocol": PROTOCOL,
        "prospective": True,
        "decision": decision,
        "attempted_seeds": len(records),
        "prediction_nodes": len(rows),
        "primary": primary,
        "models": metrics,
        "gates": gates,
        "elapsed_seconds": time.time() - started,
        "claim_boundary": frozen_protocol["claim_boundary"],
    }
    (out / "report.json").write_text(json.dumps(report, indent=2))
    (out / "protocol.json").write_text(json.dumps(frozen_protocol, indent=2))
    scalar_columns = [k for k, val in rows[0].items() if not isinstance(val, dict)]
    with (out / "node_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=scalar_columns)
        writer.writeheader()
        writer.writerows([{k: row[k] for k in scalar_columns} for row in rows])
    with (out / "loo_predictions.csv").open("w", newline="") as handle:
        fields = ["seed", "step", "future_heldout_gain", "baseline_prediction",
                  "current_capacity_prediction", "source_capacity_prediction",
                  "permuted_capacity_prediction"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, row in enumerate(rows):
            writer.writerow({
                "seed": row["seed"], "step": row["step"],
                "future_heldout_gain": row["future_heldout_gain"],
                "baseline_prediction": baseline_pred[index],
                "current_capacity_prediction": current_pred[index],
                "source_capacity_prediction": source_pred[index],
                "permuted_capacity_prediction": permuted_pred[index],
            })
    print("=" * 88)
    print(json.dumps({"decision": decision, "primary": primary, "gates": gates}, indent=2), flush=True)
    print("RESULT DIRECTORY:", out, flush=True)


if __name__ == "__main__":
    main()
