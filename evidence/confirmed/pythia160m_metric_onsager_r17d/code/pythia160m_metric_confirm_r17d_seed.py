#!/usr/bin/env python3
"""Per-seed worker for frozen R17d metric-Onsager confirmation.

The learning objective L (SST-2) and response map R (frozen AG News anchors)
remain disjoint.  Before running, the candidate is fixed as the exact solution
of

  min_delta <mhat, delta> + delta^T M delta/(2 eta)
  subject to DR(theta) delta = 0.

Here M=diag(sqrt(vhat)+eps) is frozen from the same Adam statistics in every
arm. This is a one-seed development audit, not confirmation and not a
universal variational law. The finite nonlinear response ball is enforced
identically for every arm by backtracking.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import statistics
import time
from pathlib import Path

PROTOCOL = "PYTHIA160M_SST2_AGNEWS_METRIC_ONSAGER_R17D_SEED_CONFIRMATORY"
BUDGET = 0.004543482202852718
ARMS = (
    "current_metric_m115",
    "current_projected_adamw",
    "source_frozen_metric_m100",
)
MULTIPLIERS = {
    "current_metric_m115": 1.15,
}


def load_engine():
    path = Path(__file__).with_name("r15d_engine.py")
    spec = importlib.util.spec_from_file_location("r15d_engine", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def adam_statistics(g, state, step, args, torch):
    state["m"] = args.beta1 * state["m"] + (1 - args.beta1) * g
    state["v"] = args.beta2 * state["v"] + (1 - args.beta2) * g.square()
    mh = state["m"] / (1 - args.beta1 ** (step + 1))
    vh = state["v"] / (1 - args.beta2 ** (step + 1))
    metric_diag = vh.sqrt() + args.adam_eps
    return mh, metric_diag


def metric_constrained_step(covector, metric_diag, J, step_norm, ridge_ratio, torch):
    """Exact diagonal-metric constrained Onsager minimizer, norm matched."""
    inv = metric_diag.reciprocal()
    gram = (J * inv.unsqueeze(0)) @ J.T
    ridge = ridge_ratio * max(float(gram.diag().mean()), 1e-30)
    eye = torch.eye(gram.shape[0], dtype=gram.dtype, device=gram.device)
    multiplier = torch.linalg.solve(gram + ridge * eye, J @ (inv * covector))
    raw = -(inv * (covector - J.T @ multiplier))
    eta = step_norm / max(float(raw.norm()), 1e-30)
    delta = eta * raw
    _, diag = null_basis_local(J, torch)
    stationarity = covector + metric_diag * delta / eta - J.T @ multiplier
    return delta, eta, {
        "constraint_residual": float((J @ delta).norm()),
        "tangent_kkt_residual": float((diag @ stationarity).norm()),
        "ridge": ridge,
    }


def null_basis_local(J, torch):
    _, s, vh = torch.linalg.svd(J.double(), full_matrices=True)
    tol = max(J.shape) * torch.finfo(torch.float64).eps * s.max()
    rank = int((s > tol).sum())
    N = vh[rank:].T.contiguous()
    return N, N.T


def run_arm(E, model, coordinate, warm, arm, seed, train, validation, schedule,
            response_rows, learning_ids, topic_ids, args, device, pad_id, torch, out):
    coordinate.data.copy_(warm)
    model.train()
    R0 = E.response_values(model, response_rows, topic_ids, args.response_batch_size,
                           device, pad_id, torch)
    J0 = E.response_jacobian(model, coordinate, response_rows, topic_ids,
                             args.response_batch_size, device, pad_id, torch)
    N0, d0 = E.null_basis(J0, torch)
    initial_loss, initial_acc = E.evaluate(model, validation, learning_ids,
                                           args.eval_batch_size, device, pad_id, torch)
    state = {"m": torch.zeros_like(coordinate, dtype=torch.float64),
             "v": torch.zeros_like(coordinate, dtype=torch.float64)}
    trace = []
    regrets, kkt_residuals, linear_leaks, drifts, backtracks = [], [], [], [], []
    ranks, max_leak, max_idem, zero_steps = [], 0., 0., 0
    E.sync(torch, device); started = time.perf_counter()
    for step in range(args.steps):
        batch = [train[i] for i in schedule[args.warm_steps + step]]
        loss = E.task_loss(model, batch, learning_ids, args.batch_size,
                           device, pad_id, torch)
        g = E.grad(loss, coordinate, torch).detach().double()
        J = E.response_jacobian(model, coordinate, response_rows, topic_ids,
                                args.response_batch_size, device, pad_id, torch)
        N, diag = E.null_basis(J, torch)
        P = N @ N.T
        mh, metric_diag = adam_statistics(g, state, step, args, torch)
        multiplier = MULTIPLIERS.get(arm, 1.0)
        exact, eta, metric_diag_record = metric_constrained_step(
            mh, metric_diag, J, args.step_norm * multiplier, args.metric_ridge, torch)
        if arm in MULTIPLIERS:
            proposed = exact
        elif arm == "source_frozen_metric_m100":
            proposed, _, _ = metric_constrained_step(
                mh, metric_diag, J0, args.step_norm, args.metric_ridge, torch)
        elif arm == "current_projected_adamw":
            direction = P @ (mh / metric_diag)
            proposed = -args.step_norm * E.normalized(direction)
        else:
            raise RuntimeError(arm)

        # Frozen current-kernel Onsager objective and KKT residual at proposal.
        q_exact = float(mh @ exact + (metric_diag * exact.square()).sum() / (2 * eta))
        q_proposed = float(mh @ proposed + (metric_diag * proposed.square()).sum() / (2 * eta))
        regret = q_proposed - q_exact
        kkt = float((N.T @ (mh + metric_diag * proposed / eta)).norm())
        linear_leak = float((J @ proposed).norm())

        old = coordinate.detach().clone()
        accepted, scale, used = False, 1., args.max_backtracks + 1
        for bt in range(args.max_backtracks + 1):
            coordinate.data.copy_(old)
            coordinate.data.add_((scale * proposed).to(coordinate.dtype))
            Rnew = E.response_values(model, response_rows, topic_ids,
                                     args.response_batch_size, device, pad_id, torch)
            drift = float((Rnew - R0).norm())
            if math.isfinite(drift) and drift <= BUDGET * (1 + 1e-8):
                accepted, used = True, bt
                break
            scale *= .5
        if not accepted:
            coordinate.data.copy_(old)
            drift, scale, zero_steps = drifts[-1] if drifts else 0., 0., zero_steps + 1
        regrets.append(regret)
        kkt_residuals.append(kkt)
        linear_leaks.append(linear_leak)
        drifts.append(drift)
        backtracks.append(used)
        ranks.append(diag["rank"])
        max_leak = max(max_leak, diag["leakage"])
        max_idem = max(max_idem, diag["idempotence"])
        if (step + 1) % args.eval_interval == 0 or step + 1 == args.steps:
            vl, va = E.evaluate(model, validation, learning_ids, args.eval_batch_size,
                                device, pad_id, torch)
            trace.append({"step": step + 1, "validation_loss": vl,
                          "validation_accuracy": va, "response_drift": drift,
                          "onsager_regret": regret, "kkt_residual": kkt})
            print(f"[{arm}] {step+1}/{args.steps} loss={vl:.6f} "
                  f"drift={drift:.3e} regret={regret:.3e}", flush=True)
    E.sync(torch, device)
    record = {
        "seed": seed,
        "arm": arm,
        "metric_step_multiplier": MULTIPLIERS.get(arm),
        "initial_validation_loss": initial_loss,
        "initial_validation_accuracy": initial_acc,
        "final_validation_loss": trace[-1]["validation_loss"],
        "final_validation_accuracy": trace[-1]["validation_accuracy"],
        "validation_loss_gain": initial_loss - trace[-1]["validation_loss"],
        "median_current_metric_onsager_regret": statistics.median(regrets),
        "maximum_negative_regret_numerical": max(0., -min(regrets)),
        "median_kkt_residual": statistics.median(kkt_residuals),
        "maximum_linearized_response_leak": max(linear_leaks),
        "maximum_global_response_drift": max(drifts),
        "median_backtracks": statistics.median(backtracks),
        "zero_step_fraction": zero_steps / args.steps,
        "response_ranks": sorted(set(ranks)),
        "maximum_projector_leakage": max_leak,
        "maximum_projector_idempotence": max_idem,
        "timed_seconds": time.perf_counter() - started,
        "trace": trace,
    }
    (out / f"{arm}.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--outdir", default="pythia_r17d_seed_results")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--steps", type=int, default=80)
    ap.add_argument("--warm-steps", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--response-batch-size", type=int, default=16)
    ap.add_argument("--eval-batch-size", type=int, default=16)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--chart-dim", type=int, default=32)
    ap.add_argument("--lora-rank", type=int, default=4)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--step-norm", type=float, default=.025)
    ap.add_argument("--warm-lr", type=float, default=.01)
    ap.add_argument("--beta1", type=float, default=.9)
    ap.add_argument("--beta2", type=float, default=.999)
    ap.add_argument("--adam-eps", type=float, default=1e-8)
    ap.add_argument("--metric-ridge", type=float, default=1e-10)
    ap.add_argument("--max-backtracks", type=int, default=12)
    ap.add_argument("--eval-interval", type=int, default=10)
    ap.add_argument("--train-examples", type=int, default=2048)
    ap.add_argument("--validation-examples", type=int, default=256)
    ap.add_argument("--response-per-class", type=int, default=4)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    E = load_engine()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if args.quick:
        args.steps, args.warm_steps, args.eval_interval = 4, 2, 2
        args.train_examples, args.validation_examples, args.response_per_class = 128, 32, 2
    device = torch.device(args.device)
    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    mode = "quick_nonclaim" if args.quick else "frozen_untouched_seed_worker"
    print(f"protocol={PROTOCOL} device={device} mode={mode}", flush=True)
    train_raw, val_raw, response_raw, data_record = E.load_data(
        args.seed, args.train_examples, args.validation_examples, args.response_per_class)
    tokenizer = AutoTokenizer.from_pretrained(E.MODEL_ID)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    learning_words, learning_ids = E.choose_single_token_verbalizers(
        tokenizer, ((" negative", " bad", " terrible"),
                    (" positive", " good", " great")))
    topic_words, topic_ids = E.choose_single_token_verbalizers(
        tokenizer, ((" World", " Global", " Politics"),
                    (" Sports", " Game", " Athletic"),
                    (" Business", " Market", " Finance"),
                    (" Technology", " Tech", " Science")))
    train = E.tokenize_sst2(train_raw, tokenizer, args.seq_len, torch)
    validation = E.tokenize_sst2(val_raw, tokenizer, args.seq_len, torch)
    response_rows = E.tokenize_agnews(response_raw, tokenizer, args.seq_len, torch)
    E.seed_all(args.seed, torch)
    model = AutoModelForCausalLM.from_pretrained(E.MODEL_ID).to(device)
    model.config.use_cache = False
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout): module.p = 0.
    coordinate, chosen_layers = E.inject_chart_lora(
        model, args.chart_dim, args.lora_rank, args.layers, torch)
    model.to(device)
    schedule = E.batch_schedule(len(train), args.warm_steps, args.steps,
                                args.batch_size, args.seed + 71)
    warm = E.warm_start(model, coordinate, train, schedule, args, learning_ids,
                        device, tokenizer.pad_token_id, torch)
    coordinate.data.copy_(warm)
    J = E.response_jacobian(model, coordinate, response_rows, topic_ids,
                            args.response_batch_size, device, tokenizer.pad_token_id, torch)
    _, preflight = E.null_basis(J, torch)
    if preflight["rank"] != 4:
        raise RuntimeError(f"expected response rank 4, got {preflight['rank']}")
    print(f"[preflight] rank={preflight['rank']} leakage={preflight['leakage']:.3e}", flush=True)
    records = [run_arm(E, model, coordinate, warm, arm, args.seed, train, validation, schedule,
                       response_rows, learning_ids, topic_ids, args, device,
                       tokenizer.pad_token_id, torch, out) for arm in ARMS]
    by = {r["arm"]: r for r in records}
    candidates = [by[a] for a in MULTIPLIERS]
    for record in candidates:
        record["response_budget_utilization"] = record["maximum_global_response_drift"] / BUDGET
    candidate = by["current_metric_m115"]
    projected = by["current_projected_adamw"]
    source = by["source_frozen_metric_m100"]
    margins = ({
        "current_projected_adamw": projected["final_validation_loss"] - candidate["final_validation_loss"],
        "source_frozen_metric_m100": source["final_validation_loss"] - candidate["final_validation_loss"],
    } if candidate else {})
    numerical = {
        "all_runs_finite": all(math.isfinite(r["final_validation_loss"]) for r in records),
        "all_response_balls_respected": all(r["maximum_global_response_drift"] <= BUDGET * (1 + 1e-8) for r in records),
        "float64_projector_leakage_at_most_1e_10": all(r["maximum_projector_leakage"] <= 1e-10 for r in records),
        "projector_idempotence_at_most_1e_10": all(r["maximum_projector_idempotence"] <= 1e-10 for r in records),
        "response_rank_constant": all(r["response_ranks"] == [4] for r in records),
        "frozen_multiplier_exact": candidate["metric_step_multiplier"] == 1.15,
        "all_current_metric_kkt_residuals_at_most_1e_8": all(r["median_kkt_residual"] <= 1e-8 for r in candidates),
        "all_current_metric_linearized_constraint_residuals_at_most_1e_8": all(r["maximum_linearized_response_leak"] <= 1e-8 for r in candidates),
        "all_current_metric_onsager_regrets_numerically_zero": all(abs(r["median_current_metric_onsager_regret"]) <= 1e-10 for r in candidates),
    }
    scientific = {
        "selected_multiplier_beats_source_frozen_metric": margins["source_frozen_metric_m100"] > 0,
        "selected_multiplier_beats_projected_adamw": margins["current_projected_adamw"] > 0,
        "selected_multiplier_positive_learning_gain": candidate["validation_loss_gain"] > 0,
    }
    supported = all(numerical.values()) and all(scientific.values()) and not args.quick
    summary = {
        "protocol": PROTOCOL, "mode": mode, "model": E.MODEL_ID,
        "pretrained": True, "seed": args.seed,
        "learning_target_L": "GLUE/SST-2 prompted binary sentiment loss",
        "response_map_R": "four AG News topic-margin coordinates on frozen disjoint inputs",
        "r_l_separation": "different datasets, prompts, labels, verbalizers and declared functionals",
        "development_reference": "R17C_BUDGET_MATCHED_METRIC_ONSAGER_CANDIDATE_SELECTED",
        "frozen_step_multiplier": 1.15,
        "frozen_incremental_principle": "min <mhat,delta>+delta^T M delta/(2 eta) subject to DR(theta)delta=0; M=diag(sqrt(vhat)+eps)",
        "analytic_solution": "delta*=-eta[M^-1-M^-1 J^T(J M^-1 J^T)^-1 J M^-1]mhat",
        "frozen_global_response_budget": BUDGET,
        "arms": list(ARMS), "data": data_record,
        "learning_verbalizers": learning_words, "response_verbalizers": topic_words,
        "chart": {"dimension": args.chart_dim, "lora_rank": args.lora_rank,
                  "layers": chosen_layers},
        "records": records,
        "selected_multiplier": candidate["metric_step_multiplier"],
        "selected_arm": candidate["arm"],
        "control_minus_selected_final_loss": margins,
        "numerical_gates": numerical, "scientific_gates": scientific,
        "scientific_status": ("R17D_SEED_SUPPORTS_FROZEN_METRIC_ONSAGER"
                              if supported else "R17D_SEED_DOES_NOT_SUPPORT_ALL_GATES"),
        "wall_seconds": time.time() - started,
        "claim_boundary": "One worker of a frozen five-seed R17d confirmation. It has no standalone confirmatory status.",
    }
    (out / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0 if (supported or args.quick) else 2


if __name__ == "__main__":
    raise SystemExit(main())
