#!/usr/bin/env python3
"""R15d: tangent variation with fixed-target normal recovery.

This is a development experiment, never a confirmation.  The learning target
L is GLUE/SST-2 sentiment classification.  The independently declared response
R is a four-coordinate topic-logit response on frozen AG News inputs.  The
The candidate follows the current response-kernel tangent and then applies a
damped pseudoinverse normal feedback step.  A frozen contraction gate,
residual-decrease line search, local tolerance, and normal trust region are all
mandatory.  This is inspired by normally attracting/K=1 transverse recovery;
it is not a K=1 bridge theorem.

All arms share the same warm start, batches, chart, step norm (where
meaningful), nominated response balls and backtracking implementation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import statistics
import time
from pathlib import Path

PROTOCOL = "PYTHIA160M_SST2_AGNEWS_FIXED_TARGET_NORMAL_RECOVERY_R15D_DEVELOPMENT"
MODEL_ID = "EleutherAI/pythia-160m"
LEARNING_DATASET_ID = "glue/sst2"
RESPONSE_DATASET_ID = "ag_news"
SEED = 41217
FROZEN_BUDGETS = (0.002271741101426359, 0.004543482202852718)
ARMS = ("normal_recovery_momentum", "naive_momentum", "covariant_momentum",
        "source_momentum", "current_adam", "budgeted_adamw")
CONTROLS = ARMS[1:]


def seed_all(seed, torch):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sync(torch, device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def stable_sha(obj):
    raw = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def load_data(seed, train_n, val_n, response_per_class):
    from datasets import load_dataset

    sst = load_dataset("glue", "sst2")
    train = [{"text": str(x["sentence"]), "label": int(x["label"])} for x in sst["train"]]
    val = [{"text": str(x["sentence"]), "label": int(x["label"])} for x in sst["validation"]]
    rng = random.Random(seed)
    rng.shuffle(train)
    rng.shuffle(val)
    train, val = train[:train_n], val[:val_n]
    ag = load_dataset("ag_news")
    pool = [{"text": str(x["text"]), "label": int(x["label"])} for x in ag["test"]]
    random.Random(seed + 991).shuffle(pool)
    response = []
    for label in range(4):
        response.extend([x for x in pool if x["label"] == label][:response_per_class])
    if len(train) < train_n or len(val) < val_n:
        raise RuntimeError("insufficient deterministic SST-2 subset")
    if len(response) != 4 * response_per_class:
        raise RuntimeError("insufficient deterministic AG News response subset")
    return train, val, response, {
        "learning_dataset": LEARNING_DATASET_ID,
        "response_dataset": RESPONSE_DATASET_ID,
        "learning_train_sha256": stable_sha(train),
        "learning_validation_sha256": stable_sha(val),
        "response_anchors_sha256": stable_sha(response),
        "learning_train_examples": len(train),
        "learning_validation_examples": len(val),
        "response_anchor_examples": len(response),
        "response_anchor_examples_per_topic": response_per_class,
        "r_l_datasets_distinct": True,
    }


def tokenize_sst2(examples, tokenizer, seq_len, torch):
    rows = []
    prefix = "Review: "
    suffix = "\nSentiment:"
    for ex in examples:
        enc = tokenizer(prefix + ex["text"] + suffix, add_special_tokens=False,
                        truncation=True, max_length=seq_len)
        ids = enc["input_ids"]
        if not ids:
            raise RuntimeError("empty tokenized example")
        rows.append((torch.tensor(ids, dtype=torch.long), int(ex["label"])))
    return rows


def tokenize_agnews(examples, tokenizer, seq_len, torch):
    rows = []
    for ex in examples:
        enc = tokenizer("Article: " + ex["text"] + "\nTopic:", add_special_tokens=False,
                        truncation=True, max_length=seq_len)
        if not enc["input_ids"]:
            raise RuntimeError("empty tokenized AG News response example")
        rows.append((torch.tensor(enc["input_ids"], dtype=torch.long), int(ex["label"])))
    return rows


def choose_single_token_verbalizers(tokenizer, candidate_groups):
    words, ids = [], []
    for candidates in candidate_groups:
        chosen = None
        for word in candidates:
            got = tokenizer.encode(word, add_special_tokens=False)
            if len(got) == 1:
                chosen = (word, got[0])
                break
        if chosen is None:
            raise RuntimeError(f"no one-token verbalizer among {candidates}")
        words.append(chosen[0]); ids.append(chosen[1])
    if len(set(ids)) != len(ids):
        raise RuntimeError("verbalizer collision")
    return tuple(words), tuple(ids)


def collate(rows, device, pad_id, torch):
    max_len = max(len(x[0]) for x in rows)
    input_ids = torch.full((len(rows), max_len), pad_id, dtype=torch.long, device=device)
    attention = torch.zeros((len(rows), max_len), dtype=torch.long, device=device)
    labels = torch.tensor([x[1] for x in rows], dtype=torch.long, device=device)
    last = []
    for i, (ids, _) in enumerate(rows):
        n = len(ids)
        input_ids[i, :n] = ids.to(device)
        attention[i, :n] = 1
        last.append(n - 1)
    return input_ids, attention, labels, torch.tensor(last, device=device)


def inject_chart_lora(model, chart_dim, rank, layers, torch):
    import torch.nn as nn

    class ChartLoRALinear(nn.Module):
        def __init__(self, base, coordinate, generator):
            super().__init__()
            self.in_features = base.in_features
            self.out_features = base.out_features
            self.rank = rank
            self.scale = 1.0 / rank
            self.weight = nn.Parameter(base.weight.detach().clone(), requires_grad=False)
            self.bias = None if base.bias is None else nn.Parameter(base.bias.detach().clone(), requires_grad=False)
            count = rank * self.in_features + self.out_features * rank
            q, _ = torch.linalg.qr(torch.randn(count, chart_dim, generator=generator,
                                               dtype=torch.float64), mode="reduced")
            self.register_buffer("basis", q.float())
            base_vec = torch.zeros(count)
            base_vec[:rank * self.in_features] = .02 * torch.randn(rank * self.in_features,
                                                                   generator=generator)
            self.register_buffer("base_vec", base_vec)
            self.coordinate = coordinate

        def forward(self, x):
            v = self.base_vec + self.basis @ self.coordinate
            k = self.rank * self.in_features
            A = v[:k].view(self.rank, self.in_features)
            B = v[k:].view(self.out_features, self.rank)
            return torch.nn.functional.linear(x, self.weight, self.bias) + ((x @ A.T) @ B.T) * self.scale

    for p in model.parameters():
        p.requires_grad_(False)
    coordinate = nn.Parameter(torch.zeros(chart_dim))
    model.register_parameter("intrinsic_lora_coordinate", coordinate)
    generator = torch.Generator(device="cpu").manual_seed(140317)
    blocks = model.gpt_neox.layers
    chosen = list(range(len(blocks) - layers, len(blocks)))
    for i in chosen:
        old = blocks[i].attention.query_key_value
        if not isinstance(old, nn.Linear):
            raise RuntimeError(f"unexpected query_key_value type: {type(old)}")
        blocks[i].attention.query_key_value = ChartLoRALinear(old, coordinate, generator)
    return coordinate, chosen


def class_logits(model, rows, verbalizers, batch_size, device, pad_id, torch, require_grad):
    chunks = []
    context = torch.enable_grad() if require_grad else torch.no_grad()
    with context:
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            ids, mask, _, last = collate(batch, device, pad_id, torch)
            logits = model(input_ids=ids, attention_mask=mask).logits
            chosen = logits[torch.arange(len(batch), device=device), last]
            chunks.append(chosen[:, list(verbalizers)])
    return torch.cat(chunks, dim=0)


def task_loss(model, rows, verbalizers, batch_size, device, pad_id, torch):
    logits = class_logits(model, rows, verbalizers, batch_size, device, pad_id, torch, True)
    labels = torch.tensor([r[1] for r in rows], device=device)
    return torch.nn.functional.cross_entropy(logits, labels)


def per_example_task_gradients(model, coordinate, rows, verbalizers, device,
                               pad_id, torch):
    """Exact chart gradients used to form a small empirical Fisher."""
    grads = []
    for row in rows:
        loss = task_loss(model, [row], verbalizers, 1, device, pad_id, torch)
        grads.append(torch.autograd.grad(loss, coordinate)[0].detach().double())
    return torch.stack(grads)


def fisher_power_proposal(g, sample_grads, N, alpha, damping_ratio, torch):
    """Return N(F_N+lambda I)^(-alpha)N^T g and diagnostics."""
    gn = N.T @ g
    projected_samples = sample_grads @ N
    fisher = projected_samples.T @ projected_samples / len(projected_samples)
    fisher = 0.5 * (fisher + fisher.T)
    eig, vec = torch.linalg.eigh(fisher)
    eig = eig.clamp_min(0.)
    scale = float(eig.mean().clamp_min(1e-30))
    damping = 0.0 if alpha == 0 else max(damping_ratio * scale, 1e-12)
    weights = torch.ones_like(eig) if alpha == 0 else (eig + damping).pow(-alpha)
    direction = N @ (vec @ (weights * (vec.T @ gn)))
    condition = float((eig.max() + damping) / (eig.min() + damping)) if damping else 1.0
    return direction, {
        "fisher_trace": float(eig.sum()),
        "fisher_min_eigenvalue": float(eig.min()),
        "fisher_max_eigenvalue": float(eig.max()),
        "effective_damping": damping,
        "regularized_condition_number": condition,
    }


def grad(loss, coordinate, torch, retain_graph=False):
    return torch.autograd.grad(loss, coordinate, retain_graph=retain_graph)[0]


def response_values(model, response_rows, topic_verbalizers, batch_size, device, pad_id,
                    torch, detach=True):
    logits = class_logits(model, response_rows, topic_verbalizers, batch_size,
                          device, pad_id, torch, not detach)
    labels = torch.tensor([r[1] for r in response_rows], device=device)
    coordinates = []
    for topic in range(4):
        own = logits[:, topic]
        mask = torch.ones(4, dtype=torch.bool, device=device); mask[topic] = False
        margin = own - torch.logsumexp(logits[:, mask], dim=1)
        coordinates.append(margin[labels == topic].mean())
    vals = torch.stack(coordinates)
    return vals.detach().double() if detach else vals


def response_jacobian(model, coordinate, response_rows, topic_verbalizers, batch_size,
                      device, pad_id, torch):
    vals = response_values(model, response_rows, topic_verbalizers, batch_size,
                           device, pad_id, torch, detach=False)
    rows = []
    for i in range(len(vals)):
        rows.append(grad(vals[i], coordinate, torch,
                         retain_graph=i + 1 < len(vals)).detach().double())
    return torch.stack(rows)


def null_basis(J, torch):
    _, s, vh = torch.linalg.svd(J.double(), full_matrices=True)
    tol = max(J.shape) * torch.finfo(torch.float64).eps * (s.max() if s.numel() else 1.)
    rank = int((s > tol).sum())
    N = vh[rank:].T.contiguous()
    if N.shape[1] < 1:
        raise RuntimeError("empty numerical response kernel")
    P = N @ N.T
    diag = {
        "rank": rank,
        "leakage": float((J @ P).norm() / J.norm().clamp_min(1e-30)),
        "idempotence": float((P @ P - P).norm() / P.norm().clamp_min(1e-30)),
    }
    return N, diag


def normalized(v):
    return v / v.norm().clamp_min(1e-30)


def batch_schedule(n, warm_steps, steps, batch_size, seed):
    rng = random.Random(seed)
    return [[rng.randrange(n) for _ in range(batch_size)] for _ in range(warm_steps + steps)]


def evaluate(model, rows, verbalizers, batch_size, device, pad_id, torch):
    model.eval()
    logits = class_logits(model, rows, verbalizers, batch_size, device, pad_id, torch, False)
    labels = torch.tensor([r[1] for r in rows], device=device)
    loss = float(torch.nn.functional.cross_entropy(logits, labels))
    acc = float((logits.argmax(dim=1) == labels).float().mean())
    model.train()
    return loss, acc


def warm_start(model, coordinate, train, schedule, args, verbalizers, device, pad_id, torch):
    opt = torch.optim.AdamW([coordinate], lr=args.warm_lr, weight_decay=0.)
    for k in range(args.warm_steps):
        rows = [train[i] for i in schedule[k]]
        opt.zero_grad(set_to_none=True)
        task_loss(model, rows, verbalizers, args.batch_size, device, pad_id, torch).backward()
        opt.step()
    return coordinate.detach().clone()


def adam_proposal(g, state, step, args, torch):
    state["m"] = args.beta1 * state["m"] + (1 - args.beta1) * g
    state["v"] = args.beta2 * state["v"] + (1 - args.beta2) * g.square()
    mh = state["m"] / (1 - args.beta1 ** (step + 1))
    vh = state["v"] / (1 - args.beta2 ** (step + 1))
    return -mh / (vh.sqrt() + args.adamw_eps)


def optimizer_direction(g, state, step, kind, args, torch):
    """Matched optimizer decomposition before response-kernel projection."""
    state["m"] = args.beta1 * state["m"] + (1 - args.beta1) * g
    state["v"] = args.beta2 * state["v"] + (1 - args.beta2) * g.square()
    mh = state["m"] / (1 - args.beta1 ** (step + 1))
    vh = state["v"] / (1 - args.beta2 ** (step + 1))
    if kind == "sgd":
        return g
    if kind == "momentum":
        return mh
    if kind == "rms":
        return g / (vh.sqrt() + args.adamw_eps)
    if kind == "adam":
        return mh / (vh.sqrt() + args.adamw_eps)
    raise RuntimeError(f"unknown optimizer component: {kind}")


def procrustes_transport(old_basis, new_basis, vector, torch):
    """Canonical partial isometry from the old kernel to the new kernel."""
    overlap = new_basis.T @ old_basis
    u, singular, vh = torch.linalg.svd(overlap, full_matrices=False)
    transported = new_basis @ (u @ (vh @ (old_basis.T @ vector)))
    residual = float((transported.norm() - (old_basis.T @ vector).norm()).abs())
    return transported, {
        "minimum_principal_cosine": float(singular.min()),
        "maximum_principal_angle_sine": float(
            (1. - singular.min().square()).clamp_min(0.).sqrt()),
        "transport_norm_residual": residual,
    }


def run_arm(model, coordinate, warm, arm, budget, train, validation, schedule,
            response_rows, learning_verbalizers, topic_verbalizers, args, device,
            pad_id, torch, out, steps, phase):
    coordinate.data.copy_(warm)
    model.train()
    J0 = response_jacobian(model, coordinate, response_rows, topic_verbalizers,
                           args.response_batch_size, device, pad_id, torch)
    N0, d0 = null_basis(J0, torch)
    R0 = response_values(model, response_rows, topic_verbalizers,
                         args.response_batch_size, device, pad_id, torch)
    Rprev = R0.clone()
    initial_validation_loss, initial_validation_accuracy = evaluate(
        model, validation, learning_verbalizers, args.eval_batch_size, device, pad_id, torch)
    adam = {"m": torch.zeros_like(coordinate, dtype=torch.float64),
            "v": torch.zeros_like(coordinate, dtype=torch.float64)}
    trace, accepted_norms, backtracks, transport_diags = [], [], [], []
    recovery_residuals, recovery_corrections, recovery_iterations = [], [], []
    recovery_contraction_ratios = []
    recovery_predictor_drifts, recovery_corrected_drifts = [], []
    net_lyapunov_nonexpanding = 0
    recovery_failed_attempts = line_search_failures = 0
    maximum_correction_to_tangent_ratio = 0.
    covariant_momentum = torch.zeros_like(coordinate, dtype=torch.float64)
    previous_basis = None
    finite_response_action = linearized_response_action = 0.
    zero = 0
    max_drift = max_leak = max_idem = max_repeat = 0.
    ranks = []
    sync(torch, device)
    t0 = time.perf_counter()
    for step in range(steps):
        rows = [train[i] for i in schedule[args.warm_steps + step]]
        loss = task_loss(model, rows, learning_verbalizers, args.batch_size,
                         device, pad_id, torch)
        g = grad(loss, coordinate, torch).detach().double()
        if not torch.isfinite(g).all():
            raise RuntimeError(f"non-finite gradient in {arm} step {step}")
        J = response_jacobian(model, coordinate, response_rows, topic_verbalizers,
                              args.response_batch_size, device, pad_id, torch)
        Ncurrent, dcurrent = null_basis(J, torch)
        N, dg = (N0, d0) if arm == "source_momentum" else (Ncurrent, dcurrent)
        if arm == "budgeted_adamw":
            proposal = args.adamw_lr * adam_proposal(g, adam, step, args, torch)
        elif arm == "covariant_momentum":
            tangent_gradient = N @ (N.T @ g)
            if previous_basis is None:
                transported = torch.zeros_like(covariant_momentum)
                td = {"minimum_principal_cosine": 1.,
                      "maximum_principal_angle_sine": 0.,
                      "transport_norm_residual": 0.}
            else:
                transported, td = procrustes_transport(
                    previous_basis, N, covariant_momentum, torch)
            covariant_momentum = args.beta1 * transported + (1 - args.beta1) * tangent_gradient
            previous_basis = N.detach().clone()
            transport_diags.append(td)
            proposal = -args.step_norm * normalized(covariant_momentum)
        else:
            kind = "momentum" if arm in ("normal_recovery_momentum", "naive_momentum", "source_momentum") else "adam"
            direction = optimizer_direction(g, adam, step, kind, args, torch)
            proposal = -args.step_norm * normalized(N @ (N.T @ direction))
        old = coordinate.detach().clone()
        scale, accepted, used = 1., False, 0
        accepted_correction = accepted_iterations = 0.
        accepted_local_residual = 0.
        for bt in range(args.max_backtracks + 1):
            coordinate.data.copy_(old)
            coordinate.data.add_((scale * proposal).to(coordinate.dtype))
            Rnew = response_values(model, response_rows, topic_verbalizers,
                                   args.response_batch_size, device, pad_id, torch)
            trial_correction, trial_iterations = 0., 0
            if arm == "normal_recovery_momentum":
                predictor_drift = float((Rnew - R0).norm())
                trust_budget = args.recovery_trust_ratio * float((scale * proposal).norm())
                for iteration in range(args.recovery_iters):
                    residual = Rnew - R0
                    residual_norm = float(residual.norm())
                    if (residual_norm <= args.recovery_tol or
                            residual_norm <= args.recovery_target_fraction * predictor_drift):
                        break
                    remaining = trust_budget - trial_correction
                    if remaining <= 0:
                        break
                    Jtrial = response_jacobian(
                        model, coordinate, response_rows, topic_verbalizers,
                        args.response_batch_size, device, pad_id, torch)
                    gram = Jtrial @ Jtrial.T
                    ridge = args.recovery_ridge * max(float(gram.diag().mean()), 1e-30)
                    correction = -args.recovery_gain * Jtrial.T @ torch.linalg.solve(
                        gram + ridge * torch.eye(gram.shape[0], dtype=gram.dtype,
                                                  device=gram.device), residual)
                    correction_norm = float(correction.norm())
                    if correction_norm > remaining:
                        correction = correction * (remaining / max(correction_norm, 1e-30))
                        correction_norm = remaining
                    base_coordinate = coordinate.detach().clone()
                    accepted_line_search = False
                    line_scale = 1.
                    for _ in range(args.recovery_line_search):
                        coordinate.data.copy_(base_coordinate)
                        coordinate.data.add_((line_scale * correction).to(coordinate.dtype))
                        candidate_R = response_values(
                            model, response_rows, topic_verbalizers,
                            args.response_batch_size, device, pad_id, torch)
                        candidate_residual = float((candidate_R - R0).norm())
                        if candidate_residual <= args.recovery_contraction * residual_norm:
                            Rnew = candidate_R
                            trial_correction += line_scale * correction_norm
                            trial_iterations = iteration + 1
                            accepted_line_search = True
                            break
                        line_scale *= .5
                    if not accepted_line_search:
                        coordinate.data.copy_(base_coordinate)
                        line_search_failures += 1
                        break
                local_residual = float((Rnew - R0).norm())
                local_ok = (local_residual <= args.recovery_tol or
                            local_residual <= args.recovery_target_fraction * predictor_drift)
                if not local_ok:
                    recovery_failed_attempts += 1
            else:
                local_residual, local_ok = float((Rnew - Rprev).norm()), True
            if local_ok and float((Rnew - R0).norm()) <= budget * (1 + 1e-8):
                accepted, used = True, bt
                accepted_correction = trial_correction
                accepted_iterations = trial_iterations
                accepted_local_residual = local_residual
                accepted_predictor_drift = predictor_drift if arm == "normal_recovery_momentum" else None
                break
            scale *= .5
        if not accepted:
            coordinate.data.copy_(old)
            Ra = response_values(model, response_rows, topic_verbalizers,
                                 args.response_batch_size, device, pad_id, torch)
            Rb = response_values(model, response_rows, topic_verbalizers,
                                 args.response_batch_size, device, pad_id, torch)
            max_repeat = max(max_repeat, float((Ra - Rb).norm()))
            Rnew, scale, used = Rb, 0., args.max_backtracks + 1
            zero += 1
            actual_delta = torch.zeros_like(proposal)
        else:
            actual_delta = coordinate.detach().double() - old.double()
        drift = float((Rnew - R0).norm())
        finite_response_action += float((Rnew - Rprev).norm())
        linearized_response_action += float((J @ actual_delta).norm())
        if arm == "normal_recovery_momentum" and accepted:
            recovery_residuals.append(accepted_local_residual)
            recovery_corrections.append(accepted_correction)
            recovery_iterations.append(accepted_iterations)
            final_ratio = accepted_local_residual / max(accepted_predictor_drift, 1e-30)
            recovery_contraction_ratios.append(final_ratio)
            recovery_predictor_drifts.append(accepted_predictor_drift)
            recovery_corrected_drifts.append(accepted_local_residual)
            if accepted_local_residual <= float((Rprev - R0).norm()) + args.recovery_tol:
                net_lyapunov_nonexpanding += 1
            tangent_norm = float((scale * proposal).norm())
            maximum_correction_to_tangent_ratio = max(
                maximum_correction_to_tangent_ratio,
                accepted_correction / max(tangent_norm, 1e-30))
        Rprev = Rnew.clone()
        max_drift = max(max_drift, drift)
        max_leak = max(max_leak, dg["leakage"])
        max_idem = max(max_idem, dg["idempotence"])
        ranks.append(dg["rank"])
        accepted_norms.append(float((scale * proposal).norm()))
        backtracks.append(used)
        interval = args.scan_eval_interval if phase == "scan" else args.eval_interval
        if (step + 1) % interval == 0 or step == steps - 1:
            vl, va = evaluate(model, validation, learning_verbalizers, args.eval_batch_size,
                              device, pad_id, torch)
            trace.append({"step": step + 1, "validation_loss": vl,
                          "validation_accuracy": va, "global_response_drift": drift,
                          "accepted_step_norm": accepted_norms[-1]})
            print(f"[{phase}:{arm} B={budget:.3e}] {step+1}/{steps} "
                  f"loss={vl:.6f} acc={va:.4f} drift={drift:.2e}", flush=True)
    sync(torch, device)
    rec = {
        "seed": SEED, "phase": phase, "arm": arm, "budget": budget,
        "final_validation_loss": trace[-1]["validation_loss"],
        "final_validation_accuracy": trace[-1]["validation_accuracy"],
        "best_validation_loss": min(x["validation_loss"] for x in trace),
        "initial_validation_loss": initial_validation_loss,
        "initial_validation_accuracy": initial_validation_accuracy,
        "validation_loss_gain": initial_validation_loss - trace[-1]["validation_loss"],
        "finite_response_path_action": finite_response_action,
        "linearized_response_path_action": linearized_response_action,
        "loss_gain_per_finite_response_action": ((initial_validation_loss - trace[-1]["validation_loss"]) /
            max(finite_response_action, 1e-30)),
        "max_global_response_drift": max_drift,
        "zero_step_fraction": zero / steps,
        "accepted_steps": steps - zero,
        "median_accepted_step_norm": statistics.median(accepted_norms),
        "median_backtracks": statistics.median(backtracks),
        "max_projector_leakage": max_leak,
        "max_projector_idempotence": max_idem,
        "response_ranks": sorted(set(ranks)),
        "max_restored_state_response_repeat_error": max_repeat,
        "minimum_principal_cosine": (min(x["minimum_principal_cosine"] for x in transport_diags)
                                      if transport_diags else None),
        "maximum_principal_angle_sine": (max(x["maximum_principal_angle_sine"] for x in transport_diags)
                                          if transport_diags else None),
        "maximum_transport_norm_residual": (max(x["transport_norm_residual"] for x in transport_diags)
                                             if transport_diags else None),
        "maximum_local_recovery_residual": (max(recovery_residuals)
                                               if recovery_residuals else None),
        "cumulative_normal_correction_norm": (sum(recovery_corrections)
                                               if recovery_corrections else None),
        "median_recovery_iterations": (statistics.median(recovery_iterations)
                                         if recovery_iterations else None),
        "maximum_normal_recovery_contraction_ratio": (
            max(recovery_contraction_ratios) if recovery_contraction_ratios else 0.),
        "normal_recovery_contraction_events": len(recovery_contraction_ratios),
        "maximum_predictor_response_drift": (max(recovery_predictor_drifts)
                                                if recovery_predictor_drifts else None),
        "maximum_corrected_response_drift": (max(recovery_corrected_drifts)
                                                if recovery_corrected_drifts else None),
        "net_response_lyapunov_nonexpanding_fraction": (
            net_lyapunov_nonexpanding / max(len(recovery_corrected_drifts), 1)
            if arm == "normal_recovery_momentum" else None),
        "recovery_failed_attempts": (recovery_failed_attempts
                                       if arm == "normal_recovery_momentum" else None),
        "recovery_line_search_failures": (line_search_failures
                                            if arm == "normal_recovery_momentum" else None),
        "maximum_correction_to_tangent_ratio": (maximum_correction_to_tangent_ratio
                                                if arm == "normal_recovery_momentum" else None),
        "timed_seconds": time.perf_counter() - t0,
        "trace": trace,
    }
    name = f"{phase}_{arm}_B{budget:.3e}_{SEED}.json".replace("-", "m")
    (out / name).write_text(json.dumps(rec, indent=2) + "\n")
    return rec


def response_probe_scale(model, coordinate, warm, train, schedule, response_rows,
                         learning_verbalizers, topic_verbalizers, args, device,
                         pad_id, torch):
    coordinate.data.copy_(warm)
    R0 = response_values(model, response_rows, topic_verbalizers,
                         args.response_batch_size, device, pad_id, torch)
    J = response_jacobian(model, coordinate, response_rows, topic_verbalizers,
                          args.response_batch_size, device, pad_id, torch)
    N, _ = null_basis(J, torch)
    rows = [train[i] for i in schedule[args.warm_steps]]
    g = grad(task_loss(model, rows, learning_verbalizers, args.batch_size,
                       device, pad_id, torch), coordinate, torch).detach().double()
    state = {"m": torch.zeros_like(coordinate, dtype=torch.float64),
             "v": torch.zeros_like(coordinate, dtype=torch.float64)}
    raw = adam_proposal(g, state, 0, args, torch)
    proposals = {
        "current_gradient": -args.step_norm * normalized(N @ (N.T @ g)),
        "current_projected_adamw": args.step_norm * normalized(N @ (N.T @ raw)),
        "raw_adamw": args.adamw_lr * raw,
    }
    old = coordinate.detach().clone(); drifts = {}
    for name, proposal in proposals.items():
        coordinate.data.copy_(old); coordinate.data.add_(proposal.to(coordinate.dtype))
        R1 = response_values(model, response_rows, topic_verbalizers,
                             args.response_batch_size, device, pad_id, torch)
        drifts[name] = float((R1 - R0).norm())
    coordinate.data.copy_(old)
    positive = [v for v in drifts.values() if math.isfinite(v) and v > 0]
    if not positive:
        raise RuntimeError("response probe produced no positive finite drift")
    scale = max(max(positive), 1e-8)
    return scale, drifts


def nominate_budgets(scan_records):
    by_budget = {}
    for rec in scan_records:
        by_budget.setdefault(rec["budget"], {})[rec["arm"]] = rec
    rows = []
    for budget, arms in sorted(by_budget.items()):
        complete = all(a in arms for a in SCAN_ARMS)
        feasible = complete and all(
            math.isfinite(arms[a]["final_validation_loss"]) and
            arms[a]["zero_step_fraction"] <= .75 and
            arms[a]["max_global_response_drift"] <= budget * (1 + 1e-8)
            for a in SCAN_ARMS)
        if complete:
            cur = arms["current_kernel"]["final_validation_loss"]
            margins = {a: arms[a]["final_validation_loss"] - cur
                       for a in ("source_frozen", "projected_adamw")}
            worst_margin = min(margins.values())
            max_zero = max(arms[a]["zero_step_fraction"] for a in SCAN_ARMS)
        else:
            margins, worst_margin, max_zero = {}, -math.inf, 1.
        rows.append({"budget": budget, "complete": complete, "feasible": feasible,
                     "current_loss_margins": margins,
                     "current_worst_control_loss_margin": worst_margin,
                     "max_zero_step_fraction": max_zero})
    eligible = [x for x in rows if x["feasible"]]
    ranked = sorted(eligible, key=lambda x: (x["current_worst_control_loss_margin"],
                                             -x["max_zero_step_fraction"]), reverse=True)
    nominated = sorted(x["budget"] for x in ranked[:2]) if len(ranked) >= 2 else []
    return nominated, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--outdir", default="pythia_r15d_results")
    ap.add_argument("--steps", type=int, default=150)
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
    ap.add_argument("--adamw-lr", type=float, default=.01)
    ap.add_argument("--beta1", type=float, default=.9)
    ap.add_argument("--beta2", type=float, default=.999)
    ap.add_argument("--adamw-eps", type=float, default=1e-8)
    ap.add_argument("--max-backtracks", type=int, default=12)
    ap.add_argument("--recovery-iters", type=int, default=8)
    ap.add_argument("--recovery-tol", type=float, default=2e-6)
    ap.add_argument("--recovery-ridge", type=float, default=1e-8)
    ap.add_argument("--recovery-gain", type=float, default=.5)
    ap.add_argument("--recovery-contraction", type=float, default=.95)
    ap.add_argument("--recovery-target-fraction", type=float, default=.8)
    ap.add_argument("--recovery-trust-ratio", type=float, default=.5)
    ap.add_argument("--recovery-line-search", type=int, default=8)
    ap.add_argument("--eval-interval", type=int, default=15)
    ap.add_argument("--train-examples", type=int, default=2048)
    ap.add_argument("--validation-examples", type=int, default=256)
    ap.add_argument("--response-per-class", type=int, default=4)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if args.quick:
        args.steps, args.warm_steps = 4, 2
        args.train_examples, args.validation_examples, args.response_per_class = 128, 32, 2
        args.eval_interval = 2
    device = torch.device(args.device)
    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    mode = "quick_nonclaim" if args.quick else "one_seed_fixed_target_normal_recovery_development"
    print(f"protocol={PROTOCOL} device={device} mode={mode}", flush=True)
    train_raw, val_raw, response_raw, data_record = load_data(
        SEED, args.train_examples, args.validation_examples, args.response_per_class)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    learning_words, learning_verbalizers = choose_single_token_verbalizers(
        tokenizer, ((" negative", " bad", " terrible"),
                    (" positive", " good", " great")))
    topic_words, topic_verbalizers = choose_single_token_verbalizers(
        tokenizer, ((" World", " Global", " Politics"),
                    (" Sports", " Game", " Athletic"),
                    (" Business", " Market", " Finance"),
                    (" Technology", " Tech", " Science")))
    train = tokenize_sst2(train_raw, tokenizer, args.seq_len, torch)
    validation = tokenize_sst2(val_raw, tokenizer, args.seq_len, torch)
    response_rows = tokenize_agnews(response_raw, tokenizer, args.seq_len, torch)
    seed_all(SEED, torch)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID).to(device)
    model.config.use_cache = False
    dropout_modules = 0
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.; dropout_modules += 1
    coordinate, chosen_layers = inject_chart_lora(
        model, args.chart_dim, args.lora_rank, args.layers, torch)
    model.to(device)
    schedule = batch_schedule(len(train), args.warm_steps, args.steps,
                              args.batch_size, SEED + 71)
    warm = warm_start(model, coordinate, train, schedule, args, learning_verbalizers,
                      device, tokenizer.pad_token_id, torch)
    coordinate.data.copy_(warm)
    repeats = [response_values(model, response_rows, topic_verbalizers,
                               args.response_batch_size, device,
                               tokenizer.pad_token_id, torch) for _ in range(3)]
    repeat_error = max(float((x - repeats[0]).norm()) for x in repeats[1:])
    J = response_jacobian(model, coordinate, response_rows, topic_verbalizers,
                          args.response_batch_size, device, tokenizer.pad_token_id, torch)
    _, preflight = null_basis(J, torch)
    print(f"[preflight] response_rank={preflight['rank']} repeat={repeat_error:.3e} "
          f"leakage={preflight['leakage']:.3e}", flush=True)
    if repeat_error > 1e-10:
        raise RuntimeError("fixed-state response determinism preflight failed")
    if preflight["rank"] != 4:
        raise RuntimeError(f"expected independent AG News response rank 4, got {preflight['rank']}")
    records = []
    for budget in FROZEN_BUDGETS:
        for arm in ARMS:
            records.append(run_arm(
                model, coordinate, warm, arm, budget, train, validation, schedule,
                response_rows, learning_verbalizers, topic_verbalizers, args, device,
                tokenizer.pad_token_id, torch, out, args.steps, "variational"))
    by = {(r["budget"], r["arm"]): r for r in records}
    comparisons = []
    for budget in FROZEN_BUDGETS:
        candidate_rec, naive = by[budget, "normal_recovery_momentum"], by[budget, "naive_momentum"]
        item, wins = {"budget": budget}, True
        for control in CONTROLS:
            c = by[budget, control]
            dl = c["final_validation_loss"] - candidate_rec["final_validation_loss"]
            da = candidate_rec["final_validation_accuracy"] - c["final_validation_accuracy"]
            item[f"{control}_minus_normal_recovery_loss"] = dl
            item[f"normal_recovery_minus_{control}_accuracy"] = da
            wins = wins and dl > 0 and da >= 0
        item["naive_minus_normal_recovery_finite_response_action"] = (
            naive["finite_response_path_action"] - candidate_rec["finite_response_path_action"])
        item["normal_recovery_variationally_dominates_naive"] = (
            item["naive_momentum_minus_normal_recovery_loss"] > 0 and
            item["normal_recovery_minus_naive_momentum_accuracy"] >= 0 and
            item["naive_minus_normal_recovery_finite_response_action"] >= 0)
        item["normal_recovery_beats_all_controls"] = wins
        comparisons.append(item)
    numerical_gates = {
        "pretrained_model_loaded": getattr(model.config, "_name_or_path", "") == MODEL_ID,
        "dropout_disabled": dropout_modules > 0 and all(
            not isinstance(m, torch.nn.Dropout) or m.p == 0 for m in model.modules()),
        "fixed_state_repeat_error_at_most_1e_10": repeat_error <= 1e-10,
        "all_runs_finite": all(math.isfinite(r["final_validation_loss"]) for r in records),
        "all_global_budgets_respected": all(
            r["max_global_response_drift"] <= r["budget"] * (1 + 1e-8) for r in records),
        "projector_leakage_at_most_1e_10": max(r["max_projector_leakage"] for r in records) <= 1e-10,
        "projector_idempotence_at_most_1e_10": max(r["max_projector_idempotence"] for r in records) <= 1e-10,
        "response_rank_constant_four": all(r["response_ranks"] == [4] for r in records),
        "every_arm_accepts_at_least_one_step": all(r["accepted_steps"] >= 1 for r in records),
        "r_l_datasets_and_tasks_distinct": data_record["r_l_datasets_distinct"],
        "r14a_r1_budgets_frozen_exactly": tuple(FROZEN_BUDGETS) == (
            0.002271741101426359, 0.004543482202852718),
        "covariant_transport_norm_residual_at_most_1e_10": max(
            r["maximum_transport_norm_residual"] or 0. for r in records) <= 1e-10,
        "all_path_actions_finite_nonnegative": all(
            math.isfinite(r["finite_response_path_action"]) and
            r["finite_response_path_action"] >= 0 and
            math.isfinite(r["linearized_response_path_action"]) and
            r["linearized_response_path_action"] >= 0 for r in records),
        "fixed_target_recovery_has_accepted_steps": all(
            r["arm"] != "normal_recovery_momentum" or r["accepted_steps"] >= 1
            for r in records),
        "recovery_trust_ratio_respected": max(
            r["maximum_correction_to_tangent_ratio"] or 0. for r in records
        ) <= args.recovery_trust_ratio * (1 + 1e-8),
        "accepted_predictor_to_fixed_target_contraction_gate": all(
            r["arm"] != "normal_recovery_momentum" or
            (r["accepted_steps"] >= 1 and
             r["normal_recovery_contraction_events"] >= 1 and
             r["maximum_normal_recovery_contraction_ratio"] <=
             args.recovery_target_fraction * (1 + 1e-8)) for r in records),
    }
    candidate = all(numerical_gates.values()) and all(
        x["normal_recovery_variationally_dominates_naive"] and
        x["normal_recovery_beats_all_controls"] for x in comparisons) and not args.quick
    summary = {
        "protocol": PROTOCOL, "mode": mode, "model": MODEL_ID, "pretrained": True,
        "learning_target_L": "GLUE/SST-2 prompted binary sentiment classification loss",
        "response_map_R": "four class-conditional AG News topic-logit-margin coordinates on frozen inputs",
        "r_l_separation": "different datasets, prompts, labels, verbalizers and declared functionals",
        "seed": SEED, "development_reference": "R15C_R1_TANGENT_NORMAL_RECOVERY_INCONCLUSIVE_FAIL_CLOSED",
        "frozen_budgets": FROZEN_BUDGETS,
        "variational_action": "sum_t ||R(theta_{t+1})-R(theta_t)||_2 with a separately reported sum_t ||DR(theta_t) Delta theta_t||_2",
        "recovery": "damped pseudoinverse feedback toward the fixed source response R0; accepted corrected drift must be at most 0.8 of the tangent-predictor drift and remain inside the frozen global response ball",
        "r15c_r1_failure_repair": "R15c/r1 incorrectly targeted Rprev and thereby imposed near-exact response equality at every step; R15d targets the fixed response fibre R(theta)=R0 and reports finite path action independently",
        "geometric_origin": "Geometric-Flow normally attracting response-fibre control law; K=1-inspired transverse stability only, not a K=1 bridge result",
        "arms": ARMS,
        "candidate_gate": "at both frozen budgets normal_recovery momentum beats every control in loss with noninferior accuracy and has no larger finite response path action than naive momentum",
        "learning_verbalizers": {"words": learning_words, "token_ids": learning_verbalizers},
        "response_verbalizers": {"words": topic_words, "token_ids": topic_verbalizers},
        "adapted_layers": chosen_layers, "data_record": data_record,
        "configuration": vars(args),
        "preflight": {**preflight, "fixed_state_response_repeat_error": repeat_error,
                      "disabled_dropout_modules": dropout_modules},
        "comparisons": comparisons,
        "numerical_gates": numerical_gates,
        "records": records,
        "scientific_status": ("R15D_FIXED_TARGET_NORMAL_RECOVERY_CANDIDATE"
                              if candidate else ("R15D_QUICK_NONCLAIM" if args.quick
                                                 else "R15D_FIXED_TARGET_NORMAL_RECOVERY_INCONCLUSIVE_FAIL_CLOSED")),
        "wall_seconds": time.time() - started,
        "claim_boundary": "One-seed fail-closed tangent-plus-normal-recovery development audit on pretrained Pythia-160M. K=1 supplies motivation only: this does not establish the repository's open K=1 bridge. A positive outcome only nominates the frozen recovery rule for new-seed testing; it is not confirmation, exact Principle-R attainment, a universal variational principle, or a Picard theorem."
    }
    (out / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0 if candidate or args.quick else 2


if __name__ == "__main__":
    raise SystemExit(main())
