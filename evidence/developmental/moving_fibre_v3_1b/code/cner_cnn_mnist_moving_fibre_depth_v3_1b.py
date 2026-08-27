#!/usr/bin/env python3
"""Eight-seed deep-path moving-response-fibre audit for the F16 programme."""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.util
import json
import math
import sys
import zipfile
from pathlib import Path

import numpy as np


def load_engine():
    path = Path(__file__).with_name("cner_cnn_mnist_fisher_confirm_v16.py")
    spec = importlib.util.spec_from_file_location("f16_engine", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["f16_engine"] = mod
    spec.loader.exec_module(mod)
    return mod


E = load_engine()


@dataclasses.dataclass(frozen=True)
class Protocol(E.Protocol):
    protocol_name: str = "CNER_CNN_MNIST_MOVING_FIBRE_DEPTH_V3_1B"
    seeds: int = 8
    base_seed: int = 67726
    adapt_steps: int = 192
    capability_loss_reduction_fraction: float = 0.20
    executable_step_radius: float = 0.04
    response_retraction_iterations: int = 2
    response_retraction_ridge: float = 1e-6
    response_retraction_relative_gate: float = 2e-3
    response_retraction_step_ratio_gate: float = 0.20
    moving_kernel_residual_gate: float = 1e-5
    moving_kernel_rank_gate: int = 8
    max_step_principal_angle_gate_rad: float = 0.80
    fibre_rotation_signal_gate_rad: float = 0.02
    action_left_trapezoid_gate: float = 0.08
    minimum_comparable_seeds: int = 7
    success_required: int = 6
    wrong_metric_natural_win_ceiling: int = 2
    minimum_natural_steps: int = 8
    minimum_median_path_steps: float = 10.0
    rowspace_rotation_signal_gate_rad: float = 0.02
    fixed_replay_action_difference_gate: float = 0.02
    fixed_replay_difference_required: int = 6
    chart_projection_iterations: int = 3
    chart_internal_kernel_target: float = 1e-7


def load_protocol(path):
    p = Protocol()
    if path:
        raw = json.loads(Path(path).read_text())
        valid = {f.name for f in dataclasses.fields(p)}
        unknown = set(raw) - valid
        if unknown:
            raise ValueError(f"unknown protocol keys: {sorted(unknown)}")
        p = Protocol(**raw)
    locked = (
        p.protocol_name == "CNER_CNN_MNIST_MOVING_FIBRE_DEPTH_V3_1B"
        and p.seeds == 8
        and p.base_seed == 67726
        and p.adapt_steps == 192
        and p.capability_loss_reduction_fraction == 0.20
        and p.minimum_comparable_seeds == 7
        and p.success_required == 6
        and p.minimum_natural_steps == 8
        and p.minimum_median_path_steps == 10.0
        and p.fixed_replay_difference_required == 6
        and p.response_retraction_iterations == 2
    )
    if not locked:
        raise ValueError("v3.1b frozen depth protocol violated")
    return p


def response_vector(theta, model, spec, anchor_x, functional_call):
    return functional_call(model, E.params_from_vector(theta, spec), (anchor_x,)).reshape(-1)


def row_space_projector(J, p, torch):
    _, singular, vh = torch.linalg.svd(J, full_matrices=False)
    threshold = singular.max().clamp_min(1e-12) * p.rcond
    rank = int((singular > threshold).sum())
    vr = vh[:rank].T if rank else torch.empty((J.shape[1], 0), device=J.device, dtype=J.dtype)

    def project(v):
        return v - vr @ (vr.T @ v) if rank else v

    return project, rank, J.shape[1] - rank, float(threshold), vr


def align_basis(candidate, reference, torch):
    u, _, vh = torch.linalg.svd(candidate.T @ reference, full_matrices=False)
    return candidate @ (u @ vh)


def moving_kernel_basis(J, reference, source_row_basis, p, torch):
    project, rank, null_dim, threshold, row_basis = row_space_projector(J, p, torch)
    if null_dim < p.chart_dim:
        raise RuntimeError(f"moving numerical null dimension {null_dim} < {p.chart_dim}")
    basis = project(reference)
    if int(torch.linalg.matrix_rank(basis)) < p.chart_dim:
        raise RuntimeError("transported source basis lost rank")
    for _ in range(p.chart_projection_iterations):
        basis = project(basis)
        basis, _ = torch.linalg.qr(basis, mode="reduced")
    basis = align_basis(basis, reference, torch)
    residual = float(torch.linalg.norm(J @ basis) / torch.linalg.norm(J).clamp_min(1e-12))
    orth = float(torch.linalg.norm(basis.T @ basis - torch.eye(p.chart_dim, device=basis.device)))
    s = torch.linalg.svdvals(reference.T @ basis).clamp(0, 1)
    angles = torch.acos(s)
    max_angle = float(angles.max())
    if rank == source_row_basis.shape[1]:
        overlap = source_row_basis.T @ row_basis
        row_cosines = torch.linalg.svdvals(overlap).clamp(0, 1)
        rowspace_max_angle = float(torch.acos(row_cosines).max())
        projector_distance = float(
            torch.sqrt((2 * rank - 2 * overlap.square().sum()).clamp_min(0))
            / math.sqrt(max(2 * rank, 1))
        )
    else:
        rowspace_max_angle = math.pi / 2
        projector_distance = 1.0
    if residual > p.chart_internal_kernel_target:
        raise RuntimeError(f"moving kernel residual {residual} exceeds internal target")
    return basis.detach(), {
        "response_rank": rank,
        "numerical_null_dimension": null_dim,
        "singular_value_threshold": threshold,
        "kernel_residual": residual,
        "orthogonality_residual": orth,
        "max_principal_angle_from_previous_rad": max_angle,
        "rowspace_max_angle_from_source_rad": rowspace_max_angle,
        "rowspace_projector_distance_normalized": projector_distance,
    }


def initial_kernel_basis(theta, J, task_x, task_y, model, spec, p, seed, deps):
    torch, _, F, functional_call = deps[:4]
    project, rank, null_dim, threshold, _ = row_space_projector(J, p, torch)
    if null_dim < p.chart_dim:
        raise RuntimeError("source numerical kernel too small")
    v = theta.detach().clone().requires_grad_(True)
    loss = F.cross_entropy(functional_call(model, E.params_from_vector(v, spec), (task_x,)), task_y)
    grad = torch.autograd.grad(loss, v)[0].detach()
    cols = []
    first = project(-grad)
    if first.norm() <= 1e-8:
        raise RuntimeError("source projected task gradient vanished")
    cols.append(first / first.norm())
    gen = torch.Generator(device=theta.device).manual_seed(seed + 910001)
    for _ in range(p.chart_dim * 50):
        if len(cols) == p.chart_dim:
            break
        q = project(torch.randn(theta.numel(), generator=gen, device=theta.device))
        for b in cols:
            q = q - b * torch.dot(b, q)
        q = project(q)
        if q.norm() > 1e-8:
            cols.append(q / q.norm())
    if len(cols) != p.chart_dim:
        raise RuntimeError("could not construct source kernel chart")
    basis = torch.stack(cols, dim=1)
    for _ in range(p.chart_projection_iterations):
        basis = project(basis)
        basis, _ = torch.linalg.qr(basis, mode="reduced")
    residual = float(torch.linalg.norm(J @ basis) / torch.linalg.norm(J).clamp_min(1e-12))
    return basis.detach(), {
        "response_rank": rank,
        "numerical_null_dimension": null_dim,
        "singular_value_threshold": threshold,
        "kernel_residual": residual,
    }


def full_gradient(theta, model, spec, x, y, torch, F, functional_call):
    v = theta.detach().clone().requires_grad_(True)
    loss = F.cross_entropy(functional_call(model, E.params_from_vector(v, spec), (x,)), y)
    return torch.autograd.grad(loss, v)[0].detach()


def capability_loss(theta, model, spec, x, y, torch, F, functional_call):
    with torch.no_grad():
        return float(F.cross_entropy(functional_call(model, E.params_from_vector(theta, spec), (x,)), y))


def state_geometry(theta, basis, model, spec, metric_x, dev_x, dev_y, source_scales, p, deps):
    torch, _, F, functional_call = deps[:4]
    raw = E.output_fisher_raw(theta, basis, model, spec, metric_x, torch, functional_call)
    metric, meta = E.regularize_metric(raw, p.metric_eigen_floor_relative, torch)
    grad_full = full_gradient(theta, model, spec, dev_x, dev_y, torch, F, functional_call)
    grad = basis.T @ grad_full
    h = p.h0 + float(torch.sqrt((grad @ torch.linalg.solve(metric, grad)).clamp_min(1e-24))) / max(source_scales["true"], 1e-12)
    reverse = torch.arange(p.chart_dim - 1, -1, -1, device=theta.device)
    perm = torch.eye(p.chart_dim, device=theta.device)[:, reverse]
    wrong = perm.T @ metric @ perm
    h_wrong = p.h0 + float(torch.sqrt((grad @ torch.linalg.solve(wrong, grad)).clamp_min(1e-24))) / max(source_scales["wrong"], 1e-12)
    eigen = torch.linalg.eigvalsh(raw)
    effective_rank = int((eigen > eigen.max().clamp_min(1e-12) * p.metric_eigen_floor_relative).sum())
    return {
        "metric": metric,
        "wrong_metric": wrong,
        "h": h,
        "h_wrong": h_wrong,
        "condition": meta["condition"],
        "effective_rank": effective_rank,
    }


def retract_response(theta_trial, target_response, model, spec, anchor_x, p, deps):
    torch, _, _, functional_call = deps[:4]
    theta = theta_trial.detach()
    correction_total = torch.zeros_like(theta)
    initial_error = None
    for _ in range(p.response_retraction_iterations):
        response = response_vector(theta, model, spec, anchor_x, functional_call)
        error = response - target_response
        if initial_error is None:
            initial_error = float(error.norm())
        J = E.response_jacobian(theta, model, spec, anchor_x, torch, functional_call)
        system = J @ J.T + p.response_retraction_ridge * torch.eye(J.shape[0], device=theta.device)
        correction = -J.T @ torch.linalg.solve(system, error)
        theta = (theta + correction).detach()
        correction_total += correction
    final_response = response_vector(theta, model, spec, anchor_x, functional_call)
    final_relative = float((final_response - target_response).norm() / target_response.norm().clamp_min(1e-12))
    J_final = E.response_jacobian(theta, model, spec, anchor_x, torch, functional_call)
    return theta, J_final, {
        "initial_error": initial_error,
        "final_relative_response_error": final_relative,
        "correction_norm": float(correction_total.norm()),
    }


def normalized_direction(direction, metric, radius, torch):
    norm = torch.sqrt((direction @ metric @ direction).clamp_min(1e-24))
    return radius * direction / norm


def algorithm_direction(name, grad, metric, wrong_metric, state, p, torch):
    if name == "natural_gradient":
        raw = -torch.linalg.solve(metric, grad)
    elif name == "wrong_fisher_natural_gradient":
        raw = -torch.linalg.solve(wrong_metric, grad)
    elif name == "normalized_sgd":
        raw = -grad
    elif name == "sign_gradient":
        raw = -torch.sign(grad)
    elif name == "normalized_momentum":
        state["momentum"] = p.momentum_beta * state["momentum"] + grad
        raw = -state["momentum"]
    elif name == "adam":
        state["t"] += 1
        state["m"] = 0.9 * state["m"] + 0.1 * grad
        state["v"] = 0.999 * state["v"] + 0.001 * grad.square()
        mhat = state["m"] / (1 - 0.9 ** state["t"])
        vhat = state["v"] / (1 - 0.999 ** state["t"])
        raw = -mhat / (torch.sqrt(vhat) + 1e-8)
    else:
        raise ValueError(name)
    return normalized_direction(raw, metric, p.executable_step_radius, torch)


def transport_state(state, old_basis, new_basis):
    transport = new_basis.T @ old_basis
    for key in ("momentum", "m"):
        state[key] = transport @ state[key]
    # Adam's coordinatewise second moment is intentionally diagnostic and is
    # transported by squared overlaps rather than treated as a tensor.
    state["v"] = transport.square() @ state["v"]


def execute_online_path(name, theta0, basis0, target_response, batches, target_loss, model, spec,
                        anchor_x, cap_x, cap_y, metric_x, dev_x, dev_y, source_scales,
                        source_row_basis, p, deps, move_basis=True):
    torch, _, F, functional_call = deps[:4]
    theta = theta0.detach().clone()
    basis = basis0.detach().clone()
    state = {
        "momentum": torch.zeros(p.chart_dim, device=theta.device),
        "m": torch.zeros(p.chart_dim, device=theta.device),
        "v": torch.zeros(p.chart_dim, device=theta.device),
        "t": 0,
    }
    geom = state_geometry(theta, basis, model, spec, metric_x, dev_x, dev_y, source_scales, p, deps)
    action_left = action_trap = wrong_left = wrong_trap = 0.0
    length_left = length_trap = wrong_length_left = wrong_length_trap = 0.0
    max_kernel = max_angle_step = max_response = max_retract_ratio = 0.0
    max_angle_source = 0.0
    max_rowspace_angle_source = 0.0
    max_rowspace_projector_distance = 0.0
    max_condition = geom["condition"]
    min_effective_rank = geom["effective_rank"]
    tangent_residuals = []
    records = []
    hit = False
    initial_cap = capability_loss(theta, model, spec, cap_x, cap_y, torch, F, functional_call)
    final_cap = initial_cap
    for step in range(p.adapt_steps):
        x, y = batches[step % len(batches)]
        grad_full = full_gradient(theta, model, spec, x, y, torch, F, functional_call)
        grad = basis.T @ grad_full
        dz = algorithm_direction(name, grad, geom["metric"], geom["wrong_metric"], state, p, torch)
        tangent_step = basis @ dz
        trial = theta + tangent_step
        new_theta, J_new, retract = retract_response(trial, target_response, model, spec, anchor_x, p, deps)
        candidate_cap = capability_loss(new_theta, model, spec, cap_x, cap_y, torch, F, functional_call)
        endpoint_bisection = False
        if candidate_cap <= target_loss:
            endpoint_bisection = True
            low, high = 0.0, 1.0
            best = (new_theta, J_new, retract, candidate_cap, 1.0)
            for _ in range(10):
                alpha = 0.5 * (low + high)
                candidate, candidate_J, candidate_retract = retract_response(
                    theta + alpha * tangent_step, target_response, model, spec, anchor_x, p, deps
                )
                candidate_loss = capability_loss(
                    candidate, model, spec, cap_x, cap_y, torch, F, functional_call
                )
                if candidate_loss <= target_loss:
                    high = alpha
                    best = (candidate, candidate_J, candidate_retract, candidate_loss, alpha)
                else:
                    low = alpha
            new_theta, J_new, retract, candidate_cap, endpoint_alpha = best
            tangent_step = endpoint_alpha * tangent_step
        else:
            endpoint_alpha = 1.0
        transported_basis, chart = moving_kernel_basis(J_new, basis, source_row_basis, p, torch)
        new_basis = transported_basis if move_basis else basis0
        transport_state(state, basis, new_basis)
        new_geom = state_geometry(new_theta, new_basis, model, spec, metric_x, dev_x, dev_y, source_scales, p, deps)
        delta = new_theta - theta
        z_left = basis.T @ delta
        z_right = new_basis.T @ delta
        residual = float((delta - basis @ z_left).norm() / delta.norm().clamp_min(1e-12))
        dl_l = float(torch.sqrt((z_left @ geom["metric"] @ z_left).clamp_min(0)))
        dl_r = float(torch.sqrt((z_right @ new_geom["metric"] @ z_right).clamp_min(0)))
        wdl_l = float(torch.sqrt((z_left @ geom["wrong_metric"] @ z_left).clamp_min(0)))
        wdl_r = float(torch.sqrt((z_right @ new_geom["wrong_metric"] @ z_right).clamp_min(0)))
        action_left += dl_l / geom["h"]
        action_trap += 0.5 * (dl_l / geom["h"] + dl_r / new_geom["h"])
        wrong_left += wdl_l / geom["h_wrong"]
        wrong_trap += 0.5 * (wdl_l / geom["h_wrong"] + wdl_r / new_geom["h_wrong"])
        length_left += dl_l
        length_trap += 0.5 * (dl_l + dl_r)
        wrong_length_left += wdl_l
        wrong_length_trap += 0.5 * (wdl_l + wdl_r)
        retract_ratio = retract["correction_norm"] / max(float(tangent_step.norm()), 1e-12)
        source_s = torch.linalg.svdvals(basis0.T @ new_basis).clamp(0, 1)
        source_angle = float(torch.acos(source_s).max())
        max_kernel = max(max_kernel, chart["kernel_residual"])
        max_angle_step = max(max_angle_step, chart["max_principal_angle_from_previous_rad"])
        max_angle_source = max(max_angle_source, source_angle)
        max_rowspace_angle_source = max(max_rowspace_angle_source, chart["rowspace_max_angle_from_source_rad"])
        max_rowspace_projector_distance = max(
            max_rowspace_projector_distance, chart["rowspace_projector_distance_normalized"]
        )
        max_response = max(max_response, retract["final_relative_response_error"])
        max_retract_ratio = max(max_retract_ratio, retract_ratio)
        max_condition = max(max_condition, new_geom["condition"])
        min_effective_rank = min(min_effective_rank, new_geom["effective_rank"])
        tangent_residuals.append(residual)
        final_cap = candidate_cap
        records.append({
            "step": step + 1,
            "capability_loss": final_cap,
            "kernel_residual": chart["kernel_residual"],
            "step_principal_angle_rad": chart["max_principal_angle_from_previous_rad"],
            "source_principal_angle_rad": source_angle,
            "rowspace_max_angle_from_source_rad": chart["rowspace_max_angle_from_source_rad"],
            "rowspace_projector_distance_normalized": chart["rowspace_projector_distance_normalized"],
            "response_relative_error": retract["final_relative_response_error"],
            "retraction_step_ratio": retract_ratio,
            "tangent_residual": residual,
            "metric_condition": new_geom["condition"],
            "metric_effective_rank": new_geom["effective_rank"],
            "endpoint_bisection": endpoint_bisection,
            "endpoint_step_fraction": endpoint_alpha,
        })
        theta, basis, geom = new_theta, new_basis, new_geom
        if final_cap <= target_loss:
            hit = True
            break
    action_rel = abs(action_trap - action_left) / max(abs(action_trap), 1e-12)
    wrong_rel = abs(wrong_trap - wrong_left) / max(abs(wrong_trap), 1e-12)
    admissible = bool(
        hit
        and (max_kernel <= p.moving_kernel_residual_gate if move_basis else True)
        and max_angle_step <= p.max_step_principal_angle_gate_rad
        and max_response <= p.response_retraction_relative_gate
        and max_retract_ratio <= p.response_retraction_step_ratio_gate
        and min_effective_rank >= p.moving_kernel_rank_gate
        and action_rel <= p.action_left_trapezoid_gate
        and wrong_rel <= p.action_left_trapezoid_gate
    )
    return {
        "hit_capability": hit,
        "steps": len(records),
        "initial_capability_loss": initial_cap,
        "final_capability_loss": final_cap,
        "moving_fibre_action": action_trap,
        "wrong_moving_fibre_action": wrong_trap,
        "moving_fisher_length": length_trap,
        "wrong_moving_fisher_length": wrong_length_trap,
        "left_trapezoid_relative_change": action_rel,
        "wrong_left_trapezoid_relative_change": wrong_rel,
        "max_moving_kernel_residual": max_kernel,
        "max_step_principal_angle_rad": max_angle_step,
        "max_source_principal_angle_rad": max_angle_source,
        "fibre_rotation_signal": (
            max_angle_source >= p.fibre_rotation_signal_gate_rad if move_basis else False
        ),
        "max_rowspace_angle_from_source_rad": max_rowspace_angle_source,
        "max_rowspace_projector_distance_normalized": max_rowspace_projector_distance,
        "rowspace_rotation_signal": max_rowspace_angle_source >= p.rowspace_rotation_signal_gate_rad,
        "basis_mode": "moving_kernel" if move_basis else "fixed_source_replay",
        "max_response_relative_error": max_response,
        "max_retraction_step_ratio": max_retract_ratio,
        "median_discrete_tangent_residual": float(np.median(tangent_residuals)) if tangent_residuals else None,
        "max_metric_condition": max_condition,
        "min_metric_effective_rank": min_effective_rank,
        "admissible": admissible,
        "trajectory": records,
    }


def metric_gate(cert, p):
    return (
        cert["kl_spearman"] >= p.kl_spearman_gate
        and cert["kl_median_relative_error"] <= p.kl_median_relative_error_gate
        and cert["gauge_logit_relative_residual"] <= p.gauge_logit_residual_gate
        and cert["gauge_fisher_relative_residual"] <= p.gauge_fisher_relative_gate
        and cert["representation_gauge_relative_change"] >= p.representation_gauge_change_gate
    )


def one_seed(seed, p, outdir, device, deps):
    torch, nn, F, functional_call, DataLoader, _, Subset, datasets, transforms = deps
    E.seed_everything(seed, torch)
    root = str(outdir.parent / "mnist_data")
    train = datasets.MNIST(root, train=True, download=True, transform=transforms.ToTensor())
    test = datasets.MNIST(root, train=False, download=True, transform=transforms.ToTensor())
    src, adapt, dev, cap, metric = E.disjoint_train_subsets(
        train,
        [p.source_train_size, p.adapt_train_size, p.intelligence_probe_count,
         p.capability_probe_count, p.metric_probe_count],
        seed, Subset, torch,
    )
    src_test = E.take_subset(test, p.source_test_size, seed + 1, Subset, torch)
    src_loader = DataLoader(src, batch_size=p.source_batch_size, shuffle=True,
                            generator=torch.Generator().manual_seed(seed), num_workers=0)
    test_loader = DataLoader(src_test, batch_size=512, shuffle=False, num_workers=0)
    adapt_loader = DataLoader(adapt, batch_size=p.adapt_batch_size, shuffle=True,
                              generator=torch.Generator().manual_seed(seed + 4), num_workers=0)
    model = E.build_components(torch, nn, F)().to(device)
    E.train_source(model, src_loader, device, p, torch, F)
    source_accuracy = E.accuracy_model(model, test_loader, device, torch, shifted=False)
    spec, _ = E.vector_spec(model)
    theta0 = E.flatten_model(model, torch).to(device)
    anchor_x, _ = next(iter(DataLoader(src, batch_size=p.anchor_count, shuffle=False)))
    dev_x, dev_y = next(iter(DataLoader(dev, batch_size=p.intelligence_probe_count, shuffle=False)))
    cap_x, cap_y = next(iter(DataLoader(cap, batch_size=p.capability_probe_count, shuffle=False)))
    metric_x, _ = next(iter(DataLoader(metric, batch_size=p.metric_probe_count, shuffle=False)))
    first_x, first_y = next(iter(adapt_loader))
    anchor_x = anchor_x.to(device)
    dev_x, dev_y = E.development_batch(dev_x.to(device), torch), dev_y.to(device)
    cap_x, cap_y = E.shifted_batch(cap_x.to(device), torch), cap_y.to(device)
    metric_x = metric_x.to(device)
    first_x, first_y = E.shifted_batch(first_x.to(device), torch), first_y.to(device)
    batches = [(E.shifted_batch(x.to(device), torch), y.to(device)) for x, y in adapt_loader]
    target_response = response_vector(theta0, model, spec, anchor_x, functional_call).detach()
    J0 = E.response_jacobian(theta0, model, spec, anchor_x, torch, functional_call)
    _, _, _, _, source_row_basis = row_space_projector(J0, p, torch)
    basis0, source_chart = initial_kernel_basis(theta0, J0, first_x, first_y, model, spec, p, seed, deps)
    raw0 = E.output_fisher_raw(theta0, basis0, model, spec, metric_x, torch, functional_call)
    G0, _ = E.regularize_metric(raw0, p.metric_eigen_floor_relative, torch)
    dev_grad0 = basis0.T @ full_gradient(theta0, model, spec, dev_x, dev_y, torch, F, functional_call)
    reverse = torch.arange(p.chart_dim - 1, -1, -1, device=device)
    perm = torch.eye(p.chart_dim, device=device)[:, reverse]
    G0_wrong = perm.T @ G0 @ perm
    source_scales = {
        "true": float(torch.sqrt((dev_grad0 @ torch.linalg.solve(G0, dev_grad0)).clamp_min(1e-24))),
        "wrong": float(torch.sqrt((dev_grad0 @ torch.linalg.solve(G0_wrong, dev_grad0)).clamp_min(1e-24))),
    }
    cert = E.metric_certification(theta0, basis0, model, spec, metric_x, raw0, p, seed, torch, functional_call)
    initial_cap = capability_loss(theta0, model, spec, cap_x, cap_y, torch, F, functional_call)
    target_loss = initial_cap * (1 - p.capability_loss_reduction_fraction)
    names = (
        "adam", "normalized_sgd", "normalized_momentum", "sign_gradient",
        "natural_gradient", "wrong_fisher_natural_gradient",
    )
    algorithms = {}
    for name in names:
        algorithms[name] = execute_online_path(
            name, theta0, basis0, target_response, batches, target_loss, model, spec,
            anchor_x, cap_x, cap_y, metric_x, dev_x, dev_y, source_scales,
            source_row_basis, p, deps, move_basis=True,
        )
    fixed_replay = execute_online_path(
        "natural_gradient", theta0, basis0, target_response, batches, target_loss, model, spec,
        anchor_x, cap_x, cap_y, metric_x, dev_x, dev_y, source_scales,
        source_row_basis, p, deps, move_basis=False,
    )
    return {
        "seed": seed,
        "source_accuracy": source_accuracy,
        "source_gate": source_accuracy >= p.source_accuracy_gate,
        "source_chart": source_chart,
        "source_chart_gate": source_chart["kernel_residual"] <= p.chart_internal_kernel_target,
        "source_metric_certification": cert,
        "source_metric_gate": metric_gate(cert, p),
        "target_capability_loss": target_loss,
        "algorithms": algorithms,
        "fixed_source_chart_natural_replay": fixed_replay,
    }


def summarize(rows, p):
    comparable = []
    exclusions = []
    for row in rows:
        bad = []
        if not row["source_gate"]:
            bad.append("source_accuracy")
        if not row["source_chart_gate"]:
            bad.append("source_chart")
        if not row["source_metric_gate"]:
            bad.append("source_metric")
        for name, result in row["algorithms"].items():
            if not result["admissible"]:
                bad.append(f"path:{name}")
        if bad:
            exclusions.append({"seed": row["seed"], "reasons": bad})
        else:
            comparable.append(row)
    counts = {
        "moving_fibre_natural_wins": 0,
        "true_natural_beats_wrong_natural": 0,
        "wrong_metric_natural_wins": 0,
        "all_paths_show_fibre_rotation": 0,
        "all_paths_show_rowspace_rotation": 0,
        "natural_meets_depth_gate": 0,
        "seed_median_depth_gate": 0,
        "fixed_replay_action_difference": 0,
        "fixed_replay_admissible": 0,
        "fixed_replay_structurally_separated": 0,
        "moving_natural_beats_fixed_replay": 0,
        "moving_fisher_length_natural_wins": 0,
        "action_length_winner_agreement": 0,
    }
    per_seed = []
    for row in comparable:
        a = row["algorithms"]
        winner = min(a, key=lambda n: a[n]["moving_fibre_action"])
        length_winner = min(a, key=lambda n: a[n]["moving_fisher_length"])
        wrong_winner = min(a, key=lambda n: a[n]["wrong_moving_fibre_action"])
        counts["moving_fibre_natural_wins"] += winner == "natural_gradient"
        counts["true_natural_beats_wrong_natural"] += (
            a["natural_gradient"]["moving_fibre_action"]
            < a["wrong_fisher_natural_gradient"]["moving_fibre_action"]
        )
        counts["wrong_metric_natural_wins"] += wrong_winner == "natural_gradient"
        counts["all_paths_show_fibre_rotation"] += all(v["fibre_rotation_signal"] for v in a.values())
        counts["all_paths_show_rowspace_rotation"] += all(v["rowspace_rotation_signal"] for v in a.values())
        counts["natural_meets_depth_gate"] += a["natural_gradient"]["steps"] >= p.minimum_natural_steps
        median_depth = float(np.median([v["steps"] for v in a.values()]))
        counts["seed_median_depth_gate"] += median_depth >= p.minimum_median_path_steps
        fixed = row["fixed_source_chart_natural_replay"]
        fixed_relative_difference = None
        if fixed["admissible"]:
            counts["fixed_replay_admissible"] += 1
            fixed_relative_difference = abs(
                a["natural_gradient"]["moving_fibre_action"] - fixed["moving_fibre_action"]
            ) / max(abs(a["natural_gradient"]["moving_fibre_action"]), 1e-12)
            differs = fixed_relative_difference >= p.fixed_replay_action_difference_gate
            counts["fixed_replay_action_difference"] += differs
            counts["fixed_replay_structurally_separated"] += differs
            counts["moving_natural_beats_fixed_replay"] += (
                a["natural_gradient"]["moving_fibre_action"] < fixed["moving_fibre_action"]
            )
        else:
            # The negative control is structurally separated if its fixed source
            # directions cease to define an admissible deep response-preserving path.
            counts["fixed_replay_structurally_separated"] += 1
        counts["moving_fisher_length_natural_wins"] += length_winner == "natural_gradient"
        counts["action_length_winner_agreement"] += winner == length_winner
        per_seed.append({
            "seed": row["seed"],
            "moving_fibre_winner": winner,
            "moving_fisher_length_winner": length_winner,
            "wrong_metric_winner": wrong_winner,
            "natural_action": a["natural_gradient"]["moving_fibre_action"],
            "wrong_natural_action": a["wrong_fisher_natural_gradient"]["moving_fibre_action"],
            "natural_max_source_angle_rad": a["natural_gradient"]["max_source_principal_angle_rad"],
            "natural_max_response_error": a["natural_gradient"]["max_response_relative_error"],
            "natural_steps": a["natural_gradient"]["steps"],
            "median_six_path_steps": median_depth,
            "natural_moving_fisher_length": a["natural_gradient"]["moving_fisher_length"],
            "fixed_replay_action": fixed["moving_fibre_action"],
            "fixed_replay_admissible": fixed["admissible"],
            "fixed_replay_relative_action_difference": fixed_relative_difference,
            "natural_max_rowspace_angle_rad": a["natural_gradient"]["max_rowspace_angle_from_source_rad"],
            "natural_max_rowspace_projector_distance": a["natural_gradient"]["max_rowspace_projector_distance_normalized"],
        })
    n = len(comparable)
    gate = bool(
        n >= p.minimum_comparable_seeds
        and counts["moving_fibre_natural_wins"] >= p.success_required
        and counts["true_natural_beats_wrong_natural"] >= p.success_required
        and counts["wrong_metric_natural_wins"] <= p.wrong_metric_natural_win_ceiling
        and counts["all_paths_show_fibre_rotation"] >= p.success_required
        and counts["all_paths_show_rowspace_rotation"] >= p.success_required
        and counts["natural_meets_depth_gate"] >= p.success_required
        and counts["seed_median_depth_gate"] >= p.success_required
        and counts["fixed_replay_structurally_separated"] >= p.fixed_replay_difference_required
    )
    if n < p.minimum_comparable_seeds:
        status = "MOVING_FIBRE_V31B_INADMISSIBLE"
    elif gate:
        status = "MOVING_FIBRE_V31B_DEPTH_CANDIDATE_SUPPORTED"
    else:
        status = "MOVING_FIBRE_V31B_DEPTH_CANDIDATE_NOT_SUPPORTED"
    return {
        "scientific_status": status,
        "seeds": len(rows),
        "fully_comparable": n,
        "required_comparable": p.minimum_comparable_seeds,
        "counts": counts,
        "moving_fibre_depth_gate": gate,
        "excluded": exclusions,
        "per_seed": per_seed,
        "claim_boundary": (
            "Eight-seed deep-path audit of six online paths whose local directions, "
            "Fisher pullbacks, and capacity dual norms are recomputed from the current state. "
            "A fixed-source-chart natural replay, pure moving-Fisher lengths, and full response-row-space "
            "rotation are reported. Newton response retraction approximates finite constrained motion. "
            "This is not a 16-seed confirmation, exact continuum horizontal lift, arbitrary-path result, "
            "GPT-2 transfer, stationarity theorem, or global variational law."
        ),
    }


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--output", type=Path, default=Path("cner_cnn_mnist_moving_fibre_depth_v3_1b_results"))
    parser.add_argument("--no-download", action="store_true")
    args, _ = parser.parse_known_args()
    p = load_protocol(args.protocol)
    deps = E.import_torch()
    torch = deps[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[preflight] device={device} torch={torch.__version__} seeds={p.seeds}", flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for index in range(p.seeds):
        seed = p.base_seed + index
        print(f"[Moving-fibre depth v3.1b seed {index + 1}/{p.seeds}] {seed}", flush=True)
        row = one_seed(seed, p, args.output, device, deps)
        rows.append(row)
        (args.output / f"seed_{seed}.json").write_text(json.dumps(row, indent=2) + "\n")
    summary = summarize(rows, p)
    result = {
        "protocol": dataclasses.asdict(p),
        "provenance": {"script_sha256": sha256(__file__)},
        "summary": summary,
        "seeds": rows,
    }
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    (args.output / "protocol.json").write_text(json.dumps(dataclasses.asdict(p), indent=2) + "\n")
    (args.output / "REPORT.md").write_text(
        "# Moving-response-fibre depth audit v3.1b\n\n```json\n"
        + json.dumps(summary, indent=2) + "\n```\n"
    )
    result_zip = args.output.parent / f"{args.output.name}.zip"
    with zipfile.ZipFile(result_zip, "w", zipfile.ZIP_DEFLATED) as archive:
        for file in args.output.rglob("*"):
            if file.is_file():
                archive.write(file, file.relative_to(args.output.parent))
    print("=" * 96)
    print(json.dumps(summary, indent=2))
    print("Results ZIP:", result_zip.resolve())
    if not args.no_download:
        try:
            from google.colab import files
            files.download(str(result_zip))
        except Exception:
            pass


if __name__ == "__main__":
    main()
