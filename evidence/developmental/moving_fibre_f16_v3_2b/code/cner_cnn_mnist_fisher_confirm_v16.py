#!/usr/bin/env python3
"""Final prospective confirmation of the output-Fisher natural-flow result.

The audit freezes an anchor-logit response Jacobian at a source checkpoint,
constructs a low-dimensional source-tangent chart, records the realized Adam
trajectory during shifted-MNIST adaptation, and compares its discrete action
using the g-norm of a prospectively frozen development-utility gradient in the
denominator. The numerator length and denominator intelligence gradient use
the same locked chart metric in both numerator and denominator. The primary
candidate is a source-frozen representation pullback; g=I is diagnostic only.

This is a restricted frozen-chart test. It is not a proof of a universal
learning law and it does not implement the moving full-space projector.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import shutil
import sys
import time
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/cner-cnn-matplotlib")

import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class Protocol:
    protocol_name: str = "CNER_CNN_MNIST_FISHER_NATURAL_CONFIRM_V16"
    seeds: int = 16
    base_seed: int = 18726
    source_train_size: int = 12000
    source_test_size: int = 2000
    source_epochs: int = 3
    source_batch_size: int = 128
    source_lr: float = 0.002
    source_accuracy_gate: float = 0.90
    anchor_count: int = 12
    intelligence_probe_count: int = 256
    chart_dim: int = 8
    rcond: float = 1e-6
    ridge: float = 1e-10
    adapt_steps: int = 240
    adapt_batch_size: int = 256
    adapt_lr: float = 0.08
    adapt_train_size: int = 4096
    adapt_test_size: int = 2000
    target_gain_gate_pp: float = 0.5
    h0: float = 0.10
    shuffle_controls: int = 3
    ordering_success_gate: int = 12
    leakage_relative_gate: float = 0.08
    kernel_residual_gate: float = 1e-5
    integration_refinement_tol: float = 0.02
    intelligence_relative_span_gate: float = 0.05
    shuffled_success_ceiling: int = 8
    mcnemar_alpha: float = 0.05
    median_effect_gate: float = -0.01
    capability_probe_count: int = 256
    capability_loss_reduction_fraction: float = 0.20
    executable_step_radius: float = 0.20
    momentum_beta: float = 0.90
    minimum_alternative_hitters: int = 2
    adam_win_gate: int = 12
    action_relative_tolerance: float = 1e-8
    primary_depth_min_median_steps: int = 5
    metric_probe_count: int = 32
    metric_eigen_floor_relative: float = 1e-3
    metric_condition_gate: float = 1e6
    metric_capacity_span_gate: float = 0.05
    metric_length_spearman_ceiling: float = 0.98
    locked_candidate_metric: str = "representation_pullback"
    diagnostic_baseline_metric: str = "identity"
    metric_cause_improvement_gate: int = 12
    cner_argmin_success_gate: int = 12
    kl_perturbations: int = 24
    kl_radius_min: float = 0.005
    kl_radius_max: float = 0.02
    kl_spearman_gate: float = 0.90
    kl_median_relative_error_gate: float = 0.25
    gauge_logit_residual_gate: float = 1e-6
    gauge_fisher_relative_gate: float = 1e-4
    representation_gauge_change_gate: float = 0.05
    natural_gradient_success_gate: int = 12
    wrong_natural_success_ceiling: int = 8
    natural_specificity_success_gate: int = 12
    wrong_metric_permutation: str = "reverse_chart_coordinates"
    integration_subdivisions_initial: int = 4
    integration_subdivisions_max: int = 32
    primary_metric: str = "output_fisher_quotient"


def canonical_json(x) -> str:
    return json.dumps(x, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_text(x: str) -> str:
    return hashlib.sha256(x.encode()).hexdigest()


def exact_binomial_upper(k: int, n: int) -> float:
    return sum(math.comb(n, j) for j in range(k, n + 1)) / (2**n)


def import_torch():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torch.func import functional_call
        from torch.utils.data import DataLoader, Dataset, Subset
        from torchvision import datasets, transforms
    except Exception as exc:
        raise RuntimeError(
            "PyTorch and torchvision are required. In Colab use the default runtime; "
            "locally install torch torchvision."
        ) from exc
    return torch, nn, F, functional_call, DataLoader, Dataset, Subset, datasets, transforms


def seed_everything(seed: int, torch) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def build_components(torch, nn, F):
    class TinyCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 4, 3, padding=1)
            self.conv2 = nn.Conv2d(4, 8, 3, padding=1)
            self.fc = nn.Linear(8 * 7 * 7, 10)

        def forward(self, x, return_features=False):
            x = F.relu(self.conv1(x)); x = F.max_pool2d(x, 2)
            x = F.relu(self.conv2(x)); x = F.max_pool2d(x, 2)
            feat = F.adaptive_avg_pool2d(x, 1).flatten(1)
            logits = self.fc(x.flatten(1))
            return (logits, feat) if return_features else logits
    return TinyCNN


def vector_spec(model):
    spec, offset = [], 0
    for name, param in model.named_parameters():
        n = param.numel(); spec.append((name, tuple(param.shape), offset, offset + n)); offset += n
    return spec, offset


def flatten_model(model, torch):
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()])


def params_from_vector(vec, spec):
    return {name: vec[a:b].view(shape) for name, shape, a, b in spec}


def shifted_batch(x, torch):
    x = torch.roll(x, shifts=(2, 1), dims=(-2, -1))
    return torch.clamp((x - 0.5) * 1.25 + 0.5, 0.0, 1.0)


def development_batch(x, torch):
    """Frozen development shift, different from the held-out target shift."""
    x = torch.roll(x, shifts=(-2, -1), dims=(-2, -1))
    return torch.clamp((x - 0.5) * 1.15 + 0.5, 0.0, 1.0)


def take_subset(dataset, n, seed, Subset, torch):
    gen = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(dataset), generator=gen)[: min(n, len(dataset))].tolist()
    return Subset(dataset, idx)


def disjoint_train_subsets(dataset, sizes, seed, Subset, torch):
    total = sum(sizes)
    if total > len(dataset):
        raise ValueError("Requested disjoint subsets exceed dataset size")
    idx = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(seed))[:total].tolist()
    out, start = [], 0
    for n in sizes:
        out.append(Subset(dataset, idx[start:start + n])); start += n
    return out


def accuracy_model(model, loader, device, torch, shifted=False):
    model.eval(); good = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if shifted: x = shifted_batch(x, torch)
            pred = model(x).argmax(1); good += int((pred == y).sum()); total += y.numel()
    return good / max(total, 1)


def accuracy_vector(theta, model, spec, loader, device, torch, functional_call, shifted=True):
    good = total = 0; params = params_from_vector(theta, spec)
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if shifted: x = shifted_batch(x, torch)
            pred = functional_call(model, params, (x,)).argmax(1)
            good += int((pred == y).sum()); total += y.numel()
    return good / max(total, 1)


def train_source(model, loader, device, p, torch, F):
    model.train(); opt = torch.optim.Adam(model.parameters(), lr=p.source_lr)
    for _ in range(p.source_epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x), y); loss.backward(); opt.step()


def response_jacobian(theta0, model, spec, anchors_x, torch, functional_call):
    def fn(v):
        return functional_call(model, params_from_vector(v, spec), (anchors_x,)).reshape(-1)
    return torch.autograd.functional.jacobian(fn, theta0, vectorize=True).detach()


def projected(v, J, rcond, torch):
    gram = J @ J.T
    coeff = torch.linalg.pinv(gram, rtol=rcond) @ (J @ v)
    return v - J.T @ coeff


def build_chart(theta0, J, target_x, target_y, model, spec, p, torch, F, functional_call, seed):
    v = theta0.detach().clone().requires_grad_(True)
    logits = functional_call(model, params_from_vector(v, spec), (target_x,))
    loss = F.cross_entropy(logits, target_y)
    grad = torch.autograd.grad(loss, v)[0].detach()
    cols = []
    first = projected(-grad, J, p.rcond, torch)
    cols.append(first / first.norm().clamp_min(1e-12))
    gen = torch.Generator(device=theta0.device).manual_seed(seed + 900001)
    attempts = 0
    while len(cols) < p.chart_dim and attempts < p.chart_dim * 20:
        attempts += 1
        q = projected(torch.randn(theta0.numel(), generator=gen, device=theta0.device), J, p.rcond, torch)
        for b in cols: q = q - b * torch.dot(b, q)
        if q.norm() > 1e-8: cols.append(q / q.norm())
    if len(cols) != p.chart_dim:
        raise RuntimeError("Could not construct requested source-tangent chart")
    B = torch.stack(cols, dim=1)
    B, _ = torch.linalg.qr(B, mode="reduced")
    return B.detach()


def intelligence_gradient(z, theta0, B, model, spec, dev_x, dev_y, torch, F, functional_call):
    w = z.detach().clone().requires_grad_(True)
    theta = theta0 + B @ w
    logits = functional_call(model, params_from_vector(theta, spec), (dev_x,))
    loss = F.cross_entropy(logits, dev_y)
    return torch.autograd.grad(loss, w)[0].detach()


def intelligence_density(z, direction, theta0, B, model, spec, dev_x, dev_y,
                         scale, p, torch, F, functional_call):
    # Same-metric scalar capacity:
    #   H_g(z) = h0 + ||grad^g A_dev(z)||_g / ||grad^g A_dev(0)||_g,
    # with A_dev=-L_dev.  In the orthonormal frozen chart g=I, so the
    # numerator and denominator are both measured by the same Euclidean g.
    # `direction` is intentionally unused: this is the conformal Riemannian
    # version, not the directional/Finsler v3 mapping.
    grad = intelligence_gradient(z, theta0, B, model, spec, dev_x, dev_y, torch, F, functional_call)
    intelligence_norm = torch.linalg.vector_norm(grad) / max(scale, 1e-12)
    return float((p.h0 + intelligence_norm).detach().cpu())


def path_action(path, theta0, B, model, spec, dev_x, dev_y, scale,
                p, torch, F, functional_call, subdivisions=1):
    total = 0.0; hs = []
    for a, b in zip(path[:-1], path[1:]):
        for j in range(subdivisions):
            aa = a + (b - a) * (j / subdivisions)
            bb = a + (b - a) * ((j + 1) / subdivisions)
            mid = 0.5 * (aa + bb)
            h = intelligence_density(mid, bb - aa, theta0, B, model, spec,
                                     dev_x, dev_y, scale, p, torch, F, functional_call)
            total += float(torch.linalg.vector_norm(bb - aa).detach().cpu()) / h
            hs.append(h)
    return total, hs


def control_paths(path, count, seed, torch):
    dz = path[1:] - path[:-1]; z0 = path[0]
    controls = {}
    controls["straight"] = torch.stack([z0 + (path[-1] - z0) * (i / (len(path) - 1)) for i in range(len(path))])
    controls["reverse"] = torch.cat([z0[None], z0[None] + torch.cumsum(torch.flip(dz, dims=[0]), dim=0)], dim=0)
    gen = torch.Generator(device=path.device).manual_seed(seed + 700003)
    for k in range(count):
        perm = torch.randperm(len(dz), generator=gen, device=path.device)
        controls[f"shuffle_{k:02d}"] = torch.cat([z0[None], z0[None] + torch.cumsum(dz[perm], dim=0)], dim=0)
    return controls


def random_direction_paths(path, count, seed, torch):
    """Matched step norms in random chart directions; each arm starts at z0."""
    dz = path[1:] - path[:-1]
    norms = torch.linalg.vector_norm(dz, dim=1)
    gen = torch.Generator(device=path.device).manual_seed(seed + 710003)
    controls = {}
    for k in range(count):
        q = torch.randn(dz.shape, generator=gen, device=path.device)
        q = q / torch.linalg.vector_norm(q, dim=1, keepdim=True).clamp_min(1e-12)
        matched = q * norms[:, None]
        controls[f"random_direction_{k:02d}"] = torch.cat(
            [path[0][None], path[0][None] + torch.cumsum(matched, dim=0)], dim=0
        )
    return controls


def path_response_leakage(path, theta0, B, model, spec, anchor_x, r0, torch, functional_call):
    values = []
    denom = torch.linalg.vector_norm(r0).clamp_min(1e-12)
    with torch.no_grad():
        for z in path:
            theta = theta0 + B @ z
            r = functional_call(model, params_from_vector(theta, spec), (anchor_x,)).reshape(-1)
            values.append(float((torch.linalg.vector_norm(r - r0) / denom).cpu()))
    return {
        "max": max(values),
        "median": float(np.median(values)),
        "final": values[-1],
    }


def adapt_in_chart(theta0, B, model, spec, batches, p, torch, F, functional_call):
    z = torch.zeros(B.shape[1], device=theta0.device, requires_grad=True)
    opt = torch.optim.Adam([z], lr=p.adapt_lr); path = [z.detach().clone()]
    for step in range(p.adapt_steps):
        x, y = batches[step % len(batches)]
        opt.zero_grad(set_to_none=True)
        theta = theta0 + B @ z
        logits = functional_call(model, params_from_vector(theta, spec), (x,))
        loss = F.cross_entropy(logits, y); loss.backward(); opt.step()
        path.append(z.detach().clone())
    return torch.stack(path)


def one_seed(seed, p, outdir, device, deps):
    torch, nn, F, functional_call, DataLoader, Dataset, Subset, datasets, transforms = deps
    seed_everything(seed, torch)
    transform = transforms.ToTensor()
    root = str(outdir.parent / "mnist_data")
    train_ds = datasets.MNIST(root, train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(root, train=False, download=True, transform=transform)
    src_train, adapt_train, intelligence_dev = disjoint_train_subsets(
        train_ds,
        [p.source_train_size, p.adapt_train_size, p.intelligence_probe_count],
        seed,
        Subset,
        torch,
    )
    src_test = take_subset(test_ds, p.source_test_size, seed + 1, Subset, torch)
    adapt_test = take_subset(test_ds, p.adapt_test_size, seed + 3, Subset, torch)
    src_loader = DataLoader(src_train, batch_size=p.source_batch_size, shuffle=True,
                            generator=torch.Generator().manual_seed(seed), num_workers=0)
    src_test_loader = DataLoader(src_test, batch_size=512, shuffle=False, num_workers=0)
    adapt_loader = DataLoader(adapt_train, batch_size=p.adapt_batch_size, shuffle=True,
                              generator=torch.Generator().manual_seed(seed + 4), num_workers=0)
    adapt_test_loader = DataLoader(adapt_test, batch_size=512, shuffle=False, num_workers=0)

    TinyCNN = build_components(torch, nn, F); model = TinyCNN().to(device)
    train_source(model, src_loader, device, p, torch, F)
    source_acc = accuracy_model(model, src_test_loader, device, torch, shifted=False)
    pre_acc = accuracy_model(model, adapt_test_loader, device, torch, shifted=True)
    spec, nparam = vector_spec(model); theta0 = flatten_model(model, torch).to(device)

    anchor_x, _ = next(iter(DataLoader(src_train, batch_size=p.anchor_count, shuffle=False)))
    dev_x, dev_y = next(iter(DataLoader(intelligence_dev, batch_size=p.intelligence_probe_count, shuffle=False)))
    tx, ty = next(iter(adapt_loader)); anchor_x = anchor_x.to(device)
    dev_x, dev_y = development_batch(dev_x.to(device), torch), dev_y.to(device)
    label_gen = torch.Generator(device=device).manual_seed(seed + 880003)
    dev_y_shuffled = dev_y[torch.randperm(dev_y.numel(), generator=label_gen, device=device)]
    tx, ty = shifted_batch(tx.to(device), torch), ty.to(device)
    batches = [(shifted_batch(x.to(device), torch), y.to(device)) for x, y in adapt_loader]

    J = response_jacobian(theta0, model, spec, anchor_x, torch, functional_call)
    svals = torch.linalg.svdvals(J)
    ranks = {str(rc): int((svals > rc * svals.max()).sum()) for rc in (1e-5, 1e-6, 1e-7)}
    B = build_chart(theta0, J, tx, ty, model, spec, p, torch, F, functional_call, seed)
    kernel_residual = float(torch.linalg.norm(J @ B) / torch.linalg.norm(J).clamp_min(1e-12))

    path = adapt_in_chart(theta0, B, model, spec, batches, p, torch, F, functional_call)
    final_theta = theta0 + B @ path[-1]
    post_acc = accuracy_vector(final_theta, model, spec, adapt_test_loader, device, torch, functional_call, shifted=True)
    with torch.no_grad():
        r0 = functional_call(model, params_from_vector(theta0, spec), (anchor_x,)).reshape(-1)

    z0 = torch.zeros(B.shape[1], device=device)
    true_scale = float(torch.linalg.vector_norm(intelligence_gradient(
        z0, theta0, B, model, spec, dev_x, dev_y, torch, F, functional_call
    )).cpu())
    shuffled_scale = float(torch.linalg.vector_norm(intelligence_gradient(
        z0, theta0, B, model, spec, dev_x, dev_y_shuffled, torch, F, functional_call
    )).cpu())
    actual_action, actual_h = path_action(path, theta0, B, model, spec, dev_x, dev_y,
                                           true_scale, p, torch, F, functional_call, 1)
    refined_action, _ = path_action(path, theta0, B, model, spec, dev_x, dev_y,
                                     true_scale, p, torch, F, functional_call, 2)
    actual_action_shuffled_map, _ = path_action(
        path, theta0, B, model, spec, dev_x, dev_y_shuffled,
        shuffled_scale, p, torch, F, functional_call, 1
    )
    controls = control_paths(path, p.shuffle_controls, seed, torch)
    control_actions = {}
    shuffled_map_control_actions = {}
    control_leakage = {}
    for name, cp in controls.items():
        control_actions[name] = path_action(
            cp, theta0, B, model, spec, dev_x, dev_y, true_scale,
            p, torch, F, functional_call, 1
        )[0]
        shuffled_map_control_actions[name] = path_action(
            cp, theta0, B, model, spec, dev_x, dev_y_shuffled, shuffled_scale,
            p, torch, F, functional_call, 1
        )[0]
        control_leakage[name] = path_response_leakage(cp, theta0, B, model, spec, anchor_x, r0, torch, functional_call)
    actual_leakage = path_response_leakage(path, theta0, B, model, spec, anchor_x, r0, torch, functional_call)
    direction_controls = random_direction_paths(path, p.random_direction_controls, seed, torch)
    direction_control_accuracy = {}
    direction_control_leakage = {}
    for name, cp in direction_controls.items():
        theta_cp = theta0 + B @ cp[-1]
        direction_control_accuracy[name] = accuracy_vector(
            theta_cp, model, spec, adapt_test_loader, device, torch, functional_call, shifted=True
        )
        direction_control_leakage[name] = path_response_leakage(
            cp, theta0, B, model, spec, anchor_x, r0, torch, functional_call
        )
    admissible_direction_names = [
        name for name in direction_controls
        if direction_control_leakage[name]["max"] <= p.leakage_relative_gate
    ]
    direction_controls_admissible = len(admissible_direction_names) == p.random_direction_controls
    median_random_direction_accuracy = (
        float(np.median([direction_control_accuracy[name] for name in admissible_direction_names]))
        if admissible_direction_names else None
    )
    functional_margin_pp = (
        100 * (post_acc - median_random_direction_accuracy)
        if median_random_direction_accuracy is not None else None
    )
    functional_value_success = bool(
        direction_controls_admissible and functional_margin_pp is not None
        and functional_margin_pp >= p.functional_margin_gate_pp
    )
    admissible_shuffle_names = [k for k in control_actions if k.startswith("shuffle_") and control_leakage[k]["max"] <= p.leakage_relative_gate]
    admissible_shuffle_vals = [control_actions[k] for k in admissible_shuffle_names]
    median_shuffle = float(np.median(admissible_shuffle_vals)) if admissible_shuffle_vals else None
    shuffled_map_shuffle_vals = [shuffled_map_control_actions[k] for k in admissible_shuffle_names]
    shuffled_map_median_shuffle = float(np.median(shuffled_map_shuffle_vals)) if shuffled_map_shuffle_vals else None
    temporal_controls_admissible = (
        control_leakage["reverse"]["max"] <= p.leakage_relative_gate
        and len(admissible_shuffle_names) == p.shuffle_controls
    )
    straight_admissible = control_leakage["straight"]["max"] <= p.leakage_relative_gate
    temporal_success = bool(
        temporal_controls_admissible and median_shuffle is not None
        and actual_action < control_actions["reverse"] and actual_action < median_shuffle
    )
    shuffled_map_temporal_success = bool(
        temporal_controls_admissible and shuffled_map_median_shuffle is not None
        and actual_action_shuffled_map < shuffled_map_control_actions["reverse"]
        and actual_action_shuffled_map < shuffled_map_median_shuffle
    )
    straight_success = bool(straight_admissible and actual_action < control_actions["straight"])
    strong_success = temporal_success and straight_success
    h_span = (max(actual_h) - min(actual_h)) / max(float(np.median(actual_h)), 1e-12)

    return {
        "seed": seed, "device": str(device), "parameter_count": nparam,
        "source_accuracy": source_acc, "target_pre_accuracy": pre_acc,
        "target_post_accuracy": post_acc, "target_gain_pp": 100 * (post_acc - pre_acc),
        "jacobian_shape": list(J.shape), "rank_by_rcond": ranks,
        "kernel_residual": kernel_residual,
        "actual_path_leakage_max": actual_leakage["max"],
        "actual_path_leakage_median": actual_leakage["median"],
        "actual_path_leakage_final": actual_leakage["final"],
        "actual_action": actual_action, "actual_action_refined": refined_action,
        "actual_action_shuffled_intelligence_map": actual_action_shuffled_map,
        "integration_relative_change": abs(refined_action - actual_action) / max(abs(refined_action), 1e-12),
        "straight_action": control_actions["straight"], "reverse_action": control_actions["reverse"],
        "median_admissible_shuffle_action": median_shuffle,
        "shuffled_map_reverse_action": shuffled_map_control_actions["reverse"],
        "shuffled_map_median_admissible_shuffle_action": shuffled_map_median_shuffle,
        "shuffle_actions": {k: control_actions[k] for k in control_actions if k.startswith("shuffle_")},
        "control_path_leakage": control_leakage,
        "admissible_shuffle_count": len(admissible_shuffle_names),
        "temporal_controls_admissible": temporal_controls_admissible,
        "straight_control_admissible": straight_admissible,
        "temporal_order_success": temporal_success,
        "shuffled_intelligence_temporal_success": shuffled_map_temporal_success,
        "straight_competition_success": straight_success,
        "strong_joint_success": strong_success,
        "random_direction_control_accuracy": direction_control_accuracy,
        "random_direction_control_leakage": direction_control_leakage,
        "admissible_random_direction_count": len(admissible_direction_names),
        "direction_controls_admissible": direction_controls_admissible,
        "median_random_direction_accuracy": median_random_direction_accuracy,
        "functional_value_margin_pp": functional_margin_pp,
        "functional_value_success": functional_value_success,
        "euclidean_actual_length": float(torch.linalg.vector_norm(path[1:] - path[:-1], dim=1).sum().cpu()),
        "euclidean_straight_length": float(torch.linalg.vector_norm(path[-1] - path[0]).cpu()),
        "intelligence_density_min": min(actual_h),
        "intelligence_density_median": float(np.median(actual_h)),
        "intelligence_density_max": max(actual_h),
        "intelligence_relative_span": h_span,
        "intelligence_variation_gate": h_span >= p.intelligence_relative_span_gate,
        "true_intelligence_scale": true_scale,
        "shuffled_intelligence_scale": shuffled_scale,
        "source_gate": source_acc >= p.source_accuracy_gate,
        "gain_gate": 100 * (post_acc - pre_acc) >= p.target_gain_gate_pp,
        "kernel_gate": kernel_residual <= p.kernel_residual_gate,
        "leakage_gate": actual_leakage["max"] <= p.leakage_relative_gate,
        "integration_gate": abs(refined_action - actual_action) / max(abs(refined_action), 1e-12) <= p.integration_refinement_tol,
    }


def run(p, outdir: Path):
    deps = import_torch(); torch = deps[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    outdir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(p.seeds):
        seed = p.base_seed + i
        print(f"[seed {i+1}/{p.seeds}] {seed}", flush=True)
        row = one_seed(seed, p, outdir, device, deps); rows.append(row)
        print(f"  source={row['source_accuracy']:.4f} pre={row['target_pre_accuracy']:.4f} post={row['target_post_accuracy']:.4f} "
              f"T={row['actual_action']:.4g} controls=({row['straight_action']:.4g},{row['reverse_action']:.4g},{row['median_admissible_shuffle_action']}) "
              f"temporal={row['temporal_order_success']} shuffled-map={row['shuffled_intelligence_temporal_success']} "
              f"functional={row['functional_value_success']} margin_pp={row['functional_value_margin_pp']} "
              f"straight={row['straight_competition_success']} "
              f"leak_max={row['actual_path_leakage_max']:.3g}", flush=True)

    valid = [r for r in rows if r["source_gate"] and r["gain_gate"] and r["kernel_gate"] and r["leakage_gate"] and r["integration_gate"]]
    temporal_valid = [r for r in valid if r["temporal_controls_admissible"]]
    straight_valid = [r for r in valid if r["straight_control_admissible"]]
    strong_valid = [r for r in valid if r["temporal_controls_admissible"] and r["straight_control_admissible"]]
    temporal_k = sum(r["temporal_order_success"] for r in temporal_valid)
    shuffled_map_k = sum(r["shuffled_intelligence_temporal_success"] for r in temporal_valid)
    straight_k = sum(r["straight_competition_success"] for r in straight_valid)
    strong_k = sum(r["strong_joint_success"] for r in strong_valid)
    n = len(valid)
    all_admissible = n == p.seeds
    temporal_comparable = len(temporal_valid) == p.seeds
    functional_valid = [r for r in valid if r["direction_controls_admissible"]]
    functional_k = sum(r["functional_value_success"] for r in functional_valid)
    functional_comparable = len(functional_valid) == p.seeds
    functional_median_margin = float(np.median([
        r["functional_value_margin_pp"] for r in functional_valid
    ])) if functional_valid else None
    functional_pass = (
        all_admissible and functional_comparable
        and functional_k >= p.functional_success_gate
        and functional_median_margin is not None
        and functional_median_margin >= p.functional_margin_gate_pp
    )
    straight_comparable = len(straight_valid) == p.seeds
    true_only = sum(
        r["temporal_order_success"] and not r["shuffled_intelligence_temporal_success"]
        for r in temporal_valid
    )
    shuffled_only = sum(
        r["shuffled_intelligence_temporal_success"] and not r["temporal_order_success"]
        for r in temporal_valid
    )
    discordant = true_only + shuffled_only
    mcnemar_p = exact_binomial_upper(true_only, discordant) if discordant else 1.0
    effects = [r["actual_action"] - r["median_admissible_shuffle_action"] for r in temporal_valid]
    median_effect = float(np.median(effects)) if effects else None
    temporal_count_pass = temporal_k >= p.ordering_success_gate
    shuffled_ceiling_pass = shuffled_map_k <= p.shuffled_success_ceiling
    mcnemar_pass = mcnemar_p < p.mcnemar_alpha
    effect_pass = median_effect is not None and median_effect <= p.median_effect_gate
    confirmatory_pass = (
        all_admissible and temporal_comparable and temporal_count_pass
        and shuffled_ceiling_pass and mcnemar_pass and effect_pass
    )
    if functional_pass and confirmatory_pass:
        status = "TANGENT_FUNCTIONAL_VALUE_AND_CNER_BOTH_SUPPORTED"
    elif functional_pass and not confirmatory_pass:
        status = "TANGENT_FUNCTIONAL_VALUE_SUPPORTED_CNER_NOT_SUPPORTED"
    elif confirmatory_pass and not functional_pass:
        status = "CNER_SUPPORTED_TANGENT_FUNCTIONAL_VALUE_NOT_SUPPORTED"
    elif all_admissible and temporal_comparable and functional_comparable:
        status = "TANGENT_FUNCTIONAL_VALUE_AND_CNER_BOTH_NOT_SUPPORTED"
    else:
        status = "DUAL_BRIDGE_INADMISSIBLE_OR_NONCOMPARABLE"
    result = {
        "scientific_status": status,
        "scope": "TinyCNN/MNIST dual bridge: matched-norm tangent functional value and same-metric CNER temporal ordering in one frozen source-tangent chart; not an exact paper reproduction or universal-law certificate.",
        "protocol": asdict(p), "protocol_sha256": sha256_text(canonical_json(asdict(p))),
        "summary": {
            "independent_seeds": p.seeds, "admissible_actual_paths": n,
            "temporal_comparable_seeds": len(temporal_valid), "temporal_order_successes": temporal_k,
            "temporal_exact_one_sided_binomial_p": exact_binomial_upper(temporal_k, len(temporal_valid)) if temporal_valid else None,
            "shuffled_intelligence_temporal_successes": shuffled_map_k,
            "shuffled_intelligence_exact_one_sided_binomial_p": exact_binomial_upper(
                shuffled_map_k, len(temporal_valid)
            ) if temporal_valid else None,
            "true_only_successes": true_only, "shuffled_only_successes": shuffled_only,
            "discordant_pairs": discordant, "mcnemar_exact_one_sided_p": mcnemar_p,
            "temporal_count_gate_pass": temporal_count_pass,
            "shuffled_success_ceiling_gate_pass": shuffled_ceiling_pass,
            "mcnemar_gate_pass": mcnemar_pass, "median_effect_gate_pass": effect_pass,
            "confirmatory_gate_pass": confirmatory_pass,
            "functional_comparable_seeds": len(functional_valid),
            "functional_value_successes": functional_k,
            "functional_value_exact_one_sided_binomial_p": exact_binomial_upper(
                functional_k, len(functional_valid)
            ) if functional_valid else None,
            "median_actual_minus_random_direction_accuracy_pp": functional_median_margin,
            "functional_value_gate_pass": functional_pass,
            "straight_comparable_seeds": len(straight_valid), "straight_successes": straight_k,
            "straight_exact_one_sided_binomial_p": exact_binomial_upper(straight_k, len(straight_valid)) if straight_valid else None,
            "strong_joint_comparable_seeds": len(strong_valid), "strong_joint_successes": strong_k,
            "all_actual_paths_admissible": all_admissible,
            "straight_result_is_secondary": True,
            "median_target_gain_pp": float(np.median([r["target_gain_pp"] for r in rows])),
            "median_actual_minus_admissible_shuffle_action": median_effect,
            "median_intelligence_relative_span": float(np.median([r["intelligence_relative_span"] for r in rows])),
            "intelligence_variation_gate_passes": sum(r["intelligence_variation_gate"] for r in rows),
        },
        "seeds": rows,
        "interpretation_boundary": [
            "R, the anchor set, development-utility probes, metric g=I, rcond, gates and controls are frozen by protocol.",
            "Both numerator path length and denominator development-utility-gradient norm use the same chart metric g=I.",
            "The denominator is the local maximum development-utility rate per unit g-motion, normalized at the source checkpoint.",
            "Development shift and target shift are distinct, and their training samples are disjoint.",
            "A fixed shuffled-development-label mapping is the negative control; success without specificity is not support.",
            "The chart lies in the source-checkpoint kernel; the full moving projector is not recomputed along the path.",
            "Same-endpoint step permutations test temporal path ordering, not global geodesic optimality.",
            "Every control is compared only after its full path satisfies the same response-leakage gate.",
            "The tangent-value gate and the CNER gate are evaluated separately on the same new seeds; neither may substitute for the other.",
            "The tangent-value arm matches every realized chart-step norm but randomizes its direction; this is a restricted bridge to H2-F, not an exact reproduction of the paper's full normal-plus-tangent intervention.",
            "The functional-value gate is >=12/16 seed wins with a median accuracy margin >=5 pp over eight within-seed random-direction controls.",
            "The CNER gate remains true >=12, shuffled <=8, paired exact one-sided McNemar p<0.05, and median actual-minus-shuffle action <=-0.01.",
            "Straight-path competition is secondary because the post-hoc straight curve need not belong to the causal online path class.",
            "A failed gate is retained and cannot be overridden by a pooled p-value."
        ],
        "runtime": {"python": sys.version, "torch": torch.__version__, "numpy": np.__version__,
                    "platform": platform.platform(), "device": str(device), "unix_time": time.time()},
    }
    (outdir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    flat_keys = [k for k, v in rows[0].items() if isinstance(v, (str, int, float, bool))]
    with (outdir / "seed_summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=flat_keys); w.writeheader(); w.writerows({k: r[k] for k in flat_keys} for r in rows)
    make_plot(rows, outdir / "action_comparison.png")
    make_report(result, outdir / "REPORT.md")
    return result


def loss_at_z(z, theta0, B, model, spec, x, y, torch, F, functional_call):
    with torch.no_grad():
        theta = theta0 + B @ z
        logits = functional_call(model, params_from_vector(theta, spec), (x,))
        return float(F.cross_entropy(logits, y).cpu())


def executable_path(name, theta0, B, model, spec, batches, p, torch, F, functional_call):
    z = torch.zeros(B.shape[1], device=theta0.device)
    path = [z.clone()]
    momentum = torch.zeros_like(z)
    if name == "adam":
        z = z.requires_grad_(True)
        opt = torch.optim.Adam([z], lr=p.adapt_lr)
    for step in range(p.adapt_steps):
        x, y = batches[step % len(batches)]
        if name == "adam":
            opt.zero_grad(set_to_none=True)
            theta = theta0 + B @ z
            loss = F.cross_entropy(functional_call(model, params_from_vector(theta, spec), (x,)), y)
            loss.backward(); opt.step()
            path.append(z.detach().clone())
            continue
        w = z.detach().clone().requires_grad_(True)
        theta = theta0 + B @ w
        loss = F.cross_entropy(functional_call(model, params_from_vector(theta, spec), (x,)), y)
        grad = torch.autograd.grad(loss, w)[0].detach()
        if name == "normalized_sgd":
            direction = grad / grad.norm().clamp_min(1e-12)
        elif name == "normalized_momentum":
            momentum = p.momentum_beta * momentum + (1.0 - p.momentum_beta) * grad
            direction = momentum / momentum.norm().clamp_min(1e-12)
        elif name == "sign_gradient":
            direction = torch.sign(grad) / math.sqrt(grad.numel())
        else:
            raise ValueError(f"Unknown executable algorithm: {name}")
        z = (w.detach() - p.executable_step_radius * direction).detach()
        path.append(z.clone())
    return torch.stack(path)


def truncate_at_capability(path, target_loss, theta0, B, model, spec, x, y,
                           torch, F, functional_call):
    losses = [loss_at_z(z, theta0, B, model, spec, x, y, torch, F, functional_call) for z in path]
    hit = next((i for i, value in enumerate(losses) if value <= target_loss), None)
    if hit is None:
        return path, False, losses[-1], len(path) - 1
    if hit == 0:
        return path[:1], True, losses[0], 0
    lo, hi = path[hit - 1].clone(), path[hit].clone()
    for _ in range(24):
        mid = 0.5 * (lo + hi)
        value = loss_at_z(mid, theta0, B, model, spec, x, y, torch, F, functional_call)
        if value <= target_loss: hi = mid
        else: lo = mid
    clipped = torch.cat([path[:hit], hi[None]], dim=0)
    final_loss = loss_at_z(hi, theta0, B, model, spec, x, y, torch, F, functional_call)
    return clipped, True, final_loss, hit


def one_seed_m5a(seed, p, outdir, device, deps):
    torch, nn, F, functional_call, DataLoader, Dataset, Subset, datasets, transforms = deps
    seed_everything(seed, torch)
    root = str(outdir.parent / "mnist_data")
    train_ds = datasets.MNIST(root, train=True, download=True, transform=transforms.ToTensor())
    test_ds = datasets.MNIST(root, train=False, download=True, transform=transforms.ToTensor())
    src_train, adapt_train, intelligence_dev, capability_probe = disjoint_train_subsets(
        train_ds, [p.source_train_size, p.adapt_train_size, p.intelligence_probe_count,
                   p.capability_probe_count], seed, Subset, torch
    )
    src_test = take_subset(test_ds, p.source_test_size, seed + 1, Subset, torch)
    adapt_test = take_subset(test_ds, p.adapt_test_size, seed + 3, Subset, torch)
    src_loader = DataLoader(src_train, batch_size=p.source_batch_size, shuffle=True,
                            generator=torch.Generator().manual_seed(seed), num_workers=0)
    src_test_loader = DataLoader(src_test, batch_size=512, shuffle=False, num_workers=0)
    adapt_loader = DataLoader(adapt_train, batch_size=p.adapt_batch_size, shuffle=True,
                              generator=torch.Generator().manual_seed(seed + 4), num_workers=0)
    adapt_test_loader = DataLoader(adapt_test, batch_size=512, shuffle=False, num_workers=0)
    TinyCNN = build_components(torch, nn, F); model = TinyCNN().to(device)
    train_source(model, src_loader, device, p, torch, F)
    source_acc = accuracy_model(model, src_test_loader, device, torch, shifted=False)
    pre_acc = accuracy_model(model, adapt_test_loader, device, torch, shifted=True)
    spec, nparam = vector_spec(model); theta0 = flatten_model(model, torch).to(device)
    anchor_x, _ = next(iter(DataLoader(src_train, batch_size=p.anchor_count, shuffle=False)))
    dev_x, dev_y = next(iter(DataLoader(intelligence_dev, batch_size=p.intelligence_probe_count, shuffle=False)))
    cap_x, cap_y = next(iter(DataLoader(capability_probe, batch_size=p.capability_probe_count, shuffle=False)))
    tx, ty = next(iter(adapt_loader))
    anchor_x = anchor_x.to(device)
    dev_x, dev_y = development_batch(dev_x.to(device), torch), dev_y.to(device)
    cap_x, cap_y = shifted_batch(cap_x.to(device), torch), cap_y.to(device)
    tx, ty = shifted_batch(tx.to(device), torch), ty.to(device)
    batches = [(shifted_batch(x.to(device), torch), y.to(device)) for x, y in adapt_loader]
    J = response_jacobian(theta0, model, spec, anchor_x, torch, functional_call)
    B = build_chart(theta0, J, tx, ty, model, spec, p, torch, F, functional_call, seed)
    kernel_residual = float(torch.linalg.norm(J @ B) / torch.linalg.norm(J).clamp_min(1e-12))
    with torch.no_grad():
        r0 = functional_call(model, params_from_vector(theta0, spec), (anchor_x,)).reshape(-1)
    z0 = torch.zeros(B.shape[1], device=device)
    scale = float(torch.linalg.vector_norm(intelligence_gradient(
        z0, theta0, B, model, spec, dev_x, dev_y, torch, F, functional_call
    )).cpu())
    initial_cap_loss = loss_at_z(z0, theta0, B, model, spec, cap_x, cap_y, torch, F, functional_call)
    target_cap_loss = initial_cap_loss * (1.0 - p.capability_loss_reduction_fraction)
    names = ("adam", "normalized_sgd", "normalized_momentum", "sign_gradient")
    algorithms = {}
    for name in names:
        raw_path = executable_path(name, theta0, B, model, spec, batches, p, torch, F, functional_call)
        path, hit, cap_loss, hit_step = truncate_at_capability(
            raw_path, target_cap_loss, theta0, B, model, spec, cap_x, cap_y,
            torch, F, functional_call
        )
        leakage = path_response_leakage(path, theta0, B, model, spec, anchor_x, r0, torch, functional_call)
        action, hs = path_action(path, theta0, B, model, spec, dev_x, dev_y, scale,
                                  p, torch, F, functional_call, 1)
        refined, _ = path_action(path, theta0, B, model, spec, dev_x, dev_y, scale,
                                  p, torch, F, functional_call, 2)
        integration_rel = abs(refined - action) / max(abs(refined), 1e-12)
        final_theta = theta0 + B @ path[-1]
        test_acc = accuracy_vector(final_theta, model, spec, adapt_test_loader, device,
                                   torch, functional_call, shifted=True)
        admissible = bool(hit and leakage["max"] <= p.leakage_relative_gate
                          and integration_rel <= p.integration_refinement_tol)
        algorithms[name] = {
            "hit_capability": hit, "hit_step": hit_step,
            "capability_final_loss": cap_loss, "capability_target_loss": target_cap_loss,
            "capability_relative_error": abs(cap_loss - target_cap_loss) / max(target_cap_loss, 1e-12),
            "path_points": len(path), "action": action, "action_refined": refined,
            "integration_relative_change": integration_rel,
            "leakage_max": leakage["max"], "test_accuracy": test_acc,
            "test_gain_pp": 100 * (test_acc - pre_acc), "admissible": admissible,
            "path_length": float(torch.linalg.vector_norm(path[1:] - path[:-1], dim=1).sum().cpu()),
            "capacity_density_min": min(hs), "capacity_density_max": max(hs),
        }
        algorithms[name]["effective_capacity_length_over_action"] = (
            algorithms[name]["path_length"] / max(algorithms[name]["action"], 1e-12)
        )
    alternatives = [(n, algorithms[n]) for n in names[1:] if algorithms[n]["admissible"]]
    comparable = algorithms["adam"]["admissible"] and len(alternatives) >= p.minimum_alternative_hitters
    best_alt_name, best_alt = min(alternatives, key=lambda item: item[1]["action"]) if alternatives else (None, None)
    best_alt_action = best_alt["action"] if best_alt is not None else None
    adam_win = bool(comparable and algorithms["adam"]["action"] <=
                    best_alt_action * (1.0 + p.action_relative_tolerance))
    return {
        "seed": seed, "device": str(device), "parameter_count": nparam,
        "source_accuracy": source_acc, "target_pre_accuracy": pre_acc,
        "kernel_residual": kernel_residual, "initial_capability_loss": initial_cap_loss,
        "target_capability_loss": target_cap_loss, "algorithms": algorithms,
        "admissible_alternative_count": len(alternatives), "comparable_seed": comparable,
        "best_alternative_action": best_alt_action, "adam_action": algorithms["adam"]["action"],
        "best_alternative_name": best_alt_name,
        "adam_to_best_length_ratio": (
            algorithms["adam"]["path_length"] / max(best_alt["path_length"], 1e-12)
            if best_alt is not None else None
        ),
        "adam_to_best_effective_capacity_ratio": (
            algorithms["adam"]["effective_capacity_length_over_action"] /
            max(best_alt["effective_capacity_length_over_action"], 1e-12)
            if best_alt is not None else None
        ),
        "adam_minus_best_alternative_action": (
            algorithms["adam"]["action"] - best_alt_action if best_alt_action is not None else None
        ),
        "adam_executable_argmin_success": adam_win,
        "source_gate": source_acc >= p.source_accuracy_gate,
        "kernel_gate": kernel_residual <= p.kernel_residual_gate,
    }


def run_m5a(p, outdir: Path):
    deps = import_torch(); torch = deps[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    outdir.mkdir(parents=True, exist_ok=True); rows = []
    for i in range(p.seeds):
        seed = p.base_seed + i
        print(f"[seed {i+1}/{p.seeds}] {seed}", flush=True)
        row = one_seed_m5a(seed, p, outdir, device, deps); rows.append(row)
        alg = row["algorithms"]
        print("  " + " ".join(f"{n}:hit={alg[n]['hit_capability']},adm={alg[n]['admissible']},T={alg[n]['action']:.4g}" for n in alg)
              + f" adam_win={row['adam_executable_argmin_success']}", flush=True)
    valid = [r for r in rows if r["source_gate"] and r["kernel_gate"]]
    comparable = [r for r in valid if r["comparable_seed"]]
    wins = sum(r["adam_executable_argmin_success"] for r in comparable)
    diffs = [r["adam_minus_best_alternative_action"] for r in comparable]
    hit_steps = [r["algorithms"]["adam"]["hit_step"] for r in comparable]
    median_hit_step = float(np.median(hit_steps)) if hit_steps else None
    depth_pass = median_hit_step is not None and median_hit_step >= p.primary_depth_min_median_steps
    length_ratios = [r["adam_to_best_length_ratio"] for r in comparable]
    capacity_ratios = [r["adam_to_best_effective_capacity_ratio"] for r in comparable]
    all_comparable = len(comparable) == p.seeds
    gate = (all_comparable and depth_pass and wins >= p.adam_win_gate
            and float(np.median(diffs)) <= 0.0)
    pairwise = {}
    for name in ("normalized_sgd", "normalized_momentum", "sign_gradient"):
        pairs = [r for r in valid if r["algorithms"]["adam"]["admissible"] and r["algorithms"][name]["admissible"]]
        k = sum(r["algorithms"]["adam"]["action"] <= r["algorithms"][name]["action"] *
                (1.0 + p.action_relative_tolerance) for r in pairs)
        pairwise[name] = {"comparable": len(pairs), "adam_wins": k,
                          "exact_one_sided_binomial_p": exact_binomial_upper(k, len(pairs)) if pairs else None}
    status = ("M5A_30PCT_CNER_COHORT_ARGMIN_CONFIRMED" if gate else
              "M5A_30PCT_CNER_COHORT_ARGMIN_REFUTED" if all_comparable and depth_pass else
              "M5A_30PCT_CONFIRMATION_INADMISSIBLE_OR_TOO_SHALLOW")
    result = {
        "scientific_status": status,
        "scope": "Prospective TinyCNN/MNIST confirmation at one frozen 30% capability-loss reduction target over four executable optimizer paths in a frozen tangent chart; not global and not optimizer-derived Hcap.",
        "protocol": asdict(p), "protocol_sha256": sha256_text(canonical_json(asdict(p))),
        "summary": {"independent_seeds": p.seeds, "valid_source_chart_seeds": len(valid),
                    "comparable_seeds": len(comparable), "adam_best_action_successes": wins,
                    "exact_one_sided_binomial_p": exact_binomial_upper(wins, len(comparable)) if comparable else None,
                    "median_adam_minus_best_alternative_action": float(np.median(diffs)) if diffs else None,
                    "median_adam_hit_step": median_hit_step,
                    "primary_depth_gate_pass": depth_pass,
                    "median_adam_to_best_length_ratio": float(np.median(length_ratios)) if length_ratios else None,
                    "median_adam_to_best_effective_capacity_ratio": float(np.median(capacity_ratios)) if capacity_ratios else None,
                    "all_seeds_comparable": all_comparable, "primary_gate_pass": gate,
                    "pairwise": pairwise},
        "seeds": rows,
        "interpretation_boundary": [
            "Every arm is an actually executed gradient-based algorithm, not a permutation of recorded increments.",
            "All actions are truncated by bisection at the same frozen 30% capability-probe loss reduction target.",
            "The primary depth gate requires Adam's median target-hit step to be at least five.",
            "The capability probe, capacity-development probe, adaptation batches, and final test set are disjoint.",
            "The primary comparison asks whether Adam has no larger action than the best admissible alternative in each seed.",
            "A cohort minimum among four algorithms is not a global argmin over all causal paths or all optimizers.",
            "Hcap remains the prospectively declared same-metric development-utility gradient density; deriving it from optimizer dynamics is reserved for M5b.",
            "Failed, inadmissible, and non-hitting algorithms are retained and never pooled into support."
        ],
        "runtime": {"python": sys.version, "torch": torch.__version__, "numpy": np.__version__,
                    "platform": platform.platform(), "device": str(device), "unix_time": time.time()},
    }
    (outdir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with (outdir / "seed_summary.csv").open("w", newline="", encoding="utf-8") as f:
        fields = ["seed", "source_accuracy", "kernel_residual", "comparable_seed",
                  "admissible_alternative_count", "adam_action", "best_alternative_action",
                  "best_alternative_name", "adam_minus_best_alternative_action",
                  "adam_to_best_length_ratio", "adam_to_best_effective_capacity_ratio",
                  "adam_executable_argmin_success"]
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows({k:r[k] for k in fields} for r in rows)
    make_plot_m5a(rows, outdir / "executable_action_comparison.png")
    make_report_m5a(result, outdir / "REPORT.md")
    return result


def make_plot_m5a(rows, target):
    names = ("adam", "normalized_sgd", "normalized_momentum", "sign_gradient")
    x = np.arange(len(rows)); width = 0.2
    fig, ax = plt.subplots(figsize=(12, 5))
    for j, name in enumerate(names):
        ax.bar(x + (j - 1.5) * width, [r["algorithms"][name]["action"] for r in rows], width, label=name)
    ax.set_xticks(x); ax.set_xticklabels([str(r["seed"]) for r in rows], rotation=45)
    ax.set_ylabel("CNER action at matched capability target"); ax.set_xlabel("Independent seed")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(target, dpi=180); plt.close(fig)


def make_report_m5a(result, target):
    s = result["summary"]
    lines = ["# M5a 30% Executable-Optimizer Confirmation v8", "",
             f"**Status:** `{result['scientific_status']}`", "", result["scope"], "",
             "## Summary", "", f"- Comparable seeds: {s['comparable_seeds']}/{s['independent_seeds']}",
             f"- Adam best-action successes: {s['adam_best_action_successes']}/{s['comparable_seeds']}",
             f"- Exact one-sided binomial p: {s['exact_one_sided_binomial_p']}",
             f"- Median Adam-minus-best-alternative action: {s['median_adam_minus_best_alternative_action']}",
             f"- Median Adam hit step: {s['median_adam_hit_step']}",
             f"- Depth gate: {s['primary_depth_gate_pass']}",
             f"- Median Adam/best length ratio: {s['median_adam_to_best_length_ratio']}",
             f"- Median Adam/best effective-capacity ratio: {s['median_adam_to_best_effective_capacity_ratio']}",
             f"- Primary gate: {s['primary_gate_pass']}", "", "## Pairwise", ""]
    lines += [f"- Adam vs {name}: {v['adam_wins']}/{v['comparable']}, p={v['exact_one_sided_binomial_p']}"
              for name, v in s["pairwise"].items()]
    lines += ["", "## Interpretation boundary", ""] + [f"- {x}" for x in result["interpretation_boundary"]]
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")


def regularize_metric(G, floor_relative, torch):
    G = 0.5 * (G + G.T)
    vals, vecs = torch.linalg.eigh(G)
    scale = vals.abs().max().clamp_min(1e-12)
    vals = vals.clamp_min(floor_relative * scale)
    G = (vecs * vals[None, :]) @ vecs.T
    G = G * (G.shape[0] / torch.trace(G).clamp_min(1e-12))
    vals = torch.linalg.eigvalsh(G)
    return G.detach(), {"eigen_min": float(vals.min().cpu()),
                        "eigen_max": float(vals.max().cpu()),
                        "condition": float((vals.max()/vals.min()).cpu())}


def source_frozen_metrics(theta0, B, model, spec, x, y, p, torch, F, functional_call):
    grads = []
    for i in range(x.shape[0]):
        z = torch.zeros(B.shape[1], device=theta0.device, requires_grad=True)
        theta = theta0 + B @ z
        logits = functional_call(model, params_from_vector(theta, spec), (x[i:i+1],))
        loss = F.cross_entropy(logits, y[i:i+1])
        grads.append(torch.autograd.grad(loss, z)[0].detach())
    grads = torch.stack(grads)
    fisher = grads.T @ grads / max(len(grads), 1)
    adam_diag = torch.diag(torch.diag(fisher))
    def q_fn(z):
        theta = theta0 + B @ z
        _, feat = functional_call(model, params_from_vector(theta, spec), (x,), {"return_features": True})
        return feat.mean(0)
    z0 = torch.zeros(B.shape[1], device=theta0.device, requires_grad=True)
    DQ = torch.autograd.functional.jacobian(q_fn, z0, vectorize=True).detach()
    representation = DQ.T @ DQ
    raw = {"identity": torch.eye(B.shape[1], device=theta0.device),
           "adam_source_diag": adam_diag, "fisher_source_full": fisher,
           "representation_pullback": representation}
    metrics = {}; metadata = {}
    for name, G in raw.items():
        metrics[name], metadata[name] = regularize_metric(G, p.metric_eigen_floor_relative, torch)
    return metrics, metadata


def output_fisher_raw(theta, B, model, spec, x, torch, functional_call):
    """Pull back the categorical Fisher--Rao metric to chart coordinates."""
    def logits_from_z(z, sample):
        params=params_from_vector(theta+B@z,spec)
        return functional_call(model,params,(sample,)).reshape(-1)
    z0=torch.zeros(B.shape[1],device=theta.device,requires_grad=True)
    total=torch.zeros((B.shape[1],B.shape[1]),device=theta.device)
    for i in range(x.shape[0]):
        sample=x[i:i+1]
        logits=logits_from_z(z0,sample)
        prob=torch.softmax(logits,dim=0).detach()
        J=torch.autograd.functional.jacobian(lambda z: logits_from_z(z,sample),z0,vectorize=True).detach()
        C=torch.diag(prob)-prob[:,None]*prob[None,:]
        total=total+J.T@C@J
    return (0.5*(total+total.T)/max(x.shape[0],1)).detach()


def representation_raw(theta,B,model,spec,x,torch,functional_call):
    def q_fn(z):
        params=params_from_vector(theta+B@z,spec)
        _,feat=functional_call(model,params,(x,),{"return_features":True})
        return feat.mean(0)
    z0=torch.zeros(B.shape[1],device=theta.device,requires_grad=True)
    DQ=torch.autograd.functional.jacobian(q_fn,z0,vectorize=True).detach()
    return (DQ.T@DQ).detach()


def gauge_factors(spec,device,torch):
    """Positive conv2-channel rescaling with inverse compensation in fc."""
    total=max(b for _,_,_,b in spec);f=torch.ones(total,device=device)
    scales=torch.tensor([0.55,0.70,0.85,1.15,1.35,1.55,1.80,2.10],device=device)
    for name,shape,a,b in spec:
        if name=="conv2.weight":
            f[a:b]=scales[:,None,None,None].expand(shape).reshape(-1)
        elif name=="conv2.bias": f[a:b]=scales
        elif name=="fc.weight":
            f[a:b]=(1.0/scales)[None,:,None].expand(shape[0],shape[1]//49,49).reshape(-1)
    return f


def mean_output_kl(theta_a,theta_b,model,spec,x,torch,functional_call):
    with torch.no_grad():
        la=functional_call(model,params_from_vector(theta_a,spec),(x,)).double()
        lb=functional_call(model,params_from_vector(theta_b,spec),(x,)).double()
        logpa=torch.log_softmax(la,dim=1);logpb=torch.log_softmax(lb,dim=1)
        pa=torch.softmax(la,dim=1)
        return float((pa*(logpa-logpb)).sum(1).mean().cpu())


def metric_certification(theta0,B,model,spec,x,Graw,p,seed,torch,functional_call):
    gen=torch.Generator(device=theta0.device).manual_seed(seed+931177)
    true=[];quad=[]
    for i in range(p.kl_perturbations):
        q=torch.randn(B.shape[1],generator=gen,device=theta0.device);q=q/q.norm().clamp_min(1e-12)
        radius=p.kl_radius_min+(p.kl_radius_max-p.kl_radius_min)*(i/max(p.kl_perturbations-1,1))
        dz=radius*q
        true.append(mean_output_kl(theta0,theta0+B@dz,model,spec,x,torch,functional_call))
        quad.append(float((0.5*dz@Graw@dz).cpu()))
    rho=spearman(true,quad)
    rel=[abs(a-b)/max(abs(a),1e-12) for a,b in zip(true,quad)]
    factors=gauge_factors(spec,theta0.device,torch);theta_g=theta0*factors;B_g=B*factors[:,None]
    with torch.no_grad():
        l0=functional_call(model,params_from_vector(theta0,spec),(x,))
        lg=functional_call(model,params_from_vector(theta_g,spec),(x,))
    logit_res=float(torch.linalg.norm(l0-lg)/torch.linalg.norm(l0).clamp_min(1e-12))
    Gg=output_fisher_raw(theta_g,B_g,model,spec,x,torch,functional_call)
    fisher_res=float(torch.linalg.norm(Gg-Graw)/torch.linalg.norm(Graw).clamp_min(1e-12))
    R0=representation_raw(theta0,B,model,spec,x,torch,functional_call)
    Rg=representation_raw(theta_g,B_g,model,spec,x,torch,functional_call)
    rep_change=float(torch.linalg.norm(Rg-R0)/torch.linalg.norm(R0).clamp_min(1e-12))
    eig=torch.linalg.eigvalsh(Graw);rank=int((eig>eig.max().clamp_min(1e-12)*p.metric_eigen_floor_relative).sum())
    return {"kl_spearman":rho,"kl_median_relative_error":float(np.median(rel)),
            "kl_max_relative_error":max(rel),"gauge_logit_relative_residual":logit_res,
            "gauge_fisher_relative_residual":fisher_res,"representation_gauge_relative_change":rep_change,
            "raw_fisher_effective_rank":rank,"raw_fisher_eigenvalues":[float(v.cpu()) for v in eig]}


def executable_path_natural(theta0,B,G,model,spec,batches,p,torch,F,functional_call):
    z=torch.zeros(B.shape[1],device=theta0.device);path=[z.clone()];Ginv=torch.linalg.inv(G)
    for step in range(p.adapt_steps):
        x,y=batches[step%len(batches)];w=z.detach().clone().requires_grad_(True)
        theta=theta0+B@w;loss=F.cross_entropy(functional_call(model,params_from_vector(theta,spec),(x,)),y)
        grad=torch.autograd.grad(loss,w)[0].detach();direction=Ginv@grad
        direction=direction/torch.sqrt((direction@G@direction).clamp_min(1e-24))
        z=(w.detach()-p.executable_step_radius*direction).detach();path.append(z.clone())
    return torch.stack(path)


def metric_path_action(path, G, theta0, B, model, spec, dev_x, dev_y,
                       p, torch, F, functional_call, subdivisions=1):
    Ginv = torch.linalg.inv(G)
    z0 = torch.zeros(B.shape[1], device=path.device)
    g0 = intelligence_gradient(z0, theta0, B, model, spec, dev_x, dev_y, torch, F, functional_call)
    scale = float(torch.sqrt((g0 @ Ginv @ g0).clamp_min(1e-24)).cpu())
    total = 0.0; hs = []; length = 0.0
    for a,b in zip(path[:-1],path[1:]):
        for j in range(subdivisions):
            aa=a+(b-a)*(j/subdivisions);bb=a+(b-a)*((j+1)/subdivisions);mid=.5*(aa+bb);dz=bb-aa
            grad = intelligence_gradient(mid,theta0,B,model,spec,dev_x,dev_y,torch,F,functional_call)
            h = p.h0 + float(torch.sqrt((grad @ Ginv @ grad).clamp_min(1e-24)).cpu())/max(scale,1e-12)
            dl = float(torch.sqrt((dz @ G @ dz).clamp_min(0)).cpu())
            total += dl/h; length += dl; hs.append(h)
    return total, length, hs


def adaptive_metric_path_action(path,G,theta0,B,model,spec,dev_x,dev_y,
                                p,torch,F,functional_call):
    """Frozen T4/T8 refinement, escalating to T8/T16 and T16/T32 if needed."""
    values={};details={}
    for subdivisions in (p.integration_subdivisions_initial,2*p.integration_subdivisions_initial):
        values[subdivisions],details[subdivisions],hs=metric_path_action(
            path,G,theta0,B,model,spec,dev_x,dev_y,p,torch,F,functional_call,subdivisions)
    low=p.integration_subdivisions_initial;high=2*low
    rel=abs(values[high]-values[low])/max(abs(values[high]),1e-12)
    if rel<=p.integration_refinement_tol:
        return values[high],details[high],hs,rel,f"T{low}/T{high}",True,values
    while high < p.integration_subdivisions_max:
        final=2*high
        values[final],details[final],hs=metric_path_action(
            path,G,theta0,B,model,spec,dev_x,dev_y,p,torch,F,functional_call,final)
        rel=abs(values[final]-values[high])/max(abs(values[final]),1e-12)
        if rel<=p.integration_refinement_tol:
            return values[final],details[final],hs,rel,f"T{high}/T{final}",True,values
        low,high=high,final
    return values[high],details[high],hs,rel,f"T{low}/T{high}",False,values


def rankdata(values):
    order=np.argsort(values);ranks=np.empty(len(values),dtype=float);i=0
    while i<len(values):
        j=i+1
        while j<len(values) and values[order[j]]==values[order[i]]:j+=1
        ranks[order[i:j]]=.5*(i+j-1)+1;i=j
    return ranks


def spearman(x,y):
    if len(x)<3:return None
    rx,ry=rankdata(np.asarray(x)),rankdata(np.asarray(y))
    if np.std(rx)==0 or np.std(ry)==0:return None
    return float(np.corrcoef(rx,ry)[0,1])


def one_seed_metric(seed,p,outdir,device,deps):
    torch,nn,F,functional_call,DataLoader,Dataset,Subset,datasets,transforms=deps
    seed_everything(seed,torch);root=str(outdir.parent/"mnist_data")
    train_ds=datasets.MNIST(root,train=True,download=True,transform=transforms.ToTensor())
    test_ds=datasets.MNIST(root,train=False,download=True,transform=transforms.ToTensor())
    src_train,adapt_train,intelligence_dev,capability_probe,metric_probe=disjoint_train_subsets(
        train_ds,[p.source_train_size,p.adapt_train_size,p.intelligence_probe_count,
                  p.capability_probe_count,p.metric_probe_count],seed,Subset,torch)
    src_test=take_subset(test_ds,p.source_test_size,seed+1,Subset,torch)
    adapt_test=take_subset(test_ds,p.adapt_test_size,seed+3,Subset,torch)
    src_loader=DataLoader(src_train,batch_size=p.source_batch_size,shuffle=True,generator=torch.Generator().manual_seed(seed),num_workers=0)
    src_test_loader=DataLoader(src_test,batch_size=512,shuffle=False,num_workers=0)
    adapt_loader=DataLoader(adapt_train,batch_size=p.adapt_batch_size,shuffle=True,generator=torch.Generator().manual_seed(seed+4),num_workers=0)
    TinyCNN=build_components(torch,nn,F);model=TinyCNN().to(device);train_source(model,src_loader,device,p,torch,F)
    source_acc=accuracy_model(model,src_test_loader,device,torch,shifted=False)
    spec,nparam=vector_spec(model);theta0=flatten_model(model,torch).to(device)
    anchor_x,_=next(iter(DataLoader(src_train,batch_size=p.anchor_count,shuffle=False)))
    dev_x,dev_y=next(iter(DataLoader(intelligence_dev,batch_size=p.intelligence_probe_count,shuffle=False)))
    cap_x,cap_y=next(iter(DataLoader(capability_probe,batch_size=p.capability_probe_count,shuffle=False)))
    met_x,met_y=next(iter(DataLoader(metric_probe,batch_size=p.metric_probe_count,shuffle=False)))
    tx,ty=next(iter(adapt_loader));anchor_x=anchor_x.to(device)
    dev_x,dev_y=development_batch(dev_x.to(device),torch),dev_y.to(device)
    cap_x,cap_y=shifted_batch(cap_x.to(device),torch),cap_y.to(device);met_x,met_y=met_x.to(device),met_y.to(device)
    tx,ty=shifted_batch(tx.to(device),torch),ty.to(device)
    batches=[(shifted_batch(x.to(device),torch),y.to(device)) for x,y in adapt_loader]
    J=response_jacobian(theta0,model,spec,anchor_x,torch,functional_call)
    B=build_chart(theta0,J,tx,ty,model,spec,p,torch,F,functional_call,seed)
    kernel_residual=float(torch.linalg.norm(J@B)/torch.linalg.norm(J).clamp_min(1e-12))
    Graw=output_fisher_raw(theta0,B,model,spec,met_x,torch,functional_call)
    Gf,fisher_meta=regularize_metric(Graw,p.metric_eigen_floor_relative,torch)
    reverse=torch.arange(B.shape[1]-1,-1,-1,device=device)
    permutation=torch.eye(B.shape[1],device=device)[:,reverse]
    Gwrong=permutation.T@Gf@permutation
    Grep,rep_meta=regularize_metric(representation_raw(theta0,B,model,spec,met_x,torch,functional_call),p.metric_eigen_floor_relative,torch)
    metrics={"identity":torch.eye(B.shape[1],device=device),"representation_pullback":Grep,"output_fisher_quotient":Gf}
    metadata={"identity":{"condition":1.0},"representation_pullback":rep_meta,"output_fisher_quotient":fisher_meta}
    certification=metric_certification(theta0,B,model,spec,met_x,Graw,p,seed,torch,functional_call)
    with torch.no_grad():r0=functional_call(model,params_from_vector(theta0,spec),(anchor_x,)).reshape(-1)
    z0=torch.zeros(B.shape[1],device=device);initial_loss=loss_at_z(z0,theta0,B,model,spec,cap_x,cap_y,torch,F,functional_call)
    target_loss=initial_loss*(1-p.capability_loss_reduction_fraction)
    algorithms={};names=("adam","normalized_sgd","normalized_momentum","sign_gradient","natural_gradient","wrong_fisher_natural_gradient")
    for name in names:
        if name=="natural_gradient":
            raw=executable_path_natural(theta0,B,Gf,model,spec,batches,p,torch,F,functional_call)
        elif name=="wrong_fisher_natural_gradient":
            raw=executable_path_natural(theta0,B,Gwrong,model,spec,batches,p,torch,F,functional_call)
        else:
            raw=executable_path(name,theta0,B,model,spec,batches,p,torch,F,functional_call)
        path,hit,cap_loss,hit_step=truncate_at_capability(raw,target_loss,theta0,B,model,spec,cap_x,cap_y,torch,F,functional_call)
        leak=path_response_leakage(path,theta0,B,model,spec,anchor_x,r0,torch,functional_call)
        metric_results={}
        for mn,G in metrics.items():
            action,length,hs,integ,pair,converged,values=adaptive_metric_path_action(
                path,G,theta0,B,model,spec,dev_x,dev_y,p,torch,F,functional_call)
            metric_results[mn]={"action":action,
                                "action_by_subdivisions":{str(k):v for k,v in values.items()},
                                "integration_pair_used":pair,"integration_converged":converged,
                                "length":length,"effective_capacity":length/max(action,1e-12),
                                "capacity_relative_span":(max(hs)-min(hs))/max(float(np.median(hs)),1e-12),
                                "integration_relative_change":integ,
                                "admissible":bool(hit and leak["max"]<=p.leakage_relative_gate and converged)}
        algorithms[name]={"hit_capability":hit,"hit_step":hit_step,"capability_final_loss":cap_loss,
                          "capability_relative_error":abs(cap_loss-target_loss)/max(target_loss,1e-12),
                          "leakage_max":leak["max"],"metrics":metric_results}
    return {"seed":seed,"parameter_count":nparam,"source_accuracy":source_acc,"kernel_residual":kernel_residual,
            "source_gate":source_acc>=p.source_accuracy_gate,"kernel_gate":kernel_residual<=p.kernel_residual_gate,
            "metric_metadata":metadata,"metric_certification":certification,
            "wrong_metric":{"permutation":reverse.detach().cpu().tolist(),
                            "condition":fisher_meta["condition"]},"algorithms":algorithms}


def run_fisher_final(p,outdir:Path):
    deps=import_torch();torch=deps[0];device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    outdir.mkdir(parents=True,exist_ok=True);rows=[]
    for i in range(p.seeds):
        seed=p.base_seed+i;print(f"[v16 seed {i+1}/{p.seeds}] {seed}",flush=True)
        row=one_seed_metric(seed,p,outdir,device,deps);rows.append(row);print("  complete",flush=True)
    metric_names=("identity","representation_pullback","output_fisher_quotient")
    summaries={}
    for mn in metric_names:
        valid=[r for r in rows if r["source_gate"] and r["kernel_gate"]]
        comps=[];natural_wins=0;wrong_wins=0;adam_wins=0;specificity_wins=0
        natural_diffs=[];wrong_diffs=[];adam_diffs=[];true_minus_wrong=[];actions=[];lengths=[];spans=[]
        for r in valid:
            a=r["algorithms"]
            if all(a[n]["metrics"][mn]["admissible"] for n in a):
                comps.append(r);nat=a["natural_gradient"]["metrics"][mn];wrong=a["wrong_fisher_natural_gradient"]["metrics"][mn];adam=a["adam"]["metrics"][mn]
                best_other_nat=min(a[n]["metrics"][mn]["action"] for n in a if n!="natural_gradient")
                best_other_wrong=min(a[n]["metrics"][mn]["action"] for n in a if n!="wrong_fisher_natural_gradient")
                best_other_adam=min(a[n]["metrics"][mn]["action"] for n in a if n!="adam")
                natural_wins+=nat["action"]<=best_other_nat*(1+p.action_relative_tolerance);natural_diffs.append(nat["action"]-best_other_nat)
                wrong_wins+=wrong["action"]<=best_other_wrong*(1+p.action_relative_tolerance);wrong_diffs.append(wrong["action"]-best_other_wrong)
                specificity_wins+=nat["action"]<wrong["action"];true_minus_wrong.append(nat["action"]-wrong["action"])
                adam_wins+=adam["action"]<=best_other_adam*(1+p.action_relative_tolerance);adam_diffs.append(adam["action"]-best_other_adam)
            for n in a:
                q=a[n]["metrics"][mn]
                if q["admissible"]:actions.append(q["action"]);lengths.append(q["length"]);spans.append(q["capacity_relative_span"])
        rho=spearman(actions,lengths);maxcond=max(r["metric_metadata"][mn]["condition"] for r in rows)
        medspan=float(np.median(spans)) if spans else None
        ok=(len(comps)==p.seeds and maxcond<=p.metric_condition_gate and medspan is not None and
            medspan>=p.metric_capacity_span_gate)
        summaries[mn]={"comparable_seeds":len(comps),"natural_gradient_argmin_successes":natural_wins,
                       "median_natural_minus_best_other_action":float(np.median(natural_diffs)) if natural_diffs else None,
                       "wrong_fisher_natural_argmin_successes":wrong_wins,
                       "median_wrong_natural_minus_best_other_action":float(np.median(wrong_diffs)) if wrong_diffs else None,
                       "true_natural_beats_wrong_successes":specificity_wins,
                       "true_vs_wrong_exact_one_sided_binomial_p":exact_binomial_upper(specificity_wins,len(comps)) if comps else None,
                       "median_true_minus_wrong_action":float(np.median(true_minus_wrong)) if true_minus_wrong else None,
                       "adam_argmin_successes":adam_wins,"median_adam_minus_best_other_action":float(np.median(adam_diffs)) if adam_diffs else None,
                       "action_length_spearman":rho,"median_capacity_relative_span":medspan,
                       "max_condition":maxcond,"metric_admissible":ok}
    valid=[r for r in rows if r["source_gate"] and r["kernel_gate"]];certs=[r["metric_certification"] for r in valid]
    cert_summary={"valid_seeds":len(valid),
                  "median_kl_spearman":float(np.median([c["kl_spearman"] for c in certs])) if certs else None,
                  "median_kl_relative_error":float(np.median([c["kl_median_relative_error"] for c in certs])) if certs else None,
                  "max_gauge_logit_relative_residual":max((c["gauge_logit_relative_residual"] for c in certs),default=None),
                  "max_gauge_fisher_relative_residual":max((c["gauge_fisher_relative_residual"] for c in certs),default=None),
                  "median_representation_gauge_relative_change":float(np.median([c["representation_gauge_relative_change"] for c in certs])) if certs else None,
                  "median_raw_fisher_effective_rank":float(np.median([c["raw_fisher_effective_rank"] for c in certs])) if certs else None}
    cert_gate=(len(valid)==p.seeds and cert_summary["median_kl_spearman"]>=p.kl_spearman_gate and
               cert_summary["median_kl_relative_error"]<=p.kl_median_relative_error_gate and
               cert_summary["max_gauge_logit_relative_residual"]<=p.gauge_logit_residual_gate and
               cert_summary["max_gauge_fisher_relative_residual"]<=p.gauge_fisher_relative_gate and
               cert_summary["median_representation_gauge_relative_change"]>=p.representation_gauge_change_gate)
    fs=summaries[p.primary_metric];primary_admissible=fs["metric_admissible"]
    diagnostic_admissibility={m:summaries[m]["metric_admissible"] for m in metric_names if m!=p.primary_metric}
    natural_gate=(primary_admissible and fs["natural_gradient_argmin_successes"]>=p.natural_gradient_success_gate and
                  fs["median_natural_minus_best_other_action"]<=0 and
                  fs["wrong_fisher_natural_argmin_successes"]<=p.wrong_natural_success_ceiling and
                  fs["true_natural_beats_wrong_successes"]>=p.natural_specificity_success_gate and
                  fs["true_vs_wrong_exact_one_sided_binomial_p"]<.05 and fs["median_true_minus_wrong_action"]<0)
    if not primary_admissible or len(valid)!=p.seeds:status="V16_FISHER_PRIMARY_INADMISSIBLE"
    elif cert_gate and natural_gate:status="V16_RESTRICTED_CNER_F_CONFIRMED_IN_FROZEN_TINYCNN_MNIST_CHART"
    elif cert_gate:status="V16_FISHER_METRIC_RECONFIRMED_NATURAL_FLOW_NOT_CONFIRMED"
    else:status="V16_FISHER_METRIC_NOT_RECONFIRMED"
    result={"scientific_status":status,"scope":"Prospective new-seed TinyCNN/MNIST confirmation of restricted CNER-F in one frozen eight-dimensional response-fibre chart. It is not CNER-S, global optimality, ordinary-training minimization, cross-architecture replication, or a universal learning law. All six named output-Fisher paths in every seed are blocking. Each executable path receives the same frozen 240-step budget and is truncated only at the common 20% capability-loss target. Integration uses frozen adaptive T4/T8/T16/T32 refinement.",
            "protocol":asdict(p),"protocol_sha256":sha256_text(canonical_json(asdict(p))),
            "summary":{"metric_certification":cert_summary,"metric_certification_gate_pass":cert_gate,
                       "optimizer_metric_summaries":summaries,"natural_gradient_gate_pass":natural_gate,
                       "primary_metric":p.primary_metric,"all_primary_fisher_paths_admissible":primary_admissible,
                       "diagnostic_metric_admissibility":diagnostic_admissibility,
                       "frozen_decision_rule":"Reconfirm Fisher only by KL and gauge gates. Primary admissibility requires all six named output-Fisher paths in all 16 seeds to hit the common target within the same 240-step budget, pass leakage, and converge under frozen T4/T8/T16/T32 refinement. Identity and hidden-representation diagnostics cannot veto. Confirm restricted CNER-F only if true Fisher natural gradient is restricted argmin in >=12/16 with nonpositive median gap, beats wrong-Fisher natural gradient in >=12/16 with exact one-sided binomial p<0.05 and negative median paired action, while wrong-Fisher natural argmin successes are <=8/16."},
            "seeds":rows,"runtime":{"python":sys.version,"torch":torch.__version__,"numpy":np.__version__,"platform":platform.platform(),"device":str(device),"unix_time":time.time()}}
    (outdir/"result.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    (outdir/"protocol.json").write_text(json.dumps(asdict(p),indent=2)+"\n",encoding="utf-8")
    make_report_fisher(result,outdir/"REPORT.md")
    write_fisher_seed_summary(result,outdir/"seed_summary_v16.csv")
    make_fisher_action_plot(result,outdir/"action_comparison_v16.png")
    return result


def make_report_fisher(result,target):
    s=result["summary"];lines=["# Restricted CNER-F Prospective Confirmation v16","",f"**Status:** `{result['scientific_status']}`","",result["scope"],"","## Metric recertification","",json.dumps(s["metric_certification"],indent=2),"",f"Certification gate: {s['metric_certification_gate_pass']}",f"All primary Fisher paths admissible: {s['all_primary_fisher_paths_admissible']}",f"Diagnostic admissibility: {s['diagnostic_metric_admissibility']}","","## Optimizer audit",""]
    for n,v in s["optimizer_metric_summaries"].items():lines += [f"- {n}: admissible={v['metric_admissible']}, comparable={v['comparable_seeds']}, true natural minima={v['natural_gradient_argmin_successes']}, median true gap={v['median_natural_minus_best_other_action']}, wrong natural minima={v['wrong_fisher_natural_argmin_successes']}, true beats wrong={v['true_natural_beats_wrong_successes']}, paired p={v['true_vs_wrong_exact_one_sided_binomial_p']}, Adam minima={v['adam_argmin_successes']}, H-span={v['median_capacity_relative_span']}, cond={v['max_condition']}"]
    lines += ["",f"Natural-gradient confirmation gate: {s['natural_gradient_gate_pass']}"]
    target.write_text("\n".join(lines)+"\n",encoding="utf-8")


def write_fisher_seed_summary(result,target):
    names=("adam","normalized_sgd","normalized_momentum","sign_gradient",
           "natural_gradient","wrong_fisher_natural_gradient")
    fields=["seed","source_accuracy","kernel_residual","all_six_paths_admissible",
            "natural_gradient_is_argmin","natural_gradient_beats_wrong_fisher",
            "max_leakage","max_integration_relative_change"]
    fields += [f"{n}_hit_step" for n in names]
    fields += [f"{n}_fisher_action" for n in names]
    with target.open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for row in result["seeds"]:
            alg=row["algorithms"]
            q={n:alg[n]["metrics"]["output_fisher_quotient"] for n in names}
            nat=q["natural_gradient"]["action"]
            wrong=q["wrong_fisher_natural_gradient"]["action"]
            best_other=min(q[n]["action"] for n in names if n!="natural_gradient")
            out={"seed":row["seed"],"source_accuracy":row["source_accuracy"],
                 "kernel_residual":row["kernel_residual"],
                 "all_six_paths_admissible":all(q[n]["admissible"] for n in names),
                 "natural_gradient_is_argmin":nat<=best_other*(1+result["protocol"]["action_relative_tolerance"]),
                 "natural_gradient_beats_wrong_fisher":nat<wrong,
                 "max_leakage":max(alg[n]["leakage_max"] for n in names),
                 "max_integration_relative_change":max(q[n]["integration_relative_change"] for n in names)}
            out.update({f"{n}_hit_step":alg[n]["hit_step"] for n in names})
            out.update({f"{n}_fisher_action":q[n]["action"] for n in names})
            writer.writerow(out)


def make_fisher_action_plot(result,target):
    names=("adam","normalized_sgd","normalized_momentum","sign_gradient",
           "natural_gradient","wrong_fisher_natural_gradient")
    labels=("Adam","Normalized SGD","Normalized momentum","Sign gradient",
            "True-Fisher natural","Wrong-Fisher natural")
    rows=result["seeds"];x=np.arange(len(rows));width=0.13
    fig,ax=plt.subplots(figsize=(13,6))
    for j,(name,label) in enumerate(zip(names,labels)):
        values=[r["algorithms"][name]["metrics"]["output_fisher_quotient"]["action"] for r in rows]
        ax.bar(x+(j-2.5)*width,values,width,label=label)
    ax.set_xticks(x);ax.set_xticklabels([str(r["seed"]) for r in rows],rotation=45)
    ax.set_ylabel("True output-Fisher CNER action")
    ax.set_xlabel("Independent seed; common 20% capability-loss-reduction target")
    ax.set_title("Restricted CNER-F v16: all six paths admissible in all 16 seeds")
    ax.legend(fontsize=8,ncol=3);fig.tight_layout();fig.savefig(target,dpi=180);plt.close(fig)


def make_plot(rows, target):
    x = np.arange(len(rows)); width = 0.2
    fig, ax = plt.subplots(figsize=(10, 5))
    for j, key in enumerate(("actual_action", "straight_action", "reverse_action", "median_admissible_shuffle_action")):
        vals = [np.nan if r[key] is None else r[key] for r in rows]
        ax.bar(x + (j - 1.5) * width, vals, width, label=key)
    ax.set_xticks(x); ax.set_xticklabels([str(r["seed"]) for r in rows], rotation=45)
    ax.set_ylabel("Same-metric intelligence action"); ax.set_xlabel("Independent seed"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(target, dpi=180); plt.close(fig)


def make_report(result, target):
    s = result["summary"]
    lines = ["# CNN/MNIST Tangent-Value / CNER Dual Bridge v5", "", f"**Status:** `{result['scientific_status']}`", "",
             result["scope"], "", "## Summary", "",
             f"- Admissible actual paths: {s['admissible_actual_paths']}/{s['independent_seeds']}",
             f"- Temporal ordering: {s['temporal_order_successes']}/{s['temporal_comparable_seeds']}",
             f"- Temporal exact one-sided binomial p: {s['temporal_exact_one_sided_binomial_p']}",
             f"- Shuffled-intelligence temporal successes: {s['shuffled_intelligence_temporal_successes']}/{s['temporal_comparable_seeds']}",
             f"- Paired discordance true-only/shuffled-only: {s['true_only_successes']}/{s['shuffled_only_successes']}",
             f"- Exact one-sided McNemar p: {s['mcnemar_exact_one_sided_p']}",
             f"- Confirmatory gate: {s['confirmatory_gate_pass']}",
             f"- Tangent functional-value successes: {s['functional_value_successes']}/{s['functional_comparable_seeds']}",
             f"- Tangent functional-value exact p: {s['functional_value_exact_one_sided_binomial_p']}",
             f"- Median actual-minus-random accuracy: {s['median_actual_minus_random_direction_accuracy_pp']} pp",
             f"- Tangent functional-value gate: {s['functional_value_gate_pass']}",
             f"- Straight competition: {s['straight_successes']}/{s['straight_comparable_seeds']}",
             f"- Strong joint successes: {s['strong_joint_successes']}/{s['strong_joint_comparable_seeds']}",
             f"- Median target gain: {s['median_target_gain_pp']:.4f} pp",
             f"- Median actual-minus-admissible-shuffle action: {s['median_actual_minus_admissible_shuffle_action']}",
             f"- Median intelligence-density relative span: {s['median_intelligence_relative_span']:.6g}",
             "", "## Interpretation boundary", ""]
    lines += [f"- {x}" for x in result["interpretation_boundary"]]
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_protocol(path):
    if path is None: return Protocol()
    raw = json.loads(Path(path).read_text()); unknown = set(raw) - set(Protocol.__dataclass_fields__)
    if unknown: raise ValueError(f"Unknown protocol keys: {sorted(unknown)}")
    return Protocol(**raw)


def maybe_colab_upload():
    try: from google.colab import files
    except Exception: return None
    print("Optional: upload protocol.json, or cancel to use built-in frozen defaults.")
    try: uploaded = files.upload()
    except Exception as exc: print(f"Upload skipped: {exc}"); return None
    js = [Path(x) for x in uploaded if x.lower().endswith(".json")]
    return js[0] if js else None


def zip_results(outdir):
    target = outdir.parent / "cner_cnn_mnist_fisher_confirm_v16_results.zip"
    required=("result.json","REPORT.md","protocol.json","seed_summary_v16.csv","action_comparison_v16.png")
    missing=[name for name in required if not (outdir/name).is_file()]
    if missing: raise RuntimeError(f"Missing required v16 outputs: {missing}")
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as z:
        for name in required:
            p=outdir/name;z.write(p,Path("cner_cnn_results")/name)
    return target


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--protocol", type=Path)
    ap.add_argument("--output", type=Path, default=Path("cner_cnn_results"))
    ap.add_argument("--no-upload-dialog", action="store_true"); ap.add_argument("--no-download", action="store_true")
    args, unknown = ap.parse_known_args()
    if unknown: print(f"[notice] ignored notebook arguments: {unknown}")
    protocol_path = args.protocol
    if protocol_path is None and not args.no_upload_dialog: protocol_path = maybe_colab_upload()
    p = load_protocol(protocol_path)
    if (p.protocol_name != "CNER_CNN_MNIST_FISHER_NATURAL_CONFIRM_V16" or
        p.wrong_metric_permutation != "reverse_chart_coordinates" or
        p.primary_metric != "output_fisher_quotient" or
        p.base_seed != 18726 or p.seeds != 16 or p.adapt_steps != 240 or
        (p.integration_subdivisions_initial,p.integration_subdivisions_max)!=(4,32)):
        raise ValueError("v16 protocol lock violated")
    result = run_fisher_final(p, args.output); zp = zip_results(args.output)
    print("=" * 96); print("CNN/MNIST RESTRICTED CNER-F PROSPECTIVE CONFIRMATION v16"); print("=" * 96)
    print(json.dumps(result["summary"], indent=2)); print(f"scientific_status: {result['scientific_status']}"); print(f"Results ZIP: {zp.resolve()}")
    if not args.no_download:
        try:
            from google.colab import files
            files.download(str(zp))
        except Exception: pass


if __name__ == "__main__": main()
