#!/usr/bin/env python3
"""Float64 precision repair, implementation revision r1."""
from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms

PROTOCOL = "CNER_RESNET18_CIFAR10_FIBRE_EXTERNAL_PRECISION_CONFIRM_V4_0C"
IMPLEMENTATION_REVISION = "v4.0c-r1"
SEEDS = list(range(64726, 64742))
ANCHOR_COUNTS = [4, 16, 32]
PRIMARY_ANCHORS = 16
N_RANDOM = 32
STEP_RADIUS = 0.08
PRIMARY_REQUIRED = 12
SENSITIVITY_REQUIRED = 10
MIN_BASE_ACCURACY = 0.70
MAX_KERNEL_RELATIVE_RESIDUAL = 1e-12
MAX_FINITE_RESPONSE_LEAKAGE = 1e-9
MIN_TRUE_IMPROVEMENT = 1e-3
MIN_ADV_ANTI = 2e-3
MIN_ADV_RANDOM_BEST = 1e-3
MIN_ADV_SHUFFLED = 1e-3
MIN_AMBIENT_ABS_LEAKAGE = 1e-2


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def centered(x: torch.Tensor) -> torch.Tensor:
    return x - x.mean(-1, keepdim=True)


def unit(x: torch.Tensor) -> torch.Tensor:
    return x / (x.norm() + 1e-12)


def cache_features(net, loader, device):
    xs, ys = [], []
    net.eval()
    with torch.no_grad():
        for x, y in loader:
            xs.append(net(x.to(device, non_blocking=True)).cpu())
            ys.append(y)
    return torch.cat(xs), torch.cat(ys)


def fit_head(x, y, seed, device):
    seed_all(seed)
    head = nn.Linear(x.shape[1], 10).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=3e-3, weight_decay=1e-3)
    x, y = x.to(device), y.to(device)
    for _ in range(120):
        idx = torch.randint(0, len(x), (min(256, len(x)),), device=device)
        value = F.cross_entropy(head(x[idx]), y[idx])
        opt.zero_grad()
        value.backward()
        opt.step()
    return head


def head_vector(head):
    return torch.cat((head.weight.detach().flatten(), head.bias.detach()))


def unpack(w, dim):
    return w[: 10 * dim].reshape(10, dim), w[10 * dim :]


def logits(w, x):
    weight, bias = unpack(w, x.shape[1])
    return x @ weight.T + bias


def loss(w, x, y):
    return F.cross_entropy(logits(w, x), y)


def exact_response_jacobian(anchor_x):
    """Exact Jacobian of centered anchor logits w.r.t. the complete linear head."""
    anchors, dim = anchor_x.shape
    nparam = 10 * dim + 10
    rows = []
    eye10 = torch.eye(10, device=anchor_x.device, dtype=anchor_x.dtype)
    centering = eye10 - torch.ones_like(eye10) / 10.0
    for i in range(anchors):
        raw = torch.zeros((10, nparam), device=anchor_x.device, dtype=anchor_x.dtype)
        for cls in range(10):
            raw[cls, cls * dim : (cls + 1) * dim] = anchor_x[i]
            raw[cls, 10 * dim + cls] = 1.0
        rows.append(centering @ raw)
    return torch.cat(rows, dim=0)


def kernel_projector_data(jac):
    _, singular, vh = torch.linalg.svd(jac, full_matrices=False)
    tol = max(jac.shape) * torch.finfo(jac.dtype).eps * float(singular.max())
    rank = int((singular > tol).sum().item())
    row_basis = vh[:rank]
    return row_basis, rank, singular


def project_kernel(v, row_basis):
    return v - row_basis.T @ (row_basis @ v)


def gradient(w, x, y):
    z = w.detach().clone().requires_grad_(True)
    return torch.autograd.grad(loss(z, x, y), z)[0].detach()


def loss_delta(w, direction, x, y):
    with torch.no_grad():
        return float((loss(w + STEP_RADIUS * direction, x, y) - loss(w, x, y)).item())


def response_leakage(w, direction, anchors):
    with torch.no_grad():
        before = centered(logits(w, anchors))
        after = centered(logits(w + STEP_RADIUS * direction, anchors))
        return float((after - before).abs().max().item())


def kernel_relative_residual(jac, direction, largest_singular):
    denominator = largest_singular * direction.norm() + 1e-30
    return float((jac @ direction).norm().div(denominator).item())


def evaluate_anchor_count(seed, count, w, dev_x, dev_y, confirm_x, confirm_y, all_anchors):
    anchors = all_anchors[:count]
    jac = exact_response_jacobian(anchors)
    row_basis, rank, singular = kernel_projector_data(jac)
    grad_dev = gradient(w, dev_x, dev_y)
    true = unit(-project_kernel(grad_dev, row_basis))
    anti = -true
    ambient = unit(-grad_dev)

    gen = torch.Generator(device=w.device)
    gen.manual_seed(seed + 177 + 1000 * count)
    perm = torch.randperm(len(dev_y), generator=gen, device=w.device)
    shuffled = unit(-project_kernel(gradient(w, dev_x, dev_y[perm]), row_basis))

    random_dirs = []
    for idx in range(N_RANDOM):
        gen.manual_seed(seed * 10000 + count * 100 + idx)
        draw = torch.randn(
            w.numel(), generator=gen, device=w.device, dtype=w.dtype
        )
        random_dirs.append(unit(project_kernel(draw, row_basis)))

    values = {
        "true": loss_delta(w, true, confirm_x, confirm_y),
        "anti": loss_delta(w, anti, confirm_x, confirm_y),
        "shuffled": loss_delta(w, shuffled, confirm_x, confirm_y),
        "ambient_raw": loss_delta(w, ambient, confirm_x, confirm_y),
    }
    random_values = [loss_delta(w, d, confirm_x, confirm_y) for d in random_dirs]
    values["random_median"] = float(np.median(random_values))
    values["random_best"] = float(np.min(random_values))
    values["random_all"] = random_values

    leaks = {
        "true": response_leakage(w, true, anchors),
        "anti": response_leakage(w, anti, anchors),
        "shuffled": response_leakage(w, shuffled, anchors),
        "ambient_raw": response_leakage(w, ambient, anchors),
    }
    kernel_residuals = {
        "true": kernel_relative_residual(jac, true, singular[0]),
        "anti": kernel_relative_residual(jac, anti, singular[0]),
        "shuffled": kernel_relative_residual(jac, shuffled, singular[0]),
    }
    margins = {
        "anti_minus_true": values["anti"] - values["true"],
        "random_best_minus_true": values["random_best"] - values["true"],
        "shuffled_minus_true": values["shuffled"] - values["true"],
    }
    gates = {
        "nontrivial_kernel": rank < w.numel(),
        "kernel_relative_residual": max(kernel_residuals.values()) <= MAX_KERNEL_RELATIVE_RESIDUAL,
        "finite_response_leakage": max(leaks["true"], leaks["anti"], leaks["shuffled"]) <= MAX_FINITE_RESPONSE_LEAKAGE,
        "true_improves": values["true"] <= -MIN_TRUE_IMPROVEMENT,
        "anti_effect": margins["anti_minus_true"] >= MIN_ADV_ANTI,
        "random_best_effect": margins["random_best_minus_true"] >= MIN_ADV_RANDOM_BEST,
        "shuffled_effect": margins["shuffled_minus_true"] >= MIN_ADV_SHUFFLED,
        "ambient_is_unconstrained": leaks["ambient_raw"] >= MIN_AMBIENT_ABS_LEAKAGE,
    }
    return {
        "anchor_count": count,
        "response_rank": rank,
        "null_dimension": w.numel() - rank,
        "min_nonzero_singular": float(singular[rank - 1].item()),
        "projected_gradient_share": float(project_kernel(grad_dev, row_basis).norm().div(grad_dev.norm() + 1e-12).item()),
        "confirm_loss_deltas": values,
        "response_leakage_max": leaks,
        "kernel_relative_residual": kernel_residuals,
        "effect_margins": margins,
        "gates": gates,
        "candidate": all(gates.values()),
    }


def run_seed(seed, data, device):
    source_x, source_y, dev_x, dev_y, confirm_x, confirm_y, anchors = data
    head = fit_head(source_x, source_y, seed, device)
    # The backbone and source-head fit remain float32. All objects entering the
    # response Jacobian, null projection, finite step, and confirmation loss
    # are frozen and promoted to float64.
    w = head_vector(head).to(device=device, dtype=torch.float64)
    dev_x, dev_y = dev_x.to(device=device, dtype=torch.float64), dev_y.to(device)
    confirm_x, confirm_y = confirm_x.to(device=device, dtype=torch.float64), confirm_y.to(device)
    anchors = anchors.to(device=device, dtype=torch.float64)
    with torch.no_grad():
        accuracy = float((logits(w, confirm_x).argmax(1) == confirm_y).float().mean().item())
    results = {str(count): evaluate_anchor_count(seed, count, w, dev_x, dev_y, confirm_x, confirm_y, anchors) for count in ANCHOR_COUNTS}
    primary = results[str(PRIMARY_ANCHORS)]
    return {
        "seed": seed,
        "base_confirm_accuracy": accuracy,
        "accuracy_gate": accuracy >= MIN_BASE_ACCURACY,
        "anchor_results": results,
        "primary_candidate": accuracy >= MIN_BASE_ACCURACY and primary["candidate"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="cner_resnet18_cifar10_fibre_external_precision_confirm_v4_0c_r1_results")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[preflight] protocol={PROTOCOL} device={device} torch={torch.__version__}", flush=True)

    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    source_transform = transforms.Compose([transforms.Resize(224), transforms.ToTensor(), normalize])
    shift_transform = transforms.Compose([transforms.Resize(224), transforms.GaussianBlur(7, 1.4), transforms.ColorJitter(0.25, 0.25, 0.2, 0.05), transforms.ToTensor(), normalize])
    root = Path("data")
    train = datasets.CIFAR10(root, train=True, download=True, transform=source_transform)
    shifted_test = datasets.CIFAR10(root, train=False, download=True, transform=shift_transform)
    backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    backbone.fc = nn.Identity()
    backbone.to(device).eval()
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)

    def loader(dataset, indices, batch=128):
        return DataLoader(Subset(dataset, list(indices)), batch_size=batch, shuffle=False, num_workers=2, pin_memory=True)

    seed_all(63000)
    print("[cache] frozen ImageNet ResNet-18 features", flush=True)
    source_x, source_y = cache_features(backbone, loader(train, range(6000)), device)
    anchor_x, _ = cache_features(backbone, loader(train, range(6000, 6032), 32), device)
    dev_x, dev_y = cache_features(backbone, loader(shifted_test, range(0, 1500)), device)
    confirm_x, confirm_y = cache_features(backbone, loader(shifted_test, range(1500, 3500)), device)
    data = (source_x, source_y, dev_x, dev_y, confirm_x, confirm_y, anchor_x)

    records = []
    for index, seed in enumerate(SEEDS, 1):
        print(f"[external precision confirm v4.0c seed {index}/{len(SEEDS)}] {seed}", flush=True)
        record = run_seed(seed, data, device)
        records.append(record)
        (output / f"seed_{seed}.json").write_text(json.dumps(record, indent=2))

    primary_count = sum(r["primary_candidate"] for r in records)
    sensitivity_counts = {
        str(count): sum(r["accuracy_gate"] and r["anchor_results"][str(count)]["candidate"] for r in records)
        for count in ANCHOR_COUNTS
    }
    primary_pass = primary_count >= PRIMARY_REQUIRED
    sensitivity_pass = all(sensitivity_counts[str(c)] >= SENSITIVITY_REQUIRED for c in (4, 32))
    confirmation = primary_pass and sensitivity_pass
    report = {
        "protocol": PROTOCOL,
        "implementation_revision": IMPLEMENTATION_REVISION,
        "scientific_status": "RESNET_CIFAR_EXTERNAL_FIBRE_V40C_CONFIRMED" if confirmation else "RESNET_CIFAR_EXTERNAL_FIBRE_V40C_NOT_CONFIRMED",
        "claim": "selected response-fibre tangent value in the complete final-classifier parameter space of a frozen ImageNet-pretrained ResNet-18 representation",
        "seeds": len(SEEDS),
        "primary_anchor_count": PRIMARY_ANCHORS,
        "primary_candidate_seeds": primary_count,
        "primary_required_seeds": PRIMARY_REQUIRED,
        "sensitivity_candidate_seeds": sensitivity_counts,
        "sensitivity_required_seeds": SENSITIVITY_REQUIRED,
        "primary_gate_pass": primary_pass,
        "anchor_sensitivity_gate_pass": sensitivity_pass,
        "confirmation_gate_pass": confirmation,
        "per_seed": records,
        "v4_0b_failure_reclassified": False,
        "precision_repair": "float64 Jacobian, SVD projection, finite step, response difference, and confirmation loss; functional gates unchanged from v4.0b",
        "claim_boundary": "prospective 16-seed numerical-precision repair in the complete final-classifier parameter space with frozen ResNet-18 backbone; not backbone adaptation, moving-kernel transport, realizability-cost scaling, Moving-Fibre F16, LLM, global variation, or a universal learning law",
    }
    (output / "report.json").write_text(json.dumps(report, indent=2))
    (output / "protocol.json").write_text(Path(__file__).with_name("protocol.json").read_text())
    (output / "run_record.json").write_text(json.dumps({"torch": torch.__version__, "cuda": torch.version.cuda, "protocol": PROTOCOL, "implementation_revision": IMPLEMENTATION_REVISION}, indent=2))
    print("=" * 100)
    print(json.dumps(report, indent=2))
    shutil.make_archive(str(output), "zip", output.parent, output.name)
    print(f"Results ZIP: {output}.zip")


if __name__ == "__main__":
    main()
