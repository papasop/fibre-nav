#!/usr/bin/env python3
"""Prospective ordinary-Adam path audit for moving response fibres."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms

PROTOCOL = "CNER_RESNET18_CIFAR10_REAL_ADAM_PATH_V4_4_R1"
SEEDS = list(range(81726, 81734))
RANK = 8
DIM = 512
CLASSES = 10
SOURCE_STEPS = 120
ADAPT_STEPS = 32
AUDIT_EVERY = 2
N_RANDOM = 8
HARD_LIMIT = 3300
MIN_SOURCE_ACC = 0.60
MIN_GAIN = 0.002
MAX_CUR_SOURCE = 0.90
MIN_CUR_BELOW_SOURCE = 0.70
MAX_CUR_SHUFFLED = 0.90
MAX_CUR_RANDOM = 0.90
MIN_ROTATION = 0.001
REQUIRED = 6
ADAPTER_SCALE = 0.20
RESPONSE_COORDS = 9
SOURCE_LR = 2e-3
SOURCE_WEIGHT_DECAY = 1e-3
ADAPT_LR = 6e-4
ADAPT_WEIGHT_DECAY = 1e-4
TIME_SHIFT = 1


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def unpack(theta):
    i = 0
    u = theta[i:i + DIM * RANK].reshape(DIM, RANK); i += DIM * RANK
    v = theta[i:i + RANK * DIM].reshape(RANK, DIM); i += RANK * DIM
    w = theta[i:i + CLASSES * DIM].reshape(CLASSES, DIM); i += CLASSES * DIM
    b = theta[i:i + CLASSES]
    return u, v, w, b


def forward(theta, x):
    u, v, w, b = unpack(theta)
    hidden = x + ADAPTER_SCALE * torch.tanh(x @ u) @ v
    return hidden @ w.T + b


def response(theta, anchors):
    z = forward(theta, anchors)
    z = z - z.mean(-1, keepdim=True)
    # Centered logits have one exact linear dependence; retain 9 coordinates.
    return z[:, :RESPONSE_COORDS].reshape(-1)


def init_theta(seed, device):
    seed_all(seed)
    gen = torch.Generator(device=device).manual_seed(seed + 91)
    u = 0.01 * torch.randn(DIM, RANK, generator=gen, device=device)
    v = torch.zeros(RANK, DIM, device=device)
    w = 0.02 * torch.randn(CLASSES, DIM, generator=gen, device=device)
    b = torch.zeros(CLASSES, device=device)
    return torch.nn.Parameter(torch.cat([u.flatten(), v.flatten(), w.flatten(), b]))


def jacobian(theta, anchors):
    z = theta.detach().double().requires_grad_(True)
    a = anchors.double()
    j = torch.autograd.functional.jacobian(
        lambda q: response(q, a), z, vectorize=True
    ).detach()
    singular = torch.linalg.svdvals(j)
    sigma = singular[0].clamp_min(1e-30)
    tol = max(j.shape) * torch.finfo(j.dtype).eps * sigma
    rank = int((singular > tol).sum().item())
    _, _, vh = torch.linalg.svd(j, full_matrices=False)
    return j, vh[:rank], sigma, rank


def leakage(j, sigma, delta):
    d = delta.double()
    return float((j @ d).norm().div(sigma * d.norm() + 1e-30).item())


def tangent_share(row_basis, delta):
    d = delta.double()
    tangent = d - row_basis.T @ (row_basis @ d)
    return float(tangent.norm().div(d.norm() + 1e-30).item())


def row_rotation(a, b):
    k = min(a.shape[0], b.shape[0])
    if k == 0:
        return float("nan")
    s = torch.linalg.svdvals(a[:k] @ b[:k].T).clamp(0, 1)
    return float(torch.acos(s.min()).item())


def accuracy(theta, x, y):
    with torch.no_grad():
        return float((forward(theta, x).argmax(1) == y).float().mean().item())


def cache(backbone, dataset, indices, device, batch=128):
    loader = DataLoader(
        Subset(dataset, list(indices)), batch_size=batch, shuffle=False,
        num_workers=2, pin_memory=True
    )
    xs, ys = [], []
    backbone.eval()
    with torch.no_grad():
        for x, y in loader:
            xs.append(backbone(x.to(device, non_blocking=True)).cpu())
            ys.append(y)
    return torch.cat(xs), torch.cat(ys)


def batch_indices(seed, n, steps, size):
    gen = torch.Generator().manual_seed(seed)
    return [torch.randint(0, n, (min(size, n),), generator=gen) for _ in range(steps)]


def fit_source(theta, x, y, seed):
    opt = torch.optim.AdamW([theta], lr=SOURCE_LR, weight_decay=SOURCE_WEIGHT_DECAY)
    for idx in batch_indices(seed + 1, len(x), SOURCE_STEPS, 256):
        loss = F.cross_entropy(forward(theta, x[idx]), y[idx])
        opt.zero_grad(); loss.backward(); opt.step()


def run_seed(seed, data, device, started):
    sx, sy, anchors, dx, dy, cx, cy = [
        t.to(device) for t in data
    ]
    theta = init_theta(seed, device)
    fit_source(theta, sx, sy, seed)
    source_acc = accuracy(theta, sx[:1600], sy[:1600])
    pre_acc = accuracy(theta, cx, cy)
    j0, row0, sigma0, rank0 = jacobian(theta, anchors)

    opt = torch.optim.AdamW([theta], lr=ADAPT_LR, weight_decay=ADAPT_WEIGHT_DECAY)
    schedule = batch_indices(seed + 2, len(dx), ADAPT_STEPS, 192)
    nodes, jacobians, sigmas, deltas = [], [], [], []

    for step, idx_cpu in enumerate(schedule):
        if time.time() - started > HARD_LIMIT:
            raise TimeoutError("55-minute hard limit exceeded")
        idx = idx_cpu.to(device)
        before = theta.detach().clone()
        loss = F.cross_entropy(forward(theta, dx[idx]), dy[idx])
        opt.zero_grad(); loss.backward(); opt.step()
        delta = theta.detach() - before
        if step % AUDIT_EVERY:
            continue
        jt, rowt, sigmat, rankt = jacobian(before, anchors)
        gen = torch.Generator(device=device).manual_seed(seed * 1000 + step)
        random_q = []
        source_random_q = []
        for _ in range(N_RANDOM):
            r = torch.randn(delta.numel(), generator=gen, device=device, dtype=delta.dtype)
            r = r * delta.norm() / (r.norm() + 1e-30)
            random_q.append(leakage(jt, sigmat, r))
            source_random_q.append(leakage(j0, sigma0, r))
        current_q = leakage(jt, sigmat, delta)
        source_q = leakage(j0, sigma0, delta)
        current_random_median = float(np.median(random_q))
        source_random_median = float(np.median(source_random_q))
        ambient_normalized_ratio = (
            current_q / (current_random_median + 1e-30)
        ) / (source_q / (source_random_median + 1e-30) + 1e-30)
        node = {
            "step": step,
            "loss": float(loss.item()),
            "current_leakage": current_q,
            "source_leakage": source_q,
            "current_tangent_share": tangent_share(rowt, delta),
            "source_tangent_share": tangent_share(row0, delta),
            "row_space_rotation": row_rotation(rowt, row0),
            "random_median_leakage": current_random_median,
            "source_random_median_leakage": source_random_median,
            "ambient_normalized_current_to_source_ratio": ambient_normalized_ratio,
            "response_rank": rankt,
            "update_norm": float(delta.norm().item()),
        }
        nodes.append(node); jacobians.append(jt.cpu()); sigmas.append(float(sigmat)); deltas.append(delta.cpu())

    # Frozen circular shift: realized updates are evaluated at the wrong nodes.
    for i, node in enumerate(nodes):
        wrong = deltas[(i + TIME_SHIFT) % len(deltas)].double()
        node["time_shuffled_leakage"] = leakage(
            jacobians[i], torch.tensor(sigmas[i], dtype=torch.float64), wrong
        )

    post_acc = accuracy(theta, cx, cy)
    cur = np.array([n["current_leakage"] for n in nodes])
    src = np.array([n["source_leakage"] for n in nodes])
    shf = np.array([n["time_shuffled_leakage"] for n in nodes])
    rnd = np.array([n["random_median_leakage"] for n in nodes])
    src_rnd = np.array([n["source_random_median_leakage"] for n in nodes])
    ambient_ratio = np.array([n["ambient_normalized_current_to_source_ratio"] for n in nodes])
    rotations = np.array([n["row_space_rotation"] for n in nodes])
    summary = {
        "source_accuracy": source_acc,
        "pre_shift_confirm_accuracy": pre_acc,
        "post_shift_confirm_accuracy": post_acc,
        "confirmation_gain": post_acc - pre_acc,
        "audited_nodes": len(nodes),
        "median_current_leakage": float(np.median(cur)),
        "median_source_leakage": float(np.median(src)),
        "median_time_shuffled_leakage": float(np.median(shf)),
        "median_random_leakage": float(np.median(rnd)),
        "median_source_random_leakage": float(np.median(src_rnd)),
        "median_ambient_normalized_current_to_source_ratio": float(np.median(ambient_ratio)),
        "median_current_to_source_ratio": float(np.median(cur / (src + 1e-30))),
        "fraction_current_below_source": float(np.mean(cur < src)),
        "median_current_to_time_shuffled_ratio": float(np.median(cur / (shf + 1e-30))),
        "median_current_to_random_ratio": float(np.median(cur / (rnd + 1e-30))),
        "maximum_row_space_rotation": float(rotations.max()),
        "source_response_rank": rank0,
    }
    gates = {
        "source_accuracy": source_acc >= MIN_SOURCE_ACC,
        "confirmation_improvement": post_acc - pre_acc >= MIN_GAIN,
        "current_vs_source": summary["median_current_to_source_ratio"] <= MAX_CUR_SOURCE,
        "current_below_source_fraction": summary["fraction_current_below_source"] >= MIN_CUR_BELOW_SOURCE,
        "current_vs_time_shuffled": summary["median_current_to_time_shuffled_ratio"] <= MAX_CUR_SHUFFLED,
        "current_vs_random": summary["median_current_to_random_ratio"] <= MAX_CUR_RANDOM,
        "nontrivial_rotation": summary["maximum_row_space_rotation"] >= MIN_ROTATION,
    }
    return {"seed": seed, "summary": summary, "gates": gates, "supported": all(gates.values()), "nodes": nodes}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="cner_resnet18_cifar10_real_optimizer_path_v4_4_r1_results")
    parser.add_argument("--allow-non-a100", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    # Colab/IPython injects exactly -f <kernel.json>. Keep all other unknown
    # arguments fatal so protocol-affecting command typos cannot pass silently.
    raw_args = sys.argv[1:]
    clean_args, ignored = [], []
    i = 0
    while i < len(raw_args):
        if raw_args[i] == "-f" and i + 1 < len(raw_args):
            ignored.extend(raw_args[i:i + 2]); i += 2
        else:
            clean_args.append(raw_args[i]); i += 1
    args = parser.parse_args(clean_args)
    if ignored:
        print(f"[notice] ignored notebook arguments: {ignored}", flush=True)
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    if not args.allow_non_a100 and "A100" not in gpu.upper():
        raise RuntimeError(f"A100 required by frozen protocol; detected {gpu}")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    started = time.time()
    print(f"[preflight] {PROTOCOL} device={device} gpu={gpu}", flush=True)

    norm = transforms.Normalize([0.485, .456, .406], [.229, .224, .225])
    source_tf = transforms.Compose([transforms.Resize(224), transforms.ToTensor(), norm])
    shift_tf = transforms.Compose([transforms.Resize(224), transforms.GaussianBlur(7, 1.4), transforms.ToTensor(), norm])
    root = Path("data")
    train = datasets.CIFAR10(root, train=True, download=True, transform=source_tf)
    shifted = datasets.CIFAR10(root, train=False, download=True, transform=shift_tf)
    backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    backbone.fc = torch.nn.Identity(); backbone.to(device).eval()
    for p in backbone.parameters(): p.requires_grad_(False)
    print("[cache] frozen ResNet-18 features", flush=True)
    sx, sy = cache(backbone, train, range(5000), device)
    ax, _ = cache(backbone, train, range(5000, 5012), device, 12)
    dx, dy = cache(backbone, shifted, range(1600), device)
    cx, cy = cache(backbone, shifted, range(1600, 3200), device)
    data = (sx, sy, ax, dx, dy, cx, cy)

    seeds = SEEDS[:1] if args.smoke else SEEDS
    records = []
    for i, seed in enumerate(seeds, 1):
        print(f"[seed {i}/{len(seeds)}] {seed}", flush=True)
        rec = run_seed(seed, data, device, started)
        records.append(rec)
        (out / f"seed_{seed}.json").write_text(json.dumps(rec, indent=2))
        print(json.dumps(rec["summary"], indent=2), flush=True)

    supporting = sum(r["supported"] for r in records)
    decision = (
        "REAL_ADAM_CURRENT_FIBRE_ALIGNMENT_SUPPORTED"
        if len(records) == len(SEEDS) and supporting >= REQUIRED
        else "REAL_ADAM_CURRENT_FIBRE_ALIGNMENT_NOT_SUPPORTED"
    )
    report = {
        "protocol": PROTOCOL,
        "scientific_status": "PROSPECTIVE_AUDIT_COMPLETED",
        "decision": decision,
        "supporting_seeds": supporting,
        "required_supporting_seeds": REQUIRED,
        "attempted_seeds": len(records),
        "elapsed_seconds": time.time() - started,
        "primary_metric_is_raw_unprojected_update": True,
        "ambient_normalized_spectral_control_is_report_only": True,
        "records": records,
        "claim_boundary": "Ordinary AdamW alignment in the declared frozen-backbone ResNet-18/CIFAR-10 construction only; not full-backbone training, arbitrary optimizers, complete kernels, or a global variational law."
    }
    (out / "report.json").write_text(json.dumps(report, indent=2))
    (out / "protocol.json").write_text(Path(__file__).with_name("protocol.json").read_text())
    with (out / "node_metrics.csv").open("w", newline="") as stream:
        rows = [(r["seed"], n) for r in records for n in r["nodes"]]
        writer = csv.DictWriter(stream, fieldnames=["seed"] + list(rows[0][1]) if rows else ["seed"])
        writer.writeheader()
        for seed, node in rows: writer.writerow({"seed": seed, **node})
    print("=" * 88); print(json.dumps(report, indent=2), flush=True)
    shutil.make_archive(str(out), "zip", out.parent, out.name)
    print(f"Results ZIP: {out}.zip", flush=True)


if __name__ == "__main__":
    main()
