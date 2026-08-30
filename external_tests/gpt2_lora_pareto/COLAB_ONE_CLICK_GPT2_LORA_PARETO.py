#!/usr/bin/env python3
"""Zero-upload external reproduction for the GPT-2 LoRA-B Pareto audit.

The frozen scientific program is not modified. This wrapper obtains a pinned
repository checkout, verifies archived evidence identity, imports the parent
engine and eight-seed wrapper, then optionally smoke-tests or fully reruns the
frozen eight-seed cohort.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path


REPOSITORY = "https://github.com/papasop/neural-fibre-geometry.git"
DEFAULT_SOURCE_REF = os.environ.get(
    "NFG_SOURCE_REF", "codex/readme-behaviour-learning-hierarchy"
)
DEFAULT_SOURCE_COMMIT = os.environ.get("NFG_SOURCE_COMMIT", "")
DEFAULT_WORK = os.environ.get(
    "NFG_WORK",
    "/content/gpt2_lora_pareto_external"
    if Path("/content").exists()
    else str(Path(tempfile.gettempdir()) / "gpt2_lora_pareto_external"),
)
SNAPSHOT_REL = Path("evidence/low_response_pareto_v1/gpt2_lora_b_v1")
PARENT = "gptw_lora_low_response_pareto_cpu.py"
WRAPPER = "gptw_lora_pareto_cpu_8seed.py"
PROTOCOL = "GPTW_GPT2_NATIVE_LORA_B_LOW_RESPONSE_PARETO_CPU_V1_R1_8SEED"
DECISION = "GPT2_LORA_LOW_RESPONSE_PARETO_CONFIRMED"


def run(command: list[str], *, cwd: Path | None = None, stream=None) -> subprocess.CompletedProcess:
    print("+", " ".join(map(str, command)), flush=True)
    if stream is None:
        return subprocess.run(command, cwd=cwd, check=True, text=True)
    proc = subprocess.run(
        command,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    stream.write(proc.stdout)
    stream.flush()
    print(proc.stdout, end="", flush=True)
    if proc.returncode:
        raise subprocess.CalledProcessError(proc.returncode, command)
    return proc


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def acquire_source(work: Path, source_root: str | None, source_ref: str) -> Path:
    if source_root:
        root = Path(source_root).resolve()
    else:
        root = work / "neural-fibre-geometry"
        if root.exists():
            shutil.rmtree(root)
        run(["git", "clone", "--quiet", REPOSITORY, str(root)])
        run(["git", "checkout", "--quiet", source_ref], cwd=root)
    return root


def verify_commit(root: Path, expected_commit: str) -> str:
    actual = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    if expected_commit and actual != expected_commit:
        raise RuntimeError(
            f"Pinned source mismatch: expected {expected_commit}, observed {actual}"
        )
    return actual


def verify_archived(root: Path, *, check_imports: bool = False) -> dict:
    run(["shasum", "-a", "256", "-c", "MANIFEST_LOW_RESPONSE_PARETO_V1.sha256"], cwd=root)
    run([sys.executable, "verify_low_response_snapshot.py"], cwd=root)
    snapshot = root / SNAPSHOT_REL
    if check_imports:
        sys.path.insert(0, str(snapshot))
        __import__("gptw_lora_low_response_pareto_cpu")
        __import__("gptw_lora_pareto_cpu_8seed")
    report = json.loads((snapshot / "results/report.json").read_text())
    if report.get("protocol") != PROTOCOL:
        raise RuntimeError(f"Unexpected protocol: {report.get('protocol')}")
    if report.get("decision") != DECISION:
        raise RuntimeError(f"Unexpected decision: {report.get('decision')}")
    if report.get("attempted") != 8 or report.get("supporting_seeds") != 8:
        raise RuntimeError("Expected attempted=8 and supporting_seeds=8")
    noninitial = sum(
        row.get("summary", {}).get("primary_noninitial_nodes", 0)
        for row in report.get("records", [])
    )
    if noninitial != 24:
        raise RuntimeError(f"Expected exactly 24 noninitial nodes, observed {noninitial}")
    return {
        "parent_engine_sha256": sha256(snapshot / PARENT),
        "eight_seed_wrapper_sha256": sha256(snapshot / WRAPPER),
        "protocol_sha256": sha256(snapshot / "protocol.json"),
        "archived_report_sha256": sha256(snapshot / "results/report.json"),
        "attempted": report["attempted"],
        "supporting_seeds": report["supporting_seeds"],
        "noninitial_nodes": noninitial,
        "decision": report["decision"],
    }


def smoke_test(root: Path, work: Path, skip_install: bool) -> dict:
    if not skip_install:
        run([
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "-r",
            str(root / SNAPSHOT_REL / "requirements.txt"),
        ])
    code = f"""
import json, sys, torch
sys.path.insert(0, {str(root / SNAPSHOT_REL)!r})
import gptw_lora_low_response_pareto_cpu as v
from transformers import AutoModelForCausalLM, AutoTokenizer
v.seed_all(86841)
torch.set_num_threads(1)
tok = AutoTokenizer.from_pretrained(v.MODEL_ID)
tok.pad_token = tok.eos_token
tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(v.MODEL_ID)
model.eval()
model.config.use_cache = False
for param in model.parameters():
    param.requires_grad_(False)
adapter = v.NativeLoraB(model, 86841)
theta = torch.nn.Parameter(torch.zeros(adapter.dimension, dtype=torch.float32))
anchors = v.encode_prompts(tok, v.ANCHORS)
response_ids = v.target_ids(tok, v.RESPONSE_WORDS)
row, rank, sigma = v.jacobian(model, adapter, theta, anchors, response_ids)
train, train_y = v.build_task(tok, v.TRAIN_TEMPLATES)
loss = v.task_loss(model, adapter, theta, train, train_y)
loss.backward()
delta = -theta.grad.detach().clone()
projected = v.project_kernel(row, delta)
curve = v.curve(model, adapter, theta.detach(), projected, anchors, response_ids, *v.build_task(tok, v.CAL_TEMPLATES), *v.build_task(tok, v.EVAL_TEMPLATES))
for handle in adapter.handles:
    handle.remove()
print(json.dumps({{"rank": rank, "dimension": adapter.dimension, "curve_points": len(curve), "alpha_grid": v.ALPHAS}}))
"""
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True,
    )
    print(proc.stdout, end="", flush=True)
    return json.loads(proc.stdout.strip().splitlines()[-1])


def validate_full_result(report_path: Path) -> dict:
    report = json.loads(report_path.read_text())
    if report.get("protocol") != PROTOCOL:
        raise RuntimeError(f"Unexpected protocol: {report.get('protocol')}")
    if report.get("decision") != DECISION:
        raise RuntimeError(f"Unexpected decision: {report.get('decision')}")
    if report.get("attempted") != 8 or report.get("supporting_seeds") != 8:
        raise RuntimeError("Expected attempted=8 and supporting_seeds=8")
    noninitial = sum(
        row.get("summary", {}).get("primary_noninitial_nodes", 0)
        for row in report.get("records", [])
    )
    if noninitial != 24:
        raise RuntimeError(f"Expected exactly 24 noninitial nodes, observed {noninitial}")
    return {
        "decision": report["decision"],
        "attempted": report["attempted"],
        "supporting_seeds": report["supporting_seeds"],
        "noninitial_nodes": noninitial,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", default=DEFAULT_WORK)
    parser.add_argument("--source-root", help="Use an existing checkout")
    parser.add_argument("--source-ref", default=DEFAULT_SOURCE_REF)
    parser.add_argument("--source-commit", default=DEFAULT_SOURCE_COMMIT)
    parser.add_argument("--skip-install", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--verify-archived", action="store_true")
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--full", action="store_true")
    args, _ = parser.parse_known_args()

    print(
        "Scientific role: same-cohort external reproduction of the prospectively "
        "frozen 8-seed/24-node GPT-2 native-LoRA-B Pareto audit; not an "
        "independent new-seed confirmation.",
        flush=True,
    )

    work = Path(args.work).resolve()
    work.mkdir(parents=True, exist_ok=True)
    root = acquire_source(work, args.source_root, args.source_ref)
    actual_commit = verify_commit(root, args.source_commit)
    archived = verify_archived(root)
    if args.verify_archived:
        print(json.dumps({"source_commit": actual_commit, **archived}, indent=2), flush=True)
        return

    started = datetime.now(timezone.utc)
    smoke = None
    if args.smoke:
        smoke = smoke_test(root, work, args.skip_install)
        print(json.dumps({"source_commit": actual_commit, "smoke": smoke}, indent=2), flush=True)
        return

    if not args.skip_install:
        run([
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "-r",
            str(root / SNAPSHOT_REL / "requirements.txt"),
        ])
    archived = verify_archived(root, check_imports=True)

    result_dir = work / "gptw_lora_pareto_cpu_8seed_results"
    log_path = work / "GPT2_LORA_PARETO_EXTERNAL_REPRODUCTION.log"
    with log_path.open("w") as log:
        run([
            sys.executable,
            "-u",
            str(root / SNAPSHOT_REL / WRAPPER),
            "--output",
            str(result_dir),
        ], stream=log)
    full = validate_full_result(result_dir / "report.json")
    finished = datetime.now(timezone.utc)
    manifest = {
        "test_role": "same-cohort external reproduction; not an independent new-seed confirmation",
        "repository": REPOSITORY,
        "source_ref": args.source_ref,
        "source_commit": actual_commit,
        "expected_source_commit": args.source_commit or None,
        "protocol": PROTOCOL,
        "started_utc": started.isoformat(),
        "finished_utc": finished.isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        **archived,
        **full,
        "report_sha256": sha256(result_dir / "report.json"),
    }
    (result_dir / "EXTERNAL_REPRODUCTION_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    shutil.copy2(log_path, result_dir / log_path.name)
    archive = Path(shutil.make_archive(str(result_dir), "zip", result_dir.parent, result_dir.name))
    print(json.dumps(manifest, indent=2), flush=True)
    print(f"Result ZIP: {archive}", flush=True)

    try:
        from google.colab import files

        files.download(str(archive))
    except ImportError:
        pass


if __name__ == "__main__":
    main()
