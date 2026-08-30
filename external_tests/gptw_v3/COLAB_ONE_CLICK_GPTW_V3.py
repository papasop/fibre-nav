#!/usr/bin/env python3
"""Zero-upload external reproduction entry point for GPTW-V3.

The scientific protocol is not modified. This wrapper pins the archived source,
verifies it, runs V3, validates the result schema and returns a provenance-rich
ZIP. A same-cohort rerun is a reproduction, not an independent confirmation.
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
from datetime import datetime, timezone
from pathlib import Path


REPOSITORY = "https://github.com/papasop/neural-fibre-geometry.git"
SOURCE_REF = "v1.4.0-gptw-natural-text-confirmed"
SOURCE_COMMIT = "236f646c472018a7e38be11fd658519763bc2346"
SNAPSHOT_REL = Path("evidence/gptw_response_fibre_v1_1_0")
RUN_REL = SNAPSHOT_REL / "experiments/v3_natural_text/run.py"
PROTOCOL = "CNER_GPT2_LORA_NATURAL_PROMPT_MOVING_FIBRE_V3"


def run(command, *, cwd=None, stream=None):
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


def acquire_source(work: Path, source_root: str | None) -> Path:
    if source_root:
        root = Path(source_root).resolve()
    else:
        root = work / "neural-fibre-geometry"
        run(["git", "clone", "--quiet", REPOSITORY, str(root)])
        run(["git", "checkout", "--quiet", SOURCE_REF], cwd=root)
    actual = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    if actual != SOURCE_COMMIT:
        raise RuntimeError(
            f"Pinned source mismatch: expected {SOURCE_COMMIT}, observed {actual}"
        )
    return root


def validate_report(report_path: Path) -> dict:
    report = json.loads(report_path.read_text())
    if report.get("protocol") != PROTOCOL:
        raise RuntimeError(f"Unexpected protocol: {report.get('protocol')}")
    if report.get("total_interior_nodes") != 32:
        raise RuntimeError("Expected exactly 32 V3 interior nodes")
    summaries = report.get("instance_summaries", [])
    if len(summaries) != 8:
        raise RuntimeError("Expected exactly eight V3 seed summaries")
    if any(len(row.get("node_summaries", [])) != 4 for row in summaries):
        raise RuntimeError("Expected four interior nodes per seed")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", default="/content/gptw_v3_external")
    parser.add_argument("--source-root", help="Use an existing pinned checkout")
    parser.add_argument("--skip-install", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    args, _ = parser.parse_known_args()

    work = Path(args.work).resolve()
    if work.exists() and not args.source_root:
        shutil.rmtree(work)
    work.mkdir(parents=True, exist_ok=True)
    root = acquire_source(work, args.source_root)
    snapshot = root / SNAPSHOT_REL

    run(["sha256sum", "-c", "SHA256SUMS"], cwd=snapshot)
    run([sys.executable, "verify_snapshot.py"], cwd=snapshot)
    if args.preflight_only:
        print("GPTW-V3 external preflight passed; no GPU run requested.")
        return

    if not args.skip_install:
        run([
            sys.executable, "-m", "pip", "install", "-q", "-r",
            str(snapshot / "requirements.txt"),
        ])

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. In Colab select an A100 GPU runtime.")
    gpu = torch.cuda.get_device_name(0)
    if "A100" not in gpu.upper():
        print(f"WARNING: V3 was budgeted for A100; detected {gpu}", flush=True)

    result_dir = work / "cner_gpt2_lora_natural_prompt_moving_fibre_v3_results"
    log_path = work / "GPTW_V3_EXTERNAL_REPRODUCTION.log"
    started = datetime.now(timezone.utc)
    with log_path.open("w") as log:
        run([
            sys.executable, "-u", str(root / RUN_REL),
            "--output", str(result_dir),
        ], stream=log)
    finished = datetime.now(timezone.utc)

    report = validate_report(result_dir / "report.json")
    positive = (
        report.get("decision") == "NATURAL_TEXT_CURRENT_FIBRE_ADVANTAGE_SUPPORTED"
        and report.get("supporting_instances", 0) >= report.get("required_instances", 6)
        and all(row.get("passing_nodes") == 4 for row in report["instance_summaries"])
    )
    manifest = {
        "test_role": "same-cohort external reproduction; not an independent new-cohort confirmation",
        "repository": REPOSITORY,
        "source_ref": SOURCE_REF,
        "source_commit": SOURCE_COMMIT,
        "protocol": PROTOCOL,
        "started_utc": started.isoformat(),
        "finished_utc": finished.isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "gpu": gpu,
        "cuda": torch.version.cuda,
        "decision": report["decision"],
        "supporting_instances": report["supporting_instances"],
        "required_instances": report["required_instances"],
        "total_interior_nodes": report["total_interior_nodes"],
        "strict_reference_reproduction_pass": positive,
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

    if not positive:
        raise RuntimeError(
            "The run completed and was archived, but did not reproduce the frozen positive V3 decision"
        )


if __name__ == "__main__":
    main()
