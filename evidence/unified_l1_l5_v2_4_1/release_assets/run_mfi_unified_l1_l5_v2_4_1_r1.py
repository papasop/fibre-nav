#!/usr/bin/env python3
"""Compact Colab launcher for MFI unified L1-L5 v2.4.1-r1."""
import argparse
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

PACKAGE = "moving_fibre_intelligence_unified_l1_l5_v2_4_1_r1.zip"
FOLDER = Path(PACKAGE).stem


def select_archive(requested):
    path = Path(requested)
    if path.is_file() and zipfile.is_zipfile(path):
        return path.resolve()
    from google.colab import files
    print(f"Please select {PACKAGE}", flush=True)
    uploaded = files.upload()
    matches = []
    for name in uploaded:
        stem = Path(name).stem
        if stem == FOLDER or stem.startswith(FOLDER + " ("):
            matches.append(Path("/content") / name)
    if len(matches) != 1 or not zipfile.is_zipfile(matches[0]):
        raise FileNotFoundError(
            f"Expected one {FOLDER}[ (n)].zip; received {list(uploaded)}")
    print("[upload] accepted:", matches[0].name, flush=True)
    return matches[0]


def run_streamed(command, cwd, log_path):
    print("+", " ".join(map(str, command)), flush=True)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            list(map(str, command)), cwd=cwd, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1)
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        return process.wait()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", default=PACKAGE)
    parser.add_argument("--workdir", default="/content/mfi_unified_v2_4_1_r1")
    args, unknown = parser.parse_known_args()
    if unknown:
        print("[notice] ignored notebook arguments:", unknown, flush=True)

    archive = select_archive(args.zip)
    work = Path(args.workdir)
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    with zipfile.ZipFile(archive) as handle:
        handle.extractall(work)
    source = work / FOLDER
    if not source.is_dir():
        raise FileNotFoundError(f"Missing package directory: {source}")

    print("[preflight] python:", sys.version, flush=True)
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-q", "-r",
        source / "requirements.txt"])
    subprocess.call([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"])

    results = work / "results"
    results.mkdir()
    print("[MFI v2.4.1-r1] frozen seeds: 83131, 83161, 83191", flush=True)
    status = run_streamed([
        sys.executable, "-u", source / "run_three_seed_audit.py",
        "--config", source / "config_audit.json", "--output", results,
    ], source, results / "launcher_console.log")

    target = Path("/content/moving_fibre_intelligence_unified_l1_l5_results_v2_4_1_r1.zip")
    shutil.make_archive(str(target.with_suffix("")), "zip", root_dir=results)
    print("RESULT_ZIP=", target, flush=True)
    try:
        from google.colab import files
        files.download(str(target))
    except Exception:
        print("Download manually:", target, flush=True)
    raise SystemExit(status)


if __name__ == "__main__":
    main()
