#!/usr/bin/env python3
"""Colab launcher for MFI L3 reassignment margin polish v2.3.2.1."""
import argparse
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

PACKAGE = "moving_fibre_intelligence_l1_l5_independent_smoke_v2_3_2_1.zip"
STEM = Path(PACKAGE).stem


def run(cmd, cwd=None):
    print("+", " ".join(map(str, cmd)), flush=True)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    return subprocess.call(list(map(str, cmd)), cwd=cwd, env=env)


def choose_zip(requested):
    candidate = Path(requested)
    if candidate.is_file() and zipfile.is_zipfile(candidate):
        return candidate.resolve()
    try:
        from google.colab import files
        print(f"Please select {PACKAGE}", flush=True)
        uploaded = files.upload()
        matches = [Path("/content") / name for name in uploaded
                   if Path(name).stem == STEM or Path(name).stem.startswith(STEM + " (")]
        if len(matches) != 1 or not zipfile.is_zipfile(matches[0]):
            raise FileNotFoundError(f"Expected one {STEM}[ (n)].zip; received {list(uploaded)}")
        print("[upload] accepted:", matches[0].name, flush=True)
        return matches[0]
    except ImportError as exc:
        raise FileNotFoundError(f"Provide --zip /path/to/{PACKAGE}") from exc


def download(path):
    try:
        from google.colab import files
        files.download(str(path))
    except Exception:
        print("Download manually:", path, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", default=PACKAGE)
    ap.add_argument("--workdir", default="/content/mfi_joint_l1_l5_smoke_v2_3_2_1")
    args, unknown = ap.parse_known_args()
    if unknown:
        print("[notice] ignored notebook arguments:", unknown, flush=True)
    archive = choose_zip(args.zip)
    work = Path(args.workdir)
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(work)
    source = work / STEM
    if not source.is_dir():
        raise FileNotFoundError(f"Missing package directory: {source}")
    print("[preflight] python:", sys.version, flush=True)
    run([sys.executable, "-m", "pip", "install", "-q", "-r", source / "requirements.txt"])
    run([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"])
    results = work / "results"
    results.mkdir()
    print("[MFI joint L1-L5 v2.3.2.1 margin polish] development seed: 83001", flush=True)
    code = run([sys.executable, "-u", source / "run_joint_smoke.py",
                "--config", source / "config_audit.json", "--output", results])
    output = Path("/content/moving_fibre_intelligence_l1_l5_independent_smoke_results_v2_3_2_1.zip")
    shutil.make_archive(str(output.with_suffix("")), "zip", root_dir=results)
    print("RESULT_ZIP=", output, flush=True)
    download(output)
    raise SystemExit(code)


if __name__ == "__main__":
    main()
