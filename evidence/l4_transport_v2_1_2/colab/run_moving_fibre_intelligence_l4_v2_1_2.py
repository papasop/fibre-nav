#!/usr/bin/env python3
"""Robust Colab launcher for MFI L4 large-atlas confirmation v2.1.2."""
import argparse
import os
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

PACKAGE_NAME = "moving_fibre_intelligence_l4_v2_1_2.zip"
ROOT_NAME = "moving_fibre_intelligence_l4_v2_1_2"
EXPECTED_ENTRY = f"{ROOT_NAME}/run_l4_confirmation.py"


def validate(path):
    if not zipfile.is_zipfile(path):
        raise zipfile.BadZipFile(f"Invalid ZIP: {path}")
    with zipfile.ZipFile(path) as zf:
        if EXPECTED_ENTRY not in zf.namelist():
            raise zipfile.BadZipFile(f"Wrong package; missing {EXPECTED_ENTRY}")
    return path


def choose_zip(requested, explicit):
    candidate = Path(requested)
    if explicit and candidate.is_file():
        return validate(candidate)
    try:
        from google.colab import files
        print(f"Please select {PACKAGE_NAME}")
        uploaded = files.upload()
        expected = Path(PACKAGE_NAME)
        pattern = re.compile(
            rf"^{re.escape(expected.stem)}(?: \(\d+\))?{re.escape(expected.suffix)}$")
        matches = [name for name in uploaded if pattern.fullmatch(Path(name).name)]
        if not matches:
            got = ", ".join(uploaded) if uploaded else "nothing"
            raise FileNotFoundError(
                f"Expected {PACKAGE_NAME} (numeric suffix allowed); received {got}")
        selected = validate(Path("/content") / matches[-1])
        print(f"[upload] accepted: {selected.name}")
        return selected
    except ImportError:
        if candidate.is_file():
            return validate(candidate)
        raise FileNotFoundError(f"ZIP not found: {candidate}")


def stream(cmd, cwd, log_path):
    print("+", " ".join(map(str, cmd)), flush=True)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    with open(log_path, "w", encoding="utf-8") as log:
        proc = subprocess.Popen(list(map(str, cmd)), cwd=cwd, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        return proc.wait()


def download(path):
    try:
        from google.colab import files
        files.download(str(path))
    except Exception:
        print("Download manually:", path)


def main():
    raw = sys.argv[1:]
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", default=f"/content/{PACKAGE_NAME}")
    parser.add_argument("--workdir", default="/content/mfi_l4_run_v2_1_2")
    args, unknown = parser.parse_known_args()
    if unknown:
        print("[notice] ignored notebook arguments:", unknown)
    archive_in = choose_zip(args.zip, "--zip" in raw)
    work = Path(args.workdir)
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    with zipfile.ZipFile(archive_in) as zf:
        zf.extractall(work)
    root = work / ROOT_NAME
    print("[preflight] python:", sys.version)
    print("[preflight] executable:", sys.executable)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-cache-dir",
                    "-r", str(root / "requirements.txt")], check=True)
    print("[preflight] removing optional torchao to avoid PEFT dispatcher conflicts", flush=True)
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"],
                   check=False, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    subprocess.run([sys.executable, "-c",
        "import torch,transformers,peft; print('[preflight] torch',torch.__version__,"
        "'cuda',torch.cuda.is_available(),'transformers',transformers.__version__,'peft',peft.__version__)"],
        check=True)
    results = work / "results"
    results.mkdir(exist_ok=True)
    print("[MFI L4 v2.1.2 large-atlas confirmation] frozen seed: 82803", flush=True)
    cmd = [sys.executable, "-u", root / "run_l4_confirmation.py",
           "--config", root / "config_audit.json", "--output", results]
    code = stream(cmd, root, results / "console.log")
    archive = Path(shutil.make_archive(
        "/content/moving_fibre_intelligence_l4_results_v2_1_2", "zip", results))
    if code:
        print(f"\n[FAILED] experiment exit code={code}")
        print("The complete traceback is above and in console.log.")
        print("DIAGNOSTIC_ZIP=", archive)
    else:
        print("RESULT_ZIP=", archive)
    download(archive)


if __name__ == "__main__":
    main()
