#!/usr/bin/env python3
"""Robust Colab launcher for GPT-2 Fibre Memory v1.3.2."""
import argparse
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path


def stream(cmd, cwd, log_path):
    print("+", " ".join(map(str, cmd)), flush=True)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    with open(log_path, "w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            list(map(str, cmd)), cwd=cwd, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            log.write(line); log.flush()
        code = proc.wait()
    return code


def colab_download(path):
    try:
        from google.colab import files
        files.download(str(path))
    except Exception:
        print("Download manually:", path)


def choose_zip(requested):
    z = Path(requested)
    if z.exists():
        return z
    candidates = sorted(
        Path("/content").glob("gpt2_fibre_memory_v1_3_2*.zip"),
        key=lambda p: p.stat().st_mtime, reverse=True,
    )
    if candidates:
        print("[notice] using uploaded ZIP:", candidates[0])
        return candidates[0]
    try:
        from google.colab import files
        print("Please select gpt2_fibre_memory_v1_3_2.zip")
        uploaded = files.upload()
        names = [name for name in uploaded if name.lower().endswith(".zip")]
        if not names:
            raise FileNotFoundError("No ZIP was uploaded")
        return Path("/content") / names[0]
    except ImportError:
        raise FileNotFoundError(f"ZIP not found: {z}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--zip", default="/content/gpt2_fibre_memory_v1_3_2.zip")
    p.add_argument("--mode", choices=["smoke", "quick"], default="smoke")
    p.add_argument("--workdir", default="/content/fibre_memory_run_v1_3_2")
    args, unknown = p.parse_known_args()
    if unknown:
        print("[notice] ignored notebook arguments:", unknown)

    z = choose_zip(args.zip)
    work = Path(args.workdir)
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    with zipfile.ZipFile(z) as f:
        f.extractall(work)
    roots = list(work.glob("*/fibre_memory_audit.py"))
    if len(roots) != 1:
        raise RuntimeError("Expected exactly one fibre_memory_audit.py in ZIP")
    root = roots[0].parent

    print("[preflight] python:", sys.version)
    print("[preflight] executable:", sys.executable)
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "--no-cache-dir",
         "-r", str(root / "requirements.txt")], check=True
    )
    subprocess.run(
        [sys.executable, "-c",
         "import torch,transformers,peft; print('[preflight] torch',torch.__version__,'cuda',torch.cuda.is_available(),'transformers',transformers.__version__,'peft',peft.__version__)"],
        check=True,
    )

    results = work / "results"
    results.mkdir(exist_ok=True)
    log_path = results / "console.log"
    cmd = [sys.executable, "-u", root / "fibre_memory_audit.py",
           "--config", root / "config_quick.json", "--output", results]
    if args.mode == "smoke":
        cmd += ["--seeds", "81401"]
    code = stream(cmd, root, log_path)

    archive = Path(shutil.make_archive("/content/fibre_memory_results_v1_3_2", "zip", results))
    if code != 0:
        print(f"\n[FAILED] experiment exit code={code}")
        print("The complete root traceback is above and in console.log.")
        print("DIAGNOSTIC_ZIP=", archive)
        colab_download(archive)
        return
    print("RESULT_ZIP=", archive)
    colab_download(archive)


if __name__ == "__main__":
    main()

