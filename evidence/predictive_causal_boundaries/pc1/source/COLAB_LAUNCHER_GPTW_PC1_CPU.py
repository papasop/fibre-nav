#!/usr/bin/env python3
"""Upload-and-run Colab CPU launcher for the frozen GPTW-PC1 package."""
from __future__ import annotations
import hashlib
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path("/content") if Path("/content").exists() else Path.cwd()
print("=" * 78)
print("GPTW-PC1 predictive response-fibre capacity audit — CPU")
print("No GPU required; frozen hard limit: two hours.")
print("=" * 78)
try:
    from google.colab import files
except Exception as exc:
    raise RuntimeError("Run this launcher inside Google Colab.") from exc

print("Upload gptw_pc1_predictive_capacity_v1.zip")
uploaded = files.upload()
if len(uploaded) != 1:
    raise RuntimeError("Upload exactly one GPTW-PC1 source ZIP.")
name, payload = next(iter(uploaded.items()))
if not name.lower().endswith(".zip") or "pc1" not in name.lower():
    raise RuntimeError(f"Unexpected package: {name}")
source = ROOT / name
source.write_bytes(payload)
run_root = ROOT / "gptw_pc1_external_run"
if run_root.exists():
    shutil.rmtree(run_root)
run_root.mkdir(parents=True)
with zipfile.ZipFile(source) as archive:
    base = run_root.resolve()
    for member in archive.infolist():
        target = (run_root / member.filename).resolve()
        if target != base and base not in target.parents:
            raise RuntimeError(f"Unsafe ZIP member: {member.filename}")
    archive.extractall(run_root)

subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q", "-r",
    str(run_root / "requirements.txt"),
])
output = ROOT / "gptw_pc1_predictive_capacity_results"
log_path = ROOT / "GPTW_PC1_CPU.log"
with log_path.open("w", encoding="utf-8") as log:
    proc = subprocess.Popen(
        [sys.executable, "-u", str(run_root / "gptw_pc1_predictive_capacity.py"),
         "--output", str(output)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="")
        log.write(line)
        log.flush()
    code = proc.wait()
if code:
    files.download(str(log_path))
    raise RuntimeError(f"GPTW-PC1 exited with code {code}; log downloaded.")
shutil.copy2(log_path, output / log_path.name)
result = Path(shutil.make_archive(str(output), "zip", output.parent, output.name))
print("RESULT READY:", result)
print("SHA-256:", hashlib.sha256(result.read_bytes()).hexdigest())
files.download(str(result))
