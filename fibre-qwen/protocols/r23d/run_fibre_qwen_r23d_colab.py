#!/usr/bin/env python3
"""Fail-closed Colab launcher for Fibre-Qwen R23d."""
import hashlib, json, os, shutil, subprocess, sys, time, zipfile
from pathlib import Path

ROOT = Path("/content") if Path("/content").exists() else Path.cwd()
EXPECTED_PROTOCOL = "FIBRE_QWEN_GENERAL_MOVING_RESPONSE_KERNEL_R23D_CONFIRMATORY"
try:
    from google.colab import files
except ImportError:
    files = None

pkg = next(ROOT.glob("fibre_qwen_r23d*.zip"), None)
if pkg is None and files:
    print("Upload fibre_qwen_r23d.zip", flush=True)
    uploaded = files.upload()
    name = next((n for n in uploaded if n.lower().endswith(".zip")), None)
    pkg = ROOT/name if name else None
if pkg is None or not pkg.is_file(): raise SystemExit("R23d package ZIP not found")
print("Package:", pkg.name, "sha256=", hashlib.sha256(pkg.read_bytes()).hexdigest(), flush=True)
run = ROOT/"fibre_qwen_r23d_run"; shutil.rmtree(run, ignore_errors=True); run.mkdir()
with zipfile.ZipFile(pkg) as archive:
    bad = archive.testzip()
    if bad: raise SystemExit(f"Corrupt ZIP member: {bad}")
    base = run.resolve()
    for member in archive.infolist():
        target = (run/member.filename).resolve()
        if target != base and base not in target.parents: raise SystemExit(f"Unsafe ZIP path: {member.filename}")
    archive.extractall(run)
scripts = [p for p in run.rglob("fibre_qwen_r23d.py") if "__pycache__" not in p.parts]
reqs = [p for p in run.rglob("requirements.txt") if "__pycache__" not in p.parts]
if len(scripts) != 1 or len(reqs) != 1: raise SystemExit(f"Expected one benchmark and requirements file: {scripts}, {reqs}")
script, req = scripts[0], reqs[0]
source = script.read_text(encoding="utf-8")
if EXPECTED_PROTOCOL not in source or 'if __name__ == "__main__"' not in source: raise SystemExit("Benchmark identity preflight failed")
print("Installing pinned dependencies...", flush=True)
subprocess.run([sys.executable,"-m","pip","install","-q","-r",str(req)], check=True)
out = ROOT/"fibre_qwen_r23d_results"; shutil.rmtree(out, ignore_errors=True); out.mkdir()
cmd = [sys.executable,"-u",str(script),"--device","cuda","--outdir",str(out)]
print("Running:", " ".join(cmd), flush=True); started = time.time()
with (out/"child_combined.log").open("w", encoding="utf-8") as log:
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                            env={**os.environ,"PYTHONUNBUFFERED":"1"})
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="", flush=True); log.write(line); log.flush()
    code = proc.wait()
summary_path = out/"run_summary.json"
launcher = {"protocol":"FIBRE_QWEN_R23D_FAIL_CLOSED_LAUNCHER", "child_exit_code":code,
            "child_wall_seconds":time.time()-started, "summary_exists":summary_path.exists(),
            "benchmark_sha256":hashlib.sha256(script.read_bytes()).hexdigest()}
final_code = code
if not summary_path.exists():
    launcher["status"] = "FAIL_NO_RUN_SUMMARY"; final_code = final_code or 3
    print("FAIL-CLOSED: no run_summary.json; inspect child_combined.log", flush=True)
else:
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8")); launcher["summary_protocol"] = summary.get("protocol")
        if summary.get("protocol") != EXPECTED_PROTOCOL:
            launcher["status"] = "FAIL_PROTOCOL_MISMATCH"; final_code = final_code or 4
        else: launcher["status"] = "SUMMARY_VALIDATED"
        print(summary_path.read_text(encoding="utf-8"), flush=True)
    except Exception as exc:
        launcher["status"] = f"FAIL_INVALID_SUMMARY:{type(exc).__name__}:{exc}"; final_code = final_code or 5
(out/"launcher_summary.json").write_text(json.dumps(launcher, indent=2)+"\n", encoding="utf-8")
result_zip = shutil.make_archive(str(out), "zip", ROOT, out.name)
print("Results archive:", result_zip, flush=True)
if files: files.download(result_zip)
raise SystemExit(final_code)
