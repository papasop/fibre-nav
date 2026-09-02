#!/usr/bin/env python3
"""One-upload, CPU-only Colab launcher for R21c-r1."""
import shutil,subprocess,sys,zipfile
from pathlib import Path
ROOT=Path("/content") if Path("/content").exists() else Path.cwd()
try:from google.colab import files
except ImportError:files=None
pkg=next(ROOT.glob("fibre_qwen_r21c_r1_evaluator_audit*.zip"),None)
if pkg is None and files:
 print("Upload fibre_qwen_r21c_r1_evaluator_audit.zip");u=files.upload();n=next((x for x in u if x.endswith(".zip")),None);pkg=ROOT/n if n else None
if pkg is None:raise SystemExit("Package ZIP not found")
run=ROOT/"fibre_qwen_r21c_r1_run";shutil.rmtree(run,ignore_errors=True);run.mkdir()
with zipfile.ZipFile(pkg) as z:z.extractall(run)
out=ROOT/"fibre_qwen_r21c_r1_results";shutil.rmtree(out,ignore_errors=True);out.mkdir();s=next(run.rglob("audit_evaluator_r21c_r1.py"))
cmd=[sys.executable,"-u",str(s),"--outdir",str(out)];print("Running:"," ".join(cmd));code=subprocess.run(cmd).returncode
if (out/"run_summary.json").exists():print((out/"run_summary.json").read_text())
archive=shutil.make_archive(str(out),"zip",ROOT,out.name)
if files:files.download(archive)
raise SystemExit(0 if code in (0,2) else code)
