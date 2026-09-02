#!/usr/bin/env python3
"""One-upload, CPU-only Colab launcher for R21d-r1."""
import shutil,subprocess,sys,zipfile
from pathlib import Path
ROOT=Path("/content") if Path("/content").exists() else Path.cwd()
try:from google.colab import files
except ImportError:files=None
pkg=next(ROOT.glob("fibre_qwen_r21d_r1*.zip"),None)
if pkg is None and files:
 print("Upload fibre_qwen_r21d_r1.zip");up=files.upload();name=next((n for n in up if n.endswith(".zip")),None);pkg=ROOT/name if name else None
if pkg is None:raise SystemExit("fibre_qwen_r21d_r1.zip not found")
run=ROOT/"fibre_qwen_r21d_r1_run";shutil.rmtree(run,ignore_errors=True);run.mkdir()
with zipfile.ZipFile(pkg) as z:z.extractall(run)
out=ROOT/"fibre_qwen_r21d_r1_results";shutil.rmtree(out,ignore_errors=True);out.mkdir()
script=next(run.rglob("audit_r21d_r1.py"));cmd=[sys.executable,"-u",str(script),"--outdir",str(out)];print("Running:"," ".join(cmd),flush=True);code=subprocess.run(cmd).returncode
if (out/"audit_summary.json").exists():print((out/"audit_summary.json").read_text())
archive=shutil.make_archive(str(out),"zip",ROOT,out.name)
if files:files.download(archive)
raise SystemExit(code)
