#!/usr/bin/env python3
"""One-upload Colab launcher for Fibre-Qwen R21d."""
import os,shutil,subprocess,sys,zipfile
from pathlib import Path
ROOT=Path("/content") if Path("/content").exists() else Path.cwd()
try:from google.colab import files
except ImportError:files=None
pkg=next(ROOT.glob("fibre_qwen_r21d*.zip"),None)
if pkg is None and files:
 print("Upload fibre_qwen_r21d.zip");up=files.upload();name=next((n for n in up if n.endswith(".zip")),None);pkg=ROOT/name if name else None
if pkg is None:raise SystemExit("fibre_qwen_r21d.zip not found")
run=ROOT/"fibre_qwen_r21d_run";shutil.rmtree(run,ignore_errors=True);run.mkdir()
with zipfile.ZipFile(pkg) as z:z.extractall(run)
print("Installing pinned dependencies...",flush=True)
subprocess.run([sys.executable,"-m","pip","install","-q","transformers==4.56.2","accelerate==1.10.1","safetensors>=0.4.5","sentencepiece>=0.2.0"],check=True)
out=ROOT/"fibre_qwen_r21d_results";shutil.rmtree(out,ignore_errors=True);out.mkdir()
script=next(run.rglob("evaluate_r21d.py"));cmd=[sys.executable,"-u",str(script),"--outdir",str(out)];print("Running:"," ".join(cmd),flush=True)
code=subprocess.run(cmd).returncode
if (out/"run_summary.json").exists():print((out/"run_summary.json").read_text())
else:print("No run_summary.json was produced; inspect the traceback above.")
archive=shutil.make_archive(str(out),"zip",ROOT,out.name)
if files:files.download(archive)
raise SystemExit(code)
