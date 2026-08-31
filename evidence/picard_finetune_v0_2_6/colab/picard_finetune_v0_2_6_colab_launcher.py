#!/usr/bin/env python3
"""Single-process Colab launcher for intrinsic Picard v0.2.6."""
from __future__ import annotations
import glob,hashlib,json,runpy,shutil,sys,traceback,zipfile
from pathlib import Path
from google.colab import files

EXPECTED="f08dcc4258ba5bb3e102b87b2748ca06279a5acff63fef0c2b9bb6ee92d9c044"
WORK=Path("/content/picard_v0_2_6_run");OUT=Path("/content/picard_v0_2_6_results")
def sha(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
print("Upload picard_finetune_demo_v0_2_6.zip",flush=True)
up=files.upload();zs=[Path(x) for x in up if x.endswith(".zip")]
if len(zs)!=1:raise RuntimeError("Upload exactly one v0.2.6 ZIP")
if sha(zs[0])!=EXPECTED:raise RuntimeError("v0.2.6 ZIP SHA-256 mismatch")
for p in (WORK,OUT):
 if p.exists():shutil.rmtree(p)
WORK.mkdir(parents=True);zipfile.ZipFile(zs[0]).extractall(WORK)
scripts=glob.glob(str(WORK/"**"/"picard_finetune_benchmark_v0_2_6.py"),recursive=True)
if len(scripts)!=1:raise RuntimeError(f"Expected one benchmark, found {len(scripts)}")
import torch
if not torch.cuda.is_available():raise RuntimeError("Select a Colab GPU runtime")
print("GPU:",torch.cuda.get_device_name(0),flush=True)
sys.argv=[scripts[0],"--device","cuda","--outdir",str(OUT),"--data-root","/content/data"]
code=0
try:runpy.run_path(scripts[0],run_name="__main__")
except SystemExit as e:code=int(e.code or 0)
except Exception:
 code=3;traceback.print_exc()
summary=OUT/"run_summary.json"
if summary.exists():
 d=json.loads(summary.read_text());print("STATUS:",d.get("scientific_status"));print("EQUAL-LOSS SPEEDUP:",d.get("median_time_to_equal_loss_speedup_fraction"));print("STEP REDUCTION:",d.get("median_step_reduction_fraction"));print("FIXED-BUDGET SPEEDUP:",d.get("median_fixed_budget_speedup_fraction"))
 z=Path(shutil.make_archive("/content/picard_v0_2_6_results","zip",root_dir=OUT));files.download(str(z))
else:print("No run_summary.json was produced; use the traceback above.")
if code:print("v0.2.6 did not pass every frozen gate or stopped in preflight.")
