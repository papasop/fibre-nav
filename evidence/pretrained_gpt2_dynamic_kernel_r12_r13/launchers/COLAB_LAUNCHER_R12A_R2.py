#!/usr/bin/env python3
import os,shutil,subprocess,sys,zipfile
from pathlib import Path
subprocess.run([sys.executable,"-m","pip","uninstall","-y","-q","torchao"],check=False)
subprocess.run([sys.executable,"-m","pip","install","-q","transformers==4.56.2"],check=True)
if os.path.exists("/content"):
 from google.colab import files
 print("Upload picard_gpt2_pretrained_dynamic_r12a_r2.zip",flush=True); z=next(iter(files.upload()))
else: z="picard_gpt2_pretrained_dynamic_r12a_r2.zip"
root=Path("/content/r12a_r2_run" if os.path.exists("/content") else "r12a_r2_run"); root.mkdir(parents=True,exist_ok=True)
with zipfile.ZipFile(z) as f:f.extractall(root)
scripts=list(root.rglob("picard_gpt2_pretrained_r12a_r2.py")); assert len(scripts)==1,scripts
out=Path("/content/picard_r12a_r2_results" if os.path.exists("/content") else "picard_r12a_r2_results")
forward=["--quick"] if "--quick" in sys.argv[1:] else []
cmd=[sys.executable,str(scripts[0]),"--device","cuda","--outdir",str(out),"--data-root","/content/data" if os.path.exists("/content") else "data",*forward]
gpu=subprocess.run(["nvidia-smi","--query-gpu=name","--format=csv,noheader"],capture_output=True,text=True).stdout.strip()
print("GPU:",gpu,flush=True); print("Running:"," ".join(cmd),flush=True); code=subprocess.run(cmd).returncode
if (out/"run_summary.json").exists(): print((out/"run_summary.json").read_text())
if out.is_dir():
 archive=shutil.make_archive(str(out),"zip",out)
 if os.path.exists("/content"):
  from google.colab import files
  files.download(archive)
raise SystemExit(code)
