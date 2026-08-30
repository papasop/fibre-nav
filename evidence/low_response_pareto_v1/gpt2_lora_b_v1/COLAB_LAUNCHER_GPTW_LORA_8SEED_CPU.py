#!/usr/bin/env python3
"""One-click Colab CPU launcher for GPTW GPT-2/LoRA eight-seed confirmation."""
from pathlib import Path
import hashlib,shutil,subprocess,sys,zipfile
ROOT=Path("/content") if Path("/content").exists() else Path.cwd()
print("="*78);print("GPTW GPT-2/native-LoRA-B Pareto — CPU 8-new-seed confirmation");print("No GPU required; allow up to 90 minutes.");print("="*78)
try:from google.colab import files
except Exception as exc:raise RuntimeError("Run inside Google Colab.") from exc
print("Upload gptw_gpt2_lora_pareto_cpu_8seed_v1_r1.zip");uploaded=files.upload()
if len(uploaded)!=1:raise RuntimeError("Upload exactly one source ZIP.")
name,payload=next(iter(uploaded.items()))
if not name.lower().endswith(".zip") or "8seed" not in name.lower():raise RuntimeError(f"Unexpected package: {name}")
source=ROOT/name;source.write_bytes(payload);run=ROOT/"gptw_lora_8seed_cpu_run"
if run.exists():shutil.rmtree(run)
run.mkdir(parents=True)
with zipfile.ZipFile(source) as z:
    base=run.resolve()
    for m in z.infolist():
        target=(run/m.filename).resolve()
        if target!=base and base not in target.parents:raise RuntimeError(f"Unsafe ZIP member: {m.filename}")
    z.extractall(run)
subprocess.check_call([sys.executable,"-m","pip","install","-q","-r",str(run/"requirements.txt")])
output=ROOT/"gptw_lora_pareto_cpu_8seed_results"
subprocess.check_call([sys.executable,"-u",str(run/"gptw_lora_pareto_cpu_8seed.py"),"--output",str(output)])
result=Path(str(output)+".zip");print("RESULT READY:",result);print("SHA-256:",hashlib.sha256(result.read_bytes()).hexdigest());files.download(str(result))
