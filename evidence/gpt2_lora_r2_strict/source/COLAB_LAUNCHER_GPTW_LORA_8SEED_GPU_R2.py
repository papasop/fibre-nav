#!/usr/bin/env python3
"""One-click Colab launcher for GPTW Pareto GPU R2 strict audit."""
from pathlib import Path
import hashlib,shutil,subprocess,sys,zipfile
ROOT=Path("/content") if Path("/content").exists() else Path.cwd()
print("="*78);print("GPTW GPT-2/native-LoRA-B Pareto — GPU V1-R2 strict")
print("CUDA required; A100 optional; frozen hard limit: two hours.");print("="*78)
try:from google.colab import files
except Exception as exc:raise RuntimeError("Run inside Google Colab.") from exc
try:gpu=subprocess.check_output(["nvidia-smi","--query-gpu=name","--format=csv,noheader"],text=True).strip().splitlines()[0]
except Exception as exc:raise RuntimeError("No CUDA GPU. Select Runtime > Change runtime type > GPU.") from exc
print("GPU:",gpu);print("Upload gptw_gpt2_lora_pareto_gpu_8seed_v1_r2_strict.zip")
uploaded=files.upload()
if len(uploaded)!=1:raise RuntimeError("Upload exactly one R2 source ZIP.")
name,payload=next(iter(uploaded.items()))
if not name.lower().endswith(".zip") or "r2_strict" not in name.lower():raise RuntimeError(f"Unexpected package: {name}")
source=ROOT/name;source.write_bytes(payload);run=ROOT/"gptw_lora_gpu_r2_run"
if run.exists():shutil.rmtree(run)
run.mkdir(parents=True)
with zipfile.ZipFile(source) as z:
    base=run.resolve()
    for member in z.infolist():
        target=(run/member.filename).resolve()
        if target!=base and base not in target.parents:raise RuntimeError(f"Unsafe ZIP member: {member.filename}")
    z.extractall(run)
subprocess.check_call([sys.executable,"-m","pip","install","-q","-r",str(run/"requirements.txt")])
output=ROOT/"gptw_lora_pareto_gpu_8seed_r2_results";log=ROOT/"GPTW_LORA_PARETO_GPU_R2.log"
with log.open("w",encoding="utf-8") as stream:
    proc=subprocess.Popen([sys.executable,"-u",str(run/"gptw_lora_pareto_gpu_8seed_r2.py"),"--output",str(output)],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    assert proc.stdout is not None
    for line in proc.stdout:print(line,end="");stream.write(line);stream.flush()
    code=proc.wait()
if code:
    files.download(str(log));raise RuntimeError(f"R2 exited with code {code}; log downloaded.")
shutil.copy2(log,output/log.name)
result=Path(shutil.make_archive(str(output),"zip",output.parent,output.name))
print("RESULT READY:",result);print("SHA-256:",hashlib.sha256(result.read_bytes()).hexdigest());files.download(str(result))
