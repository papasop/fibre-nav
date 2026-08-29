#!/usr/bin/env python3
from pathlib import Path
import shutil, subprocess, sys, zipfile

try:
    from google.colab import files
except ImportError as exc:
    raise RuntimeError("请在 Google Colab 中运行") from exc

expected = "cner_gpt2_lora_natural_prompt_moving_fibre_v3.zip"
print(f"请先选择 A100 GPU，然后上传：{expected}")
uploaded = files.upload()
archives = [Path(name) for name in uploaded if name.endswith(".zip")]
if len(archives) != 1: raise RuntimeError(f"必须且只能上传一个 ZIP，当前为 {len(archives)}")
root = Path("/content/gptw_v3_run")
if root.exists(): shutil.rmtree(root)
root.mkdir(parents=True)
with zipfile.ZipFile(archives[0]) as zf: zf.extractall(root)
scripts = list(root.rglob("cner_gpt2_lora_natural_prompt_moving_fibre_v3.py"))
if len(scripts) != 1: raise RuntimeError(f"主程序数量异常：{len(scripts)}；请确认上传的是 {expected}")
subprocess.run([sys.executable,"-m","pip","install","-q","transformers>=4.45,<5","accelerate>=0.34"],check=True)
out = Path("/content/cner_gpt2_lora_natural_prompt_moving_fibre_v3_results")
log = Path("/content/GPTW_V3_RUN.log")
with log.open("w") as stream:
    proc = subprocess.run([sys.executable,"-u",str(scripts[0]),"--output",str(out)],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    stream.write(proc.stdout); print(proc.stdout)
if proc.returncode:
    files.download(str(log)); raise RuntimeError(f"主程序退出码 {proc.returncode}；请上传自动下载的日志")
result = Path(str(out)+".zip")
if not result.exists(): raise RuntimeError("结果 ZIP 未生成")
files.download(str(result))
