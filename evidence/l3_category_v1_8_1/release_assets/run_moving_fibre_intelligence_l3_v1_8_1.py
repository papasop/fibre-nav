#!/usr/bin/env python3
"""Robust Colab launcher for Moving Fibre Intelligence L3 v1.8.1."""
import argparse
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path


def stream(cmd, cwd, log_path):
    print("+", " ".join(map(str, cmd)), flush=True)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    with open(log_path, "w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            list(map(str, cmd)), cwd=cwd, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            log.write(line); log.flush()
        code = proc.wait()
    return code


def colab_download(path):
    try:
        from google.colab import files
        files.download(str(path))
    except Exception:
        print("Download manually:", path)


def choose_zip(requested, explicit):
    z = Path(requested)
    if explicit and z.exists():
        return z
    try:
        from google.colab import files
        print("Please select moving_fibre_intelligence_l3_v1_8_1.zip")
        uploaded = files.upload()
        names = [name for name in uploaded if name.lower().endswith(".zip")]
        if not names:
            raise FileNotFoundError("No ZIP was uploaded")
        return Path("/content") / names[0]
    except ImportError:
        raise FileNotFoundError(f"ZIP not found: {z}")


def main():
    raw_args=sys.argv[1:]
    explicit_zip="--zip" in raw_args
    p = argparse.ArgumentParser()
    p.add_argument("--zip", default="/content/moving_fibre_intelligence_l3_v1_8_1.zip")
    p.add_argument("--workdir", default="/content/moving_fibre_intelligence_l3_run_v1_8_1")
    args, unknown = p.parse_known_args()
    if unknown:
        print("[notice] ignored notebook arguments:", unknown)

    z = choose_zip(args.zip,explicit_zip)
    work = Path(args.workdir)
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    with zipfile.ZipFile(z) as f:
        f.extractall(work)
    roots = list(work.glob("*/fibre_memory_audit.py"))
    if len(roots) != 1:
        raise RuntimeError("Expected exactly one fibre_memory_audit.py in ZIP")
    root = roots[0].parent

    print("[preflight] python:", sys.version)
    print("[preflight] executable:", sys.executable)
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "--no-cache-dir",
         "-r", str(root / "requirements.txt")], check=True
    )
    print("[preflight] removing optional torchao to avoid PEFT dispatcher conflicts", flush=True)
    subprocess.run(
        [sys.executable, "-m", "pip", "uninstall", "-y", "torchao"],
        check=False, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
    )
    subprocess.run(
        [sys.executable, "-c",
         "import torch,transformers,peft; print('[preflight] torch',torch.__version__,'cuda',torch.cuda.is_available(),'transformers',transformers.__version__,'peft',peft.__version__)"],
        check=True,
    )
    subprocess.run(
        [sys.executable, "-c",
         "from transformers import GPT2Config,GPT2LMHeadModel;"
         "from peft import LoraConfig,get_peft_model;"
         "m=GPT2LMHeadModel(GPT2Config(vocab_size=128,n_positions=16,n_ctx=16,n_embd=32,n_layer=2,n_head=4));"
         "m=get_peft_model(m,LoraConfig(r=2,lora_alpha=4,target_modules=['c_attn'],task_type='CAUSAL_LM'));"
         "print('[preflight] PEFT GPT-2 LoRA injection: OK')"],
        check=True,
    )
    subprocess.run(
        [sys.executable, "-c",
         "import importlib.util,json,numpy as np,sys;"
         "s=importlib.util.spec_from_file_location('audit',sys.argv[1]);"
         "m=importlib.util.module_from_spec(s);s.loader.exec_module(m);"
         "json.dumps({'numpy_bool':np.bool_(True),'numpy_float':np.float64(1)},default=m.json_default);"
         "print('[preflight] JSON scalar serialization: OK')",
         str(root / "fibre_memory_audit.py")],
        check=True,
    )

    results = work / "results"
    results.mkdir(exist_ok=True)
    log_path = results / "console.log"
    cmd = [sys.executable, "-u", root / "fibre_memory_audit.py",
           "--config", root / "config_quick.json", "--output", results]
    print("[MFI L3 v1.8.1 shared category-conditioned writer] frozen seed: 82001", flush=True)
    code = stream(cmd, root, log_path)

    archive = Path(shutil.make_archive("/content/moving_fibre_intelligence_l3_results_v1_8_1", "zip", results))
    if code != 0:
        print(f"\n[FAILED] experiment exit code={code}")
        print("The complete root traceback is above and in console.log.")
        print("DIAGNOSTIC_ZIP=", archive)
        colab_download(archive)
        return
    print("RESULT_ZIP=", archive)
    colab_download(archive)


if __name__ == "__main__":
    main()
