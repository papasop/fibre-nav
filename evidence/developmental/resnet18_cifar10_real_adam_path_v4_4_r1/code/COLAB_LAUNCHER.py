from pathlib import Path
import subprocess, sys, threading

try:
    from google.colab import files
except ImportError as exc:
    raise RuntimeError("Run this launcher in Google Colab") from exc

root = Path(__file__).resolve().parent
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "-r", str(root / "requirements.txt")],
    check=True,
)
out = Path("/content/cner_resnet18_cifar10_real_optimizer_path_v4_4_r1_results")
log = Path("/content/CNER_REAL_ADAM_V4_4_R1_RUN.log")
cmd = [sys.executable, "-u", str(root / "cner_resnet18_cifar10_real_optimizer_path_v4_4_r1.py"), "--output", str(out)]
with log.open("w", encoding="utf-8") as stream:
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    timed_out = [False]
    def stop_process():
        timed_out[0] = True
        proc.kill()
    timer = threading.Timer(3300, stop_process)
    timer.start()
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="")
        stream.write(line); stream.flush()
    return_code = proc.wait()
    timer.cancel()
if timed_out[0]:
    files.download(str(log))
    raise TimeoutError("strict 3300-second audit timeout; log downloaded")
if return_code:
    files.download(str(log))
    raise RuntimeError(f"audit exited with code {return_code}; log downloaded")
result = Path(str(out) + ".zip")
if not result.exists():
    files.download(str(log))
    raise FileNotFoundError("result ZIP missing; log downloaded")
files.download(str(result))
