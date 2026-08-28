from google.colab import files
from pathlib import Path
import shutil, subprocess, sys, zipfile

PACKAGE = "cner_resnet18_cifar10_fibre_external_precision_confirm_v4_0c_r1.zip"
ROOT = Path("/content/cner_resnet18_cifar10_fibre_external_precision_confirm_v4_0c_r1")
RESULT = Path("/content/cner_resnet18_cifar10_fibre_external_precision_confirm_v4_0c_r1_results.zip")
LOG = Path("/content/cner_resnet18_cifar10_fibre_external_precision_confirm_v4_0c_r1.log")

print(f"请上传：{PACKAGE}")
uploaded = files.upload()
if not uploaded:
    raise RuntimeError("没有收到 ZIP")
if ROOT.exists():
    shutil.rmtree(ROOT)
if RESULT.exists():
    RESULT.unlink()
with zipfile.ZipFile(next(iter(uploaded))) as archive:
    archive.extractall("/content")
script = ROOT / "cner_resnet18_cifar10_fibre_external_precision_confirm_v4_0c_r1.py"
if not script.exists():
    raise FileNotFoundError(script)
print("开始 ResNet-18/CIFAR-10 float64 外部确认 v4.0c-r1；16 seeds，推荐 A100。")
with LOG.open("w", encoding="utf-8") as handle:
    process = subprocess.run(
        [sys.executable, str(script), "--output", str(RESULT.with_suffix(""))],
        cwd=ROOT,
        stdout=handle,
        stderr=subprocess.STDOUT,
        text=True,
    )
print(LOG.read_text(encoding="utf-8", errors="replace"))
if process.returncode:
    failure = Path("/content/cner_resnet18_cifar10_fibre_external_precision_confirm_v4_0c_r1_failure.zip")
    output_dir = RESULT.with_suffix("")
    if output_dir.exists():
        shutil.make_archive(str(failure.with_suffix("")), "zip", output_dir.parent, output_dir.name)
        files.download(str(failure))
    files.download(str(LOG))
    raise RuntimeError(f"v4.0c-r1 运行失败，exit={process.returncode}")
if not RESULT.exists():
    raise FileNotFoundError(RESULT)
print(f"测试完成：{RESULT}")
files.download(str(RESULT))
