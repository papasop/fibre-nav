from google.colab import files
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

PACKAGE = "cner_cnn_mnist_moving_fibre_f16_scaling_v3_2b.zip"
ROOT = Path("/content/cner_cnn_mnist_moving_fibre_f16_scaling_v3_2b")
OUTPUT = Path("/content/cner_cnn_mnist_moving_fibre_f16_scaling_v3_2b_results")

print(f"请上传：{PACKAGE}")
uploaded = files.upload()
archive_name = next((name for name in uploaded if Path(name).name == PACKAGE), None)
if archive_name is None:
    raise RuntimeError(f"未找到 {PACKAGE}")
if ROOT.exists():
    shutil.rmtree(ROOT)
with zipfile.ZipFile(archive_name) as archive:
    archive.extractall("/content")

print("安装依赖……")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "torchvision", "numpy"], check=True)
print("开始 Moving-Fibre F16 v3.2b 多步长分叉审计；4 seeds、96 paths，推荐 A100。")
proc = subprocess.run([
    sys.executable,
    str(ROOT / "cner_cnn_mnist_moving_fibre_f16_scaling_v3_2b.py"),
    "--protocol", str(ROOT / "protocol.json"),
    "--output", str(OUTPUT),
    "--no-download",
])
if proc.returncode != 0:
    failure = Path("/content/cner_cnn_mnist_moving_fibre_f16_scaling_v3_2b_failure.zip")
    if OUTPUT.exists():
        shutil.make_archive(str(failure.with_suffix("")), "zip", OUTPUT.parent, OUTPUT.name)
        files.download(str(failure))
    raise RuntimeError(f"v3.2b 运行失败，exit={proc.returncode}")

result_zip = Path(str(OUTPUT) + ".zip")
if not result_zip.exists():
    raise RuntimeError(f"结果 ZIP 不存在：{result_zip}")
print("测试完成：", result_zip)
files.download(str(result_zip))
