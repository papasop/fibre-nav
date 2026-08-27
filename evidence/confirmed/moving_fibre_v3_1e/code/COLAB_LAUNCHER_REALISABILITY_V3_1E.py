from pathlib import Path
import shutil
import subprocess
import sys
from google.colab import files

PACKAGE = "cner_cnn_mnist_realisability_flat_confirm_v3_1e.zip"
ROOT = Path("/content/cner_cnn_mnist_realisability_flat_confirm_v3_1e")
OUT = Path("/content/cner_cnn_mnist_realisability_flat_confirm_v3_1e_results")

print(f"请上传：{PACKAGE}")
uploaded = files.upload()
if PACKAGE not in uploaded:
    raise RuntimeError(f"未找到 {PACKAGE}")
if ROOT.exists():
    shutil.rmtree(ROOT)
if OUT.exists():
    shutil.rmtree(OUT)
result_zip = Path(str(OUT) + ".zip")
if result_zip.exists():
    result_zip.unlink()
shutil.unpack_archive(f"/content/{PACKAGE}", "/content")
print("安装依赖……")
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "torchvision", "scipy", "matplotlib"],
    check=True,
)
print("开始 v3.1e 零斜率门修正确认；16 seeds、每种子八条深路径，强烈推荐 A100。")
proc = subprocess.run([
    sys.executable,
    str(ROOT / "cner_cnn_mnist_realisability_flat_confirm_v3_1e.py"),
    "--protocol", str(ROOT / "protocol.json"),
    "--output", str(OUT),
    "--no-download",
])
if proc.returncode != 0:
    failure = Path("/content/cner_cnn_mnist_realisability_flat_confirm_v3_1e_failure.zip")
    if OUT.exists():
        shutil.make_archive(str(failure.with_suffix("")), "zip", OUT.parent, OUT.name)
        files.download(str(failure))
    raise RuntimeError(f"v3.1e 运行失败，exit={proc.returncode}")
if not result_zip.exists():
    raise RuntimeError(f"结果 ZIP 不存在：{result_zip}")
print("测试完成：", result_zip)
files.download(str(result_zip))
