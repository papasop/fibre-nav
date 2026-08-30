from google.colab import files
from pathlib import Path
import shutil, subprocess, sys, zipfile

print("请上传：cner_resnet18_cifar10_moving_fibre_dual_scaling_confirm_v4_1b.zip")
uploaded=files.upload(); source=next(iter(uploaded))
root=Path("/content/cner_resnet18_cifar10_moving_fibre_dual_scaling_confirm_v4_1b")
if root.exists(): shutil.rmtree(root)
with zipfile.ZipFile(source) as z: z.extractall("/content")
engine=root/"cner_resnet18_cifar10_moving_fibre_dual_scaling_confirm_v4_1b.py"
print("开始 ResNet-18/CIFAR-10 Moving-Fibre v4.1b 双幂律分离前瞻确认；16 个新种子，推荐 A100。")
proc=subprocess.run([sys.executable,str(engine),"--output",str(root)+"_results"])
result=Path(str(root)+"_results.zip")
if proc.returncode or not result.exists():
    failure=Path("/content/cner_resnet18_cifar10_moving_fibre_dual_scaling_confirm_v4_1b_failure.zip")
    if Path(str(root)+"_results").exists(): shutil.make_archive(str(failure.with_suffix("")),"zip",root.parent,root.name+"_results")
    if failure.exists(): files.download(str(failure))
    raise RuntimeError(f"v4.1b failed, exit={proc.returncode}")
files.download(str(result)); print("测试完成：",result)
