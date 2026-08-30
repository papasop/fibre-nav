from google.colab import files
from pathlib import Path
import shutil, subprocess, sys, zipfile

name="cner_resnet18_cifar10_full_layer4_transverse_confirm_v4_2d"
print(f"请上传：{name}.zip")
uploaded=files.upload(); source=next(iter(uploaded))
root=Path("/content")/name
if root.exists(): shutil.rmtree(root)
with zipfile.ZipFile(source) as z: z.extractall("/content")
engine=root/f"{name}.py"
print("开始 ResNet-18/CIFAR-10 v4.2d 横向放大前瞻确认；16 new seeds，必须使用 A100。")
proc=subprocess.run([sys.executable,str(engine),"--output",str(root)+"_results"])
result=Path(str(root)+"_results.zip")
if proc.returncode or not result.exists():
    failure=Path("/content")/f"{name}_failure.zip"; partial=Path(str(root)+"_results")
    if partial.exists(): shutil.make_archive(str(failure.with_suffix("")),"zip",partial.parent,partial.name)
    if failure.exists(): files.download(str(failure))
    raise RuntimeError(f"v4.2d failed, exit={proc.returncode}")
files.download(str(result)); print("测试完成：",result)
