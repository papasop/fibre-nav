from google.colab import files
from pathlib import Path
import json,shutil,subprocess,sys,zipfile
EXPECTED="CNER_CNN_MNIST_FISHER_NATURAL_CONFIRM_V16"
print("请上传：cner_cnn_mnist_fisher_confirm_v16.zip")
u=files.upload();zs=[Path(x) for x in u if x.lower().endswith('.zip')]
if len(zs)!=1:raise RuntimeError("必须且只能上传一个ZIP")
d=Path('/content/cner_fisher_v16_run')
if d.exists():shutil.rmtree(d)
d.mkdir()
with zipfile.ZipFile(zs[0]) as z:z.extractall(d)
ss=list(d.rglob('cner_cnn_mnist_fisher_confirm_v16.py'));ps=list(d.rglob('protocol.json'))
if len(ss)!=1 or len(ps)!=1:raise RuntimeError(f"文件定位失败：{ss},{ps}")
p=json.loads(ps[0].read_text())
if p.get('protocol_name')!=EXPECTED or p.get('seeds')!=16 or p.get('base_seed')!=18726 or p.get('adapt_steps')!=240 or p.get('natural_gradient_success_gate')!=12 or p.get('primary_metric')!='output_fisher_quotient' or p.get('integration_subdivisions_max')!=32:raise RuntimeError('协议校验失败')
print('协议通过：v16受限CNER-F终局确认，seed 18726至18741；统一240步预算；T4/T8/T16/T32')
out=Path('/content/cner_fisher_v16_results')
if out.exists():shutil.rmtree(out)
out.mkdir(parents=True,exist_ok=False)
subprocess.run([sys.executable,str(ss[0]),'--output',str(out),'--protocol',str(ps[0]),'--no-upload-dialog','--no-download'],check=True)
zp=Path('/content/cner_cnn_mnist_fisher_confirm_v16_results.zip')
if not zp.exists():raise RuntimeError(f"结果不存在：{zp}")
files.download(str(zp))
