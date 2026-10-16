import os,sys,time,json,zipfile,shutil,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT),str(ROOT/'inherited'),str(ROOT/'inherited/inherited'),str(ROOT/'inherited/inherited/legacy')]
from c29_protocol import *

def start(use_drive=True,input_zip=''):
 os.environ.update(CUBLAS_WORKSPACE_CONFIG=':4096:8',TOKENIZERS_PARALLELISM='false',USE_TF='0',USE_FLAX='0')
 from process_io import visible
 check_inherited(ROOT)
 boot='import sys,runpy;sys.path.insert(0,sys.argv.pop(1));sys.path.insert(0,sys.argv.pop(1));sys.path.insert(0,sys.argv.pop(1));sys.path.insert(0,sys.argv.pop(1));s=sys.argv[1];sys.argv=sys.argv[1:];runpy.run_path(s,run_name="__main__")'
 paths=[str(ROOT/'inherited/inherited/legacy'),str(ROOT/'inherited/inherited'),str(ROOT/'inherited'),str(ROOT)]
 prefix=[sys.executable,'-u','-I','-c',boot]+paths
 visible(prefix+[str(ROOT/'test_c29_protocol.py')],'C29 protocol/archive self-tests',timeout=120)
 visible([sys.executable,'-I','-c',"import torch;assert torch.cuda.is_available();n=torch.cuda.get_device_name(0);assert 'A100' in n,n;print(n,torch.__version__,flush=True)"],'A100 device',timeout=60)
 for p in Path('/proc').iterdir():
  if not p.name.isdigit():continue
  try:argv=(p/'cmdline').read_bytes().decode(errors='replace').split('\0')
  except OSError:continue
  require(not any(Path(x).name.startswith('worker_') and Path(x).suffix=='.py' and ('/gpt2_' in ' '.join(argv) or '/qwen_' in ' '.join(argv)) for x in argv),'Another model worker is active: '+p.name)
 dd=''
 if use_drive:
  from google.colab import drive
  if not Path('/content/drive/MyDrive').is_dir():drive.mount('/content/drive')
  dd='/content/drive/MyDrive/'+DRIVE_NAME
 base=Path('/content/gpt2_c29_frozen_direction_diagnosis_a100_v1');base.mkdir(parents=True,exist_ok=True)
 if not input_zip and dd:
  good=[]
  for p in Path(dd).glob(RESULT+'_*.zip'):
   try:good.append((resume_score(p,ROOT),p.stat().st_mtime_ns,str(p)))
   except Exception as e:print('跳过不兼容C29',p.name,str(e),flush=True)
  if good:input_zip=max(good)[2]
 if not input_zip:
  pp=list(Path('/content/gpt2_c28_response_aware_four_route_a100_v1').glob('session_*/gpt2_c28_response_aware_four_route_results.zip'))
  if use_drive:pp+=list(Path('/content/drive/MyDrive/GPT2_C28_RESPONSE_AWARE_FOUR_ROUTE_A100_V1_RECOVERY').glob('gpt2_c28_response_aware_four_route_results*.zip'))
  for p in sorted(pp,key=lambda p:p.stat().st_mtime_ns,reverse=True):
   try:
    with zipfile.ZipFile(p) as z:parent_probe(z,ROOT,check_source=False)
    input_zip=str(p);break
   except Exception:continue
 if not input_zip:
  from google.colab import files
  print('请上传已完成的C28四路由对照结果ZIP；也可上传本版C29诊断包续跑。不是源码ZIP。',flush=True)
  u=files.upload();require(len(u)==1,'Upload exactly one result ZIP');input_zip=str(Path(next(iter(u))).resolve())
 parent=Path(input_zip)
 with zipfile.ZipFile(parent) as z:
  require('identity.json' in z.namelist(),'输入不是结果包，不能使用源码ZIP')
  kind=read_json(z,'identity.json')['protocol']
 require(kind in (PROTOCOL,PARENT_PROTOCOL),'C29 requires exact completed C28 or matching C29 recovery')
 session=base/('session_'+str(time.time_ns()));out=session/'run';out.mkdir(parents=True)
 if kind==PROTOCOL:
  score=resume_score(parent,ROOT)
  with zipfile.ZipFile(parent) as z:z.extractall(out)
  print('恢复C29已完成诊断、固定参数探测点和诊断Jacobian：',score,flush=True)
 else:
  dump(out/'identity.json',{'protocol':PROTOCOL,'settings':SETTINGS,'source_hashes':source_hashes(ROOT)})
  local=session/'parent_c28_readonly.zip';shutil.copyfile(parent,local)
  try:visible(prefix+[str(ROOT/'prepare_c29.py'),'--parent',str(local),'--out',str(out)],'C29 verified compact C28 import')
  finally:
   if local.exists():local.unlink()
 if not (out/'source').exists():shutil.copytree(ROOT,out/'source',ignore=shutil.ignore_patterns('__pycache__'))
 print('明确输入：',parent,flush=True)
 print('C29：冻结C27与C28两组实际权重；不训练、不搜索、不提交新头。',flush=True)
 print('复用76份训练Jacobian；只对已检查16态在C27终点新增诊断Jacobian。',flush=True)
 print('对比实际提交位移的Jδ与真实响应变化；两组各作固定0.5半步探测，不用于回溯或选头。',flush=True)
 print('开发态导数仅用于本次外部诊断；不是在线写入，也不建立新的泛化结果。',flush=True)
 print('一个诊断总包、无限会话时长；只清理本C29目录超过最近3个的已校验备份。',flush=True)
 check="import sys;sys.path.insert(0,sys.argv[1]);import torch,numpy,transformers,peft,tokenizers,huggingface_hub,accelerate,safetensors;assert transformers.__version__=='4.56.2';assert peft.__version__=='0.17.1';print('DEPENDENCIES_OK',flush=True)"
 deps=base/'deps'
 for reuse in (Path('/content/gpt2_c28_response_aware_four_route_a100_v1/deps'),Path('/content/gpt2_c27_state_residual_comparison_a100_v1/deps')):
  if reuse.is_dir() and visible([sys.executable,'-u','-I','-c',check,str(reuse)],'复用隔离HF依赖',timeout=90,check=False)==0:deps=reuse;break
 deps.mkdir(parents=True,exist_ok=True)
 if visible([sys.executable,'-u','-I','-c',check,str(deps)],'HF依赖检查',timeout=90,check=False):
  visible([sys.executable,'-u','-m','pip','--disable-pip-version-check','install','--upgrade','--no-deps','--only-binary=:all:','--progress-bar','off','--timeout','30','--retries','2','--target',str(deps),'transformers==4.56.2','peft==0.17.1','tokenizers==0.22.0','huggingface_hub==0.34.4','accelerate==1.10.1','safetensors==0.6.2'],'隔离HF依赖；保留CUDA PyTorch',timeout=600)
  visible([sys.executable,'-u','-I','-c',check,str(deps)],'安装后检查',timeout=90)
 b4='import sys,runpy;sys.path.insert(0,sys.argv.pop(1));sys.path.insert(0,sys.argv.pop(1));sys.path.insert(0,sys.argv.pop(1));sys.path.insert(0,sys.argv.pop(1));sys.path.insert(0,sys.argv.pop(1));s=sys.argv[1];sys.argv=sys.argv[1:];runpy.run_path(s,run_name="__main__")'
 prefix=[sys.executable,'-u','-I','-c',b4,str(deps)]+paths
 visible(prefix+[str(ROOT/'test_c29_math.py')],'C29 direction/Taylor/isolation self-tests',timeout=180)
 result=session/(RESULT+'.zip')
 try:visible(prefix+[str(ROOT/'worker_c29.py'),'--out',str(out),'--drive-dir',dd],'C29 A100 frozen direction diagnosis',log_path=session/'console.log')
 finally:
  if (session/'console.log').exists():shutil.copyfile(session/'console.log',out/'console.log')
  try:result=snapshot(out,dd)
  except Exception as e:print('最终快照未完成；保留已有C29备份',type(e).__name__,str(e),flush=True)
  if result.exists():
   print('唯一结果总包：',result,flush=True)
   try:
    from google.colab import files
    files.download(str(result))
   except Exception:print('请手动下载：',result,flush=True)

if __name__=='__main__':start()
