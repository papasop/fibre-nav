from pathlib import Path
import sys,os,time,zipfile,json,shutil,subprocess
from b5_protocol import *
from process_io import visible

def start(use_drive=True,input_zip=''):
 os.environ.update(CUBLAS_WORKSPACE_CONFIG=':4096:8',TOKENIZERS_PARALLELISM='false',USE_TF='0',USE_FLAX='0')
 root=Path(__file__).resolve().parent
 visible([sys.executable,'-u',str(root/'test_dispatch.py')],'B5 runtime self-tests',timeout=60)
 visible([sys.executable,'-I','-c',"import torch;assert torch.cuda.is_available();n=torch.cuda.get_device_name(0);assert 'A100' in n,n;print(n,torch.__version__,flush=True)"],'A100 device',timeout=45)
 drive=''
 if use_drive:
  from google.colab import drive as gd
  gd.mount('/content/drive');drive='/content/drive/MyDrive/GPT2_B5_TRAINED_DISPATCH_A100_V2_RECOVERY'
 base=Path('/content/gpt2_b5_trained_dispatch_a100_v2');base.mkdir(exist_ok=True)
 # Reject a duplicate B5 worker, even if launched in another session.
 for proc in Path('/proc').iterdir():
  if not proc.name.isdigit():continue
  try:a=(proc/'cmdline').read_bytes().decode(errors='replace').split('\0')
  except (OSError,UnicodeError):continue
  if any(Path(x).name=='worker_b5.py' for x in a):raise RuntimeError('B5 worker already running; stop it first')
 if not input_zip:
  candidates=sorted(Path(drive).glob('gpt2_b5_trained_dispatch_a100_results_*.zip'),reverse=True) if drive else []
  for p in candidates:
   try:verified_zip(p,PROTOCOL);input_zip=str(p);break
   except Exception as e:print('跳过损坏B5备份',p,str(e),flush=True)
  if candidates and not input_zip:raise RuntimeError('All B5 backups invalid; no fresh fallback')
 if not input_zip:
  print('请上传已通过的 gpt2_b4_minimal_causal_a100_results.zip；也可上传V1结果提取父包，或本版V2结果续跑。不要上传启动器ZIP。',flush=True)
  from google.colab import files
  uploaded=files.upload()
  if len(uploaded)!=1:raise ValueError('Upload exactly one result ZIP')
  input_zip=str(Path('/content')/next(iter(uploaded)))
 p=Path(input_zip)
 try:verified_zip(p)
 except KeyError as e:raise ValueError('上传的是启动器/源码包。请上传已通过的B4结果ZIP或本版V2结果ZIP；不要上传GPT2_B5_*.zip启动器。') from e
 with zipfile.ZipFile(p) as z:protocol=json.loads(z.read('identity.json'))['protocol']
 session=base/('session_'+str(time.time_ns()));out=session/'run';out.mkdir(parents=True)
 if protocol==PROTOCOL:
  with zipfile.ZipFile(p) as z:z.extractall(out)
  print('明确恢复B5：',p,flush=True)
 elif protocol=='GPT2_B5_MODEL_DISPATCH_A100_V1':
  with zipfile.ZipFile(p) as z:(out/'parent_b4.zip').write_bytes(z.read('parent_b4.zip'))
  verified_zip(out/'parent_b4.zip','GPT2_B4_MINIMAL_CAUSAL_A100_V1')
  print('从V1结果提取原始B4父包；新训V2控制器，不恢复失败的V1动作记录。',flush=True)
 elif protocol=='GPT2_B4_MINIMAL_CAUSAL_A100_V1':
  shutil.copyfile(p,out/'parent_b4.zip');print('明确B4父存档：',p,flush=True)
 else:raise ValueError('Wrong input protocol; refusing source ZIP or unrelated checkpoints')
 deps=base/'deps';deps.mkdir(exist_ok=True)
 check="import sys;sys.path.insert(0,sys.argv[1]);import torch,transformers,peft,tokenizers,huggingface_hub,accelerate,safetensors;assert transformers.__version__=='4.56.2';assert peft.__version__=='0.17.1';print('DEPENDENCIES_OK',flush=True)"
 if visible([sys.executable,'-u','-I','-c',check,str(deps)],'依赖检查',timeout=90,check=False):
  visible([sys.executable,'-u','-m','pip','--disable-pip-version-check','install','--upgrade','--no-deps','--only-binary=:all:','--progress-bar','off','--timeout','30','--retries','2','--target',str(deps),'transformers==4.56.2','peft==0.17.1','tokenizers==0.22.0','huggingface_hub==0.34.4','accelerate==1.10.1','safetensors==0.6.2'],'隔离HF依赖；保留CUDA PyTorch',timeout=600)
  visible([sys.executable,'-u','-I','-c',check,str(deps)],'安装后检查',timeout=90)
 bootstrap='import sys,runpy;sys.path.insert(0,sys.argv.pop(1));sys.path.insert(0,sys.argv.pop(1));s=sys.argv[1];sys.argv=sys.argv[1:];runpy.run_path(s,run_name="__main__")'
 print('B5 V2：独立控制器SFT 600步；未训练地址组合8例检查；通过后才实际写回。无限会话时长。',flush=True)
 print('4种状态：normal/stored_substitution/reset/restored；每例最多12指令、2次写入，每次最多480步。不是完整B5或内部参数更新。',flush=True)
 try:
  visible([sys.executable,'-u','-I','-c',bootstrap,str(deps),str(root),str(root/'worker_b5.py'),'--out',str(out),'--drive-dir',drive],'B5运行',log_path=session/'console.log')
 except KeyboardInterrupt:print('已中断；使用最近校验B5备份恢复，不重建丢失权重。',flush=True)
 finally:
  result=session/'gpt2_b5_trained_dispatch_a100_results.zip'
  if result.exists():
   print('唯一结果总包：',result,flush=True)
   try:
    from google.colab import files
    files.download(str(result))
   except Exception as e:print('自动下载未完成，请手动保存：',result,type(e).__name__,flush=True)
if __name__=='__main__':start()
