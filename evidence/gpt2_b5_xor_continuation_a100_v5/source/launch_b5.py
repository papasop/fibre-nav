from pathlib import Path
import sys,os,time,zipfile,json,shutil,subprocess
from b5_protocol import *
from process_io import visible
from continuation_plan import initialise, resume_score, V4_PROTOCOL

def start(use_drive=True,input_zip=''):
 os.environ.update(CUBLAS_WORKSPACE_CONFIG=':4096:8',TOKENIZERS_PARALLELISM='false',USE_TF='0',USE_FLAX='0')
 root=Path(__file__).resolve().parent
 visible([sys.executable,'-u',str(root/'test_dispatch.py')],'B5 runtime self-tests',timeout=60)
 visible([sys.executable,'-u',str(root/'test_continuation.py')],'V5恢复与门控自检',timeout=60)
 visible([sys.executable,'-I','-c',"import torch;assert torch.cuda.is_available();n=torch.cuda.get_device_name(0);assert 'A100' in n,n;print(n,torch.__version__,flush=True)"],'A100 device',timeout=45)
 drive=''
 if use_drive:
  from google.colab import drive as gd
  gd.mount('/content/drive');drive='/content/drive/MyDrive/GPT2_B5_XOR_CONTINUATION_A100_V5_RECOVERY'
 base=Path('/content/gpt2_b5_xor_continuation_a100_v5');base.mkdir(exist_ok=True)
 # Reject a duplicate B5 worker, even if launched in another session.
 for proc in Path('/proc').iterdir():
  if not proc.name.isdigit():continue
  try:a=(proc/'cmdline').read_bytes().decode(errors='replace').split('\0')
  except (OSError,UnicodeError):continue
  if any(Path(x).name=='worker_b5.py' for x in a):raise RuntimeError('B5 worker already running; stop it first')
 if not input_zip:
  candidates=list(Path(drive).glob('gpt2_b5_xor_continuation_a100_results_*.zip')) if drive else []
  valid=[]
  for p in candidates:
   try:valid.append((resume_score(p),p.stat().st_mtime,str(p)))
   except Exception as e:print('跳过损坏V5备份',p,str(e),flush=True)
  if valid:input_zip=max(valid)[2]
  elif candidates:raise RuntimeError('All V5 backups invalid; no fresh fallback')
 if not input_zip and use_drive:
  recovery=Path('/content/drive/MyDrive/GPT2_B5_XOR_TRUTH_TABLE_A100_V4_RECOVERY')
  for p in sorted(recovery.glob('gpt2_b5_xor_truth_table_a100_results_*.zip'),reverse=True):
   try:
    verified_zip(p,V4_PROTOCOL)
    with zipfile.ZipFile(p) as z:
     from continuation_plan import validate_parent_summary
     validate_parent_summary(json.loads(z.read('SUMMARY.json')))
    input_zip=str(p);break
   except Exception as e:print('跳过不完整/不兼容V4备份',p,str(e),flush=True)
 if not input_zip:
  print('请上传完整V4结果ZIP（2/4通过）或V5结果续跑；不要上传启动器、Qwen或V3包。',flush=True)
  from google.colab import files
  uploaded=files.upload()
  if len(uploaded)!=1:raise ValueError('Upload exactly one result ZIP')
  input_zip=str(Path('/content')/next(iter(uploaded)))
 p=Path(input_zip)
 try:verified_zip(p)
 except KeyError as e:raise ValueError('不是结果ZIP；需要完整V4或V5结果包。') from e
 with zipfile.ZipFile(p) as z:protocol=json.loads(z.read('identity.json'))['protocol']
 session=base/('session_'+str(time.time_ns()));out=session/'run';out.mkdir(parents=True)
 if protocol==PROTOCOL:
  with zipfile.ZipFile(p) as z:z.extractall(out)
  print('恢复V5实际进度：',resume_score(p),'输入：',p,flush=True)
 elif protocol==V4_PROTOCOL:
  initialise(p,out)
  print('V4输入完整校验通过：',p,flush=True)
  print('继承00/01通过记录；10恢复真实600步检查点；11从已保存准备态重新写回，旧被拒候选已丢失。',flush=True)
 else:raise ValueError('V5 only accepts complete V4 or matching V5 results')
 deps=Path('/content/gpt2_b5_xor_truth_table_a100_v4/deps');deps.mkdir(exist_ok=True)
 check="import sys;sys.path.insert(0,sys.argv[1]);import torch,transformers,peft,tokenizers,huggingface_hub,accelerate,safetensors;assert transformers.__version__=='4.56.2';assert peft.__version__=='0.17.1';print('DEPENDENCIES_OK',flush=True)"
 if visible([sys.executable,'-u','-I','-c',check,str(deps)],'依赖检查',timeout=90,check=False):
  visible([sys.executable,'-u','-m','pip','--disable-pip-version-check','install','--upgrade','--no-deps','--only-binary=:all:','--progress-bar','off','--timeout','30','--retries','2','--target',str(deps),'transformers==4.56.2','peft==0.17.1','tokenizers==0.22.0','huggingface_hub==0.34.4','accelerate==1.10.1','safetensors==0.6.2'],'隔离HF依赖；保留CUDA PyTorch',timeout=600)
  visible([sys.executable,'-u','-I','-c',check,str(deps)],'安装后检查',timeout=90)
 bootstrap='import sys,runpy;sys.path.insert(0,sys.argv.pop(1));sys.path.insert(0,sys.argv.pop(1));s=sys.argv[1];sys.argv=sys.argv[1:];runpy.run_path(s,run_name="__main__")'
 print('V5：固定V3控制器和原门槛；10准备累计最多1200步；失败案例写回每次最多1080步。',flush=True)
 print('无限会话；每5步保留真实检查点并备份Drive，最近3份V5备份；旧版本不修改。开发续训，不是独立确认。',flush=True)
 try:
  visible([sys.executable,'-u','-I','-c',bootstrap,str(deps),str(root),str(root/'worker_b5.py'),'--out',str(out),'--drive-dir',drive],'B5运行',log_path=session/'console.log')
 except KeyboardInterrupt:print('已中断；使用最近校验B5备份恢复，不重建丢失权重。',flush=True)
 finally:
  result=session/'gpt2_b5_xor_continuation_a100_results.zip'
  if result.exists():
   print('唯一结果总包：',result,flush=True)
   try:
    from google.colab import files
    files.download(str(result))
   except Exception as e:print('自动下载未完成，请手动保存：',result,type(e).__name__,flush=True)
if __name__=='__main__':start()
