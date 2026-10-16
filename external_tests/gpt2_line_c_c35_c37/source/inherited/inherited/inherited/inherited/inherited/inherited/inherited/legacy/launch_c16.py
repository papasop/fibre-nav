import os,shutil,sys,time,zipfile
from pathlib import Path
from c16_protocol import (PROTOCOL,PARENT_PROTOCOL,RESULT,validate_parent,verified_zip,resume_score,snapshot,inherited_check)
from process_io import visible

def start(use_drive=True,input_zip=''):
 os.environ.update(CUBLAS_WORKSPACE_CONFIG=':4096:8',TOKENIZERS_PARALLELISM='false',USE_TF='0',USE_FLAX='0')
 root=Path(__file__).resolve().parent;inherited_check(root)
 visible([sys.executable,'-u',str(root/'test_c16.py')],'C16 checkpoint/paired-gate/archive self-tests',timeout=90)
 visible([sys.executable,'-I','-c',"import torch;assert torch.cuda.is_available();n=torch.cuda.get_device_name(0);assert 'A100' in n,n;print(n,torch.__version__,flush=True)"],'A100 device',timeout=60)
 for proc in Path('/proc').iterdir():
  if not proc.name.isdigit():continue
  try:argv=(proc/'cmdline').read_bytes().decode(errors='replace').split('\0')
  except OSError:continue
  if any(Path(x).name.startswith('worker_') and Path(x).suffix=='.py' and ('/gpt2_' in ' '.join(argv) or '/qwen_' in ' '.join(argv)) for x in argv):raise RuntimeError('Another model worker is active (PID '+proc.name+')')
 backup=''
 if use_drive:
  from google.colab import drive
  drive.mount('/content/drive');backup='/content/drive/MyDrive/GPT2_C16_CHECKPOINT_DIAGNOSIS_A100_V1_RECOVERY'
 base=Path('/content/gpt2_c16_checkpoint_diagnosis_a100_v1');base.mkdir(parents=True,exist_ok=True)
 if not input_zip and backup:
  candidates=[]
  for p in Path(backup).glob(RESULT+'_*.zip'):
   try:candidates.append((resume_score(p),p.stat().st_mtime,str(p)))
   except Exception as e:print('跳过不兼容C16备份：',p,str(e),flush=True)
  if candidates:input_zip=max(candidates)[2]
 if not input_zip:
  candidates=list(Path('/content').glob('gpt2_c15_raw_writer_results*.zip'))
  if use_drive:candidates+=list(Path('/content/drive/MyDrive/GPT2_C15_RAW_WRITER_A100_V1_RECOVERY').glob('gpt2_c15_raw_writer_results_*.zip'))
  for p in sorted(candidates,key=lambda p:p.stat().st_mtime,reverse=True):
   try:validate_parent(p);input_zip=str(p);break
   except Exception as e:print('跳过未完成或不匹配C15：',p,str(e),flush=True)
 if not input_zip:
  print('请上传完整 gpt2_c15_raw_writer_results.zip，或本版C16结果续跑；不要上传源码/启动器ZIP。',flush=True)
  from google.colab import files
  uploaded=files.upload()
  if len(uploaded)!=1:raise ValueError('Upload exactly one result ZIP')
  input_zip=str(Path('/content')/next(iter(uploaded)))
 path=Path(input_zip);ident=verified_zip(path)
 session=base/('session_'+str(time.time_ns()));out=session/'run';out.mkdir(parents=True,exist_ok=True)
 if ident['protocol']==PROTOCOL:
  progress=resume_score(path)
  with zipfile.ZipFile(path) as z:z.extractall(out)
  validate_parent(out/'parent_c15.zip');print('恢复C16实际诊断进度：',progress,flush=True)
 elif ident['protocol']==PARENT_PROTOCOL:
  validate_parent(path);shutil.copyfile(path,out/'parent_c15.zip');print('完整C15、原始源码、两阶段真实检查点和52个状态校验通过。',flush=True)
 else:raise ValueError('Requires complete matching C15 or own C16 results')
 print('明确输入：',path,flush=True)
 print('C16：C9基线 / warm1000 / functional720；同36训练态+16已检查新态；不新增训练或状态。',flush=True)
 print('使用同一保存输入作主要对照；另核对新计算特征、参数端点和原记录；不在线校正，不选正式新头。',flush=True)
 print('全门控保持0.002/0.005/0.05；教师仅作误差/正参考；已检查16态是诊断数据，不是新留出。',flush=True)
 print('无限会话时长；每8状态备份；只保留本C16目录最近3个已校验备份；C15/C9存档不修改。',flush=True)
 check="import sys;sys.path.insert(0,sys.argv[1]);import torch,numpy,transformers,peft,tokenizers,huggingface_hub,accelerate,safetensors;assert transformers.__version__=='4.56.2';assert peft.__version__=='0.17.1';print('DEPENDENCIES_OK',flush=True)"
 deps=base/'deps';reuse=Path('/content/gpt2_c15_raw_writer_a100_v1/deps')
 if reuse.exists() and visible([sys.executable,'-u','-I','-c',check,str(reuse)],'复用C15隔离依赖检查',timeout=90,check=False)==0:deps=reuse
 else:
  deps.mkdir(parents=True,exist_ok=True)
  if visible([sys.executable,'-u','-I','-c',check,str(deps)],'HF依赖检查',timeout=90,check=False):
   visible([sys.executable,'-u','-m','pip','--disable-pip-version-check','install','--upgrade','--no-deps','--only-binary=:all:','--progress-bar','off','--timeout','30','--retries','2','--target',str(deps),
    'transformers==4.56.2','peft==0.17.1','tokenizers==0.22.0','huggingface_hub==0.34.4','accelerate==1.10.1','safetensors==0.6.2'],'隔离HF依赖；保留CUDA PyTorch',timeout=600)
   visible([sys.executable,'-u','-I','-c',check,str(deps)],'安装后检查',timeout=90)
 bootstrap='import sys,runpy;sys.path.insert(0,sys.argv.pop(1));sys.path.insert(0,sys.argv.pop(1));s=sys.argv[1];sys.argv=sys.argv[1:];runpy.run_path(s,run_name="__main__")'
 prefix=[sys.executable,'-u','-I','-c',bootstrap,str(deps),str(root)]
 visible(prefix+[str(root/'test_c16_torch.py')],'C16 immutable-head/replay Torch self-tests',timeout=90)
 result=session/(RESULT+'.zip')
 try:visible(prefix+[str(root/'worker_c16.py'),'--out',str(out),'--drive-dir',backup],'C16 A100 paired checkpoint diagnosis',log_path=session/'console.log')
 except KeyboardInterrupt:print('已中断；用本C16校验结果续跑，已完成状态和臂不会重做。',flush=True)
 finally:
  try:
   if (session/'console.log').exists():shutil.copyfile(session/'console.log',out/'console.log')
   saved=snapshot(out,backup)
   if saved:result=saved
  except Exception as e:print('最终快照未完成；保留已有已校验备份：',type(e).__name__,str(e),flush=True)
  if result.exists():
   print('唯一结果总包：',result,flush=True)
   try:
    from google.colab import files
    files.download(str(result))
   except Exception as e:print('请手动保存：',result,type(e).__name__,flush=True)

if __name__=='__main__':start()
