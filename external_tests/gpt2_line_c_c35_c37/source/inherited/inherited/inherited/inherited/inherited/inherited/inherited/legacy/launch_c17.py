import os,shutil,sys,time,zipfile
from pathlib import Path
from c17_protocol import (PROTOCOL,RESULT,C15_SHA,input_kind,validate_c15,verified_zip,resume_score,snapshot,inherited_check)
from process_io import visible

def start(use_drive=True,input_zip=''):
 os.environ.update(CUBLAS_WORKSPACE_CONFIG=':4096:8',TOKENIZERS_PARALLELISM='false',USE_TF='0',USE_FLAX='0')
 root=Path(__file__).resolve().parent;inherited_check(root)
 visible([sys.executable,'-u',str(root/'test_c17.py')],'C17 guard/rollback/archive self-tests',timeout=90)
 visible([sys.executable,'-I','-c',"import torch;assert torch.cuda.is_available();n=torch.cuda.get_device_name(0);assert 'A100' in n,n;print(n,torch.__version__,flush=True)"],'A100 device',timeout=60)
 for proc in Path('/proc').iterdir():
  if not proc.name.isdigit():continue
  try:argv=(proc/'cmdline').read_bytes().decode(errors='replace').split('\0')
  except OSError:continue
  if any(Path(x).name.startswith('worker_') and Path(x).suffix=='.py' and ('/gpt2_' in ' '.join(argv) or '/qwen_' in ' '.join(argv)) for x in argv):raise RuntimeError('Another model worker is active (PID '+proc.name+')')
 backup=''
 if use_drive:
  from google.colab import drive
  drive.mount('/content/drive');backup='/content/drive/MyDrive/GPT2_C17_GUARDED_RAW_WRITER_A100_V1_RECOVERY'
 base=Path('/content/gpt2_c17_guarded_raw_writer_a100_v1');base.mkdir(parents=True,exist_ok=True)
 if not input_zip and backup:
  candidates=[]
  for p in Path(backup).glob(RESULT+'_*.zip'):
   try:candidates.append((resume_score(p),p.stat().st_mtime,str(p)))
   except Exception as e:print('跳过不兼容C17备份：',p,str(e),flush=True)
  if candidates:input_zip=max(candidates)[2]
 if not input_zip:
  candidates=list(Path('/content').glob('gpt2_c16_checkpoint_diagnosis_results*.zip'))+list(Path('/content').glob('gpt2_c15_raw_writer_results*.zip'))
  if use_drive:
   candidates+=list(Path('/content/drive/MyDrive/GPT2_C16_CHECKPOINT_DIAGNOSIS_A100_V1_RECOVERY').glob('gpt2_c16_checkpoint_diagnosis_results_*.zip'))
   candidates+=list(Path('/content/drive/MyDrive/GPT2_C15_RAW_WRITER_A100_V1_RECOVERY').glob('gpt2_c15_raw_writer_results_*.zip'))
  for p in sorted(candidates,key=lambda p:p.stat().st_mtime,reverse=True):
   try:input_kind(p);input_zip=str(p);break
   except Exception as e:print('跳过未完成或不匹配C15：',p,str(e),flush=True)
 if not input_zip:
  print('请上传已完成C16或C15结果ZIP，或本版C17结果续跑；不要上传源码/启动器ZIP。',flush=True)
  from google.colab import files
  uploaded=files.upload()
  if len(uploaded)!=1:raise ValueError('Upload exactly one result ZIP')
  input_zip=str(Path('/content')/next(iter(uploaded)))
 path=Path(input_zip);kind=input_kind(path)
 session=base/('session_'+str(time.time_ns()));out=session/'run';out.mkdir(parents=True,exist_ok=True)
 if kind=='resume':
  progress=resume_score(path)
  with zipfile.ZipFile(path) as z:z.extractall(out)
  validate_c15(out/'parent_c15.zip');print('恢复C17真实接受检查点和优化器历史：',progress,flush=True)
 elif kind=='c15':
  shutil.copyfile(path,out/'parent_c15.zip');validate_c15(out/'parent_c15.zip')
 elif kind=='c16':
  with zipfile.ZipFile(path) as z:
   with z.open('parent_c15.zip') as src,open(out/'parent_c15.zip','wb') as dst:shutil.copyfileobj(src,dst)
   (out/'C16_TRANSITION_SUMMARY.json').write_bytes(z.read('SUMMARY.json'))
  validate_c15(out/'parent_c15.zip')
 else:raise ValueError('Unsupported input')
 print('明确输入：',path,flush=True)
 print('C17：实际 warm1000 起点；离线小步AdamW + 最多8次回溯；全部36训练态前向验收。',flush=True)
 print('保护当前通过集合和内容/KL/margin；最坏响应不许增大；不接受就回退权重及优化器历史。',flush=True)
 print('最多64次离线尝试；冻结后生成16个新局部状态；旧C15的16个诊断态不参与训练。',flush=True)
 print('在线各臂单次提案，无Jacobian/SVD/校正；保持0.002/0.005/0.05；仍是开发评估。',flush=True)
 print('无限会话时长；每次完整离线尝试备份；Drive只保留本C17最近3个校验备份；旧档不改。',flush=True)
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
 visible(prefix+[str(root/'test_c17_torch.py')],'C17 transactional optimiser/isolation Torch self-tests',timeout=90)
 result=session/(RESULT+'.zip')
 try:visible(prefix+[str(root/'worker_c17.py'),'--out',str(out),'--drive-dir',backup],'C17 A100 guarded writer',log_path=session/'console.log')
 except KeyboardInterrupt:print('已中断；用本C17校验结果续跑，已接受检查点、真实优化器和已完成新状态将恢复。',flush=True)
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
