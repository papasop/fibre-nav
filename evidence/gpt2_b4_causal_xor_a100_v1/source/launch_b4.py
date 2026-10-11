from pathlib import Path
import subprocess,sys,zipfile,json,hashlib,shutil,time,os
from archive_io import restore_archive

def start(use_drive=True,resume_zip=''):
 root=Path(__file__).resolve().parent
 gpu_check=subprocess.run([sys.executable,'-I','-c',
  "import torch;assert torch.cuda.is_available(),'Colab CUDA unavailable';n=torch.cuda.get_device_name(0);assert 'A100' in n,'Select A100, observed '+n;print('GPU:',n,'torch:',torch.__version__,'CUDA:',torch.version.cuda)"],capture_output=True,text=True)
 if gpu_check.returncode:raise RuntimeError('请先在 Colab 选择 A100 GPU 运行时。\n'+gpu_check.stderr)
 print(gpu_check.stdout,flush=True)
 os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
 os.environ['TOKENIZERS_PARALLELISM']='false' 
 subprocess.run([sys.executable,str(root/'test_b4.py')],check=True)
 subprocess.run([sys.executable,str(root/'test_a100.py')],check=True)
 drive=''
 if use_drive:
  from google.colab import drive as gd
  gd.mount('/content/drive')
  drive='/content/drive/MyDrive/GPT2_B4_CAUSAL_XOR_A100_V1_RECOVERY'
 base=Path('/content/gpt2_b4_causal_xor_a100_v1');base.mkdir(parents=True,exist_ok=True)
 session=base/('session_'+str(time.time_ns()));out=session/'run';out.mkdir(parents=True)
 candidates=[Path(resume_zip)] if resume_zip else (list(Path(drive).glob('gpt2_b4_a100_results_*.zip')) if drive else [])
 chosen=None;score=None
 for p in candidates:
  temp=session/('check_'+str(time.time_ns()))
  try:
   restore_archive(p,temp)
   records=json.loads((temp/'records.json').read_text()) if (temp/'records.json').exists() else {}
   progress=json.loads((temp/'progress.json').read_text()) if (temp/'progress.json').exists() else {}
   key=(len(records),progress.get('steps',0) if progress.get('phase') not in records else 0)
   if score is None or key>score:chosen=p;score=key
  except Exception as e:
   if resume_zip:raise
   print('跳过损坏备份：',p,type(e).__name__,str(e))
  finally:shutil.rmtree(temp,ignore_errors=True)
 if candidates and chosen is None:raise RuntimeError('No valid continuation archive; refusing fresh fallback')
 if chosen:restore_archive(chosen,out);print('明确续跑：',chosen,'实际进度：',score,flush=True)
 else:print('首次 B4 开发测试：从零 B 重建 B3 共享状态。',flush=True)
 print('GPT-2 B4 A100：float32、TF32关闭；外部 XOR；无限会话时长；每5步备份；不是B5。',flush=True)
 # Reuse Colab's working CUDA torch; never replace it with a CPU wheel.
 deps=base/'deps';deps.mkdir(exist_ok=True)
 bootstrap='import sys,runpy;sys.path.insert(0,sys.argv.pop(1));sys.path.insert(0,sys.argv.pop(1));s=sys.argv[1];sys.argv=sys.argv[1:];runpy.run_path(s,run_name="__main__")'
 prefix=[sys.executable,'-I','-c',bootstrap,str(deps),str(root)]
 check_code="import sys;sys.path.insert(0,sys.argv[1]);import torch,transformers,peft,tokenizers,huggingface_hub,accelerate,safetensors;assert torch.cuda.is_available();assert transformers.__version__=='4.56.2';assert peft.__version__=='0.17.1'"
 check=subprocess.run([sys.executable,'-I','-c',check_code,str(deps)],capture_output=True,text=True)
 if check.returncode:
  print('安装独立 Hugging Face 依赖；保留 Colab 已安装的 CUDA PyTorch。',flush=True)
  subprocess.run([sys.executable,'-m','pip','install','--upgrade','--no-deps','--target',str(deps),
   'transformers==4.56.2','peft==0.17.1','tokenizers==0.22.0','huggingface_hub==0.34.4',
   'accelerate==1.10.1','safetensors==0.6.2'],check=True)
  subprocess.run([sys.executable,'-I','-c',check_code,str(deps)],check=True)
 subprocess.run(prefix+[str(root/'test_resume.py')],check=True)
 try:
  subprocess.run(prefix+[str(root/'worker_b4.py'),'--out',str(out),'--drive-dir',drive],check=True)
 except KeyboardInterrupt:
  print('已中断；请用最近校验通过的 Drive 结果 ZIP 续跑。',flush=True)
  return
 result=out.parent/'gpt2_b4_a100_results.zip'
 if result.exists():
  print('请保存本次结果：',result,flush=True)
  from google.colab import files
  files.download(str(result))
