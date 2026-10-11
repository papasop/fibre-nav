from pathlib import Path
import subprocess,sys,zipfile,json,hashlib,shutil,time,os
from archive_io import restore_archive,archive_progress
from process_io import visible,refuse_existing_worker

def start(use_drive=True,resume_zip=''):
 root=Path(__file__).resolve().parent
 gpu_check=subprocess.run([sys.executable,'-I','-c',
  "import torch;assert torch.cuda.is_available(),'Colab CUDA unavailable';n=torch.cuda.get_device_name(0);assert 'A100' in n,'Select A100, observed '+n;print('GPU:',n,'torch:',torch.__version__,'CUDA:',torch.version.cuda)"],capture_output=True,text=True,timeout=45)
 if gpu_check.returncode:raise RuntimeError('请先在 Colab 选择 A100 GPU 运行时。\n'+gpu_check.stderr)
 print(gpu_check.stdout,flush=True)
 os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
 os.environ['TOKENIZERS_PARALLELISM']='false'
 os.environ['USE_TF']='0'
 os.environ['USE_FLAX']='0'
 visible([sys.executable,'-u',str(root/'test_b4.py')],'协议自检',timeout=60)
 visible([sys.executable,'-u',str(root/'test_a100.py')],'A100配置自检',timeout=60)
 drive=''
 if use_drive:
  from google.colab import drive as gd
  gd.mount('/content/drive')
  drive='/content/drive/MyDrive/GPT2_B4_FULL_XOR_A100_V2_RECOVERY'
 base=Path('/content/gpt2_b4_full_xor_a100_v2');base.mkdir(parents=True,exist_ok=True)
 refuse_existing_worker(base)
 session=base/('session_'+str(time.time_ns()));out=session/'run';out.mkdir(parents=True)
 candidates=[Path(resume_zip)] if resume_zip else (list(Path(drive).glob('gpt2_b4_full_xor_a100_results_*.zip')) if drive else [])
 chosen=None;score=None
 for p in candidates:
  temp=session/('check_'+str(time.time_ns()))
  try:
   restore_archive(p,temp)
   key=archive_progress(temp)
   if score is None or key>score:chosen=p;score=key
  except Exception as e:
   if resume_zip:raise
   print('跳过损坏备份：',p,type(e).__name__,str(e))
  finally:shutil.rmtree(temp,ignore_errors=True)
 if candidates and chosen is None:raise RuntimeError('No valid continuation archive; refusing fresh fallback')
 if chosen:restore_archive(chosen,out);print('明确续跑：',chosen,'实际进度：',score,flush=True)
 else:print('首次 B4 完整 XOR：三个固定种子、四种输入，各分支从零 B 构建；不读取旧 B4 结果。',flush=True)
 print('B4 XOR V2：82931开发 + 82961/82991新种子；正常/替换写回均须翻转；实际错误读取写回对照；训练每5步备份；无会话时限。',flush=True)
 print('规模：60个训练阶段，最多30240步；可能需要数小时。Drive每5步校验备份，可中断续跑。',flush=True)
 # Reuse Colab's working CUDA torch; never replace it with a CPU wheel.
 deps=base/'deps_visible_v1_1';deps.mkdir(exist_ok=True)
 bootstrap='import sys,runpy;sys.path.insert(0,sys.argv.pop(1));sys.path.insert(0,sys.argv.pop(1));s=sys.argv[1];sys.argv=sys.argv[1:];runpy.run_path(s,run_name="__main__")'
 prefix=[sys.executable,'-u','-I','-c',bootstrap,str(deps),str(root)]
 check_code="import sys;sys.path.insert(0,sys.argv[1]);import torch,transformers,peft,tokenizers,huggingface_hub,accelerate,safetensors;assert torch.cuda.is_available();assert transformers.__version__=='4.56.2';assert peft.__version__=='0.17.1';print('DEPENDENCIES_OK',torch.__version__,transformers.__version__,peft.__version__,flush=True)"
 check=visible([sys.executable,'-u','-I','-c',check_code,str(deps)],'依赖导入检查',timeout=90,check=False)
 if check:
  print('安装隔离依赖：实时显示日志；最多600秒；保留 CUDA PyTorch。',flush=True)
  visible([sys.executable,'-u','-m','pip','--disable-pip-version-check','install',
   '--upgrade','--no-deps','--only-binary=:all:','--progress-bar','off',
   '--timeout','30','--retries','2','--target',str(deps),
   'transformers==4.56.2','peft==0.17.1','tokenizers==0.22.0','huggingface_hub==0.34.4',
   'accelerate==1.10.1','safetensors==0.6.2'],'安装 Hugging Face 依赖',timeout=600,log_path=session/'install.log')
  visible([sys.executable,'-u','-I','-c',check_code,str(deps)],'安装后导入检查',timeout=90)
 visible(prefix+[str(root/'test_resume.py')],'写入器续跑自检',timeout=120)
 try:
  visible(prefix+[str(root/'worker_b4.py'),'--out',str(out),'--drive-dir',drive],
    'B4 FULL XOR A100 运行',log_path=session/'worker_console.log')
 except KeyboardInterrupt:
  print('已停止本次子进程；请用最近校验通过的 Drive 结果 ZIP 续跑。',flush=True)
  return
 result=out.parent/'gpt2_b4_full_xor_a100_results.zip'
 if result.exists():
  print('请保存本次结果：',result,flush=True)
  from google.colab import files
  files.download(str(result))
