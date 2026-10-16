import os,shutil,sys,time,zipfile,json
from pathlib import Path
from c24_protocol import PROTOCOL,RESULT,PARENT_PROTOCOL,input_kind,validate_parent,resume_score,snapshot,inherited_check,validate_capsule
from process_io import visible
from c24_parent import validate_companion

def start(use_drive=True,input_zip='',c22_input_zip=''):
 os.environ.update(CUBLAS_WORKSPACE_CONFIG=':4096:8',TOKENIZERS_PARALLELISM='false',USE_TF='0',USE_FLAX='0')
 root=Path(__file__).resolve().parent;inherited_check(root)
 visible([sys.executable,'-u',str(root/'test_c24.py')],'C24 protocol/archive self-tests',timeout=90)
 visible([sys.executable,'-I','-c',"import torch;assert torch.cuda.is_available();n=torch.cuda.get_device_name(0);assert 'A100' in n,n;print(n,torch.__version__,flush=True)"],'A100 device',timeout=60)
 for proc in Path('/proc').iterdir():
  if not proc.name.isdigit():continue
  try:argv=(proc/'cmdline').read_bytes().decode(errors='replace').split('\0')
  except OSError:continue
  if any(Path(x).name.startswith('worker_') and Path(x).suffix=='.py' and ('/gpt2_' in ' '.join(argv) or '/qwen_' in ' '.join(argv)) for x in argv):raise RuntimeError('Another model worker is active (PID '+proc.name+')')
 backup=''
 if use_drive:
  from google.colab import drive
  if not Path('/content/drive/MyDrive').is_dir():
   try:drive.mount('/content/drive')
   except Exception as e:raise RuntimeError('Drive授权未完成；请在独立单元运行 drive.mount("/content/drive")，成功后重启本PY。尚未开始模型校准。') from e
  backup='/content/drive/MyDrive/GPT2_C24_RELINEARISED_CALIBRATION_A100_V1_RECOVERY'
 base=Path('/content/gpt2_c24_relinearised_calibration_a100_v1');base.mkdir(parents=True,exist_ok=True)
 if not input_zip and backup:
  candidates=[]
  for p in Path(backup).glob(RESULT+'_*.zip'):
   try:candidates.append((resume_score(p),p.stat().st_mtime,str(p)))
   except Exception as e:print('跳过不兼容C24备份：',p,str(e),flush=True)
  if candidates:input_zip=max(candidates)[2]
 if not input_zip:
  candidates=list(Path('/content').glob('gpt2_c23_frozen_step_diagnosis_results*.zip'))
  candidates+=list(Path('/content/gpt2_c23_frozen_step_diagnosis_a100_v1').glob('session_*/gpt2_c23_frozen_step_diagnosis_results.zip'))
  if use_drive:candidates+=list(Path('/content/drive/MyDrive/GPT2_C23_FROZEN_STEP_DIAGNOSIS_A100_V1_RECOVERY').glob('gpt2_c23_frozen_step_diagnosis_results_*.zip'))
  for p in sorted(candidates,key=lambda p:p.stat().st_mtime,reverse=True):
   try:validate_parent(p);input_zip=str(p);break
   except Exception as e:print('跳过未完成或不匹配C23：',p,str(e),flush=True)
 if not input_zip:
  print('请上传完整 gpt2_c23_frozen_step_diagnosis_results.zip，或本C24结果续跑；不要上传源码ZIP。',flush=True)
  from google.colab import files
  uploaded=files.upload()
  if len(uploaded)!=1:raise ValueError('Upload exactly one result ZIP')
  input_zip=str(Path('/content')/next(iter(uploaded)))
 path=Path(input_zip);kind=input_kind(path)
 session=base/('session_'+str(time.time_ns()));out=session/'run';out.mkdir(parents=True,exist_ok=True)
 if kind=='resume':
  score=resume_score(path)
  with zipfile.ZipFile(path) as z:z.extractall(out)
  validate_capsule(out/'parent_capsule.zip');print('恢复C24真实校准检查点和已完成候选/几何/评估记录：',score,flush=True)
 else:
  info=validate_parent(path)
  if not c22_input_zip:
   companions=list(Path('/content').glob('gpt2_c22_budget_calibration_results*.zip'))
   companions+=list(Path('/content/gpt2_c22_budget_calibration_a100_v1').glob('session_*/gpt2_c22_budget_calibration_results.zip'))
   if use_drive:companions+=list(Path('/content/drive/MyDrive/GPT2_C22_BUDGET_CALIBRATION_A100_V1_RECOVERY').glob('gpt2_c22_budget_calibration_results_*.zip'))
   for old in sorted(companions,key=lambda p:p.stat().st_mtime,reverse=True):
    try:
     print('检查原始C22评估存档：',old,flush=True);validate_companion(old,info);c22_input_zip=str(old);break
    except Exception as e:print('跳过非原始C22：',old,str(e),flush=True)
  if not c22_input_zip:
   print('请上传C23使用的原始完整C22结果包，供只读导入原先16个评估态；不重建状态。',flush=True)
   from google.colab import files
   uploaded=files.upload()
   if len(uploaded)!=1:raise ValueError('Upload exactly one original C22 result ZIP')
   c22_input_zip=str(Path('/content')/next(iter(uploaded)))
  visible([sys.executable,'-u',str(root/'prepare_c24.py'),'--parent',str(path),'--companion',str(c22_input_zip),'--capsule',str(out/'parent_capsule.zip')],'C24 verified compact candidate/state import')
  print('C23最佳候选与原始C22评估态导入校验通过；启动后重算候选真实头哈希。',flush=True)
 print('明确输入：',path,flush=True)
 print('C24：从C23 candidate_022真实权重开始；原54/60通过项受保护；重置阶段历史，不继承trust缩小。',flush=True)
 print('每轮重建xor_10全部15态Jacobian；三ridge间隔方向；全90个预声明半径/步长项均实测，无线性淘汰、无提前结束。',flush=True)
 print('同头按哈希复用；按训练结果排名后全60态复验，只有原guard接受的真实候选才能提交。最多16轮，不限制会话时长。',flush=True)
 print('阈值保持0.002/0.005/0.05；最终冻结后评估原C22十六个已检查状态，不参与选头、不称新留出。',flush=True)
 print('在线一提案+外部提交/前向核验，无Jacobian/SVD/校正/教师兜底；离线校准仍用外部autograd/求解。',flush=True)
 print('每5候选/5几何态备份；Drive只保留本C24最近3个已校验备份；C22/C23存档不修改。',flush=True)
 check="import sys;sys.path.insert(0,sys.argv[1]);import torch,numpy,transformers,peft,tokenizers,huggingface_hub,accelerate,safetensors;assert transformers.__version__=='4.56.2';assert peft.__version__=='0.17.1';print('DEPENDENCIES_OK',flush=True)"
 deps=base/'deps'
 for reuse in [Path('/content/gpt2_c22_budget_calibration_a100_v1/deps'),Path('/content/gpt2_c19_ridge_comparison_a100_v1/deps'),Path('/content/gpt2_c18_joint_response_calibration_a100_v1/deps'),Path('/content/gpt2_c17_guarded_raw_writer_a100_v1/deps'),Path('/content/gpt2_c15_raw_writer_a100_v1/deps')]:
  if reuse.exists() and visible([sys.executable,'-u','-I','-c',check,str(reuse)],'复用隔离HF依赖检查',timeout=90,check=False)==0:deps=reuse;break
 deps.mkdir(parents=True,exist_ok=True)
 if visible([sys.executable,'-u','-I','-c',check,str(deps)],'HF依赖检查',timeout=90,check=False):
  visible([sys.executable,'-u','-m','pip','--disable-pip-version-check','install','--upgrade','--no-deps','--only-binary=:all:','--progress-bar','off','--timeout','30','--retries','2','--target',str(deps),
   'transformers==4.56.2','peft==0.17.1','tokenizers==0.22.0','huggingface_hub==0.34.4','accelerate==1.10.1','safetensors==0.6.2'],'隔离HF依赖；保留CUDA PyTorch',timeout=600)
  visible([sys.executable,'-u','-I','-c',check,str(deps)],'安装后检查',timeout=90)
 bootstrap='import sys,runpy;sys.path.insert(0,sys.argv.pop(1));sys.path.insert(0,sys.argv.pop(1));s=sys.argv[1];sys.argv=sys.argv[1:];runpy.run_path(s,run_name="__main__")'
 prefix=[sys.executable,'-u','-I','-c',bootstrap,str(deps),str(root)]
 visible(prefix+[str(root/'test_c22_math.py')],'C24 inherited interval-solver/math self-tests',timeout=90)
 visible(prefix+[str(root/'test_c24_torch.py')],'C24 candidate reconstruction/physical scan self-tests',timeout=90)
 result=session/(RESULT+'.zip')
 try:visible(prefix+[str(root/'worker_c24.py'),'--out',str(out),'--drive-dir',backup],'C24 A100 relinearised guarded calibration',log_path=session/'console.log')
 except KeyboardInterrupt:print('已中断；恢复真实已接受检查点；续跑复用同头几何/候选记录，不从零重训。',flush=True)
 finally:
  try:
   if (session/'console.log').exists():shutil.copyfile(session/'console.log',out/'console.log')
   saved=snapshot(out,backup)
   if saved:result=saved
  except Exception as e:print('最终快照未完成；保留已有备份：',type(e).__name__,str(e),flush=True)
  if result.exists():
   print('唯一结果总包：',result,flush=True)
   try:
    from google.colab import files
    files.download(str(result))
   except Exception as e:print('请手动保存：',result,type(e).__name__,flush=True)

if __name__=='__main__':start()
