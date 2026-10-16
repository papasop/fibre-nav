import os,shutil,sys,time,zipfile,json
from pathlib import Path
from c25_protocol import PROTOCOL,RESULT,PARENT_PROTOCOL,input_kind,validate_parent,resume_score,snapshot,inherited_check,validate_capsule
from process_io import visible

def start(use_drive=True,input_zip=''):
 os.environ.update(CUBLAS_WORKSPACE_CONFIG=':4096:8',TOKENIZERS_PARALLELISM='false',USE_TF='0',USE_FLAX='0')
 root=Path(__file__).resolve().parent;inherited_check(root)
 visible([sys.executable,'-u',str(root/'test_c25.py')],'C25 protocol/archive self-tests',timeout=90)
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
  backup='/content/drive/MyDrive/GPT2_C25_FOUR_ROUTE_COVERAGE_A100_V1_RECOVERY'
 base=Path('/content/gpt2_c25_four_route_coverage_a100_v1');base.mkdir(parents=True,exist_ok=True)
 if not input_zip and backup:
  candidates=[]
  for p in Path(backup).glob(RESULT+'_*.zip'):
   try:candidates.append((resume_score(p),p.stat().st_mtime,str(p)))
   except Exception as e:print('跳过不兼容C25备份：',p,str(e),flush=True)
  if candidates:input_zip=max(candidates)[2]
 if not input_zip:
  candidates=list(Path('/content').glob('gpt2_c24_relinearised_calibration_results*.zip'))
  candidates+=list(Path('/content/gpt2_c24_relinearised_calibration_a100_v1').glob('session_*/gpt2_c24_relinearised_calibration_results.zip'))
  if use_drive:candidates+=list(Path('/content/drive/MyDrive/GPT2_C24_RELINEARISED_CALIBRATION_A100_V1_RECOVERY').glob('gpt2_c24_relinearised_calibration_results_*.zip'))
  for p in sorted(candidates,key=lambda p:p.stat().st_mtime,reverse=True):
   try:validate_parent(p);input_zip=str(p);break
   except Exception as e:print('跳过未完成或不匹配C24：',p,str(e),flush=True)
 if not input_zip:
  print('请上传完整 gpt2_c24_relinearised_calibration_results.zip，或本C25结果续跑；不要上传源码ZIP。',flush=True)
  from google.colab import files
  uploaded=files.upload()
  if len(uploaded)!=1:raise ValueError('Upload exactly one result ZIP')
  input_zip=str(Path('/content')/next(iter(uploaded)))
 path=Path(input_zip);kind=input_kind(path)
 session=base/('session_'+str(time.time_ns()));out=session/'run';out.mkdir(parents=True,exist_ok=True)
 if kind=='resume':
  score=resume_score(path)
  with zipfile.ZipFile(path) as z:z.extractall(out)
  validate_capsule(out/'parent_capsule.zip');print('恢复C25真实校准检查点和已完成候选/几何/评估记录：',score,flush=True)
 else:
  visible([sys.executable,'-u',str(root/'prepare_c25.py'),'--parent',str(path),'--capsule',str(out/'parent_capsule.zip')],'C25 verified compact C24 import')
  print('完整C24、真实checkpoint_000008_final和60/60记录校验通过；启动后重算最终头哈希。',flush=True)
 print('明确输入：',path,flush=True)
 print('C25：真实C24最终头；原60态+已检查16态明确转训练，共76态，每路由19态；原60门全保护。',flush=True)
 print('四路由轮流校准；每轮重新求导并实测完整90项；全76态复验才提交；最多32轮，无总会话时长限制。',flush=True)
 print('阈值保持0.002/0.005/0.05；原60态响应可在预算内变化，不要求其已测最大漂移继续单调下降。',flush=True)
 print('最终冻结后生成16个预声明新种子局部态；不重抽、不参与选头；仍是同模型/图的开发评估。',flush=True)
 print('在线一提案+外部提交/前向核验，无Jacobian/SVD/校正/教师兜底；离线校准和状态准备使用外部几何。',flush=True)
 print('每5候选/5几何态备份；Drive仅保留本C25最近3个已校验备份；旧版本存档不修改。',flush=True)
 check="import sys;sys.path.insert(0,sys.argv[1]);import torch,numpy,transformers,peft,tokenizers,huggingface_hub,accelerate,safetensors;assert transformers.__version__=='4.56.2';assert peft.__version__=='0.17.1';print('DEPENDENCIES_OK',flush=True)"
 deps=base/'deps'
 for reuse in [Path('/content/gpt2_c24_relinearised_calibration_a100_v1/deps'),Path('/content/gpt2_c22_budget_calibration_a100_v1/deps'),Path('/content/gpt2_c19_ridge_comparison_a100_v1/deps'),Path('/content/gpt2_c18_joint_response_calibration_a100_v1/deps'),Path('/content/gpt2_c17_guarded_raw_writer_a100_v1/deps'),Path('/content/gpt2_c15_raw_writer_a100_v1/deps')]:
  if reuse.exists() and visible([sys.executable,'-u','-I','-c',check,str(reuse)],'复用隔离HF依赖检查',timeout=90,check=False)==0:deps=reuse;break
 deps.mkdir(parents=True,exist_ok=True)
 if visible([sys.executable,'-u','-I','-c',check,str(deps)],'HF依赖检查',timeout=90,check=False):
  visible([sys.executable,'-u','-m','pip','--disable-pip-version-check','install','--upgrade','--no-deps','--only-binary=:all:','--progress-bar','off','--timeout','30','--retries','2','--target',str(deps),
   'transformers==4.56.2','peft==0.17.1','tokenizers==0.22.0','huggingface_hub==0.34.4','accelerate==1.10.1','safetensors==0.6.2'],'隔离HF依赖；保留CUDA PyTorch',timeout=600)
  visible([sys.executable,'-u','-I','-c',check,str(deps)],'安装后检查',timeout=90)
 bootstrap='import sys,runpy;sys.path.insert(0,sys.argv.pop(1));sys.path.insert(0,sys.argv.pop(1));s=sys.argv[1];sys.argv=sys.argv[1:];runpy.run_path(s,run_name="__main__")'
 prefix=[sys.executable,'-u','-I','-c',bootstrap,str(deps),str(root)]
 visible(prefix+[str(root/'test_c22_math.py')],'C25 inherited interval-solver/math self-tests',timeout=90)
 visible(prefix+[str(root/'test_c25_torch.py')],'C25 real head/four-route scan self-tests',timeout=90)
 result=session/(RESULT+'.zip')
 try:visible(prefix+[str(root/'worker_c25.py'),'--out',str(out),'--drive-dir',backup],'C25 A100 four-route coverage calibration',log_path=session/'console.log')
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
