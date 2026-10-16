import os,shutil,sys,time,zipfile,json
from pathlib import Path
from c22_protocol import PROTOCOL,RESULT,PARENT_PROTOCOL,input_kind,validate_parent,resume_score,snapshot,inherited_check,validate_capsule
from process_io import visible

def start(use_drive=True,input_zip=''):
 os.environ.update(CUBLAS_WORKSPACE_CONFIG=':4096:8',TOKENIZERS_PARALLELISM='false',USE_TF='0',USE_FLAX='0')
 root=Path(__file__).resolve().parent;inherited_check(root)
 visible([sys.executable,'-u',str(root/'test_c22.py')],'C22 protocol/archive self-tests',timeout=90)
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
  backup='/content/drive/MyDrive/GPT2_C22_BUDGET_CALIBRATION_A100_V1_RECOVERY'
 base=Path('/content/gpt2_c22_budget_calibration_a100_v1');base.mkdir(parents=True,exist_ok=True)
 if not input_zip and backup:
  candidates=[]
  for p in Path(backup).glob(RESULT+'_*.zip'):
   try:candidates.append((resume_score(p),p.stat().st_mtime,str(p)))
   except Exception as e:print('跳过不兼容C22备份：',p,str(e),flush=True)
  if candidates:input_zip=max(candidates)[2]
 if not input_zip:
  candidates=list(Path('/content').glob('gpt2_c21_radius_comparison_results*.zip'))
  if use_drive:candidates+=list(Path('/content/drive/MyDrive/GPT2_C21_RADIUS_COMPARISON_A100_V1_RECOVERY').glob('gpt2_c21_radius_comparison_results_*.zip'))
  for p in sorted(candidates,key=lambda p:p.stat().st_mtime,reverse=True):
   try:validate_parent(p);input_zip=str(p);break
   except Exception as e:print('跳过未完成或不匹配C21：',p,str(e),flush=True)
 if not input_zip:
  print('请上传完整 gpt2_c21_radius_comparison_results.zip，或本C22结果续跑；不要上传源码ZIP。',flush=True)
  from google.colab import files
  uploaded=files.upload()
  if len(uploaded)!=1:raise ValueError('Upload exactly one result ZIP')
  input_zip=str(Path('/content')/next(iter(uploaded)))
 path=Path(input_zip);kind=input_kind(path)
 session=base/('session_'+str(time.time_ns()));out=session/'run';out.mkdir(parents=True,exist_ok=True)
 if kind=='resume':
  score=resume_score(path)
  with zipfile.ZipFile(path) as z:z.extractall(out)
  validate_capsule(out/'parent_capsule.zip');print('恢复C22真实校准检查点及已保存Jacobian：',score,flush=True)
 else:
  visible([sys.executable,'-u',str(root/'prepare_c22.py'),'--parent',str(path),'--capsule',str(out/'parent_capsule.zip')],'C22 verified compact parent import')
  print('完整C21与真实checkpoint_000016校验通过；复用全部60训练态，不新增或重抽训练态。',flush=True)
 print('明确输入：',path,flush=True)
 print('C22：从C21真实最终头开始；区间目标0.0018、实际响应门0.002，比较增量半径0.05/0.10/0.20；只校准最后线性层，不恢复旧Adam历史。',flush=True)
 print('复用C21全部60训练态；旧C15/C17/C18/C19/C20/C21已查看评估态不加入训练。',flush=True)
 print('每轮比较3个半径×3个ridge各自首个可接受回溯候选；先通过数，再最大漂移，再平方预算超额、平方残差。',flush=True)
 print('最多16轮校准；每分支15个训练态；候选实测15态、其余45态结构不变；提交前重测全60态；保留当前50/60通过项、原36态的30项通过及最大漂移及0.002/0.005/0.05原门槛。',flush=True)
 print('冻结后生成16个新局部状态；在线单次头提案，无Jacobian/SVD/校正/教师回退；开发评估。',flush=True)
 print('保存精确小系数和原始导数，可重建方向；记录实际位移、预测误差；无限会话时长；Drive只保留本C22最近3份校验备份。',flush=True)
 check="import sys;sys.path.insert(0,sys.argv[1]);import torch,numpy,transformers,peft,tokenizers,huggingface_hub,accelerate,safetensors;assert transformers.__version__=='4.56.2';assert peft.__version__=='0.17.1';print('DEPENDENCIES_OK',flush=True)"
 deps=base/'deps'
 for reuse in [Path('/content/gpt2_c21_radius_comparison_a100_v1/deps'),Path('/content/gpt2_c19_ridge_comparison_a100_v1/deps'),Path('/content/gpt2_c18_joint_response_calibration_a100_v1/deps'),Path('/content/gpt2_c17_guarded_raw_writer_a100_v1/deps'),Path('/content/gpt2_c15_raw_writer_a100_v1/deps')]:
  if reuse.exists() and visible([sys.executable,'-u','-I','-c',check,str(reuse)],'复用隔离HF依赖检查',timeout=90,check=False)==0:deps=reuse;break
 deps.mkdir(parents=True,exist_ok=True)
 if visible([sys.executable,'-u','-I','-c',check,str(deps)],'HF依赖检查',timeout=90,check=False):
  visible([sys.executable,'-u','-m','pip','--disable-pip-version-check','install','--upgrade','--no-deps','--only-binary=:all:','--progress-bar','off','--timeout','30','--retries','2','--target',str(deps),
   'transformers==4.56.2','peft==0.17.1','tokenizers==0.22.0','huggingface_hub==0.34.4','accelerate==1.10.1','safetensors==0.6.2'],'隔离HF依赖；保留CUDA PyTorch',timeout=600)
  visible([sys.executable,'-u','-I','-c',check,str(deps)],'安装后检查',timeout=90)
 bootstrap='import sys,runpy;sys.path.insert(0,sys.argv.pop(1));sys.path.insert(0,sys.argv.pop(1));s=sys.argv[1];sys.argv=sys.argv[1:];runpy.run_path(s,run_name="__main__")'
 prefix=[sys.executable,'-u','-I','-c',bootstrap,str(deps),str(root)]
 visible(prefix+[str(root/'test_c22_math.py')],'C22 interval-dual/radius algebra self-tests',timeout=90)
 visible(prefix+[str(root/'test_c22_torch.py')],'C22 output-layer/autograd/isolation self-tests',timeout=90)
 result=session/(RESULT+'.zip')
 try:visible(prefix+[str(root/'worker_c22.py'),'--out',str(out),'--drive-dir',backup],'C22 A100 interval-budget calibration',log_path=session/'console.log')
 except KeyboardInterrupt:print('已中断；续跑恢复上次完整接受头与分支步幅，复用同头已保存导数和候选实测记录。',flush=True)
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
