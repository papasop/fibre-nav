import os,sys,time,json,zipfile,shutil
from pathlib import Path
from c27_protocol import *
ROOT=Path(__file__).resolve().parent


def start(use_drive=True,input_zip=''):
 os.environ.update(CUBLAS_WORKSPACE_CONFIG=':4096:8',TOKENIZERS_PARALLELISM='false',USE_TF='0',USE_FLAX='0')
 # Process wrapper is immutable inherited source.
 sys.path.insert(0,str(ROOT/'legacy'));from process_io import visible
 check_legacy(ROOT)
 pure='import sys,runpy;sys.path.insert(0,sys.argv.pop(1));sys.path.insert(0,sys.argv.pop(1));s=sys.argv[1];sys.argv=sys.argv[1:];runpy.run_path(s,run_name="__main__")'
 pure_prefix=[sys.executable,'-u','-I','-c',pure,str(ROOT/'legacy'),str(ROOT)]
 visible(pure_prefix+[str(ROOT/'test_c27_protocol.py')],'C27 protocol/archive self-tests',timeout=120)
 visible([sys.executable,'-I','-c',"import torch;assert torch.cuda.is_available();n=torch.cuda.get_device_name(0);assert 'A100' in n,n;print(n,torch.__version__,flush=True)"],'A100 device',timeout=60)
 for p in Path('/proc').iterdir():
  if not p.name.isdigit():continue
  try:argv=(p/'cmdline').read_bytes().decode(errors='replace').split('\0')
  except OSError:continue
  if any(Path(x).name.startswith('worker_') and Path(x).suffix=='.py' and ('/gpt2_' in ' '.join(argv) or '/qwen_' in ' '.join(argv)) for x in argv):raise RuntimeError('Another model worker is active: PID '+p.name)
 dd=''
 if use_drive:
  from google.colab import drive
  if not Path('/content/drive/MyDrive').is_dir():drive.mount('/content/drive')
  dd='/content/drive/MyDrive/GPT2_C27_STATE_RESIDUAL_COMPARISON_A100_V1_RECOVERY'
 base=Path('/content/gpt2_c27_state_residual_comparison_a100_v1');base.mkdir(parents=True,exist_ok=True)
 if not input_zip and dd:
  good=[]
  for p in Path(dd).glob(RESULT+'_*.zip'):
   try:good.append((resume_score(p,ROOT),p.stat().st_mtime_ns,str(p)))
   except Exception as e:print('跳过不兼容C27',p.name,str(e),flush=True)
  if good:input_zip=max(good)[2]
 if not input_zip:
  paths=list(Path('/content/gpt2_c25_four_route_coverage_a100_v1').glob('session_*/gpt2_c25_four_route_coverage_results.zip'))
  if use_drive:paths+=list(Path('/content/drive/MyDrive/GPT2_C25_FOUR_ROUTE_COVERAGE_A100_V1_RECOVERY').glob('gpt2_c25_four_route_coverage_results*.zip'))
  for p in sorted(paths,key=lambda p:p.stat().st_mtime_ns,reverse=True):
   try:
    with zipfile.ZipFile(p) as z:
     s=read_json(z,'SUMMARY.json')
     require(s['complete'] and s['writer_head_sha256']==HEAD_SHA,'Exact completed C25 required')
    input_zip=str(p);break
   except Exception:continue
 if not input_zip:
  from google.colab import files
  print('请上传完整C25结果ZIP，或本版C27结果续跑；不是C26诊断包，也不是源码ZIP。',flush=True)
  u=files.upload();require(len(u)==1,'Upload one result ZIP');input_zip=str(Path(next(iter(u))).resolve())
 path=Path(input_zip)
 with zipfile.ZipFile(path) as z:
  require('identity.json' in z.namelist(),'输入不是结果包：请使用完整C25结果ZIP或本版C27恢复包，不能上传源码ZIP。')
  kind=read_json(z,'identity.json')['protocol']
 require(kind in ('GPT2_C25_FOUR_ROUTE_COVERAGE_A100_V1',PROTOCOL),'C27 accepts exact completed C25 or matching C27 recovery result')
 session=base/('session_'+str(time.time_ns()));out=session/'run';out.mkdir(parents=True)
 if kind==PROTOCOL:
  score=resume_score(path,ROOT)
  with zipfile.ZipFile(path) as z:z.extractall(out)
  print('恢复C27真实残差矩阵、已完成候选与冻结评估记录：',score,flush=True)
 else:
  dump(out/'identity.json',{'protocol':PROTOCOL,'settings':SETTINGS,'source_hashes':source_hashes(ROOT),'parent_C25_sha256':PARENT_SHA})
  visible(pure_prefix+[str(ROOT/'prepare_c27.py'),'--parent',str(path),'--out',str(out)],'C27 verified compact C25 import')
 if not (out/'source').exists():shutil.copytree(ROOT,out/'source',ignore=shutil.ignore_patterns('__pycache__'))
 print('C27 FIX1：支持继承INPUT.pt未单独保存extras的格式；原缓存输入逐值保持，记录可核查范围。',flush=True)
 print('明确输入：',path,flush=True)
 print('C27：原C25头全部冻结；xor10训练19态；旧隐藏/未饱和输入两组9通道残差分支；每组固定18候选、单轮离线求解。',flush=True)
 print('保护原76态中的所有74个通过态；完整76态实测后才接受。阈值仍0.002/0.005/0.05。',flush=True)
 print('两组冻结后才加载已检查C25开发16态；不是新留出、不生成新状态。使用原缓存准备特征。',flush=True)
 print('在线单次组合提案+外部提交；无在线Jacobian/SVD/校正。离线复用19份Jacobians和外部数值求解；不测试B5控制器或原生更新。',flush=True)
 print('不设总会话时长限制；保存真实候选矩阵；Drive只保留本C27目录最近3个已校验总包。',flush=True)
 check="import sys;sys.path.insert(0,sys.argv[1]);import torch,numpy,transformers,peft,tokenizers,huggingface_hub,accelerate,safetensors;assert transformers.__version__=='4.56.2';assert peft.__version__=='0.17.1';print('DEPENDENCIES_OK',flush=True)"
 deps=base/'deps'
 for reuse in (Path('/content/gpt2_c25_four_route_coverage_a100_v1/deps'),Path('/content/gpt2_c24_relinearised_calibration_a100_v1/deps')):
  if reuse.is_dir() and visible([sys.executable,'-u','-I','-c',check,str(reuse)],'复用隔离HF依赖',timeout=90,check=False)==0:deps=reuse;break
 deps.mkdir(parents=True,exist_ok=True)
 if visible([sys.executable,'-u','-I','-c',check,str(deps)],'HF依赖检查',timeout=90,check=False):
  visible([sys.executable,'-u','-m','pip','--disable-pip-version-check','install','--upgrade','--no-deps','--only-binary=:all:','--progress-bar','off','--timeout','30','--retries','2','--target',str(deps),
   'transformers==4.56.2','peft==0.17.1','tokenizers==0.22.0','huggingface_hub==0.34.4','accelerate==1.10.1','safetensors==0.6.2'],'隔离HF依赖；保留CUDA PyTorch',timeout=600)
  visible([sys.executable,'-u','-I','-c',check,str(deps)],'安装后检查',timeout=90)
 boot='import sys,runpy;sys.path.insert(0,sys.argv.pop(1));sys.path.insert(0,sys.argv.pop(1));sys.path.insert(0,sys.argv.pop(1));s=sys.argv[1];sys.argv=sys.argv[1:];runpy.run_path(s,run_name="__main__")'
 prefix=[sys.executable,'-u','-I','-c',boot,str(deps),str(ROOT/'legacy'),str(ROOT)]
 visible(prefix+[str(ROOT/'test_c27_torch.py')],'C27 residual/activation/geometry self-tests',timeout=180)
 visible(prefix+[str(ROOT/'test_c27_input_schema.py')],'C27 historical input-schema regression self-tests',timeout=120)
 result=session/(RESULT+'.zip')
 try:visible(prefix+[str(ROOT/'worker_c27.py'),'--out',str(out),'--drive-dir',dd],'C27 A100 frozen-base residual comparison',log_path=session/'console.log')
 finally:
  if (session/'console.log').exists():shutil.copyfile(session/'console.log',out/'console.log')
  try:result=snapshot(out,dd)
  except Exception as e:print('最终快照未完成；保留已有C27备份：',type(e).__name__,str(e),flush=True)
  if result.exists():
   print('唯一结果总包：',result,flush=True)
   try:
    from google.colab import files
    files.download(str(result))
   except Exception as e:print('结果已保存，请手动下载：',result,type(e).__name__,flush=True)


if __name__=='__main__':start()
