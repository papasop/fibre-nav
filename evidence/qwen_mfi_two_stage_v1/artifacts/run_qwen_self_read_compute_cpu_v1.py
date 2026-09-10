"""CPU two-stage read/compute diagnostic launcher. No total runtime timeout. Python3.11-3.13."""
import os,sys,time,json,signal,subprocess,zipfile,hashlib,argparse,platform
from pathlib import Path
ARCHIVE='qwen_self_read_compute_cpu_v1.zip'
EXPECTED_SHA='d354abca180e1ce4fae21908277147d5a7beaaff2425122bb6bebe9f9f762b77'

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--archive',default=ARCHIVE);p.add_argument('--threads',type=int,default=4)
 args,unknown=p.parse_known_args()
 if unknown:print('Ignored notebook arguments:',unknown,flush=True)
 if not (3,11)<=sys.version_info[:2]<=(3,13):raise RuntimeError('Use Python3.11,3.12 or3.13 for pinned dependencies')
 if args.threads<1:raise ValueError('threads must be positive')
 if os.name!='posix':raise RuntimeError('Use Linux/Colab/macOS or WSL')
 colab=False;archive=Path(args.archive)
 if not archive.exists():
  try:from google.colab import files
  except ImportError:raise FileNotFoundError('Place '+ARCHIVE+' beside this launcher or specify --archive')
  colab=True;print('请选择 '+ARCHIVE,flush=True);files.upload()
 else:
  try:import google.colab;colab=True
  except ImportError:pass
 if hashlib.sha256(archive.read_bytes()).hexdigest()!=EXPECTED_SHA:raise ValueError('ZIP checksum mismatch')
 work=Path('qwen_self_read_compute_cpu_v1_'+str(time.time_ns())).resolve();work.mkdir()
 with zipfile.ZipFile(archive) as z:
  if any(Path(n).name!=n or n in ('.','..') for n in z.namelist()):raise ValueError('Unsafe archive path')
  if z.testzip() is not None:raise ValueError('ZIP CRC failure')
  z.extractall(work)
 env_vars=os.environ.copy();env_vars['CUDA_VISIBLE_DEVICES']='';env_vars['TOKENIZERS_PARALLELISM']='false'
 env_vars['OMP_NUM_THREADS']=str(args.threads);env_vars['MKL_NUM_THREADS']=str(args.threads)
 start=time.monotonic();stage=None;status='INCOMPLETE'
 def event(kind,**kw):
  with (work/'diagnostics.jsonl').open('a') as f:
   f.write(json.dumps(dict(event=kind,stage=stage,elapsed_seconds=time.monotonic()-start,**kw))+'\n');f.flush()
 def run(command,label):
  nonlocal stage
  stage=label;print('Starting',label,'(no time limit)',flush=True);event('stage_start')
  with (work/(label+'.log')).open('w') as log:
   proc=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env_vars,start_new_session=True)
   next_ping=0
   try:
    while proc.poll() is None:
     now=time.monotonic()
     if now>=next_ping:
      path=work/(label+'.log')
      with path.open('rb') as f:f.seek(0,2);size=f.tell();f.seek(max(0,size-1600));tail=f.read().decode(errors='replace')
      print(f'[{label}] total_elapsed={int(now-start)}s log_bytes={size}\n{tail}',flush=True)
      event('heartbeat',pid=proc.pid,log_bytes=size);next_ping=now+30
     time.sleep(2)
   finally:
    if proc.poll() is None:
     event('cleanup_signal',signal='SIGTERM')
     try:os.killpg(proc.pid,signal.SIGTERM)
     except ProcessLookupError:pass
     try:proc.wait(timeout=20)
     except subprocess.TimeoutExpired:
      event('cleanup_signal',signal='SIGKILL')
      try:os.killpg(proc.pid,signal.SIGKILL)
      except ProcessLookupError:pass
      proc.wait()
    event('stage_end',returncode=proc.returncode)
   if proc.returncode:raise RuntimeError(label+' failed; inspect '+label+'.log')
 try:
  venv=work/'env';run([sys.executable,'-m','venv','--without-pip',str(venv)],'create_env');py=str(venv/'bin/python')
  pip=[sys.executable,'-m','pip','--isolated','--python',py,'install']
  if platform.system()=='Linux' and platform.machine() in ('x86_64','amd64'):
   run(pip+['--index-url','https://download.pytorch.org/whl/cpu','torch==2.8.0+cpu'],'install_cpu_torch')
  elif platform.system()=='Darwin' or platform.machine() in ('aarch64','arm64'):
   run(pip+['torch==2.8.0'],'install_cpu_torch')
  else:raise RuntimeError('Unsupported wheel platform; use Linux x86_64 Colab')
  run(pip+['-r',str(work/'requirements_cpu.txt')],'install')
  run([py,'-m','unittest','discover','-s',str(work),'-p','test_*.py'],'tests')
  run([py,'-u',str(work/'run_cpu_audit.py'),'--out',str(work/'results'),'--threads',str(args.threads)],'audit')
  status='COMPLETED_CHECK_EXPERIMENT_SUMMARY'
 except KeyboardInterrupt:status='USER_INTERRUPTED';print('Interrupted; packaging completed rows.',flush=True)
 except Exception as e:status='ERROR';event('exception',message=repr(e));print(repr(e),flush=True)
 finally:
  (work/'launcher_summary.json').write_text(json.dumps(dict(status=status,stage=stage,seconds=time.monotonic()-start,
   time_limit_seconds=None,threads=args.threads,warning='Host/session limits still apply; VM loss can prevent ZIP creation. SIGKILL alone does not prove OOM.'),indent=2))
  output=Path(str(work)+'_results.zip')
  with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED) as z:
   for f in work.rglob('*'):
    if f.is_file() and 'env' not in f.relative_to(work).parts and '__pycache__' not in f.parts:z.write(f,str(f.relative_to(work)))
  if (work/'results/summary.json').exists():print((work/'results/summary.json').read_text(),flush=True)
  print('Results:',output,flush=True)
  if colab:
   from google.colab import files
   files.download(str(output))
if __name__=='__main__':main()
