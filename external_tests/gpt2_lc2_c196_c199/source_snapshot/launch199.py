"""Pure NumPy read-only CPU audit launcher with separate bounded child processes."""
from pathlib import Path
import os,shutil,signal,subprocess,sys,threading,time,traceback,zipfile
from core199 import *

def dependencies(deps,env,log):
    visible([sys.executable,'-I','-c',"import torch,numpy;assert torch.ones(1,device='cpu').device.type=='cpu';print('CPU_TORCH_READY',torch.__version__,numpy.__version__,flush=True)"],'C199 CPU PyTorch',90,env,log)
    script="import sys;sys.path.insert(0,sys.argv[1]);import transformers,peft,tokenizers,huggingface_hub,accelerate,safetensors;v=[transformers.__version__,peft.__version__,tokenizers.__version__,huggingface_hub.__version__,accelerate.__version__,safetensors.__version__];assert v==['4.56.2','0.17.1','0.22.0','0.34.4','1.10.1','0.6.2'],v;print('PINNED_HF_OK',v,flush=True)"
    check=[sys.executable,'-I','-c',script,str(deps)];r=subprocess.run(check,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,env=env,timeout=90)
    if r.returncode:
        print('C199：隔离安装固定HF包，保留系统PyTorch与NumPy；不安装SciPy。',flush=True)
        visible([sys.executable,'-I','-m','pip','install','--upgrade','--no-deps','--only-binary=:all:','--progress-bar','off','--timeout','30','--retries','2','--target',str(deps),'transformers==4.56.2','peft==0.17.1','tokenizers==0.22.0','huggingface_hub==0.34.4','accelerate==1.10.1','safetensors==0.6.2'],'C199 isolated pinned HF dependencies',600,env,log)
    visible(check,'C199 pinned HF check',90,env,log)


def child_environment():return dict(os.environ,CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1',TOKENIZERS_PARALLELISM='false',USE_TF='0',USE_FLAX='0')

def visible(cmd,label,timeout,env,log):
    print('START',label,flush=True);started=time.monotonic();p=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1,env=env,start_new_session=True)
    def drain():
        for line in p.stdout:print(line,end='',flush=True);log.write(line);log.flush()
    reader=threading.Thread(target=drain,daemon=True);reader.start()
    try:
        while True:
            try:code=p.wait(timeout=min(30,max(.1,timeout-(time.monotonic()-started))));break
            except subprocess.TimeoutExpired:
                if time.monotonic()-started>=timeout:raise TimeoutError(label+' exceeded fixed budget')
                print(label,'运行中',int(time.monotonic()-started),'秒',flush=True)
        reader.join(timeout=5);p.stdout.close();need(code==0,label+' exit='+str(code));print('END',label,'exit=',code,flush=True)
    except BaseException:
        try:os.killpg(p.pid,signal.SIGKILL)
        except ProcessLookupError:pass
        p.wait();reader.join(timeout=5);p.stdout.close();raise

def choose(kind,explicit='',drive=Path('/content/drive/MyDrive')):
    r=clean_json((ROOT/'INPUT_PINS.json').read_bytes())[kind]
    if explicit:
        p=Path(explicit).expanduser().resolve();need(p.is_file() and sha(p)==r['sha256'],'Wrong fixed original '+kind);return p
    preferred=drive/Path(r['path']).relative_to('/content/drive/MyDrive')
    for p in [preferred,Path(r.get('local_path',str(preferred))),Path('/content')/preferred.name]:
        if p.is_file() and sha(p)==r['sha256']:return p
    raise FileNotFoundError('未找到固定'+kind+'科学结果ZIP。请挂载原Drive或填写'+kind+'_ZIP绝对路径；内容仍须匹配原哈希。')

def start(use_drive=True,inputs=None,auto_download=True,base=None):
    manifest=clean_json((ROOT/'SOURCE_MANIFEST.json').read_bytes());actual={p.relative_to(ROOT).as_posix() for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    need(actual==set(manifest)|{'SOURCE_MANIFEST.json'},'C199 source inventory changed')
    for n,h in manifest.items():need(sha(ROOT/n)==h,'C199 delivered source changed: '+n)
    base=Path(base or ('/content' if Path('/content').is_dir() else Path.cwd()));session=base/'gpt2_lc2_c199_frozen_new_state_confirmation_cpu_v1'/('session_'+str(time.time_ns()));out=session/'run';out.mkdir(parents=True)
    shutil.copytree(ROOT,out/'source',ignore=shutil.ignore_patterns('__pycache__'));dump(out/'STATUS.json',dict(protocol=PROTOCOL,complete=False,stage='launcher_started'))
    originals={};failure=None;drive=Path('/content/drive/MyDrive');pins=clean_json((ROOT/'INPUT_PINS.json').read_bytes());result=session/('c199_frozen_new_state_confirmation_'+str(time.time_ns())+'_results.zip')
    deps=session/'deps';deps.mkdir()
    boot='import sys,runpy;d=sys.argv.pop(1);r=sys.argv.pop(1);sys.path[:0]=[d,r];p=sys.argv[1];sys.argv=sys.argv[1:];runpy.run_path(p,run_name="__main__")'
    prefix=[sys.executable,'-u','-I','-c',boot,str(deps),str(ROOT)];env=child_environment()
    try:
        if use_drive and not drive.is_dir():
            from google.colab import drive as gd
            gd.mount('/content/drive',timeout_ms=180000)
        for k in INPUT_KEYS:
            originals[k]=choose(k,(inputs or {}).get(k,''),drive);print('C199 原始只读输入',k,str(originals[k]),flush=True)
        dump(out/'ORIGINAL_INPUTS.json',{k:dict(path=str(p),sha256=sha(p)) for k,p in originals.items()})
        with (out/'console.log').open('w',encoding='utf-8') as log:
            visible(prefix+[str(ROOT/'test_math199.py')],'C199 frozen map/route/full-feature/gate checks',120,env,log)
            visible(prefix+[str(ROOT/'test_novelty199.py')],'C199 bounded archive novelty/source-exclusion controls',120,env,log)
            visible(prefix+[str(ROOT/'test199.py')],'C199 synthetic known84/new48/two-round/no-repair controls',1200,env,log)
            visible(prefix+[str(ROOT/'test_binding199.py')],'C199 original C198 source/feature/route/gate binding controls',1200,env,log)
            visible(prefix+[str(ROOT/'test_delivery199.py')],'C199 isolated entrypoint/failure packaging controls',180,env,log)
            dependencies(deps,env,log)
            visible(prefix+[str(ROOT/'torch_smoke199.py')],'C199 real tiny CPU Torch/reference/restore controls (not GPT-2)',180,env,log)
            print('C199：固定C196最终头；44原包重绑；先完成84态已知控制，再测固定48个新方向态，三个已训练类各16态；幅度0.125/0.375，不重抽。',flush=True)
            print('预登记624条记录、最多1429次CPU前向；两轮96个新提案均先封存。无反传/训练/求解/修复/选头/排除输入；制备和写入失败全保留。已知控制失败则不测新状态；上限240分钟，自测与依赖另计。',flush=True)
            args=[]
            for k,p in originals.items():args+=['--'+k.lower(),str(p)]
            visible(prefix+[str(ROOT/'worker199.py')]+args+['--out',str(out)],'C199 frozen prospective48 local confirmation',TIMEOUT,env,log)
        s=clean_json((out/'SUMMARY.json').read_bytes());need(s.get('complete') is True,'C199 incomplete');dump(out/'STATUS.json',dict(protocol=PROTOCOL,complete=True,stage='conditional_confirmation_completed_result_in_summary'))
    except BaseException as e:
        failure=e;dump(out/'LAUNCHER_ERROR.json',dict(type=type(e).__name__,message=str(e),traceback=traceback.format_exc()));print('C199 stopped:',type(e).__name__,str(e),flush=True)
    finally:
        checks={}
        for k,p in originals.items():
            try:checks[k]=sha(p)==pins[k]['sha256']
            except Exception:checks[k]=False
        dump(out/'ORIGINAL_INPUTS_UNCHANGED.json',checks)
        if set(checks)!=set(INPUT_KEYS) or not all(checks.values()):failure=failure or ValueError('Original input missing or changed')
        if failure:
            s=clean_json((out/'SUMMARY.json').read_bytes()) if (out/'SUMMARY.json').is_file() else dict(protocol=PROTOCOL,new_model_forwards=0,new_backward_calls=0,new_training_steps=0,new_head_fits=0,new_Jacobians=0,new_SVD_calls=0,linear_systems_solved=0,LC2_closed=False,LC2_single_fold_passed=False,candidate_selected=False,parent_decision_reclassified=False)
            s.update(complete=False,decision='C199_INCOMPLETE_NO_CONFIRMATION',runtime_counts_finalized=False,launcher_failure=str(failure));dump(out/'SUMMARY.json',s);dump(out/'STATUS.json',dict(protocol=PROTOCOL,complete=False,stage='stopped_and_packaged'))
        dump(out/'REQUESTED_WORK.json',dict(known_training_states_requested=84,prospective_confirmation_states_requested=48,new_model_forwards_requested=1429,scalar_backward_calls_requested=0,new_training_steps_requested=0))
        result_manifest={p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file() and p!=out/'RESULT_MANIFEST.json'};dump(out/'RESULT_MANIFEST.json',result_manifest)
        with zipfile.ZipFile(result,'w',zipfile.ZIP_DEFLATED) as z:
            for p in sorted(out.rglob('*')):
                if p.is_file():z.write(p,p.relative_to(out).as_posix())
        print('唯一结果总包：',result,flush=True);print('RESULT_SHA256',sha(result),flush=True)
        if use_drive and drive.is_dir():
            try:
                target=drive/(PROTOCOL+'_RECOVERY')/result.name;target.parent.mkdir(exist_ok=True);shutil.copy2(result,target);need(sha(target)==sha(result),'Drive backup hash differs');print('DRIVE_BACKUP_OK',target,flush=True)
            except Exception as e:print('Drive备份失败；本地总包保留：',repr(e),flush=True)
        if auto_download:
            try:
                from google.colab import files
                files.download(str(result))
            except Exception as e:print('请从上方路径下载总包：',repr(e),flush=True)
    if failure:raise RuntimeError('C199未完成；已封装失败结果：'+str(result)) from failure
    return result
