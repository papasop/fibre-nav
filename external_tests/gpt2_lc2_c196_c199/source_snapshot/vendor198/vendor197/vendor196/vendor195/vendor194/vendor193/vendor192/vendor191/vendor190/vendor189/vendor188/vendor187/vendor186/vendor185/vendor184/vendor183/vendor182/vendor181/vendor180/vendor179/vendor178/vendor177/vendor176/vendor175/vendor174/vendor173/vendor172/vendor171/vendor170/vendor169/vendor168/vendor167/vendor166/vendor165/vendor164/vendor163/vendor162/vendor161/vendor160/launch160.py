"""Pinned CPU model diagnostic launcher with separate bounded child processes."""
from pathlib import Path
import os,shutil,signal,subprocess,sys,threading,time,traceback,zipfile
from core160 import *

def child_environment():return dict(os.environ,CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1',TOKENIZERS_PARALLELISM='false',USE_TF='0',USE_FLAX='0')

def dependencies(deps,env,log):
    visible([sys.executable,'-I','-c',"import torch,numpy;v=torch.ones(2,device='cpu');assert v.device.type=='cpu';print('CPU_TORCH_READY',torch.__version__,numpy.__version__,flush=True)"],'C160 CPU PyTorch',90,env,log)
    script="import sys;sys.path.insert(0,sys.argv[1]);import transformers,peft,tokenizers,huggingface_hub,accelerate,safetensors;v=[transformers.__version__,peft.__version__,tokenizers.__version__,huggingface_hub.__version__,accelerate.__version__,safetensors.__version__];assert v==['4.56.2','0.17.1','0.22.0','0.34.4','1.10.1','0.6.2'],v;print('PINNED_HF_DEPENDENCIES_OK',v,flush=True)"
    check=[sys.executable,'-I','-c',script,str(deps)];p=subprocess.run(check,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,env=env,timeout=90)
    if p.returncode:
        print('C160：在隔离目录安装固定HF依赖；保留系统PyTorch与NumPy。',flush=True)
        visible([sys.executable,'-I','-m','pip','install','--upgrade','--no-deps','--only-binary=:all:','--progress-bar','off','--timeout','30','--retries','2','--target',str(deps),'transformers==4.56.2','peft==0.17.1','tokenizers==0.22.0','huggingface_hub==0.34.4','accelerate==1.10.1','safetensors==0.6.2'],'C160 isolated pinned HF dependencies',600,env,log)
    visible(check,'C160 pinned HF check',90,env,log)

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
    need(actual==set(manifest)|{'SOURCE_MANIFEST.json'},'C160 source inventory changed')
    for n,h in manifest.items():need(sha(ROOT/n)==h,'C160 delivered source changed: '+n)
    base=Path(base or ('/content' if Path('/content').is_dir() else Path.cwd()));session=base/'gpt2_lc2_c160_full84_fp64_baseline_cpu_v1'/('session_'+str(time.time_ns()));out=session/'run';out.mkdir(parents=True)
    shutil.copytree(ROOT,out/'source',ignore=shutil.ignore_patterns('__pycache__'));dump(out/'STATUS.json',dict(protocol=PROTOCOL,complete=False,stage='launcher_started'))
    originals={};failure=None;drive=Path('/content/drive/MyDrive');pins=clean_json((ROOT/'INPUT_PINS.json').read_bytes());result=session/('c160_full84_fp64_baseline_'+str(time.time_ns())+'_results.zip')
    deps=session/'deps';deps.mkdir()
    boot='import sys,runpy;d=sys.argv.pop(1);r=sys.argv.pop(1);sys.path[:0]=[d,r];p=sys.argv[1];sys.argv=sys.argv[1:];runpy.run_path(p,run_name="__main__")'
    prefix=[sys.executable,'-u','-I','-c',boot,str(deps),str(ROOT)];env=child_environment()
    try:
        if use_drive and not drive.is_dir():
            from google.colab import drive as gd
            gd.mount('/content/drive')
        for k in INPUT_KEYS:
            originals[k]=choose(k,(inputs or {}).get(k,''),drive);print('C160 原始只读输入',k,str(originals[k]),flush=True)
        dump(out/'ORIGINAL_INPUTS.json',{k:dict(path=str(p),sha256=sha(p)) for k,p in originals.items()})
        with (out/'console.log').open('w',encoding='utf-8') as log:
            visible(prefix+[str(ROOT/'test160.py')],'C160 fixed path/seals/denominators/no-training tests',2400,env,log)
            visible(prefix+[str(ROOT/'test_bootstrap160.py')],'C160 entrypoint and import-only checks (no tensor validation)',90,env,log)
            visible(prefix+[str(ROOT/'test_launcher160.py')],'C160 launcher success/failure packaging tests (synthetic only)',120,env,log)
            dependencies(deps,env,log)
            visible(prefix+[str(ROOT/'torch_smoke160.py')],'C160 actual CPU Torch scalar AD/FD and restoration smoke',600,env,log)
            print('C160：真实GPT-2 CPU；独立同次运行协议；完整84训练态的响应/读取/KL/半径/正则目标与梯度；原生FP32资格分列。',flush=True)
            print('固定2019次模型前向、168次标量反传；完整84态三个尺度差分与两轮原生FP32检查；保留三个正负步长及重复。无训练/新态/选头/确认/排除输入。工作上限120分钟，自测另计。',flush=True)
            args=[]
            for k,p in originals.items():args+=['--'+k.lower(),str(p)]
            visible(prefix+[str(ROOT/'worker160.py')]+args+['--out',str(out)],'C160 full84 FP64 objective/gradient baseline',TIMEOUT,env,log)
        s=clean_json((out/'SUMMARY.json').read_bytes());need(s.get('complete') is True,'C160 incomplete');dump(out/'STATUS.json',dict(protocol=PROTOCOL,complete=True,stage='full84_fp64_baseline_diagnostic_complete'))
    except BaseException as e:
        failure=e;dump(out/'LAUNCHER_ERROR.json',dict(type=type(e).__name__,message=str(e),traceback=traceback.format_exc()));print('C160 stopped:',type(e).__name__,str(e),flush=True)
    finally:
        checks={}
        for k,p in originals.items():
            try:checks[k]=sha(p)==pins[k]['sha256']
            except Exception:checks[k]=False
        dump(out/'ORIGINAL_INPUTS_UNCHANGED.json',checks)
        if set(checks)!=set(INPUT_KEYS) or not all(checks.values()):failure=failure or ValueError('Original input missing or changed')
        if failure:
            s=clean_json((out/'SUMMARY.json').read_bytes()) if (out/'SUMMARY.json').is_file() else dict(protocol=PROTOCOL,new_model_forwards=0,new_backward_calls=0,new_training_steps=0,new_head_fits=0,new_Jacobians=0,new_SVD_calls=0,linear_systems_solved=0,LC2_closed=False,LC2_single_fold_passed=False,candidate_selected=False,parent_decision_reclassified=False)
            s.update(complete=False,decision='C160_INCOMPLETE_NO_BASELINE_ACCEPTED',runtime_counts_finalized=False,launcher_failure=str(failure));dump(out/'SUMMARY.json',s);dump(out/'STATUS.json',dict(protocol=PROTOCOL,complete=False,stage='stopped_and_packaged'))
        dump(out/'REQUESTED_WORK.json',dict(training_states_requested=84,new_model_forwards_requested=2019,scalar_backward_calls_requested=168,new_training_steps_requested=0))
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
    if failure:raise RuntimeError('C160未完成；已封装失败结果：'+str(result)) from failure
    return result
